"""Matched warm-start control/contrast runs on a versioned draft label set."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from pcb_parser.unet_model import WirePinUNet
from pcb_parser.unet_retraining import CachedTrainingTiles
from train_unet import epoch


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def train_arm(arm, cfg, source, manifest_hash, rows, output, device, benchmark_steps=None):
    folder = output/arm
    if folder.exists() and not benchmark_steps:
        raise ValueError(f"Training output already exists: {folder}")
    torch.manual_seed(cfg["seed"]); random.seed(cfg["seed"]); np.random.seed(cfg["seed"])
    generator = torch.Generator().manual_seed(cfg["seed"])
    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows = [r for r in rows if r["split"] == "val"]
    train_set = CachedTrainingTiles(train_rows, cfg["tile_size"], cfg["stride"],
                    cfg["augmentation"] if arm == "contrast" else None, cfg["seed"])
    val_set = CachedTrainingTiles(val_rows, cfg["tile_size"], cfg["stride"])
    loader = DataLoader(train_set, batch_size=cfg["batch_size"], shuffle=True,
                        num_workers=0, generator=generator, pin_memory=device.type == "cuda")
    validation = DataLoader(val_set, batch_size=cfg["batch_size"], shuffle=False,
                            num_workers=0, pin_memory=device.type == "cuda")
    model = WirePinUNet(source["base_channels"]).to(device)
    model.load_state_dict(source["state_dict"], strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg["epochs"], eta_min=cfg["minimum_learning_rate"])
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and cfg["amp"])
    start = time.perf_counter()
    initial_output = model.out.conv.weight.detach().cpu().clone()
    if benchmark_steps:
        loss, values, _ = epoch(model, loader, device, cfg, optimizer, scaler,
                                max_steps=benchmark_steps, phase="performance_smoke_only")
        seconds = time.perf_counter()-start
        result = {"status": "performance_smoke_only_not_quality_checkpoint", "steps": len(values),
                  "seconds": seconds, "training_batches_per_epoch": len(loader),
                  "validation_batches": len(validation), "loss": loss, "cuda_peak_allocated_gib": torch.cuda.max_memory_allocated()/1024**3,
                  "cache_estimated_bytes": sum(r["width"]*r["height"]*5 for r in rows)}
        train_set.clear_cache(); val_set.clear_cache()
        return result
    folder.mkdir(parents=True)
    write_json(folder/"configuration.json", cfg | {"arm": arm, "manifest_sha256": manifest_hash})
    best_loss = float("inf"); history = []
    print(json.dumps({"phase": "arm_start", "arm": arm, "epochs": cfg["epochs"], "training_batches": len(loader),
                      "validation_batches": len(validation), "initial_epoch": source["epoch"], "device": str(device)}), flush=True)
    for index in range(cfg["epochs"]):
        tick = time.perf_counter(); lr = optimizer.param_groups[0]["lr"]
        train_loss, batches, train_metrics = epoch(model, loader, device, cfg, optimizer, scaler, phase=f"{arm}/train/{index+1}")
        val_loss, _, val_metrics = epoch(model, validation, device, cfg, phase=f"{arm}/validation/{index+1}")
        scheduler.step()
        record = {"epoch": index+1, "train_loss": train_loss, "val_loss": val_loss,
                  "steps": len(batches), "learning_rate": lr, "epoch_seconds": time.perf_counter()-tick,
                  "train_wire_pixels": train_metrics, "val_wire_pixels": val_metrics}
        history.append(record); improved = val_loss < best_loss; best_loss = min(best_loss, val_loss)
        model_cfg = dict(source["config"])
        model_cfg["training"] = dict(cfg)
        model_cfg["training"]["arm"] = arm
        model_cfg["labels_status"] = cfg["label_status"]
        checkpoint = {"state_dict": model.state_dict(), "base_channels": source["base_channels"],
                      "kind": "trained_on_draft_labels", "config": model_cfg, "outputs": source["outputs"],
                      "source_commit": source["source_commit"], "epoch": index+1, "best_loss": best_loss,
                      "history": history, "manifest_sha256": manifest_hash, "optimizer_state": optimizer.state_dict(),
                      "scaler_state": scaler.state_dict(), "scheduler_state": scheduler.state_dict(),
                      "initial_weights_sha256": cfg["initial_sha256"], "initial_epoch": source["epoch"],
                      "arm": arm, "augmentation_rng_state": train_set.rng.bit_generator.state,
                      "torch_rng_state": torch.get_rng_state(), "shuffle_rng_state": generator.get_state()}
        torch.save(checkpoint, folder/"last.pt")
        if improved:
            torch.save(checkpoint, folder/"best.pt")
        write_json(folder/"history.json", history)
        progress = {"status": "training", "arm": arm, "epoch": index+1, "epochs": cfg["epochs"],
                    "best_loss": best_loss, "latest": record, "elapsed_seconds": time.perf_counter()-start}
        write_json(output/"progress.json", progress)
        print(json.dumps(progress), flush=True)
    best = min(history, key=lambda r: r["val_loss"])
    summary = {"status": "completed_draft_label_warm_start_retraining", "arm": arm, "device": str(device),
               "epochs_completed": len(history), "steps": sum(r["steps"] for r in history),
               "wall_seconds": time.perf_counter()-start, "best_epoch": best["epoch"], "best_validation": best,
               "initial_weights_sha256": cfg["initial_sha256"], "initial_epoch": source["epoch"],
               "checkpoint": str((folder/"best.pt").resolve()), "checkpoint_sha256": sha(folder/"best.pt"),
               "manifest_sha256": manifest_hash, "train_cases": [r["case_id"] for r in train_rows],
               "validation_cases": [r["case_id"] for r in val_rows],
               "output_layer_changed": not torch.equal(initial_output, model.out.conv.weight.detach().cpu()),
               "production_config_changed": False, "official_score": None}
    write_json(folder/"summary.json", summary)
    train_set.clear_cache(); val_set.clear_cache()
    del model, optimizer, scaler, scheduler, checkpoint
    torch.cuda.empty_cache()
    print(json.dumps({"phase": "arm_complete", "arm": arm, "best_epoch": best["epoch"], "wall_seconds": summary["wall_seconds"]}), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT/"configs/unet_retraining_v2.json")
    parser.add_argument("--allow-draft", action="store_true")
    parser.add_argument("--benchmark-steps", type=int)
    args = parser.parse_args()
    if not args.allow_draft:
        raise ValueError("This is a partial-review draft-label experiment; --allow-draft is required")
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    output = ROOT/cfg["output"]; manifest = ROOT/cfg["manifest"]
    rows = list(map(json.loads, manifest.read_text(encoding="utf-8").splitlines()))
    source_path = ROOT/cfg["initial_weights"]
    assert sha(source_path) == cfg["initial_sha256"]
    assert len([r for r in rows if r["split"] == "train"]) == 103
    assert len([r for r in rows if r["split"] == "val"]) == 26
    assert "0036" not in [r["case_id"] for r in rows if r["split"] == "train"]
    for r in rows:
        assert sha(r["image"]) == r["image_sha256"]
        assert sha(r["wire_mask"]) == r["wire_mask_sha256"]
        assert sha(r["pin_heatmap"]) == r["pin_heatmap_sha256"]
    review = json.loads((output/"label_review/review_rois.json").read_text(encoding="utf-8"))
    assert len(review["regions"]) >= 8 and all(r["status"].startswith("agent_visual_reviewed") for r in review["regions"])
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; do not silently start a long CPU training job")
    free, _ = torch.cuda.mem_get_info()
    if free < 4*1024**3:
        raise RuntimeError("Less than 4 GiB free GPU memory; no other processes will be interrupted")
    device = torch.device("cuda:0")
    source = torch.load(source_path, map_location="cpu", weights_only=True)
    assert source["kind"] == "trained_on_draft_labels"
    manifest_hash = sha(manifest)
    if args.benchmark_steps:
        result = train_arm("control", cfg, source, manifest_hash, rows, output, device, args.benchmark_steps)
        write_json(output/"performance_smoke.json", result)
        print(json.dumps(result), flush=True)
        return
    protocol = {"config": cfg, "config_sha256": sha(args.config), "manifest_sha256": manifest_hash,
                "source_weights_sha256": sha(source_path), "label_review_sha256": sha(output/"label_review/review_rois.json"),
                "training_comparison": "same init, labels, optimizer, LR schedule, epochs and shuffled order; augmentation is the arm difference",
                "source_code_sha256": {str(p.relative_to(ROOT)): sha(p) for p in (Path(__file__), ROOT/"src/pcb_parser/unet_retraining.py", ROOT/"src/pcb_parser/unet_model.py")},
                "label_status": cfg["label_status"], "independent_test_set": False, "official_score": None}
    write_json(output/"training_protocol.json", protocol)
    summaries = []
    for arm in cfg["arms"]:
        summaries.append(train_arm(arm, cfg, source, manifest_hash, rows, output, device))
    assert sha(source_path) == cfg["initial_sha256"] and sha(manifest) == manifest_hash
    write_json(output/"training_summary.json", {"status": "both_runs_completed", "runs": summaries,
                "original_weights_unchanged": True, "production_settings_changed": False})
    write_json(output/"progress.json", {"status": "both_training_runs_completed_evaluation_pending", "runs": summaries})


if __name__ == "__main__":
    main()
