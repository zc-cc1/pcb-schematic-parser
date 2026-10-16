"""Rotation-only warm-start trial against the completed matched v2 control."""
import argparse
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from pcb_parser.unet_model import WirePinUNet
from pcb_parser.unet_retraining import CachedTrainingTiles
from pcb_parser.unet_rotation_training import RotationTrainingTiles
from pcb_parser.unet_data import predict_tiled
from retrain_unet_v2 import sha, write_json
from train_unet import epoch

MATCHED_KEYS = ("initial_sha256", "epochs", "batch_size", "learning_rate", "minimum_learning_rate",
                "weight_decay", "amp", "seed", "tile_size", "stride", "wire_positive_weight", "pin_positive_weight")


def monitor(probability):
    values = {}
    for name, y, x0, x1 in (("X1_left", 80, 620, 680), ("DOUT", 181, 605, 628), ("X1_right", 80, 727, 760)):
        scores = probability[0, y-1:y+2, x0:x1+1].max(axis=0)
        values[name] = dict(median=float(np.median(scores)), minimum=float(scores.min()),
                            samples_above_050=int((scores >= .5).sum()), samples=len(scores))
    values["manual_nonwire_IC_border"] = dict(box=[591, 86, 594, 106],
        pixels_above_050=int((probability[0, 86:106, 591:594] >= .5).sum()), pixels=60)
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT/"configs/unet_rotation_retraining_v3.json")
    parser.add_argument("--allow-draft", action="store_true")
    parser.add_argument("--smoke-steps", type=int)
    args = parser.parse_args()
    if not args.allow_draft:
        raise ValueError("Explicit draft-label experiment flag --allow-draft required")
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    output = (ROOT/cfg["output"]).resolve()
    if ROOT not in output.parents:
        raise ValueError("Output must be inside project")
    output.mkdir(parents=True, exist_ok=True)
    folder = output/"rotation"
    if folder.exists():
        raise ValueError("A rotation training already exists; do not overwrite it")
    source_path = ROOT/cfg["initial_weights"]
    manifest = ROOT/cfg["manifest"]
    control = ROOT/cfg["reference_control"]
    control_cfg = json.loads((control/"configuration.json").read_text(encoding="utf-8"))
    control_summary = json.loads((control/"summary.json").read_text(encoding="utf-8"))
    assert control_cfg["arm"] == "control"
    assert all(cfg[k] == control_cfg[k] for k in MATCHED_KEYS)
    assert control_summary["epochs_completed"] == cfg["epochs"]
    assert sha(source_path) == cfg["initial_sha256"]
    manifest_hash = sha(manifest)
    assert manifest_hash == control_cfg["manifest_sha256"] == control_summary["manifest_sha256"]
    assert sha(control/"best.pt") == control_summary["checkpoint_sha256"]
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows = [r for r in rows if r["split"] == "val"]
    assert len(train_rows) == 103 and len(val_rows) == 26 and "0036" not in [r["case_id"] for r in train_rows]
    assert control_summary["train_cases"] == [r["case_id"] for r in train_rows]
    assert control_summary["validation_cases"] == [r["case_id"] for r in val_rows]
    original = json.loads((ROOT/cfg["protected_source_summary"]).read_text(encoding="utf-8"))["protected_source_sha256"]
    assert all(sha(path) == digest for path, digest in original.items())
    for row in rows:
        assert all(sha(row[k]) == row[k+"_sha256"] for k in ("image", "wire_mask", "pin_heatmap"))
    reviewed = ROOT/cfg["label_review"]
    review = json.loads(reviewed.read_text(encoding="utf-8"))
    assert len(review["regions"]) >= 8 and all(r["status"].startswith("agent_visual_reviewed") for r in review["regions"])
    torch.set_num_threads(4)
    if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0] < 4*1024**3:
        raise RuntimeError("CUDA with 4 GiB free required; do not start a long CPU job")
    torch.manual_seed(cfg["seed"]); random.seed(cfg["seed"]); np.random.seed(cfg["seed"])
    generator = torch.Generator().manual_seed(cfg["seed"])
    device = torch.device("cuda:0")
    source = torch.load(source_path, map_location="cpu", weights_only=True)
    assert source["kind"] == "trained_on_draft_labels"
    train_set = RotationTrainingTiles(train_rows, cfg["tile_size"], cfg["stride"], cfg["rotation_augmentation"], cfg["seed"])
    val_set = CachedTrainingTiles(val_rows, cfg["tile_size"], cfg["stride"])
    loader = DataLoader(train_set, batch_size=cfg["batch_size"], shuffle=True, num_workers=0,
                        generator=generator, pin_memory=True)
    validation = DataLoader(val_set, batch_size=cfg["batch_size"], shuffle=False, num_workers=0, pin_memory=True)
    model = WirePinUNet(source["base_channels"]).to(device)
    model.load_state_dict(source["state_dict"], strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg["epochs"], eta_min=cfg["minimum_learning_rate"])
    scaler = torch.amp.GradScaler("cuda", enabled=cfg["amp"])
    tick = time.perf_counter()
    if args.smoke_steps:
        if args.smoke_steps <= 0:
            raise ValueError("Positive smoke steps required")
        loss, values, _ = epoch(model, loader, device, cfg, optimizer, scaler,
                                max_steps=args.smoke_steps, phase="rotation_performance_smoke_only")
        result = dict(status="performance_smoke_only_no_quality_checkpoint", steps=len(values), loss=loss,
            seconds=time.perf_counter()-tick, batches_per_epoch=len(loader), validation_batches=len(validation),
            rotation_counts=train_set.rotation_counts.tolist(), cuda_peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3)
        assert all(sha(path) == digest for path, digest in original.items())
        write_json(output/"performance_smoke.json", result)
        print(json.dumps(result), flush=True)
        return
    folder.mkdir()
    write_json(folder/"configuration.json", cfg | {"arm": "rotation", "manifest_sha256": manifest_hash})
    protocol = dict(config=cfg, config_sha256=sha(args.config), manifest_sha256=manifest_hash,
        initial_weights_sha256=sha(source_path), label_review_sha256=sha(reviewed),
        source_code_sha256={str(p.relative_to(ROOT)): sha(p) for p in (Path(__file__), ROOT/"scripts/train_unet.py",
            ROOT/"src/pcb_parser/unet_rotation_training.py", ROOT/"src/pcb_parser/unet_retraining.py", ROOT/"src/pcb_parser/unet_model.py")},
        reference_control=dict(reused=True, summary_sha256=sha(control/"summary.json"),
            checkpoint_sha256=sha(control/"best.pt"), matched_keys=list(MATCHED_KEYS)),
        training_comparison="same initial weights, labels, case order, shuffle seed, loss, schedule and 16 epochs; only joint quarter-turn augmentation differs",
        labels_status=cfg["label_status"], independent_test_set=False, official_score=None,
        monitor_used_for_checkpoint_selection=False)
    write_json(output/"training_protocol.json", protocol)
    monitor_rgb = np.asarray(Image.open(ROOT/"data/standardized/0036/0036.png").convert("RGB"))
    history = []; best_loss = float("inf")
    initial_out = model.out.conv.weight.detach().cpu().clone()
    monitor_folder = output/"monitor_0036"; monitor_folder.mkdir()
    print(json.dumps(dict(phase="rotation_arm_start", epochs=cfg["epochs"], training_batches=len(loader),
                         validation_batches=len(validation), reference_control_reused=True)), flush=True)
    for index in range(cfg["epochs"]):
        epoch_tick = time.perf_counter(); lr = optimizer.param_groups[0]["lr"]
        old_counts = train_set.rotation_counts.copy()
        train_loss, batches, train_metrics = epoch(model, loader, device, cfg, optimizer, scaler, phase=f"rotation/train/{index+1}")
        val_loss, _, val_metrics = epoch(model, validation, device, cfg, phase=f"rotation/validation/{index+1}")
        scheduler.step()
        probability = predict_tiled(model, monitor_rgb, device, cfg["tile_size"], cfg["stride"])
        local = monitor(probability)
        np.savez_compressed(monitor_folder/f"epoch_{index+1:02d}.npz", wire=probability[0], pin_tip=probability[1])
        record = dict(epoch=index+1, train_loss=train_loss, val_loss=val_loss, steps=len(batches), learning_rate=lr,
            epoch_seconds=time.perf_counter()-epoch_tick, train_wire_pixels=train_metrics, val_wire_pixels=val_metrics,
            rotation_counts=(train_set.rotation_counts-old_counts).tolist(), monitor_0036=local)
        history.append(record)
        improved = val_loss < best_loss; best_loss = min(best_loss, val_loss)
        model_cfg = dict(source["config"]); model_cfg["training"] = cfg | {"arm": "rotation"}
        model_cfg["labels_status"] = cfg["label_status"]
        checkpoint = dict(state_dict=model.state_dict(), base_channels=source["base_channels"], kind="trained_on_draft_labels",
            config=model_cfg, outputs=source["outputs"], source_commit=source["source_commit"], epoch=index+1,
            best_loss=best_loss, history=history, manifest_sha256=manifest_hash, optimizer_state=optimizer.state_dict(),
            scaler_state=scaler.state_dict(), scheduler_state=scheduler.state_dict(),
            initial_weights_sha256=cfg["initial_sha256"], initial_epoch=source["epoch"], arm="rotation",
            rotation_rng_state=train_set.rotation_rng.bit_generator.state, torch_rng_state=torch.get_rng_state(),
            shuffle_rng_state=generator.get_state(), reference_control_checkpoint_sha256=sha(control/"best.pt"))
        torch.save(checkpoint, folder/"last.pt")
        if improved:
            torch.save(checkpoint, folder/"best.pt")
        write_json(folder/"history.json", history)
        progress = dict(status="training", epoch=index+1, epochs=cfg["epochs"], best_loss=best_loss,
                        latest=record, elapsed_seconds=time.perf_counter()-tick)
        write_json(output/"progress.json", progress)
        print(json.dumps(progress), flush=True)
    best = min(history, key=lambda r: r["val_loss"])
    assert all(sha(path) == digest for path, digest in original.items())
    assert sha(source_path) == cfg["initial_sha256"] and sha(manifest) == manifest_hash
    assert sha(control/"best.pt") == protocol["reference_control"]["checkpoint_sha256"]
    summary = dict(status="completed_rotation_draft_label_warm_start", arm="rotation", device=str(device),
        epochs_completed=len(history), steps=sum(r["steps"] for r in history), wall_seconds=time.perf_counter()-tick,
        best_epoch=best["epoch"], best_validation=best, initial_weights_sha256=cfg["initial_sha256"], initial_epoch=source["epoch"],
        checkpoint=str((folder/"best.pt").resolve()), checkpoint_sha256=sha(folder/"best.pt"), manifest_sha256=manifest_hash,
        train_cases=[r["case_id"] for r in train_rows], validation_cases=[r["case_id"] for r in val_rows],
        output_layer_changed=not torch.equal(initial_out, model.out.conv.weight.detach().cpu()),
        production_config_changed=False, original_files_unchanged=True, reference_control_reused=True, official_score=None)
    write_json(folder/"summary.json", summary)
    write_json(output/"training_summary.json", dict(status="rotation_completed_control_reused", rotation=summary,
        control_summary_sha256=sha(control/"summary.json"), original_weights_unchanged=True, production_settings_changed=False))
    write_json(output/"progress.json", dict(status="training_completed_evaluation_pending", rotation=summary))
    train_set.clear_cache(); val_set.clear_cache()
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
