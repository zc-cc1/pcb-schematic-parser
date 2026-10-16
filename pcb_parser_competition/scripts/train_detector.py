"""Fine-tune the detector locally on CUDA, preserving shared output rows by name."""
import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "artifacts/ultralytics"))
Path(os.environ["YOLO_CONFIG_DIR"]).mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_AUTOINSTALL", "false")
os.environ.setdefault("MPLBACKEND", "Agg")

import torch
from ultralytics import YOLO, settings
from ultralytics.models.yolo.detect.train import DetectionTrainer

settings.update({"sync": False})


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class ClassNameTransferTrainer(DetectionTrainer):
    def get_model(self, cfg=None, weights=None, verbose=True):
        model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
        copied = []
        if weights is not None:
            source = weights["model"] if isinstance(weights, dict) else weights
            source_head, target_head = source.model[-1], model.model[-1]
            source_names = source.names
            source_lookup = {name: int(idx) for idx, name in source_names.items()}
            with torch.no_grad():
                for branch_index, (old_branch, new_branch) in enumerate(zip(source_head.cv3, target_head.cv3)):
                    old, new = old_branch[-1], new_branch[-1]
                    if old.weight.shape[1:] != new.weight.shape[1:]:
                        continue
                    for target_idx, name in self.data["names"].items():
                        if name not in source_lookup:
                            continue
                        source_idx = source_lookup[name]
                        new.weight[int(target_idx)].copy_(old.weight[source_idx])
                        new.bias[int(target_idx)].copy_(old.bias[source_idx])
                        assert torch.equal(new.weight[int(target_idx)], old.weight[source_idx])
                        assert torch.equal(new.bias[int(target_idx)], old.bias[source_idx])
                        copied.append({"branch": branch_index, "class_name": name, "source_index": source_idx, "target_index": int(target_idx)})
        transfer = {"copied_classification_rows": copied, "target_names": self.data["names"],
                    "policy": "base matching tensors + explicitly copied output classification rows by class name"}
        (self.save_dir / "class_transfer.json").write_text(json.dumps(transfer, indent=2), encoding="utf-8")
        return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/detector_training.json")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--run-name")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    training = config["training"].copy()
    if args.epochs is not None:
        training["epochs"] = args.epochs
    if args.batch is not None:
        training["batch"] = args.batch
    run_name = args.run_name or config["run_name"]
    if not torch.cuda.is_available():
        raise RuntimeError("GPU training requested but CUDA is unavailable")
    dataset = ROOT / config["dataset"]
    manifest_path = dataset.parent / "manifest.jsonl"
    rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()]
    train_groups = {r["group"] for r in rows if r["split"] == "train"}
    val_groups = {r["group"] for r in rows if r["split"] == "val"}
    if train_groups & val_groups:
        raise RuntimeError("Source groups overlap across splits")
    for row in rows:
        if digest(row["label"]) != row["label_sha256"] or digest(row["image"]) != row["image_sha256"]:
            raise RuntimeError(f"Training inputs changed since export: {row['case_id']}")
    audit = json.loads((dataset.parent / "audit.json").read_text(encoding="utf-8"))
    initial = ROOT / config["initial_weights"]
    record = {"status": "training_started", "started_at_utc": datetime.now(timezone.utc).isoformat(),
              "python": sys.executable, "config": config, "effective_training": training,
              "initial_architecture": config.get("initial_architecture"),
              "initial_weights_sha256": digest(initial), "dataset_manifest_sha256": digest(manifest_path),
              "data_config_sha256": digest(dataset), "image_counts": audit["image_counts"],
              "annotation_status": "experimental_bbox_category_labels_with_targeted_manual_corrections", "official_f1": None}
    run_record = ROOT / f"runs/training/{run_name}_record.json"
    run_record.parent.mkdir(parents=True, exist_ok=True)
    if run_record.exists():
        raise RuntimeError("Run record already exists; choose a new --run-name")
    run_record.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    started = time.perf_counter()
    architecture = config.get("initial_architecture")
    if architecture:
        # Build the requested architecture (e.g. "yolov8s-p2.yaml" resolves inside
        # the ultralytics package) and transfer every compatible layer from the
        # previous checkpoint; mismatched shapes (deeper/wider head) stay random.
        model = YOLO(str(architecture))
        model.load(str(initial))
    else:
        model = YOLO(str(initial))
    def record_epoch(trainer):
        progress = {"epoch_completed": trainer.epoch+1, "epochs_planned": trainer.epochs,
                    "wall_seconds": time.perf_counter()-started,
                    "losses": [float(v) for v in trainer.tloss.detach().cpu()],
                    "metrics": {key: float(value) for key,value in trainer.metrics.items()},
                    "peak_allocated_mib": torch.cuda.max_memory_allocated()/2**20,
                    "run_directory": str(trainer.save_dir)}
        live = ROOT / f"runs/training/{run_name}_progress.json"
        live.write_text(json.dumps(progress, ensure_ascii=False, indent=2), encoding="utf-8")
    model.add_callback("on_fit_epoch_end", record_epoch)
    try:
        model.train(trainer=ClassNameTransferTrainer, data=str(dataset), project=str(ROOT / "runs/training"), name=run_name,
                    exist_ok=False, **training)
        actual_run = Path(model.trainer.save_dir)
        best = actual_run / "weights/best.pt"
        if not best.is_file():
            raise RuntimeError("Training returned without best.pt")
        destination = ROOT / f"weights/{run_name}.pt"
        shutil.copy2(best, destination)
        record.update(status="completed", finished_at_utc=datetime.now(timezone.utc).isoformat(),
                      wall_seconds=time.perf_counter()-started, run_directory=str(actual_run),
                      best_weights=str(destination), best_weights_sha256=digest(destination),
                      metrics={key:float(value) for key,value in model.trainer.metrics.items()},
                      device=str(model.trainer.device))
    except Exception as exc:
        record.update(status="failed", error=repr(exc), wall_seconds=time.perf_counter()-started)
        raise
    finally:
        run_record.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
