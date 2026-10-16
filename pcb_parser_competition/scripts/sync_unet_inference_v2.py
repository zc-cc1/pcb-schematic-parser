"""Selectively copy accepted U-Net inference files, keeping backups and hashes."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=Path, required=True)
    args = parser.parse_args()
    target = args.target.resolve(strict=True)
    if target == ROOT or not (target / "scripts/train_unet.py").is_file():
        raise ValueError("Expected the existing destination U-Net training project")
    config = json.loads((ROOT / "configs/unet_inference_v2.json").read_text(encoding="utf-8"))
    if digest(target / "runs/training/unet_wire_pin_v1/best.pt") != config["checkpoint_sha256"]:
        raise ValueError("Target trained checkpoint no longer matches calibrated model")
    protected_names = ["scripts/run_baseline.py", "scripts/train_detector.py", "configs/baseline_v3_adopted.json",
                       "configs/class_thresholds_v3_adopted.json", "weights/detector_dataset300_v3_nano.pt",
                       "runs/training/unet_wire_pin_v1/best.pt", "runs/training/unet_wire_pin_v1/last.pt",
                       "configs/unet_label_reviews.json"]
    protected = {name: digest(target / name) for name in protected_names if (target / name).is_file()}
    names = ["src/pcb_parser/unet_inference.py", "scripts/run_unet.py", "scripts/run_unet_optimized.ps1",
             "configs/unet_inference_v2.json", "tests/test_unet_inference.py", "docs/unet_inference_optimization_v2.md",
             "runs/unet_inference_v2/adopted_report.md", "runs/unet_inference_v2/adopted_comparison.json",
             "runs/unet_inference_v2/comparison.json", "runs/unet_inference_v2/frozen_inference.json",
             "runs/unet_inference_v2/split.json"]
    names += [path.relative_to(ROOT).as_posix() for path in sorted((ROOT / "runs/unet_inference_v2/adopted_examples").glob("*.png"))]
    folder = target / "data/local_imports" / ("unet_inference_v2_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    folder.mkdir(parents=True, exist_ok=False)
    records = []
    for name in names:
        source, destination = ROOT / name, target / name
        if target not in destination.resolve().parents:
            raise ValueError("Destination escapes the intended project")
        destination.parent.mkdir(parents=True, exist_ok=True)
        backup = None
        if destination.exists() and digest(destination) != digest(source):
            backup = folder / "backup" / name
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(destination, backup)
        if name.endswith("adopted_report.md"):
            # Keep copied report image links self-contained in the target project.
            body = source.read_text(encoding="utf-8").replace(ROOT.as_posix(), target.as_posix())
            destination.write_text(body, encoding="utf-8")
        else:
            shutil.copy2(source, destination)
            assert digest(source) == digest(destination)
        records.append({"file": name, "source_sha256": digest(source), "destination_sha256": digest(destination),
                        "backup": str(backup) if backup else None, "report_links_rebased": name.endswith("adopted_report.md")})
    assert protected == {name: digest(target / name) for name in protected}
    result = {"status": "copied_with_backups_YOLO_and_training_artifacts_preserved", "target": str(target),
              "files": records, "protected_sha256": protected}
    (folder / "import_manifest.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": result["status"], "file_count": len(records), "manifest": str(folder / "import_manifest.json")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
