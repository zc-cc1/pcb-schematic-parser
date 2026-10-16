"""Prepare a separate frozen-data experiment without using the GPU."""
from pathlib import Path
import copy
import hashlib
import json
import shutil
from datetime import datetime, timezone
import yaml

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / "y11n_matched_control_20261015"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    if (HERE / "protocol.json").exists():
        print("Already prepared; frozen protocol preserved.")
        return
    cfg = copy.deepcopy(json.loads((PARENT / "protocol.json").read_text(encoding="utf-8")))
    assert sha(PARENT / "protocol.json") == "1d56120c702133d488deca1580f5fa2fb9faa15e3b3763fbfe379d1b7d0423ad"
    cfg["parent_protocol"] = {"path": str(PARENT / "protocol.json"), "sha256": sha(PARENT / "protocol.json")}
    rows = [json.loads(line) for line in (PARENT / "dataset/manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    paths = {}
    for row in rows:
        for key in ("image", "label"):
            source = Path(row[key])
            assert sha(source) == row[key + "_sha256"]
            target = HERE / "dataset" / source.relative_to(PARENT / "dataset")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            assert sha(target) == row[key + "_sha256"]
            paths[str(source)] = str(target)
            row[key] = str(target)
    manifest = HERE / "dataset/manifest.jsonl"
    manifest.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    cfg["copied_manifest_sha256"] = sha(manifest)
    for split in ("train", "calibration", "audit"):
        lines = (PARENT / "dataset" / (split + ".txt")).read_text(encoding="utf-8").splitlines()
        (HERE / "dataset" / (split + ".txt")).write_text("\n".join(paths[p] for p in lines) + "\n", encoding="utf-8")
    data = yaml.safe_load((PARENT / "dataset/dataset.yaml").read_text(encoding="utf-8"))
    data["path"] = str(HERE / "dataset")
    dataset = HERE / "dataset/dataset.yaml"
    dataset.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    (HERE / "dataset/audit_dataset.yaml").write_text(yaml.safe_dump({**data, "val": "audit.txt"}, sort_keys=False), encoding="utf-8")
    cfg.update(dataset=str(dataset), dataset_yaml_sha256=sha(dataset), seeds=[20261012], run_order=["11s_seed20261012"])
    initial = Path("C:/Users/Pc/Desktop/新建文件夹/pcb_parser_competition/weights/yolo11s.pt")
    expected = "85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5"
    assert sha(initial) == expected
    destination = HERE / "initial_weights/yolo11s.pt"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(initial, destination)
    cfg["initial_weights"]["11s"] = {"path": str(destination), "sha256": expected,
        "source_url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt",
        "cached_source": str(initial), "classes": cfg["initial_weights"]["11n"]["classes"]}
    shutil.copy2(PARENT / "run_experiment.py", HERE / "matched_helpers.py")
    shutil.copy2(PARENT / "yolo11n.pt", HERE / "yolo11n.pt")
    cfg["code_inputs"].append({"path": str(HERE / "matched_helpers.py"), "sha256": sha(HERE / "matched_helpers.py")})
    cfg["references"] = {}
    for key, filename in (("comparison", "resumed_v8n_comparison.json"), ("diagnostics", "resumed_v8n_diagnostics.json")):
        entry = {"path": str(PARENT / filename), "sha256": sha(PARENT / filename)}
        cfg["references"][key] = entry
        cfg["original_inputs"].append(entry)
    cfg["original_inputs"].append(cfg["parent_protocol"])
    cfg["comparability"] = "One matched seed 20261012, architecture-specific COCO weights; identical final-stage recipe and frozen train/calibration/audit content. v8n resumed after epoch59; incumbent is a separate historical reference."
    cfg["limitations"] = ["Only one matched YOLO11s seed; no claim of stable architecture superiority.",
        "Old development pages already examined; not a new blind engineering-source test.",
        "Architecture-specific COCO pretraining differs; incumbent has task-specific warm starts.",
        "v8n resumed after epoch59 without complete data-loader/RNG state.",
        "BBox/category metrics are proxies, not official ComponentF1/PinF1/NetF1."]
    cfg["idle_policy"] = {"consecutive_samples": 3, "sample_seconds": 10, "max_utilization_percent": 10,
                          "min_free_mib": 6500, "require_no_compute_workload_process": True}
    cfg["experiment_created_at_utc"] = datetime.now(timezone.utc).isoformat()
    (HERE / "protocol.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    (HERE / "status.json").write_text(json.dumps({"status": "waiting_for_gpu", "created_at_utc": cfg["experiment_created_at_utc"]}, indent=2), encoding="utf-8")
    (HERE / "README.md").write_text("# YOLO11s匹配训练实验\n\n等待GPU空闲后启动一个种子20261012。训练418张，校准30页，评分29页；1536、batch4、AMP、AdamW、lr0=0.0003、最多120轮、patience25及其他参数复制上一组冻结配方。独立保存数据副本、权重与报告；不改部署模型。若显存不足，不自动更改batch或分辨率。\n\n使用run_y11s.py --launch-if-idle检查并启动；完成后生成comparison.json和comparison.md，包含与现有v8n、同配方v8n及11n的P/R/F1、整图AP和逐类差异。\n", encoding="utf-8")
    print(json.dumps({"prepared": True, "experiment": str(HERE), "images": len(rows), "protocol_sha256": sha(HERE / "protocol.json")}))


if __name__ == "__main__":
    main()
