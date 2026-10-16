"""Prepare one warm-start refinement trial and the user-selected YOLO11s config."""
from pathlib import Path
from datetime import datetime, timezone
import copy
import csv
import hashlib
import json
import shutil
import yaml

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / "y11s_matched_control_20261015"
PROJECT = HERE.parents[1] / "pcb_parser_competition"
CHANGES = {"epochs": 40, "patience": 12, "lr0": .00005, "warmup_epochs": 1, "mosaic": 0., "close_mosaic": 0}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    assert not (HERE / "protocol.json").exists(), "Preserve an already-prepared trial."
    cfg = copy.deepcopy(read(PARENT / "protocol.json"))
    previous = read(PARENT / "comparison.json")
    record = read(PARENT / "training_record.json")
    assert previous["status"] == "completed" and record["status"] == "trained"
    assert record["best_calibration_epoch"] == 76 and record["completed_epochs"] == 101
    source = PARENT / "runs/11s_seed20261012/weights/best.pt"
    assert sha(source) == record["best_sha256"] == previous["models"]["11s_seed20261012_best"]["weights_sha256"]
    rows = [json.loads(line) for line in (PARENT / "dataset/manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    paths = {}
    for row in rows:
        for key in ("image", "label"):
            old = Path(row[key]); assert sha(old) == row[key + "_sha256"]
            new = HERE / "dataset" / old.relative_to(PARENT / "dataset")
            new.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(old, new)
            assert sha(new) == row[key + "_sha256"]
            paths[str(old)] = str(new); row[key] = str(new)
    manifest = HERE / "dataset/manifest.jsonl"
    manifest.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    for split in ("train", "calibration", "audit"):
        members = (PARENT / "dataset" / (split + ".txt")).read_text(encoding="utf-8").splitlines()
        (HERE / "dataset" / (split + ".txt")).write_text("\n".join(paths[p] for p in members) + "\n", encoding="utf-8")
    data = yaml.safe_load((PARENT / "dataset/dataset.yaml").read_text(encoding="utf-8")); data["path"] = str(HERE / "dataset")
    (HERE / "dataset/dataset.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    (HERE / "dataset/audit_dataset.yaml").write_text(yaml.safe_dump({**data, "val": "audit.txt"}, sort_keys=False), encoding="utf-8")
    initial = HERE / "initial_weights/yolo11s_parent_best.pt"; initial.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, initial)
    cfg["initial_weights"]["11s_finetune"] = {"path": str(initial), "sha256": sha(initial), "classes": cfg["names"],
        "source": str(source), "parent_selected_epoch": 76, "policy": "Warm start from calibration-selected best, not audit-selected last; new optimizer and new 40-epoch schedule."}
    cfg["parent_protocol"] = {"path": str(PARENT / "protocol.json"), "sha256": sha(PARENT / "protocol.json")}
    cfg["parent_training"] = copy.deepcopy(cfg["training"])
    cfg["training"].update(CHANGES)
    cfg["fine_tune_changes"] = CHANGES
    cfg["dataset"] = str(HERE / "dataset/dataset.yaml")
    cfg["dataset_yaml_sha256"] = sha(cfg["dataset"]); cfg["copied_manifest_sha256"] = sha(manifest)
    cfg["run_order"] = ["11s_finetune_seed20261012"]
    cfg["seeds"] = [20261012]
    cfg["references"] = {"comparison": {"path": str(PARENT / "comparison.json"), "sha256": sha(PARENT / "comparison.json")}}
    cfg["original_inputs"] += [cfg["parent_protocol"], cfg["references"]["comparison"], {"path": str(source), "sha256": sha(source)}]
    for old_name, new_name in (("matched_helpers.py", "matched_helpers.py"), ("run_y11s.py", "experiment_runtime.py"), ("yolo11n.pt", "yolo11n.pt")):
        shutil.copy2(PARENT / old_name, HERE / new_name)
    for name in ("matched_helpers.py", "experiment_runtime.py"):
        cfg["code_inputs"].append({"path": str(HERE / name), "sha256": sha(HERE / name)})
    font = PARENT / "ultralytics/Ultralytics/Arial.ttf"
    assert sha(font) == "525979822591a3447cfc49d943d6f7683508e25543407871c0ed8fed05fd2bd9"
    for directory in (HERE / "ultralytics/Ultralytics", HERE / "Ultralytics"):
        directory.mkdir(parents=True, exist_ok=True); shutil.copy2(font, directory / "Arial.ttf")
    with (PARENT / "runs/11s_seed20261012/results.csv").open(encoding="utf-8") as stream:
        history = [{k.strip(): v for k, v in row.items()} for row in csv.DictReader(stream)]
    parent_peak = max(history, key=lambda r: float(r["metrics/mAP50-95(B)"]))
    cfg["parent_calibration_mAP50_95"] = float(parent_peak["metrics/mAP50-95(B)"])
    cfg["recommendation_gate"] = {"minimum_calibrated_fused_f1_gain": .002, "maximum_capacitor_f1_drop": .01, "maximum_calibration_map_drop": .005,
        "selection_data": "30 calibration pages only; 29 score pages are report-only and already examined development data"}
    cfg["comparability"] = "One planned YOLO11s warm-start refinement recipe versus its unchanged parent, same train/calibration/audit content and inference/threshold policy."
    cfg["limitations"] = ["One refinement run and one seed; no per-hyperparameter causal attribution.", "Old development data already examined; no independent test claim.", "New optimizer warm-start stage; not resuming the old schedule.", "Only bbox/category metrics; not official pin, net or component identity scores."]
    cfg["prepared_at_utc"] = datetime.now(timezone.utc).isoformat()
    write(HERE / "protocol.json", cfg)
    write(HERE / "status.json", {"status": "waiting_for_gpu", "updated_at_utc": cfg["prepared_at_utc"]})
    # Prepare an explicit YOLO11s entry point without altering existing v8n/downstream cached runs.
    base_config = read(PROJECT / "configs/baseline_v3_adopted.json")
    base_weight = PROJECT / "weights/detector_yolo11s_base_20261015.pt"
    assert not base_weight.exists() or sha(base_weight) == sha(initial)
    shutil.copy2(initial, base_weight)
    thresholds = previous["models"]["11s_seed20261012_best"]["modes"]["fused"]["thresholds_from_calibration"]
    base_config.update(version="0.5-yolo11s-base", weights="weights/" + base_weight.name, weights_sha256=sha(base_weight),
        class_thresholds_file="configs/class_thresholds_y11s_base.json", status="user_selected_yolo11s_refinement_baseline",
        recommended_mode_basis="User-selected YOLO11s architecture; checkpoint chosen at calibration epoch76, thresholds fitted on 30 calibration pages.")
    base_config["detector"]["class_confidence"] = thresholds
    base_config["detector_provenance"] = {"experiment": str(PARENT), "checkpoint": "best.pt", "calibration_epoch": 76,
        "audit_fused_f1": previous["models"]["11s_seed20261012_best"]["modes"]["fused"]["audit_calibrated"]["overall"]["f1"]}
    baseline_path = PROJECT / "configs/baseline_y11s.json"
    assert not baseline_path.exists(), "Preserve an existing YOLO11s integration config."
    write(baseline_path, base_config)
    write(PROJECT / "configs/class_thresholds_y11s_base.json", {"thresholds": thresholds, "calibration_cases": cfg["calibration_cases"], "weights_sha256": sha(base_weight)})
    write(HERE / "integration.json", {"yolo11s_baseline_config": str(baseline_path), "weights": str(base_weight),
        "previous_v8n_configs_preserved": True, "fine_tune_outputs": "A separate candidate config is written after evaluation; no automatic deployment replacement."})
    (HERE / "README.md").write_text("# YOLO11s小学习率无Mosaic精修\n\n从原11s第76轮best开始新精修阶段。418张训练、30页校准、29页评分不变；1536、batch4、AdamW等保持，lr0=0.00005，最多40轮、patience12、warmup1、mosaic0、close_mosaic0。初始模型、图像和标注均另存副本。上一轮第101轮早停，尚未执行原定106轮开始的无Mosaic阶段，本次检验精修是否有收益。\n\n以相同整图/切片/WBF和阈值政策对比父11s。推荐门槛只使用30页校准集；29页评分只报告结果，属于已查看的旧开发数据。保留best和last，last仅作诊断。GPU空闲后以run_finetune.py --launch-if-idle启动。\n\nYOLO11s独立推理配置：pcb_parser_competition/configs/baseline_y11s.json；微调结束另写候选配置，不覆盖旧v8n和已有下游缓存。\n", encoding="utf-8")
    print(json.dumps({"prepared": True, "experiment": str(HERE), "training_changes": CHANGES, "images": len(rows), "yolo11s_config": str(baseline_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
