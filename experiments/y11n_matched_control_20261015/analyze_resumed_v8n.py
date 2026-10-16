"""Audit AP diagnostics for the resumed control; never select on audit results."""
from pathlib import Path
import csv
import json
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_experiment import read, write, sha, libraries, verify_inputs, utc


def main():
    import yaml
    cfg = read(HERE / "protocol.json")
    comparison = read(HERE / "resumed_v8n_comparison.json")
    assert comparison["status"] == "completed"
    verify_inputs(cfg)
    torch, _, YOLO = libraries()
    copied = [json.loads(line) for line in (HERE / "dataset/manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    val = sorted([r for r in copied if r["split"] == "val"], key=lambda r: r["case_id"])
    for split, expected in [
        ("train", [r["image"] for r in copied if r["split"] == "train"]),
        ("calibration", [r["image"] for r in val if r["case_id"] in cfg["calibration_cases"]]),
        ("audit", [r["image"] for r in val if r["case_id"] in cfg["audit_cases"]]),
    ]:
        assert (HERE / "dataset" / (split + ".txt")).read_text(encoding="utf-8").splitlines() == expected
    training_yaml = yaml.safe_load(Path(cfg["dataset"]).read_text(encoding="utf-8"))
    audit_yaml = HERE / "dataset/audit_dataset.yaml"
    assert yaml.safe_load(audit_yaml.read_text(encoding="utf-8")) == {**training_yaml, "val": "audit.txt"}
    resume = read(HERE / "v8n_resume_record.json")
    assert resume["status"] == "trained"
    v8args = read(HERE / "records/v8n_seed20261012_resumed_args.json")
    y11args = read(HERE / "records/11n_seed20261012_args.json")
    allowed = {"model", "name", "save_dir", "resume"}
    assert not {k: [v8args.get(k), y11args.get(k)] for k in set(v8args) | set(y11args)
                if k not in allowed and v8args.get(k) != y11args.get(k)}
    diagnostics = {
        "created_at_utc": utc(), "protocol_sha256": sha(HERE / "protocol.json"),
        "selection_policy": "best.pt selected only on 30 calibration pages; last.pt diagnostic only; audit AP does not select a checkpoint",
        "effective_args_and_file_membership_verified": True, "training": {},
        "audit_ap": dict(read(HERE / "diagnostics.json")["audit_ap"]),
        "ap_recipe": {"imgsz": 1536, "batch": 4, "workers": 2, "device": 0,
                      "half": False, "conf": .001, "iou": .7, "max_det": 1000},
    }
    for run in ("v8n_seed20261012", "11n_seed20261012", "11n_seed20261015"):
        with (HERE / "runs" / run / "results.csv").open(encoding="utf-8") as stream:
            rows = [{k.strip(): v for k, v in r.items()} for r in csv.DictReader(stream)]
        assert [int(float(r["epoch"])) for r in rows] == list(range(1, len(rows) + 1))
        peak = max(float(r["metrics/mAP50-95(B)"]) for r in rows)
        peaks = [int(float(r["epoch"])) for r in rows if float(r["metrics/mAP50-95(B)"]) == peak]
        diagnostics["training"][run] = {"completed_epochs": len(rows), "calibration_peak_epochs": peaks,
                                        "best_calibration_mAP50_95": peak}
    for checkpoint in ("best", "last"):
        name = "v8n_seed20261012_" + checkpoint
        entry = comparison["models"][name]
        assert sha(entry["weights"]) == entry["weights_sha256"]
        write(HERE / "resumed_v8n_analysis_status.json", {"status": "audit_ap_evaluation", "model": name, "updated_at_utc": utc()})
        model = YOLO(entry["weights"])
        metrics = model.val(data=str(audit_yaml), **diagnostics["ap_recipe"],
                            plots=False, save_json=False, project=str(HERE / "audit_ap"),
                            name="resumed_" + name, exist_ok=False, verbose=False)
        diagnostics["audit_ap"][name] = {
            "mAP50": float(metrics.box.map50), "mAP50_95": float(metrics.box.map),
            "framework_precision": float(metrics.box.mp), "framework_recall": float(metrics.box.mr),
            "weights_sha256": entry["weights_sha256"],
            "definition": "Ultralytics whole-image AP on the same 29 audit pages; distinct from fused F1",
        }
        del model
        torch.cuda.empty_cache()
    verify_inputs(cfg)
    diagnostics["frozen_inputs_unchanged_after_analysis"] = True
    write(HERE / "resumed_v8n_diagnostics.json", diagnostics)
    lines = ["## 补充：整图AP与训练轮次", "", "best仅按30页校准集选择；以下29页评分集AP用于诊断，不用于重新选择模型。", "",
             "| 模型 | 实际轮数 | 校准集mAP50-95峰值轮次 |", "|---|---:|---:|"]
    for name, row in diagnostics["training"].items():
        lines.append(f"| {name} | {row['completed_epochs']} | {', '.join(map(str, row['calibration_peak_epochs']))} |")
    lines += ["", "AP采用整图1536、conf0.001、NMS IoU0.7、max_det1000，和前述整图+切片融合F1口径不同。", "",
              "| 模型 | 评分集mAP50 | 评分集mAP50-95 |", "|---|---:|---:|"]
    for name in ("v8n_adopted", "v8n_seed20261012_best", "v8n_seed20261012_last", "11n_seed20261012_best", "11n_seed20261012_last", "11n_seed20261015_best"):
        row = diagnostics["audit_ap"][name]
        lines.append(f"| {name} | {row['mAP50']:.4%} | {row['mAP50_95']:.4%} |")
    deployed = comparison["models"]["v8n_adopted"]["historical_deployment_thresholds_audit"]["overall"]
    lines += ["", f"原部署v8n沿用原部署阈值时，这29页的F1为{deployed['f1']:.4%}（TP={deployed['tp']}，FP={deployed['fp']}，FN={deployed['fn']}）。该阈值历史上使用过全部59页开发数据，和新模型仅30页校准的结果不能视为同等选择条件。"]
    report = HERE / "resumed_v8n_comparison.md"
    existing = report.read_text(encoding="utf-8").split("\n## 补充：整图AP与训练轮次")[0]
    report.write_text(existing.rstrip() + "\n\n" + "\n".join(lines) + "\n", encoding="utf-8")
    write(HERE / "resumed_v8n_analysis_status.json", {"status": "completed", "updated_at_utc": utc()})
    print(json.dumps({"resumed_analysis_completed": True, "report": str(report)}), flush=True)


if __name__ == "__main__":
    main()
