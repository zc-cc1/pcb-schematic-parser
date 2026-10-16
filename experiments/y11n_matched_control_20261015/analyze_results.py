"""Post-training diagnostics; does not select checkpoints or change thresholds."""
from pathlib import Path
import csv
import json
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_experiment import read, write, sha, libraries, verify_inputs, utc


def main():
    import numpy as np
    import yaml
    cfg = read(HERE / "protocol.json")
    result = read(HERE / "comparison.json")
    assert result["status"] == "completed" and (HERE/"comparison.md").exists()
    verify_inputs(cfg)
    torch, _, YOLO = libraries()
    deployment_only=result.get("comparison_kind")=="practical_replacement_against_existing_deployment"
    training_roster=[f"11n_seed{s}" for s in cfg["seeds"]] if deployment_only else cfg["run_order"]
    argument_pairs=[(training_roster[0],training_roster[1],{"name","save_dir","seed"})] if deployment_only else [
        (f"v8n_seed{s}",f"11n_seed{s}",{"model","name","save_dir"}) for s in cfg["seeds"]]
    for left,right,allowed in argument_pairs:
        a=read(HERE/"records"/(left+"_args.json")); b=read(HERE/"records"/(right+"_args.json"))
        differences={k:[a.get(k),b.get(k)] for k in set(a)|set(b) if k not in allowed and a.get(k)!=b.get(k)}
        assert not differences, f"Unmatched effective arguments: {left}/{right}: {differences}"
    copied_rows=[json.loads(s) for s in (HERE/"dataset/manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    val_rows=sorted([r for r in copied_rows if r["split"]=="val"],key=lambda r:r["case_id"])
    for label,expected in [("train",[r["image"] for r in copied_rows if r["split"]=="train"]),
                           ("calibration",[r["image"] for r in val_rows if r["case_id"] in cfg["calibration_cases"]]),
                           ("audit",[r["image"] for r in val_rows if r["case_id"] in cfg["audit_cases"]])]:
        assert (HERE/"dataset"/f"{label}.txt").read_text(encoding="utf-8").splitlines()==expected
    validation_data = yaml.safe_load(Path(cfg["dataset"]).read_text(encoding="utf-8"))
    validation_data["val"] = "audit.txt"
    audit_yaml = HERE / "dataset/audit_dataset.yaml"
    audit_yaml.write_text(yaml.safe_dump(validation_data, sort_keys=False), encoding="utf-8")
    diagnostics = {"created_at_utc": utc(), "selection_policy": "best.pt preselected only on calibration mAP50-95; last.pt diagnostic only",
                   "training": {}, "audit_ap": {}, "class_differences": [],
                   "memory_caveat": "Training runner records cumulative process CUDA allocation peaks, not independently reset per-model peaks; do not rank model VRAM using this column.",
                   "effective_args_between_listed_runs_verified": True, "training_roster":training_roster,
                   "train_calibration_audit_file_membership_verified": True}
    for run in training_roster:
        with (HERE/"runs"/run/"results.csv").open(encoding="utf-8") as stream:
            rows = [{k.strip():v for k,v in r.items()} for r in csv.DictReader(stream)]
        best = max(rows, key=lambda r: float(r["metrics/mAP50-95(B)"]))
        record = read(HERE/"records"/(run+".json"))
        diagnostics["training"][run] = {"epochs_completed": len(rows), "best_calibration_epoch": int(float(best["epoch"])),
            "best_calibration_mAP50": float(best["metrics/mAP50(B)"]),
            "best_calibration_mAP50_95": float(best["metrics/mAP50-95(B)"]),
            "best_epoch_source": "maximum calibration mAP50-95 in CSV; checkpoint history can contain all epochs after optimizer stripping",
            "seconds": record["seconds_this_process"], "parameters": record["parameter_count"],
            "peak_allocated_mib": record["peak_allocated_mib"], "effective_arguments_verified": record["effective_arguments_verified"]}
    for name, entry in result["models"].items():
        write(HERE/"analysis_status.json", {"status":"audit_ap_evaluation", "model":name,"updated_at_utc":utc()})
        checkpoint = Path(entry["weights"])
        assert sha(checkpoint) == entry["weights_sha256"]
        model = YOLO(str(checkpoint))
        metrics = model.val(data=str(audit_yaml), imgsz=1536, batch=4, workers=2, device=0,
            half=False, conf=.001, iou=.7, max_det=1000, plots=False, save_json=False,
            project=str(HERE/"audit_ap"), name=name, exist_ok=False, verbose=False)
        diagnostics["audit_ap"][name] = {"mAP50":float(metrics.box.map50), "mAP50_95":float(metrics.box.map),
            "framework_precision":float(metrics.box.mp), "framework_recall":float(metrics.box.mr),
            "definition":"Ultralytics whole-image AP on same 29 audit pages; distinct from fixed/conf-calibrated fused F1"}
        del model
        torch.cuda.empty_cache()
        write(HERE/"diagnostics.json",diagnostics)
    from evaluate_detector import metric
    for cls in cfg["names"].values():
        values = {"class":cls}
        for architecture in ("v8n","11n"):
            per_seed=[]
            for seed in cfg["seeds"]:
                if deployment_only and architecture=="v8n":
                    classes=result["models"]["v8n_adopted"]["historical_deployment_thresholds_audit"]["by_class"]
                else:
                    classes=result["models"][f"{architecture}_seed{seed}_best"]["modes"]["fused"]["audit_calibrated"]["by_class"]
                per_seed.append(classes.get(cls,metric({"tp":0,"fp":0,"fn":0})))
            values[architecture+"_mean_f1"] = float(np.mean([r["f1"] for r in per_seed]))
            values[architecture+"_per_seed"] = per_seed
        values["audit_gt"] = values["v8n_per_seed"][0]["tp"] + values["v8n_per_seed"][0]["fn"]
        values["delta_f1"] = values["11n_mean_f1"]-values["v8n_mean_f1"]
        diagnostics["class_differences"].append(values)
    for architecture in (("11n",) if deployment_only else ("v8n","11n")):
        diagnostics[architecture+"_best_seed_f1"] = [result["models"][f"{architecture}_seed{s}_best"]["modes"]["fused"]["audit_calibrated"]["overall"]["f1"] for s in cfg["seeds"]]
    diagnostics["post_analysis_original_inputs_unchanged"] = True
    verify_inputs(cfg)
    write(HERE/"diagnostics.json",diagnostics)
    lines=["", "## 训练与评分集AP", "", "| 模型/种子 | 实际轮数 | 校准最佳轮 | 校准mAP50-95 | 参数量 | 训练分钟 | 进程累计峰值MiB |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name,r in diagnostics["training"].items():
        lines.append(f"| {name} | {r['epochs_completed']} | {r['best_calibration_epoch']} | {r['best_calibration_mAP50_95']:.4%} | {r['parameters']} | {r['seconds']/60:.1f} | {r['peak_allocated_mib']:.0f} |")
    lines += ["", "显存列为训练进程截至该组结束的累计PyTorch分配峰值，未逐模型重置，不能用于比较模型独立显存占用。", "", "评分集AP采用整图推理、conf0.001、NMS IoU0.7、max_det1000；与前面的融合固定阈值F1口径不同。", "", "| 模型 | 评分集mAP50 | 评分集mAP50-95 |", "|---|---:|---:|"]
    for name,r in diagnostics["audit_ap"].items():
        lines.append(f"| {name} | {r['mAP50']:.4%} | {r['mAP50_95']:.4%} |")
    lines += ["", "## 逐类差异（11n两个种子best平均；v8n为现有部署工作点）" if deployment_only else "## 逐类差异（两个种子best平均，评分集校准阈值）", "", "| 类别 | 评分集GT数 | v8n F1 | 11n平均F1 | 差值（百分点） |", "|---|---:|---:|---:|---:|"]
    for r in sorted(diagnostics["class_differences"],key=lambda r: -r["audit_gt"]):
        lines.append(f"| {r['class']} | {r['audit_gt']} | {r['v8n_mean_f1']:.4%} | {r['11n_mean_f1']:.4%} | {r['delta_f1']*100:+.3f} |")
    lines += ["", "这里只评价检测器，没有声称元件身份/文本、引脚或网络得分已经提高。零GT及极少GT类别的F1不能支持泛化结论。"]
    report = HERE/"comparison.md"
    existing = report.read_text(encoding="utf-8")
    report.write_text(existing+"\n".join(lines)+"\n",encoding="utf-8")
    write(HERE/"analysis_status.json",{"status":"completed","updated_at_utc":utc()})
    print(json.dumps({"analysis_completed":True,"report":str(report)}),flush=True)


if __name__ == "__main__":
    main()
