"""Practical replacement comparison: trained YOLO11n versus existing deployed v8n."""
from pathlib import Path
import json
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_experiment import (CONFIG, read, write, sha, utc, status, libraries,
    metric_self_check, baseline_check, train_run, collect, tune, counts, bootstrap, verify_inputs)


def summarize_deployment(cfg):
    import numpy as np
    names = {"v8n_adopted": Path(cfg["baseline"]["path"])}
    for seed in cfg["seeds"]:
        for checkpoint in ("best", "last"):
            name=f"11n_seed{seed}_{checkpoint}"
            names[name]=HERE/"runs"/f"11n_seed{seed}"/f"weights/{checkpoint}.pt"
    revision=read(HERE/"scope_revision.json")
    result={"status":"completed", "comparison_kind":"practical_replacement_against_existing_deployment",
        "protocol_sha256":sha(CONFIG), "scope_revision":revision, "models":{}, "deployment_comparisons":{},
        "limitations":[x for x in cfg["limitations"] if not x.startswith("Same COCO dataset")] + ["No retrained v8n control used; comparison does not isolate architecture from task-training history.",
        "Existing v8n checkpoint and deployment thresholds historically used all 59 validation pages. The 29 pages are a protected score subset for the new 11n runs, not an untouched test for the incumbent."]}
    all_ids=set(cfg["calibration_cases"])|set(cfg["audit_cases"])
    historical_thresholds=read(cfg["baseline"]["reference"])["class_thresholds"]
    for name,weights in names.items():
        saved=collect(cfg,name,weights)
        entry={"weights":str(weights),"weights_sha256":sha(weights),"modes":{}}
        for mode in ("whole","sliced","fused"):
            thresholds,details=tune(cfg,saved,mode)
            entry["modes"][mode]={"thresholds_from_calibration":thresholds,"threshold_diagnostics":details,
                "audit_global_0_2":counts(saved,mode,set(cfg["audit_cases"]),{}),
                "audit_calibrated":counts(saved,mode,set(cfg["audit_cases"]),thresholds),
                "calibration_global_0_2":counts(saved,mode,set(cfg["calibration_cases"]),{}),
                "calibration_calibrated":counts(saved,mode,set(cfg["calibration_cases"]),thresholds),
                "full59_global_0_2_development_only":counts(saved,mode,all_ids,{}),
                "latency_ms_median":float(np.median([p["timings_ms"][mode] for p in saved["cases"]]))}
        if name=="v8n_adopted":
            entry["historical_deployment_thresholds_audit"]=counts(saved,"fused",set(cfg["audit_cases"]),historical_thresholds)
            entry["historical_deployment_thresholds_full59"]=counts(saved,"fused",all_ids,historical_thresholds)
        result["models"][name]=entry
        write(HERE/"comparison.json",result)
    baseline=result["models"]["v8n_adopted"]
    actual=baseline["historical_deployment_thresholds_audit"]
    for seed in cfg["seeds"]:
        new=result["models"][f"11n_seed{seed}_best"]["modes"]["fused"]
        result["deployment_comparisons"][str(seed)]={
            "against_actual_deployment":{"delta_f1":new["audit_calibrated"]["overall"]["f1"]-actual["overall"]["f1"],
                "bootstrap":bootstrap(actual,new["audit_calibrated"],cfg["case_groups"])},
            "same_calibration_method":{"delta_f1":new["audit_calibrated"]["overall"]["f1"]-baseline["modes"]["fused"]["audit_calibrated"]["overall"]["f1"],
                "bootstrap":bootstrap(baseline["modes"]["fused"]["audit_calibrated"],new["audit_calibrated"],cfg["case_groups"])},
            "fixed_global_0_2":{"delta_f1":new["audit_global_0_2"]["overall"]["f1"]-baseline["modes"]["fused"]["audit_global_0_2"]["overall"]["f1"]}}
    result["mean_delta_against_actual_deployment"]=float(np.mean([x["against_actual_deployment"]["delta_f1"] for x in result["deployment_comparisons"].values()]))
    result["completed_at_utc"]=utc()
    verify_inputs(cfg)
    result["frozen_inputs_unchanged_after_experiment"]=True
    write(HERE/"comparison.json",result)
    b=baseline["modes"]["fused"]
    lines=["# YOLO11n与现有YOLOv8n替换实验", "", "本次回答新11n方案能否替换现有部署v8n。新增v8n重训已取消，部分59轮对照权重只归档，不用于正式比较。原protocol.json保留，范围修订见scope_revision.json。", "",
        "新11n的两个种子使用同一418张训练样本、30张校准页、29张评分页。评分页不用于新模型选checkpoint或调阈值。现有v8n有多阶段PCB训练历史，并曾使用全部59页做开发；它作为实际部署参照，不称为同初始化架构对照。", "",
        f"现有部署工作点：本次29页F1={actual['overall']['f1']:.4%}，P={actual['overall']['precision']:.4%}，R={actual['overall']['recall']:.4%}；原59页F1={baseline['historical_deployment_thresholds_full59']['overall']['f1']:.4%}已复现。", "",
        "| 模型 | 评分集固定conf0.2 F1 | 同法30页校准阈值F1 | 校准后P | 校准后R | fused中位ms |", "|---|---:|---:|---:|---:|---:|"]
    for name,entry in result["models"].items():
        m=entry["modes"]["fused"]; g=m["audit_global_0_2"]["overall"]; c=m["audit_calibrated"]["overall"]
        lines.append(f"| {name} | {g['f1']:.4%} | {c['f1']:.4%} | {c['precision']:.4%} | {c['recall']:.4%} | {m['latency_ms_median']:.1f} |")
    lines += ["", "## 新11n best对现有部署工作点的差异", ""]
    for seed,comparison in result["deployment_comparisons"].items():
        x=comparison["against_actual_deployment"]; lo,hi=x["bootstrap"]["ci95_delta_f1"]
        lines.append(f"- seed {seed}：{x['delta_f1']*100:+.3f}个百分点；条件分组bootstrap95%区间[{lo*100:+.3f}, {hi*100:+.3f}]。")
    lines += ["", f"两次11n best的平均差：{result['mean_delta_against_actual_deployment']*100:+.3f}个百分点。未按评分集挑选更高的种子或best/last来作为唯一结果。", "", "## 限制", ""]
    lines += ["- "+x for x in result["limitations"]]
    lines += ["", "原部署权重与配置未替换。全部原始预测、阈值、逐类/逐页计数和whole/sliced结果保留在evaluation/与comparison.json。"]
    (HERE/"comparison.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    status(status="completed",comparison=str(HERE/"comparison.md"),mean_delta_against_actual_deployment=result["mean_delta_against_actual_deployment"])
    print(json.dumps({"deployment_comparison_completed":True,"mean_delta_against_actual_deployment":result["mean_delta_against_actual_deployment"]}),flush=True)


def main():
    cfg=read(CONFIG)
    revision={"revised_at_utc":utc(), "original_protocol_sha256":sha(CONFIG),
        "reason":"User clarified direct comparison with existing v8n is the intended replacement decision; new v8n training is not necessary.",
        "training_roster":[f"11n_seed{s}" for s in cfg["seeds"]], "baseline":"existing deployed v8n",
        "unchanged_conditions":["data and source group split","initial 11n weights","training recipe and two seeds","inference","calibration-only threshold search","29-page score subset"],
        "excluded_control":"v8n_seed20261012 partial 59 epochs; v8n_seed20261015 not started",
        "interpretation":"Practical comparison of available model solutions, not causal architecture-only comparison."}
    if not (HERE/"scope_revision.json").exists():
        write(HERE/"scope_revision.json",revision)
    (HERE/"README.md").write_text("# YOLO11n与现有v8n替换试验\n\n按用户澄清，主目标是比较新11n方案与现有部署v8n。新增v8n对照已停止；59轮部分记录仅归档，不用于比较。\n\n两个11n种子沿用冻结的v3末阶段训练参数、418张训练样本、30页校准集和29页评分集。新模型选checkpoint和调阈值只用校准页。同组页面不跨集合。原始protocol.json保留，scope_revision.json记录范围修订。\n\n比较固定conf0.2、相同校准方法及现有v8n实际部署工作点。报告best和last、逐类、逐页、速度和分组bootstrap区间。现有v8n拥有多阶段任务训练历史，并曾使用全部59页做开发；结果回答实际替换问题，不称为架构因果对照或全新工程独立测试。\n\nrun_deployment_experiment.py为当前入口，analyze_results.py为完成后的评分集AP和详细诊断。正式报告为comparison.md。原部署权重与配置保持。\n",encoding="utf-8")
    libraries(); metric_self_check(); baseline_check(cfg)
    for seed in cfg["seeds"]:
        train_run(cfg,f"11n_seed{seed}")
    summarize_deployment(cfg)


if __name__=="__main__":
    try:
        main()
    except BaseException as exc:
        status(status="failed",error=repr(exc))
        raise
