"""Resume the preserved 59-epoch COCO v8n control, then compare frozen models."""
from pathlib import Path
import copy
import csv
import json
import math
import os
import shutil
import sys
import time

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from run_experiment import (CONFIG, ROOT, read, write, sha, utc, libraries, verify_inputs,
    metric_self_check, baseline_check, collect, tune, counts, bootstrap, status as base_status)
RUN="v8n_seed20261012"
FOLDER=HERE/"runs"/RUN
RECORD=HERE/"records"/(RUN+".json")
ARCHIVE=HERE/"resume_archive"/"v8n_seed20261012_59epochs"


def emit(**values):
    base_status(**values)
    write(HERE/"resume_v8n_status.json",{"updated_at_utc":utc(),"pid":os.getpid(),**values})


def csv_rows():
    with (FOLDER/"results.csv").open(encoding="utf-8") as stream:
        return [{k.strip():v for k,v in r.items()} for r in csv.DictReader(stream)]


def archive_prefix():
    if (ARCHIVE/"manifest.json").exists():
        for entry in read(ARCHIVE/"manifest.json"):
            assert sha(entry["archive"])==entry["sha256"]
        return
    paths=[FOLDER/"weights/last.pt",FOLDER/"weights/best.pt",FOLDER/"results.csv",
           FOLDER/"args.yaml",FOLDER/"class_transfer.json",RECORD,
           HERE/"records"/(RUN+"_args.json")]
    entries=[]
    for path in paths:
        destination=ARCHIVE/path.relative_to(HERE)
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,destination)
        assert sha(path)==sha(destination)
        entries.append({"source":str(path),"archive":str(destination),"sha256":sha(path)})
    write(ARCHIVE/"manifest.json",entries)


def train(cfg):
    torch,_,YOLO=libraries()
    from train_detector import ClassNameTransferTrainer
    verify_inputs(cfg)
    if (HERE/"v8n_resume_record.json").exists() and read(HERE/"v8n_resume_record.json").get("status")=="trained":
        return
    archive_prefix()
    checkpoint=FOLDER/"weights/last.pt"
    ckpt=torch.load(checkpoint,map_location="cpu",weights_only=False)
    prefix=csv_rows()
    completed=int(ckpt["epoch"])+1
    assert completed==len(prefix)==59, "This entry point resumes the original 59-epoch checkpoint only."
    assert sha(checkpoint)==read(ARCHIVE/"records"/(RUN+".json"))["partial_last_sha256"]
    assert [int(float(r["epoch"])) for r in prefix]==list(range(1,60))
    assert ckpt.get("optimizer") and ckpt.get("scaler") and ckpt.get("ema") is not None
    expected=dict(cfg["training"],seed=20261012)
    for k,v in expected.items():
        actual=ckpt["train_args"].get(k)
        assert (str(v)==str(actual) if k=="device" else v==actual),(k,v,actual)
    peak=max(prefix,key=lambda r:float(r["metrics/mAP50-95(B)"]))
    peak_epoch=int(float(peak["epoch"]))
    assert abs(float(peak["metrics/mAP50-95(B)"])-float(ckpt["best_fitness"]))<1e-5
    old_lrs=[g["lr"] for g in ckpt["optimizer"]["param_groups"]]
    old_scaler=ckpt["scaler"]
    old_updates=ckpt["updates"]
    old_optimizer_count=len(ckpt["optimizer"]["state"])
    before=read(RECORD)
    record={"status":"resuming","started_at_utc":utc(),"starting_completed_epochs":59,
        "target_total_epochs":120,"checkpoint_sha256_before":sha(checkpoint),
        "prior_record":before,"expected_training":expected,"initial_weights":cfg["initial_weights"]["v8n"],
        "prior_best_epoch":peak_epoch,"prior_best_fitness":float(ckpt["best_fitness"]),
        "prefix_csv_training_seconds":float(prefix[-1]["time"]),
        "resume_caveat":"Framework checkpoint resume; data-loader/RNG states are not serialized, so future stochastic ordering is not bitwise identical to an uninterrupted run."}
    write(HERE/"v8n_resume_record.json",record)
    write(HERE/"resume_scope.json",{"created_at_utc":utc(),"user_request":"恢复已训练59轮的COCO起点v8n作为对照",
        "run":RUN,"original_protocol_sha256":sha(CONFIG),"changes":"Only resume existing checkpoint; no changed data, hyperparameters, threshold policy or additional training budget.",
        "comparison":"Same-seed COCO-initialized v8n and 11n; existing deployed v8n remains an additional practical reference.",
        "limits":[record["resume_caveat"],"Only one matched v8n seed; other 11n seed is a repeat reference, not a second paired architecture trial.","Score pages were previously examined; old development data, not a new blind engineering-source test."]})
    model=YOLO(str(checkpoint))
    started=time.perf_counter()
    def on_start(trainer):
        actual=vars(trainer.args)
        differences={k:[v,actual.get(k)] for k,v in expected.items()
                     if (str(v)!=str(actual.get(k)) if k=="device" else v!=actual.get(k))}
        assert not differences,differences
        assert trainer.start_epoch==59 and trainer.epochs==120
        assert trainer.scheduler.last_epoch==58
        assert Path(actual["data"]).resolve()==Path(cfg["dataset"]).resolve()
        assert len(trainer.optimizer.state_dict()["state"])==old_optimizer_count
        assert all(math.isclose(g["lr"],old,rel_tol=0,abs_tol=1e-12) for g,old in zip(trainer.optimizer.param_groups,old_lrs))
        assert trainer.scaler.state_dict()==old_scaler
        assert trainer.ema.updates==old_updates
        # Ultralytics restores best_fitness, but not the EarlyStopping counter.
        trainer.stopper.best_epoch=peak_epoch
        trainer.stopper.best_fitness=float(ckpt["best_fitness"])
        trainer.stopper.possible_stop=(59-peak_epoch)>=trainer.stopper.patience-1
        record.update(status="training",parameter_count=sum(p.numel() for p in trainer.model.parameters()),
            effective_arguments_verified=True,optimizer_restored=True,amp_scaler_restored=True,
            ema_updates_restored=True,scheduler_last_epoch=trainer.scheduler.last_epoch,
            early_stopping_best_epoch_restored=peak_epoch,early_stopping_best_fitness_restored=trainer.stopper.best_fitness,
            restored_learning_rates=[g["lr"] for g in trainer.optimizer.param_groups])
        write(HERE/"v8n_resume_record.json",record)
        write(HERE/"records"/(RUN+"_resumed_args.json"),actual)
        # Compare all effective options with same-seed 11n, allowing only identity/resume fields.
        reference=read(HERE/"records/11n_seed20261012_args.json")
        allowed={"model","name","save_dir","resume"}
        mismatches={k:[reference.get(k),actual.get(k)] for k in set(reference)|set(actual)
                    if k not in allowed and reference.get(k)!=actual.get(k)}
        assert not mismatches,mismatches
        record["same_seed_11n_effective_args_verified"]=True
        write(HERE/"v8n_resume_record.json",record)
        emit(status="training",run=RUN,epoch_completed=59,epochs_planned=120,resumed=True)
    def on_epoch(trainer):
        state={"status":"training","run":RUN,"epoch_completed":trainer.epoch+1,"epochs_planned":trainer.epochs,
            "resumed":True,"seconds_this_process":time.perf_counter()-started,
            "metrics":{k:float(v) for k,v in trainer.metrics.items()}}
        emit(**state)
        print(json.dumps(state),flush=True)
    model.add_callback("on_train_start",on_start)
    model.add_callback("on_fit_epoch_end",on_epoch)
    model.train(resume=True,trainer=ClassNameTransferTrainer)
    rows=csv_rows()
    assert [int(float(r["epoch"])) for r in rows]==list(range(1,len(rows)+1))
    assert 59<len(rows)<=120
    record.update(status="trained",finished_at_utc=utc(),completed_epochs=len(rows),
        resumed_wall_seconds=time.perf_counter()-started,best_sha256=sha(FOLDER/"weights/best.pt"),
        last_sha256=sha(FOLDER/"weights/last.pt"))
    write(HERE/"v8n_resume_record.json",record)
    write(RECORD,{**before,"status":"trained","resume_record":str(HERE/"v8n_resume_record.json"),
        "resumed":True,"completed_epochs":len(rows),"best_sha256":record["best_sha256"],"last_sha256":record["last_sha256"]})
    del model
    torch.cuda.empty_cache()
    verify_inputs(cfg)


def compare(cfg):
    import numpy as np
    previous=read(HERE/"comparison.json")
    result={"status":"completed","comparison_kind":"same_seed_COCO_initialized_v8n_resumed_vs_11n",
        "protocol_sha256":sha(CONFIG),"resume_scope":read(HERE/"resume_scope.json"),
        "resume_record":read(HERE/"v8n_resume_record.json"),"models":copy.deepcopy(previous["models"])}
    for checkpoint in ("best","last"):
        name=RUN+"_"+checkpoint
        weights=FOLDER/f"weights/{checkpoint}.pt"
        emit(status="evaluating",model=name)
        saved=collect(cfg,name,weights)
        entry={"weights":str(weights),"weights_sha256":sha(weights),"modes":{}}
        for mode in ("whole","sliced","fused"):
            thresholds,details=tune(cfg,saved,mode)
            entry["modes"][mode]={"thresholds_from_calibration":thresholds,"threshold_diagnostics":details,
                "audit_global_0_2":counts(saved,mode,set(cfg["audit_cases"]),{}),
                "audit_calibrated":counts(saved,mode,set(cfg["audit_cases"]),thresholds),
                "calibration_calibrated":counts(saved,mode,set(cfg["calibration_cases"]),thresholds),
                "latency_ms_median":float(np.median([p["timings_ms"][mode] for p in saved["cases"]]))}
        result["models"][name]=entry
    a=result["models"][RUN+"_best"]["modes"]["fused"]
    b=result["models"]["11n_seed20261012_best"]["modes"]["fused"]
    result["same_seed_comparison_11n_minus_v8n"]={endpoint:{
        "delta_f1":b[endpoint]["overall"]["f1"]-a[endpoint]["overall"]["f1"],
        "bootstrap":bootstrap(a[endpoint],b[endpoint],cfg["case_groups"])} for endpoint in ("audit_global_0_2","audit_calibrated")}
    from evaluate_detector import metric
    class_diffs=[]
    for cls in cfg["names"].values():
        x=a["audit_calibrated"]["by_class"].get(cls,metric({"tp":0,"fp":0,"fn":0}))
        y=b["audit_calibrated"]["by_class"].get(cls,metric({"tp":0,"fp":0,"fn":0}))
        class_diffs.append({"class":cls,"gt":x["tp"]+x["fn"],"v8n":x,"11n":y,"delta_f1":y["f1"]-x["f1"]})
    result["same_seed_class_differences"]=class_diffs
    verify_inputs(cfg)
    result["frozen_inputs_unchanged_after_resume"]=True
    result["completed_at_utc"]=utc()
    write(HERE/"resumed_v8n_comparison.json",result)
    lines=["# 恢复59轮v8n后的同起点类型对照", "", "v8n从第60轮继续，恢复优化器、EMA、AMP缩放、学习率位置和原早停计数，总预算仍为120轮。两种架构均使用各自COCO预训练权重、同一末阶段配方、同种子20261012、418张训练样本与30页校准/29页评分划分。", "",
        "框架断点不保存完整数据加载器/RNG状态，因此续训不保证逐位等同于不中断训练。只有一个v8n匹配种子；评分页属于已经查看过的旧开发数据，不能称新盲测。原部署v8n继续单独作为实际参照，未替换。", "",
        "| 模型 | 评分集固定conf0.2 F1 | 同法校准阈值F1 | 校准后P | 校准后R | fused中位ms |", "|---|---:|---:|---:|---:|---:|"]
    for name in ["v8n_adopted",RUN+"_best",RUN+"_last","11n_seed20261012_best","11n_seed20261012_last","11n_seed20261015_best"]:
        m=result["models"][name]["modes"]["fused"]; g=m["audit_global_0_2"]["overall"]; c=m["audit_calibrated"]["overall"]
        lines.append(f"| {name} | {g['f1']:.4%} | {c['f1']:.4%} | {c['precision']:.4%} | {c['recall']:.4%} | {m['latency_ms_median']:.1f} |")
    lines += ["", "## 同种子best差异（11n减新v8n）", ""]
    for endpoint,c in result["same_seed_comparison_11n_minus_v8n"].items():
        lo,hi=c["bootstrap"]["ci95_delta_f1"]
        lines.append(f"- {endpoint}：{c['delta_f1']*100:+.3f}个百分点；条件分组bootstrap95%区间[{lo*100:+.3f}, {hi*100:+.3f}]。")
    lines += ["", "## 同种子逐类结果", "", "| 类别 | GT | 新v8n F1 | 11n F1 | 差值（百分点） |", "|---|---:|---:|---:|---:|"]
    for r in sorted(class_diffs,key=lambda x:-x["gt"]):
        if r["gt"]:
            lines.append(f"| {r['class']} | {r['gt']} | {r['v8n']['f1']:.4%} | {r['11n']['f1']:.4%} | {r['delta_f1']*100:+.3f} |")
    lines += ["", "该结果是固定训练配方下现成COCO初始化模型的对照；不同架构的COCO预训练权重本身训练历史也不完全相同。未接入官方ComponentF1/PinF1/NetF1。此前实际替换试验报告comparison.md保留，部分59轮检查点及历史记录见resume_archive。"]
    (HERE/"resumed_v8n_comparison.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    emit(status="completed",comparison=str(HERE/"resumed_v8n_comparison.md"),
         same_seed_delta_f1_11n_minus_v8n=result["same_seed_comparison_11n_minus_v8n"]["audit_calibrated"]["delta_f1"])
    print(json.dumps({"resume_comparison_completed":True,"comparison":str(HERE/"resumed_v8n_comparison.md")}),flush=True)


def main():
    cfg=read(CONFIG)
    libraries(); metric_self_check(); baseline_check(cfg)
    train(cfg)
    compare(cfg)


if __name__=="__main__":
    try:
        main()
    except BaseException as exc:
        emit(status="failed",error=repr(exc))
        raise
