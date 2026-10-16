"""Serialized v6/v7 label ablation, GPU resource queue, and fixed full-page evaluation."""
import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from pcb_parser.unet_data import SegmentationTiles,predict_tiled
from pcb_parser.unet_model import WirePinUNet
from pcb_parser.unet_retraining import CachedTrainingTiles
from pcb_parser.unet_inference import decode_pins
from pcb_parser.dataset_converter import load_source_annotation
from pcb_parser.training_dependency import prerequisite_state
from train_unet import epoch
from verify_unet_contrast import path_specs,path_records,pin_metrics


def read(path):return json.loads(Path(path).read_text(encoding="utf-8-sig"))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
    path=Path(path);temp=path.with_suffix(path.suffix+".tmp")
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n",encoding="utf-8");temp.replace(path)
def read_rows(path):return [json.loads(s) for s in Path(path).read_text(encoding="utf-8").splitlines()]
def state_hash(state):
    h=hashlib.sha256()
    for name,value in sorted(state.items()):
        h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def pixel_metrics(tp,fp,fn):
    return dict(tp=int(tp),fp=int(fp),fn=int(fn),precision=tp/max(1,tp+fp),recall=tp/max(1,tp+fn),
                dice=2*tp/max(1,2*tp+fp+fn),iou=tp/max(1,tp+fp+fn))


def validate_inputs(cfg):
    initial=ROOT/cfg["initial_weights"]
    assert sha(initial)==cfg["initial_sha256"]
    datasets={name:read_rows(ROOT/arm["manifest"]) for name,arm in cfg["arms"].items()}
    reference=datasets["control_v6"]
    assert len(reference)==199
    assert sum(r["split"]=="train" for r in reference)==171
    assert sum(r["split"]=="val" for r in reference)==28
    for name,rows in datasets.items():
        assert [(r["case_id"],r["split"]) for r in rows]==[(r["case_id"],r["split"]) for r in reference]
        assert all(r["structurally_usable_draft"] for r in rows)
        assert [r for r in rows if r["split"]=="val"]==[r for r in reference if r["split"]=="val"]
        assert sha(ROOT/cfg["arms"][name]["manifest"])==cfg["arms"][name]["manifest_sha256"]
        for row in rows:
            for key in ("image","wire_mask","pin_heatmap","source_annotation_path"):
                expected=row.get("source_annotation_sha256" if key=="source_annotation_path" else key+"_sha256")
                if expected and row.get(key):assert sha(row[key])==expected
    checks=read(ROOT/"runs/unet_v7_label_review_20261015/final_checks.json")
    assert checks["status"]=="passed" and checks["changed_training_pages"]==4
    summary=read(ROOT/"data/segmentation_unet_v7_train_review_20261015/summary.json")
    assert all(sha(p)==digest for p,digest in summary["protected_inputs_sha256"].items())
    return datasets


def free_gpu_mib():
    result=subprocess.run(["nvidia-smi","--id=0","--query-gpu=memory.free","--format=csv,noheader,nounits"],
                          capture_output=True,text=True,timeout=10,check=True)
    return int(result.stdout.strip().splitlines()[0])


def wait_for_gpu(cfg,out,stage,queue):
    start=time.monotonic();last_report=0;last_state=None
    while True:
        dependency=prerequisite_state(cfg["prerequisite_job"]) if cfg.get("prerequisite_job") else dict(ready=True)
        # A validation-time memory release is insufficient: wait for the controller to finish and exit first.
        if not dependency["ready"]:
            record=dict(status="waiting_for_yolo",prerequisite=dependency)
        else:
            free=free_gpu_mib()
            if free>=cfg["minimum_free_gpu_mib"]:return free
            record=dict(status="waiting_for_gpu",free_gpu_mib=free,prerequisite=dependency)
        record.update(next_stage=stage,required_gpu_mib=cfg["minimum_free_gpu_mib"],wait_seconds=round(time.monotonic()-start),
                    process_id=os.getpid(),other_processes_interrupted=False)
        write(out/"progress.json",record)
        current_state=(record["status"],dependency.get("reason"),dependency.get("job_status"))
        if current_state!=last_state or time.monotonic()-last_report>=300:
            print(json.dumps(record),flush=True);last_report=time.monotonic();last_state=current_state
        if not queue:raise RuntimeError("Prerequisite job or GPU busy; use --queue to wait")
        if time.monotonic()-start>=cfg["maximum_queue_seconds"]:raise TimeoutError("Training dependency/resource queue timed out")
        time.sleep(30)


def checkpoint_for(model,source,cfg,arm,manifest_hash,history,best_loss,optimizer,scaler,scheduler,generator):
    model_cfg=dict(source["config"]);model_cfg["training"]=dict(cfg["training"])
    model_cfg["labels_status"]="draft_v7_matched_training_only_local_review"
    return dict(state_dict=model.state_dict(),base_channels=source["base_channels"],kind="trained_on_draft_labels",
        config=model_cfg,outputs=source["outputs"],source_commit=source["source_commit"],epoch=len(history),
        initial_epoch=source["epoch"],initial_weights_sha256=cfg["initial_sha256"],arm=arm,best_loss=best_loss,
        history=history,manifest_sha256=manifest_hash,optimizer_state=optimizer.state_dict(),
        scaler_state=scaler.state_dict(),scheduler_state=scheduler.state_dict(),
        torch_rng_state=torch.get_rng_state(),shuffle_rng_state=generator.get_state(),
        cuda_rng_state=torch.cuda.get_rng_state_all())


def train_arm(name,cfg,source,rows,out):
    folder=out/name;manifest_hash=cfg["arms"][name]["manifest_sha256"]
    if (folder/"summary.json").exists():
        saved=read(folder/"summary.json")
        assert saved["epochs_completed"]==cfg["training"]["epochs"] and sha(saved["checkpoint"])==saved["checkpoint_sha256"]
        return saved
    folder.mkdir(exist_ok=True)
    settings=cfg["training"];seed=cfg["seed"]
    torch.manual_seed(seed);random.seed(seed);np.random.seed(seed)
    generator=torch.Generator().manual_seed(seed)
    train_rows=[r for r in rows if r["split"]=="train"];val_rows=[r for r in rows if r["split"]=="val"]
    train_set=CachedTrainingTiles(train_rows,512,384);val_set=CachedTrainingTiles(val_rows,512,384)
    loader=DataLoader(train_set,batch_size=settings["batch_size"],shuffle=True,generator=generator,num_workers=0,pin_memory=True)
    validation=DataLoader(val_set,batch_size=settings["batch_size"],shuffle=False,num_workers=0,pin_memory=True)
    model=WirePinUNet(source["base_channels"]).to("cuda:0");model.load_state_dict(source["state_dict"],strict=True)
    optimizer=torch.optim.AdamW(model.parameters(),lr=settings["learning_rate"],weight_decay=settings["weight_decay"])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=settings["epochs"],eta_min=settings["minimum_learning_rate"])
    scaler=torch.amp.GradScaler("cuda",enabled=settings["amp"])
    history=[];best_loss=float("inf");initial_state=state_hash(source["state_dict"])
    started=time.perf_counter()
    if (folder/"last.pt").exists():
        previous=torch.load(folder/"last.pt",map_location="cpu",weights_only=True)
        assert previous["manifest_sha256"]==manifest_hash and previous["initial_weights_sha256"]==cfg["initial_sha256"]
        model.load_state_dict(previous["state_dict"]);optimizer.load_state_dict(previous["optimizer_state"])
        scheduler.load_state_dict(previous["scheduler_state"]);scaler.load_state_dict(previous["scaler_state"])
        history=previous["history"];best_loss=previous["best_loss"]
        torch.set_rng_state(previous["torch_rng_state"]);generator.set_state(previous["shuffle_rng_state"])
        torch.cuda.set_rng_state_all(previous["cuda_rng_state"])
        del previous
    else:
        assert state_hash(model.state_dict())==initial_state
        val_loss,_,val_pixels=epoch(model,validation,torch.device("cuda:0"),settings,phase=name+"/initial_validation")
        best_loss=val_loss
        baseline=checkpoint_for(model,source,cfg,name,manifest_hash,history,best_loss,optimizer,scaler,scheduler,generator)
        torch.save(baseline,folder/"best.pt")
        write(folder/"initial_validation.json",dict(val_loss=val_loss,val_wire_pixels=val_pixels,initial_state_sha256=initial_state))
        del baseline
    write(folder/"configuration.json",cfg|dict(arm=name,initial_state_sha256=initial_state))
    print(json.dumps(dict(status="training_arm_started",arm=name,epochs_completed=len(history),epochs=settings["epochs"],batches=len(loader))),flush=True)
    for index in range(len(history),settings["epochs"]):
        tick=time.perf_counter();lr=optimizer.param_groups[0]["lr"]
        write(out/"progress.json",dict(status="training",arm=name,epoch=index+1,epochs=settings["epochs"],
              initial_epoch=source["epoch"],learning_rate=lr,process_id=os.getpid()))
        train_loss,steps,train_pixels=epoch(model,loader,torch.device("cuda:0"),settings,optimizer,scaler,phase=f"{name}/train/{index+1}")
        val_loss,_,val_pixels=epoch(model,validation,torch.device("cuda:0"),settings,phase=f"{name}/validation/{index+1}")
        scheduler.step()
        record=dict(epoch=index+1,source_total_epoch=source["epoch"]+index+1,train_loss=train_loss,val_loss=val_loss,
                    learning_rate=lr,steps=len(steps),epoch_seconds=time.perf_counter()-tick,
                    train_wire_pixels=train_pixels,val_wire_pixels=val_pixels)
        history.append(record);improved=val_loss<best_loss;best_loss=min(best_loss,val_loss)
        checkpoint=checkpoint_for(model,source,cfg,name,manifest_hash,history,best_loss,optimizer,scaler,scheduler,generator)
        torch.save(checkpoint,folder/"last.pt")
        if improved:torch.save(checkpoint,folder/"best.pt")
        write(folder/"history.json",history)
        with (folder/"epochs.jsonl").open("a",encoding="utf-8") as f:f.write(json.dumps(record)+"\n")
        progress=dict(status="training",arm=name,epoch=index+1,epochs=settings["epochs"],epoch_completed=True,
                      best_loss=best_loss,latest=record,elapsed_seconds=time.perf_counter()-started)
        write(out/"progress.json",progress);print(json.dumps(progress),flush=True)
        del checkpoint
    selected=torch.load(folder/"best.pt",map_location="cpu",weights_only=True)
    result=dict(status="completed",arm=name,epochs_completed=len(history),steps=sum(r["steps"] for r in history),
        best_finetune_epoch=selected["epoch"],initial_epoch=source["epoch"],best_loss=best_loss,
        initial_weights_sha256=cfg["initial_sha256"],initial_state_sha256=initial_state,manifest_sha256=manifest_hash,
        checkpoint=str((folder/"best.pt").resolve()),checkpoint_sha256=sha(folder/"best.pt"),
        wall_seconds=time.perf_counter()-started,learned_state_changed=state_hash(selected["state_dict"])!=initial_state)
    write(folder/"summary.json",result)
    train_set.clear_cache();val_set.clear_cache()
    del model,optimizer,scheduler,scaler,selected
    gc.collect();torch.cuda.empty_cache()
    return result


def evaluate(cfg,datasets,out):
    val=[r for r in datasets["control_v6"] if r["split"]=="val"]
    split=read(ROOT/"runs/unet_inference_v2/split.json");unique=set(split["development"]+split["acceptance_holdout"])
    raw={r["case_id"]:r for r in read_rows(ROOT/"data/manifest.jsonl")}
    variants={"baseline_v6":ROOT/cfg["initial_weights"]}|{name:out/name/"best.pt" for name in cfg["arms"]}
    results={};directory=out/"evaluation";directory.mkdir(exist_ok=True)
    for name,weight in variants.items():
        checkpoint=torch.load(weight,map_location="cpu",weights_only=True)
        model=WirePinUNet(checkpoint["base_channels"]).to("cuda:0");model.load_state_dict(checkpoint["state_dict"],strict=True)
        records=[]
        for index,row in enumerate(val,1):
            case=row["case_id"]
            write(out/"progress.json",dict(status="evaluating",variant=name,case_id=case,pages=index,total=28))
            rgb=np.asarray(Image.open(row["image"]).convert("RGB"))
            probability=predict_tiled(model,rgb,"cuda:0",512,384)
            pred=probability[0]>=.5;truth=np.asarray(Image.open(row["wire_mask"]))>0
            valid=np.ones_like(truth) if not row.get("valid_mask") else np.asarray(Image.open(row["valid_mask"]))>0
            wire=pixel_metrics(int((pred&truth&valid).sum()),int((pred&~truth&valid).sum()),int((~pred&truth&valid).sum()))
            tips=read(Path(row["pin_heatmap"]).parent/"pin_points.json")
            candidates=decode_pins(probability[1],threshold=.8,kernel=5,refinement="quadratic",nms_radius=0.,limit=500)
            pin=pin_metrics(candidates,tips)
            annotation,_=load_source_annotation(raw[case]["raw_annotation_path"])
            paths=path_records(pred,path_specs(annotation,truth.shape))
            record=dict(case_id=case,unique26=case in unique,wire=wire,pin_coordinates_5px=pin,
                        connected_paths=sum(p["connected_within_reference_tube"] for p in paths),paths=len(paths))
            records.append(record)
            folder=directory/name/case;folder.mkdir(parents=True,exist_ok=True)
            Image.fromarray(pred.astype(np.uint8)*255).save(folder/"wire_mask.png")
            write(folder/"metrics.json",record);write(folder/"pin_candidates.json",candidates)
            if case in {"0036","0208","0007","0286"}:np.savez_compressed(folder/"probabilities.npz",wire=probability[0],pin=probability[1])
            print(json.dumps(dict(status="evaluation",variant=name,case_id=case,pages=index,total=28,iou=wire["iou"])),flush=True)
        groups={}
        for group in ("all28","unique26"):
            selected=[r for r in records if group=="all28" or r["unique26"]]
            groups[group]=dict(pages=len(selected),wire=pixel_metrics(*(sum(r["wire"][k] for r in selected) for k in ("tp","fp","fn"))),
                pin_coordinates_5px=pixel_metrics(*(sum(r["pin_coordinates_5px"][k] for r in selected) for k in ("tp","fp","fn"))),
                connected_paths=sum(r["connected_paths"] for r in selected),paths=sum(r["paths"] for r in selected))
        results[name]=dict(checkpoint=str(weight.resolve()),checkpoint_sha256=sha(weight),selected_finetune_epoch=checkpoint.get("epoch"),groups=groups,cases=records)
        del model,checkpoint;gc.collect();torch.cuda.empty_cache()
    base=results["baseline_v6"]["groups"]["unique26"]
    assert abs(base["wire"]["iou"]-.7271300114263004)<.001
    checks={}
    for name in cfg["arms"]:
        candidate=results[name]["groups"]["unique26"]
        flags=dict(wire_iou_improves=candidate["wire"]["iou"]>base["wire"]["iou"],
                   connected_paths_not_lower=candidate["connected_paths"]>=base["connected_paths"],
                   pin_coordinate_f1_drop_within_1pp=candidate["pin_coordinates_5px"]["dice"]>=base["pin_coordinates_5px"]["dice"]-.01)
        checks[name]=dict(checks=flags,passes=all(flags.values()))
    accepted=[name for name in checks if checks[name]["passes"]]
    winner=max(accepted,key=lambda n:results[n]["groups"]["unique26"]["wire"]["iou"]) if accepted else "baseline_v6"
    result=dict(status="completed",variants=results,adoption_checks=checks,recommended_variant=winner,
        recommended_checkpoint=results[winner]["checkpoint"],independent_test=False,
        evaluation_thresholds=dict(wire=.5,pin=.8,pin_radius=5),default_weights_replaced=False,
        frozen_validation_labels_changed=False,augmentation_used=False,official_score=None)
    write(directory/"comparison.json",result)
    lines=["# v6低学习率微调与v7标签对照", "", "两组从相同v6第60轮权重开始，各20轮；相同样本、顺序、优化器及3e-5→3e-6余弦学习率。仅4张训练页的标签不同。验证28页保持原样，固定阈值0.5，另报告与旧模型相同的26页整图。没有图像增强或结构更换。", "", "| 模型 | 26页IoU | Dice | P | R | 局部连通路径 | 引脚坐标F1 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for name,value in results.items():
        g=value["groups"]["unique26"];w=g["wire"]
        lines.append(f"| {name} | {w['iou']:.4%} | {w['dice']:.4%} | {w['precision']:.4%} | {w['recall']:.4%} | {g['connected_paths']}/{g['paths']} | {g['pin_coordinates_5px']['dice']:.4%} |")
    lines += ["",f"按固定比较门槛推荐：{winner}。默认权重未替换。", "", "best允许保持微调第0轮的原v6权重，避免所有微调轮次损失更差时强行采用退步模型。验证集已经参与模型选择与开发，指标基于草稿标签；不属于独立工程测试，也不是官方网络指标。",
              "", "完整结果与门槛：comparison.json。训练标签局部证据：runs/unet_v7_label_review_20261015/。"]
    (directory/"report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    return result


def cpu_smoke(cfg,datasets,out):
    torch.manual_seed(cfg["seed"])
    rows=[r for r in datasets["reviewed_v7"] if r["case_id"] in {"0025","0088"}]
    dataset=SegmentationTiles(rows,512,384);loader=DataLoader(dataset,batch_size=2,shuffle=False,num_workers=0)
    source=torch.load(ROOT/cfg["initial_weights"],map_location="cpu",weights_only=True)
    model=WirePinUNet(source["base_channels"]);model.load_state_dict(source["state_dict"])
    initial=state_hash(model.state_dict());initial_output=model.out.conv.weight.detach().clone()
    settings=cfg["training"]|dict(amp=False)
    optimizer=torch.optim.AdamW(model.parameters(),lr=settings["learning_rate"],weight_decay=settings["weight_decay"])
    scaler=torch.amp.GradScaler("cuda",enabled=False)
    loss,steps,_=epoch(model,loader,torch.device("cpu"),settings,optimizer,scaler,max_steps=1,phase="cpu_full512_smoke")
    assert len(steps)==1 and math.isfinite(loss) and not torch.equal(initial_output,model.out.conv.weight.detach())
    result=dict(status="cpu_smoke_passed_not_quality_training",tile_size=512,batch_size=2,steps=1,loss=loss,
                warm_start_state_matches_source=initial==state_hash(source["state_dict"]),parameter_update_verified=True,
                manifests_verified=True,GPU_used=False,production_weights_changed=False)
    write(out/"cpu_smoke.json",result);print(json.dumps(result),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--allow-draft",action="store_true");parser.add_argument("--queue",action="store_true")
    parser.add_argument("--cpu-smoke",action="store_true");args=parser.parse_args()
    if not args.allow_draft:raise ValueError("Explicit --allow-draft is required")
    cfg=read(args.config);out=ROOT/cfg["output"];out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);datasets=validate_inputs(cfg)
    if args.cpu_smoke:cpu_smoke(cfg,datasets,out);return
    import msvcrt
    with (out/"experiment.lock").open("a+b") as lock:
        if lock.tell()==0:lock.write(b"0");lock.flush()
        lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        write(out/"worker.json",dict(process_id=os.getpid(),status="started",config=str(args.config.resolve()),config_sha256=sha(args.config)))
        protocol=read(ROOT/"data/segmentation_unet_v7_train_review_20261015/summary.json")
        protected=protocol["protected_inputs_sha256"]|{str(ROOT/arm["manifest"]):arm["manifest_sha256"] for arm in cfg["arms"].values()}
        for rows in datasets.values():
            for row in rows:
                for key in ("image","wire_mask","pin_heatmap","valid_mask","source_annotation_path"):
                    if row.get(key):protected[row[key]]=sha(row[key])
        protected[str(args.config.resolve())]=sha(args.config)
        write(out/"protocol.json",dict(config=cfg,protected_sha256=protected,worker_script_sha256=sha(Path(__file__)),
            comparison="same initial weights, shuffle order, LR schedule and images; labels differ on four train pages only",
            independent_test=False,official_score=None))
        try:
            wait_for_gpu(cfg,out,"control_v6",args.queue)
            assert all(sha(p)==digest for p,digest in protected.items())
            source=torch.load(ROOT/cfg["initial_weights"],map_location="cpu",weights_only=True)
            assert source["epoch"]==60 and source["kind"]=="trained_on_draft_labels"
            summaries=[]
            for name in cfg["arms"]:
                wait_for_gpu(cfg,out,name,args.queue)
                summaries.append(train_arm(name,cfg,source,datasets[name],out))
                assert all(sha(p)==digest for p,digest in protected.items())
            wait_for_gpu(cfg,out,"full_page_evaluation",args.queue)
            result=evaluate(cfg,datasets,out)
            assert all(sha(p)==digest for p,digest in protected.items())
            write(out/"summary.json",dict(status="training_and_evaluation_complete",training_runs=summaries,
                recommended_variant=result["recommended_variant"],recommended_checkpoint=result["recommended_checkpoint"],
                protected_original_files_unchanged=True,default_weights_changed=False))
            write(out/"progress.json",dict(status="completed",recommended_variant=result["recommended_variant"],
                                          evaluation=str((out/"evaluation/comparison.json").resolve())))
            print(json.dumps(dict(status="completed",recommended_variant=result["recommended_variant"])),flush=True)
        except BaseException as exc:
            write(out/"progress.json",dict(status="failed",error_type=type(exc).__name__,error=str(exc),process_id=os.getpid()))
            raise


if __name__=="__main__":main()
