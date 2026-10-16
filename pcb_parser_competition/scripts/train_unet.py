"""Train the U-Net adapter; CPU smoke is explicitly separate from model quality."""
import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path
import torch
from torch.utils.data import DataLoader

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from pcb_parser.unet_data import SegmentationTiles
from pcb_parser.unet_model import WirePinUNet,segmentation_loss


def epoch(model,loader,device,cfg,optimizer=None,scaler=None,max_steps=None,phase="train"):
    model.train(optimizer is not None)
    values=[]
    tp=fp=fn=0
    for images,targets,valid in loader:
        images,targets,valid=images.to(device),targets.to(device),valid.to(device)
        with torch.set_grad_enabled(optimizer is not None):
            with torch.autocast(device_type=device.type,enabled=bool(cfg["amp"] and device.type=="cuda")):
                logits=model(images)
                loss,_=segmentation_loss(logits,targets,valid,cfg["wire_positive_weight"],cfg["pin_positive_weight"])
            if not torch.isfinite(loss):
                raise ValueError("Non-finite U-Net loss")
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        values.append(float(loss.detach()))
        with torch.no_grad():
            predicted=logits[:,0].sigmoid()>=.5
            expected=targets[:,0]>=.5
            mask=valid.bool()
            tp+=int((predicted&expected&mask).sum())
            fp+=int((predicted&~expected&mask).sum())
            fn+=int((~predicted&expected&mask).sum())
        if len(values)%20==0:
            print(json.dumps({"phase":phase,"batch":len(values),"batches":len(loader),"loss":values[-1]}),flush=True)
        if max_steps and len(values)>=max_steps:
            break
    metrics={"wire_pixel_precision":tp/(tp+fp) if tp+fp else None,
             "wire_pixel_recall":tp/(tp+fn) if tp+fn else None,
             "wire_pixel_dice":2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
             "wire_pixel_iou":tp/(tp+fp+fn) if tp+fp+fn else None,
             "threshold":.5,"scope":"tile_pixels_against_selected_labels_not_net_or_pin_F1"}
    return sum(values)/len(values),values,metrics


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--config",type=Path,default=ROOT/"configs/unet_wire_pin.json")
    p.add_argument("--manifest",type=Path,default=ROOT/"data/segmentation_unet_v1/manifest.jsonl")
    p.add_argument("--reviews",type=Path,default=ROOT/"configs/unet_label_reviews.json")
    p.add_argument("--device")
    p.add_argument("--allow-draft",action="store_true")
    p.add_argument("--smoke",action="store_true")
    p.add_argument("--case-id",action="append")
    p.add_argument("--steps",type=int,default=4)
    p.add_argument("--epochs",type=int)
    p.add_argument("--batch-size",type=int)
    p.add_argument("--resume",type=Path)
    p.add_argument("--max-train-steps",type=int,help="Debug-only bound; marks checkpoint as partial_training_debug")
    p.add_argument("--max-val-steps",type=int,help="Debug-only bound; marks checkpoint as partial_training_debug")
    p.add_argument("--output",type=Path,default=ROOT/"runs/training/unet_wire_pin_v1")
    args=p.parse_args()
    cfg=json.loads(args.config.read_text(encoding="utf-8"))
    if args.epochs is not None:
        cfg["training"]["epochs"]=args.epochs
    if args.batch_size is not None:
        cfg["training"]["batch_size"]=args.batch_size
    if min(cfg["training"]["epochs"],cfg["training"]["batch_size"],args.steps)<=0:
        raise ValueError("Epochs, batch size and smoke steps must be positive")
    all_rows=list(map(json.loads,args.manifest.read_text(encoding="utf-8").splitlines()))
    reviews=json.loads(args.reviews.read_text(encoding="utf-8"))
    approved=set(reviews["approved_cases"])
    rows=[r for r in all_rows if r["structurally_usable_draft"] and (args.allow_draft or r["case_id"] in approved)]
    if args.case_id:
        rows=[r for r in rows if r["case_id"] in args.case_id]
    if args.smoke and not args.allow_draft and not rows:
        raise ValueError("No reviewed labels; explicit --allow-draft is required for draft smoke testing")
    train_rows=rows if args.smoke else [r for r in rows if r["split"]=="train"]
    val_rows=[] if args.smoke else [r for r in rows if r["split"]=="val"]
    if not train_rows or (not args.smoke and not val_rows):
        raise ValueError("No usable reviewed train/val labels. Review labels first, or explicitly select --allow-draft for a draft-label experiment.")
    torch.manual_seed(cfg["training"]["seed"])
    random.seed(cfg["training"]["seed"])
    torch.set_num_threads(4)
    device=torch.device(args.device or ("cpu" if args.smoke else cfg["training"]["device"]))
    if args.smoke and device.type!="cpu":
        raise ValueError("Smoke mode uses CPU to avoid other GPU jobs")
    if not args.smoke and device.type=="cuda":
        free,total=torch.cuda.mem_get_info(device)
        if free<training_minimum_bytes(cfg):
            raise RuntimeError(f"GPU has only {free/1024**3:.2f}GiB free; leave the existing job running and start U-Net when enough memory is available")
    model=WirePinUNet(cfg["base_channels"]).to(device)
    training=cfg["training"]
    optimizer=torch.optim.AdamW(model.parameters(),lr=training["learning_rate"],weight_decay=training["weight_decay"])
    scaler=torch.amp.GradScaler("cuda",enabled=device.type=="cuda" and training["amp"])
    size=128 if args.smoke else cfg["tile_size"]
    stride=128 if args.smoke else cfg["stride"]
    loader=DataLoader(SegmentationTiles(train_rows,size,stride),batch_size=1 if args.smoke else training["batch_size"],shuffle=True,num_workers=0)
    validation=DataLoader(SegmentationTiles(val_rows,size,stride),batch_size=training["batch_size"],num_workers=0) if val_rows else None
    args.output.mkdir(parents=True,exist_ok=True)
    history=[]
    best=float("inf")
    start_epoch=0
    start=time.perf_counter()
    initial=model.out.conv.weight.detach().clone()
    kind="cpu_smoke_only" if args.smoke else "trained_on_draft_labels" if args.allow_draft else "trained_on_reviewed_labels"
    if (args.max_train_steps or args.max_val_steps) and not args.smoke:
        kind="partial_training_debug"
    manifest_hash=hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    if args.resume:
        previous=torch.load(args.resume,map_location=device,weights_only=True)
        if args.smoke or kind!=previous["kind"] or previous["manifest_sha256"]!=manifest_hash:
            raise ValueError("Resume requires the same training kind and dataset manifest")
        if previous["config"]["tile_size"]!=cfg["tile_size"] or previous["config"]["stride"]!=cfg["stride"]:
            raise ValueError("Resume tile configuration changed")
        model.load_state_dict(previous["state_dict"])
        optimizer.load_state_dict(previous["optimizer_state"])
        scaler.load_state_dict(previous["scaler_state"])
        start_epoch=previous["epoch"]
        best=previous["best_loss"]
        history=previous["history"]
    for index in range(start_epoch,1 if args.smoke else training["epochs"]):
        loss,steps,train_metrics=epoch(model,loader,device,training,optimizer,scaler,args.steps if args.smoke else args.max_train_steps)
        val_result=epoch(model,validation,device,training,max_steps=args.max_val_steps,phase="validation") if validation else None
        val_loss=val_result[0] if val_result else None
        record={"epoch":index+1,"train_loss":loss,"val_loss":val_loss,"steps":len(steps),
                "train_wire_pixels":train_metrics,"val_wire_pixels":val_result[2] if val_result else None}
        history.append(record)
        print(json.dumps(record),flush=True)
        score=val_loss if val_loss is not None else loss
        improved=score<best
        best=min(best,score)
        checkpoint={"state_dict":model.state_dict(),"base_channels":cfg["base_channels"],"kind":kind,
                    "config":cfg,"outputs":cfg["outputs"],"source_commit":cfg["source_commit"],
                    "optimizer_state":optimizer.state_dict(),"scaler_state":scaler.state_dict(),
                    "epoch":index+1,"best_loss":best,"history":history,"manifest_sha256":manifest_hash}
        torch.save(checkpoint,args.output/"last.pt")
        if improved:
            torch.save(checkpoint,args.output/"best.pt")
        (args.output/"history.json").write_text(json.dumps(history,ensure_ascii=False,indent=2),encoding="utf-8")
        with (args.output/"epochs.jsonl").open("a",encoding="utf-8") as stream:
            stream.write(json.dumps(record,ensure_ascii=False)+"\n")
    if not (args.output/"best.pt").exists():
        raise ValueError("No new best checkpoint in output; resume into its original run directory and set total epochs above the saved epoch")
    result={"status":kind,"device":str(device),"train_cases":[r["case_id"] for r in train_rows],
            "validation_cases":[r["case_id"] for r in val_rows],"parameters":sum(v.numel() for v in model.parameters()),
            "steps":sum(r["steps"] for r in history),"history":history,"wall_seconds":time.perf_counter()-start,
            "output_layer_changed":not torch.equal(initial,model.out.conv.weight.detach()),
            "tile_size_used":size,"checkpoint":str((args.output/"best.pt").resolve()),
            "checkpoint_sha256":hashlib.sha256((args.output/"best.pt").read_bytes()).hexdigest(),
            "manifest_sha256":hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
            "official_score":None,"quality_metrics":None if args.smoke or kind=="partial_training_debug" else history[-1]["val_wire_pixels"],
            "limited_train_steps":args.max_train_steps,"limited_val_steps":args.max_val_steps,
            "resumed_from":str(args.resume) if args.resume else None,
            "engineering_split_status":"existing_case_split_reused_engineering_dedup_pending"}
    (args.output/"summary.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False))


def training_minimum_bytes(cfg):
    return cfg["training"].get("minimum_free_gpu_gib",4)*1024**3


if __name__=="__main__":
    main()
