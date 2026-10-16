"""Fixed16 paired pilot; only checked negative crop sampling differs."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
import numpy as np
import torch
from torch.utils.data import DataLoader

from unet_audit_common import ROOT, OUT, read, write, sha, assert_protected
from pcb_parser.unet_model import WirePinUNet
from pcb_parser.unet_retraining import CachedTrainingTiles
from pcb_parser.unet_negative_sampling import NegativeFocusTrainingTiles
from train_unet import epoch


def state_hash(state):
    h=hashlib.sha256()
    for key,value in sorted(state.items()):
        h.update(key.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--allow-agent-draft",action="store_true")
    parser.add_argument("--check-only",action="store_true");args=parser.parse_args()
    if not args.allow_agent_draft:raise ValueError("This controlled pilot needs explicit agent-draft mode; no certified labels exist")
    config_path=ROOT/"configs/unet_negative_sampling_pilot_v4.json";cfg=read(config_path)
    rows=[json.loads(s) for s in (ROOT/cfg["manifest"]).read_text(encoding="utf-8").splitlines()]
    train=[r for r in rows if r["split"]=="train"];cal=[r for r in rows if r["split"]=="calibration"]
    split=read(ROOT/cfg["split"]);regions=read(ROOT/cfg["regions"])["regions"]
    assert len(train)==93 and len(cal)==10 and set(r["case_id"] for r in train)==set(split["training_cases"])
    assert set(r["case_id"] for r in cal)==set(split["calibration_cases"])
    external=set(read(OUT/"protocol.json")["original_validation_cases"])
    assert all(r["source_split"]=="train" and r["case_id"] not in external for r in rows)
    assert_protected()
    frozen=read(ROOT/"runs/unet_training_label_review_v4/protected_inputs.json")
    for path,digest in frozen.items():assert sha(path)==digest
    for row in rows:
        assert sha(row["wire_mask"])==row["wire_mask_sha256"] and sha(row["pin_heatmap"])==row["pin_heatmap_sha256"]
    frozen=frozen|{str(config_path):sha(config_path),str(ROOT/cfg["manifest"]):sha(ROOT/cfg["manifest"]),
        str(ROOT/cfg["regions"]):sha(ROOT/cfg["regions"]),str(ROOT/cfg["split"]):sha(ROOT/cfg["split"])}
    for row in rows:
        for key in ("image","wire_mask","pin_heatmap"):frozen[row[key]]=sha(row[key])
    datasets={name:NegativeFocusTrainingTiles(train,regions,fraction,cfg["seed"],cfg["tile_size"],cfg["stride"]) for name,fraction in cfg["arms"].items()}
    for number in range(1,cfg["epochs"]+1):
        first=datasets["control"].set_epoch(number);second=datasets["negative_focus"].set_epoch(number)
        assert first["tiles"]==second["tiles"]==925 and second["focus_slots"]==231
        untouched=set(range(925))-set(datasets["negative_focus"].replaced_slots)
        assert all(datasets["control"].tiles[i]==datasets["negative_focus"].tiles[i] for i in untouched)
    if args.check_only:
        print(json.dumps(dict(status="paired_data_and16_epoch_schedules_verified",training_pages=93,calibration_pages=10,
            external_pages_used_for_training=0,tiles_per_arm=925,focus_slots_per_epoch=231,source_files=assert_protected())));return
    output=ROOT/cfg["output"]
    if output.exists():raise ValueError("Pilot output exists; no overwrite or silent restart")
    if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0]<4*1024**3:raise RuntimeError("Need CUDA with4GiB free")
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    # CUDA bilinear backward may lack a deterministic kernel. Record warnings;
    # never misrepresent a fixed RNG seed as guaranteed bitwise reproducibility.
    torch.use_deterministic_algorithms(True,warn_only=True)
    seed_all(cfg["seed"]);initial_model=WirePinUNet(cfg["base_channels"])
    initial={k:v.detach().cpu().clone() for k,v in initial_model.state_dict().items()};initial_hash=state_hash(initial)
    output.mkdir(parents=True);torch.save(dict(state_dict=initial,base_channels=cfg["base_channels"],epoch=0),output/"initial.pt")
    write(output/"configuration.json",cfg|dict(config_sha256=sha(config_path),manifest_sha256=sha(ROOT/cfg["manifest"]),
        initial_state_sha256=initial_hash,initial_checkpoint_sha256=sha(output/"initial.pt"),
        labels_status="partial_agent_visual_corrections_not_full_page_certified",training_started=True,
        runtime=dict(torch=torch.__version__,device=torch.cuda.get_device_name(0),deterministic_algorithms="warn_only",CUBLAS_WORKSPACE_CONFIG=os.environ["CUBLAS_WORKSPACE_CONFIG"])))
    write(output/"protected_inputs.json",frozen);start=time.perf_counter();results=[]
    calibration=CachedTrainingTiles(cal,cfg["tile_size"],cfg["stride"])
    cal_loader=DataLoader(calibration,batch_size=cfg["batch_size"],shuffle=False,num_workers=0)
    for name,dataset in datasets.items():
        seed_all(cfg["seed"]);model=WirePinUNet(cfg["base_channels"]);model.load_state_dict(initial,strict=True)
        assert state_hash(model.state_dict())==initial_hash
        model=model.to("cuda:0");optimizer=torch.optim.AdamW(model.parameters(),lr=cfg["learning_rate"],weight_decay=cfg["weight_decay"])
        scaler=torch.amp.GradScaler("cuda",enabled=cfg["amp"]);history=[];arm_start=time.perf_counter()
        folder=output/name;folder.mkdir()
        for number in range(1,cfg["epochs"]+1):
            schedule=dataset.set_epoch(number);loader=DataLoader(dataset,batch_size=cfg["batch_size"],shuffle=False,num_workers=0)
            before=time.perf_counter();loss,steps,train_metrics=epoch(model,loader,torch.device("cuda:0"),cfg,optimizer,scaler,phase=f"{name}:{number}:train")
            cal_loss,_,cal_metrics=epoch(model,cal_loader,torch.device("cuda:0"),cfg,phase=f"{name}:{number}:internal_calibration")
            record=dict(epoch=number,train_loss=loss,calibration_loss=cal_loss,steps=len(steps),sampling=schedule,
                train_sampled_tile_metrics=train_metrics,internal_calibration_tile_metrics=cal_metrics,
                epoch_seconds=time.perf_counter()-before,learning_rate=cfg["learning_rate"],external_validation_read=False)
            history.append(record);write(folder/"history.json",history)
            print(json.dumps(dict(event="epoch_complete",arm=name,epoch=number,train_loss=loss,calibration_loss=cal_loss,
                train_Dice=train_metrics["wire_pixel_dice"],calibration_Dice=cal_metrics["wire_pixel_dice"],seconds=record["epoch_seconds"])),flush=True)
        # Selection is fixed in advance, irrespective of any calibration curve.
        cpu_state={k:v.detach().cpu() for k,v in model.state_dict().items()}
        torch.save(dict(kind="training_internal_agent_draft_sampling_pilot",state_dict=cpu_state,base_channels=cfg["base_channels"],
            epoch=16,configuration=cfg,arm=name,initial_state_sha256=initial_hash,manifest_sha256=sha(ROOT/cfg["manifest"])),folder/"last.pt")
        result=dict(arm=name,epochs=16,selected_epoch=16,checkpoint_selection="fixed_last16_only",initial_state_sha256=initial_hash,
            checkpoint_sha256=sha(folder/"last.pt"),elapsed_seconds=time.perf_counter()-arm_start,
            history_last=history[-1],external_validation_read=False,default_replaced=False)
        write(folder/"summary.json",result);results.append(result)
        del model,optimizer,scaler,cpu_state;gc.collect();torch.cuda.empty_cache()
    assert all(sha(p)==digest for p,digest in frozen.items());assert_protected()
    write(output/"summary.json",dict(status="paired_fixed16_training_complete_external_evaluation_not_started",arms=results,
        elapsed_seconds=time.perf_counter()-start,only_treatment_variable="25pct manually checked negative focus crop sampling",default_replaced=False,
        frozen_external_validation_labels_changed=False,training_pages=93,internal_calibration_pages=10,protected_original_files=assert_protected()))


if __name__=="__main__":main()
