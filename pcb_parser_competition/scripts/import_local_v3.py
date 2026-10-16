"""Import the user's local v3 detector with hashes and backups; never train."""
import argparse
import copy
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--source",type=Path,default=Path(r"C:\Users\Pc\Desktop\新建文件夹\pcb_parser_competition"))
    args=parser.parse_args()
    source=args.source.resolve()
    target=ROOT.resolve()
    if source==target: raise ValueError("Source and destination must differ")
    snapshot=ROOT/"data/local_imports/v3_20261011"
    evidence=json.loads((source/"runs/eval_v3_fused_v2thr.json").read_text(encoding="utf-8"))
    if evidence["mode"]!="fused" or evidence["iou_match_threshold"]!=0.5:
        raise ValueError("Unexpected adoption evaluation contract")
    weight=source/"weights/detector_dataset300_v3_nano.pt"
    if digest(weight)!=evidence["weights_sha256"]: raise ValueError("Source weight differs from final evaluation")
    if digest(ROOT/"data/detection_v1/manifest.jsonl")!=evidence["manifest_sha256"]:
        raise ValueError("Local detection validation manifest differs from source evaluation")
    paths=["src/pcb_parser/fusion.py","scripts/run_baseline.py","scripts/evaluate_detector.py",
           "scripts/tune_class_thresholds.py","scripts/error_breakdown.py","scripts/train_detector.py",
           "scripts/download_yolo_weights.py","tests/test_fusion.py",
           "configs/detector_training_v3.json","configs/detector_training_v3_big.json",
           "configs/detector_training_v3_v11s.json","configs/detector_training_v3_y11s_quick.json",
           "configs/baseline_v3.json","weights/detector_dataset300_v3_nano.pt"]
    files=[]
    for relative in paths:
        src=source/relative;dst=ROOT/relative
        if not src.is_file(): raise FileNotFoundError(src)
        old_hash=digest(dst) if dst.is_file() else None
        new_hash=digest(src)
        if dst.exists() and old_hash!=new_hash:
            backup=snapshot/"prior_workspace_files"/relative
            if not backup.exists():
                backup.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(dst,backup)
        dst.parent.mkdir(parents=True,exist_ok=True)
        if old_hash!=new_hash: shutil.copy2(src,dst)
        if digest(dst)!=new_hash: raise RuntimeError(f"Copy verification failed: {relative}")
        files.append({"source":str(src),"destination":str(dst),"sha256":new_hash,"previous_sha256":old_hash})
    for relative in ["runs/eval_v3_fused_v2thr.json","runs/eval_v3_fused.json",
                     "runs/training/detector_dataset300_v3_nano_record.json","configs/class_thresholds.json",
                     "docs/optimization_v3.md"]:
        src=source/relative;dst=snapshot/"source_evidence"/relative
        dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
        files.append({"source":str(src),"destination":str(dst),"sha256":digest(dst),"evidence_only":True})
    config=json.loads((source/"configs/baseline_v3.json").read_text(encoding="utf-8"))
    config["detector"]=copy.deepcopy(evidence["detector_config"])
    config["detector"]["class_confidence"]=evidence["class_thresholds"]
    config["sahi"]=evidence["sahi_config"]
    config.update(version="0.4-adopted",status="v3_fused_final_thresholds_imported_for_downstream",
                  recommended_mode="fused",recommended_mode_basis="Final source evaluation 59-case micro F1 0.8830433; validation reused for threshold selection",
                  weights_sha256=evidence["weights_sha256"],class_thresholds_file="configs/class_thresholds_v3_adopted.json")
    config.pop("adoption_checklist",None)
    save(ROOT/"configs/baseline_v3_adopted.json",config)
    save(ROOT/"configs/class_thresholds_v3_adopted.json",{
        "thresholds":evidence["class_thresholds"],"weights_sha256":evidence["weights_sha256"],
        "source_evaluation":str(snapshot/"source_evidence/runs/eval_v3_fused_v2thr.json"),
        "mode":"fused","inference_confidence":evidence["confidence"],"iou_match_threshold":0.5,
        "status":"frozen_final_evaluation_thresholds_not_retuned_here"})
    save(snapshot/"import_manifest.json",{"source_root":str(source),"destination_root":str(ROOT),
        "imported_at_utc":datetime.now(timezone.utc).isoformat(),"files":files,
        "source_overall":evidence["overall"],"source_validation_cases":len(evidence["cases"]),
        "new_training_started":False,"original_data_copied_or_modified":False})
    print(json.dumps({"files_verified":len(files),"weights_sha256":evidence["weights_sha256"],
                      "source_f1":evidence["overall"]["f1"],"config":"configs/baseline_v3_adopted.json"}))


if __name__=="__main__": main()
