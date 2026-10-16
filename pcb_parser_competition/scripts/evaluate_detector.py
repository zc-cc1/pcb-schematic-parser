"""Offline bbox/category evaluation. This entrypoint may read labels; inference does not."""
import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "artifacts/ultralytics"))
Path(os.environ["YOLO_CONFIG_DIR"]).mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_AUTOINSTALL", "false")
os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO
from run_baseline import fused_predictions, sliced_predictions, whole_predictions, apply_class_confidence


def bbox_iou(a, b):
    area_a = max(0, a[2]-a[0]) * max(0, a[3]-a[1])
    area_b = max(0, b[2]-b[0]) * max(0, b[3]-b[1])
    overlap = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    return overlap/(area_a+area_b-overlap) if area_a+area_b-overlap else 0.


def counts_for_case(predicted, expected, threshold):
    # Maximum one-to-one matching within each class avoids greedy assignment artifacts.
    from scipy.optimize import linear_sum_assignment
    counts = {}
    for name in sorted({o["class_name"] for o in expected+predicted}):
        preds = [o for o in predicted if o["class_name"] == name]
        truth = [o for o in expected if o["class_name"] == name]
        matched = 0
        if preds and truth:
            ious = np.asarray([[bbox_iou(p["bbox"], g["bbox"]) for g in truth] for p in preds])
            # Reward valid matches before overlap so cardinality wins over higher IoU.
            reward = (ious >= threshold).astype(float) * (min(len(preds),len(truth))+1) + ious
            pidx, gidx = linear_sum_assignment(-reward)
            matched = int(sum(ious[p,g] >= threshold for p,g in zip(pidx,gidx)))
        counts[name] = {"tp": matched, "fp": len(preds)-matched, "fn": len(truth)-matched}
    return counts


def metric(counts):
    tp, fp, fn = (counts.get(k,0) for k in ("tp", "fp", "fn"))
    return {**counts, "precision": tp/(tp+fp) if tp+fp else 0., "recall": tp/(tp+fn) if tp+fn else 0.,
            "f1": 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/detection_v1/manifest.jsonl")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/baseline.json")
    parser.add_argument("--mode", choices=["whole", "sliced", "fused"], default="whole")
    parser.add_argument("--split", default="val")
    parser.add_argument("--case-id", action="append")
    parser.add_argument("--iou", type=float, default=.5)
    parser.add_argument("--thresholds", type=Path, help="Per-class confidence JSON from tune_class_thresholds.py")
    parser.add_argument("--device", default="cuda:0", help="Use cpu while the GPU is occupied")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    thresholds = json.loads(args.thresholds.read_text(encoding="utf-8")) if args.thresholds else cfg["detector"].get("class_confidence")
    if isinstance(thresholds, dict) and "thresholds" in thresholds:
        thresholds = thresholds["thresholds"]
    rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines()]
    rows = [r for r in rows if r["split"] == args.split and (not args.case_id or r["case_id"] in args.case_id)]
    if not rows:
        raise RuntimeError("No evaluation cases")
    if args.mode in ("whole", "fused"):
        whole_model = YOLO(str(args.weights))
    if args.mode in ("sliced", "fused"):
        from sahi import AutoDetectionModel
        sliced_conf = cfg.get("fusion", {}).get("sliced_confidence", cfg["detector"]["confidence"]) if args.mode == "fused" else cfg["detector"]["confidence"]
        sliced_model = AutoDetectionModel.from_pretrained(model_type="ultralytics", model_path=str(args.weights), confidence_threshold=sliced_conf, device=args.device, image_size=cfg["detector"]["imgsz"])
        sliced_model.model.overrides.update({"iou":cfg["detector"]["iou"], "max_det":cfg["detector"]["max_det"], "half":cfg["detector"]["half"]})
    summary = {"status": "provisional_offline_detection_evaluation", "mode": args.mode, "split": args.split,
               "confidence": cfg["detector"]["confidence"], "iou_match_threshold": args.iou,
               "detector_config": cfg["detector"], "sahi_config": cfg["sahi"] if args.mode in ("sliced", "fused") else None,
               "class_thresholds": thresholds,
               "fusion_config": cfg.get("fusion"), "device": args.device,
               "weights": str(args.weights.resolve()), "weights_sha256": hashlib.sha256(args.weights.read_bytes()).hexdigest(),
               "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(), "cases": [],
               "definition": "Class-name matching + bbox IoU threshold + maximum-cardinality one-to-one matching; micro P/R/F1 at fixed confidence",
               "official_competition_f1": False, "label_caveat": "Corrected raw bbox labels; incomplete manual certification and engineering provenance"}
    aggregate = defaultdict(Counter)
    started = time.perf_counter()
    for index, row in enumerate(rows):
        rgb = np.asarray(Image.open(row["image"]).convert("RGB"))
        if args.mode == "whole":
            predictions = whole_predictions(whole_model, np.ascontiguousarray(rgb[:,:,::-1]), cfg["detector"], args.device)
        elif args.mode == "sliced":
            predictions = sliced_predictions(sliced_model, rgb, cfg["sahi"])
        else:
            predictions = fused_predictions(whole_model, sliced_model, rgb, args.device, cfg)
        if thresholds:
            predictions = apply_class_confidence(predictions, thresholds)
        counts = counts_for_case(predictions, row["objects"], args.iou)
        for name, count in counts.items():
            aggregate[name].update(count)
        total = Counter()
        for count in counts.values():
            total.update(count)
        summary["cases"].append({"case_id":row["case_id"], "ground_truth_count":len(row["objects"]), "prediction_count":len(predictions),
                                 "counts":metric(total), "by_class":counts, "predictions":predictions})
        if (index+1) % 10 == 0:
            print(json.dumps({"evaluated":index+1,"total":len(rows)},ensure_ascii=False),flush=True)
    total = Counter()
    for count in aggregate.values():
        total.update(count)
    summary["overall"] = metric(total)
    summary["by_class"] = {name:metric(count) for name,count in aggregate.items()}
    summary["wall_seconds"] = time.perf_counter()-started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"cases":len(rows), "overall":summary["overall"], "output":str(args.output)},ensure_ascii=False),flush=True)


if __name__ == "__main__":
    main()
