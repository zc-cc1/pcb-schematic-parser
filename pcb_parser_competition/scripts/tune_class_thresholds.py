"""Sweep per-class confidence thresholds on the validation split, offline.

One low-confidence inference pass over the validation manifest; thresholds are
then swept per class against the same Hungarian matching protocol that
evaluate_detector.py uses, so the chosen thresholds optimize F1 directly
instead of eyeballing a global value. Output feeds evaluate_detector.py
--thresholds and the detector.class_confidence block of the run config.
"""
import argparse
import copy
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "artifacts/ultralytics"))
Path(os.environ["YOLO_CONFIG_DIR"]).mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_AUTOINSTALL", "false")
os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, str(ROOT / "scripts"))

from scipy.optimize import linear_sum_assignment

from evaluate_detector import bbox_iou
from run_baseline import fused_predictions, sliced_predictions, whole_predictions


def class_counts(predictions, expected, name, conf_threshold, iou):
    """TP/FP/FN for one class: confidence filter + IoU matching kept separate.

    The sweep must vary conf_threshold while holding the IoU match threshold
    fixed at the evaluation value; conflating them (earlier bug) inflated F1 at
    low thresholds by counting almost-any-overlap boxes as matches.
    """
    preds = [o for o in predictions if o["class_name"] == name and o["confidence"] >= conf_threshold]
    truth = [o for o in expected if o["class_name"] == name]
    matched = 0
    if preds and truth:
        ious = np.asarray([[bbox_iou(p["bbox"], g["bbox"]) for g in truth] for p in preds])
        reward = (ious >= iou).astype(float) * (min(len(preds), len(truth)) + 1) + ious
        pidx, gidx = linear_sum_assignment(-reward)
        matched = int(sum(ious[p, g] >= iou for p, g in zip(pidx, gidx)))
    return matched, len(preds) - matched, len(truth) - matched


def prf(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.
    recall = tp / (tp + fn) if tp + fn else 0.
    return precision, recall, (2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/baseline_finetuned.json")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/detection_v1/manifest.jsonl")
    parser.add_argument("--mode", choices=["whole", "sliced", "fused"], default="fused")
    parser.add_argument("--split", default="val")
    parser.add_argument("--iou", type=float, default=.5)
    parser.add_argument("--base-confidence", type=float, default=.05,
                        help="Inference confidence for the single collection pass; the sweep runs offline above it")
    parser.add_argument("--max-candidates", type=int, default=24)
    parser.add_argument("--global-threshold", type=float, default=.25, help="Current global confidence, for the comparison row")
    parser.add_argument("--output", type=Path, default=ROOT / "configs/class_thresholds.json")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines()]
    rows = [r for r in rows if r["split"] == args.split]
    if not rows:
        raise RuntimeError("No validation cases in manifest")

    detector_cfg = copy.deepcopy(cfg["detector"])
    detector_cfg["confidence"] = args.base_confidence
    whole_model = sliced_model = None
    if args.mode in ("whole", "fused"):
        from ultralytics import YOLO
        whole_model = YOLO(str(args.weights))
    if args.mode in ("sliced", "fused"):
        from sahi import AutoDetectionModel
        sliced_model = AutoDetectionModel.from_pretrained(model_type="ultralytics", model_path=str(args.weights),
                                                          confidence_threshold=args.base_confidence, device="cuda:0",
                                                          image_size=detector_cfg["imgsz"])
        sliced_model.model.overrides.update({"iou": detector_cfg["iou"], "max_det": detector_cfg["max_det"],
                                             "half": detector_cfg["half"]})

    from PIL import Image
    collected = []  # (case_id, predictions, expected)
    for row in rows:
        rgb = np.asarray(Image.open(row["image"]).convert("RGB"))
        if args.mode == "whole":
            predictions = whole_predictions(whole_model, np.ascontiguousarray(rgb[:, :, ::-1]), detector_cfg, "cuda:0")
        elif args.mode == "sliced":
            predictions = sliced_predictions(sliced_model, rgb, cfg["sahi"])
        else:
            predictions = fused_predictions(whole_model, sliced_model, rgb, "cuda:0", {**cfg, "detector": detector_cfg})
        collected.append((row["case_id"], predictions, row["objects"]))
        print(json.dumps({"collected": len(collected), "total": len(rows), "case_id": row["case_id"]},
                         ensure_ascii=False), flush=True)

    per_class_predictions = defaultdict(list)
    for _, predictions, _ in collected:
        for p in predictions:
            per_class_predictions[p["class_name"]].append(float(p["confidence"]))

    thresholds, sweep_report = {}, {}
    for name, confs in sorted(per_class_predictions.items()):
        unique = np.unique(np.asarray(confs))
        if len(unique) > args.max_candidates:
            unique = np.quantile(unique, np.linspace(0, 1, args.max_candidates))
        candidates = sorted({round(float(v), 4) for v in unique} | {args.global_threshold})
        best = None
        for t in candidates:
            tp = fp = fn = 0
            for _, predictions, expected in collected:
                m, p, n = class_counts(predictions, expected, name, t, args.iou)
                tp += m; fp += p; fn += n
            _, _, f1 = prf(tp, fp, fn)
            sweep_report.setdefault(name, []).append({"threshold": t, "f1": round(f1, 6)})
            # Ties prefer the lower threshold: recall costs nothing at equal F1.
            if best is None or f1 > best[1] + 1e-9:
                best = (t, f1, tp, fp, fn)
        thresholds[name] = best[0]
        sweep_report[name] = {"best": {"threshold": best[0], "f1": round(best[1], 6),
                                       "tp": best[2], "fp": best[3], "fn": best[4]},
                              "candidates": sweep_report[name],
                              "global_0.25_f1": next((c["f1"] for c in sweep_report[name]
                                                      if abs(c["threshold"] - args.global_threshold) < 1e-9), None)}

    tp = fp = fn = 0
    for _, predictions, expected in collected:
        for name in {o["class_name"] for o in expected} | {o["class_name"] for o in predictions}:
            m, p, n = class_counts(predictions, expected, name, thresholds.get(name, args.global_threshold), args.iou)
            tp += m; fp += p; fn += n
    _, _, micro_f1 = prf(tp, fp, fn)
    tp0 = fp0 = fn0 = 0
    for _, predictions, expected in collected:
        for name in {o["class_name"] for o in expected} | {o["class_name"] for o in predictions}:
            m, p, n = class_counts(predictions, expected, name, args.global_threshold, args.iou)
            tp0 += m; fp0 += p; fn0 += n
    _, _, micro_f1_global = prf(tp0, fp0, fn0)

    output = {"version": "0.1", "mode": args.mode, "weights_sha256": hashlib.sha256(args.weights.read_bytes()).hexdigest(),
              "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
              "split": args.split, "iou_match_threshold": args.iou, "base_confidence": args.base_confidence,
              "thresholds": thresholds,
              "micro_f1_with_thresholds": round(micro_f1, 6),
              "micro_f1_at_global_threshold": round(micro_f1_global, 6),
              "definition": "Same class-name + IoU max-cardinality protocol as evaluate_detector.py",
              "sweep": sweep_report}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "micro_f1_with_thresholds": round(micro_f1, 6),
                      "micro_f1_at_global_threshold": round(micro_f1_global, 6),
                      "classes_tuned": len(thresholds)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
