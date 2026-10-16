"""Break detector evaluation results down by class, GT size and source group.

Offline: reads an evaluate_detector.py output JSON plus the manifest, re-runs
the same per-class Hungarian matching to recover match identity, and attributes
every FN/FP to a GT size bucket and a source group. Answers "where do the 20%
misses live" without another inference pass.
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_detector import bbox_iou

from scipy.optimize import linear_sum_assignment

SIZE_BUCKETS = [("tiny", 0.015), ("small", 0.03), ("medium", 0.06), ("large", float("inf"))]


def size_bucket(bbox, width, height):
    relative = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / max(width, height)
    for name, limit in SIZE_BUCKETS:
        if relative < limit:
            return name
    return "large"


def match_case(predictions, expected, iou):
    """Return (matched_pred_indices, matched_gt_indices, tp, fp, fn) per class name."""
    per_class = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0,
                                     "fn_objects": [], "fp_objects": []})
    for name in sorted({o["class_name"] for o in expected} | {o["class_name"] for o in predictions}):
        preds = [(i, o) for i, o in enumerate(predictions) if o["class_name"] == name]
        truth = [(i, o) for i, o in enumerate(expected) if o["class_name"] == name]
        entry = per_class[name]
        if preds and truth:
            ious = np.asarray([[bbox_iou(p["bbox"], g["bbox"]) for _, g in truth] for _, p in preds])
            reward = (ious >= iou).astype(float) * (min(len(preds), len(truth)) + 1) + ious
            pidx, gidx = linear_sum_assignment(-reward)
            matched_preds = {p for p, g in zip(pidx, gidx) if ious[p, g] >= iou}
            matched_gts = {g for p, g in zip(pidx, gidx) if ious[p, g] >= iou}
            entry["tp"] += len(matched_preds)
            for local_p, (global_p, obj) in enumerate(preds):
                if local_p not in matched_preds:
                    entry["fp"] += 1
                    entry["fp_objects"].append(obj)
            for local_g, (global_g, obj) in enumerate(truth):
                if local_g not in matched_gts:
                    entry["fn"] += 1
                    entry["fn_objects"].append(obj)
        else:
            entry["fp"] += len(preds)
            entry["fp_objects"].extend(o for _, o in preds)
            entry["fn"] += len(truth)
            entry["fn_objects"].extend(o for _, o in truth)
    return per_class


def prf(tp, fp, fn):
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": round(tp / (tp + fp), 4) if tp + fp else 0.,
            "recall": round(tp / (tp + fn), 4) if tp + fn else 0.,
            "f1": round(2 * tp / (2 * tp + fp + fn), 4) if 2 * tp + fp + fn else 0.}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-json", type=Path, required=True, help="evaluate_detector.py output")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/detection_v1/manifest.jsonl")
    parser.add_argument("--iou", type=float, default=.5, help="Must match the evaluation run")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    report = json.loads(args.eval_json.read_text(encoding="utf-8"))
    manifest = {r["case_id"]: r for r in
                (json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines())}

    by_class = defaultdict(Counter)
    by_group = defaultdict(Counter)
    by_gt_size = defaultdict(Counter)
    by_fp_size = defaultdict(Counter)
    fn_histogram, fp_histogram = Counter(), Counter()
    fp_confidences = defaultdict(list)
    worst_cases = []
    for case in report["cases"]:
        row = manifest[case["case_id"]]
        group = row.get("group", "unknown")
        width, height = Image.open(row["image"]).size
        matched = match_case(case["predictions"], row["objects"], args.iou)
        case_tp = case_fp = case_fn = 0
        for name, entry in matched.items():
            for key in ("tp", "fp", "fn"):
                by_class[name][key] += entry[key]
            case_tp += entry["tp"]; case_fp += entry["fp"]; case_fn += entry["fn"]
            for obj in entry["fn_objects"]:
                by_gt_size[size_bucket(obj["bbox"], width, height)]["fn"] += 1
                fn_histogram[name] += 1
            for obj in entry["fp_objects"]:
                bucket = size_bucket(obj["bbox"], width, height)
                by_fp_size[bucket]["fp"] += 1
                fp_histogram[name] += 1
                fp_confidences[name].append(round(float(obj["confidence"]), 3))
        by_group[group]["tp"] += case_tp; by_group[group]["fp"] += case_fp; by_group[group]["fn"] += case_fn
        worst_cases.append({"case_id": case["case_id"], "group": group, "gt": len(row["objects"]),
                            "tp": case_tp, "fp": case_fp, "fn": case_fn})
    # GT size buckets also need tp counts: recompute per bucket by re-matching is
    # equivalent to tp = gt_in_bucket - fn_in_bucket.
    all_gt_sizes = defaultdict(Counter)
    for case in report["cases"]:
        row = manifest[case["case_id"]]
        width, height = Image.open(row["image"]).size
        for obj in row["objects"]:
            all_gt_sizes[size_bucket(obj["bbox"], width, height)]["gt"] += 1
    for bucket, counts in list(by_gt_size.items()):
        counts["gt"] = all_gt_sizes[bucket]["gt"]
        counts["tp"] = counts["gt"] - counts["fn"]

    bucket_order = {name: i for i, (name, _) in enumerate(SIZE_BUCKETS)}
    output = {
        "status": "offline_error_breakdown",
        "eval_json": str(args.eval_json),
        "manifest_sha256": None,
        "iou_match_threshold": args.iou,
        "by_class": {name: prf(c["tp"], c["fp"], c["fn"]) for name, c in sorted(by_class.items())},
        "by_group": {name: prf(c["tp"], c["fp"], c["fn"]) for name, c in sorted(by_group.items())},
        "by_gt_size": {name: {**prf(c["tp"], 0, c["fn"]), "gt": c["gt"]} for name, c in
                       sorted(by_gt_size.items(), key=lambda kv: bucket_order.get(kv[0], 99))},
        "false_negatives": {"by_class": dict(fn_histogram.most_common()),
                            "by_gt_size": {k: v["fn"] for k, v in sorted(by_gt_size.items())},
                            "share_pct": {k: round(100 * v / max(1, sum(fn_histogram.values())), 1)
                                          for k, v in fn_histogram.most_common(10)}},
        "false_positives": {"by_class": dict(fp_histogram.most_common()),
                            "by_pred_size": {k: v["fp"] for k, v in sorted(by_fp_size.items())},
                            "mean_confidence": {k: round(sum(v) / len(v), 3) for k, v in fp_confidences.items() if v}},
        "worst_cases": sorted(worst_cases, key=lambda c: (-c["fn"], -c["fp"]))[:15],
    }
    import hashlib
    output["manifest_sha256"] = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    target = args.output or args.eval_json.with_name(args.eval_json.stem + "_breakdown.json")
    target.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# 误差拆解 (error breakdown)", "",
             f"评测文件: {args.eval_json}  IoU={args.iou}", "",
             "## 按类别", "| 类别 | TP | FP | FN | P | R | F1 |", "|---|---|---|---|---|---|---|"]
    for name, m in output["by_class"].items():
        lines.append(f"| {name} | {m['tp']} | {m['fp']} | {m['fn']} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} |")
    lines += ["", "## 按来源组", "| 组 | TP | FP | FN | P | R | F1 |", "|---|---|---|---|---|---|---|"]
    for name, m in output["by_group"].items():
        lines.append(f"| {name} | {m['tp']} | {m['fp']} | {m['fn']} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} |")
    lines += ["", "## 按 GT 尺寸桶（相对图长边）", "| 桶 | GT | TP | FN | 召回 |", "|---|---|---|---|---|"]
    for name in ("tiny", "small", "medium", "large"):
        c = by_gt_size.get(name, Counter())
        gt, fn = c.get("gt", 0), c.get("fn", 0)
        lines.append(f"| {name} | {gt} | {gt - fn} | {fn} | {1 - fn / gt:.3f} |" if gt else f"| {name} | 0 | 0 | 0 | - |")
    lines += ["", "## FN 大户（漏检最多类别）", ""]
    lines += [f"- {k}: {v} ({output['false_negatives']['share_pct'].get(k, 0)}% 的全部漏检)"
              for k, v in list(fn_histogram.most_common(10))]
    lines += ["", "## 最差案例 Top10", "", "| case | 组 | GT | TP | FP | FN |", "|---|---|---|---|---|---|"]
    for c in sorted(worst_cases, key=lambda x: (-x["fn"], -x["fp"]))[:10]:
        lines.append(f"| {c['case_id']} | {c['group']} | {c['gt']} | {c['tp']} | {c['fp']} | {c['fn']} |")
    print("\n".join(lines))
    print(f"\nwritten: {target}")


if __name__ == "__main__":
    main()
