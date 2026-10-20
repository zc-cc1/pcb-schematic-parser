"""Offline diagnosis: is the U-Net wire/pin gap a model problem or a label/geometry problem?

Reads only cached probability maps and draft labels; writes a JSON report.
Does not train, does not modify labels, does not call the official scorer.

Four questions:
  Q1 wire: how much of the pixel mismatch survives when the label is dilated
     to the measured rendered stroke width (2..4 px)?  Remaining part = model error.
  Q2 wire: how much does allowing a +/-1 px shift (label erosion/dilation window)
     explain?  A large jump means the centerline is fine and the width is the issue.
  Q3 pin: what is the point-set F1 under the official 5 px tolerance, and how much
     of the FP mass sits close to a labeled wire (i.e. model predicting
     'wire endpoint' instead of 'component terminal')?
  Q4 label self-consistency: tips far from any labeled wire are unreachable
     false negatives regardless of model quality.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return {"tp": int(tp), "fp": int(fp), "fn": int(fn),
            "precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4)}


def peaks(heat, threshold, kernel=5):
    """Local maxima on the probability map, sub-pixel via quadratic fit on log-prob."""
    pad = kernel // 2
    maxima = ndimage.maximum_filter(heat, size=kernel, mode="constant", cval=0)
    candidates = (heat == maxima) & (heat >= threshold)
    labels, count = ndimage.label(candidates, structure=np.ones((3, 3), bool))
    points = []
    for index, center in enumerate(ndimage.center_of_mass(candidates, labels, range(1, count + 1)), 1):
        y, x = center
        yi, xi = int(round(y)), int(round(x))
        if not (pad <= yi < heat.shape[0] - pad and pad <= xi < heat.shape[1] - pad):
            points.append((float(x), float(y), float(heat[yi, xi])))
            continue
        # parabola on log probability along each axis
        def vertex(a, b, c):
            denominator = a - 2 * b + c
            return 0.0 if abs(denominator) < 1e-9 else 0.5 * (a - c) / denominator
        logp = np.log(np.clip(heat[yi - 1:yi + 2, xi - 1:xi + 2], 1e-9, None))
        dx = vertex(logp[1, 0], logp[1, 1], logp[1, 2])
        dy = vertex(logp[0, 1], logp[1, 1], logp[2, 1])
        points.append((x + dx, y + dy, float(heat[yi, xi])))
    return points


def bipartite_match(pred, truth, tolerance, iou_pred=None):
    """Greedy-by-distance one-to-one matching; returns tp, list of unmatched pred idx."""
    if not truth or not pred:
        return 0, list(range(len(pred))), list(range(len(truth)))
    tree = cKDTree(np.asarray(truth))
    distances, indices = tree.query(np.asarray(pred), k=1)
    order = np.argsort(distances)
    used_pred, used_truth, tp = set(), set(), 0
    for i in order:
        j = int(indices[i])
        if distances[i] <= tolerance and j not in used_truth:
            used_truth.add(j)
            used_pred.add(int(i))
            tp += 1
    return (tp,
            [i for i in range(len(pred)) if i not in used_pred],
            [j for j in range(len(truth)) if j not in used_truth])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=ROOT / "runs/unet_inference_v2/cache")
    parser.add_argument("--labels", type=Path, default=ROOT / "data/segmentation_unet_v1")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/segmentation_unet_v1/manifest.jsonl")
    parser.add_argument("--wire-threshold", type=float, default=0.5)
    parser.add_argument("--pin-threshold", type=float, default=0.8)
    parser.add_argument("--pin-tolerance", type=float, default=5.0)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/unet_ceiling_diagnosis.json")
    args = parser.parse_args()

    manifest = {r["case_id"]: r for r in
                (json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines())}
    cases = sorted(p.name for p in args.cache.iterdir() if p.is_dir())
    rows = []
    for case_id in cases:
        npz = args.cache / case_id / "scale_1.0.npz"
        if not npz.exists():
            continue
        probability = np.load(npz)["probability"]
        wire_pred = probability[0] >= args.wire_threshold
        wire_label = np.asarray(Image.open(args.labels / case_id / "wire_mask.png")) > 0
        entry = {"case_id": case_id, "split": manifest[case_id]["split"],
                 "valid_label": bool(manifest[case_id]["structurally_usable_draft"]),
                 "width": int(wire_label.shape[1]), "height": int(wire_label.shape[0])}

        # Q1/Q2: tolerance-aware wire matching.
        for name, radius in (("exact", 0), ("tolerance_1px", 1), ("tolerance_2px", 2), ("tolerance_3px", 3)):
            if radius:
                label_d = ndimage.binary_dilation(wire_label, ndimage.generate_binary_structure(2, 2), iterations=radius)
                pred_d = ndimage.binary_dilation(wire_pred, ndimage.generate_binary_structure(2, 2), iterations=radius)
                # one-to-one approximation: a predicted pixel counts if it is near a label
                # pixel AND the label pixel is only claimed once, measured by set intersection
                # of the skeleton-free masks (centroid-free, so this is an upper bound).
                tp = int((wire_pred & label_d).sum())
                fn = int((wire_label & ~pred_d).sum())
                fp = int((wire_pred & ~label_d).sum())
            else:
                tp = int((wire_pred & wire_label).sum())
                fp = int((wire_pred & ~wire_label).sum())
                fn = int((wire_label & ~wire_pred).sum())
            entry[f"wire_{name}"] = prf(tp, fp, fn)

        # Q4: label self-consistency (tips vs labeled wire) and distance stats.
        tips = [t["point"] for t in json.loads((args.labels / case_id / "pin_points.json").read_text(encoding="utf-8"))]
        if wire_label.any() and tips:
            distance = ndimage.distance_transform_edt(~wire_label)
            tip_dist = [float(distance[min(int(y), distance.shape[0] - 1), min(int(x), distance.shape[1] - 1)])
                        for x, y in tips]
        else:
            tip_dist = []
        entry["label_self_consistency"] = {
            "tips": len(tips),
            "within_5px_of_labeled_wire": int(sum(1 for d in tip_dist if d <= 5)),
            "unreachable_fraction": round(1 - sum(1 for d in tip_dist if d <= 5) / len(tip_dist), 4) if tip_dist else None,
            "median_px": round(float(np.median(tip_dist)), 2) if tip_dist else None,
            "p95_px": round(float(np.percentile(tip_dist, 95)), 2) if tip_dist else None,
        }

        # Q3: pin point-set F1 under the official tolerance.
        pred_points = peaks(probability[1], args.pin_threshold)
        tp, fp_idx, fn_idx = bipartite_match([(p[0], p[1]) for p in pred_points], tips, args.pin_tolerance)
        entry["pin_points"] = prf(tp, len(fp_idx), len(fn_idx))
        if fp_idx and wire_label.any():
            wire_tree = cKDTree(np.argwhere(wire_label)[:, ::-1])  # (x, y)
            fp_positions = np.asarray([pred_points[i][:2] for i in fp_idx])
            fp_to_wire, _ = wire_tree.query(fp_positions, k=1)
            entry["pin_false_positives"] = {
                "count": int(len(fp_idx)),
                "share_within_2px_of_wire": round(float((fp_to_wire <= 2).mean()), 4),
                "share_within_5px_of_wire": round(float((fp_to_wire <= 5).mean()), 4),
                "median_px_to_wire": round(float(np.median(fp_to_wire)), 2),
            }
        rows.append(entry)

    def aggregate(subset, key):
        total = {"tp": 0, "fp": 0, "fn": 0}
        for row in subset:
            for metric in total:
                total[metric] += row[key][metric]
        return prf(total["tp"], total["fp"], total["fn"])

    usable = [r for r in rows if r["valid_label"]]
    report = {
        "status": "offline_ceiling_diagnosis",
        "scope": "cached v1 probabilities (26 unique pages) and draft labels; not official scores",
        "config": {"wire_threshold": args.wire_threshold, "pin_threshold": args.pin_threshold,
                   "pin_tolerance_px": args.pin_tolerance},
        "cases": rows,
        "wire_by_tolerance": {key: aggregate(usable, key) for key in
                              ("wire_exact", "wire_tolerance_1px", "wire_tolerance_2px", "wire_tolerance_3px")},
        "pin_points": aggregate(usable, "pin_points"),
        "label_self_consistency": {
            "tips": int(sum(r["label_self_consistency"]["tips"] for r in usable)),
            "unreachable": int(sum(r["label_self_consistency"]["tips"] - r["label_self_consistency"]["within_5px_of_labeled_wire"] for r in usable)),
        },
        "pin_false_positive_wire_proximity": {
            key: round(float(np.mean([r["pin_false_positives"][key] for r in usable if "pin_false_positives" in r])), 4)
            for key in ("share_within_2px_of_wire", "share_within_5px_of_wire", "median_px_to_wire")
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("== wire (26 cached pages, threshold %.2f) ==" % args.wire_threshold)
    for key, value in report["wire_by_tolerance"].items():
        print("  %-18s P %.4f R %.4f F1 %.4f" % (key, value["precision"], value["recall"], value["f1"]))
    print("== pin (point-set, %.0f px tolerance, threshold %.2f) ==" % (args.pin_tolerance, args.pin_threshold))
    print("  P %.4f R %.4f F1 %.4f  TP %d FP %d FN %d" % (
        report["pin_points"]["precision"], report["pin_points"]["recall"], report["pin_points"]["f1"],
        report["pin_points"]["tp"], report["pin_points"]["fp"], report["pin_points"]["fn"]))
    print("  FP near labeled wire:", report["pin_false_positive_wire_proximity"])
    print("  unreachable tips (label inconsistency): %d / %d" % (
        report["label_self_consistency"]["unreachable"], report["label_self_consistency"]["tips"]))
    print("written:", args.output)


if __name__ == "__main__":
    main()
