"""Why is pin precision 0.657?  Attribute every predicted peak before changing anything.

For a sample of validation pages this records, for every local peak above a low
threshold:

  confidence          peak probability
  matched             within 5 px of a ground-truth tip
  in_component_box    inside any component bbox (symbol body / designator area)
  ink_local           mean darkness in a 5 px window (text and strokes are ink)
  distance_to_wire    distance to the nearest supervised wire pixel
  distance_to_tip     distance to the nearest true tip (when unmatched)

Then it sweeps the decode threshold and reports the precision/recall curve, plus
where the false positives live.  Output drives the decision: threshold-only fix,
body/ignore supervision, or loss reweighting.  No training, no writes to data.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from pcb_parser.unet_model import WirePinUNet  # noqa: E402
from evaluate_unet_baseline import match, peaks  # noqa: E402
from evaluate_unet_v2 import predict  # noqa: E402

V2 = Path(r"D:/pcb_unet_v2/data/segmentation_unet_v2")
STD = ROOT / "data/standardized"
THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95)


def f1(tp, fp, fn):
    return 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0


def load_boxes(case_id, shape):
    path = STD / case_id / f"{case_id}.json"
    boxes = np.zeros(shape[:2], bool)
    if not path.exists():
        return boxes, 0
    data = json.loads(path.read_text(encoding="utf-8"))
    count = 0
    for component in (data.get("components") or {}).values():
        box = component.get("bbox")
        if not box or len(box) != 4:
            continue
        x1, y1, x2, y2 = (int(round(v)) for v in box)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(shape[1], max(x1 + 1, x2)), min(shape[0], max(y1 + 1, y2))
        boxes[y1:y2, x1:x2] = True
        count += 1
    return boxes, count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path(r"D:/pcb_unet_v2/runs/A2_full60/best.pt"))
    parser.add_argument("--pages", type=int, default=12)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=Path(r"D:/pcb_unet_v2/reports/pin_error_analysis.json"))
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model = WirePinUNet(checkpoint["base_channels"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])

    rows = [json.loads(line) for line in (V2 / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    val = [r for r in rows if r["split"] == "val"]
    # spread the sample over tip counts so dense pages are represented
    val = sorted(val, key=lambda r: -r["stats"]["pin_tips"])[: args.pages]

    per_page, sweep = [], {t: [0, 0, 0] for t in THRESHOLDS}
    records = []
    truth_tips = {}
    for row in val:
        case_id = row["case_id"]
        with Image.open(row["image"]) as source:
            image = np.asarray(source.convert("RGB"))
        grey = image.mean(axis=2)
        with Image.open(row["wire_mask"]) as mask:
            wire = np.asarray(mask) > 0
        tips = [tuple(p["point"]) for p in json.loads(Path(row["pin_points"]).read_text(encoding="utf-8"))]
        if not tips:
            continue
        boxes, box_count = load_boxes(case_id, image.shape)
        probability = predict(model, image, device, 512, 384)
        heat = probability[1]
        wire_distance = ndimage.distance_transform_edt(~wire)
        candidates = []
        for x, y in peaks(heat, min(THRESHOLDS), 5):
            xi, yi = int(round(x)), int(round(y))
            if 0 <= yi < heat.shape[0] and 0 <= xi < heat.shape[1]:
                candidates.append((float(x), float(y), float(heat[yi, xi])))
        tip_tree = cKDTree(np.asarray(tips))
        tip_distance = tip_tree.query(np.asarray([(c[0], c[1]) for c in candidates]))[0] if candidates else []

        for threshold in THRESHOLDS:
            kept = [c for c in candidates if c[2] >= threshold]
            tp, unmatched_pred, unmatched_truth = match([(c[0], c[1]) for c in kept], tips, 5.0)
            sweep[threshold][0] += tp
            sweep[threshold][1] += len(unmatched_pred)
            sweep[threshold][2] += len(unmatched_truth)

        for index, (x, y, _conf) in enumerate(candidates):
            xi, yi = int(round(x)), int(round(y))
            if not (0 <= yi < grey.shape[0] and 0 <= xi < grey.shape[1]):
                continue
            records.append({
                "case_id": case_id,
                "x": round(float(x), 1), "y": round(float(y), 1),
                "confidence": round(float(probability[1][yi, xi]), 3),
                "distance_to_tip": round(float(tip_distance[index]), 2) if len(tip_distance) else None,
                "matched_5px": bool(tip_distance[index] <= 5.0) if len(tip_distance) else False,
                "in_component_box": bool(boxes[yi, xi]),
                "ink_local": round(float(grey[max(0, yi - 2):yi + 3, max(0, xi - 2):xi + 3].mean()), 1),
                "distance_to_wire": round(float(wire_distance[yi, xi]), 2),
            })
        per_page.append({"case_id": case_id, "tips": len(tips), "component_boxes": box_count,
                         "peaks": len(candidates)})
        truth_tips[case_id] = [[round(float(x), 1), round(float(y), 1)] for x, y in tips]
        print(f"  {case_id}: {len(candidates)} peaks, {len(tips)} tips", flush=True)

    matched = [r for r in records if r["matched_5px"]]
    unmatched = [r for r in records if not r["matched_5px"]]
    def share(rows_, predicate):
        return round(float(np.mean([predicate(r) for r in rows_])), 4) if rows_ else None
    report = {
        "status": "pin_error_analysis",
        "checkpoint": str(args.checkpoint),
        "pages": per_page,
        "threshold_sweep": {str(t): {"tp": v[0], "fp": v[1], "fn": v[2],
                                     "precision": round(v[0] / (v[0] + v[1]), 4) if v[0] + v[1] else None,
                                     "recall": round(v[0] / (v[0] + v[2]), 4) if v[0] + v[2] else None,
                                     "f1": round(f1(*v), 4)} for t, v in sweep.items()},
        "peaks": {"total": len(records), "matched": len(matched), "unmatched": len(unmatched)},
        "fp_structure": {
            "share_in_component_box": share(unmatched, lambda r: r["in_component_box"]),
            "share_on_ink": share(unmatched, lambda r: r["ink_local"] < 200),
            "share_near_wire_5px": share(unmatched, lambda r: r["distance_to_wire"] <= 5),
            "median_confidence": round(float(np.median([r["confidence"] for r in unmatched])), 3) if unmatched else None,
            "median_distance_to_wire": round(float(np.median([r["distance_to_wire"] for r in unmatched])), 2) if unmatched else None,
            "median_distance_to_tip": round(float(np.median([r["distance_to_tip"] for r in unmatched])), 2) if unmatched else None,
        },
        "tp_structure": {
            "share_in_component_box": share(matched, lambda r: r["in_component_box"]),
            "median_confidence": round(float(np.median([r["confidence"] for r in matched])), 3) if matched else None,
            "median_distance_to_wire": round(float(np.median([r["distance_to_wire"] for r in matched])), 2) if matched else None,
        },
        "records": records,
        "truth_tips": truth_tips,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # Re-score the same peaks with geometric reweighting, one-to-one matched.
    def score_sweep(scorer, grid):
        best = None
        for threshold in grid:
            tp = fp = fn = 0
            for case_id, tips_ in truth_tips.items():
                rows_ = [r for r in records if r["case_id"] == case_id and scorer(r) >= threshold]
                hit, unmatched_pred, unmatched_truth = match([(r["x"], r["y"]) for r in rows_],
                                                             [tuple(t) for t in tips_], 5.0)
                tp += hit
                fp += len(unmatched_pred)
                fn += len(unmatched_truth)
            value = f1(tp, fp, fn)
            if best is None or value > best[0]:
                best = (value, threshold, tp, fp, fn)
        return best

    reweighting = {}
    plain = score_sweep(lambda r: r["confidence"], np.arange(0.5, 0.99, 0.01))
    reweighting["confidence_only"] = {"f1": round(plain[0], 4), "threshold": round(float(plain[1]), 2),
                                      "tp": plain[2], "fp": plain[3], "fn": plain[4]}
    for k in (2.0, 3.0, 5.0, 8.0, 12.0):
        best = score_sweep(lambda r, k=k: r["confidence"] / (1 + r["distance_to_wire"] / k), np.arange(0.2, 0.95, 0.01))
        reweighting[f"conf/(1+dw/{k:g})"] = {"f1": round(best[0], 4), "threshold": round(float(best[1]), 2),
                                             "tp": best[2], "fp": best[3], "fn": best[4]}
    report["reweighting"] = reweighting
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nthreshold sweep (12 densest val pages):")
    for threshold in THRESHOLDS:
        value = report["threshold_sweep"][str(threshold)]
        print(f"  >={threshold:.2f}  P {value['precision']}  R {value['recall']}  F1 {value['f1']}")
    print(f"\npeaks {report['peaks']}, false positives {len(unmatched)}")
    for key, value in report["fp_structure"].items():
        print(f"  FP {key}: {value}")
    print("  TP structure:", report["tp_structure"])
    print("\ngeometric reweighting (one-to-one matched, same peaks):")
    for name, value in reweighting.items():
        print(f"  {name:<18} F1 {value['f1']:.4f}  at t={value['threshold']}  (TP {value['tp']} FP {value['fp']} FN {value['fn']})")
    print("written:", args.output)


if __name__ == "__main__":
    main()
