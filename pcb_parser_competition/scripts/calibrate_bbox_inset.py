"""How much would insetting the exported bbox buy, under the official 20 px rule?

The reference `components.bbox` is the symbol BODY box; the detector's box also covers the
leads.  Measured on the public Golden pages, every reference pin tip sits OUTSIDE the
reference box (286/292), with a median lead of 20 px and a maximum of 40 px, while the
official tolerance is 20 px per side.  So a pure geometry mistake can cost a component
that was found in exactly the right place.

This script does not change the pipeline.  It shrinks the stored prediction boxes by a
constant per side (and, in a second pass, by a per-type constant taken from the measured
median deviation) and re-scores, so the size of the prize is known before anyone touches
the exporter.

    python scripts/calibrate_bbox_inset.py --predictions D:/pcb_unet_v2/autoid_ab
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pcb_parser.contest_metrics import evaluate_cases, normalise_type  # noqa: E402

GT_DIR = Path(r"C:/Users/Pc/Desktop/新建文件夹/赛题六公开数据集/赛题六公开数据集/10_GTcase")


def image_height(path: Path) -> float:
    from PIL import Image
    with Image.open(path) as handle:
        return float(handle.height)


def load(cases, root: Path) -> dict:
    subs = {}
    for case in cases:
        subs[case["case"]] = json.loads(
            (root / case["case"] / "partial_submission.json").read_text(encoding="utf-8"))
    return subs


def inset_boxes(prediction, per_type, default=0.0):
    out = {"components": {}, "pins": prediction.get("pins") or {},
           "nets": prediction.get("nets") or {}}
    for key, component in (prediction.get("components") or {}).items():
        x1, y1, x2, y2 = (float(v) for v in component["bbox"])
        cut = per_type.get(normalise_type(component.get("type")), default)
        half_w, half_h = (x2 - x1) / 2, (y2 - y1) / 2
        cut_x, cut_y = min(cut, half_w * 0.8), min(cut, half_h * 0.8)
        out["components"][key] = {**component,
                                  "bbox": [x1 + cut_x, y1 + cut_y, x2 - cut_x, y2 - cut_y]}
    return out


def measure(preds, cases, per_type, default=0.0):
    pairs = [(case["case"], inset_boxes(preds[case["case"]], per_type, default),
              case["truth"], case["height"]) for case in cases]
    report = evaluate_cases(pairs, text_mode="designator_name",
                            anonymous_components="by_position")
    return report["metrics"]["component"], report["weighted_total"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path,
                        default=Path(r"D:/pcb_unet_v2/reports/bbox_inset_calibration.json"))
    args = parser.parse_args()

    cases = []
    for folder in sorted(GT_DIR.iterdir()):
        if folder.is_dir():
            case = folder.name.split("-")[0]
            cases.append({"case": case,
                          "truth": json.loads(next(folder.glob("*_target.json")).read_text(encoding="utf-8")),
                          "height": image_height(ROOT / f"data/standardized/{case}/{case}.png")})
    preds = load(cases, args.predictions)

    # measured per-type median deviation on key-matched pairs
    deviations = {}
    for case in cases:
        truth, pred = case["truth"]["components"], preds[case["case"]]["components"]
        for key, reference in truth.items():
            candidate = pred.get(key)
            if not candidate:
                continue
            kind = normalise_type(reference.get("type"))
            ref_box = [float(v) for v in reference["bbox"]]
            box = [float(v) for v in candidate["bbox"]]
            deviations.setdefault(kind, []).append(
                statistics.median(abs(box[i] - ref_box[i]) for i in range(4)))
    per_type_inset = {kind: round(statistics.median(values), 1)
                      for kind, values in deviations.items() if len(values) >= 2}
    print("measured median per-side deviation (px) -> candidate inset:")
    for kind, value in sorted(per_type_inset.items()):
        print(f"  {kind:<14}{value:>7.1f}   n={len(deviations[kind])}")

    report = {"predictions": str(args.predictions),
              "measured_median_deviation_px": per_type_inset,
              "uniform": {}, "per_type": {}}
    print(f"\n{'inset/side':>12}{'ComponentF1':>13}{'P':>8}{'R':>8}{'weighted':>10}")
    for cut in (0, 4, 8, 10, 12, 15, 20):
        metric, weighted = measure(preds, cases, {}, float(cut))
        report["uniform"][cut] = {"component": metric, "weighted_total": weighted}
        print(f"{cut:>12}{metric['f1']:>13.4f}{metric['precision']:>8.4f}"
              f"{metric['recall']:>8.4f}{weighted:>10.4f}")

    metric, weighted = measure(preds, cases, per_type_inset)
    report["per_type"]["applied"] = per_type_inset
    report["per_type"]["component"] = metric
    report["per_type"]["weighted_total"] = weighted
    print(f"\nper-type inset {per_type_inset}")
    print(f"  ComponentF1 {metric['f1']:.4f}  P {metric['precision']:.4f}  R {metric['recall']:.4f}"
          f"  weighted {weighted:.4f}")

    best_cut = max(report["uniform"], key=lambda c: report["uniform"][c]["component"]["f1"])
    print(f"\nbest uniform inset: {best_cut} px/side "
          f"(ComponentF1 {report['uniform'][best_cut]['component']['f1']:.4f}, "
          f"weighted {report['uniform'][best_cut]['weighted_total']:.4f})")
    report["best_uniform_inset_px"] = best_cut

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("written:", args.output)


if __name__ == "__main__":
    main()
