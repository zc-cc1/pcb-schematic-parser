"""Measure reference detection-box sizes so page scale can be estimated at runtime.

A page drawn at 3x the usual symbol size gets boxes three times larger from the same
detector (probe_scale_sensitivity.py: median capacitor width 58.5 -> 176.7 between
1x and 3x). That means the detector's own output carries the page's scale, so the
canonical symbol table can be scaled per page instead of being applied as absolute
pixels - which is what collapsed to bbox accuracy 0.00 on the hidden pages 12/13.

This script writes those reference medians into the geometry config:

    configs/bbox_convention_v1.json -> page_scale.reference_detected_widths

Read-only on predictions; no GT needed for the reference itself.
"""
import argparse
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from evaluate_official_component import pred_class_name  # noqa: E402

# Classes whose drawn size is essentially constant, so their detected width is a
# usable ruler for the page. ICs, switches, opamps are deliberately absent.
RULERS = ("capacitor", "resistor", "ground", "voltage_source", "inductor", "led", "diode")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", type=Path, default=ROOT / "runs/eval_v3_fused_v2thr.json")
    ap.add_argument("--geometry", type=Path, default=ROOT / "configs/bbox_convention_v1.json")
    ap.add_argument("--min-samples", type=int, default=15)
    args = ap.parse_args()

    data = json.loads(args.pred.read_text(encoding="utf-8"))
    widths = defaultdict(list)
    heights = defaultdict(list)
    for case in data.get("cases") or []:
        for p in case.get("predictions") or []:
            name = pred_class_name(p["class_name"])
            if name not in RULERS:
                continue
            bbox = [float(v) for v in p["bbox"]]
            widths[name].append(bbox[2] - bbox[0])
            heights[name].append(bbox[3] - bbox[1])

    reference = {}
    print(f"{'class':>16} {'n':>6} {'median W':>9} {'median H':>9}")
    for name in RULERS:
        if len(widths[name]) < args.min_samples:
            continue
        reference[name] = {"w": round(st.median(widths[name]), 2),
                           "h": round(st.median(heights[name]), 2),
                           "n": len(widths[name])}
        print(f"{name:>16} {reference[name]['n']:>6} {reference[name]['w']:>9} "
              f"{reference[name]['h']:>9}")
    if not reference:
        raise SystemExit("no ruler classes had enough detections")

    config = json.loads(args.geometry.read_text(encoding="utf-8"))
    config["page_scale"] = {
        "status": "reference medians measured on the public validation predictions",
        "measured_from": str(args.pred.relative_to(ROOT)),
        "ruler_classes": sorted(reference),
        "reference_detected_widths": {k: v["w"] for k, v in reference.items()},
        "note": ("scale = median(detected width / reference width) over the ruler classes, "
                 "clamped; the canonical symbol sizes are multiplied by it before export"),
    }
    args.geometry.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote page_scale.reference_detected_widths ({len(reference)} classes) "
          f"to {args.geometry.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
