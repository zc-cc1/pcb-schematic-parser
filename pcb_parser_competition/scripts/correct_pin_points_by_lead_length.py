"""Correct each pin point to "body edge + 10 px" using that pin's own lead length.

The reference puts the pin point exactly 10 px outside the body box.  Our point sits at
the outer end of the drawn lead, so the correction for one pin is (lead length - 10)
along the lead, measured on that pin rather than estimated from a whole component box.
This keeps per-pin accuracy and avoids the 8 px error of a fitted box.
"""
import json
import math
import statistics
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from evaluate_golden_cases import GT_DIR  # noqa: E402
from recover_body_box_by_lead_walk import body_edge  # noqa: E402

BASE = Path(r"D:/pcb_unet_v2/golden_eval")
STUB = 10.0


def main():
    total = hits_now = extra = 0
    leads = []
    for folder in sorted(GT_DIR.iterdir()):
        if not folder.is_dir():
            continue
        case = folder.name.split("-")[0]
        submission_path = BASE / case / "wire_graph" / "partial_submission.json"
        if not submission_path.exists():
            continue
        truth = json.loads(next(folder.glob("*_target.json")).read_text(encoding="utf-8"))
        pred = json.loads(submission_path.read_text(encoding="utf-8"))
        with Image.open(next(folder.glob("*.png"))) as image:
            gray = np.asarray(image.convert("L"))
            height = gray.shape[0]
        ink = gray < 150
        pred_pins = pred.get("pins") or {}
        pred_components = pred.get("components") or {}
        for component, pins in (truth.get("pins") or {}).items():
            ours = pred_pins.get(component)
            det = (pred_components.get(component) or {}).get("bbox")
            if not ours or not det:
                continue
            cx = (det[0] + det[2]) / 2.0
            cy = height - (det[1] + det[3]) / 2.0
            for key, pin in pins.items():
                candidate = ours.get(key)
                if candidate is None:
                    continue
                total += 1
                reference = (pin["point"]["x"], pin["point"]["y"])
                px, py = candidate["point"]["x"], candidate["point"]["y"]
                if math.hypot(px - reference[0], py - reference[1]) <= 5.0:
                    hits_now += 1
                    continue
                tip = (px, height - py)
                edge = body_edge(ink, tip, (cx, cy))
                if not edge:
                    continue
                length = math.hypot(edge[0] - tip[0], edge[1] - tip[1])
                reach = math.hypot(cx - tip[0], cy - tip[1])
                if reach < 1e-6:
                    continue
                ux, uy = (cx - tip[0]) / reach, (cy - tip[1]) / reach
                nx = tip[0] + ux * (length - STUB)
                ny = tip[1] + uy * (length - STUB)
                leads.append(length)
                if math.hypot(nx - reference[0], (height - ny) - reference[1]) <= 5.0:
                    extra += 1
    print(f"evaluable pins: {total}")
    print(f"  matched today                      : {hits_now}")
    print(f"  additionally fixed by lead-length  : {extra}")
    print(f"  total                              : {hits_now + extra}")
    if leads:
        leads.sort()
        print()
        print(f"measured lead lengths: n={len(leads)} median {statistics.median(leads):.1f} "
              f"(p25 {leads[len(leads)//4]:.1f}, p75 {leads[3*len(leads)//4]:.1f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
