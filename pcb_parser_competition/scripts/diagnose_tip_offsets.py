"""Is the 5 px pin failure a numbering error or a tip-definition error?

The PDF (p26) says a pin tip sits on the OUTER bbox edge, and base connects to the
inner body box.  Two failure modes look identical in a plain distance, so each offset
is decomposed against its component box:

  tangential  - along the pin row/column.  About one pin pitch means the number was
                attached to the wrong physical pin (a numbering error).
  radial      - along the pin axis.  A few pixels in one consistent direction means the
                tip was taken at the body edge instead of the outer edge (a definition
                error).

Run:  python scripts/diagnose_tip_offsets.py
"""
import json
import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from evaluate_golden_cases import GT_DIR  # noqa: E402

BASE = Path(r"D:/pcb_unet_v2/golden_eval")


def main():
    tangential, radial = [], []
    inward = outward = 0
    for folder in sorted(GT_DIR.iterdir()):
        if not folder.is_dir():
            continue
        case = folder.name.split("-")[0]
        submission_path = BASE / case / "wire_graph" / "partial_submission.json"
        if not submission_path.exists():
            continue
        truth = json.loads(next(folder.glob("*_target.json")).read_text(encoding="utf-8"))
        pred = json.loads(submission_path.read_text(encoding="utf-8"))
        pred_pins = pred.get("pins") or {}
        for component, pins in (truth.get("pins") or {}).items():
            ours = pred_pins.get(component)
            box = (truth.get("components", {}).get(component) or {}).get("bbox")
            if not ours or not box:
                continue
            cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
            for key, pin in pins.items():
                candidate = ours.get(key)
                if candidate is None:
                    continue
                dx = candidate["point"]["x"] - pin["point"]["x"]
                dy = candidate["point"]["y"] - pin["point"]["y"]
                if math.hypot(dx, dy) <= 5.0:
                    continue
                # pin axis: from the box centre out to the reference tip
                ax, ay = pin["point"]["x"] - cx, pin["point"]["y"] - cy
                norm = math.hypot(ax, ay)
                if norm < 1e-6:
                    continue
                ax, ay = ax / norm, ay / norm
                along = dx * ax + dy * ay                       # radial component
                across = abs(dx * (-ay) + dy * ax)              # tangential component
                tangential.append(across)
                radial.append(abs(along))
                d_ref = math.hypot(pin["point"]["x"] - cx, pin["point"]["y"] - cy)
                d_our = math.hypot(candidate["point"]["x"] - cx, candidate["point"]["y"] - cy)
                if d_our < d_ref:
                    inward += 1
                else:
                    outward += 1
    if not tangential:
        print("no failing pins with a usable box")
        return 0
    tangential.sort()
    radial.sort()
    median = lambda values: values[len(values) // 2]
    print(f"pins failing the 5 px test with a component box: {len(tangential)}")
    print(f"  tangential |offset| median {median(tangential):.1f} px "
          f"(p25 {tangential[len(tangential)//4]:.1f}, p75 {tangential[3*len(tangential)//4]:.1f})")
    print(f"  radial     |offset| median {median(radial):.1f} px "
          f"(p25 {radial[len(radial)//4]:.1f}, p75 {radial[3*len(radial)//4]:.1f})")
    print()
    mostly_tangential = sum(1 for t, r in zip(tangential, radial) if t > r)
    print(f"  offset dominated by the tangential axis : {mostly_tangential} "
          f"({100 * mostly_tangential / len(tangential):.1f}%)  -> numbering / wrong pin")
    print(f"  offset dominated by the radial axis     : {len(tangential) - mostly_tangential} "
          f"({100 * (len(tangential) - mostly_tangential) / len(tangential):.1f}%)  -> tip definition")
    print()
    print(f"  our point closer to the box centre than the reference : {inward}")
    print(f"  our point further out                                 : {outward}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
