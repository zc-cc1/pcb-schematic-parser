"""Pin failure attribution under the official rule (PDF p8).

Official rule: a pin is correct iff its component KEY matches, its pin NUMBER
matches, its pinname matches, and the two points are within 5 px.  The component
itself does NOT have to pass type/Name/bbox for its pins to be scored.

Every reference pin is placed in exactly one bucket, so the per-bucket counts are
directly the score loss and can be attacked in size order.
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
TOLERANCE = 5.0


def main():
    totals = Counter()
    per_case = []
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
        counts = Counter()
        for component, pins in (truth.get("pins") or {}).items():
            ours = pred_pins.get(component)
            for key, pin in pins.items():
                totals["reference pins"] += 1
                counts["reference pins"] += 1
                if ours is None:
                    counts["component key absent"] += 1
                    continue
                candidate = ours.get(key)
                if candidate is None:
                    counts["pin number absent"] += 1
                    continue
                distance = math.hypot(candidate["point"]["x"] - pin["point"]["x"],
                                      candidate["point"]["y"] - pin["point"]["y"])
                if distance > TOLERANCE:
                    counts["point too far"] += 1
                    continue
                if str(pin.get("pinname")) != str(candidate.get("pinname")):
                    counts["pinname differs"] += 1
                    continue
                counts["matched"] += 1
        for label, value in counts.items():
            totals[label] += value
        per_case.append((case, counts["matched"], counts["reference pins"]))
    order = ("matched", "point too far", "pin number absent", "component key absent",
             "pinname differs")
    print(f"{'reference pins':<26}{totals['reference pins']:>6}")
    for label in order:
        count = totals[label]
        share = 100 * count / max(1, totals["reference pins"])
        print(f"  {label:<24}{count:>6}  {share:5.1f}%")
    print()
    print(f"  {'case':<8}{'matched':>9}{'reference':>11}")
    for case, matched, reference in per_case:
        print(f"  {case:<8}{matched:>9}{reference:>11}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
