"""Why is each reference pin still a false negative?

Classifies every reference pin that our export does not turn into a true positive:
  component not matched  - its component never matched, so none of its pins can match
  no such pin number     - the component matched but we exported a different pin set
  point too far          - the key exists but the tip is beyond the 5 px tolerance
  pinname differs        - the key and the point are right, only the name is wrong
  matched                - counted as a true positive

This is the decomposition the pin score needs: it says whether the remaining loss is
perception, numbering, geometry or text.
"""
import json
import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from pcb_parser.contest_metrics import match_components, PIN_TOLERANCE_PX  # noqa: E402
from evaluate_golden_cases import GT_DIR  # noqa: E402

BASE = Path(r"D:/pcb_unet_v2/golden_eval")


def main():
    totals = Counter()
    by_case = []
    for folder in sorted(GT_DIR.iterdir()):
        if not folder.is_dir():
            continue
        case = folder.name.split("-")[0]
        submission_path = BASE / case / "wire_graph" / "partial_submission.json"
        if not submission_path.exists():
            continue
        from PIL import Image
        truth = json.loads(next(folder.glob("*_target.json")).read_text(encoding="utf-8"))
        pred = json.loads(submission_path.read_text(encoding="utf-8"))
        with Image.open(next(folder.glob("*.png"))) as image:
            height = image.height
        mapping, _, _, _ = match_components(pred, truth, height)
        reverse = {v: k for k, v in mapping.items()}
        case_counts = Counter()
        for component, pins in (truth.get("pins") or {}).items():
            our_component = reverse.get(component)
            ours = (pred.get("pins") or {}).get(our_component) or {} if our_component else {}
            for key, pin in pins.items():
                totals["reference pins"] += 1
                if not our_component:
                    totals["component not matched"] += 1
                    continue
                candidate = ours.get(key)
                if candidate is None:
                    totals["no such pin number"] += 1
                    continue
                distance = math.hypot(candidate["point"]["x"] - pin["point"]["x"],
                                      candidate["point"]["y"] - pin["point"]["y"])
                if distance > PIN_TOLERANCE_PX:
                    totals["point too far"] += 1
                    continue
                expected = str(pin.get("pinname"))
                actual = str(candidate.get("pinname"))
                if expected != actual:
                    totals["pinname differs"] += 1
                    continue
                totals["matched"] += 1
                case_counts["matched"] += 1
        by_case.append((case, case_counts["matched"], len([p for pins in (truth.get("pins") or {}).values() for p in pins])))

    print(f"{'reference pins':<28}{totals['reference pins']:>6}")
    for label in ("matched", "pinname differs", "point too far", "no such pin number", "component not matched"):
        count = totals[label]
        share = 100 * count / max(1, totals["reference pins"])
        print(f"  {label:<26}{count:>6}  {share:5.1f}%")
    print()
    print(f"  {'case':<8}{'matched':>9}{'reference':>11}")
    for case, matched, reference in by_case:
        print(f"  {case:<8}{matched:>9}{reference:>11}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
