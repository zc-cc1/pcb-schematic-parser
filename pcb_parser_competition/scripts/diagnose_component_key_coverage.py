"""Why are reference component keys missing from the submission?

Pin scoring (PDF p8) first matches the component key, so a key we never emit costs
every pin under it.  For each reference component this reports whether we emitted that
key, and if not, whether we at least emitted a component at the same place under a
different key (a designator failure) or nothing at all (a detection failure).
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
NEAR = 20.0


def centre(box):
    return (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0


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
        pred_components = pred.get("components") or {}
        counts = Counter()
        for key, component in (truth.get("components") or {}).items():
            if key in pred_components:
                counts["key present"] += 1
                continue
            target = centre(component["bbox"])
            near = None
            for other_key, other in pred_components.items():
                distance = math.dist(centre(other["bbox"]), target)
                if near is None or distance < near[0]:
                    near = (distance, other_key)
            if near and near[0] <= NEAR:
                counts["detected but wrong key"] += 1
            else:
                counts["not detected"] += 1
        for label, value in counts.items():
            totals[label] += value
        per_case.append((case, counts))
    order = ("key present", "detected but wrong key", "not detected")
    header = f"  {'case':<8}" + "".join(f"{label:>24}" for label in order)
    print(header)
    for case, counts in per_case:
        row = f"  {case:<8}" + "".join(f"{counts[label]:>24}" for label in order)
        print(row)
    print()
    for label in order:
        print(f"  {label:<24}{totals[label]:>5}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
