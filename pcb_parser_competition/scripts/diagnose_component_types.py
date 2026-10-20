"""Diagnose component-type disagreements that cost component matches.

For every reference component that did not match, find the nearest exported
component within the 20 px centre tolerance and record the type pair.  A frequent
pair is a missing alias rather than a detection error.
"""
import collections
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pcb_parser.contest_metrics import normalise_type  # noqa: E402

GT = Path(r"C:/Users/Pc/Desktop/新建文件夹/赛题六公开数据集/赛题六公开数据集/10_GTcase")
BASE = Path(r"D:/pcb_unet_v2/golden_eval")


def centre(box):
    return (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0


def main():
    mismatches = collections.Counter()
    missed_far = 0
    for folder in sorted(GT.iterdir()):
        if not folder.is_dir():
            continue
        case = folder.name.split("-")[0]
        submission = BASE / case / "wire_graph" / "partial_submission.json"
        if not submission.exists():
            continue
        truth = json.loads(next(folder.glob("*_target.json")).read_text(encoding="utf-8"))
        pred = json.loads(submission.read_text(encoding="utf-8"))
        predicted = list((pred.get("components") or {}).items())
        for truth_type in (truth.get("components") or {}).values():
            truth_centre = centre(truth_type["bbox"])
            nearest = None
            for key, candidate in predicted:
                distance = math.dist(centre(candidate["bbox"]), truth_centre)
                if nearest is None or distance < nearest[0]:
                    nearest = (distance, key, candidate["type"])
            if nearest is None or nearest[0] > 20:
                missed_far += 1
                continue
            if normalise_type(nearest[2]) != normalise_type(truth_type["type"]):
                mismatches[(str(truth_type["type"]), str(nearest[2]))] += 1
    print("reference components with no exported component within 20 px:", missed_far)
    print("reference components with a nearby export but a different type:")
    for (expected, actual), count in mismatches.most_common(20):
        print(f"  reference {expected:<14} -> ours {actual:<14} {count}")
    print("total type mismatches:", sum(mismatches.values()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
