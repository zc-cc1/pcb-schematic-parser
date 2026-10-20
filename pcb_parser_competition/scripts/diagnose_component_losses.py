"""Break the component false negatives down by reference type.

Answers "which reference class are we losing" rather than "how many".
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
    total = collections.Counter()
    matched = collections.Counter()
    nearest_type = collections.Counter()
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
        for component in (truth.get("components") or {}).values():
            kind = str(component["type"])
            total[kind] += 1
            truth_centre = centre(component["bbox"])
            best = None
            for key, candidate in predicted:
                distance = math.dist(centre(candidate["bbox"]), truth_centre)
                if best is None or distance < best[0]:
                    best = (distance, candidate["type"])
            same = best is not None and best[0] <= 20 and \
                normalise_type(best[1]) == normalise_type(kind)
            if same:
                matched[kind] += 1
            else:
                nearest_type[(kind, str(best[1]) if best and best[0] <= 20 else "none within 20px")] += 1
    print(f"{'reference type':<16}{'total':>7}{'matched':>9}{'lost':>6}")
    for kind, count in total.most_common():
        print(f"{kind:<16}{count:>7}{matched[kind]:>9}{count - matched[kind]:>6}")
    print()
    print("lost components, by nearest exported type:")
    for (kind, actual), count in nearest_type.most_common(20):
        print(f"  {kind:<14} -> {actual:<18} {count}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
