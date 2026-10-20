"""Can the reference's generated designators be reproduced from geometry?

Every case is checked for two things:
  1) are the keys of one prefix a gapless 1..n sequence (i.e. synthesised, not printed)?
  2) does any simple geometric ordering reproduce the index order?

If some ordering does, unnamed components can be given reproducible keys, which per the
official rule (p8: pins match on component key first) unlocks their pins and nets.

Run:  python scripts/probe_generated_designators.py
"""
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from evaluate_golden_cases import GT_DIR  # noqa: E402

ORDERS = {
    "top->bottom, left->right": lambda c: (round(c[1], 1), round(c[0], 1)),
    "left->right, top->bottom": lambda c: (round(c[0], 1), round(c[1], 1)),
    "bottom->top, left->right": lambda c: (-round(c[1], 1), round(c[0], 1)),
    "right->left, top->bottom": lambda c: (-round(c[0], 1), round(c[1], 1)),
}


def centre(box):
    return (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0


def main():
    patterns = Counter()
    score = Counter()
    tested = 0
    for folder in sorted(GT_DIR.iterdir()):
        if not folder.is_dir():
            continue
        case = folder.name.split("-")[0]
        truth = json.loads(next(folder.glob("*_target.json")).read_text(encoding="utf-8"))
        groups = defaultdict(list)
        for key, component in (truth.get("components") or {}).items():
            match = re.fullmatch(r"([A-Za-z]+)(\d+)", key)
            if not match:
                patterns["internal ID (not a designator)"] += 1
                continue
            groups[match.group(1).upper()].append((int(match.group(2)), centre(component["bbox"])))
        for prefix, items in groups.items():
            numbers = sorted(number for number, _ in items)
            gapless = numbers == list(range(1, len(numbers) + 1))
            patterns["gapless 1..n (synthesised)" if gapless else "has gaps (printed)"] += 1
            if not gapless or len(items) < 2:
                continue
            tested += 1
            for label, key_fn in ORDERS.items():
                ordered = sorted(items, key=lambda item: key_fn(item[1]))
                if [number for number, _ in ordered] == list(range(1, len(ordered) + 1)):
                    score[label] += 1
    total = sum(patterns.values())
    print("reference component keys:")
    for label, count in patterns.most_common():
        print(f"  {label:<34}{count:>5}  {100 * count / max(1, total):5.1f}%")
    print()
    print(f"prefix groups that are gapless and testable: {tested}")
    print("geometric orderings that reproduce the index order:")
    for label in ORDERS:
        count = score[label]
        print(f"  {label:<28}{count:>4} / {tested}"
              f"  {100 * count / max(1, tested):5.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
