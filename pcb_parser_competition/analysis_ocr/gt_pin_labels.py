"""Are GT IC pin numbers decimal and printed? BGA pages get renumbered balls."""
import glob
import json
import sys
from pathlib import Path

ROOT = Path(r"C:\Users\Pc\Desktop\新建文件夹\赛题六公开数据集\赛题六公开数据集\200_train_cases")
CASES = ["0001", "0002", "0018", "0080", "0102"]


def main():
    for case in CASES:
        f = glob.glob(str(ROOT / case / "*_target_named.json"))[0]
        gt = json.load(open(f, encoding="utf-8"))
        ics = [(k, v) for k, v in gt["components"].items() if str(v.get("type")) in ("box", "ic")]
        if not ics:
            continue
        biggest = max(ics, key=lambda kv: len(gt["pins"].get(kv[0]) or {}))
        key, comp = biggest
        pins = gt["pins"].get(key) or {}
        labels = list(pins)
        digits = [int(p[4:]) for p in labels if p[4:].isdigit()]
        alnum = [p for p in labels if p[4:] and not p[4:].isdigit()]
        print(f"{case} {key} {comp.get('Name')!r}: {len(pins)} GT pins, "
              f"decimal {len(digits)}"
              + (f" range {min(digits)}-{max(digits)}" if digits else "")
              + f", non-decimal {len(alnum)} {alnum[:5]}")
        names = [p.get("pinname") for p in pins.values() if p.get("pinname")]
        print(f"    pinname non-empty {len(names)}; sample {names[:5]}")


if __name__ == "__main__":
    sys.exit(main())
