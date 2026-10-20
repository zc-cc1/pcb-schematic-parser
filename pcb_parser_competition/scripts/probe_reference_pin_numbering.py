"""Probe the reference pin-numbering direction on the public Golden cases.

The auto-numbering rule (R15) has to choose a direction.  This measures the one the
reference files actually use, so the rule is calibrated rather than assumed.
"""
import json
import re
import sys
from pathlib import Path

GT = Path(r"C:/Users/Pc/Desktop/新建文件夹/赛题六公开数据集/赛题六公开数据集/10_GTcase")


def main():
    vertical_lower_first = 0
    horizontal_left_first = 0
    vertical_upper_first = 0
    horizontal_right_first = 0
    total = 0
    for folder in sorted(GT.iterdir()):
        if not folder.is_dir():
            continue
        annotations = list(folder.glob("*_target.json"))
        if not annotations:
            continue
        truth = json.loads(annotations[0].read_text(encoding="utf-8"))
        for component, pins in (truth.get("pins") or {}).items():
            if len(pins) != 2:
                continue
            keys = sorted(pins, key=lambda k: int(k.replace("pin_", "")))
            if keys != ["pin_1", "pin_2"]:
                continue
            first, second = pins["pin_1"]["point"], pins["pin_2"]["point"]
            total += 1
            if abs(first["y"] - second["y"]) > abs(first["x"] - second["x"]):
                if first["y"] > second["y"]:
                    vertical_lower_first += 1
                else:
                    vertical_upper_first += 1
            else:
                if first["x"] < second["x"]:
                    horizontal_left_first += 1
                else:
                    horizontal_right_first += 1
    print(f"two-pin components: {total}")
    print(f"  vertical  : pin_1 lower {vertical_lower_first}, pin_1 upper {vertical_upper_first}")
    print(f"  horizontal: pin_1 left  {horizontal_left_first}, pin_1 right {horizontal_right_first}")
    if vertical_lower_first + horizontal_left_first >= vertical_upper_first + horizontal_right_first:
        print("=> 参考更倾向从下/左开始编号")
    else:
        print("=> 参考更倾向上/右开始编号")
    return 0


if __name__ == "__main__":
    sys.exit(main())
