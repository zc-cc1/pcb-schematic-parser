"""Can the collapsed label types be told apart from geometry alone?

`class_mapping` exports pin_port / testpoint / net_input all as `vbus_in`, and every
crystal as `crystal`, so those components can never match the GT type string. This
measures the GT box size of each label type to see whether a post-export rule can
recover the distinction without touching the detector.
"""
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(r"C:\Users\Pc\Desktop\新建文件夹\赛题六公开数据集\赛题六公开数据集\200_train_cases")
TYPES = ["net_input", "net_output", "net_short", "net_bidirection", "pin",
         "testpoint", "crystal_2pin", "crystal_3pin", "crystal_4pin", "crystal",
         "led", "d", "switch"]


def main():
    boxes = {t: [] for t in TYPES}
    for case in sorted(p for p in ROOT.iterdir() if p.is_dir()):
        f = next(case.glob("*_target_named.json"), None)
        if f is None:
            continue
        g = json.loads(f.read_text(encoding="utf-8"))
        for comp in (g.get("components") or {}).values():
            t = str(comp.get("type") or "")
            if t not in boxes:
                continue
            x1, y1, x2, y2 = comp["bbox"]
            boxes[t].append((abs(x2 - x1), abs(y2 - y1)))
    print(f"{'type':16s} {'n':>5} {'w med':>7} {'h med':>7} {'w p10-p90':>14} {'h p10-p90':>14} {'h>w':>6}")
    for t in TYPES:
        rows = boxes[t]
        if not rows:
            continue
        w = sorted(r[0] for r in rows)
        h = sorted(r[1] for r in rows)
        p10 = lambda s: s[len(s)//10]
        p90 = lambda s: s[min(len(s)-1, len(s)*9//10)]
        tall = sum(1 for a, b in rows if b > a)
        print(f"{t:16s} {len(rows):5d} {statistics.median(w):7.1f} {statistics.median(h):7.1f} "
              f"{p10(w):6.1f}-{p90(w):<7.1f} {p10(h):6.1f}-{p90(h):<7.1f} {tall/len(rows):6.0%}")
    # how separable is net_input (wide strip) from testpoint (square pad)?
    print("\naspect ratio (w/h) deciles:")
    for t in ("net_input", "testpoint", "pin", "crystal_2pin", "crystal_3pin", "crystal_4pin"):
        rows = boxes[t]
        if not rows:
            continue
        ratios = sorted((r[0] / r[1]) if r[1] else 0 for r in rows)
        dec = [ratios[min(len(ratios)-1, len(ratios)*k//10)] for k in range(0, 10, 2)]
        print(f"  {t:14s} " + " ".join(f"{d:5.2f}" for d in dec))


if __name__ == "__main__":
    sys.exit(main())
