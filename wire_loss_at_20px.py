"""Decompose the remaining wire loss with the real protocol (20 px tolerance).

fn splits into "no segment at all within 20 px" versus "segment exists but its net got
assigned to another reference net".  fp is simply what we draw that nothing matches.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, r"C:/Users/Pc/Desktop/新建文件夹/pcb_parser_competition/scripts")
from platform_metrics import _segments  # noqa: E402

GT = Path(r"C:/Users/Pc/Desktop/新建文件夹/赛题六公开数据集/赛题六公开数据集/open_GT/open_GT")
IMAGES = Path(r"D:/pcb_open_gt_in")
TOLERANCE = 20.0


def close(a, b, tol=TOLERANCE):
    direct = math.dist(a[0], b[0]) <= tol and math.dist(a[-1], b[-1]) <= tol
    reverse = math.dist(a[0], b[-1]) <= tol and math.dist(a[-1], b[0]) <= tol
    return direct or reverse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, default=Path(r"D:/pcb_open_gt_v37dots/final_results"))
    args = parser.parse_args()

    buckets = Counter()
    missing_ink = Counter()
    gt_total = our_total = 0

    for folder in sorted(GT.iterdir(), key=lambda p: int(p.name)):
        case = folder.name
        truth = json.loads(next(folder.glob("*.json")).read_text(encoding="utf-8"))
        path = args.predictions / f"{case}_Pred.json"
        pred = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {"nets": {}}
        with Image.open(IMAGES / f"{case}.png") as handle:
            rgb = np.array(handle.convert("RGB"))
            height = rgb.shape[0]

        gt_nets = [_segments(net) for net in (truth.get("nets") or {}).values()]
        our_nets = [_segments(net) for net in (pred.get("nets") or {}).values()]
        gt_flat = [seg for net in gt_nets for seg in net]
        our_flat = [seg for net in our_nets for seg in net]
        gt_total += len(gt_flat)
        our_total += len(our_flat)

        # which of our segments land inside which reference net (geometry only)
        for net in gt_nets:
            for seg in net:
                if any(close(seg, other) for other in our_flat):
                    continue
                buckets["真没画（20px 内没有任何我们的线）"] += 1
                samples = []
                steps = max(3, int(math.dist(seg[0], seg[-1])) + 1)
                for t in np.linspace(0, 1, steps):
                    x = int(round(seg[0][0] + t * (seg[-1][0] - seg[0][0])))
                    y = int(round(height - (seg[0][1] + t * (seg[-1][1] - seg[0][1]))))
                    if 0 <= y < rgb.shape[0] and 0 <= x < rgb.shape[1]:
                        samples.append(float(rgb[y, x].mean()))
                if samples:
                    values = np.asarray(samples)
                    missing_ink["深色线" if (values < 110).mean() >= .5 else "浅色/空白"] += 1

        for seg in our_flat:
            if not any(close(seg, other) for other in gt_flat):
                buckets["多画的（20px 内没有真值线）"] += 1

    print(f"真值线段 {gt_total}，我们画了 {our_total}")
    for key, value in buckets.items():
        print(f"  {key}: {value}")
    print(f"  缺失线段处的墨迹: {dict(missing_ink)}")


if __name__ == "__main__":
    main()
