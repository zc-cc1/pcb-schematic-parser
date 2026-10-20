"""Are the reference's cut points ('nodes') visible on the page as distinct glyphs?

Samples the image colour at the endpoints of reference wire segments (which are exactly
the electrical nodes) and compares with the colour along the wire away from any endpoint.
"""
from __future__ import annotations

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

bucket_nodes = Counter()
bucket_mid = Counter()


def classify(px):
    r, g, b = float(px[0]), float(px[1]), float(px[2])
    brightness = (r + g + b) / 3
    if brightness > 215 and max(r, g, b) - min(r, g, b) < 30:
        return "背景（浅色）"
    if r > 140 and g < 110 and b < 110:
        return "红色"
    if g > 110 and r < 130:
        return "绿色"
    if brightness < 110:
        return "深色/黑"
    return "其他"


for case in ("13", "6", "12"):
    truth = json.loads((GT / case / f"{case}.json").read_text(encoding="utf-8"))
    with Image.open(IMAGES / f"{case}.png") as handle:
        rgb = np.array(handle.convert("RGB"))
        height = rgb.shape[0]

    endpoint_hits = Counter()
    mid_hits = Counter()
    for net in (truth.get("nets") or {}).values():
        for seg in _segments(net):
            a, b = seg[0], seg[-1]
            for point in (a, b):
                x = int(round(point[0]))
                y = int(round(height - point[1]))
                if 0 <= y < rgb.shape[0] and 0 <= x < rgb.shape[1]:
                    endpoint_hits[classify(rgb[y, x])] += 1
            for t in (0.35, 0.5, 0.65):
                x = int(round(a[0] + t * (b[0] - a[0])))
                y = int(round(height - (a[1] + t * (b[1] - a[1]))))
                if 0 <= y < rgb.shape[0] and 0 <= x < rgb.shape[1]:
                    mid_hits[classify(rgb[y, x])] += 1
    print(f"=== case {case} ===")
    print("  线段端点（=电节点）颜色分布:", dict(endpoint_hits))
    print("  线段中段颜色分布        :", dict(mid_hits))
    bucket_nodes.update(endpoint_hits)
    bucket_mid.update(mid_hits)

print("\n三页合计：")
print("  端点:", dict(bucket_nodes))
print("  中段:", dict(bucket_mid))
