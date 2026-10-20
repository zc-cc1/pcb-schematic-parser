"""Crop the gap between two fragments so the drawing can be inspected directly."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, r"C:/Users/Pc/Desktop/新建文件夹/pcb_parser_competition/scripts")
from platform_metrics import _members, _rails, _vertex_key  # noqa: E402

GT = Path(r"C:/Users/Pc/Desktop/新建文件夹/赛题六公开数据集/赛题六公开数据集/open_GT/open_GT")
IMAGES = Path(r"D:/pcb_open_gt_in")
PRED = Path(r"D:/pcb_open_gt_v32best/final_results")
OUT = Path(r"C:/Users/Pc/Desktop/新建文件夹/gap_crop_case13.png")


def ends(net):
    out = []
    for edge in (net.get("edges") or {}).values():
        if len(edge) >= 2:
            out.append((float(edge[0]["x"]), float(edge[0]["y"])))
            out.append((float(edge[-1]["x"]), float(edge[-1]["y"])))
    return out


def verts(net, components, rail):
    keys = set()
    for token in _members(net):
        c, p = _vertex_key(token, components, rail)
        keys.add(f"{c}.{p}" if p else c)
    return keys


truth = json.loads((GT / "13/13.json").read_text(encoding="utf-8"))
components = truth.get("components") or {}
rail = _rails(components)
wanted = verts((truth.get("nets") or {})["net_7"], components, rail)
pred = json.loads((PRED / "13_Pred.json").read_text(encoding="utf-8"))
p_components = pred.get("components") or {}
p_rail = _rails(p_components)
pieces = []
for net in (pred.get("nets") or {}).values():
    common = verts(net, p_components, p_rail) & wanted
    if common:
        pieces.append((sorted(common), ends(net)))
pieces.sort(key=lambda item: -len(item[1]))

best = None
for i in range(len(pieces)):
    for j in range(i + 1, len(pieces)):
        a, b = pieces[i][1], pieces[j][1]
        if not a or not b:
            continue
        distance, p, q = min((math.dist(u, v), u, v) for u in a for v in b)
        if 30 <= distance <= 70 and (best is None or distance < best[0]):
            best = (distance, p, q, pieces[i][0], pieces[j][0])

distance, p, q, left, right = best
with Image.open(IMAGES / "13.png") as handle:
    image = handle.convert("RGB")
    height = image.height
    ty1, ty2 = height - p[1], height - q[1]
    cx, cy = (p[0] + q[0]) / 2, (ty1 + ty2) / 2
    size = 90
    box = (int(cx - size), int(cy - size), int(cx + size), int(cy + size))
    crop = image.crop(box).resize(((box[2] - box[0]) * 5, (box[3] - box[1]) * 5), Image.NEAREST)
    crop.save(OUT)
print(f"缺口 {distance:.0f}px  端点 {left[:2]} {p} -> {right[:2]} {q}")
print(f"裁剪区域 {box}（放大 5 倍）已保存 {OUT}")
