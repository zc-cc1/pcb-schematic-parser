"""Show how the reference annotates one net: every edge is a straight two-point piece,
cut at pin/junction positions.  Endpoints are numbered in drawing order.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, r"C:/Users/Pc/Desktop/新建文件夹/pcb_parser_competition/scripts")
from platform_metrics import _members, _segments  # noqa: E402

GT = Path(r"C:/Users/Pc/Desktop/新建文件夹/赛题六公开数据集/赛题六公开数据集/open_GT/open_GT")
IMAGES = Path(r"D:/pcb_open_gt_in")
CASE, NET = "13", "net_7"
OUT = Path(r"C:/Users/Pc/Desktop/新建文件夹/gt_wire_annotation.png")
COLOURS = [(214, 39, 40), (31, 119, 180), (44, 160, 44), (255, 127, 14), (148, 103, 189),
           (140, 86, 75), (227, 119, 194), (188, 189, 34), (23, 190, 207), (174, 199, 232)]

truth = json.loads((GT / CASE / f"{CASE}.json").read_text(encoding="utf-8"))
net = (truth.get("nets") or {})[NET]
edges = _segments(net)
components = truth.get("components") or {}

with Image.open(IMAGES / f"{CASE}.png") as handle:
    image = handle.convert("RGB")
    width, height = image.size


def ty(point):
    return (point[0], height - point[1])


xs, ys = [], []
for seg in edges:
    for point in (seg[0], seg[-1]):
        x, y = ty(point)
        xs.append(x); ys.append(y)
margin = 30
box = (max(0, int(min(xs)) - margin), max(0, int(min(ys)) - margin),
       min(width, int(max(xs)) + margin), min(height, int(max(ys)) + margin))

canvas = image.crop(box)
scale = 1.6
canvas = canvas.resize((int(canvas.width * scale), int(canvas.height * scale)), Image.LANCZOS)
draw = ImageDraw.Draw(canvas)
try:
    font = ImageFont.truetype(r"C:\Windows\Fonts\segoeui.ttf", 22)
except OSError:
    font = ImageFont.load_default()

def conv(point):
    x, y = ty(point)
    return ((x - box[0]) * scale, (y - box[1]) * scale)

for index, seg in enumerate(edges):
    colour = COLOURS[index % len(COLOURS)]
    pts = [conv(seg[0]), conv(seg[-1])]
    draw.line(pts, fill=colour, width=3)
    for end in pts:
        draw.ellipse([end[0] - 5, end[1] - 5, end[0] + 5, end[1] + 5], fill=colour, outline=(255, 255, 255))
    mid = ((pts[0][0] + pts[1][0]) / 2, (pts[0][1] + pts[1][1]) / 2)
    draw.text((mid[0] + 6, mid[1] - 24), str(index + 1), fill=colour, font=font)

summary = (f"{CASE} {NET}: {len(edges)} 条线段, {len(_members(net))} 个引脚引用\n"
           f"每条线段 = 两个端点; 端点落在引脚/交叉点上")
draw.rectangle([8, 8, 8 + 12 * len(summary.split(chr(10))[0]) + 260, 78], fill=(255, 255, 255))
draw.multiline_text((16, 14), summary, fill=(0, 0, 0), font=font)
canvas.save(OUT)
print(f"已保存 {OUT}  ({canvas.width}x{canvas.height})")

print("\n前 6 条线段的原始标注：")
for index, seg in enumerate(edges[:6]):
    (x1, y1), (x2, y2) = seg[0], seg[-1]
    length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** .5
    print(f"  #{index+1}: ({x1:.0f}, {y1:.0f}) -> ({x2:.0f}, {y2:.0f})   长 {length:.0f}px   点数 {len(seg)}")
