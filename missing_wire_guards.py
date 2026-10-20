"""Are the missing wires the ones that run through the guarded zones?

The mask removes a disk around every pin candidate (pin_protection_radius_px) and insets
every component body.  If most of the reference wire we fail to draw passes through those
zones, the guards - not the model - are what breaks the net.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, r"C:/Users/Pc/Desktop/新建文件夹/pcb_parser_competition/scripts")
from platform_metrics import _close, _segments  # noqa: E402

GT = Path(r"C:/Users/Pc/Desktop/新建文件夹/赛题六公开数据集/赛题六公开数据集/open_GT/open_GT")
PRED = Path(r"D:/pcb_open_gt_v32best/final_results")
PIN_RADIUS = float(json.loads(Path(r"D:/pcb5_submission/configs/wire_graph_v1.json").read_text(encoding="utf-8"))["pin_protection_radius_px"])

categories = {"穿过引脚保护圈": 0, "穿过元件本体": 0, "两样都不占": 0}
total = 0


def point_segment_distance(point, a, b):
    ax, ay = a
    bx, by = b
    vx, vy = bx - ax, by - ay
    norm = vx * vx + vy * vy
    if norm < 1e-9:
        return math.dist(point, a)
    t = max(0.0, min(1.0, ((point[0] - ax) * vx + (point[1] - ay) * vy) / norm))
    return math.dist(point, (ax + t * vx, ay + t * vy))


for folder in sorted(GT.iterdir(), key=lambda p: int(p.name)):
    case = folder.name
    truth = json.loads(next(folder.glob("*.json")).read_text(encoding="utf-8"))
    path = PRED / f"{case}_Pred.json"
    pred = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {"components": {}, "pins": {}, "nets": {}}
    our_edges = [seg for net in (pred.get("nets") or {}).values() for seg in _segments(net)]
    pins = []
    for group in (pred.get("pins") or {}).values():
        for pin in (group.values() if isinstance(group, dict) else group):
            point = (pin or {}).get("point") or {}
            if "x" in point and "y" in point:
                pins.append((float(point["x"]), float(point["y"])))
    bodies = [c["bbox"] for c in (pred.get("components") or {}).values() if c.get("bbox")]

    for net in (truth.get("nets") or {}).values():
        for seg in _segments(net):
            if any(_close(seg, other) for other in our_edges):
                continue
            total += 1
            a, b = seg[0], seg[-1]
            length = math.dist(a, b)
            steps = max(3, int(length) + 1)
            near_pin = inside_body = False
            for t in np.linspace(0, 1, steps):
                x = a[0] + t * (b[0] - a[0])
                y = a[1] + t * (b[1] - a[1])
                if any(math.dist((x, y), pin) <= PIN_RADIUS for pin in pins):
                    near_pin = True
                if any(box[0] <= x <= box[2] and box[1] <= y <= box[3] for box in bodies):
                    inside_body = True
            if inside_body:
                categories["穿过元件本体"] += 1
            elif near_pin:
                categories["穿过引脚保护圈"] += 1
            else:
                categories["两样都不占"] += 1

print(f"缺失线段 {total} 条（引脚保护半径 {PIN_RADIUS}px）：")
for key, value in categories.items():
    print(f"  {key:<16}{value:>5}  {value/max(1,total):.1%}")
