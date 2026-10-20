"""How much of each missing reference segment actually sits inside a pin-guard disk?

A segment that merely ends at a pin is not affected by the guard; one that runs through
the disk is.  Buckets by share of sampled points inside any disk.
"""
from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, r"C:/Users/Pc/Desktop/新建文件夹/pcb_parser_competition/scripts")
from platform_metrics import _segments  # noqa: E402

GT = Path(r"C:/Users/Pc/Desktop/新建文件夹/赛题六公开数据集/赛题六公开数据集/open_GT/open_GT")
PRED = Path(r"D:/pcb_open_gt_v37dots/final_results")
CFG = json.loads(Path(r"D:/pcb5_submission/configs/wire_graph_v1.json").read_text(encoding="utf-8"))
RADIUS = float(CFG["pin_protection_radius_px"])


def close(a, b, tol=20.0):
    return ((math.dist(a[0], b[0]) <= tol and math.dist(a[-1], b[-1]) <= tol)
            or (math.dist(a[0], b[-1]) <= tol and math.dist(a[-1], b[0]) <= tol))


shares = Counter()
examples = []
for folder in sorted(GT.iterdir(), key=lambda p: int(p.name)):
    case = folder.name
    truth = json.loads(next(folder.glob("*.json")).read_text(encoding="utf-8"))
    path = PRED / f"{case}_Pred.json"
    pred = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {"pins": {}, "nets": {}}
    our = [seg for net in (pred.get("nets") or {}).values() for seg in _segments(net)]
    pins = []
    for group in (pred.get("pins") or {}).values():
        for pin in (group.values() if isinstance(group, dict) else group):
            point = (pin or {}).get("point") or {}
            if "x" in point:
                pins.append((float(point["x"]), float(point["y"])))
    if not pins:
        continue
    points = np.asarray(pins)
    for net in (truth.get("nets") or {}).values():
        for seg in _segments(net):
            if any(close(seg, other) for other in our):
                continue
            steps = max(3, int(math.dist(seg[0], seg[-1])) + 1)
            samples = np.asarray([[seg[0][0] + t * (seg[-1][0] - seg[0][0]),
                                   seg[0][1] + t * (seg[-1][1] - seg[0][1])]
                                  for t in np.linspace(0, 1, steps)])
            distances = np.linalg.norm(samples[:, None, :] - points[None, :, :], axis=2).min(axis=1)
            share = float((distances <= RADIUS).mean())
            if share >= 0.8:
                shares["整段都在圈内"] += 1
            elif share >= 0.4:
                shares["一半以上在圈内"] += 1
            elif share > 0.05:
                shares["只有端头碰上"] += 1
            else:
                shares["完全在圈外"] += 1
            if 0.4 <= share < 0.8 and len(examples) < 4:
                examples.append((case, [round(v) for v in seg[0]], [round(v) for v in seg[-1]], round(share, 2)))

print(f"缺失线段 {sum(shares.values())} 条，按落在引脚保护圈（半径 {RADIUS}px）内的比例：")
for key, value in shares.items():
    print(f"  {key:<14}{value:>4}  {value/max(1,sum(shares.values())):.1%}")
print("\n示例（case, 起点, 终点, 在圈内的比例）：")
for example in examples:
    print("  ", example)
