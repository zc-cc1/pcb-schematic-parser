"""Segment geometry: are our wire polylines fragmented compared with the reference?"""
from __future__ import annotations

import json
import math
from pathlib import Path

GT = Path(r"C:/Users/Pc/Desktop/新建文件夹/赛题六公开数据集/赛题六公开数据集/open_GT/open_GT")
PRED = Path(r"D:/pcb_open_gt_v28b08/final_results")


def segs(doc):
    for net in (doc.get("nets") or {}).values():
        for edge in (net.get("edges") or {}).values():
            yield [(float(p["x"]), float(p["y"])) for p in edge]


def length(points):
    return sum(math.dist(a, b) for a, b in zip(points, points[1:]))


def stats(values):
    values = sorted(values)
    if not values:
        return "n=0"
    def q(f):
        return values[min(len(values) - 1, int(len(values) * f))]
    return (f"n={len(values)} median={q(.5):.1f} mean={sum(values)/len(values):.1f} "
            f"p90={q(.9):.1f} max={values[-1]:.1f}")


gt_len, pred_len, gt_pts, pred_pts = [], [], [], []
for folder in sorted(GT.iterdir(), key=lambda p: int(p.name)):
    case = folder.name
    truth = json.loads(next(folder.glob("*.json")).read_text(encoding="utf-8"))
    path = PRED / f"{case}_Pred.json"
    pred = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {"nets": {}}
    gt_len += [length(s) for s in segs(truth)]
    pred_len += [length(s) for s in segs(pred)]
    gt_pts += [len(s) for s in segs(truth)]
    pred_pts += [len(s) for s in segs(pred)]

print("线段长度（px）")
print("  真值 :", stats(gt_len))
print("  预测 :", stats(pred_len))
print("\n每条线的点数")
print("  真值 :", stats([float(v) for v in gt_pts]))
print("  预测 :", stats([float(v) for v in pred_pts]))
short = [x for x in pred_len if x < 5]
print(f"\n预测里 <5px 的碎线: {len(short)} / {len(pred_len)} = {len(short)/len(pred_len):.1%}")
print(f"真值里 <5px 的:      {sum(1 for x in gt_len if x < 5)} / {len(gt_len)}")
