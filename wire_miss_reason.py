"""Why do our wire segments miss: offset from the line, or cut at the wrong place?"""
from __future__ import annotations

import json
import math
from pathlib import Path

GT = Path(r"C:/Users/Pc/Desktop/新建文件夹/赛题六公开数据集/赛题六公开数据集/open_GT/open_GT")
PRED = Path(r"D:/pcb_open_gt_v28b08/final_results")


def segs(doc):
    for net in (doc.get("nets") or {}).values():
        for edge in (net.get("edges") or {}).values():
            pts = [(float(p["x"]), float(p["y"])) for p in edge]
            if len(pts) >= 2:
                yield pts[0], pts[-1]


def match(a, b, tol=5.0):
    def dist(x, y):
        return max(math.dist(p, q) for p, q in zip(x, y))
    return dist(a, b) <= tol or dist(a, list(reversed(b))) <= tol


def perp_and_along(p, a, b):
    """Distance of p from line ab, and the parameter t (0..1) of its projection."""
    ax, ay = a
    bx, by = b
    vx, vy = bx - ax, by - ay
    norm = math.hypot(vx, vy)
    if norm < 1e-6:
        return math.dist(p, a), 0.0
    t = ((p[0] - ax) * vx + (p[1] - ay) * vy) / (norm * norm)
    proj = (ax + t * vx, ay + t * vy)
    return math.dist(p, proj), t


buckets = {"同线但端点出界(切错)": 0, "平行偏移 5-12px": 0, "偏移 >12px": 0, "附近没有线": 0}
examples = []
for folder in sorted(GT.iterdir(), key=lambda p: int(p.name)):
    case = folder.name
    truth = json.loads(next(folder.glob("*.json")).read_text(encoding="utf-8"))
    path = PRED / f"{case}_Pred.json"
    pred = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {"nets": {}}
    gt = [(a, b) for a, b in segs(truth)]
    for pa, pb in segs(pred):
        if any(match((pa, pb), (a, b)) for a, b in gt):
            continue
        best = None
        for a, b in gt:
            d1, t1 = perp_and_along(pa, a, b)
            d2, t2 = perp_and_along(pb, a, b)
            score = max(d1, d2)
            if best is None or score < best[0]:
                best = (score, d1, d2, t1, t2)
        if best is None:
            continue
        score, d1, d2, t1, t2 = best
        if score <= 5 and (t1 < -0.02 or t1 > 1.02 or t2 < -0.02 or t2 > 1.02):
            buckets["同线但端点出界(切错)"] += 1
            if len(examples) < 5:
                examples.append((case, [round(v, 1) for v in (d1, d2)], [round(t1, 2), round(t2, 2)]))
        elif score <= 12:
            buckets["平行偏移 5-12px"] += 1
        elif score <= 40:
            buckets["偏移 >12px"] += 1
        else:
            buckets["附近没有线"] += 1

total = sum(buckets.values())
print(f"未命中的预测线段共 {total} 条，分类：")
for key, value in buckets.items():
    print(f"  {key:<20}{value:>5}  {value/max(1,total):.1%}")
print("\n切错样例（case, 两端到最近真值线的垂距, 投影像素参数 t）:")
for e in examples:
    print("  ", e)
