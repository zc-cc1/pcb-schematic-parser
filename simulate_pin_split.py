"""Would splitting our wire segments at pin attachment points convert the 'wrong cut' misses?

80.6% of the segments that miss lie on a truth line but are cut in the wrong place.
The graph only splits the skeleton at junctions and at excluded component bodies; a pin
that sits in the middle of an edge does not split it.  Simulate splitting there and see
how many misses turn into matches.
"""
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


def pins(doc):
    out = []
    for group in (doc.get("pins") or {}).values():
        for pin in (group.values() if isinstance(group, dict) else group):
            point = (pin or {}).get("point") or {}
            if "x" in point and "y" in point:
                out.append((float(point["x"]), float(point["y"])))
    return out


def match(a, b, tol=5.0):
    def dist(x, y):
        return max(math.dist(p, q) for p, q in zip(x, y))
    return dist(a, b) <= tol or dist(a, list(reversed(b))) <= tol


def project(p, a, b):
    ax, ay = a; bx, by = b
    vx, vy = bx - ax, by - ay
    norm2 = vx * vx + vy * vy
    if norm2 < 1e-9:
        return math.dist(p, a), 0.0
    t = ((p[0] - ax) * vx + (p[1] - ay) * vy) / norm2
    proj = (ax + t * vx, ay + t * vy)
    return math.dist(p, proj), t


total_missed = converted = split_needed = 0
case_rows = []
for folder in sorted(GT.iterdir(), key=lambda p: int(p.name)):
    case = folder.name
    truth = json.loads(next(folder.glob("*.json")).read_text(encoding="utf-8"))
    path = PRED / f"{case}_Pred.json"
    pred = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {"nets": {}}
    gt = [(a, b) for a, b in segs(truth)]
    pts = pins(pred)
    case_miss = case_fix = 0
    for pa, pb in segs(pred):
        if any(match((pa, pb), (a, b)) for a, b in gt):
            continue
        case_miss += 1
        cuts = []
        for point in pts:
            d, t = project(point, pa, pb)
            if d <= 4.0 and 0.05 < t < 0.95:
                cuts.append((t, point))
        if not cuts:
            continue
        cuts = [p for _, p in sorted(cuts)]
        chain = [pa] + cuts + [pb]
        pieces = [(chain[i], chain[i + 1]) for i in range(len(chain) - 1)]
        if any(match(piece, (a, b)) for piece in pieces for a, b in gt):
            case_fix += 1
    total_missed += case_miss
    converted += case_fix
    if case_miss:
        case_rows.append((case, case_miss, case_fix))

print(f"{'case':>4}{'未命中':>8}{'按引脚切开后能对上':>20}")
for case, miss, fix in case_rows:
    print(f"{case:>4}{miss:>8}{fix:>20}")
print(f"\n合计：未命中 {total_missed} 条，在引脚投影处切分后有 {converted} 条能对上 ({converted/max(1,total_missed):.1%})")
print(f"换算成分数：导线 tp 194 -> {194+converted}，F1 会从 .2962 升到约 "
      f"{2*(194+converted)/(2*(194+converted)+(372-converted)+(550-converted)):.4f}")
