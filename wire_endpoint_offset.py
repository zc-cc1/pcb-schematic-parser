"""Where exactly do our wire segment endpoints land relative to the reference?

For every predicted segment that misses, find the nearest truth segment and report how
far each endpoint is from the truth segment's two ends (signed, along the line).  A
positive "short" bias means we stop before the reference does - the mask is trimmed by
the component guard.  A symmetric spread means the cuts simply do not coincide.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

GT = Path(r"C:/Users/Pc/Desktop/新建文件夹/赛题六公开数据集/赛题六公开数据集/open_GT/open_GT")


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


def project(p, a, b):
    ax, ay = a; bx, by = b
    vx, vy = bx - ax, by - ay
    norm = math.hypot(vx, vy)
    if norm < 1e-6:
        return math.dist(p, a), 0.0, norm
    t = ((p[0] - ax) * vx + (p[1] - ay) * vy) / (norm * norm)
    proj = (ax + t * vx, ay + t * vy)
    return math.dist(p, proj), t, norm


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, default=Path(r"D:/pcb_open_gt_v30split/final_results"))
    args = parser.parse_args()

    short = long = extra = 0
    offsets = []
    for folder in sorted(GT.iterdir(), key=lambda p: int(p.name)):
        case = folder.name
        truth = json.loads(next(folder.glob("*.json")).read_text(encoding="utf-8"))
        path = args.predictions / f"{case}_Pred.json"
        pred = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {"nets": {}}
        gt = [(a, b) for a, b in segs(truth)]
        for pa, pb in segs(pred):
            if any(match((pa, pb), (a, b)) for a, b in gt):
                continue
            best = None
            for a, b in gt:
                score = max(project(pa, a, b)[0], project(pb, a, b)[0])
                if best is None or score < best[0]:
                    best = (score, a, b)
            if best is None:
                continue
            _, a, b = best
            _, t1, norm = project(pa, a, b)
            _, t2, _ = project(pb, a, b)
            t1, t2 = min(t1, t2), max(t1, t2)
            inside = (t1 >= -0.02) and (t2 <= 1.02)
            if t1 < -0.02 and t2 > 1.02:
                long += 1
            elif inside:
                extra += 1
            elif t1 < -0.02 and abs(t1) * norm > 4:
                long += 1
            elif t2 > 1.02 and (t2 - 1) * norm > 4:
                long += 1
            else:
                short += 1
            offsets.append((round(t1 * norm, 1), round((t2 - 1) * norm, 1)))

    total = short + long + extra
    print(f"未命中线段 {total} 条：")
    print(f"  比真值线长（越界超过 4px）: {long:>4}  {long/max(1,total):.1%}")
    print(f"  比真值线短 / 端点差一点  : {short:>4}  {short/max(1,total):.1%}")
    print(f"  完全在真值线内部（被切碎）: {extra:>4}  {extra/max(1,total):.1%}")
    buckets = Counter()
    for t1, t2 in offsets:
        buckets[(int(t1 // 10) * 10, int(t2 // 10) * 10)] += 1
    print("\n端点沿线的相对位置分布（起点偏移, 终点偏移, 条数）:")
    for (a, b), n in buckets.most_common(8):
        print(f"  起点 {a:+4d}..{a+10:+4d}px  终点 {b:+4d}..{b+10:+4d}px  {n}")


if __name__ == "__main__":
    main()
