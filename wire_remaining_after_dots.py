"""What is still missing in the wire bucket after the junction-dot split?

Two sides of the ledger on the current best run:
  * reference segments we still fail to reproduce  -> what the ink looks like there
  * our segments that match nothing                -> same line (cut mismatch) or wrong line
Plus net-level fragmentation.
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
from platform_metrics import _close, _members, _rails, _segments, _vertex_key  # noqa: E402

GT = Path(r"C:/Users/Pc/Desktop/新建文件夹/赛题六公开数据集/赛题六公开数据集/open_GT/open_GT")
IMAGES = Path(r"D:/pcb_open_gt_in")


def project(p, a, b):
    ax, ay = a
    bx, by = b
    vx, vy = bx - ax, by - ay
    norm = math.hypot(vx, vy)
    if norm < 1e-6:
        return math.dist(p, a), 0.0
    t = ((p[0] - ax) * vx + (p[1] - ay) * vy) / (norm * norm)
    return math.dist(p, (ax + t * vx, ay + t * vy)), t


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, default=Path(r"D:/pcb_open_gt_v37dots/final_results"))
    args = parser.parse_args()

    gt_total = gt_miss = 0
    our_total = our_bad = 0
    ink = Counter()
    reasons = Counter()
    fragments = []
    segment_lengths = []

    for folder in sorted(GT.iterdir(), key=lambda p: int(p.name)):
        case = folder.name
        truth = json.loads(next(folder.glob("*.json")).read_text(encoding="utf-8"))
        path = args.predictions / f"{case}_Pred.json"
        pred = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {"components": {}, "pins": {}, "nets": {}}
        our_edges = [seg for net in (pred.get("nets") or {}).values() for seg in _segments(net)]
        with Image.open(IMAGES / f"{case}.png") as handle:
            rgb = np.array(handle.convert("RGB"))
            height = rgb.shape[0]

        t_components = truth.get("components") or {}
        t_rail = _rails(t_components)
        p_components = pred.get("components") or {}
        p_rail = _rails(p_components)
        our_nets = []
        for net in (pred.get("nets") or {}).values():
            keys = set()
            for token in _members(net):
                c, p = _vertex_key(token, p_components, p_rail)
                keys.add(f"{c}.{p}" if p else c)
            our_nets.append(keys)

        for net in (truth.get("nets") or {}).values():
            keys = set()
            for token in _members(net):
                c, p = _vertex_key(token, t_components, t_rail)
                keys.add(f"{c}.{p}" if p else c)
            usable = keys & set().union(*our_nets) if our_nets else set()
            fragments.append(sum(1 for mine in our_nets if mine & usable))
            for seg in _segments(net):
                gt_total += 1
                if any(_close(seg, other) for other in our_edges):
                    continue
                gt_miss += 1
                segment_lengths.append(math.dist(seg[0], seg[-1]))
                samples = []
                steps = max(3, int(math.dist(seg[0], seg[-1])) + 1)
                for t in np.linspace(0, 1, steps):
                    x = int(round(seg[0][0] + t * (seg[-1][0] - seg[0][0])))
                    y = int(round(height - (seg[0][1] + t * (seg[-1][1] - seg[0][1]))))
                    if 0 <= y < rgb.shape[0] and 0 <= x < rgb.shape[1]:
                        samples.append(float(rgb[y, x].mean()))
                if samples:
                    values = np.asarray(samples)
                    ink["深色 (<110)" if (values < 110).mean() >= .5 else
                        ("浅色 (<200)" if (values < 200).mean() >= .5 else "空白")] += 1

        gt_all = [seg for net in (truth.get("nets") or {}).values() for seg in _segments(net)]
        for seg in our_edges:
            our_total += 1
            if any(_close(seg, other) for other in gt_all):
                continue
            our_bad += 1
            best = None
            for other in gt_all:
                score = max(project(seg[0], other[0], other[-1])[0], project(seg[-1], other[0], other[-1])[0])
                if best is None or score < best[0]:
                    best = (score, other)
            if best is None:
                reasons["没有可比的真值线"] += 1
            elif best[0] <= 5:
                reasons["落在真值线上但端点不对（切点仍不一致）"] += 1
            elif best[0] <= 15:
                reasons["偏离真值线 5-15px"] += 1
            else:
                reasons["偏离 >15px（画错了）"] += 1

    print(f"真值线段 {gt_total}，仍然缺 {gt_miss}（{gt_miss/gt_total:.1%}）")
    print(f"我们画的线段 {our_total}，其中对不上 {our_bad}（{our_bad/our_total:.1%}）")
    print(f"\n缺失线段处的墨迹：{dict(ink)}")
    print(f"我们对不上的线段，原因：{dict(reasons)}")
    if segment_lengths:
        values = np.asarray(segment_lengths)
        print(f"\n还缺的线段长度：中位 {np.median(values):.0f}px  <20px 占 {(values<20).mean():.0%}")
    counts = Counter(fragments)
    print(f"\n每个真值网被我们拆成几块（1 块 = 完整）：{dict(sorted(counts.items())[:8])}")


if __name__ == "__main__":
    main()
