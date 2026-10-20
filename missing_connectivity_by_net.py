"""Per-net understanding of what connectivity is missing across the 13 pages.

For every reference net: how many of its wire edges do we draw at all (geometry only,
ignoring which net we put them in), how many fragments we split it into, and - for the
edges we miss - what the page looks like under them (dark wire? coloured? blank?).
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
from platform_metrics import _close, _members, _rails, _segments, _vertex_key  # noqa: E402

GT = Path(r"C:/Users/Pc/Desktop/新建文件夹/赛题六公开数据集/赛题六公开数据集/open_GT/open_GT")
IMAGES = Path(r"D:/pcb_open_gt_in")
PRED = Path(r"D:/pcb_open_gt_v32best/final_results")

rows = []
missing_profile = Counter()
missing_lengths = []
per_case = []

for folder in sorted(GT.iterdir(), key=lambda p: int(p.name)):
    case = folder.name
    truth = json.loads(next(folder.glob("*.json")).read_text(encoding="utf-8"))
    path = PRED / f"{case}_Pred.json"
    pred = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {"components": {}, "pins": {}, "nets": {}}
    t_components = truth.get("components") or {}
    t_rail = _rails(t_components)
    p_components = pred.get("components") or {}
    p_rail = _rails(p_components)

    our_edges = [seg for net in (pred.get("nets") or {}).values() for seg in _segments(net)]
    our_nets = []
    for net in (pred.get("nets") or {}).values():
        keys = set()
        for token in _members(net):
            c, p = _vertex_key(token, p_components, p_rail)
            keys.add(f"{c}.{p}" if p else c)
        our_nets.append(keys)

    with Image.open(IMAGES / f"{case}.png") as handle:
        rgb = np.array(handle.convert("RGB"))
        height = rgb.shape[0]

    case_edges = case_missing = 0
    for net_id, net in (truth.get("nets") or {}).items():
        edges = _segments(net)
        if not edges:
            continue
        keys = set()
        for token in _members(net):
            c, p = _vertex_key(token, t_components, t_rail)
            keys.add(f"{c}.{p}" if p else c)
        usable = keys & set().union(*our_nets) if our_nets else set()
        pieces = sum(1 for mine in our_nets if mine & usable)
        matched = 0
        missed = []
        for seg in edges:
            if any(_close(seg, other) for other in our_edges):
                matched += 1
            else:
                missed.append(seg)
        case_edges += len(edges)
        case_missing += len(missed)
        for seg in missed:
            a, b = seg[0], seg[-1]
            missing_lengths.append(math.dist(a, b))
            samples = []
            steps = max(3, int(math.dist(a, b)) + 1)
            for t in np.linspace(0, 1, steps):
                x = int(round(a[0] + t * (b[0] - a[0])))
                y = int(round(height - (a[1] + t * (b[1] - a[1]))))
                if 0 <= y < rgb.shape[0] and 0 <= x < rgb.shape[1]:
                    samples.append(rgb[y, x].astype(float))
            if not samples:
                continue
            pixels = np.asarray(samples)
            brightness = pixels.mean(axis=1)
            if (brightness < 110).mean() >= .5:
                missing_profile["深色线（模型应该看得见）"] += 1
            elif (brightness < 200).mean() >= .5:
                missing_profile["浅色/彩色线"] += 1
            else:
                missing_profile["几乎是空白"] += 1
        rows.append((case, net_id, len(keys), pieces, len(edges), matched))
    per_case.append((case, case_edges, case_missing))

print(f"{'case':>4}{'真值线段':>10}{'我们缺失':>10}{'覆盖率':>9}")
for case, edges, missing in per_case:
    print(f"{case:>4}{edges:>10}{missing:>10}{(edges-missing)/edges:>9.1%}")

total_edges = sum(e for _, e, _ in per_case)
total_missing = sum(m for _, _, m in per_case)
print(f"\n合计：真值 {total_edges} 段，缺失 {total_missing} 段（{total_missing/total_edges:.1%}）")
if missing_lengths:
    values = np.asarray(missing_lengths)
    print(f"缺失段长度：中位 {np.median(values):.0f}px  均值 {values.mean():.0f}px  "
          f"<20px 占 {(values<20).mean():.0%}  >100px 占 {(values>100).mean():.0%}")
print("\n缺失线的画面特征：", dict(missing_profile))

big = sorted(rows, key=lambda r: -(r[4] - r[5]))[:10]
print("\n缺得最多的 10 个真值网（case, 网, 顶点数, 我们拆成几块, 总段数, 对上几段）：")
for row in big:
    print("  ", row)
