"""Wire: how much of the loss is 'net never paired' vs 'paired but wrong geometry'."""
from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, r"C:/Users/Pc/Desktop/新建文件夹/pcb_parser_competition/scripts")
from platform_metrics import _close, _segments  # noqa: E402

GT = Path(r"C:/Users/Pc/Desktop/新建文件夹/赛题六公开数据集/赛题六公开数据集/open_GT/open_GT")
PRED = Path(r"D:/pcb_open_gt_v21net/final_results")

zero_net_segs = 0
partial_net_segs = 0
zero_nets = 0
partial_nets = 0
pred_sizes, gt_sizes = [], []
for folder in sorted(GT.iterdir(), key=lambda p: int(p.name)):
    case = folder.name
    truth = json.loads(next(folder.glob("*.json")).read_text(encoding="utf-8"))
    path = PRED / f"{case}_Pred.json"
    pred = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {"nets": {}}
    tn = list((truth.get("nets") or {}).values())
    pn = list((pred.get("nets") or {}).values())
    gt_sizes += [len(_segments(n)) for n in tn if _segments(n)]
    pred_sizes += [len(_segments(n)) for n in pn if _segments(n)]
    used = set()
    for pnet in pn:
        segs = _segments(pnet)
        if not segs:
            continue
        best = None
        for i, tnet in enumerate(tn):
            if i in used:
                continue
            pool = _segments(tnet)
            hits = sum(1 for s in segs if any(_close(s, o) for o in pool))
            if best is None or hits > best[0]:
                best = (hits, i)
        if best is None or best[0] == 0:
            zero_nets += 1
            zero_net_segs += len(segs)
            continue
        used.add(best[1])
        partial_nets += 1
        partial_net_segs += len(segs) - best[0]

print(f"预测网络 {zero_nets + partial_nets} 个 / {sum(pred_sizes)} 段")
print(f"  完全没配上（0 命中）的网络: {zero_nets} 个，占 {zero_net_segs} 段")
print(f"  配上了但段对不上的:          {partial_net_segs} 段（在 {partial_nets} 个网络里）")
print()
def stats(xs):
    xs = sorted(xs)
    return f"n={len(xs)} median={xs[len(xs)//2]} mean={sum(xs)/len(xs):.1f} p90={xs[int(len(xs)*.9)]}"
print("GT 每个网络的段数:", stats(gt_sizes))
print("我们每个网络的段数:", stats(pred_sizes))
