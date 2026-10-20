"""Wire (net_line) error budget on the current best local run."""
from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, r"C:/Users/Pc/Desktop/新建文件夹/pcb_parser_competition/scripts")
from platform_metrics import _close, _segments, f1_block  # noqa: E402

GT = Path(r"C:/Users/Pc/Desktop/新建文件夹/赛题六公开数据集/赛题六公开数据集/open_GT/open_GT")
PRED = Path(r"D:/pcb_open_gt_v21net/final_results")


def seg_dist(a, b):
    if len(a) < 2 or len(b) < 2:
        return 1e9
    def d(x, y):
        return max(math.hypot(p[0] - q[0], p[1] - q[1]) for p, q in zip(x, y))
    return min(d(a, b), d(a, list(reversed(b))))


print(f"{'case':>4} {'GT nets':>8}{'pred nets':>10}{'GT seg':>8}{'pred seg':>9}{'tp':>5}{'fp':>5}{'fn':>5}{'P':>7}{'R':>7}")
rows = []
near = Counter()
for folder in sorted(GT.iterdir(), key=lambda p: int(p.name)):
    case = folder.name
    truth = json.loads(next(folder.glob("*.json")).read_text(encoding="utf-8"))
    path = PRED / f"{case}_Pred.json"
    pred = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {"nets": {}}
    tn, pn = (truth.get("nets") or {}), (pred.get("nets") or {})
    t_segs = [s for n in tn.values() for s in _segments(n)]
    p_segs = [s for n in pn.values() for s in _segments(n)]
    # greedy per-net pairing, same as platform_metrics.score_wire
    used, tp = set(), 0
    unmatched_pred = []
    for pnet in pn.values():
        segs = _segments(pnet)
        best = None
        for i, tnet in enumerate(tn.values()):
            if i in used:
                continue
            pool = _segments(tnet)
            hits = sum(1 for s in segs if any(_close(s, o) for o in pool))
            if best is None or hits > best[0]:
                best = (hits, i, pool)
        if best is None:
            unmatched_pred += segs
            continue
        hits, i, pool = best
        used.add(i)
        tp += hits
        for s in segs:
            if not any(_close(s, o) for o in pool):
                # how far is this missed segment from the closest truth segment?
                d = min((seg_dist(s, o) for o in pool), default=1e9)
                near["<=8px" if d <= 8 else "8-15px" if d <= 15 else "15-30px" if d <= 30 else ">30px"] += 1
                unmatched_pred.append(s)
    fp = len(p_segs) - tp
    fn = len(t_segs) - tp
    rows.append((case, len(tn), len(pn), len(t_segs), len(p_segs), tp, fp, fn))
    print(f"{case:>4} {len(tn):>8}{len(pn):>10}{len(t_segs):>8}{len(p_segs):>9}{tp:>5}{fp:>5}{fn:>5}"
          f"{tp/max(1,len(p_segs)):>7.2f}{tp/max(1,len(t_segs)):>7.2f}")

tot = [sum(r[i] for r in rows) for i in range(3, 8)]
blk = f1_block(tot[2], tot[3], tot[4])
print(f"{'ALL':>4} {sum(r[1] for r in rows):>8}{sum(r[2] for r in rows):>10}{tot[0]:>8}{tot[1]:>9}"
      f"{tot[2]:>5}{tot[3]:>5}{tot[4]:>5}{blk['precision']:>7.2f}{blk['recall']:>7.2f}")
print(f"\nnet_line F1 {blk['f1']:.4f}  -> {blk['f1']*10:.2f} / 10 分")
print("\n未命中预测段离最近真值段的端点距离分布:", dict(near))

