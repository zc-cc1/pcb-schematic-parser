"""Would better net splitting raise the wire score?

Upper bound 1: every predicted segment may be credited if it matches ANY truth
segment (i.e. assume a perfect one-to-one net partition).
Upper bound 2: same, but the prediction keeps only segments whose net already hit.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, r"C:/Users/Pc/Desktop/新建文件夹/pcb_parser_competition/scripts")
from platform_metrics import _close, _segments, f1_block  # noqa: E402

GT = Path(r"C:/Users/Pc/Desktop/新建文件夹/赛题六公开数据集/赛题六公开数据集/open_GT/open_GT")
PRED = Path(r"D:/pcb_open_gt_v32best/final_results")

tp_now = fp_now = fn_now = 0
tp_any = 0
gt_seg_total = 0
pred_seg_total = 0
per_case = []

for folder in sorted(GT.iterdir(), key=lambda p: int(p.name)):
    case = folder.name
    truth = json.loads(next(folder.glob("*.json")).read_text(encoding="utf-8"))
    path = PRED / f"{case}_Pred.json"
    pred = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {"nets": {}}
    tn = list((truth.get("nets") or {}).values())
    pn = list((pred.get("nets") or {}).values())
    t_all = [s for n in tn for s in _segments(n)]
    p_all = [s for n in pn for s in _segments(n)]
    gt_seg_total += len(t_all)
    pred_seg_total += len(p_all)

    # current scoring (greedy per-net pairing)
    used, tp = set(), 0
    for pnet in pn:
        segs = _segments(pnet)
        best = None
        for i, tnet in enumerate(tn):
            if i in used:
                continue
            pool = _segments(tnet)
            hits = sum(1 for s in segs if any(_close(s, o) for o in pool))
            if best is None or hits > best[0]:
                best = (hits, i, pool)
        if best is None:
            continue
        used.add(best[1])
        tp += best[0]
    tp_now += tp
    fn_now += len(t_all) - tp

    # oracle partition: a predicted segment counts when it matches any truth segment
    hit = sum(1 for s in p_all if any(_close(s, o) for o in t_all))
    tp_any += hit
    per_case.append((case, len(t_all), len(p_all), tp, hit))

fp_now = pred_seg_total - tp_now
fn_oracle = gt_seg_total - tp_any
fp_oracle = pred_seg_total - tp_any

now = f1_block(tp_now, fp_now, fn_now)
orc = f1_block(tp_any, fp_oracle, fn_oracle)

print(f"{'case':>4}{'GT seg':>8}{'pred seg':>9}{'现在命中':>9}{'完美切分后可命中':>17}")
for row in per_case:
    print(f"{row[0]:>4}{row[1]:>8}{row[2]:>9}{row[3]:>9}{row[4]:>17}")
print()
print(f"现状:            tp {tp_now}  fp {fp_now}  fn {fn_now}   F1 {now['f1']:.4f} = {now['f1']*10:.2f}/10")
print(f"完美网络切分上限: tp {tp_any}  fp {fp_oracle}  fn {fn_oracle}   F1 {orc['f1']:.4f} = {orc['f1']*10:.2f}/10")
print()
print(f"线段本身正确率（不管怎么分网）: {tp_any}/{pred_seg_total} = {tp_any/pred_seg_total:.1%}")
print(f"GT 覆盖（我们画到了多少 GT 段）: {tp_any}/{gt_seg_total} = {tp_any/gt_seg_total:.1%}")

