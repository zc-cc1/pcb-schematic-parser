"""How coupled are the wire (10%) and connectivity (35%) scores on the current best run?"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, r"C:/Users/Pc/Desktop/新建文件夹/pcb_parser_competition/scripts")
from platform_metrics import _close, _segments, f1_block  # noqa: E402

GT = Path(r"C:/Users/Pc/Desktop/新建文件夹/赛题六公开数据集/赛题六公开数据集/open_GT/open_GT")
PRED = Path(r"D:/pcb_open_gt_v28b08/final_results")
SCORE = Path(r"C:/Users/Pc/Desktop/新建文件夹/score_v28b08.json")

rows = []
tp_now = fp_now = fn_now = 0
tp_any = 0
gt_seg = pred_seg = 0

for folder in sorted(GT.iterdir(), key=lambda p: int(p.name)):
    case = folder.name
    truth = json.loads(next(folder.glob("*.json")).read_text(encoding="utf-8"))
    pred = json.loads((PRED / f"{case}_Pred.json").read_text(encoding="utf-8"))
    tn, pn = list((truth.get("nets") or {}).values()), list((pred.get("nets") or {}).values())
    t_all = [s for n in tn for s in _segments(n)]
    p_all = [s for n in pn for s in _segments(n)]
    gt_seg += len(t_all); pred_seg += len(p_all)
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
                best = (hits, i)
        if best is None:
            continue
        used.add(best[1]); tp += best[0]
    tp_now += tp; fn_now += len(t_all) - tp
    tp_any += sum(1 for s in p_all if any(_close(s, o) for o in t_all))
    rows.append((case, len(t_all), len(p_all), tp, len(tn), len(pn)))

fp_now = pred_seg - tp_now
score = json.loads(SCORE.read_text(encoding="utf-8"))
per = {c["case_id"]: c for c in score["per_case"]}

print(f"{'case':>4}{'GT seg':>8}{'pred seg':>9}{'wire F1':>9}{'连通 F1':>9}{'GT nets':>9}{'pred nets':>10}")
pairs = []
for case, gt_n, p_n, tp, gt_nets, p_nets in rows:
    w = per[case]["wire_f1"]["f1"]; c = per[case]["connectivity_f1"]["f1"]
    pairs.append((w, c))
    print(f"{case:>4}{gt_n:>8}{p_n:>9}{w:>9.3f}{c:>9.3f}{gt_nets:>9}{p_nets:>10}")

print()
now = f1_block(tp_now, fp_now, fn_now)
orc = f1_block(tp_any, pred_seg - tp_any, gt_seg - tp_any)
print(f"现状 wire F1 {now['f1']:.4f} ({now['f1']*10:.2f}/10)   tp {tp_now} fp {fp_now} fn {fn_now}")
print(f"网络完美切分后的上限 {orc['f1']:.4f} ({orc['f1']*10:.2f}/10)   tp {tp_any}")
print(f"线段本身画对的比例 {tp_any}/{pred_seg} = {tp_any/pred_seg:.1%}，GT 覆盖率 {tp_any}/{gt_seg} = {tp_any/gt_seg:.1%}")
n = len(pairs)
mw = sum(a for a, _ in pairs) / n; mc = sum(b for _, b in pairs) / n
cov = sum((a - mw) * (b - mc) for a, b in pairs) / n
sw = (sum((a - mw) ** 2 for a, _ in pairs) / n) ** .5
sc = (sum((b - mc) ** 2 for _, b in pairs) / n) ** .5
print(f"\n逐例相关系数 (wire vs 连通): r = {cov/(sw*sc):.3f}")
