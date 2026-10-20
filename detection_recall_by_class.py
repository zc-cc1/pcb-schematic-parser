"""Detection-level recall by class on the 13 released pages (current default output)."""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, r"C:/Users/Pc/Desktop/新建文件夹/pcb_parser_competition/scripts")
from platform_metrics import _box_error  # noqa: E402

GT = Path(r"C:/Users/Pc/Desktop/新建文件夹/赛题六公开数据集/赛题六公开数据集/open_GT/open_GT")
PRED = Path(r"D:/pcb_open_gt_v20/final_results")
IC = {"box", "ic"}

gt_by_type = Counter()
hit_by_type = Counter()
geo_hit_by_type = Counter()
pred_by_type = Counter()
missed = []          # GT items with no predicted box anywhere near
ic_rows = []


def same_class(a, b):
    a, b = str(a).lower(), str(b).lower()
    return a == b or (a in IC and b in IC)


for folder in sorted(GT.iterdir(), key=lambda p: int(p.name)):
    case = folder.name
    truth = json.loads(next(folder.glob("*.json")).read_text(encoding="utf-8"))["components"]
    pred = json.loads((PRED / f"{case}_Pred.json").read_text(encoding="utf-8"))["components"]
    for v in pred.values():
        pred_by_type[str(v.get("type"))] += 1
    for key, ref in truth.items():
        t = str(ref.get("type"))
        gt_by_type[t] += 1
        cand = pred.get(key)
        if cand is not None and same_class(cand.get("type"), t):
            hit_by_type[t] += 1
        # geometric recall: any predicted component of the same class within 20 px
        best = None
        for pk, pv in pred.items():
            if not same_class(pv.get("type"), t):
                continue
            err = _box_error(pv.get("bbox") or [], ref.get("bbox") or [])
            if best is None or err < best[0]:
                best = (err, pk)
        if best and best[0] <= 20:
            geo_hit_by_type[t] += 1
        else:
            missed.append((case, key, t, None if not best else round(best[0], 1)))
        if t in IC:
            ic_rows.append((case, key, str(ref.get("Name"))[:20],
                            "ok" if (cand is not None and same_class(cand.get("type"), t)) else "key/type missing",
                            None if best is None else round(best[0], 1)))

print(f"{'type':<16}{'GT':>4}{'pred':>6}{'同key命中':>10}{'几何召回':>9}")
for t, n in gt_by_type.most_common():
    print(f"{t:<16}{n:>4}{pred_by_type.get(t, 0):>6}{hit_by_type[t]:>10}{geo_hit_by_type[t]:>9}")
print()
print(f"IC/box 合计: GT {sum(n for t, n in gt_by_type.items() if t in IC)}  "
      f"预测 {sum(n for t, n in pred_by_type.items() if t in IC)}  "
      f"同 key {sum(n for t, n in hit_by_type.items() if t in IC)}  "
      f"几何 {sum(n for t, n in geo_hit_by_type.items() if t in IC)}")
print()
print("逐个 IC：")
for row in ic_rows:
    print(f"  case {row[0]:>2} {row[1]:<8} GT Name={row[2]:<22} {row[3]:<18} 最近同类框误差={row[4]}")
print()
print("完全没检测到（同类且 20px 内无任何预测）:")
for m in missed:
    print(f"  case {m[0]:>2} {m[1]:<10} {m[2]:<16} 最近误差 {m[3]}")
