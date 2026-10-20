"""Zoom into the three biggest component buckets: bbox, value, missing key."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

sys_path = r"C:/Users/Pc/Desktop/新建文件夹/pcb_parser_competition/scripts"
import sys

sys.path.insert(0, sys_path)
from platform_metrics import COMPONENT_TOLERANCE_PX, POWER_KIND, _box_error, ntype  # noqa: E402

GT = Path(r"C:/Users/Pc/Desktop/新建文件夹/赛题六公开数据集/赛题六公开数据集/open_GT/open_GT")
PRED = Path(r"D:/pcb_open_gt_v17/final_results")

bbox_errs = []
missing = []
value_fail = []
name_fail = []
net_items = []

for folder in sorted(GT.iterdir(), key=lambda p: int(p.name)):
    case = folder.name
    truth = json.loads(next(folder.glob("*.json")).read_text(encoding="utf-8"))["components"]
    pred = json.loads((PRED / f"{case}_Pred.json").read_text(encoding="utf-8"))["components"]
    for key, ref in truth.items():
        kind = ntype(ref.get("type"))
        if kind in POWER_KIND:
            continue
        cand = pred.get(key)
        where = (case, key, kind, str(cand.get("Name"))[:16] if cand else None,
                 str(ref.get("Name"))[:16], str(cand.get("value"))[:9] if cand else None,
                 str(ref.get("value"))[:9])
        if cand is None:
            missing.append(where)
            continue
        err = _box_error(cand.get("bbox") or [], ref.get("bbox") or [])
        if ntype(cand.get("type")) != kind:
            continue
        name_bad = str(cand.get("Name")) != str(ref.get("Name"))
        value_bad = str(cand.get("value")) != str(ref.get("value"))
        if err > COMPONENT_TOLERANCE_PX:
            bbox_errs.append((*where, round(err, 1)))
        elif value_bad:
            value_fail.append(where)
        elif name_bad:
            name_fail.append(where)
        if kind.startswith("net") and (cand is None or str(cand.get("type")) != str(ref.get("type"))):
            net_items.append((case, key, kind, str(cand.get("type")) if cand else None))

print(f"bbox FNs: {len(bbox_errs)}")
for e in sorted(bbox_errs, key=lambda x: x[-1]):
    print(f"   {e[0]:>2} {e[1]:<10} {e[2]:<15} Name {e[3]!s:<16}/{e[4]:<16} value {e[5]!s:<9}/{e[6]:<9} err={e[7]}")
print()
print("marginal bucket:", Counter(("20-25" if e[-1] <= 25 else "25-35" if e[-1] <= 35 else ">35") for e in bbox_errs))
print("by type:", Counter(e[2] for e in bbox_errs))
print()
print(f"value-only FNs: {len(value_fail)}")
print("   by type:", Counter(e[2] for e in value_fail))
for e in value_fail[:14]:
    print(f"   {e[0]:>2} {e[1]:<8} {e[2]:<10} value {e[5]!s:<9} -> GT {e[6]}")
print()
print(f"Name-only FNs: {len(name_fail)}")
print("   by type:", Counter(e[2] for e in name_fail))
for e in name_fail[:14]:
    print(f"   {e[0]:>2} {e[1]:<8} {e[2]:<8} Name {e[3]!s:<20} -> GT {e[4]}")
print()
print(f"missing keys: {len(missing)}")
print("   by type:", Counter(e[2] for e in missing))
for e in missing[:22]:
    print(f"   {e[0]:>2} {e[1]:<10} {e[2]:<16} GT Name={e[4]!s:<14} value={e[6]}")
