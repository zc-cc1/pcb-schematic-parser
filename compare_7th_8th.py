"""Compare 7th vs 8th evaluation: per-case metrics, box/field changes, error budget."""
from __future__ import annotations

import json
from pathlib import Path

BASE = Path(r"C:/Users/Pc/Desktop/新建文件夹/评测结果")
E7 = BASE / "第七次评测-33.525"
E8 = BASE / "第8次测评-54.92"


def rep(root: Path) -> dict:
    d = next((root / "scoring_result/report/submission").glob("*"))
    return {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in d.glob("*.json") if p.stem.isdigit()}


def f1_of(section) -> float:
    if not isinstance(section, dict):
        return float(section)
    v = section.get("f1", section)
    while isinstance(v, dict):
        v = v.get("f1", v)
        if isinstance(v, dict):
            v = v.get("f1")
            break
    return float(v)


r7, r8 = rep(E7), rep(E8)
cases = sorted(r7, key=int)

print(f"{'case':>4} | {'comp7':>7}{'comp8':>7}{'d':>7} | {'pin7':>6}{'pin8':>6}{'d':>6} | "
      f"{'H7':>5}{'H8':>6}{'d':>6} | {'L7':>5}{'L8':>6}{'d':>6}")
print("-" * 92)
tot = {"comp": 0.0, "pin": 0.0, "H": 0.0, "L": 0.0}
for c in cases:
    a = r7[c]["component"]["f1"]["f1"]
    b = r8[c]["component"]["f1"]["f1"]
    pa, pb = f1_of(r7[c]["pin"]), f1_of(r8[c]["pin"])
    ha, hb = f1_of(r7[c]["connectivity"]), f1_of(r8[c]["connectivity"])
    la, lb = f1_of(r7[c]["wire"]), f1_of(r8[c]["wire"])
    print(f"{c:>4} | {a:>7.3f}{b:>7.3f}{b-a:>+7.3f} | {pa:>6.3f}{pb:>6.3f}{pb-pa:>+6.3f} | "
          f"{ha:>5.3f}{hb:>6.3f}{hb-ha:>+6.3f} | {la:>5.3f}{lb:>6.3f}{lb-la:>+6.3f}")
    tot["comp"] += b - a
    tot["pin"] += pb - pa
    tot["H"] += hb - ha
    tot["L"] += lb - la

n = len(cases)
print("-" * 92)
print(f"macro delta: comp {tot['comp']/n:+.4f} ({(tot['comp']/n)*30:+.2f} 分)  "
      f"pin {tot['pin']/n:+.4f} ({(tot['pin']/n)*25:+.2f})  "
      f"H {tot['H']/n:+.4f} ({(tot['H']/n)*35:+.2f})  "
      f"L {tot['L']/n:+.4f} ({(tot['L']/n)*10:+.2f})")

print()
print("case  comp  matched  type   Name   value  bbox   power(match/gt)   pin(gt_scored/pred_scored)  nets(gt/pred)")
for c in cases:
    d = r8[c]["component"]["diagnostics"]
    p = r8[c]["pin"]["diagnostics"]
    h = r8[c]["connectivity"]["diagnostics"]
    print(f" {c:>3}  {r8[c]['component']['f1']['f1']:.3f}   {d['matched_id_count']:>3}"
          f"   {d['type_accuracy_on_matched']:.3f}  {d['name_accuracy_on_matched']:.3f}"
          f"  {d['value_accuracy_on_matched']:.3f}  {d['bbox_accuracy_on_matched']:.3f}"
          f"   {d['matched_power_symbol_count']}/{d['gt_power_symbol_count']}"
          f"        {p['gt_scored_pin_count']:>3}/{p['pred_scored_pin_count']:<3}"
          f"   {h['gt_logical_net_count']:>3}/{h['pred_logical_net_count']:<3}")
