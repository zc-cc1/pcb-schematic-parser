"""Why did component_f1 drop in the 8th run? Compare per-case diagnostics and fields."""
from __future__ import annotations

import json
from pathlib import Path

BASE = Path(r"C:/Users/Pc/Desktop/新建文件夹/评测结果")
E7 = BASE / "第七次评测-33.525"
E8 = BASE / "第8次测评-54.92"


def rep(root: Path) -> dict:
    d = next((root / "scoring_result/report/submission").glob("*"))
    return {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in d.glob("*.json") if p.stem.isdigit()}


def comps(root: Path) -> dict:
    d = root / "intermediate_files/final_results"
    return {p.stem.split("_")[0]: json.loads(p.read_text(encoding="utf-8")) for p in d.glob("*_Pred.json")}


r7, r8 = rep(E7), rep(E8)
c7, c8 = comps(E7), comps(E8)
cases = sorted(r7, key=int)

print(f"{'case':>4} | {'match7':>7}{'match8':>7} | {'type7':>6}{'type8':>6} | {'name7':>6}{'name8':>6} | "
      f"{'val7':>6}{'val8':>6} | {'bbox7':>6}{'bbox8':>6} | {'pow7':>6}{'pow8':>6}")
print("-" * 108)
for c in cases:
    a, b = r7[c]["component"]["diagnostics"], r8[c]["component"]["diagnostics"]
    print(f"{c:>4} | {a['matched_id_count']:>7}{b['matched_id_count']:>7} | "
          f"{a['type_accuracy_on_matched']:>6.3f}{b['type_accuracy_on_matched']:>6.3f} | "
          f"{a['name_accuracy_on_matched']:>6.3f}{b['name_accuracy_on_matched']:>6.3f} | "
          f"{a['value_accuracy_on_matched']:>6.3f}{b['value_accuracy_on_matched']:>6.3f} | "
          f"{a['bbox_accuracy_on_matched']:>6.3f}{b['bbox_accuracy_on_matched']:>6.3f} | "
          f"{a['matched_power_symbol_count']:>6}{b['matched_power_symbol_count']:>6}")

print()
print("field-level changes between the two runs (same component id):")
for c in ["1", "7", "10"]:
    A, B = c7[c]["components"], c8[c]["components"]
    changed = {"Name": 0, "value": 0, "type": 0, "bbox": 0}
    examples = []
    for cid, comp in A.items():
        other = B.get(cid)
        if not other:
            continue
        for k in ("Name", "value", "type"):
            if comp.get(k) != other.get(k):
                changed[k] += 1
                if len(examples) < 6:
                    examples.append((cid, k, comp.get(k), other.get(k)))
        if comp.get("bbox") != other.get("bbox"):
            changed["bbox"] += 1
    print(f"  case {c}: {changed}")
    for e in examples:
        print(f"     {e[0]:<12} {e[1]}: {e[2]!r} -> {e[3]!r}")
