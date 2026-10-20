"""Attribute the v13 -> v15 delta per case and per metric."""
import json
from pathlib import Path

A = json.loads(Path(r"C:/Users/Pc/Desktop/新建文件夹/score_v13.json").read_text(encoding="utf-8"))
B = json.loads(Path(r"C:/Users/Pc/Desktop/新建文件夹/score_v15.json").read_text(encoding="utf-8"))

a = {c["case_id"]: c for c in A["per_case"]}
b = {c["case_id"]: c for c in B["per_case"]}

print(f"{'case':>4} | {'C13':>6}{'C15':>6}{'d':>7} | {'P13':>6}{'P15':>6}{'d':>7} | {'N13':>6}{'N15':>6}{'d':>7} | "
      f"{'W13':>6}{'W15':>6}{'d':>7} | {'F13':>6}{'F15':>6}{'d':>7}")
print("-" * 106)
tot = {"C": 0.0, "P": 0.0, "N": 0.0, "W": 0.0}
for case in sorted(a, key=int):
    x, y = a[case], b[case]
    c = (x["component_f1"]["f1"], y["component_f1"]["f1"])
    p = (x["pin_f1"]["f1"], y["pin_f1"]["f1"])
    n = (x["connectivity_f1"]["f1"], y["connectivity_f1"]["f1"])
    w = (x["wire_f1"]["f1"], y["wire_f1"]["f1"])
    fx = 0.30 * c[0] + 0.25 * p[0] + 0.35 * n[0] + 0.10 * w[0]
    fy = 0.30 * c[1] + 0.25 * p[1] + 0.35 * n[1] + 0.10 * w[1]
    print(f"{case:>4} | {c[0]:>6.3f}{c[1]:>6.3f}{c[1]-c[0]:>+7.3f} | {p[0]:>6.3f}{p[1]:>6.3f}{p[1]-p[0]:>+7.3f} | "
          f"{n[0]:>6.3f}{n[1]:>6.3f}{n[1]-n[0]:>+7.3f} | {w[0]:>6.3f}{w[1]:>6.3f}{w[1]-w[0]:>+7.3f} | "
          f"{fx*100:>6.1f}{fy*100:>6.1f}{(fy-fx)*100:>+7.1f}")
    for name, (u, v) in (("C", c), ("P", p), ("N", n), ("W", w)):
        tot[name] += v - u

n = len(a)
weights = {"C": 0.30, "P": 0.25, "N": 0.35, "W": 0.10}
print("-" * 106)
for name in ("C", "P", "N", "W"):
    print(f"macro {name}: {tot[name]/n:+.4f}  ->  {tot[name]/n*weights[name]*100:+.2f} 分")
print(f"total: {A['final_score']*100:.2f} -> {B['final_score']*100:.2f}  ({(B['final_score']-A['final_score'])*100:+.2f})")
print()
print("metrics v15:", {k: round(v["f1"], 4) for k, v in B["metrics"].items()})
print("metrics v13:", {k: round(v["f1"], 4) for k, v in A["metrics"].items()})
