"""What changed for power symbols between v15 and v16, per case?"""
import json
from pathlib import Path

GT = Path(r"C:/Users/Pc/Desktop/新建文件夹/赛题六公开数据集/赛题六公开数据集/open_GT/open_GT")
P15 = Path(r"D:/pcb_open_gt_v15/final_results")
P16 = Path(r"D:/pcb_open_gt_v16/final_results")
POWER = {"gnd", "ground", "v", "vcc", "vdd"}

for case in ("8", "12", "13", "1"):
    gt = json.loads((GT / case / f"{case}.json").read_text(encoding="utf-8"))["components"]
    a = json.loads((P15 / f"{case}_Pred.json").read_text(encoding="utf-8"))["components"]
    b = json.loads((P16 / f"{case}_Pred.json").read_text(encoding="utf-8"))["components"]
    print(f"=== case {case} ===")
    print(f"  GT power values: {sorted(v.get('value') for v in gt.values() if str(v.get('type','')).lower() in POWER)}")
    for tag, doc in (("v15", a), ("v16", b)):
        vals = {}
        for k, v in doc.items():
            if str(v.get("type", "")).lower() in POWER:
                vals[v.get("value")] = vals.get(v.get("value"), 0) + 1
        print(f"  {tag} power values: {vals}")
