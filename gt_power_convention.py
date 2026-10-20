"""GT convention for power values, and how many of our blanks are gnd vs supply."""
import json
from collections import Counter
from pathlib import Path

GT = Path(r"C:/Users/Pc/Desktop/新建文件夹/赛题六公开数据集/赛题六公开数据集/open_GT/open_GT")
LOGS = Path(r"D:/pcb_open_gt_v15/logs")

gnd_vals, v_vals = Counter(), Counter()
for folder in sorted(GT.iterdir(), key=lambda p: int(p.name)):
    gt = json.loads(next(folder.glob("*.json")).read_text(encoding="utf-8"))["components"]
    for comp in gt.values():
        t = str(comp.get("type", "")).lower()
        val = comp.get("value")
        if t in {"gnd", "ground"}:
            gnd_vals[val] += 1
        elif t in {"v", "vcc", "vdd"}:
            v_vals[val] += 1

print("GT gnd values:", dict(gnd_vals))
print("GT supply(v/vcc) values:", dict(v_vals))

print("\nour blank power values per case (v15):")
tot = Counter()
for log in sorted(LOGS.glob("*_component_export_audit.json"), key=lambda p: int(p.stem.split("_")[0])):
    audit = json.loads(log.read_text(encoding="utf-8"))
    blank = [r for r in audit["added"] if not r["value"]]
    c = Counter(r["type"] for r in blank)
    tot.update(c)
    print(f"  case {log.stem.split('_')[0]:>2}: {dict(c)}")
print("  total blanks by type:", dict(tot))
