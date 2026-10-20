"""打印修复跑分的逐例结果 + 哪些元件补了脚。"""
import json
from pathlib import Path

path = Path(r"C:/Users/Pc/Desktop/新建文件夹/runs/g_fill6/pin_repair_report.json")
data = json.loads(path.read_text(encoding="utf-8"))
print(f"{'case':>4} {'tp':>3} {'fp':>3} {'fn':>3}  {'PinF1':>5}  {'CompF1':>6}  {'final':>6}")
total = {"tp": 0, "fp": 0, "fn": 0}
for case in data["cases"]:
    total["tp"] += case["pin"]["tp"]
    total["fp"] += case["pin"]["fp"]
    total["fn"] += case["pin"]["fn"]
    print(f"{case['case']:>4} {case['pin']['tp']:>3} {case['pin']['fp']:>3} {case['pin']['fn']:>3}"
          f"  {case['pin_f1']:>5.3f}  {case['component_f1']:>6.3f}  {case['final']:>6.4f}")
print(f"{'ALL':>4} {total['tp']:>3} {total['fp']:>3} {total['fn']:>3}")
print()
print("补了脚的元件：")
for item in data["components"]:
    if item.get("added"):
        print(f"  case {item['case']:>2} {item['key']:6s} model={item['model']!r:22s} "
              f"pins={item['pins']} added={item['added']}")
