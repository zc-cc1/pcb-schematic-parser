"""对比修复前后 / GT：某个元件的引脚键、名称、点距。"""
import json
import math
import sys
from pathlib import Path

GT = Path(r"C:/Users/Pc/Desktop/新建文件夹/赛题六公开数据集/赛题六公开数据集/open_GT/open_GT")

case = sys.argv[1] if len(sys.argv) > 1 else "11"
component = sys.argv[2] if len(sys.argv) > 2 else "USB1"
base = Path(sys.argv[3] if len(sys.argv) > 3 else r"D:/pcb_open_gt_v33angle120/final_results")
fixed = Path(sys.argv[4] if len(sys.argv) > 4 else r"C:/Users/Pc/Desktop/新建文件夹/runs/ab_all/final_results")

truth = json.loads((GT / case / f"{case}.json").read_text(encoding="utf-8"))
gt = truth["pins"][component]
before = json.loads((base / f"{case}_Pred.json").read_text(encoding="utf-8"))["pins"].get(component) or {}
after = json.loads((fixed / f"{case}_Pred.json").read_text(encoding="utf-8"))["pins"].get(component) or {}

print(f"case {case} {component}: GT {len(gt)} 脚, 修复前 {len(before)}, 修复后 {len(after)}")
for key, ref in gt.items():
    gx, gy = ref["point"]["x"], ref["point"]["y"]
    def info(pins):
        pin = pins.get(key)
        if not pin:
            return "缺失"
        d = math.hypot(pin["point"]["x"] - gx, pin["point"]["y"] - gy)
        return f"{str(pin.get('pinname'))[:12]:12s} d={d:6.1f}"
    print(f"  {key:8s} GT名={str(ref.get('pinname'))[:12]:12s} 之前: {info(before):28s} 之后: {info(after)}")
