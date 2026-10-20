"""把这几个"读不到"的元件在 GT 里的原始注释打出来。"""
import json
from pathlib import Path

GT = Path(r"C:/Users/Pc/Desktop/新建文件夹/赛题六公开数据集/赛题六公开数据集/open_GT/open_GT")

TARGETS = [("1", "USB1"), ("1", "USB2"), ("13", "H1"), ("11", "USB2"), ("6", "U1")]

for case, component in TARGETS:
    truth = json.loads((GT / case / f"{case}.json").read_text(encoding="utf-8"))
    comp = truth["components"][component]
    pins = truth["pins"][component]
    print(f"===== case {case} / {component} =====")
    print("components[%r] = %s" % (component, json.dumps(comp, ensure_ascii=False)))
    print(f'pins[%r] = {len(pins)} 条' % component)
    for key, pin in pins.items():
        print("   ", json.dumps({key: pin}, ensure_ascii=False))
    print()
