import json
from pathlib import Path

GT = Path(r"C:/Users/Pc/Desktop/新建文件夹/赛题六公开数据集/赛题六公开数据集/open_GT/open_GT")

for case in [str(i) for i in range(1, 14)]:
    j = json.loads((GT / case / f"{case}.json").read_text(encoding="utf-8"))
    d = {k: v for k, v in j["components"].items() if v.get("type") in {"d", "led"}}
    r = [(k, v.get("value")) for k, v in j["components"].items() if v.get("type") in {"r", "c", "l"}]
    print(f"=== case {case} ===")
    for k, v in d.items():
        print(f"   {v.get('type'):<4} {k:<10} Name={v.get('Name')!r}")
    numeric = [(k, val) for k, val in r if isinstance(val, str) and val.replace(".", "").isdigit()]
    print(f"   passive values: {[v for _, v in r][:8]}")
    if numeric:
        print(f"   NUMERIC-only values: {numeric}")
