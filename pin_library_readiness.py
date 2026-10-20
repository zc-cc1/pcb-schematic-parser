"""量化"型号可信"的比例：多少 box 元件的 Name 真的读对了，多少退化成了位号。

这是走库的前提。顺带看 H1 这类排针能不能用"类型+脚数"在库里找到候选。
"""
import json
from pathlib import Path

GT = Path(r"C:/Users/Pc/Desktop/新建文件夹/赛题六公开数据集/赛题六公开数据集/open_GT/open_GT")
PRED = Path(r"D:/pcb_open_gt_v33angle120/final_results")
import sys
sys.path.insert(0, r"C:/Users/Pc/Desktop/新建文件夹/pcb_parser_competition/scripts")
from platform_metrics import is_ic  # noqa: E402

LIB = Path(r"C:/Users/Pc/Desktop/新建文件夹/官方资料/BoxICLibrary.json")
library = json.loads(LIB.read_text(encoding="utf-8"))

total = right = designator = empty = 0
rows = []
for folder in sorted(GT.iterdir(), key=lambda p: int(p.name)):
    case = folder.name
    truth = json.loads((folder / f"{case}.json").read_text(encoding="utf-8"))
    pred = json.loads((PRED / f"{case}_Pred.json").read_text(encoding="utf-8"))
    for key, comp in truth["components"].items():
        if not is_ic(comp.get("type")):
            continue
        total += 1
        our = (pred.get("components", {}).get(key) or {})
        name = str(our.get("Name") or "").strip()
        gt_name = str(comp.get("Name") or "").strip()
        if not name:
            empty += 1
            kind = "空"
        elif name.upper() == key.upper():
            designator += 1
            kind = "位号"
        elif name.lower() == gt_name.lower():
            right += 1
            kind = "正确"
        else:
            kind = "其它"
        rows.append((case, key, name, gt_name, kind))

print(f"box 元件 {total} 个：名称正确 {right}，写成了位号 {designator}，空 {empty}，其它 {total - right - designator - empty}")
print("\n写错/写空的：")
for case, key, name, gt_name, kind in rows:
    if kind in ("位号", "空", "其它"):
        flag = "在库里" if name in library else ""
        print(f"  case {case:>2} {key:6s} 我们={name[:18]:18s} GT={gt_name[:18]:18s} {kind:4s} {flag}")

print("\n排针候选（库里 1x3 / 3 脚）：")
for key in ("Header-Male-2.54_1x3", "HDR-M-2.54_1x3", "Header 4"):
    if key in library:
        pins = [(str(p.get("number")), str(p.get("name"))) for p in library[key]["pins"]]
        print(" ", key, pins[:6])
