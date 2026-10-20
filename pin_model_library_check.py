"""裁切 OCR 读到的型号，拿去查库：在不在、脚数对不对、编号+名称对不对。"""
import json
from pathlib import Path

GT = Path(r"C:/Users/Pc/Desktop/新建文件夹/赛题六公开数据集/赛题六公开数据集/open_GT/open_GT")
library = json.loads(Path(r"C:/Users/Pc/Desktop/新建文件夹/官方资料/BoxICLibrary.json").read_text(encoding="utf-8"))

# 裁切 OCR 的实际输出（含两个只读到 "78M1" 的）
READ = {("1", "USB1"): "5V2A in", ("1", "USB2"): "5V2A out", ("4", "U2"): "AMS1117-3.3",
        ("6", "U1"): "78M1", ("6", "U3"): "78M1", ("11", "U1"): "TP4333",
        ("11", "USB1"): "TYPE-C 6P(073)", ("11", "USB2"): "AF 90° WJDG", ("13", "H1"): "H1"}


def lookup(name):
    if name in library:
        return name
    lowered = {str(k).lower(): k for k in library}
    return lowered.get(str(name).lower())


def table_of(model):
    pins = {}
    for pin in library[model]["pins"]:
        number = str(pin.get("number") or "").strip()
        if number:
            pins.setdefault(number, str(pin.get("name") or "").strip())
    return pins


for (case, component), read in READ.items():
    truth = json.loads((GT / case / f"{case}.json").read_text(encoding="utf-8"))
    gt = {k.replace("pin_", ""): str(v.get("pinname") or "").strip() for k, v in truth["pins"][component].items()}
    model = lookup(read)
    print(f"case {case:>2} {component:6s} OCR读到 {read!r:20s}", end="  ")
    if not model:
        print("库: 没有 ->", "查不到")
        continue
    table = table_of(model)
    hit = [n for n in gt if n in table]
    same = [n for n in hit if table[n].upper() == gt[n].upper()]
    print(f"库: {model!r} {len(table)} 脚 -> 编号覆盖 {len(hit)}/{len(gt)}，名称一致 {len(same)}")
