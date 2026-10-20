"""读不到编号的器件，能不能从官方元件库补回引脚编号/名称。

对每个有漏脚的元件：
  1. 拿我们预测里的 Name（型号）
  2. 在 BoxICLibrary 里查（先精确、再宽松）
  3. 库里的 (编号, 名称) 去重后，和 GT 的引脚表逐项比对
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

GT = Path(r"C:/Users/Pc/Desktop/新建文件夹/赛题六公开数据集/赛题六公开数据集/open_GT/open_GT")
PRED = Path(r"D:/pcb_open_gt_v33angle120/final_results")
LIB = Path(r"C:/Users/Pc/Desktop/新建文件夹/官方资料/BoxICLibrary.json")

sys.path.insert(0, r"C:/Users/Pc/Desktop/新建文件夹/pcb_parser_competition/scripts")
from platform_metrics import is_ic  # noqa: E402

library = json.loads(LIB.read_text(encoding="utf-8"))
INDEX = {str(k).strip().upper(): k for k in library}


def normalise(text):
    text = str(text or "").strip().upper()
    text = re.sub(r"[\s\-_/]+", "", text)
    return text


FUZZY = defaultdict(list)
for key in library:
    FUZZY[normalise(key)].append(key)


def lookup(name):
    raw = str(name or "").strip()
    if not raw:
        return None, "空名称"
    if raw in library:
        return raw, "精确"
    if raw.upper() in INDEX:
        return INDEX[raw.upper()], "大小写"
    key = normalise(raw)
    if key in FUZZY:
        return FUZZY[key][0], "去符号"
    # 去掉常见后缀再试（-Q1 / -TR / /T 等）
    stripped = re.sub(r"(-Q\d|-[A-Z]{1,3}\d?|/T|TR)$", "", key)
    if stripped in FUZZY:
        return FUZZY[stripped][0], "去后缀"
    return None, "查不到"


def unique_pin_table(model):
    table = {}
    for pin in library[model]["pins"]:
        number = str(pin.get("number") or "").strip()
        name = str(pin.get("name") or "").strip()
        if not number:
            continue
        table.setdefault(number, name)
    return table


def main():
    rows = json.loads(Path(r"C:/Users/Pc/Desktop/新建文件夹/pin_miss_v33_corrected.json").read_text(encoding="utf-8"))["rows"]
    by_comp = {}
    for row in rows:
        by_comp.setdefault((row["case"], row["component"]), []).append(row)

    summary = {"hit": 0, "comp": 0, "covered": 0, "pins": 0, "name_ok": 0}
    print(f"{'case':>4} {'comp':6s} {'我们读到的型号':22s} {'GT':22s} {'查库':6s} {'库编号':>5s} {'覆盖漏脚':>7s}")
    for (case, component), group in sorted(by_comp.items(), key=lambda kv: (int(kv[0][0]), kv[0][1])):
        truth = json.loads((GT / case / f"{case}.json").read_text(encoding="utf-8"))
        pred = json.loads((PRED / f"{case}_Pred.json").read_text(encoding="utf-8"))
        our_name = str((pred.get("components", {}).get(component) or {}).get("Name") or "")
        gt_name = str(truth["components"][component].get("Name") or "")
        model, how = lookup(our_name)
        summary["comp"] += 1
        if not model:
            print(f"{case:>4} {component:6s} {our_name[:22]:22s} {gt_name[:22]:22s} {how:6s} {'-':>5s} {0:>7d}")
            continue
        summary["hit"] += 1
        table = unique_pin_table(model)
        covered = [row for row in group if row["key"].replace("pin_", "") in table]
        names_ok = sum(1 for row in covered
                       if table[row["key"].replace("pin_", "")].strip().upper() == str(row["name"]).strip().upper())
        summary["covered"] += len(covered)
        summary["pins"] += len(group)
        summary["name_ok"] += names_ok
        print(f"{case:>4} {component:6s} {our_name[:22]:22s} {gt_name[:22]:22s} {how:6s} {len(table):>5d} "
              f"{len(covered):>3d}/{len(group):<3d} 名称对 {names_ok}")

    print(f"\n有漏脚的元件 {summary['comp']} 个，库里查到 {summary['hit']} 个")
    print(f"库里能覆盖的漏脚 {summary['covered']} / {summary['pins']}，其中名称也对 {summary['name_ok']}")


if __name__ == "__main__":
    main()
