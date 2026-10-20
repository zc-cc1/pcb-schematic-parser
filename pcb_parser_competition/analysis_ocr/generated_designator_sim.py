"""Simulate the only remaining option on pages with no printed designators:
assign 1..n per prefix in a fixed geometric order and count how many keys land.

This is measured against the published answer, so it is an upper bound on what a
generated designator could earn there (it uses GT counts, not our detections).
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

DATASET = Path(r"C:\Users\Pc\Desktop\新建文件夹\Dataset300")
PUBLIC = Path(r"C:\Users\Pc\Desktop\新建文件夹\赛题六公开数据集\赛题六公开数据集\200_train_cases")
REF = re.compile(r"^(?:IC|SW|TP|FB|FET|LED|JP|[RCLDQUXYJPFS])\d+$", re.I)
SPLIT = re.compile(r"^([A-Za-z]+)(\d+)$")

ORDERS = {
    "row-major (top→bottom, left→right)": lambda v: (-v["bbox"][3], v["bbox"][0]),
    "column-major (left→right, top→bottom)": lambda v: (v["bbox"][0], -v["bbox"][3]),
    "row-major bottom-up": lambda v: (v["bbox"][1], v["bbox"][0]),
    "diagonal": lambda v: (-v["bbox"][3] + v["bbox"][0],),
}


def printed_ref_count(case):
    std = next((DATASET / case).glob("*_CVJsonStd.json"), None)
    if std is None:
        return None
    doc = json.loads(std.read_text(encoding="utf-8"))
    return sum(1 for item in (doc.get("texts") or {}).values()
               if REF.match(str(item.get("content") or "").strip()))


def main():
    pages = []
    for folder in sorted(PUBLIC.iterdir()):
        if not folder.is_dir():
            continue
        if printed_ref_count(folder.name) != 0:
            continue
        f = next(folder.glob("*_target_named.json"), None)
        if f is None:
            continue
        gt = json.loads(f.read_text(encoding="utf-8"))
        comps = [(k, v) for k, v in gt["components"].items()
                 if str(v.get("type")) in {"r", "c", "l", "d", "led"} and SPLIT.match(k)]
        if len(comps) >= 4:
            pages.append((folder.name, comps))

    print(f"pages simulated: {len(pages)}, GT passives: {sum(len(c) for _, c in pages)}")
    for label, key in ORDERS.items():
        hits = total = 0
        per_page = []
        for _, comps in pages:
            groups = defaultdict(list)
            for k, v in comps:
                groups[SPLIT.match(k)[1].upper()].append((k, v))
            page_hits = 0
            for prefix, items in groups.items():
                ordered = sorted(items, key=lambda kv: key(kv[1]))
                for index, (true_key, _) in enumerate(ordered, start=1):
                    total += 1
                    if true_key.upper() == f"{prefix}{index}".upper():
                        hits += 1
                        page_hits += 1
            per_page.append(page_hits / len(comps))
        print(f"  {label:38s} {hits}/{total} = {hits/total:.1%} "
              f"(per-page mean {sum(per_page)/len(per_page):.1%})")

    # what a per-prefix count mismatch would cost: we only see our own detections
    print("\nper-page detail (row-major):")
    for case, comps in pages:
        groups = defaultdict(list)
        for k, v in comps:
            groups[SPLIT.match(k)[1].upper()].append((k, v))
        hits = 0
        for prefix, items in groups.items():
            ordered = sorted(items, key=lambda kv: ORDERS["row-major (top→bottom, left→right)"](kv[1]))
            for index, (true_key, _) in enumerate(ordered, start=1):
                hits += true_key.upper() == f"{prefix}{index}".upper()
        print(f"  {case}: {hits}/{len(comps)}  "
              f"groups={{{', '.join(f'{p}:{len(v)}' for p, v in sorted(groups.items()))}}}")


if __name__ == "__main__":
    sys.exit(main())
