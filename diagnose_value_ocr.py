"""Was the missing passive value actually read by OCR, or never seen?"""
from __future__ import annotations

import json
import re
from pathlib import Path

LOGS = Path(r"D:/pcb_open_gt_v17/logs")
PRED = Path(r"D:/pcb_open_gt_v17/final_results")
CASES = {
    "1": [("R13", "5k1"), ("R14", "5k1")],
    "3": [("L1", "3.3uH")],
    "5": [("L1", "10uH"), ("L2", "10uH")],
    "6": [("C11", "100nF")],
    "7": [("L1", "10uH")],
    "9": [("L1", "10uH")],
    "10": [("L1", "10uH")],
    "12": [("C2", "10-100uF"), ("C3", "10-100uF")],
}


def gap(a, b):
    return ((max(a[0] - b[2], b[0] - a[2], 0)) ** 2 + (max(a[1] - b[3], b[1] - a[3], 0)) ** 2) ** .5


for case, targets in CASES.items():
    ocr = json.loads((LOGS / f"{case}_ocr_texts.json").read_text(encoding="utf-8"))
    _, height = ocr["size"]
    pred = json.loads((PRED / f"{case}_Pred.json").read_text(encoding="utf-8"))["components"]
    rows = []
    for row in ocr["texts"]:
        for o in [row] + row.get("observations", []):
            rows.append({"text": str(o.get("text", "")).strip(), "conf": float(o.get("confidence", 0)),
                         "bbox": o.get("bbox", row.get("bbox"))})
    print(f"=== case {case} ===")
    for key, expected in targets:
        comp = pred.get(key)
        if not comp:
            print(f"  {key}: not exported")
            continue
        bx1, bz1, bx2, bz2 = comp["bbox"]
        box = [bx1, height - bz2, bx2, height - bz1]
        near = sorted(((gap(box, o["bbox"]), i, o) for i, o in enumerate(rows) if o["bbox"] and gap(box, o["bbox"]) <= 60))
        digit = re.compile(r"\d")
        cands = [(round(g, 1), o["text"], round(o["conf"], 2)) for g, _i, o in near if digit.search(o["text"])][:6]
        print(f"  {key} (want {expected!r}) 近处带数字文本: {cands}")

