"""Why the canonical box breaks the big-scale pages: compare detection vs GT vs canonical."""
from __future__ import annotations

import json
from pathlib import Path

GT = Path(r"C:/Users/Pc/Desktop/新建文件夹/赛题六公开数据集/赛题六公开数据集/open_GT/open_GT")
PRED = Path(r"D:/pcb_open_gt_v17/final_results")
RAW = Path(r"D:/pcb_open_gt_v17/component_results")
CFG = json.loads(Path(r"D:/pcb5_submission/configs/bbox_convention_v1.json").read_text(encoding="utf-8"))
CANON = CFG["canonical"]


def wh(box):
    return (round(box[2] - box[0], 1), round(box[3] - box[1], 1))


def err(a, b):
    return max(abs(x - y) for x, y in zip(a, b))


for case, keys in {"13": ["R1", "R2", "C1", "C5", "L1"], "1": ["R12", "LED1", "LED2"], "12": ["L1"]}.items():
    truth = json.loads((GT / case / f"{case}.json").read_text(encoding="utf-8"))["components"]
    pred = json.loads((PRED / f"{case}_Pred.json").read_text(encoding="utf-8"))["components"]
    raw = json.loads((RAW / f"{case}_Pred.json").read_text(encoding="utf-8"))["components"]
    print(f"=== case {case} ===")
    for key in keys:
        ref, out = truth.get(key), pred.get(key)
        det = raw.get(key)
        if not ref or not out or not det:
            continue
        canon = CANON.get({"led": "led", "resistor": "resistor", "capacitor": "capacitor",
                           "inductor": "inductor"}.get(_kind := _kind if (_kind := None) else "", ""), {})
        print(f"  {key:<5} detection {wh(det['bbox'])!s:<14} exported {wh(out['bbox'])!s:<14} "
              f"GT {wh(ref['bbox'])!s:<14} err(det)={err(det['bbox'], ref['bbox']):5.1f}  "
              f"err(exported)={err(out['bbox'], ref['bbox']):5.1f}")

