"""Would growing a passive box to its own pin tips fix the case 1/11/12/13 bbox misses?"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, r"C:/Users/Pc/Desktop/新建文件夹/pcb_parser_competition/scripts")
from platform_metrics import COMPONENT_TOLERANCE_PX, _box_error  # noqa: E402

GT = Path(r"C:/Users/Pc/Desktop/新建文件夹/赛题六公开数据集/赛题六公开数据集/open_GT/open_GT")
PRED = Path(r"D:/pcb_open_gt_v17/final_results")
TARGETS = {"1": ["LED1", "LED2", "LED4", "R12"],
           "11": ["LED1"],
           "12": ["L1"],
           "13": ["C1", "C5", "C6", "C8", "R2", "R3", "R4", "L1"]}

for case, keys in TARGETS.items():
    truth = json.loads((GT / case / f"{case}.json").read_text(encoding="utf-8"))
    pred = json.loads((PRED / f"{case}_Pred.json").read_text(encoding="utf-8"))
    comps, pins = pred["components"], pred.get("pins") or {}
    print(f"=== case {case} ===")
    for key in keys:
        ref = truth["components"].get(key, {})
        comp = comps.get(key)
        if not comp:
            continue
        ours, gt = comp["bbox"], ref.get("bbox")
        tip_block = pins.get(key) or {}
        points = []
        for entry in (tip_block.values() if isinstance(tip_block, dict) else tip_block):
            pt = (entry or {}).get("point") or {}
            if pt:
                points.append((float(pt["x"]), float(pt["y"])))
        base_err = _box_error(ours, gt)
        if len(points) >= 2:
            xs = [p[0] for p in points] + [ours[0], ours[2]]
            ys = [p[1] for p in points] + [ours[1], ours[3]]
            grown = [min(xs), min(ys), max(xs), max(ys)]
            grown_err = _box_error(grown, gt)
        else:
            grown, grown_err = None, None
        print(f"  {key:<6} ours={[round(v) for v in ours]} wh=({round(ours[2]-ours[0])},{round(ours[3]-ours[1])}) "
              f"GT={[round(v) for v in gt]} wh=({round(gt[2]-gt[0])},{round(gt[3]-gt[1])}) "
              f"err {base_err:.1f}" + (f" | tips={len(points)} grown_err {grown_err:.1f}" if grown_err is not None
                                       else f" | tips={len(points)}"))
