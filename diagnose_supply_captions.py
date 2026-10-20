"""What OCR evidence exists near the value-less supply symbols?"""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, r"D:\pcb5_submission")
from component_export import GROUND, SUPPLY, center, gap  # noqa: E402

LOGS = Path(r"D:/pcb_open_gt_v15/logs")
PRED = Path(r"D:/pcb_open_gt_v15/final_results")

for case in ["2", "3", "5", "9", "10", "13"]:
    audit = json.loads((LOGS / f"{case}_component_export_audit.json").read_text(encoding="utf-8"))
    blank = [r for r in audit["added"] if not r["value"] and r["type"] != "gnd"]
    if not blank:
        continue
    ocr = json.loads((LOGS / f"{case}_ocr_texts.json").read_text(encoding="utf-8"))
    _, height = ocr["size"]
    pred = json.loads((PRED / f"{case}_Pred.json").read_text(encoding="utf-8"))["components"]
    rows = []
    for row in ocr["texts"]:
        for o in [row] + row.get("observations", []):
            rows.append({"text": str(o.get("text", "")).strip(), "conf": float(o.get("confidence", 0)),
                         "view": o.get("view"), "bbox": o.get("bbox", row.get("bbox"))})
    print(f"=== case {case}: {len(blank)} value-less supply symbols ===")
    for row in blank:
        comp = pred.get(row["component_id"])
        if not comp:
            continue
        bx1, by1, bx2, by2 = comp["bbox"]
        box = [bx1, height - by2, bx2, height - by1]
        found = {}
        for o in rows:
            if not o["bbox"] or gap(box, o["bbox"]) > 36:
                continue
            if not (GROUND.fullmatch(o["text"]) or SUPPLY.fullmatch(o["text"])):
                continue
            info = found.setdefault(o["text"], {"views": set(), "conf": 0})
            info["views"].add(o["view"])
            info["conf"] = max(info["conf"], o["conf"])
        desc = ", ".join(f"{t!r} v={len(i['views'])} c={i['conf']:.2f}" for t, i in found.items()) or "无匹配文本"
        print(f"   {row['component_id']}: {desc}")
