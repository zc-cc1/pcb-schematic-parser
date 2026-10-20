"""For value-less power symbols, list every nearby OCR text and which gate it fails."""
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, r"D:\pcb5_submission")
from component_export import GROUND, SUPPLY, center, gap  # noqa: E402

LOGS = Path(r"D:/pcb_open_gt_v15/logs")
PRED = Path(r"D:/pcb_open_gt_v15/final_results")

audit = json.loads((LOGS / "1_component_export_audit.json").read_text(encoding="utf-8"))
blank = [row for row in audit["added"] if not row["value"]]
print(f"case 1: {len(blank)} power symbols without a value")

ocr = json.loads((LOGS / "1_ocr_texts.json").read_text(encoding="utf-8"))
width, height = ocr["size"]
pred = json.loads((PRED / "1_Pred.json").read_text(encoding="utf-8"))["components"]

# group raw observations the same way observed_captions() does, without any threshold
obs_rows = []
for row in ocr["texts"]:
    for o in [row] + row.get("observations", []):
        text = str(o.get("text", "")).strip()
        obs_rows.append({"text": text, "conf": float(o.get("confidence", 0)),
                         "view": o.get("view"), "bbox": o.get("bbox", row.get("bbox"))})

for row in blank:
    cid = row["component_id"]
    comp = pred.get(cid)
    if not comp:
        print(f"  {cid}: not in final results")
        continue
    bx1, by1, bx2, by2 = comp["bbox"]
    box = [bx1, height - by2, bx2, height - by1]
    sx, sy = center(box)
    near = []
    for o in obs_rows:
        if not o["bbox"]:
            continue
        if gap(box, o["bbox"]) > 36:
            continue
        near.append(o)
    matches = [o for o in near if GROUND.fullmatch(o["text"]) or SUPPLY.fullmatch(o["text"])]
    best = {}
    for o in matches:
        key = o["text"]
        info = best.setdefault(key, {"views": set(), "max_conf": 0})
        info["views"].add(o["view"])
        info["max_conf"] = max(info["max_conf"], o["conf"])
    print(f"  {cid} ({row['type']}) box={[round(v) for v in box]}")
    if not best:
        print(f"     近处 {len(near)} 个文本，无一匹配 GND/供电正则: "
              f"{[o['text'] for o in near][:10]}")
    for text, info in best.items():
        print(f"     候选 {text!r}: views={sorted(info['views'])} max_conf={info['max_conf']:.3f}")
