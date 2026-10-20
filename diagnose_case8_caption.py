"""Why did the VDD_1.9V caption vanish on case 8?"""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, r"D:\pcb5_submission")
import component_export as ce  # noqa: E402

ocr = json.loads(Path(r"D:/pcb_open_gt_v16/logs/8_ocr_texts.json").read_text(encoding="utf-8"))
groups = []
for row in ocr["texts"]:
    for obs in [row] + row.get("observations", []):
        text = str(obs.get("text", "")).strip()
        conf = float(obs.get("confidence", 0))
        box = obs.get("bbox", row.get("bbox"))
        view = obs.get("view", row.get("view"))
        if conf < .90 or not view or not box or not (ce.GROUND.fullmatch(text) or ce.SUPPLY.fullmatch(text)):
            continue
        found = next((g for g in groups if all(ce.overlap(box, b) >= .4 for b in g["boxes"])), None)
        if found is None:
            found = {"boxes": [], "observations": []}
            groups.append(found)
        found["boxes"].append(box)
        found["observations"].append((conf, text, box, view))

for i, g in enumerate(groups):
    votes = {}
    for conf, text, box, view in g["observations"]:
        v = votes.setdefault(text.upper(), {"views": set(), "conf": 0.0, "spellings": Counter()})
        v["views"].add(view)
        v["conf"] = max(v["conf"], conf)
        v["spellings"][text] += 1
    
    print(f"group {i:>2}: " + " | ".join(
        f"{k} views={len(v['views'])} conf={v['conf']:.2f} spellings={dict(v['spellings'])}"
        for k, v in votes.items()) + flag)

