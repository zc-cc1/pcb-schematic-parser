"""Offline check: how many power captions does the relaxed gate admit, per case?"""
import importlib
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, r"D:\pcb5_submission")
import component_export as ce  # noqa: E402

LOG_DIRS = {"v15": Path(r"D:/pcb_open_gt_v15/logs")}


def old_captions(texts):
    groups = []
    for row in texts:
        for obs in [row] + row.get("observations", []):
            text = str(obs.get("text", "")).strip()
            conf = float(obs.get("confidence", 0))
            box = obs.get("bbox", row.get("bbox"))
            view = obs.get("view", row.get("view"))
            if conf < .97 or not view or not box or not (ce.GROUND.fullmatch(text) or ce.SUPPLY.fullmatch(text)):
                continue
            found = next((g for g in groups if all(ce.overlap(box, b) >= .4 for b in g["boxes"])), None)
            if found is None:
                found = {"boxes": [], "readings": defaultdict(set), "observations": []}
                groups.append(found)
            found["boxes"].append(box)
            found["readings"][text].add(view)
            found["observations"].append((conf, text, box))
    out = []
    for group in groups:
        supported = [t for t, views in group["readings"].items() if len(views) >= 2]
        if len(supported) != 1:
            continue
        out.append(supported[0])
    return out


print(f"{'case':>4} {'old captions':>13} {'new captions':>13}")
total_old = total_new = 0
for path in sorted(LOG_DIRS["v15"].glob("*_ocr_texts.json"), key=lambda p: int(p.stem.split("_")[0])):
    case = path.stem.split("_")[0]
    texts = json.loads(path.read_text(encoding="utf-8"))["texts"]
    old = old_captions(texts)
    new = ce.observed_captions(texts)
    total_old += len(old)
    total_new += len(new)
    print(f"{case:>4} {len(old):>13} {len(new):>13}   {sorted({c["text"] for c in new} - set(old))}")
print(f"{'ALL':>4} {total_old:>13} {total_new:>13}")

