"""Why do the 39 "name yes, number no" pins still have no number?

For every reference IC pin that is not in the submission, look for
  * a strip-audit ADOPTION  -> it should have been exported (different bug)
  * a strip-audit REJECTION -> the reader saw something and the rule refused it
  * nothing                 -> the pin was never measured (scope filter / not an IC / no crop)
"""
import json
import math
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import hashlib
from PIL import Image

GT = Path(r"C:/Users/Pc/Desktop/新建文件夹/赛题六公开数据集/赛题六公开数据集/open_GT/open_GT")
WORK = Path(r"D:/pcb_decouple/work")
PRED = Path(r"D:/pcb_decouple/final_results")
BOX = {"box", "ic"}

hashes = {hashlib.sha256(next(f.glob("*.png")).read_bytes()).hexdigest(): f.name
          for f in (p for p in GT.iterdir() if p.is_dir())}
case_of = {}
for work in WORK.glob("*"):
    for path in work.rglob("summary.json"):
        try:
            image = json.loads(path.read_text(encoding="utf-8")).get("image_sha256")
        except Exception:
            continue
        if image in hashes:
            case_of[hashes[image]] = work.name
        break

stats = Counter()
rows = []
for case in sorted(case_of, key=int):
    truth = json.loads(next((GT / case).glob("*.json")).read_text(encoding="utf-8"))
    pred = json.loads((PRED / f"{case}_Pred.json").read_text(encoding="utf-8"))
    audits = sorted((WORK / case_of[case]).rglob("pin_number_strip_audit.json"))
    adopted, rejected = [], []
    for path in audits:
        payload = json.loads(path.read_text(encoding="utf-8"))
        adopted += payload.get("adopted") or []
        rejected += payload.get("rejected") or []
    with Image.open(next((GT / case).glob("*.png"))) as handle:
        height = handle.height
    # the strip audit speaks in candidate ids; positions come from the same stage's pin doc
    pin_docs = sorted((WORK / case_of[case]).rglob("engineering_pin_text_result.json"))
    positions = {}
    if pin_docs:
        for pin in json.loads(pin_docs[-1].read_text(encoding="utf-8"))["pin_candidates"]:
            positions[pin["candidate_id"]] = (float(pin["point"][0]), float(pin["point"][1]))
    def locate(candidate_id):
        return positions.get(candidate_id)

    for key, reference_pins in (truth.get("pins") or {}).items():
        if str((truth["components"].get(key) or {}).get("type")) not in BOX:
            continue
        ours = (pred.get("pins") or {}).get(key) or {}
        for pin_key, reference in reference_pins.items():
            if pin_key in ours:
                continue
            target = (reference["point"]["x"], height - reference["point"]["y"])
            hit = None
            for row in adopted + rejected:
                xy = locate(row["candidate_id"])
                if xy and math.hypot(xy[0] - target[0], xy[1] - target[1]) <= 25:
                    hit = row
                    break
            if hit is None:
                stats["never measured (scope filter / no crop)"] += 1
                kind = "never_measured"
            elif "number" in hit and hit.get("number") and "reason" not in hit:
                stats["adopted but still absent"] += 1
                kind = "adopted_missing"
            else:
                stats[f"rejected: {hit.get('reason')}"] += 1
                kind = f"rejected:{hit.get('reason')}"
            rows.append((case, key, pin_key, reference.get("pinname"), kind))

print(dict(stats), "total", sum(stats.values()))
print()
for row in rows:
    print(f"  case {row[0]:>2} {row[1]:8s} {row[2]:8s} name={str(row[3])[:12]:12s} {row[4]}")
