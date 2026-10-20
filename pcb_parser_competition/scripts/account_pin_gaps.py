"""Where every reference IC pin goes: a full accounting of the pin score.

The platform rule is IC-only scope, matched by `key` + `pin_<number>`, then `pinname` exact
and a point within 5 px.  This walks every reference IC pin and says which single thing is
missing for the ones that do not score:

  A   component key missing            no designator for that part
  B1  no pin detected                  nothing within 35 px of the reference tip
  B2  pin detected, no text at all
  B3  pin detected, name matches, number missing
  B4  pin detected with the right number, but the number never reached the export
  B5  pin detected but owned by no component
  B6  pin detected, but the text around it belongs to a different pin
  C   exported, point beyond 5 px
  D   exported, point fine, name differs

Candidates are read from the earliest pin stage (image space) and the reference tip is flipped
into image space with the case's own height.

    python scripts/account_pin_gaps.py --predictions D:/pcb_open_gt_v20/final_results
"""
import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

GT_ROOT = Path(r"C:/Users/Pc/Desktop/新建文件夹/赛题六公开数据集/赛题六公开数据集/open_GT/open_GT")
WORK = Path(r"D:/pcb_open_gt_v5/work")
LIMIT = 35.0
BOX = {"box", "ic"}


def case_map(gt_root, work_root):
    hashes = {hashlib.sha256(next(folder.glob("*.png")).read_bytes()).hexdigest(): folder.name
              for folder in gt_root.iterdir() if folder.is_dir()}
    table = {}
    for work in work_root.glob("*"):
        for path in work.rglob("summary.json"):
            try:
                image = json.loads(path.read_text(encoding="utf-8")).get("image_sha256")
            except Exception:
                continue
            if image in hashes:
                table.setdefault(hashes[image], work.name)
            break
    return table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--gt-root", default=str(GT_ROOT))
    parser.add_argument("--work", default=str(WORK))
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    gt_root = Path(args.gt_root)
    predictions = Path(args.predictions)
    table = case_map(gt_root, Path(args.work))

    buckets = Counter()
    per_component = defaultdict(Counter)
    for case, work_name in sorted(table.items(), key=lambda kv: int(kv[0])):
        truth = json.loads(next((gt_root / case).glob("*.json")).read_text(encoding="utf-8"))
        pred = json.loads((predictions / f"{case}_Pred.json").read_text(encoding="utf-8"))
        candidates = sorted((Path(args.work) / work_name).rglob("candidate_pin_text_result.json"))
        pins = json.loads(candidates[0].read_text(encoding="utf-8"))["pin_candidates"] if candidates else []
        with Image.open(next((gt_root / case).glob("*.png"))) as handle:
            height = handle.height
        for key, reference_pins in (truth.get("pins") or {}).items():
            if str((truth["components"].get(key) or {}).get("type")) not in BOX:
                continue
            ours = (pred.get("pins") or {}).get(key)
            for pin_key, reference in reference_pins.items():
                label = None
                if key not in (pred.get("components") or {}):
                    label = "A_component_key_missing"
                else:
                    candidate = (ours or {}).get(pin_key)
                    if candidate is None:
                        target = (reference["point"]["x"], height - reference["point"]["y"])
                        want = pin_key.replace("pin_", "")
                        want_name = str(reference.get("pinname") or "")
                        nearby = sorted(
                            ((math.hypot(pin["point"][0] - target[0], pin["point"][1] - target[1]), pin)
                             for pin in pins),
                            key=lambda item: item[0])
                        nearby = [item for item in nearby if item[0] <= LIMIT]
                        if not nearby:
                            label = "B1_no_pin_detected"
                        elif any(str(pin.get("number") or "").strip().split("_")[0] == want
                                 for _d, pin in nearby):
                            label = "B4_number_not_exported"
                        elif any(want_name and str(pin.get("pinname") or "").strip() == want_name
                                 for _d, pin in nearby):
                            label = "B3_name_only_no_number"
                        else:
                            pin = nearby[0][1]
                            if pin.get("owner_status") != "assigned":
                                label = "B5_pin_ownerless"
                            elif str(pin.get("pinname") or "").strip() or str(pin.get("number") or "").strip():
                                label = "B6_other_pin_text"
                            else:
                                label = "B2_no_text_on_pin"
                    elif str(candidate.get("pinname")) != str(reference.get("pinname")):
                        label = "D_name_differs"
                    elif math.hypot(candidate["point"]["x"] - reference["point"]["x"],
                                    candidate["point"]["y"] - reference["point"]["y"]) > 5.0:
                        label = "C_point_too_far"
                    else:
                        label = "TP"
                buckets[label] += 1
                per_component[(case, key)][label] += 1

    total = sum(buckets.values())
    print(f"reference IC pins: {total}")
    for label in sorted(buckets, key=lambda name: (-buckets[name], name)):
        print(f"  {label:28s} {buckets[label]:4d}")
    print()
    print("where they are:")
    for (case, key), counts in sorted(per_component.items(), key=lambda kv: -sum(kv[1].values())):
        if counts.get("TP", 0) == sum(counts.values()):
            continue
        detail = ", ".join(f"{name.split('_')[0]}={count}" for name, count in counts.items() if name != "TP")
        print(f"  case {case:>2} {key:9s} TP {counts.get('TP', 0):2d} of {sum(counts.values()):2d}   {detail}")


if __name__ == "__main__":
    main()
