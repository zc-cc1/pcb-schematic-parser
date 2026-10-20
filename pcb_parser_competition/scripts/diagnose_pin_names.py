"""Show concrete pin-name failures inside matched components.

For each matched component, print the reference pins and our exported pins side by
side so the pattern behind a wrong name is visible (wrong number, wrong order,
missing pin, misread text).
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

GT = Path(r"C:/Users/Pc/Desktop/新建文件夹/赛题六公开数据集/赛题六公开数据集/10_GTcase")
BASE = Path(r"D:/pcb_unet_v2/golden_eval")
LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 8


def nearest_component(target_key, truth_pins, pred_pins):
    """Map reference component -> our component by pin-position overlap."""
    best = None
    for candidate, pins in pred_pins.items():
        shared = 0
        for reference in (truth_pins.get(target_key) or {}).values():
            for pin in pins.values():
                if abs(pin["point"]["x"] - reference["point"]["x"]) <= 20 and \
                        abs(pin["point"]["y"] - reference["point"]["y"]) <= 20:
                    shared += 1
                    break
        if best is None or shared > best[0]:
            best = (shared, candidate)
    return best[1] if best and best[0] else None


def main():
    shown = 0
    for folder in sorted(GT.iterdir()):
        if not folder.is_dir():
            continue
        case = folder.name.split("-")[0]
        submission = BASE / case / "wire_graph" / "partial_submission.json"
        if not submission.exists():
            continue
        truth = json.loads(next(folder.glob("*_target.json")).read_text(encoding="utf-8"))
        pred = json.loads(submission.read_text(encoding="utf-8"))
        truth_pins = truth.get("pins") or {}
        pred_pins = pred.get("pins") or {}
        for key, pins in truth_pins.items():
            ours_key = nearest_component(key, truth_pins, pred_pins)
            if ours_key is None:
                continue
            ours = pred_pins[ours_key]
            rows = []
            for pin_key, pin in pins.items():
                match = None
                for our_key, our_pin in ours.items():
                    distance = ((our_pin["point"]["x"] - pin["point"]["x"]) ** 2 +
                                (our_pin["point"]["y"] - pin["point"]["y"]) ** 2) ** 0.5
                    if match is None or distance < match[0]:
                        match = (distance, our_key, our_pin)
                rows.append((pin_key, pin.get("pinname"), match))
            wrong = [r for r in rows
                     if r[2] is None or r[2][0] > 20 or str(r[2][2].get("pinname")) != str(r[1])]
            if not wrong or shown >= LIMIT:
                continue
            shown += 1
            kind = (truth.get("components", {}).get(key) or {}).get("type")
            print(f"== {case}  reference {key} ({kind})  -> ours {ours_key}")
            for pin_key, expected, match in rows:
                if match is None:
                    print(f"   {pin_key:<8} ref name {expected!r:<10} -> no nearby pin")
                    continue
                distance, our_key, our_pin = match
                flag = "" if str(our_pin.get("pinname")) == str(expected) else "  <-- wrong"
                print(f"   {pin_key:<8} ref name {expected!r:<10} ours {our_key:<8} "
                      f"name {our_pin.get('pinname')!r:<10} d={distance:5.1f}{flag}")
            print()
        if shown >= LIMIT:
            break
    return 0


if __name__ == "__main__":
    sys.exit(main())
