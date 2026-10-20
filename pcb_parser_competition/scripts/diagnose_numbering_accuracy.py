"""Is the pin-numbering rule itself wrong, or is it only missing pins?

Restrict the check to components where we exported exactly as many pins as the
reference has: there the numbering order is the only thing that can be wrong.
"""
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from evaluate_golden_cases import GT_DIR  # noqa: E402

BASE = Path(r"D:/pcb_unet_v2/golden_eval")
TOLERANCE = 20.0


def nearest(candidate_key, truth_pins, pred_pins):
    best = None
    for key, pins in pred_pins.items():
        shared = 0
        for truth_pin in (truth_pins.get(candidate_key) or {}).values():
            for pin in pins.values():
                if abs(pin["point"]["x"] - truth_pin["point"]["x"]) <= TOLERANCE and \
                        abs(pin["point"]["y"] - truth_pin["point"]["y"]) <= TOLERANCE:
                    shared += 1
                    break
        if best is None or shared > best[0]:
            best = (shared, key)
    return best[1] if best and best[0] else None


def main():
    stats = collections.Counter()
    for folder in sorted(GT_DIR.iterdir()):
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
            ours_key = nearest(key, truth_pins, pred_pins)
            if ours_key is None:
                continue
            ours = pred_pins[ours_key]
            if len(ours) != len(pins):
                stats["different pin count"] += 1
                continue
            stats["same pin count"] += 1
            ok = True
            for pin_key, pin in pins.items():
                best = None
                for our_key, our_pin in ours.items():
                    distance = ((our_pin["point"]["x"] - pin["point"]["x"]) ** 2 +
                                (our_pin["point"]["y"] - pin["point"]["y"]) ** 2) ** 0.5
                    if best is None or distance < best[0]:
                        best = (distance, str(our_pin.get("pinname")))
                if best is None or best[0] > TOLERANCE:
                    ok = False
                    break
                if best[1] != str(pin.get("pinname")):
                    expected = str(pin.get("pinname"))
                    if expected.startswith("P_"):
                        stats["name is P_n (unrecoverable)"] += 1
                    else:
                        stats["name wrong but recoverable"] += 1
                    ok = False
                    break
            if ok:
                stats["all names correct"] += 1
    print("matched components, grouped by whether we exported the same pin count:")
    for key, value in stats.most_common():
        print(f"  {key:<32}{value:>5}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
