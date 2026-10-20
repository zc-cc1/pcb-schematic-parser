"""List the recoverable pin-name errors with enough context to see the pattern."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from evaluate_golden_cases import GT_DIR  # noqa: E402

BASE = Path(r"D:/pcb_unet_v2/golden_eval")
TOLERANCE = 20.0


def nearest_component(target_key, truth_pins, pred_pins):
    best = None
    for key, pins in pred_pins.items():
        shared = 0
        for truth_pin in (truth_pins.get(target_key) or {}).values():
            for pin in pins.values():
                if abs(pin["point"]["x"] - truth_pin["point"]["x"]) <= TOLERANCE and \
                        abs(pin["point"]["y"] - truth_pin["point"]["y"]) <= TOLERANCE:
                    shared += 1
                    break
        if best is None or shared > best[0]:
            best = (shared, key)
    return best[1] if best and best[0] else None


def main():
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
            ours_key = nearest_component(key, truth_pins, pred_pins)
            if ours_key is None:
                continue
            ours = pred_pins[ours_key]
            if len(ours) != len(pins):
                continue
            rows = []
            problem = False
            for pin_key, pin in sorted(pins.items()):
                best = None
                for our_key, our_pin in sorted(ours.items()):
                    distance = ((our_pin["point"]["x"] - pin["point"]["x"]) ** 2 +
                                (our_pin["point"]["y"] - pin["point"]["y"]) ** 2) ** 0.5
                    if best is None or distance < best[0]:
                        best = (distance, our_key, str(our_pin.get("pinname")))
                expected = str(pin.get("pinname"))
                if best is None or best[0] > TOLERANCE:
                    rows.append((pin_key, expected, "no pin", "-", "-"))
                    problem = True
                elif best[2] != expected:
                    rows.append((pin_key, expected, best[1], best[2], f"{best[0]:.1f}"))
                    if not expected.startswith("P_"):
                        problem = True
            # report only components with at least one RECOVERABLE wrong name
            if not any(r[1] and not str(r[1]).startswith("P_") and r[2] != r[1] and r[2] != "no pin"
                       and r[3] != r[1] for r in rows):
                continue
            kind = (truth.get("components", {}).get(key) or {}).get("type")
            print(f"== {case} ref {key} ({kind}) -> ours {ours_key}")
            for pin_key, expected, our_key, our_name, distance in rows:
                mark = "" if expected == our_name else "  <-- "
                print(f"   {pin_key:<8} ref {expected!r:<10} ours {our_key:<8} {our_name!r:<10} d={distance}{mark}")
            print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
