"""Pick the pin-ordering rule that best reproduces the reference numbering.

The rule must be a pure geometric function, so it can be evaluated offline against the
reference instead of re-running the whole chain per candidate.  Each candidate orders
the pins of a matched component and compares the resulting 1..n numbers with the
reference pin names where those names are plain digits.
"""
import itertools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from evaluate_golden_cases import GT_DIR  # noqa: E402

BASE = Path(r"D:/pcb_unet_v2/golden_eval")
TOLERANCE = 20.0

RULES = {
    "y then x (upper first)": lambda p: (round(p[1], 3), round(p[0], 3)),
    "-y then x (lower first)": lambda p: (-round(p[1], 3), round(p[0], 3)),
    "x then y (left first)": lambda p: (round(p[0], 3), round(p[1], 3)),
    "-x then y (right first)": lambda p: (-round(p[0], 3), round(p[1], 3)),
}


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


def collect():
    """Yield (component type, [(truth point, truth name)], [(our point)]) pairs."""
    out = []
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
            kind = (truth.get("components", {}).get(key) or {}).get("type")
            reference = []
            for pin in pins.values():
                reference.append(((pin["point"]["x"], pin["point"]["y"]), str(pin.get("pinname"))))
            # only components whose reference names are plain digits can rank a rule
            if not all(name.isdigit() for _, name in reference):
                continue
            our_points = [(pin["point"]["x"], pin["point"]["y"]) for pin in ours.values()]
            out.append((kind, reference, our_points))
    return out


def main():
    data = collect()
    print(f"components with digit-only reference pin names: {len(data)}")
    print()
    for label, key in RULES.items():
        correct = 0
        for kind, reference, our_points in data:
            order = sorted(range(len(our_points)), key=lambda i: key(our_points[i]))
            numbers = {order[i]: str(i + 1) for i in range(len(order))}
            ok = True
            for point, expected in reference:
                best = None
                for index, our_point in enumerate(our_points):
                    distance = ((our_point[0] - point[0]) ** 2 + (our_point[1] - point[1]) ** 2) ** 0.5
                    if best is None or distance < best[0]:
                        best = (distance, index)
                if best is None or best[0] > TOLERANCE or numbers[best[1]] != expected:
                    ok = False
                    break
            correct += ok
        share = 100 * correct / max(1, len(data))
        print(f"  {label:<26} {correct:>4}/{len(data)}  {share:5.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
