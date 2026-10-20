"""Calibrate the inward shift of pin tips against the official targets.

Measurement: our predicted tips reproduce the annotation's `tip` field (115.5 vs 117,
53.5 vs 54 on 0025/C_1), but the official reference puts the tip ~10 px closer to the
component body.  So the two definitions differ by a roughly constant inward offset.

This sweeps that offset, moving every tip along its pin axis towards the component box
centre, and reports PinF1 (official rule: key, number, pinname, 5 px) for each value.
The shift is a calibration constant, not a model change.
"""
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from evaluate_golden_cases import GT_DIR  # noqa: E402

BASE = Path(r"D:/pcb_unet_v2/golden_eval")
TOLERANCE = 5.0


def shifted_point(pin_point, box, shift):  # noqa: D401
    """Move a pin point towards its component box centre by `shift` pixels."""
    if shift <= 0 or not box:
        return pin_point[0], pin_point[1]
    cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
    dx, dy = cx - pin_point[0], cy - pin_point[1]
    norm = math.hypot(dx, dy)
    if norm < 1e-6:
        return pin_point[0], pin_point[1]
    step = min(shift, norm)
    return pin_point[0] + dx / norm * step, pin_point[1] + dy / norm * step


def evaluate(shift, cases):
    tp = fp = fn = 0
    for truth, pred in cases:
        truth_pins = truth.get("pins") or {}
        pred_pins = pred.get("pins") or {}
        # boxes: prefer our own detection box, fall back to the reference box
        boxes = {}
        for key, component in (pred.get("components") or {}).items():
            boxes[key] = component.get("bbox")
        for key, component in (truth.get("components") or {}).items():
            boxes.setdefault(key, component.get("bbox"))
        for component, pins in truth_pins.items():
            ours = pred_pins.get(component)
            for pin_key, pin in pins.items():
                candidate = (ours or {}).get(pin_key)
                if candidate is None:
                    fn += 1
                    continue
                px, py = shifted_point((candidate["point"]["x"], candidate["point"]["y"]),
                                       boxes.get(component), shift)
                distance = math.hypot(px - pin["point"]["x"], py - pin["point"]["y"])
                if distance > TOLERANCE or str(pin.get("pinname")) != str(candidate.get("pinname")):
                    fn += 1
                    continue
                tp += 1
        for component, pins in pred_pins.items():
            truth_here = truth_pins.get(component)
            for pin_key in pins:
                matched = truth_here is not None and pin_key in truth_here and any(
                    math.hypot(shifted_point((pins[pin_key]["point"]["x"], pins[pin_key]["point"]["y"]),
                                             boxes.get(component), shift)[0] - truth_here[pin_key]["point"]["x"],
                               shifted_point((pins[pin_key]["point"]["x"], pins[pin_key]["point"]["y"]),
                                             boxes.get(component), shift)[1] - truth_here[pin_key]["point"]["y"])
                    <= TOLERANCE and str(truth_here[pin_key].get("pinname")) == str(pins[pin_key].get("pinname"))
                    for _ in [0])
                if not matched:
                    fp += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return tp, fp, fn, precision, recall, f1


def main():
    cases = []
    for folder in sorted(GT_DIR.iterdir()):
        if not folder.is_dir():
            continue
        case = folder.name.split("-")[0]
        submission_path = BASE / case / "wire_graph" / "partial_submission.json"
        if not submission_path.exists():
            continue
        truth = json.loads(next(folder.glob("*_target.json")).read_text(encoding="utf-8"))
        pred = json.loads(submission_path.read_text(encoding="utf-8"))
        cases.append((truth, pred))
    print(f"cases: {len(cases)}")
    print(f"{'shift':>6}{'tp':>5}{'fp':>5}{'fn':>5}{'P':>8}{'R':>8}{'F1':>8}")
    best = None
    for shift in (0, 2, 4, 6, 8, 10, 12, 14, 16):
        tp, fp, fn, precision, recall, f1 = evaluate(shift, cases)
        marker = ""
        if best is None or f1 > best[1]:
            best = (shift, f1)
            marker = "  <- best so far"
        print(f"{shift:>6}{tp:>5}{fp:>5}{fn:>5}{precision:>8.4f}{recall:>8.4f}{f1:>8.4f}{marker}")
    print()
    print(f"best shift: {best[0]} px, PinF1 {best[1]:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
