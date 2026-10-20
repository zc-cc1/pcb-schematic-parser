"""Recover the body box by walking each pin's lead until the ink thickens.

The reference puts the pin point exactly 10 px outside the BODY box, and our detector
box is an outer box that already includes the leads, so `det_box + 10` overshoots by the
lead length (measured 11 hits vs 35).  To use the discovered rule we need the body box.

A lead is a thin line; the body is a thick region.  So for each pin, walk from its tip
towards the component centre and stop where the perpendicular ink run becomes clearly
wider than the lead.  That pixel is the body edge along that pin's axis.

This measures the recovered body box against the reference boxes and then the pin hit
rate when points are regenerated as body edge + 10 px.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from evaluate_golden_cases import GT_DIR  # noqa: E402

BASE = Path(r"D:/pcb_unet_v2/golden_eval")
STUB = 10.0
THICK = 6          # a perpendicular run this wide means "inside the body"
MAX_WALK = 45      # leads are short; do not run across the page


def run_width(ink, x, y, axis):
    """Ink run length through (x, y) perpendicular to `axis` (0=horizontal pin)."""
    if not (0 <= y < ink.shape[0] and 0 <= x < ink.shape[1]) or not ink[y, x]:
        return 0
    total = 1
    if axis == 0:                      # pin runs horizontally -> measure vertical run
        for step in (1, -1):
            yy = y + step
            while 0 <= yy < ink.shape[0] and ink[yy, x]:
                total += 1
                yy += step
    else:
        for step in (1, -1):
            xx = x + step
            while 0 <= xx < ink.shape[1] and ink[y, xx]:
                total += 1
                xx += step
    return total


def body_edge(ink, tip, centre):
    """First point from the tip towards the centre where the ink becomes thick."""
    dx, dy = centre[0] - tip[0], centre[1] - tip[1]
    norm = math.hypot(dx, dy)
    if norm < 1e-6:
        return None
    ux, uy = dx / norm, dy / norm
    axis = 0 if abs(ux) >= abs(uy) else 1
    for step in range(2, MAX_WALK):
        x = int(round(tip[0] + ux * step))
        y = int(round(tip[1] + uy * step))
        if not (0 <= y < ink.shape[0] and 0 <= x < ink.shape[1]):
            return None
        if run_width(ink, x, y, axis) >= THICK:
            return (x, y)
    return None


def generate(point, box, stub=STUB):
    cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
    dx, dy = point[0] - cx, point[1] - cy
    if abs(dx) >= abs(dy):
        x = box[2] + stub if dx >= 0 else box[0] - stub
        return (x, min(max(point[1], box[1]), box[3]))
    y = box[3] + stub if dy >= 0 else box[1] - stub
    return (min(max(point[0], box[0]), box[2]), y)


def main():
    errors = []
    hits_current = hits_regen = total = 0
    for folder in sorted(GT_DIR.iterdir()):
        if not folder.is_dir():
            continue
        case = folder.name.split("-")[0]
        submission_path = BASE / case / "wire_graph" / "partial_submission.json"
        if not submission_path.exists():
            continue
        truth = json.loads(next(folder.glob("*_target.json")).read_text(encoding="utf-8"))
        pred = json.loads(submission_path.read_text(encoding="utf-8"))
        with Image.open(next(folder.glob("*.png"))) as image:
            gray = np.asarray(image.convert("L"))
            height = gray.shape[0]
        ink = gray < 150
        pred_pins = pred.get("pins") or {}
        pred_components = pred.get("components") or {}
        for component, pins in (truth.get("pins") or {}).items():
            ours = pred_pins.get(component)
            det = (pred_components.get(component) or {}).get("bbox")
            if not ours or not det:
                continue
            # image-space centre of our detection box
            cx = (det[0] + det[2]) / 2.0
            cy = height - (det[1] + det[3]) / 2.0
            edges = []
            for pin in pins.values():
                candidate = ours.get(pin_key := next((k for k in ours if True), None))
                break
            # collect our pin tips and walk each one in
            for key, candidate in ours.items():
                tip = (candidate["point"]["x"], height - candidate["point"]["y"])
                edge = body_edge(ink, tip, (cx, cy))
                if edge:
                    edges.append(edge)
            body = None
            if len(edges) >= 2:
                xs = [e[0] for e in edges]
                ys = [e[1] for e in edges]
                body = [min(xs), min(ys), max(xs), max(ys)]
            reference = (truth["components"].get(component) or {}).get("bbox")
            if body and reference:
                ref_tl = [reference[0], height - reference[3], reference[2], height - reference[1]]
                errors.append(max(abs(body[i] - ref_tl[i]) for i in range(4)))
            for key, pin in pins.items():
                candidate = ours.get(key)
                if candidate is None:
                    continue
                total += 1
                ref_point = (pin["point"]["x"], pin["point"]["y"])
                cur = (candidate["point"]["x"], candidate["point"]["y"])
                if math.hypot(cur[0] - ref_point[0], cur[1] - ref_point[1]) <= 5:
                    hits_current += 1
                if body:
                    body_bl = [body[0], height - body[3], body[2], height - body[1]]
                    gen = generate(cur, body_bl)
                    if math.hypot(gen[0] - ref_point[0], gen[1] - ref_point[1]) <= 5:
                        hits_regen += 1
    errors.sort()
    if errors:
        print(f"recovered body box vs reference bbox: {len(errors)} components")
        print(f"  max-coordinate error median {errors[len(errors)//2]:.1f} px "
              f"(p25 {errors[len(errors)//4]:.1f}, p75 {errors[3*len(errors)//4]:.1f})")
        print(f"  within 20 px: {sum(1 for e in errors if e <= 20)} / {len(errors)}")
    print()
    print(f"evaluable pins: {total}")
    print(f"  current points                      : {hits_current}")
    print(f"  recovered body edge + {STUB:.0f} px      : {hits_regen}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
