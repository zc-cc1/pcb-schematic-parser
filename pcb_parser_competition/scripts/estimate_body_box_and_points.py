"""Estimate the component body box, then generate pin points as body edge + 10 px.

Discovered law: 57 % of the contest reference pin points sit EXACTLY 10.00 px outside
their own component bbox, and generating points that way lifts the pin hit rate from
35 to 60 of the 89 evaluable pins.  It only works with a body box: our detector box
includes the pin leads, so box+10 overshoots (11 hits).

So the pins item reduces to one question: can the body box be recovered from the image?
This measures that against the reference boxes and then re-measures the pin hits using
the estimated body.
"""
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from evaluate_golden_cases import GT_DIR  # noqa: E402

BASE = Path(r"D:/pcb_unet_v2/golden_eval")
STUB = 10.0


def body_box(gray, det_box, pad=2):
    """Largest enclosed ink contour inside the detection box."""
    x1, y1, x2, y2 = [int(round(v)) for v in det_box]
    x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
    x2, y2 = min(gray.shape[1], x2 + pad), min(gray.shape[0], y2 + pad)
    if x2 <= x1 or y2 <= y1:
        return None
    patch = gray[y1:y2, x1:x2]
    ink = (patch < 150).astype(np.uint8) * 255
    contours, _ = cv2.findContours(ink, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 12:
            continue
        if best is None or area > best[0]:
            best = (area, cv2.boundingRect(contour))
    if best is None:
        return None
    bx, by, bw, bh = best[1]
    return [float(bx + x1), float(by + y1), float(bx + x1 + bw), float(by + y1 + bh)]


def to_top_left(box, height):
    """Reference boxes are bottom-left; image is top-left."""
    return [box[0], height - box[3], box[2], height - box[1]]


def generate(point, box, stub=STUB):
    cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
    dx, dy = point[0] - cx, point[1] - cy
    if abs(dx) >= abs(dy):
        x = box[2] + stub if dx >= 0 else box[0] - stub
        return (x, min(max(point[1], box[1]), box[3]))
    y = box[3] + stub if dy >= 0 else box[1] - stub
    return (min(max(point[0], box[0]), box[2]), y)


def main():
    box_errors = []
    hits_current = hits_generated = total = 0
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
        pred_pins = pred.get("pins") or {}
        pred_components = pred.get("components") or {}
        for component, pins in (truth.get("pins") or {}).items():
            ours = pred_pins.get(component)
            if not ours:
                continue
            det = (pred_components.get(component) or {}).get("bbox")
            if not det:
                continue
            # det bbox is exported bottom-left; body_box works in image coordinates
            det_tl = to_top_left(det, height)
            estimated = body_box(gray, det_tl)
            reference = (truth["components"].get(component) or {}).get("bbox")
            if estimated and reference:
                ref_tl = to_top_left(reference, height)
                box_errors.append(max(abs(estimated[i] - ref_tl[i]) for i in range(4)))
            for key, pin in pins.items():
                candidate = ours.get(key)
                if candidate is None:
                    continue
                total += 1
                ref_point = (pin["point"]["x"], pin["point"]["y"])
                cur = (candidate["point"]["x"], candidate["point"]["y"])
                if math.hypot(cur[0] - ref_point[0], cur[1] - ref_point[1]) <= 5:
                    hits_current += 1
                if estimated:
                    est_bl = [estimated[0], height - estimated[3], estimated[2], height - estimated[1]]
                    generated = generate(cur, est_bl)
                    if math.hypot(generated[0] - ref_point[0], generated[1] - ref_point[1]) <= 5:
                        hits_generated += 1
    box_errors.sort()
    if box_errors:
        print(f"estimated body box vs reference bbox: {len(box_errors)} components")
        print(f"  max-coordinate error median {box_errors[len(box_errors)//2]:.1f} px"
              f"  (p25 {box_errors[len(box_errors)//4]:.1f}, p75 {box_errors[3*len(box_errors)//4]:.1f})")
        print(f"  within 20 px: {sum(1 for e in box_errors if e <= 20)} / {len(box_errors)}")
    print()
    print(f"evaluable pins: {total}")
    print(f"  current points                     : {hits_current}")
    print(f"  body edge + {STUB:.0f} px (estimated) : {hits_generated}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
