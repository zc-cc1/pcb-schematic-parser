"""Prototype: recover a whole pin-number column by reading the digit strip and
matching rows, instead of anchoring on the (unreliable) learned tip.

For one IC side, the printed numbers form a vertical column just outside the body.
Read that strip with detection on, keep the boxes that are small integers, sort them
by row, sort the un-numbered pins on the same side by row, and pair them in order.
This does not depend on the tip y being right.
"""
import argparse
import json
import math
import re
import sys

import cv2
import numpy as np

BASE = r"D:\pcb5_submission"
sys.path.insert(0, BASE + r"\src")
from pcb_parser.pin_text_ocr import create_engine  # noqa: E402

SMALL = re.compile(r"^\d{1,4}$")


def clean(text):
    digits = "".join(ch for ch in str(text or "") if ch.isdigit())
    return digits if digits and len(digits) <= 4 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="2")
    ap.add_argument("--gt-root", default=r"D:\pcb_v8_check\gt")
    ap.add_argument("--pred", default=r"D:\pcb_v8_check\out\final_results")
    ap.add_argument("--outward", type=float, default=34.0)
    ap.add_argument("--scale", type=int, default=2)
    args = ap.parse_args()

    gt = json.load(open(f"{args.gt_root}/{args.case}/{args.case}_target_named.json", encoding="utf-8"))
    pred = json.load(open(f"{args.pred}/{args.case}_Pred.json", encoding="utf-8"))
    image = cv2.cvtColor(cv2.imread(f"{args.gt_root}/{args.case}/{args.case}.png"), cv2.COLOR_BGR2RGB)
    height = image.shape[0]
    engine, _ = create_engine({"onnx_threads": 4})

    for comp, group in (gt["pins"] or {}).items():
        if str((gt["components"].get(comp) or {}).get("type")) not in ("box", "ic"):
            continue
        have = set(pred["pins"].get(comp) or {})
        missing = [k for k in group if k not in have]
        if not missing:
            print(f"{comp}: nothing missing")
            continue
        box = gt["components"][comp]["bbox"]
        x1, y1, x2, y2 = box
        top, bottom = height - y2, height - y1
        # pins of this component on the left side of the body
        side_pins = []
        for pk, want in group.items():
            px, py = want["point"]["x"], height - want["point"]["y"]
            if abs(px - x1) < max(abs(px - x2), abs(py - top), abs(py - bottom)):
                side_pins.append((py, pk, want))
        side_pins.sort()
        crop = [int(x1 - args.outward), int(top - 6), int(x1 + 6), int(bottom + 6)]
        x1c, y1c, x2c, y2c = crop
        patch = image[y1c:y2c, x1c:x2c]
        scaled = cv2.resize(patch, None, fx=args.scale, fy=args.scale, interpolation=cv2.INTER_CUBIC)
        res = engine(np.ascontiguousarray(scaled[:, :, ::-1]), use_det=True, use_cls=False, use_rec=True)
        found = []
        if res.boxes is not None:
            for points, text, score in zip(res.boxes, res.txts, res.scores):
                number = clean(text)
                if number is None:
                    continue
                ys = [p[1] for p in points]
                cy = y1c + (sum(ys) / len(ys)) / args.scale
                found.append((cy, number, round(float(score), 3), str(text)))
        found.sort()
        print(f"\n{comp}: GT pins on left {len(side_pins)}, missing {len(missing)}, digits found {len(found)}")
        print("   GT rows :", [(round(y), pk[4:]) for y, pk, _ in side_pins])
        print("   read rows:", [(round(y), n, t) for y, n, s, t in found])
        if len(found) == len(side_pins):
            pairs = [(pk, n) for (y, pk, _), (cy, n, s, t) in zip(side_pins, found)]
            print("   row-order pairing:", pairs)
            hit = sum(pk[4:] == n for pk, n in pairs)
            print(f"   correct {hit}/{len(pairs)}")


if __name__ == "__main__":
    sys.exit(main())
