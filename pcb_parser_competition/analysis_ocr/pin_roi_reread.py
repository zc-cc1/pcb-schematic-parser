"""Does re-reading the pin-number ROI at higher scale actually recover the miss?

Takes the GT pin number positions of one case, crops the small band just outside the
symbol edge where the digits are printed, and runs the recogniser on the crop at
1x / 2x / 4x (with the same engine the pipeline uses). If 1x fails and 4x succeeds,
the fix is a targeted re-OCR; if both fail, the digits are beyond this model.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

BASE = Path(r"D:\pcb5_submission")
sys.path.insert(0, str(BASE / "src"))
from pcb_parser.pin_text_ocr import create_engine  # noqa: E402

CFG = json.loads((BASE / "configs" / "pin_text_v2.json").read_text(encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--gt-root", required=True)
    ap.add_argument("--pad", type=float, default=14.0)
    args = ap.parse_args()
    gt = json.loads(next(Path(args.gt_root, args.case).glob("*_target_named.json")).read_text(encoding="utf-8"))
    image_path = next(Path(args.gt_root, args.case).glob("*.png"))
    rgb = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
    height = rgb.shape[0]

    engine, models = create_engine({"onnx_threads": 4})
    print("engine:", models["Rec"]["file"])

    for comp, group in (gt.get("pins") or {}).items():
        if str((gt["components"].get(comp) or {}).get("type")) not in ("box", "ic"):
            continue
        box = gt["components"][comp]["bbox"]
        x1, y1, x2, y2 = box
        left_line = x1
        for pin_key, want in group.items():
            px, py = float(want["point"]["x"]), float(want["point"]["y"])
            ty = height - py
            # the digit sits just outside the body edge, on the pin's own row
            crop = [int(max(0, left_line - args.pad - 12)), int(ty - args.pad),
                    int(min(rgb.shape[1], left_line + 4)), int(ty + args.pad)]
            x1c, y1c, x2c, y2c = crop
            if x2c <= x1c or y2c <= y1c:
                continue
            patch = rgb[y1c:y2c, x1c:x2c]
            readings = []
            for scale in (1, 2, 4):
                scaled = cv2.resize(patch, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                result = engine(np.ascontiguousarray(scaled[:, :, ::-1]), use_det=False,
                                use_cls=False, use_rec=True)
                texts = list(result.txts or [])
                scores = list(result.scores or [])
                best = max(zip(texts, scores), key=lambda r: r[1], default=("", 0.))
                readings.append((scale, best[0], round(float(best[1]), 3)))
            print(f"{comp}.{pin_key[4:]:>3} want={want.get('pinname')!r:10} "
                  f"crop={crop} -> " + "  ".join(f"{s}x:{t!r}({c})" for s, t, c in readings))


if __name__ == "__main__":
    sys.exit(main())
