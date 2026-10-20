"""Does lowering the detection thresholds recover the digit '1' in a pin-number column?

The column read finds 2/3/4 but misses '1' (a thin glyph). RapidOCR drops boxes below
`Det.box_thresh` (0.5) / `Det.text_score` (0.5). This sweeps those two knobs on one
IC's digit strip and reports which digits appear and at what score.
"""
import json
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
    case = sys.argv[1] if len(sys.argv) > 1 else "2"
    comp = sys.argv[2] if len(sys.argv) > 2 else "U1"
    gt_root = r"D:\pcb_v8_check\gt"
    gt = json.load(open(f"{gt_root}/{case}/{case}_target_named.json", encoding="utf-8"))
    image = cv2.cvtColor(cv2.imread(f"{gt_root}/{case}/{case}.png"), cv2.COLOR_BGR2RGB)
    height = image.shape[0]
    box = gt["components"][comp]["bbox"]
    x1, y1, x2, y2 = box
    top, bottom = height - y2, height - y1
    crop = [int(x1 - 34), int(top - 6), int(x1 + 6), int(bottom + 6)]
    patch = image[crop[1]:crop[3], crop[0]:crop[2]]
    want = sorted(pk[4:] for pk in (gt["pins"].get(comp) or {}))
    print(f"{case}/{comp}: GT numbers {want}, body x1={x1} rows(top/bottom)={top:.0f}/{bottom:.0f}")

    for box_thresh in (0.5, 0.3, 0.2, 0.1):
        for text_score in (0.5, 0.3, 0.2):
            params = {"Det.box_thresh": box_thresh, "Det.text_score": text_score,
                      "Det.thresh": 0.2, "Global.max_side_len": 2048}
            engine, _ = create_engine({"onnx_threads": 4}) if False else (None, None)
            from rapidocr import RapidOCR
            engine = RapidOCR(params=params)
            found = []
            for scale in (2, 4):
                scaled = cv2.resize(patch, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                result = engine(np.ascontiguousarray(scaled[:, :, ::-1]), use_det=True,
                                use_cls=False, use_rec=True)
                if result.txts:
                    for text, score in zip(result.txts, result.scores):
                        n = clean(text)
                        found.append((scale, n, round(float(score), 2), str(text)))
            ok = {n for _, n, _, _ in found if n}
            print(f"  box_thresh={box_thresh} text_score={text_score}: "
                  f"digits={sorted(x for x in ok if x)}  raw={found}")


if __name__ == "__main__":
    sys.exit(main())
