"""Find the printed pin numbers by colour, not by detection.

On these symbols the digits are black while the pin leads are coloured (green / red).
A thin "1" therefore merges with the lead in the detector, which is why the tiled pass
and the strip re-read both lose it. This builds a dark-ink row profile in the narrow
band beside the body edge, splits it into rows, and reads each row's own crop.
"""
import json
import sys

import cv2
import numpy as np

BASE = r"D:\pcb5_submission"
sys.path.insert(0, BASE + r"\src")
from pcb_parser.pin_text_ocr import create_engine  # noqa: E402


def dark_mask(patch):
    gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1]
    # black ink: dark and barely coloured; the leads are saturated green/red
    return (gray < 110) & (saturation < 90)


def main():
    case = sys.argv[1] if len(sys.argv) > 1 else "2"
    comp = sys.argv[2] if len(sys.argv) > 2 else "U1"
    gt_root = r"D:\pcb_v8_check\gt"
    gt = json.load(open(f"{gt_root}/{case}/{case}_target_named.json", encoding="utf-8"))
    image = cv2.cvtColor(cv2.imread(f"{gt_root}/{case}/{case}.png"), cv2.COLOR_BGR2RGB)
    height = image.shape[0]
    x1, y1, x2, y2 = gt["components"][comp]["bbox"]
    top, bottom = height - y2, height - y1
    engine, _ = create_engine({"onnx_threads": 4})

    for side in ("left", "right"):
        edge = x1 if side == "left" else x2
        band = [int(edge - 26), int(top - 8), int(edge - 2), int(bottom + 8)] if side == "left" \
            else [int(edge + 2), int(top - 8), int(edge + 26), int(bottom + 8)]
        patch = image[band[1]:band[3], band[0]:band[2]]
        mask = dark_mask(patch)
        rows = mask.sum(axis=1)
        clusters, start = [], None
        for index, count in enumerate(rows):
            if count >= 2 and start is None:
                start = index
            elif count < 2 and start is not None:
                clusters.append((start, index))
                start = None
        if start is not None:
            clusters.append((start, len(rows)))
        clusters = [c for c in clusters if c[1] - c[0] >= 4]
        print(f"\n{comp} {side}: edge={edge:.0f} rows {top:.0f}-{bottom:.0f}, ink rows -> {len(clusters)} clusters")
        readings = []
        for start, end in clusters:
            y_abs1, y_abs2 = band[1] + start - 2, band[1] + end + 2
            crop = [band[0] - 2, y_abs1, band[2] + 2, y_abs2]
            piece = image[crop[1]:crop[3], crop[0]:crop[2]]
            scaled = cv2.resize(piece, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
            result = engine(np.ascontiguousarray(scaled[:, :, ::-1]), use_det=False,
                            use_cls=False, use_rec=True)
            pairs = list(zip(result.txts or [], result.scores or []))
            text, score = max(pairs, key=lambda r: r[1], default=("", 0.))
            mid = (y_abs1 + y_abs2) / 2
            readings.append((round(mid), str(text), round(float(score), 2)))
        print("   readings (y, text, conf):", readings)
        want = gt["pins"].get(comp) or {}
        gt_rows = sorted((round(height - w["point"]["y"]), pk[4:]) for pk, w in want.items())
        print("   GT rows:", gt_rows)


if __name__ == "__main__":
    sys.exit(main())
