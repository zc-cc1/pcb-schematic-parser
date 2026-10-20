"""绑定测试：裁切 OCR 读到的编号，能不能按"离哪个 tip 最近"正确落到引脚上。

只看编号可读的元件。对每个 GT 引脚，取裁切图里最近的纯数字文本，与 GT 编号比对。
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np

GT = Path(r"C:/Users/Pc/Desktop/新建文件夹/赛题六公开数据集/赛题六公开数据集/open_GT/open_GT")

_engine = None


def engine():
    global _engine
    if _engine is None:
        from rapidocr import RapidOCR
        _engine = RapidOCR()
    return _engine


def ocr_points(image):
    result = engine()(image)
    out = []
    for box, text, score in zip(result.boxes, result.txts, result.scores):
        pts = np.asarray(box, dtype=float).reshape(-1, 2)
        out.append((str(text).strip(), float(score), float(pts[:, 0].mean()), float(pts[:, 1].mean())))
    return out


def main():
    targets = [("1", "USB1"), ("8", "U1"), ("10", "U1"), ("11", "U1"), ("13", "H1"), ("10", "U2")]
    total = right = 0
    for case, component in targets:
        truth = json.loads((GT / case / f"{case}.json").read_text(encoding="utf-8"))
        image = cv2.imdecode(np.fromfile(str(next((GT / case).glob("*.png"))), np.uint8), cv2.IMREAD_COLOR)
        height = image.shape[0]
        box = truth["components"][component]["bbox"]
        x1, y1 = max(0, int(box[0]) - 20), max(0, int(height - box[3]) - 20)
        x2, y2 = min(image.shape[1], int(box[2]) + 20), min(height, int(height - box[1]) + 20)
        scale = 3.0
        crop = cv2.resize(image[y1:y2, x1:x2], None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        texts = [(t, s, x1 + cx / scale, y1 + cy / scale) for t, s, cx, cy in ocr_points(crop)]
        numbers = [item for item in texts if item[0].isdigit()]
        hits = []
        for key, pin in truth["pins"][component].items():
            want = key.replace("pin_", "")
            gx, gy = pin["point"]["x"], height - pin["point"]["y"]
            if not numbers:
                hits.append((want, None, None))
                continue
            nearest = min(numbers, key=lambda item: math.hypot(item[2] - gx, item[3] - gy))
            hits.append((want, nearest[0], round(math.hypot(nearest[2] - gx, nearest[3] - gy), 1)))
        ok = sum(1 for want, got, _d in hits if got == want)
        total += len(hits)
        right += ok
        print(f"case {case:>2} {component:5s} 绑定正确 {ok:2d}/{len(hits):2d}   " +
              " ".join(f"{w}->{g}" for w, g, _ in hits))
    print(f"\n合计 {right}/{total} = {right / total:.0%}")


if __name__ == "__main__":
    main()
