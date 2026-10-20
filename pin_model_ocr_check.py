"""那 9 个"型号写成位号"的元件，裁切 OCR 到底能不能读到真型号。

对每个元件：按 bbox + 留白裁切、放大 3×、OCR，然后看 GT 型号有没有原样出现；
没原样出现的，给出最接近的一条文本和编辑距离，方便判断是"完全没读出来"还是"读得差不多"。
"""
from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from pathlib import Path

import cv2
import numpy as np

GT = Path(r"C:/Users/Pc/Desktop/新建文件夹/赛题六公开数据集/赛题六公开数据集/open_GT/open_GT")

TARGETS = [("1", "USB1"), ("1", "USB2"), ("4", "U2"), ("6", "U1"), ("6", "U3"),
           ("11", "U1"), ("11", "USB1"), ("11", "USB2"), ("13", "H1")]
MARGINS = (20, 40)

_engine = None


def engine():
    global _engine
    if _engine is None:
        from rapidocr import RapidOCR
        _engine = RapidOCR()
    return _engine


def texts_of(image):
    result = engine()(image)
    return [str(t).strip() for t in (result.txts or [])]


def norm(text):
    return re.sub(r"[\s\-_()（）]+", "", str(text).upper())


def best_match(model, texts):
    target = norm(model)
    scored = [(SequenceMatcher(None, target, norm(t)).ratio(), t) for t in texts]
    return max(scored) if scored else (0.0, "")


for case, component in TARGETS:
    truth = json.loads((GT / case / f"{case}.json").read_text(encoding="utf-8"))
    image = cv2.imdecode(np.fromfile(str(next((GT / case).glob("*.png"))), np.uint8), cv2.IMREAD_COLOR)
    height, width = image.shape[:2]
    box = truth["components"][component]["bbox"]
    model = str(truth["components"][component].get("Name") or "")
    print(f"=== case {case:>2} {component:6s} GT型号 {model!r}")
    for margin in MARGINS:
        x1, y1 = max(0, int(box[0]) - margin), max(0, int(height - box[3]) - margin)
        x2, y2 = min(width, int(box[2]) + margin), min(height, int(height - box[1]) + margin)
        crop = cv2.resize(image[y1:y2, x1:x2], None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        found = texts_of(crop)
        exact = [t for t in found if norm(t) == norm(model)]
        ratio, closest = best_match(model, found)
        flag = "✅ 原样读到" if exact else ("≈ 近似" if ratio >= 0.75 else "❌ 没读到")
        print(f"   +{margin:>2}px  {flag:10s} 最接近={closest!r} 相似度={ratio:.2f}")
        if margin == 40:
            print(f"          全部文本: {' | '.join(found[:18])}")
    print()
