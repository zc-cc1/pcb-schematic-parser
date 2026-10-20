"""可行性验证：把缺信息的元件从原图切出来、放大、再做一次 OCR，能不能补回引脚编号/名称。

对照三组：
  A 整图 OCR（现有产物 logs/<case>_ocr_texts.json，本身已经带 tile 和 90° 旋转）
  B 元件裁切 OCR（bbox + margin，放大 3×）
  C 只裁切不放大（隔离变量：收益来自"隔离"还是"放大"）

判定一个引脚"编号可读"：某条 OCR 文本等于 GT 编号，且中心点距该引脚 tip ≤ 25px。
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, r"C:/Users/Pc/Desktop/新建文件夹/pcb_parser_competition/scripts")

GT = Path(r"C:/Users/Pc/Desktop/新建文件夹/赛题六公开数据集/赛题六公开数据集/open_GT/open_GT")
RUN = Path(r"D:/pcb_open_gt_v33angle120/logs")
MARGINS = (10, 20, 30, 40)
SCALE = 3.0
RADIUS = 25.0

TARGETS = [("1", "USB1"), ("11", "USB2"), ("13", "H1"), ("8", "U1"), ("10", "U1"), ("11", "U1")]

_engine = None


def engine():
    global _engine
    if _engine is None:
        from rapidocr import RapidOCR
        _engine = RapidOCR()
    return _engine


def run_ocr(image):
    result = engine()(image)
    out = []
    if hasattr(result, "boxes"):
        boxes, texts, scores = result.boxes, result.txts, result.scores
    else:
        boxes = [item[0] for item in result]
        texts = [item[1] for item in result]
        scores = [item[2] for item in result]
    for box, text, score in zip(boxes, texts, scores):
        pts = np.asarray(box, dtype=float).reshape(-1, 2)
        out.append((str(text), float(score), float(pts[:, 0].mean()), float(pts[:, 1].mean())))
    return out


def crop_texts(image, box, margin, scale):
    height, width = image.shape[:2]
    x1 = max(0, int(box[0]) - margin)
    y1 = max(0, int(box[1]) - margin)
    x2 = min(width, int(box[2]) + margin)
    y2 = min(height, int(box[3]) + margin)
    crop = image[y1:y2, x1:x2]
    if scale != 1.0:
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return [(text, score, x1 + cx / scale, y1 + cy / scale)
            for text, score, cx, cy in run_ocr(crop)]


def readable(number, name, texts, gx, gy, radius=RADIUS):
    number_ok = any(t[0].strip() == str(number) and math.hypot(t[2] - gx, t[3] - gy) <= radius
                    for t in texts)
    name_ok = any(t[0].strip().upper() == str(name).strip().upper()
                  and math.hypot(t[2] - gx, t[3] - gy) <= radius * 2.4
                  for t in texts) if name else False
    return number_ok, name_ok


def main():
    rows = []
    for case, component in TARGETS:
        truth = json.loads((GT / case / f"{case}.json").read_text(encoding="utf-8"))
        image_path = next((GT / case).glob("*.png"))
        image = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        height = image.shape[0]
        full = json.loads((RUN / f"{case}_ocr_texts.json").read_text(encoding="utf-8"))["texts"]
        full_pts = [(t["text"], float(t["confidence"]),
                     (t["bbox"][0] + t["bbox"][2]) / 2, (t["bbox"][1] + t["bbox"][3]) / 2)
                    for t in full]
        box = truth["components"][component]["bbox"]
        px_box = [box[0], height - box[3], box[2], height - box[1]]

        per_margin = {m: crop_texts(image, px_box, m, SCALE) for m in MARGINS}
        plain = crop_texts(image, px_box, 20, 1.0)

        pins = truth["pins"][component]
        record = {"case": case, "component": component, "pins": len(pins),
                  "full": 0, "crop": {str(m): 0 for m in MARGINS},
                  "crop_noscale": 0, "names_full": 0, "names_crop": 0}
        for key, pin in pins.items():
            number = key.replace("pin_", "")
            name = str(pin.get("pinname") or "")
            gx, gy = pin["point"]["x"], height - pin["point"]["y"]
            number_ok, name_ok = readable(number, name, full_pts, gx, gy)
            record["full"] += int(number_ok)
            record["names_full"] += int(bool(name_ok))
            record["crop_noscale"] += int(readable(number, name, plain, gx, gy)[0])
            for margin in MARGINS:
                record["crop"][str(margin)] += int(readable(number, name, per_margin[margin], gx, gy)[0])
            record["names_crop"] += int(bool(readable(number, name, per_margin[20], gx, gy)[1]))
        rows.append(record)
        print(f"case {case:>2} {component:5s} pins {record['pins']:2d} | 整图编号可读 {record['full']:2d} "
              f"| 裁切×3 " + " ".join(f"{m}:{record['crop'][str(m)]:2d}" for m in MARGINS)
              + f" | 裁切×1 {record['crop_noscale']:2d}"
              + f" | 名称 整图{record['names_full']:2d}/裁切{record['names_crop']:2d}")
        print("      裁切 OCR 文本(20px):", " | ".join(t[0] for t in per_margin[20][:40]))

    out = Path(r"C:/Users/Pc/Desktop/新建文件夹/pin_crop_ocr_feasibility.json")
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nwritten:", out)


if __name__ == "__main__":
    main()
