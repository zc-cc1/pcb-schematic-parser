"""对比三种切法在"漏掉的 51 个脚"上的编号可读率：

  whole   整元件 bbox + 20px，放大 3×
  side    元件单侧条带（该引脚所在的那一侧），放大 4×
  perpin  以引脚 tip 为中心的 80×50 小窗，放大 4×

"可读" = 该窗口的 OCR 文本里出现与 GT 编号完全相同的串（窗口很小，不再限制距离）。
"""
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

GT = Path(r"C:/Users/Pc/Desktop/新建文件夹/赛题六公开数据集/赛题六公开数据集/open_GT/open_GT")
RUN = Path(r"D:/pcb_open_gt_v33angle120/logs")

_engine = None


def engine():
    global _engine
    if _engine is None:
        from rapidocr import RapidOCR
        _engine = RapidOCR()
    return _engine


def texts_of(image):
    result = engine()(image)
    if hasattr(result, "txts"):
        return [str(t).strip() for t in (result.txts or [])]
    return [str(item[1]).strip() for item in (result or [])]


def upscale(crop, scale):
    return cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)


def main():
    rows = json.loads(Path(r"C:/Users/Pc/Desktop/新建文件夹/pin_miss_v33_corrected.json").read_text(encoding="utf-8"))["rows"]
    by_comp = {}
    for row in rows:
        by_comp.setdefault((row["case"], row["component"]), []).append(row)

    stats = Counter()
    per_case = {}
    for (case, component), group in sorted(by_comp.items(), key=lambda kv: (int(kv[0][0]), kv[0][1])):
        truth = json.loads((GT / case / f"{case}.json").read_text(encoding="utf-8"))
        image = cv2.imdecode(np.fromfile(str(next((GT / case).glob("*.png"))), np.uint8), cv2.IMREAD_COLOR)
        height, width = image.shape[:2]
        box = truth["components"][component]["bbox"]
        x1, y1 = int(box[0]), int(height - box[3])
        x2, y2 = int(box[2]), int(height - box[1])

        whole = texts_of(upscale(image[max(0, y1 - 20):y2 + 20, max(0, x1 - 20):x2 + 20], 3))
        hit = {"whole": 0, "side": 0, "perpin": 0}
        for row in group:
            pin = truth["pins"][component][row["key"]]
            number = row["key"].replace("pin_", "")
            gx, gy = pin["point"]["x"], height - pin["point"]["y"]
            if number in whole:
                hit["whole"] += 1

            side = "L" if gx < (x1 + x2) / 2 else "R"
            if side == "L":
                strip = image[max(0, y1 - 25):y2 + 25, max(0, int(gx) - 45):int((x1 + x2) / 2)]
            else:
                strip = image[max(0, y1 - 25):y2 + 25, int((x1 + x2) / 2):min(width, int(gx) + 45)]
            if strip.size and number in texts_of(upscale(strip, 4)):
                hit["side"] += 1

            wx1, wx2 = int(gx) - 40, int(gx) + 40
            wy1, wy2 = int(gy) - 25, int(gy) + 25
            window = image[max(0, wy1):min(height, wy2), max(0, wx1):min(width, wx2)]
            if window.size and number in texts_of(upscale(window, 4)):
                hit["perpin"] += 1
        for key, value in hit.items():
            stats[key] += value
            per_case.setdefault(key, Counter())[case] += value

    total = stats["whole"] * 0 + sum(len(v) for v in by_comp.values())
    print(f"漏掉的 {total} 个脚，编号可读数量：")
    for key, label in (("whole", "整元件 3×"), ("side", "单侧条带 4×"), ("perpin", "单脚小窗 4×")):
        print(f"  {label:12s} {stats[key]:2d} / {total}   {stats[key] / total:5.1%}")
    print()
    print("逐 case（单脚小窗）：", dict(sorted(per_case["perpin"].items(), key=lambda kv: int(kv[0]))))


if __name__ == "__main__":
    main()
