"""只看当前漏掉的引脚：裁切 OCR 能不能读出它们的编号。

输入 pin_miss_v33_corrected.json（51 个 FN 的清单），逐个判断：
  整图 OCR       该编号在 tip 25px 内出现过吗
  裁切×3 @20px   同上
顺带把每个元件的裁切图存下来，方便肉眼确认"数字到底在不在图上"。
"""
from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

GT = Path(r"C:/Users/Pc/Desktop/新建文件夹/赛题六公开数据集/赛题六公开数据集/open_GT/open_GT")
RUN = Path(r"D:/pcb_open_gt_v33angle120/logs")
OUT = Path(r"C:/Users/Pc/Desktop/新建文件夹/_crop_preview")
OUT.mkdir(exist_ok=True)
MARGIN, SCALE, RADIUS = 20, 3.0, 25.0
if len(sys.argv) > 1:
    MARGIN = int(sys.argv[1])
if len(sys.argv) > 2:
    RADIUS = float(sys.argv[2])

_engine = None


def engine():
    global _engine
    if _engine is None:
        from rapidocr import RapidOCR
        _engine = RapidOCR()
    return _engine


def run_ocr(image):
    result = engine()(image)
    if hasattr(result, "boxes"):
        boxes, texts, scores = result.boxes, result.txts, result.scores
    else:
        boxes, texts, scores = ([i[0] for i in result], [i[1] for i in result], [i[2] for i in result])
    out = []
    for box, text, score in zip(boxes, texts, scores):
        pts = np.asarray(box, dtype=float).reshape(-1, 2)
        out.append((str(text), float(score), float(pts[:, 0].mean()), float(pts[:, 1].mean())))
    return out


def main():
    rows = json.loads(Path(r"C:/Users/Pc/Desktop/新建文件夹/pin_miss_v33_corrected.json").read_text(encoding="utf-8"))["rows"]
    by_comp = {}
    for row in rows:
        by_comp.setdefault((row["case"], row["component"]), []).append(row)

    stats = Counter()
    detail = []
    for (case, component), group in sorted(by_comp.items(), key=lambda kv: (int(kv[0][0]), kv[0][1])):
        truth = json.loads((GT / case / f"{case}.json").read_text(encoding="utf-8"))
        image_path = next((GT / case).glob("*.png"))
        image = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        height = image.shape[0]
        full = json.loads((RUN / f"{case}_ocr_texts.json").read_text(encoding="utf-8"))["texts"]
        full_pts = [(t["text"].strip(), (t["bbox"][0] + t["bbox"][2]) / 2, (t["bbox"][1] + t["bbox"][3]) / 2)
                    for t in full]
        box = truth["components"][component]["bbox"]
        x1 = max(0, int(box[0]) - MARGIN); y1 = max(0, int(height - box[3]) - MARGIN)
        x2 = min(image.shape[1], int(box[2]) + MARGIN); y2 = min(height, int(height - box[1]) + MARGIN)
        crop = image[y1:y2, x1:x2]
        big = cv2.resize(crop, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_CUBIC)
        cv2.imencode(".png", big)[1].tofile(str(OUT / f"{case}_{component}.png"))
        crop_pts = [(t, s, x1 + cx / SCALE, y1 + cy / SCALE) for t, s, cx, cy in run_ocr(big)]

        for row in group:
            pin = truth["pins"][component][row["key"]]
            number = row["key"].replace("pin_", "")
            gx, gy = pin["point"]["x"], height - pin["point"]["y"]
            full_ok = any(t == number and math.hypot(x - gx, y - gy) <= RADIUS for t, x, y in full_pts)
            crop_ok = any(t.strip() == number and math.hypot(x - gx, y - gy) <= RADIUS
                          for t, _s, x, y in crop_pts)
            stats[(full_ok, crop_ok)] += 1
            detail.append({"case": case, "component": component, "pin": number,
                           "name": row["name"], "kind": row["kind"],
                           "full_read": full_ok, "crop_read": crop_ok})

    total = sum(stats.values())
    print(f"漏掉的 {total} 个引脚里：")
    print(f"  整图可读、裁切也可读 : {stats[(True, True)]:3d}")
    print(f"  整图漏读、裁切能读回 : {stats[(False, True)]:3d}   <= 这套方案真正新增的")
    print(f"  两边都读不出         : {stats[(False, False)]:3d}   <= 图上没有/太小，二次 OCR 无用")
    print()
    print("两边都读不出的那些：")
    for item in detail:
        if not item["full_read"] and not item["crop_read"]:
            print(f"  case {item['case']:>2} {item['component']:6s} pin_{item['pin']:4s} {item['name'][:10]:10s} {item['kind']}")
    Path(r"C:/Users/Pc/Desktop/新建文件夹/pin_crop_ocr_fn.json").write_text(
        json.dumps(detail, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
