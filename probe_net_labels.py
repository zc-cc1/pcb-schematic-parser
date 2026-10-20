"""Does the detector fire on the net label strips at all? Probe case 12 / case 1."""
from __future__ import annotations

import json
from pathlib import Path

from ultralytics import YOLO

WEIGHTS = Path(r"D:/pcb5_submission/weights/detector_dataset300_v3_nano.pt")
IMAGES = Path(r"D:/pcb_open_gt_in")
GT = Path(r"C:/Users/Pc/Desktop/新建文件夹/赛题六公开数据集/赛题六公开数据集/open_GT/open_GT")

model = YOLO(str(WEIGHTS))
names = model.names


def iou_cover(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    area = max(1.0, (a[2] - a[0]) * (a[3] - a[1]))
    return inter / area


for case in ("12", "1"):
    truth = json.loads((GT / case / f"{case}.json").read_text(encoding="utf-8"))["components"]
    nets = {k: v for k, v in truth.items() if str(v.get("type")).startswith("net")}
    if not nets:
        continue
    result = model.predict(str(IMAGES / f"{case}.png"), imgsz=1536, conf=0.01, iou=0.5, verbose=False)[0]
    boxes = [(float(x[0]), float(x[1]), float(x[2]), float(x[3]), float(x[4]), int(x[5])) for x in result.boxes.data]
    print(f"=== case {case}: {len(boxes)} raw detections at conf>=0.01 ===")
    for key, ref in nets.items():
        gb = ref["bbox"]
        # GT bbox is bottom-left origin; the model predicts top-left -> convert GT to top-left
        from PIL import Image
        with Image.open(IMAGES / f"{case}.png") as h:
            height = float(h.height)
        gtl = [gb[0], height - gb[3], gb[2], height - gb[1]]
        hits = sorted(((iou_cover(b, gtl), b) for b in boxes if iou_cover(b, gtl) > 0.2), reverse=True)[:3]
        desc = ", ".join(f"{names[b[5]]}:{b[4]:.2f}(iou {c:.2f})" for c, b in hits) or "无任何重叠预测"
        print(f"  {key:<8} GT top-left {[round(v) for v in gtl]}  ->  {desc}")
