"""Is the wire loss a model-coverage problem or a vectorisation problem?

Rasterise the reference wire segments and ask the U-Net directly: how much of that ink
does the probability map reach above our decode threshold?  If large parts of the
reference wires never reach 0.35, no amount of graph work can recover them - the mask
is blind there and only retraining / multi-scale inference helps.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, r"D:/pcb5_submission/src")
from pcb_parser.unet_checkpoint import load_wire_pin_checkpoint  # noqa: E402
from pcb_parser.unet_inference import predict_configured  # noqa: E402

GT = Path(r"C:/Users/Pc/Desktop/新建文件夹/赛题六公开数据集/赛题六公开数据集/open_GT/open_GT")
IMAGES = Path(r"D:/pcb_open_gt_in")
WEIGHTS = Path(r"D:/pcb5_submission/weights/unet_wire_pin_v1.pt")
INFERENCE = json.loads(Path(r"D:/pcb5_submission/configs/unet_inference_v2.json").read_text(encoding="utf-8"))
THRESHOLD = float(INFERENCE.get("wire_threshold", 0.35))

model, checkpoint, info = load_wire_pin_checkpoint(WEIGHTS, "cuda:0")
print(f"checkpoint {info['sha256'][:12]} variant={info['model_variant']} threshold={THRESHOLD}")

rows = []
total_gt = total_hit = 0
for folder in sorted(GT.iterdir(), key=lambda p: int(p.name)):
    case = folder.name
    truth = json.loads(next(folder.glob("*.json")).read_text(encoding="utf-8"))
    with Image.open(IMAGES / f"{case}.png") as handle:
        rgb = np.array(handle.convert("RGB"))
        height, width = rgb.shape[:2]
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    segments = 0
    for net in (truth.get("nets") or {}).values():
        for edge in (net.get("edges") or {}).values():
            pts = [(float(p["x"]), float(p["y"])) for p in edge]
            if len(pts) < 2:
                continue
            # reference coordinates are bottom-left; the image is top-left
            draw.line([(x, height - y) for x, y in pts], fill=255, width=3)
            segments += 1
    gt_mask = np.array(mask) > 0
    probability = np.asarray(predict_configured(model, rgb, "cuda:0", checkpoint["config"], INFERENCE)[0])
    covered = float((probability[gt_mask] >= THRESHOLD).mean()) if gt_mask.any() else 0.0
    mean_inside = float(probability[gt_mask].mean()) if gt_mask.any() else 0.0
    total_gt += int(gt_mask.sum())
    total_hit += int((probability[gt_mask] >= THRESHOLD).sum())
    rows.append((case, segments, int(gt_mask.sum()), covered, mean_inside))
    print(f"  case {case:>2}: 真值线 {segments:>3} 段 / 墨迹 {int(gt_mask.sum()):>6} px，"
          f"prob≥{THRESHOLD} 覆盖 {covered:.1%}，墨迹内平均概率 {mean_inside:.3f}")
    del probability

print(f"\n合计：真值墨迹 {total_gt} px，被模型概率（≥{THRESHOLD}）覆盖 {total_hit} px = "
      f"{total_hit/max(1,total_gt):.1%}")
