"""Which reference ink does the wire head fail to see? Sample its colour and thickness."""
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
CASES = ["1", "7", "12", "13"]

model, checkpoint, info = load_wire_pin_checkpoint(WEIGHTS, "cuda:0")
print(f"checkpoint {info['sha256'][:12]} threshold {THRESHOLD}")

for case in CASES:
    truth = json.loads((GT / case / f"{case}.json").read_text(encoding="utf-8"))
    with Image.open(IMAGES / f"{case}.png") as handle:
        rgb = np.array(handle.convert("RGB"))
        height, width = rgb.shape[:2]
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    for net in (truth.get("nets") or {}).values():
        for edge in (net.get("edges") or {}).values():
            pts = [(float(p["x"]), float(p["y"])) for p in edge]
            if len(pts) >= 2:
                draw.line([(x, height - y) for x, y in pts], fill=255, width=3)
    ink = np.array(mask) > 0
    probability = np.asarray(predict_configured(model, rgb, "cuda:0", checkpoint["config"], INFERENCE)[0])
    covered = ink & (probability >= THRESHOLD)
    missed = ink & (probability < THRESHOLD)
    def describe(sel):
        if not sel.any():
            return "n=0"
        pixels = rgb[sel].astype(float)
        brightness = pixels.mean(axis=1)
        saturation = (pixels.max(axis=1) - pixels.min(axis=1))
        return (f"n={int(sel.sum()):>6}  RGB均值 {pixels.mean(axis=0).round(0).astype(int)}  "
                f"亮度 {brightness.mean():.0f}  饱和度 {saturation.mean():.0f}  "
                f"暗像素(<100)占比 {(brightness < 100).mean():.0%}")
    print(f"  case {case:>2}: 覆盖  {describe(covered)}")
    print(f"            未覆盖 {describe(missed)}")
    del probability
