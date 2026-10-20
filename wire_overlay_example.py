"""Draw one reference net next to what we produced, so the cut mismatch is visible.

Blue = reference wire (its own segments).  Red = our wire.  Green dots = our pins.
Where a blue line has no red line on top of it, that reference segment counts as missing.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, r"C:/Users/Pc/Desktop/新建文件夹/pcb_parser_competition/scripts")
from platform_metrics import _close, _segments  # noqa: E402

GT = Path(r"C:/Users/Pc/Desktop/新建文件夹/赛题六公开数据集/赛题六公开数据集/open_GT/open_GT")
IMAGES = Path(r"D:/pcb_open_gt_in")
PRED = Path(r"D:/pcb_open_gt_v32best/final_results")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="13")
    parser.add_argument("--net", default="net_7")
    parser.add_argument("--scale", type=float, default=2.0)
    parser.add_argument("--out", type=Path, default=Path(r"C:/Users/Pc/Desktop/新建文件夹/wire_overlay_example.png"))
    args = parser.parse_args()

    truth = json.loads((GT / args.case / f"{args.case}.json").read_text(encoding="utf-8"))
    pred = json.loads((PRED / f"{args.case}_Pred.json").read_text(encoding="utf-8"))
    with Image.open(IMAGES / f"{args.case}.png") as handle:
        image = handle.convert("RGB")
        width, height = image.size

    net = (truth.get("nets") or {})[args.net]
    gt_edges = _segments(net)
    our_edges = [seg for n in (pred.get("nets") or {}).values() for seg in _segments(n)]

    def to_image(point):
        return (point[0], height - point[1])

    xs, ys = [], []
    for seg in gt_edges:
        for point in (seg[0], seg[-1]):
            x, y = to_image(point)
            xs.append(x); ys.append(y)
    margin = 45
    box = (max(0, int(min(xs) - margin)), max(0, int(min(ys) - margin)),
           min(width, int(max(xs) + margin)), min(height, int(max(ys) + margin)))

    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    for seg in gt_edges:
        pts = [to_image(p) for p in seg]
        matched = any(_close(seg, other) for other in our_edges)
        draw.line(pts, fill=(30, 90, 220) if matched else (0, 0, 0), width=3)
    for seg in our_edges:
        pts = [to_image(p) for p in seg]
        if any(p[0] < box[0] - 20 or p[0] > box[2] + 20 or p[1] < box[1] - 20 or p[1] > box[3] + 20 for p in pts):
            continue
        draw.line(pts, fill=(220, 40, 40), width=2)
    for group in (pred.get("pins") or {}).values():
        for pin in (group.values() if isinstance(group, dict) else group):
            point = (pin or {}).get("point") or {}
            if "x" in point and "y" in point:
                x, y = point["x"], height - point["y"]
                if box[0] <= x <= box[2] and box[1] <= y <= box[3]:
                    draw.ellipse([x - 4, y - 4, x + 4, y + 4], outline=(20, 160, 60), width=2)

    crop = overlay.crop(box)
    crop = crop.resize((int(crop.width * args.scale), int(crop.height * args.scale)), Image.LANCZOS)
    crop.save(args.out)
    matched = sum(1 for seg in gt_edges if any(_close(seg, other) for other in our_edges))
    print(f"case {args.case} {args.net}: 真值 {len(gt_edges)} 段，对上 {matched} 段；区域 {box}；已保存 {args.out}")
    print("蓝色=已对上的真值线  黑色=没对上的真值线  红色=我们画的线  绿圈=我们的引脚")


if __name__ == "__main__":
    main()
