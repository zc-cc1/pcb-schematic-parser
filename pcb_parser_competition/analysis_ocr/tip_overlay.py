"""Overlay GT pin tips vs our tips on one page, zoomed on the biggest IC."""
import json
import math
import sys
from pathlib import Path

from PIL import Image, ImageDraw

GT_ROOT = Path(r"C:\Users\Pc\Desktop\新建文件夹\赛题六公开数据集\赛题六公开数据集\200_train_cases")
REPLAY = Path(r"D:\pcb_v6_check\pin_replay")
OUT = Path(r"C:\Users\Pc\Desktop\新建文件夹\pcb_parser_competition\analysis_ocr\evidence")

case = sys.argv[1] if len(sys.argv) > 1 else "0042"
gt = json.loads(next((GT_ROOT / case).glob("*_target_named.json")).read_text(encoding="utf-8"))
result = json.loads((REPLAY / case / "pin_text_result.json").read_text(encoding="utf-8"))
image = Image.open(next((GT_ROOT / case).glob("*.png"))).convert("RGB")
H = image.size[1]

# biggest GT IC
ics = [(k, v) for k, v in gt["components"].items() if str(v.get("type")) in ("box", "ic")]
if not ics:
    print("no IC")
    sys.exit(0)
key, comp = max(ics, key=lambda kv: len(gt["pins"].get(kv[0]) or {}))
x1, y1, x2, y2 = comp["bbox"]
box = (x1, H - y2, x2, H - y1)
margin = 40
crop_box = (max(0, box[0] - margin), max(0, box[1] - margin),
            min(image.size[0], box[2] + margin), min(image.size[1], box[3] + margin))
scale = 4
patch = image.crop(tuple(int(v) for v in crop_box))
patch = patch.resize((patch.width * scale, patch.height * scale), Image.LANCZOS)
draw = ImageDraw.Draw(patch)


def to_crop(point):
    return ((point[0] - crop_box[0]) * scale, (point[1] - crop_box[1]) * scale)


draw.rectangle([to_crop((box[0], box[1])), to_crop((box[2], box[3]))], outline=(0, 160, 0), width=2)
gt_points = []
for pin_key, want in (gt["pins"].get(key) or {}).items():
    point = (float(want["point"]["x"]), float(want["point"]["y"]))
    gt_points.append((pin_key, point))
    cx, cy = to_crop(point)
    draw.ellipse([cx - 6, cy - 6, cx + 6, cy + 6], outline=(220, 0, 0), width=2)
for pin in result["pin_candidates"]:
    if pin.get("owner_status") != "assigned":
        continue
    point = tuple(pin["point"])
    if not (crop_box[0] - 10 <= point[0] <= crop_box[2] + 10 and crop_box[1] - 10 <= point[1] <= crop_box[3] + 10):
        continue
    cx, cy = to_crop(point)
    d = min((math.dist(point, p) for _, p in gt_points), default=999)
    colour = (0, 0, 220) if d <= 5 else (240, 140, 0)
    draw.ellipse([cx - 3, cy - 3, cx + 3, cy + 3], fill=colour)
OUT.mkdir(parents=True, exist_ok=True)
target = OUT / f"{case}_{key}_gt_vs_ours_tips.png"
patch.save(target)
print(f"{case}/{key}: {len(gt_points)} GT tips (red circles), ours blue<=5px / orange>5px -> {target.name} ({patch.size[0]}x{patch.size[1]})")
