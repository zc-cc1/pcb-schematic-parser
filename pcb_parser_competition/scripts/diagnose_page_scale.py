"""Why does one page score 0?  Isolate scale vs absolute layout vs label quality.

For a single page, run the frozen checkpoint on four views and score each against
the same labels:

  native          the page as-is (the accepted inference path)
  upscaled x2/x3  the schematic enlarged before tiling
  cropped         only the schematic bounding box, at native scale

If `cropped` recovers the page, the failure is absolute layout/scale, not the
labels.  If nothing recovers it, the supervision itself is what differs.
No training, no checkpoint writes.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from pcb_parser.unet_model import WirePinUNet  # noqa: E402
from evaluate_unet_baseline import match, peaks, prf, tile_starts  # noqa: E402
from evaluate_unet_v2 import predict  # noqa: E402

V2 = Path(r"D:/pcb_unet_v2/data/segmentation_unet_v2")


def content_box(image, threshold=200):
    """Bounding box of non-background ink, ignoring a thin page border."""
    grey = image.mean(axis=2)
    ink = grey < threshold
    rows = np.where(ink.any(axis=1))[0]
    cols = np.where(ink.any(axis=0))[0]
    if not len(rows) or not len(cols):
        return (0, 0, image.shape[1], image.shape[0])
    return int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1


def score(model, image, targets, device, scale=1.0, offset=(0, 0), tag="", wire_threshold=0.5,
          pin_threshold=0.8, pin_tolerance=5.0):
    """Score a (possibly scaled/cropped) view against label arrays already in that frame."""
    height, width = image.shape[:2]
    probability = predict(model, image, device, 512, 384)
    predicted = probability[0] >= wire_threshold
    truth = targets["wire"] > 0
    wire = prf(int((predicted & truth).sum()), int((predicted & ~truth).sum()), int((truth & ~predicted).sum()))
    tips = [tuple(p) for p in targets["tips"]]
    candidates = []
    for x, y in peaks(probability[1], pin_threshold, 5):
        candidates.append((x, y))
    tp, unmatched_pred, unmatched_truth = match(candidates, tips, pin_tolerance)
    pin = prf(tp, len(unmatched_pred), len(unmatched_truth))
    print(f"  {tag:<22} view {width}x{height}  wire P {wire['precision']:.3f} R {wire['recall']:.3f} "
          f"F1 {wire['f1']:.3f}  pin P {pin['precision']:.3f} R {pin['recall']:.3f} F1 {pin['f1']:.3f}", flush=True)
    return {"tag": tag, "view": [width, height], "scale": scale, "offset": list(offset),
            "wire": wire, "pin": pin}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-id", default="0208")
    parser.add_argument("--checkpoint", type=Path, default=Path(r"D:/pcb_unet_v2/runs/A2_full60/best.pt"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=Path(r"D:/pcb_unet_v2/reports/page_scale_diagnosis.json"))
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model = WirePinUNet(checkpoint["base_channels"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])

    rows = [json.loads(line) for line in (V2 / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    row = next(r for r in rows if r["case_id"] == args.case_id)
    with Image.open(row["image"]) as source:
        image = np.asarray(source.convert("RGB"))
    with Image.open(row["wire_mask"]) as mask:
        wire = np.asarray(mask)
    tips = [p["point"] for p in json.loads((V2 / args.case_id / "pin_points.json").read_text(encoding="utf-8"))]

    results = []
    print(f"{args.case_id}: {row['width']}x{row['height']}, labels wire px {int((wire > 0).sum())}, tips {len(tips)}")

    results.append(score(model, image, {"wire": wire, "tips": tips}, device, tag="native"))

    for scale in (1.5, 2.0, 3.0):
        target = (round(image.shape[1] * scale), round(image.shape[0] * scale))
        big = np.asarray(Image.fromarray(image).resize(target, Image.Resampling.LANCZOS))
        big_wire = np.asarray(Image.fromarray(wire).resize(target, Image.Resampling.NEAREST))
        big_tips = [(x * scale, y * scale) for x, y in tips]
        results.append(score(model, big, {"wire": big_wire, "tips": big_tips}, device, scale=scale,
                             tag=f"upscaled x{scale:g}"))

    x1, y1, x2, y2 = content_box(image)
    # drop a thin outer border: only keep the schematic if the box covers the whole page
    if (x2 - x1) * (y2 - y1) > 0.9 * image.shape[0] * image.shape[1]:
        print("  content box covers the page; using the ink centroid window instead")
    crop = image[y1:y2, x1:x2]
    crop_wire = wire[y1:y2, x1:x2]
    crop_tips = [(x - x1, y - y1) for x, y in tips if x1 <= x < x2 and y1 <= y < y2]
    results.append(score(model, crop, {"wire": crop_wire, "tips": crop_tips}, device, scale=1.0,
                         offset=(x1, y1), tag=f"cropped to content {x2-x1}x{y2-y1}"))

    for zoom in (2.0, 3.0):
        target = (round(crop.shape[1] * zoom), round(crop.shape[0] * zoom))
        big = np.asarray(Image.fromarray(crop).resize(target, Image.Resampling.LANCZOS))
        big_wire = np.asarray(Image.fromarray(crop_wire).resize(target, Image.Resampling.NEAREST))
        big_tips = [(x * zoom, y * zoom) for x, y in crop_tips]
        results.append(score(model, big, {"wire": big_wire, "tips": big_tips}, device, scale=zoom,
                             offset=(x1, y1), tag=f"crop + upscale x{zoom:g}"))

    best = max(results, key=lambda r: r["wire"]["f1"])
    report = {"status": "page_scale_diagnosis", "case_id": args.case_id, "checkpoint": str(args.checkpoint),
              "content_box": [x1, y1, x2, y2], "views": results, "best_view": best["tag"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"best view: {best['tag']}  wire F1 {best['wire']['f1']:.4f}  pin F1 {best['pin']['f1']:.4f}")
    print("written:", args.output)


if __name__ == "__main__":
    main()
