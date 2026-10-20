"""Test the panel hypothesis on a multi-panel datasheet page.

The failing page (0208) embeds three small schematics inside ~20 % of the page.
At native scale its strokes are ~1 px, so the model outputs nothing anywhere on
the page.  This tool finds the panel regions from the ink profile, crops each,
optionally upscales, and scores every view against the same labels.

If a panel crop scores normally, the failure is presentation (scale/framing) and
can be fixed by training with scale augmentation or by panel-aware inference.
No training, no writes outside the report.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from pcb_parser.unet_model import WirePinUNet  # noqa: E402
from evaluate_unet_baseline import match, peaks, prf  # noqa: E402
from evaluate_unet_v2 import predict  # noqa: E402

V2 = Path(r"D:/pcb_unet_v2/data/segmentation_unet_v2")


def panel_boxes(image, margin_ratio=0.06, gap=25, min_ink=8):
    """Cluster ink columns/rows inside the page margins into panel boxes."""
    grey = image.mean(axis=2)
    height, width = grey.shape
    my, mx = int(height * margin_ratio), int(width * margin_ratio)
    ink = grey < 200
    ink[:my, :] = False
    ink[-my:, :] = False
    ink[:, :mx] = False
    ink[:, -mx:] = False
    columns = np.where(ink.sum(axis=0) >= min_ink)[0]
    if not len(columns):
        return []
    groups, start, previous = [], columns[0], columns[0]
    for column in columns[1:]:
        if column - previous > gap:
            groups.append((start, previous))
            start = column
        previous = column
    groups.append((start, previous))
    boxes = []
    for x1, x2 in groups:
        rows = np.where(ink[:, x1:x2 + 1].any(axis=1))[0]
        if len(rows):
            boxes.append((int(x1), int(rows[0]), int(x2) + 1, int(rows[-1]) + 1))
    return boxes


def score(model, image, wire, tips, device, tag, output, pin_tolerance=5.0):
    if wire.sum() < 20:
        return {"tag": tag, "skipped": "fewer than 20 label pixels in this view"}
    probability = predict(model, image, device, 512, 384)
    predicted = probability[0] >= 0.5
    truth = wire > 0
    entry = {"tag": tag, "view": list(image.shape[1::-1]),
             "wire": prf(int((predicted & truth).sum()), int((predicted & ~truth).sum()),
                         int((truth & ~predicted).sum())),
             "label_pixels": int(truth.sum())}
    if tips:
        tp, unmatched_pred, unmatched_truth = match(peaks(probability[1], 0.8, 5), tips, pin_tolerance)
        entry["pin"] = prf(tp, len(unmatched_pred), len(unmatched_truth))
    output.append(entry)
    wire_metrics = entry["wire"]
    print(f"  {tag:<28} view {entry['view'][0]}x{entry['view'][1]}  wire P {wire_metrics['precision']:.3f} "
          f"R {wire_metrics['recall']:.3f} F1 {wire_metrics['f1']:.3f}"
          + (f"  pin F1 {entry['pin']['f1']:.3f}" if tips else ""), flush=True)
    return entry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-id", default="0208")
    parser.add_argument("--checkpoint", type=Path, default=Path(r"D:/pcb_unet_v2/runs/A2_full60/best.pt"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=Path(r"D:/pcb_unet_v2/reports/panel_diagnosis.json"))
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
    tips = [tuple(p["point"]) for p in json.loads((V2 / args.case_id / "pin_points.json").read_text(encoding="utf-8"))]

    print(f"{args.case_id} {row['width']}x{row['height']}  label px {int((wire > 0).sum())}  tips {len(tips)}")
    results = []
    score(model, image, wire, tips, device, "whole page (baseline)", results)

    boxes = panel_boxes(image)
    print(f"  detected {len(boxes)} panel column(s): {boxes}")
    for index, (x1, y1, x2, y2) in enumerate(boxes, 1):
        crop = image[y1:y2, x1:x2]
        crop_wire = wire[y1:y2, x1:x2]
        crop_tips = [(x - x1, y - y1) for x, y in tips if x1 <= x < x2 and y1 <= y < y2]
        score(model, crop, crop_wire, crop_tips, device, f"panel{index} crop {x2-x1}x{y2-y1}", results)
        for zoom in (2.0, 3.0):
            target = (round(crop.shape[1] * zoom), round(crop.shape[0] * zoom))
            big = np.asarray(Image.fromarray(crop).resize(target, Image.Resampling.LANCZOS))
            big_wire = np.asarray(Image.fromarray(crop_wire).resize(target, Image.Resampling.NEAREST))
            big_tips = [(x * zoom, y * zoom) for x, y in crop_tips]
            score(model, big, big_wire, big_tips, device, f"panel{index} crop x{zoom:g}", results)

    scored = [r for r in results if "wire" in r]
    best = max(scored, key=lambda r: r["wire"]["f1"]) if scored else None
    report = {"status": "panel_diagnosis", "case_id": args.case_id, "checkpoint": str(args.checkpoint),
              "panels": boxes, "views": results, "best_view": best["tag"] if best else None}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if best:
        print(f"best view: {best['tag']}  wire F1 {best['wire']['f1']:.4f}")
    print("written:", args.output)


if __name__ == "__main__":
    main()
