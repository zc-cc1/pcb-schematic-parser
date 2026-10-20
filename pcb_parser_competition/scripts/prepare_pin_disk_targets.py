"""Render the pin target as a tolerance disk instead of a single Gaussian.

Why: the scored criterion is "predicted peak within 5 px of the true tip", so the
target should describe the whole acceptance disk.  A sigma-2 Gaussian puts almost
all mass inside ~3 px and falls off immediately, which pulls the decoder's peaks
toward the annotated pixel even when the annotation itself is 1-2 px off.

Two variants are written so the effect can be isolated:
  disk       1.0 inside radius 3, 0.5 at radius 5, 0 beyond (a step edge)
  softdisk   1.0 inside radius 3, smooth ramp to 0 at radius 6 (BCE-friendly)

Nothing about the wire target changes.  Existing `pin_heatmap.png` is preserved;
the variants live in `pin_heatmap_disk.png` / `pin_heatmap_softdisk.png` and the
manifest gains their paths.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

V2 = Path(r"D:/pcb_unet_v2/data/segmentation_unet_v2")


def render(points, width, height, mode, inner=3.0, outer=5.0, soft=6.0):
    heat = np.zeros((height, width), np.float32)
    radius = int(np.ceil(soft))
    for x, y in points:
        x0, x1 = max(0, int(x) - radius), min(width, int(x) + radius + 1)
        y0, y1 = max(0, int(y) - radius), min(height, int(y) + radius + 1)
        if x0 >= x1 or y0 >= y1:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1]
        distance = np.hypot(xx - x, yy - y).astype(np.float32)
        if mode == "disk":
            value = np.where(distance <= inner, 1.0,
                             np.where(distance <= outer, 1.0 - 0.5 * (distance - inner) / max(outer - inner, 1e-6), 0.0))
        else:
            value = np.where(distance <= inner, 1.0, np.clip((soft - distance) / (soft - inner), 0.0, 1.0))
        heat[y0:y1, x0:x1] = np.maximum(heat[y0:y1, x0:x1], value.astype(np.float32))
    return (np.round(heat * 255).astype(np.uint8))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=V2 / "manifest.jsonl")
    parser.add_argument("--modes", nargs="*", default=["disk", "softdisk"])
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines()]
    updated = []
    for index, row in enumerate(rows, 1):
        folder = V2 / row["case_id"]
        tips = [tuple(p["point"]) for p in json.loads(Path(row["pin_points"]).read_text(encoding="utf-8"))]
        for mode in args.modes:
            array = render(tips, row["width"], row["height"], mode)
            path = folder / f"pin_heatmap_{mode}.png"
            Image.fromarray(array).save(path)
            row[f"pin_heatmap_{mode}"] = str(path.resolve())
        updated.append(row)
        if index % 50 == 0:
            print(f"  {index}/{len(rows)} pages", flush=True)
    args.manifest.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in updated), encoding="utf-8")
    sample = [r for r in updated if r["stats"]["pin_tips"] > 100][:1]
    for row in sample:
        for mode in args.modes:
            array = np.asarray(Image.open(row[f"pin_heatmap_{mode}"]))
            print(f"{row['case_id']} {mode}: max {array.max()} levels {len(np.unique(array))} "
                  f"px>128 {int((array > 128).sum())} px>0 {int((array > 0).sum())}")
    print(f"wrote pin targets for {len(rows)} pages, modes {args.modes}")


if __name__ == "__main__":
    main()
