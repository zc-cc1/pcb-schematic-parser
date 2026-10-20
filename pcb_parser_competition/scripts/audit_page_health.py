"""Page-level health check: does each page's supervision actually sit on the drawing?

A wire label is only learnable if it lands on rendered strokes.  On 0208 half of
the label pixels sit on the 243-grey background while the visible schematic is
elsewhere, so no model and no augmentation can fit it — it must be excluded, not
trained on.

For every page this reports:
  stroke_grey      25th percentile grey of label pixels (how dark the "wire" is)
  on_ink            share of label pixels that are ink (grey < 200) within 1 px
  label_on_bg       share of label pixels on light background (grey >= 225) even
                    after allowing 1 px slack - the smoking gun for misalignment
  panels            number of ink column clusters inside the page margins
Verdicts are advisory; nothing is written into the training data.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

V2 = Path(r"D:/pcb_unet_v2/data/segmentation_unet_v2")


def panel_columns(grey, margin_ratio=0.06, gap=25, min_ink=8):
    height, width = grey.shape
    my, mx = int(height * margin_ratio), int(width * margin_ratio)
    ink = grey < 200
    ink[:my, :] = False
    ink[-my:, :] = False
    ink[:, :mx] = False
    ink[:, -mx:] = False
    columns = np.where(ink.sum(axis=0) >= min_ink)[0]
    if not len(columns):
        return 0
    groups, start, previous = 0, columns[0], columns[0]
    for column in columns[1:]:
        if column - previous > gap:
            groups += 1
            start = column
        previous = column
    return groups + 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=V2 / "manifest.jsonl")
    parser.add_argument("--output", type=Path, default=Path(r"D:/pcb_unet_v2/reports/page_health.json"))
    parser.add_argument("--bg-cut", type=float, default=225.0)
    parser.add_argument("--ink-cut", type=float, default=200.0)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines()]
    structure = ndimage.generate_binary_structure(2, 2)
    report_rows = []
    for index, row in enumerate(rows, 1):
        with Image.open(row["image"]) as source:
            grey = np.asarray(source.convert("L"))
        with Image.open(row["wire_mask"]) as mask:
            label = np.asarray(mask) > 0
        if label.sum() < 20:
            report_rows.append({"case_id": row["case_id"], "split": row["split"], "label_pixels": int(label.sum()),
                                "verdict": "no_label", "stroke_grey": None, "on_ink": None, "label_on_bg": None,
                                "panels": None})
            continue
        ink = grey < args.ink_cut
        ink_d = ndimage.binary_dilation(ink, structure, iterations=1)
        on_ink = float((label & ink_d).sum() / label.sum())
        on_ink_exact = float((label & ink).sum() / label.sum())
        # Distance-based alignment: a 3 px stroke drawn along a 1 px line has its
        # own body up to 1 px away from ink, so distance is the stable statistic.
        # Raw "share of label pixels on background" is NOT usable: a correctly
        # aligned 3 px stroke puts ~2/3 of its pixels on background by design.
        distance = ndimage.distance_transform_edt(~ink)
        label_distance = distance[label]
        median_distance = float(np.median(label_distance))
        p90_distance = float(np.percentile(label_distance, 90))
        stroke_grey = float(np.percentile(grey[label], 25))
        # mean darkness of the ink near the label: a page whose drawing uses very
        # light strokes is a presentation gap, not a misalignment.
        near_ink = ink & ndimage.binary_dilation(label, structure, iterations=2)
        near_ink_grey = float(np.median(grey[near_ink])) if near_ink.any() else None
        verdict = "ok"
        if median_distance > 2.0 or p90_distance > 5.0:
            verdict = "label_image_mismatch"
        elif near_ink_grey is not None and near_ink_grey > 175:
            verdict = "light_strokes"
        report_rows.append({"case_id": row["case_id"], "split": row["split"], "label_pixels": int(label.sum()),
                            "stroke_grey": round(stroke_grey, 1), "on_ink": round(on_ink, 4),
                            "on_ink_exact": round(on_ink_exact, 4),
                            "median_distance_to_ink": round(median_distance, 2),
                            "p90_distance_to_ink": round(p90_distance, 2),
                            "near_ink_grey": None if near_ink_grey is None else round(near_ink_grey, 1),
                            "panels": panel_columns(grey), "verdict": verdict})
        if index % 50 == 0:
            print(f"  {index}/{len(rows)} pages", flush=True)

    bad = [r for r in report_rows if r["verdict"] == "label_image_mismatch"]
    light = [r for r in report_rows if r["verdict"] == "light_strokes"]
    report = {
        "status": "page_health_check",
        "pages": len(report_rows),
        "verdict_counts": {v: sum(1 for r in report_rows if r["verdict"] == v)
                           for v in {r["verdict"] for r in report_rows}},
        "label_image_mismatch": sorted(bad, key=lambda r: -r["p90_distance_to_ink"]),
        "light_strokes": [r for r in report_rows if r["verdict"] == "light_strokes"],
        "rows": report_rows,
        "criteria": {"median_distance_max": 2.0, "p90_distance_max": 5.0, "near_ink_grey_warn": 175,
                     "bg_cut": args.bg_cut, "ink_cut": args.ink_cut},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"pages": report["pages"], "verdict_counts": report["verdict_counts"]}, ensure_ascii=False))
    print("worst label/image mismatches:")
    for row in report["label_image_mismatch"][:12]:
        print(f"  {row['case_id']} split={row['split']:<5} label_px={row['label_pixels']:>6} "
              f"median_d={row['median_distance_to_ink']:.2f} p90_d={row['p90_distance_to_ink']:.2f} "
              f"near_ink_grey={row['near_ink_grey']} panels={row['panels']}")
    print("written:", args.output)


if __name__ == "__main__":
    main()

