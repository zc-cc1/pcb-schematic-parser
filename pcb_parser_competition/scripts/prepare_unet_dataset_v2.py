"""Prepare U-Net supervision v2: full wire geometry + per-channel ignore masks.

Why this exists (see D:/pcb_unet_v2/reports/source_geometry_audit.json):
  v1 built wire labels from `edge.keyPointPaths.wire` only.  692 of 14,895 wire
  edges carry their chain in `edge.coordinates` instead, so v1 dropped those wires
  and rejected every page containing one: 103/238 train and 28/59 val pages usable.
  With the fallback, every page keeps usable geometry (238 train / 59 val).

Outputs one folder per case plus a manifest. Images are not duplicated: the
manifest points at the existing standardized PNGs. Nothing under
data/segmentation_unet_v1, data/standardized, Dataset300 or weights/ is modified.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pcb_parser.dataset_converter import load_source_annotation  # noqa: E402
from pcb_parser.unet_targets_v2 import build_targets, render_review  # noqa: E402

DEFAULT_OUTPUT = Path(r"D:/pcb_unet_v2/data/segmentation_unet_v2")
DEFAULT_REVIEWS = Path(r"D:/pcb_unet_v2/reports/label_reviews")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/manifest.jsonl")
    parser.add_argument("--split-manifest", type=Path, default=ROOT / "data/detection_v1/manifest.jsonl")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/unet_wire_pin_v2.json")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--reviews", type=Path, default=DEFAULT_REVIEWS)
    parser.add_argument("--case-id", action="append", help="only these cases (spot checks)")
    parser.add_argument("--no-reviews", action="store_true")
    args = parser.parse_args()

    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    line_width = cfg["label_line_width"]
    pin_sigma = cfg["pin_sigma"]
    splits = {r["case_id"]: r["split"] for r in
              (json.loads(line) for line in args.split_manifest.read_text(encoding="utf-8").splitlines())}
    rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines()]
    if args.case_id:
        rows = [r for r in rows if r["case_id"] in set(args.case_id)]

    args.output.mkdir(parents=True, exist_ok=True)
    if not args.no_reviews:
        args.reviews.mkdir(parents=True, exist_ok=True)
    exported, failures = [], []
    for index, row in enumerate(rows, 1):
        case_id = row["case_id"]
        try:
            annotation, duplicates = load_source_annotation(row["raw_annotation_path"])
            image_path = ROOT / f"data/standardized/{case_id}/{case_id}.png"
            with Image.open(image_path) as image:
                width, height = image.size
                if image.mode != "RGB":
                    image = image.convert("RGB")
            wire, pin_heatmap, wire_ignore, pin_ignore, tips, stats = build_targets(
                annotation, width, height, line_width=line_width, pin_sigma=pin_sigma)
            folder = args.output / case_id
            folder.mkdir(parents=True, exist_ok=True)
            wire.save(folder / "wire_mask.png")
            pin_heatmap.save(folder / "pin_heatmap.png")
            np.savez_compressed(folder / "ignore_masks.npz",
                                wire_ignore=np.asarray(wire_ignore, dtype=np.uint8),
                                pin_ignore=np.asarray(pin_ignore, dtype=np.uint8))
            (folder / "pin_points.json").write_text(json.dumps(tips, ensure_ascii=False), encoding="utf-8")
            review_path = None
            if not args.no_reviews:
                review = render_review(image_path, wire, pin_heatmap, wire_ignore, pin_ignore, tips)
                review_path = args.reviews / f"{case_id}.png"
                review.save(review_path)
            split = splits.get(case_id, "excluded")
            exported.append({
                "case_id": case_id,
                "image": str(image_path.resolve()),
                "width": width, "height": height,
                "wire_mask": str((folder / "wire_mask.png").resolve()),
                "pin_heatmap": str((folder / "pin_heatmap.png").resolve()),
                "ignore_masks": str((folder / "ignore_masks.npz").resolve()),
                "pin_points": str((folder / "pin_points.json").resolve()),
                "review": str(review_path.resolve()) if review_path else None,
                "split": split,
                "source_annotation_sha256": sha256(row["raw_annotation_path"]),
                "image_sha256": sha256(image_path),
                "label_status": "draft_source_geometry_v2_with_coordinates_fallback",
                "line_width": line_width,
                "pin_sigma": pin_sigma,
                "identical_source_duplicate_keys": len(duplicates),
                "stats": stats,
            })
            if index % 25 == 0:
                print(f"  {index}/{len(rows)} pages", flush=True)
        except Exception as error:  # noqa: BLE001 - report and continue, never half-write a page
            failures.append({"case_id": case_id, "error": f"{type(error).__name__}: {error}"})

    (args.output / "manifest.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in exported), encoding="utf-8")

    def totals(field):
        return sum(r["stats"].get(field, 0) for r in exported)

    by_split = {}
    for split in ("train", "val", "excluded"):
        subset = [r for r in exported if r["split"] == split]
        if subset:
            by_split[split] = {
                "pages": len(subset),
                "physical_edges": sum(r["stats"]["physical_edges"] for r in subset),
                "edges_recovered_from_coordinates": sum(r["stats"]["edges_recovered_from_coordinates"] for r in subset),
                "edges_without_geometry": sum(r["stats"]["edges_without_geometry"] for r in subset),
                "pin_tips": sum(r["stats"]["pin_tips"] for r in subset),
                "unowned_root_pins_ignored": sum(r["stats"]["unowned_root_pins_ignored"] for r in subset),
                "wire_pixels": sum(r["stats"]["wire_positive_pixels"] for r in subset),
            }
    pages_usable = sum(1 for r in exported if r["stats"]["physical_edges"] > 0
                       and r["stats"]["wire_positive_pixels"] > 0 and r["stats"]["pin_tips"] > 0)
    summary = {
        "status": "draft_labels_v2_not_certified",
        "label_change_vs_v1": "edge.coordinates fallback for edges with empty keyPointPaths.wire; per-channel ignore masks; page-level rejection removed",
        "config": {"line_width": line_width, "pin_sigma": pin_sigma},
        "cases_exported": len(exported),
        "failures": failures,
        "pages_with_wire_and_pins": pages_usable,
        "by_split": by_split,
        "totals": {
            "physical_edges": totals("physical_edges"),
            "edges_using_key_point_paths": totals("edges_using_key_point_paths"),
            "edges_recovered_from_coordinates": totals("edges_recovered_from_coordinates"),
            "edges_without_geometry": totals("edges_without_geometry"),
            "edges_skipped_out_of_bounds": totals("edges_skipped_out_of_bounds"),
            "unowned_root_pins_ignored": totals("unowned_root_pins_ignored"),
            "invalid_pin_tips_ignored": totals("invalid_pin_tips_ignored"),
            "wire_positive_pixels": totals("wire_positive_pixels"),
            "pin_tips": totals("pin_tips"),
        },
        "manually_approved_cases": 0,
        "labels_are_generated_from_annotations_not_images": True,
        "manifest": str((args.output / "manifest.jsonl").resolve()),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "failures"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
