"""Read-only audit: how much of the source wire geometry actually reaches the U-Net drafts?

The current draft builder reads only `edge.keyPointPaths.wire`. This audit also reads
`edge.coordinates` (the pixel-level chain) and measures, per page:

  * edges whose keyPointPaths.wire is empty but whose coordinates are usable
  * wire pixels available under each source
  * how many pages would pass the U-Net usability gate if coordinates were used
  * where the geometry sits relative to component boxes (symbol body vs outside)

Nothing is written into the training data; output is a JSON report only.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pcb_parser.dataset_converter import load_source_annotation  # noqa: E402
from pcb_parser.unet_data import polylines  # noqa: E402


def box_of(component):
    location = component.get("location") or {}
    try:
        return [float(location[k]) for k in ("x1", "y1", "x2", "y2")]
    except (KeyError, TypeError, ValueError):
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/manifest.jsonl")
    parser.add_argument("--line-width", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0, help="0 = all pages")
    parser.add_argument("--output", type=Path, default=Path(r"D:/pcb_unet_v2/reports/source_geometry_audit.json"))
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines()]
    if args.limit:
        rows = rows[: args.limit]
    pages = []
    totals = Counter()
    for index, row in enumerate(rows, 1):
        annotation, _ = load_source_annotation(row["raw_annotation_path"])
        canvas = annotation["canvas"]
        width, height = int(canvas["width"]), int(canvas["height"])
        mask_kpp = Image.new("L", (width, height), 0)
        mask_all = Image.new("L", (width, height), 0)
        draw_kpp, draw_all = ImageDraw.Draw(mask_kpp), ImageDraw.Draw(mask_all)
        boxes = Image.new("L", (width, height), 0)
        draw_boxes = ImageDraw.Draw(boxes)
        stats = Counter()
        for component in annotation.get("components", {}).values():
            if component.get("state") == "deleted":
                continue
            box = box_of(component)
            if box:
                draw_boxes.rectangle(box, fill=255)
        for edge in annotation.get("edges", {}).values():
            if edge.get("state") == "deleted":
                continue
            if edge.get("isFlywire") is True or str(edge.get("isFlywire")).lower() == "true":
                stats["flywires"] += 1
                continue
            if (edge.get("category") or "wire") != "wire":
                continue
            stats["physical_edges"] += 1
            key_paths = list(polylines((edge.get("keyPointPaths") or {}).get("wire") or []))
            coordinates = edge.get("coordinates") or []
            if key_paths:
                stats["edges_with_key_paths"] += 1
                for path in key_paths:
                    if all(0 <= x < width and 0 <= y < height for x, y in path):
                        draw_kpp.line([(round(x), round(y)) for x, y in path], fill=255, width=args.line_width)
                        draw_all.line([(round(x), round(y)) for x, y in path], fill=255, width=args.line_width)
            elif len(coordinates) >= 2:
                # Only edges with no keyPointPaths fall back to coordinates. On edges that
                # do have keyPointPaths, `coordinates` additionally carries filled net
                # polygons, which must never be rasterized as a 3 px wire stroke.
                stats["edges_missing_key_paths_recoverable"] += 1
                points = [(float(p[0]), float(p[1])) for p in coordinates]
                if all(0 <= x < width and 0 <= y < height for x, y in points):
                    stats["edges_with_coordinates"] += 1
                    draw_all.line([(round(x), round(y)) for x, y in points], fill=255, width=args.line_width)
                else:
                    stats["coordinates_out_of_bounds"] += 1
            else:
                stats["edges_missing_everything"] += 1
        owned = set()
        for component in annotation.get("components", {}).values():
            if component.get("state") == "deleted":
                continue
            for pin_id, local in (component.get("pins") or {}).items():
                if local.get("state") != "deleted":
                    owned.add(pin_id)
        unowned = sum(1 for pin_id, pin in annotation.get("pins", {}).items()
                      if pin.get("state") != "deleted" and pin_id not in owned)
        count_kpp = int((np.asarray(mask_kpp) > 0).sum())
        count_all = int((np.asarray(mask_all) > 0).sum())
        box_mask = np.asarray(boxes) > 0
        all_mask = np.asarray(mask_all) > 0
        pages.append({
            "case_id": row["case_id"],
            "width": width, "height": height,
            "physical_edges": stats["physical_edges"],
            "flywires": stats["flywires"],
            "edges_with_key_paths": stats["edges_with_key_paths"],
            "edges_missing_key_paths_recoverable": stats["edges_missing_key_paths_recoverable"],
            "edges_missing_everything": stats["edges_missing_everything"],
            "coordinates_out_of_bounds": stats["coordinates_out_of_bounds"],
            "unowned_root_pins": unowned,
            "wire_pixels_key_paths": count_kpp,
            "wire_pixels_all_geometry": count_all,
            "geometry_gain": round(count_all / count_kpp, 3) if count_kpp else None,
            "all_geometry_inside_component_boxes_fraction":
                round(float((all_mask & box_mask).sum() / max(1, all_mask.sum())), 4),
        })
        for key in ("physical_edges", "edges_with_key_paths", "edges_missing_key_paths_recoverable",
                    "edges_missing_everything", "unowned_root_pins", "wire_pixels_key_paths", "wire_pixels_all_geometry"):
            totals[key] += pages[-1][key]
        if index % 25 == 0:
            print(f"  {index}/{len(rows)} pages", flush=True)

    def gate(page, ratio_limit, min_pixels, min_edges=1):
        bad = page["edges_missing_everything"] + page["coordinates_out_of_bounds"]
        return (page["physical_edges"] >= min_edges
                and bad <= ratio_limit * max(1, page["physical_edges"])
                and page["wire_pixels_all_geometry"] >= min_pixels)

    report = {
        "status": "read_only_source_geometry_audit",
        "manifest": str(args.manifest),
        "pages": len(pages),
        "totals": dict(totals),
        "gain_overall": round(totals["wire_pixels_all_geometry"] / max(1, totals["wire_pixels_key_paths"]), 3),
        "pages_with_geometry_gain_over_1_5x": sum(1 for p in pages if (p["geometry_gain"] or 0) > 1.5),
        "gate_all_geometry": {
            f"ratio<={limit:.0%}_min_px{pixels}":
                sum(1 for p in pages if gate(p, limit, pixels))
            for limit, pixels in ((0.05, 1000), (0.10, 1000), (0.15, 1000))
        },
        "rows": pages,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, ensure_ascii=False, indent=2))
    print("written:", args.output)


if __name__ == "__main__":
    main()
