"""Offline geometric diagnosis of unmatched U-Net pin points, not inference."""
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pcb_parser.dataset_converter import load_source_annotation
from evaluate_trained_unet import match_points


def point(value):
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return [float(value[0]), float(value[1])]
    return None


def main():
    run = ROOT / "runs/unet_inference_v2"
    output = run / "false_positive_analysis"
    output.mkdir(exist_ok=True)
    split = json.loads((run / "split.json").read_text(encoding="utf-8"))
    comparison = json.loads((run / "adopted_comparison.json").read_text(encoding="utf-8"))
    expected_by_case = {row["case_id"]: row for row in comparison["cases"]}
    raw_manifest = {row["case_id"]: row for row in map(json.loads, (ROOT / "data/manifest.jsonl").read_text(encoding="utf-8").splitlines())}
    records = []
    for row in split["rows"]:
        case = row["case_id"]
        candidates = json.loads((run / "predictions" / case / "pin_candidates.json").read_text(encoding="utf-8"))
        tips = json.loads((Path(row["wire_mask"]).parent / "pin_points.json").read_text(encoding="utf-8"))
        tip_points = np.array([tip["point"] for tip in tips])
        matches, actual = match_points([item["point"] for item in candidates], tip_points)
        assert actual == expected_by_case[case]["adopted"]["pin"]
        matched = {i for i, _, _ in matches}
        source, _ = load_source_annotation(raw_manifest[case]["raw_annotation_path"])
        base_records = []
        for tip in tips:
            cid, pid = tip["component_source_id"], tip["pin_source_id"]
            component = source["components"][cid]
            pin = {**source.get("pins", {}).get(pid, {}), **component.get("pins", {}).get(pid, {})}
            base = point(pin.get("base"))
            if base:
                base_records.append({"point": base, "tip": tip["point"], "component": cid, "category": component.get("category"), "pin": pid})
        bases = np.array([item["point"] for item in base_records])
        for i, candidate in enumerate(candidates):
            if i in matched:
                continue
            position = np.array(candidate["point"])
            distances = np.linalg.norm(tip_points - position, axis=1)
            nearest = int(distances.argmin())
            distance = float(distances[nearest])
            base_distances = np.linalg.norm(bases - position, axis=1) if len(bases) else np.array([])
            base_index = int(base_distances.argmin()) if len(bases) else None
            base_distance = float(base_distances[base_index]) if base_index is not None else None
            # These are mutually exclusive proximity groups, not semantic labels.
            group = "within_5px_competing_prediction" if distance <= 5 else (
                "near_base_not_tip" if base_distance is not None and base_distance <= 5 else (
                    "near_tip_5_to_15px" if distance <= 15 else "other_far_from_tip"))
            records.append({"case_id": case, "candidate_index": i, **candidate,
                            "nearest_tip_distance_px": distance, "nearest_tip": tips[nearest],
                            "nearest_base_distance_px": base_distance,
                            "nearest_base": base_records[base_index] if base_index is not None else None,
                            "geometric_group": group})
    assert len(records) == 423
    counts = Counter(item["geometric_group"] for item in records)
    base_types = Counter(item["nearest_base"]["category"] for item in records if item["geometric_group"] == "near_base_not_tip")
    summary = {"total_fp": len(records), "geometric_groups": dict(counts), "near_base_component_categories": dict(base_types),
               "tip_distance_groups": {"at_most_5px": sum(item["nearest_tip_distance_px"] <= 5 for item in records),
                                       "5_to_15px": sum(5 < item["nearest_tip_distance_px"] <= 15 for item in records),
                                       "over_15px": sum(item["nearest_tip_distance_px"] > 15 for item in records)},
               "scope": "Geometric proximity to draft labels; not a manually confirmed semantic taxonomy",
               "source": "Accepted v2 predictions; no training or threshold changes",
               "official_score": None}
    (output / "points.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    # Representative crops provide evidence for manual interpretation.
    rows_by_case = {row["case_id"]: row for row in split["rows"]}
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 13)
    for group in sorted(counts):
        selected = [item for item in records if item["geometric_group"] == group]
        selected.sort(key=lambda item: (item["case_id"], -item["confidence"]))
        # Spread examples across pages rather than showing one page only.
        picks, seen = [], set()
        for item in selected:
            if item["case_id"] not in seen:
                picks.append(item)
                seen.add(item["case_id"])
            if len(picks) == 12:
                break
        for item in selected:
            if len(picks) == 12:
                break
            if item not in picks:
                picks.append(item)
        columns, cell_w, cell_h = 4, 260, 275
        sheet = Image.new("RGB", (columns * cell_w, ((len(picks) + columns - 1) // columns) * cell_h + 35), "white")
        pen = ImageDraw.Draw(sheet)
        pen.text((10, 8), f"{group}: {counts[group]} FP | red=prediction, blue=nearest tip, green=nearest base", font=font, fill="black")
        for index, item in enumerate(picks):
            x, y = item["point"]
            x0, y0 = round(x) - 48, round(y) - 48
            with Image.open(rows_by_case[item["case_id"]]["image"]) as image:
                crop = image.convert("RGB").crop((x0, y0, x0 + 96, y0 + 96)).resize((240, 240))
            draw = ImageDraw.Draw(crop)
            for position, colour in [(item["point"], (230, 30, 30)), (item["nearest_tip"]["point"], (0, 100, 240)),
                                     (item["nearest_base"]["point"] if item["nearest_base"] else None, (0, 160, 60))]:
                if position is None:
                    continue
                px, py = (position[0] - x0) * 2.5, (position[1] - y0) * 2.5
                draw.ellipse((px - 5, py - 5, px + 5, py + 5), outline=colour, width=2)
            left, top = (index % columns) * cell_w + 10, (index // columns) * cell_h + 35
            sheet.paste(crop, (left, top + 22))
            pen.text((left, top), f"{item['case_id']} | tip {item['nearest_tip_distance_px']:.1f}px", font=font, fill="black")
        sheet.save(output / f"{group}.png")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
