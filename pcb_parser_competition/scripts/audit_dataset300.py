"""Audit user-provided CVJsonStd image pairs; never turn them into official Golden."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def kept(entries: dict) -> dict:
    # Audit policy only; official conversion semantics are still pending.
    return {k: v for k, v in entries.items() if isinstance(v, dict) and v.get("state") != "deleted"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    args = parser.parse_args()
    root = Path(args.dataset_root).expanduser().resolve()
    if not root.is_dir():
        parser.error(f"Dataset directory not found: {root}")
    records, failures, dimensions, warnings = [], [], [], []
    source_kinds, categories, states, formats, layouts = Counter(), Counter(), Counter(), Counter(), Counter()
    image_hash_groups = defaultdict(list)
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        images = sorted(p for p in folder.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
        annotations = sorted(folder.glob("*_CVJsonStd.json"))
        if len(images) != 1 or len(annotations) != 1:
            failures.append({"case_id": folder.name, "reason": "expected one image and one CVJsonStd JSON", "images": len(images), "annotations": len(annotations)})
            continue
        image, annotation = images[0], annotations[0]
        try:
            with Image.open(image) as im:
                width, height = im.size
                image_format, image_mode = im.format, im.mode
                im.verify()
            payload = annotation.read_bytes()
            data = json.loads(payload.decode("utf-8-sig"))
            required = ("canvas", "components", "pins", "texts", "edges")
            if not isinstance(data, dict) or not all(isinstance(data.get(k), dict) for k in required):
                raise ValueError("unrecognized CVJsonStd structure")
        except (OSError, ValueError, UnicodeError) as exc:
            failures.append({"case_id": folder.name, "reason": f"{type(exc).__name__}: {exc}"})
            continue
        case_warnings = []
        expected_suffixes = {"PNG": {".png"}, "JPEG": {".jpg", ".jpeg"}, "WEBP": {".webp"}}
        header_matches_suffix = image.suffix.lower() in expected_suffixes.get(image_format, set())
        if not header_matches_suffix:
            case_warnings.append("image_suffix_header_mismatch")
        if annotation.name.removesuffix("_CVJsonStd.json") != image.stem:
            case_warnings.append("image_annotation_stem_mismatch")
        canvas_matches = data["canvas"].get("width") == width and data["canvas"].get("height") == height
        if not canvas_matches:
            dimensions.append({"case_id": folder.name, "image": [width, height], "canvas": data["canvas"]})
            case_warnings.append("canvas_image_size_mismatch")
        c, p, t, e = (kept(data[k]) for k in ("components", "pins", "texts", "edges"))
        dangling = Counter()
        for component in c.values():
            categories[str(component.get("category"))] += 1
            for field in ("searchNameByOCR", "searchNameByVLM", "searchNameByLib"):
                ref = component.get(field)
                if isinstance(ref, str) and ref.startswith("T_") and ref not in t:
                    dangling["component_text_reference"] += 1
            for pin_id, pin in component.get("pins", {}).items():
                if pin_id not in p:
                    dangling["component_pin_reference"] += 1
                if isinstance(pin, dict):
                    for field in ("pinNumberByOCR", "pinNumberByVLM", "pinNameByOCR", "pinNameByVLM"):
                        ref = pin.get(field)
                        if isinstance(ref, str) and ref.startswith("T_") and ref not in t:
                            dangling["pin_text_reference"] += 1
        for edge in e.values():
            for ref in edge.get("connect", {}).values():
                if not isinstance(ref, dict):
                    dangling["malformed_edge_connection"] += 1
                    continue
                if ref.get("cpntId") not in c:
                    dangling["edge_component_reference"] += 1
                if ref.get("pinId") not in p:
                    dangling["edge_pin_reference"] += 1
        if dangling:
            case_warnings.append("raw_reference_warnings")
            warnings.append({"case_id": folder.name, "counts": dict(dangling)})
        image_digest = sha256(image)
        image_hash_groups[image_digest].append(folder.name)
        source_kind = str(data.get("nowCls", "unknown"))
        source_kinds[source_kind] += 1
        formats[image_format] += 1
        layouts[",".join(sorted(data))] += 1
        for key in ("components", "pins", "texts", "edges"):
            for obj in data[key].values():
                if isinstance(obj, dict):
                    states[f"{key}:{obj.get('state', '[no state]')}"] += 1
        records.append({
            "case_id": folder.name, "image_path": str(image), "raw_annotation_path": str(annotation),
            "golden_path": None, "annotation_format": "CVJsonStd_raw",
            "image_width": width, "image_height": height, "image_format": image_format, "image_mode": image_mode,
            "image_sha256": image_digest, "annotation_sha256": hashlib.sha256(payload).hexdigest(),
            "raw_source_kind": source_kind, "source_group": None, "source_group_status": "pending_original_engineering_review",
            "difficulty": None, "difficulty_status": "no_official_label_identified", "split": "unassigned",
            "counts_total": {k: len(data[k]) for k in ("components", "pins", "texts", "edges")},
            "counts_excluding_deleted": {"components": len(c), "pins": len(p), "texts": len(t), "edges": len(e)},
            "max_component_pin_count": max((len(v.get("pins", {})) for v in c.values()), default=0),
            "canvas_matches_image": canvas_matches, "image_verified": True, "annotation_readable": True,
            "image_suffix_matches_header": header_matches_suffix, "requires_png_conversion": image_format != "PNG",
            "spice_path": str(next(iter(sorted(folder.glob("*.net"))), "")) or None,
            "eds_path": str(next(iter(sorted(folder.glob("*.eds"))), "")) or None,
            "official_golden_status": "not_provided_in_submission_format",
            "warnings": case_warnings, "raw_reference_warning_counts": dict(dangling),
        })
    manifest_path = PROJECT_ROOT / "data/manifest.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8")
    duplicates = [ids for ids in image_hash_groups.values() if len(ids) > 1]
    report = {
        "audit_version": "0.1", "generated_at_utc": datetime.now(timezone.utc).isoformat(), "dataset_root": str(root),
        "case_directories": sum(p.is_dir() for p in root.iterdir()), "paired_readable_cases": len(records),
        "image_formats": dict(formats), "source_kind_counts": dict(source_kinds), "active_category_counts": dict(categories),
        "state_counts": dict(states), "json_top_level_layout_counts": dict(layouts),
        "pair_failures": failures, "canvas_dimension_mismatches": dimensions,
        "exact_duplicate_image_groups": duplicates, "raw_reference_warnings": warnings,
        "image_suffix_header_mismatches": [{"case_id": r["case_id"], "path": r["image_path"], "detected_format": r["image_format"]} for r in records if not r["image_suffix_matches_header"]],
        "requires_png_conversion_count": sum(r["requires_png_conversion"] for r in records),
        "official_submission_format_cases": 0,
        "notes": ["Readable pairs are raw development materials, not verified official Golden.", "Deleted objects are excluded for audit counts only; conversion policy is pending.", "Reference warnings are structural observations, not proof of annotation error.", "Source-kind labels are not engineering source groups; no train/dev/holdout split is assigned.", "Image SHA256 catches exact duplicates only, not related variants.", "Non-PNG image headers require pixel-preserving PNG conversion in preprocessing, even when the filename ends in .png.", "No official difficulty labels were found in inspected fields."],
    }
    write_json(PROJECT_ROOT / "data/dataset_audit.json", report)
    by_id = {r["case_id"]: r for r in records}
    chosen = []
    for case_id, role in [("0002", "medium_proxy"), ("0001", "hard_proxy")]:
        row = by_id.get(case_id)
        if row:
            chosen.append({"case_id": case_id, "development_role": role, "official_difficulty": None, "golden_status": row["official_golden_status"], "image_path": row["image_path"], "raw_annotation_path": row["raw_annotation_path"], "image_sha256": row["image_sha256"], "counts_excluding_deleted": row["counts_excluding_deleted"], "selection_basis": "manual visual review and geometry/text complexity; not an official difficulty label"})
    write_json(PROJECT_ROOT / "data/splits/quick_dev.json", {"split_version": "0.2", "status": "two_raw_development_cases_selected" if len(chosen) == 2 else "pending_cases", "purpose": "detector/OCR/interface debugging, not official F1", "cases": chosen, "notes": ["Official difficulty, Golden conversion and original-engineering grouping remain pending.", "Do not use raw annotation JSON in inference; load it only in offline audit/training/evaluation."]})
    print(json.dumps({"manifest_rows": len(records), "formats": dict(formats), "source_kinds": dict(source_kinds), "pair_failures": len(failures), "canvas_mismatches": len(dimensions), "duplicate_image_groups": len(duplicates), "cases_with_reference_warnings": len(warnings), "quick_dev_ids": [v["case_id"] for v in chosen]}, ensure_ascii=False, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
