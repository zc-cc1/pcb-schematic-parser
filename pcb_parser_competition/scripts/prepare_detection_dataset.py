"""Export raw bbox labels, group duplicates, audit data and render review samples."""
import hashlib
import json
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont
from scipy.fft import dctn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pcb_parser.dataset_converter import load_source_annotation


class Groups:
    def __init__(self, ids):
        self.parent = dict(zip(ids, ids))

    def find(self, item):
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def perceptual_hash(image):
    small = np.asarray(image.convert("L").resize((32, 32), Image.Resampling.LANCZOS), dtype=float)
    coefficients = dctn(small, type=2, norm="ortho")[:8, :8].flatten()
    median = np.median(coefficients[1:])
    return sum(int(value > median) << idx for idx, value in enumerate(coefficients))


def bbox_values(box):
    try:
        values = [float(box[key]) for key in ("x1", "y1", "x2", "y2")]
        return values if np.isfinite(values).all() else None
    except (KeyError, TypeError, ValueError):
        return None


def font(size):
    return ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", size) if Path("C:/Windows/Fonts/msyh.ttc").exists() else ImageFont.load_default()


def render_gallery(records, folder, prefix, cell_width=240, cell_height=170, per_page=36):
    paths = []
    for page, offset in enumerate(range(0, len(records), per_page), 1):
        selected = records[offset:offset+per_page]
        cols = 6
        rows = (len(selected) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * cell_width, rows * cell_height), "white")
        draw = ImageDraw.Draw(sheet)
        for idx, item in enumerate(selected):
            x, y = idx % cols * cell_width, idx // cols * cell_height
            with Image.open(item["image_path"]) as source:
                x1, y1, x2, y2 = item["bbox"]
                margin = 20
                left, top = max(0, int(x1)-margin), max(0, int(y1)-margin)
                right, bottom = min(source.width, int(x2)+margin), min(source.height, int(y2)+margin)
                crop = source.convert("RGB").crop((left, top, right, bottom))
                cd = ImageDraw.Draw(crop)
                cd.rectangle((x1-left, y1-top, x2-left, y2-top), outline="#D62728", width=2)
                crop.thumbnail((cell_width-12, cell_height-46), Image.Resampling.LANCZOS)
                sheet.paste(crop, (x+(cell_width-crop.width)//2, y+42+(cell_height-46-crop.height)//2))
            title = f"{item['case_id']}/{item['component_id']} {item['source_category']}"
            draw.text((x+4, y+3), title, fill="black", font=font(15))
            draw.text((x+4, y+23), ",".join(item.get("designators", []))[:26], fill="#333333", font=font(13))
            draw.rectangle((x, y, x+cell_width-1, y+cell_height-1), outline="#BBBBBB")
        path = folder / f"{prefix}_{page:02d}.png"
        sheet.save(path)
        paths.append(str(path))
    return paths


def main():
    cfg = json.loads((ROOT / "configs/detection_dataset.json").read_text(encoding="utf-8"))
    overrides = json.loads((ROOT / "configs/detection_label_overrides.json").read_text(encoding="utf-8"))
    source_rows = [json.loads(line) for line in (ROOT / "data/manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    folder = ROOT / "data/detection_v1"
    review_dir = ROOT / "artifacts/detection_data_review"
    folder.mkdir(parents=True, exist_ok=True)
    review_dir.mkdir(parents=True, exist_ok=True)
    names = cfg["class_names"]
    groups = Groups([r["case_id"] for r in source_rows])
    rows, all_objects, issues, grouping_evidence = [], [], [], []
    source_files, by_image_hash = {}, {}
    for row in source_rows:
        case_id = row["case_id"]
        annotation = Path(row["raw_annotation_path"])
        if file_hash(annotation) != row["annotation_sha256"] or file_hash(row["image_path"]) != row["image_sha256"]:
            raise RuntimeError(f"Source changed since intake: {case_id}")
        raw, duplicates = load_source_annotation(annotation)
        image_path = ROOT / f"data/standardized/{case_id}/{case_id}.png"
        with Image.open(image_path) as image:
            width, height = image.size
            phash = perceptual_hash(image)
        if (width, height) != (row["image_width"], row["image_height"]):
            raise RuntimeError(f"Image/canvas mismatch: {case_id}")
        other = by_image_hash.get(row["image_sha256"])
        if other:
            groups.union(case_id, other)
            grouping_evidence.append({"cases": [other, case_id], "reason": "exact_image_sha256"})
        by_image_hash[row["image_sha256"]] = case_id
        source_titles = []
        for text in raw["texts"].values():
            match = re.fullmatch(r"\s*File:\s*(.+\.kicad_sch)\s*", text.get("content", ""), re.I)
            if match:
                key = match.group(1).lower()
                source_titles.append(key)
                if key in source_files:
                    groups.union(case_id, source_files[key])
                    grouping_evidence.append({"cases": [source_files[key], case_id], "reason": "same_available_source_file_text", "value": key})
                source_files[key] = case_id
        objects, seen = [], set()
        for component_id, component in raw["components"].items():
            if component.get("state") == "deleted":
                continue
            correction = overrides["components"].get(case_id, {}).get(component_id, {})
            if correction.get("exclude"):
                issues.append({"case_id": case_id, "component_id": component_id, "reason": "manual_exclusion", "evidence": correction})
                continue
            category = correction.get("category", component["category"])
            class_name = cfg["source_to_class"].get(category)
            box = bbox_values(correction.get("location", component.get(cfg["bbox_source"])))
            if class_name not in names or box is None:
                issues.append({"case_id": case_id, "component_id": component_id, "reason": "invalid_category_or_bbox"})
                continue
            original_box = box[:]
            box = [max(0., box[0]), max(0., box[1]), min(float(width), box[2]), min(float(height), box[3])]
            if min(box[2]-box[0], box[3]-box[1]) < cfg["minimum_bbox_side_px"]:
                issues.append({"case_id": case_id, "component_id": component_id, "reason": "too_small_or_degenerate", "bbox": original_box})
                continue
            if box != original_box:
                issues.append({"case_id": case_id, "component_id": component_id, "reason": "bbox_clipped_to_image", "original": original_box, "exported": box})
            key = (class_name, *box)
            if key in seen:
                issues.append({"case_id": case_id, "component_id": component_id, "reason": "identical_detection_label_removed"})
                continue
            seen.add(key)
            designators = [t.get("content", "") for t in component.get("associatedTexts", {}).values() if t.get("details") == "designator"]
            item = {"case_id": case_id, "component_id": component_id, "source_category": category, "class_name": class_name,
                    "class_id": names.index(class_name), "bbox": box, "real_bbox": component.get("realBBox"),
                    "image_path": str(image_path), "designators": designators, "manual_override": correction or None}
            objects.append(item)
            all_objects.append(item)
        rows.append({**row, "canonical_image_path": str(image_path), "phash": phash, "source_file_text": source_titles,
                     "duplicate_json_keys_collapsed": len(duplicates), "detection_objects": objects})
    for index, row in enumerate(rows):
        for other in rows[:index]:
            aspect = row["image_width"] / row["image_height"]
            other_aspect = other["image_width"] / other["image_height"]
            distance = (row["phash"] ^ other["phash"]).bit_count()
            if distance <= cfg["near_duplicate_phash_max_distance"] and abs(aspect / other_aspect - 1) <= cfg["near_duplicate_aspect_ratio_relative_tolerance"]:
                groups.union(row["case_id"], other["case_id"])
                grouping_evidence.append({"cases": [other["case_id"], row["case_id"]], "reason": "near_duplicate_phash_aspect", "distance": distance})
    members = defaultdict(list)
    for row in rows:
        members[groups.find(row["case_id"])].append(row)
    val_groups = {groups.find(cid) for cid in cfg["fixed_validation_cases"]}
    rng = random.Random(cfg["seed"])
    # Stratify by source style while selecting entire groups.
    for kind in sorted({r["raw_source_kind"] for r in rows}):
        eligible = [g for g, cases in members.items() if any(c["raw_source_kind"] == kind for c in cases)]
        rng.shuffle(eligible)
        target = round(sum(r["raw_source_kind"] == kind for r in rows) * cfg["val_fraction"])
        while sum(r["raw_source_kind"] == kind for g in val_groups for r in members[g]) < target:
            candidate = next((g for g in eligible if g not in val_groups), None)
            if candidate is None:
                break
            val_groups.add(candidate)
    exported, representative_ids, redundant = [], {}, []
    label_counts = {"train": Counter(), "val": Counter()}
    image_counts = Counter()
    for row in rows:
        case_id = row["case_id"]
        group = groups.find(case_id)
        split = "val" if group in val_groups else "train"
        # Exact image duplicates use one representative; never weight repeated scans twice.
        if row["image_sha256"] in representative_ids:
            redundant.append({"case_id": case_id, "representative": representative_ids[row["image_sha256"]], "group": group, "split": split})
            continue
        representative_ids[row["image_sha256"]] = case_id
        image_dir, label_dir = folder / f"images/{split}", folder / f"labels/{split}"
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        destination = image_dir / f"{case_id}.png"
        shutil.copy2(row["canonical_image_path"], destination)
        labels = []
        for item in row["detection_objects"]:
            x1, y1, x2, y2 = item["bbox"]
            xywh = [(x1+x2)/2/row["image_width"], (y1+y2)/2/row["image_height"], (x2-x1)/row["image_width"], (y2-y1)/row["image_height"]]
            assert all(0 <= v <= 1 for v in xywh)
            labels.append(f"{item['class_id']} " + " ".join(f"{v:.9f}" for v in xywh))
            label_counts[split][item["class_name"]] += 1
        label_path = label_dir / f"{case_id}.txt"
        label_path.write_text("\n".join(labels)+"\n", encoding="utf-8")
        image_counts[split] += 1
        exported.append({"case_id": case_id, "group": group, "split": split, "source_kind": row["raw_source_kind"],
                         "source_annotation_sha256": row["annotation_sha256"], "source_image_sha256": row["image_sha256"],
                         "image": str(destination), "image_sha256": file_hash(destination), "label": str(label_path),
                         "label_sha256": file_hash(label_path), "objects": row["detection_objects"]})
    assert not ({r["group"] for r in exported if r["split"] == "train"} & {r["group"] for r in exported if r["split"] == "val"})
    assert all(next(r for r in exported if r["case_id"] == cid)["split"] == "val" for cid in cfg["fixed_validation_cases"])
    (folder / "dataset.yaml").write_text(yaml.safe_dump({"path": folder.as_posix(), "train": "images/train", "val": "images/val", "names": dict(enumerate(names))}, allow_unicode=True, sort_keys=False), encoding="utf-8")
    # Explicit lists prevent stale files from old split policies entering future runs.
    for split in ("train", "val"):
        (folder / f"{split}.txt").write_text("\n".join(Path(r["image"]).as_posix() for r in exported if r["split"] == split)+"\n", encoding="utf-8")
    (folder / "dataset.yaml").write_text(yaml.safe_dump({"path": folder.as_posix(), "train": "train.txt", "val": "val.txt", "names": dict(enumerate(names))}, allow_unicode=True, sort_keys=False), encoding="utf-8")
    (folder / "manifest.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in exported)+"\n", encoding="utf-8")
    class_samples = []
    for name in names:
        examples = [o for o in all_objects if o["class_name"] == name]
        # Spread crops across cases rather than reviewing adjacent objects only.
        selected = []
        for obj in examples:
            if obj["case_id"] not in {s["case_id"] for s in selected}:
                selected.append(obj)
            if len(selected) == 2:
                break
        class_samples.extend(selected)
    galleries = render_gallery(class_samples, review_dir, "class_samples")
    suspicious = []
    expected_prefixes = {"Fuse": r"F\d+", "C": r"C\d+", "R": r"R\d+", "L": r"L\d+", "Crystal": r"[XY]\d+"}
    for item in all_objects:
        expected = expected_prefixes.get(item["source_category"])
        if expected and item["designators"] and all(not re.fullmatch(expected, d, re.I) for d in item["designators"]):
            suspicious.append(item)
    galleries += render_gallery(suspicious[:108], review_dir, "designator_category_candidates")
    (review_dir / "sample_objects.json").write_text(json.dumps({"class_samples": class_samples, "suspicious_designators": suspicious}, ensure_ascii=False, indent=2), encoding="utf-8")
    report = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "config": cfg, "overrides": overrides,
              "status": "experimental_detection_dataset_prepared", "source_cases": len(rows), "exported_cases": len(exported),
              "image_counts": dict(image_counts), "label_counts": {k: dict(v) for k, v in label_counts.items()},
              "duplicate_images_removed": redundant, "group_count": len(members), "groups": {g: [r["case_id"] for r in cases] for g, cases in members.items()},
              "grouping_evidence": grouping_evidence, "issues": issues, "review_galleries": galleries,
              "review_status": "class samples and suspicious labels require visual inspection; source labels are not all manually certified",
              "source_grouping_limit": "Exact/near duplicates and available file text are isolated; original engineering provenance is incomplete",
              "dataset_yaml": str(folder / "dataset.yaml"), "split_overlap_groups": 0}
    (folder / "audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"source_cases": len(rows), "exported_cases": len(exported), "image_counts": dict(image_counts), "classes": len(names),
                      "label_counts": {k:sum(v.values()) for k,v in label_counts.items()}, "issues": len(issues), "suspicious_labels": len(suspicious), "galleries": galleries}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
