"""Add auditable, training-only context crops without changing validation truth."""
import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path

import yaml
from PIL import Image

from prepare_detection_dataset import file_hash, render_gallery

ROOT = Path(__file__).resolve().parents[1]


def write_yolo_labels(objects, path, width, height):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for item in objects:
        x1, y1, x2, y2 = item["bbox"]
        values = [(x1+x2)/2/width, (y1+y2)/2/height, (x2-x1)/width, (y2-y1)/height]
        assert all(0 <= v <= 1 for v in values)
        lines.append(f"{item['class_id']} " + " ".join(f"{v:.9f}" for v in values))
    path.write_text("\n".join(lines)+"\n", encoding="utf-8")


def intersects(a, b):
    return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])


def contains(window, box):
    return all((window[0] <= box[0], window[1] <= box[1], window[2] >= box[2], window[3] >= box[3]))


def complete_window(initial, objects, width, height, maximum_side, margin):
    """Expand a crop until none of the annotated symbols on its border is cut."""
    window = list(initial)
    for _ in range(len(objects) + 1):
        old = window[:]
        for item in objects:
            box = item["bbox"]
            if intersects(window, box) and not contains(window, box):
                window = [max(0, min(window[0], math.floor(box[0])-margin)),
                          max(0, min(window[1], math.floor(box[1])-margin)),
                          min(width, max(window[2], math.ceil(box[2])+margin)),
                          min(height, max(window[3], math.ceil(box[3])+margin))]
        if max(window[2]-window[0], window[3]-window[1]) > maximum_side:
            return None
        if window == old:
            assert all(not intersects(window, o["bbox"]) or contains(window, o["bbox"]) for o in objects)
            return window
    raise RuntimeError("Crop expansion did not converge")


def project_objects(objects, window):
    left, top, right, bottom = window
    projected = []
    for item in objects:
        box = item["bbox"]
        if not intersects(window, box):
            continue
        if not contains(window, box):
            raise ValueError("Crop cuts an annotated component")
        local = [box[0]-left, box[1]-top, box[2]-left, box[3]-top]
        assert 0 <= local[0] < local[2] <= right-left
        assert 0 <= local[1] < local[3] <= bottom-top
        projected.append({**item, "parent_bbox": box, "bbox": local})
    return projected


def assert_split_isolation(rows):
    train = [r for r in rows if r["split"] == "train"]
    val = [r for r in rows if r["split"] == "val"]
    for key in ("group", "source_image_sha256"):
        if {r[key] for r in train} & {r[key] for r in val}:
            raise ValueError(f"Train/validation overlap: {key}")
    train_ids = {r["case_id"] for r in train if r.get("sample_kind", "whole") == "whole"}
    if any(r.get("sample_kind") == "context_crop" and r["parent_case_id"] not in train_ids for r in train):
        raise ValueError("Crop parent must be an original training case")
    if any(r.get("sample_kind") == "context_crop" for r in val):
        raise ValueError("Validation crops are prohibited")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/detection_augmentation.json")
    parser.add_argument("--review-only", action="store_true")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    source_manifest = ROOT / cfg["source_manifest"]
    rows = [json.loads(v) for v in source_manifest.read_text(encoding="utf-8").splitlines()]
    assert_split_isolation(rows)
    targets = [(row, obj) for row in rows if row["split"] == "train"
               for obj in row["objects"] if obj["class_name"] in cfg["crop_budgets"]]
    review = ROOT / "artifacts/detection_balanced_review"
    review.mkdir(parents=True, exist_ok=True)
    review_objects = [{**obj, "image_path": row["image"]} for row, obj in targets if obj["class_name"] in ("crystal", "mosfet")]
    galleries = render_gallery(review_objects, review, "training_rare_classes", cell_width=280, cell_height=210, per_page=30)
    (review / "review_objects.json").write_text(json.dumps(review_objects, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.review_only:
        print(json.dumps({"review_objects": len(review_objects), "galleries": galleries}, ensure_ascii=False))
        return
    for row in rows:
        if file_hash(row["image"]) != row["image_sha256"] or file_hash(row["label"]) != row["label_sha256"]:
            raise RuntimeError(f"Original exported inputs changed: {row['case_id']}")
    folder = ROOT / cfg["output_directory"]
    folder.mkdir(parents=True, exist_ok=True)
    names = json.loads((ROOT / "configs/detection_dataset.json").read_text(encoding="utf-8"))["class_names"]
    corrections = []
    for row in rows:
        changes = cfg.get("training_label_corrections", {}).get(row["case_id"], {})
        if not changes:
            continue
        if row["split"] != "train":
            raise ValueError("This stage cannot modify validation labels")
        for component_id, change in changes.items():
            item = next(o for o in row["objects"] if o["component_id"] == component_id)
            corrections.append({"case_id": row["case_id"], "component_id": component_id,
                                "original_class": item["class_name"], **change})
            item.update(class_name=change["class_name"], class_id=names.index(change["class_name"]),
                        augmentation_stage_correction=change)
        with Image.open(row["image"]) as image:
            label_path = folder / f"labels/train/{row['case_id']}.txt"
            write_yolo_labels(row["objects"], label_path, image.width, image.height)
        row.update(original_label_sha256=row["label_sha256"], label=str(label_path), label_sha256=file_hash(label_path),
                   training_label_corrections=changes)
    targets = [(row, obj) for row in rows if row["split"] == "train"
               for obj in row["objects"] if obj["class_name"] in cfg["crop_budgets"]]
    exported = [{**r, "sample_kind": "whole"} for r in rows]
    seen, crop_counts, skipped = set(), Counter(), Counter()
    random.Random(cfg["seed"]).shuffle(targets)
    # Round-robin variants expose more independent objects before repeating them.
    for variant in range(cfg["maximum_variants_per_object"]):
        for row, obj in targets:
            name = obj["class_name"]
            if crop_counts[name] >= cfg["crop_budgets"][name]:
                continue
            with Image.open(row["image"]) as source:
                width, height = source.size
                box = obj["bbox"]
                side = max(cfg["crop_sizes"][variant % len(cfg["crop_sizes"])], math.ceil(max(box[2]-box[0], box[3]-box[1])+64))
                cw, ch = min(side, width), min(side, height)
                left = max(0, min(width-cw, int((box[0]+box[2]-cw)/2)))
                top = max(0, min(height-ch, int((box[1]+box[3]-ch)/2)))
                window = complete_window([left, top, left+cw, top+ch], row["objects"], width, height,
                                         cfg["maximum_crop_side"], cfg["boundary_margin"])
                if window is None:
                    skipped["expansion_exceeded_maximum_side"] += 1
                    continue
                key = (row["case_id"], *window)
                if key in seen or window == [0, 0, width, height]:
                    skipped["duplicate_or_whole_page"] += 1
                    continue
                seen.add(key)
                sample_id = f"{row['case_id']}_crop_{len(seen):04d}"
                projected = project_objects(row["objects"], window)
                if not any(o["component_id"] == obj["component_id"] for o in projected):
                    raise RuntimeError("Target disappeared from crop")
                image_path = folder / f"images/train/{sample_id}.png"
                label_path = folder / f"labels/train/{sample_id}.txt"
                image_path.parent.mkdir(parents=True, exist_ok=True)
                label_path.parent.mkdir(parents=True, exist_ok=True)
                source.convert("RGB").crop(window).save(image_path)
            crop_width, crop_height = window[2]-window[0], window[3]-window[1]
            for item in projected:
                item["image_path"] = str(image_path)
            write_yolo_labels(projected, label_path, crop_width, crop_height)
            exported.append({"case_id": sample_id, "parent_case_id": row["case_id"], "sample_kind": "context_crop",
                             "group": row["group"], "split": "train", "source_kind": row["source_kind"],
                             "source_annotation_sha256": row["source_annotation_sha256"], "source_image_sha256": row["source_image_sha256"],
                             "image": str(image_path), "image_sha256": file_hash(image_path), "label": str(label_path),
                             "label_sha256": file_hash(label_path), "objects": projected, "crop_window": window,
                             "crop_width": crop_width, "crop_height": crop_height, "target_class": name,
                             "target_component_id": obj["component_id"], "variant": variant})
            crop_counts[name] += 1
    assert_split_isolation(exported)
    original_val = [r for r in rows if r["split"] == "val"]
    exported_val = [{k:v for k,v in r.items() if k != "sample_kind"} for r in exported if r["split"] == "val"]
    assert original_val == exported_val
    counts = {split: Counter(o["class_name"] for r in exported if r["split"] == split for o in r["objects"]) for split in ("train", "val")}
    for split in ("train", "val"):
        (folder / f"{split}.txt").write_text("\n".join(Path(r["image"]).as_posix() for r in exported if r["split"] == split)+"\n", encoding="utf-8")
    (folder / "dataset.yaml").write_text(yaml.safe_dump({"path": folder.as_posix(), "train": "train.txt", "val": "val.txt", "names": dict(enumerate(names))}, sort_keys=False), encoding="utf-8")
    (folder / "manifest.jsonl").write_text("\n".join(json.dumps(r,ensure_ascii=False) for r in exported)+"\n", encoding="utf-8")
    audit = {"config": cfg, "source_manifest_sha256": file_hash(source_manifest),
             "image_counts": dict(Counter(r["split"] for r in exported)), "crop_counts": dict(crop_counts),
             "label_counts": {k:dict(v) for k,v in counts.items()}, "skipped": dict(skipped),
             "original_training_pages": sum(r["split"] == "train" for r in rows),
             "validation_unchanged": True, "train_validation_group_overlap": 0,
             "independent_target_objects": dict(Counter(o["class_name"] for _,o in targets)),
             "training_label_corrections": corrections,
             "review_galleries": galleries, "label_status": "frozen_v1_partially_reviewed_annotations_not_official_golden"}
    (folder / "audit.json").write_text(json.dumps(audit,ensure_ascii=False,indent=2), encoding="utf-8")
    gallery_records = []
    for name in cfg["crop_budgets"]:
        samples = [r for r in exported if r.get("target_class") == name]
        for row in samples[:6]:
            item = next(o for o in row["objects"] if o["component_id"] == row["target_component_id"])
            gallery_records.append({**item, "case_id": row["case_id"]})
    render_gallery(gallery_records, review, "crop_examples", cell_width=280, cell_height=210, per_page=24)
    print(json.dumps({k:audit[k] for k in ("image_counts","crop_counts","label_counts","skipped","validation_unchanged")},ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
