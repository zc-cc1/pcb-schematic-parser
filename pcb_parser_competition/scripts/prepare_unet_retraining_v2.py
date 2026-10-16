"""Versioned, partially reviewed training labels. Original labels stay immutable."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from pcb_parser.dataset_converter import load_source_annotation
from pcb_parser.unet_data import annotation_targets


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    out = ROOT/"data/segmentation_unet_v2_reviewed_draft"
    if out.exists():
        raise ValueError("New label version already exists; never overwrite it")
    patch_path = ROOT/"configs/unet_retraining_v2_label_corrections.json"
    patches = json.loads(patch_path.read_text(encoding="utf-8"))["corrections"]
    rows = list(map(json.loads, (ROOT/"data/segmentation_unet_v1/manifest.jsonl").read_text(encoding="utf-8").splitlines()))
    raw = {r["case_id"]: r for r in map(json.loads, (ROOT/"data/manifest.jsonl").read_text(encoding="utf-8").splitlines())}
    split = json.loads((ROOT/"runs/unet_inference_v2/split.json").read_text(encoding="utf-8"))
    val_ids = set(split["development"] + split["acceptance_holdout"])
    rows = [r for r in rows if r["structurally_usable_draft"] and (r["split"] == "train" or r["case_id"] in val_ids)]
    assert len([r for r in rows if r["split"] == "train"]) == 103
    assert len([r for r in rows if r["split"] == "val"]) == 26
    assert all(next(r for r in rows if r["case_id"] == c)["split"] == "train" for c in patches)
    protected = {str(p): sha(p) for p in (ROOT/"weights/unet_wire_pin_v1.pt", ROOT/"configs/unet_inference_v2.json", patch_path)}
    for row in rows:
        for name in ("wire_mask.png", "pin_heatmap.png", "pin_points.json", "metadata.json"):
            p = ROOT/"data/segmentation_unet_v1"/row["case_id"]/name
            protected[str(p)] = sha(p)
        p = ROOT/f"data/standardized/{row['case_id']}/{row['case_id']}.png"
        assert sha(p) == row["image_sha256"]
        protected[str(p)] = sha(p)
    out.mkdir(parents=True)
    corrected = []
    records = []
    for original in rows:
        row = copy.deepcopy(original); c = row["case_id"]
        old = ROOT/"data/segmentation_unet_v1"/c; folder = out/c
        folder.mkdir()
        for name in ("wire_mask.png", "pin_heatmap.png", "pin_points.json"):
            shutil.copy2(old/name, folder/name)
        row["image"] = str((ROOT/f"data/standardized/{c}/{c}.png").resolve())
        row["wire_mask"] = str((folder/"wire_mask.png").resolve())
        row["pin_heatmap"] = str((folder/"pin_heatmap.png").resolve())
        row["label_status"] = "draft_with_partial_agent_visual_corrections_not_certified"
        row["full_page_manually_approved"] = False
        if c in patches:
            a, _ = load_source_annotation(raw[c]["raw_annotation_path"])
            assert sha(raw[c]["raw_annotation_path"]) == row["source_annotation_sha256"]
            a = copy.deepcopy(a); patch = patches[c]
            for eid, paths in patch["replace_edges"].items():
                a["edges"][eid]["keyPointPaths"] = {"wire": paths}
            for eid in patch["activate_edges"]:
                a["edges"][eid]["state"] = "active"
            for eid, paths in patch["add_edges"].items():
                assert eid not in a["edges"]
                a["edges"][eid] = {"state": "active", "category": "wire", "isFlywire": False, "keyPointPaths": {"wire": paths}}
            for tip in patch["pin_tip_overrides"]:
                cid, pid = tip["component_source_id"], tip["pin_source_id"]
                a["components"][cid]["pins"][pid]["tip"] = tip["point"]
                if pid in a.get("pins", {}):
                    a["pins"][pid]["tip"] = tip["point"]
            wire, heat, tips, stats = annotation_targets(a, row["width"], row["height"], 3, 2.)
            wire.save(folder/"wire_mask.png"); heat.save(folder/"pin_heatmap.png")
            (folder/"pin_points.json").write_text(json.dumps(tips, ensure_ascii=False, indent=2), encoding="utf-8")
            row["stats"] = stats
            row["visual_correction"] = patch
            before = np.asarray(Image.open(old/"wire_mask.png")) > 0
            after = np.asarray(wire) > 0
            corrected.append({"case_id": c, "added_wire_pixels": int((after & ~before).sum()), "removed_wire_pixels": int((before & ~after).sum()),
                              "pin_tips_changed": len(patch["pin_tip_overrides"]), "reason": patch["reason"]})
        row["wire_mask_sha256"] = sha(folder/"wire_mask.png")
        row["pin_heatmap_sha256"] = sha(folder/"pin_heatmap.png")
        row["pin_points_sha256"] = sha(folder/"pin_points.json")
        if row["split"] == "val":
            assert row["wire_mask_sha256"] == sha(old/"wire_mask.png")
            assert row["pin_heatmap_sha256"] == sha(old/"pin_heatmap.png")
        (folder/"metadata.json").write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
        records.append(row)
    assert all(sha(p) == digest for p, digest in protected.items())
    (out/"manifest.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False)+"\n" for r in records), encoding="utf-8")
    summary = {"status": "partial_agent_review_draft_training_experiment", "train_cases": 103, "validation_unique_cases": 26,
               "validation_labels_changed": False, "validation_added_to_training": False,
               "full_page_human_approved_cases": 0, "corrected_training_cases": corrected, "original_files_unchanged": True,
               "protected_source_sha256": protected, "patch_sha256": sha(patch_path), "official_score": None}
    (out/"summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    # Review the new labels visually before any GPU training is started.
    review_out = ROOT/"runs/unet_retrain_v2/label_review"
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"]})
    rois = {"0073": [(430, 80, 746, 117), (440, 207, 641, 316)], "0088": [(304, 142, 379, 181), (314, 95, 424, 139)]}
    for c, boxes in rois.items():
        rgb = np.asarray(Image.open(ROOT/f"data/standardized/{c}/{c}.png").convert("RGB"))
        m = np.asarray(Image.open(out/c/"wire_mask.png")) > 0
        heat = np.asarray(Image.open(out/c/"pin_heatmap.png")) > 76
        overlay = rgb.copy(); overlay[m] = (.3*overlay[m]+.7*np.array([230, 60, 35])).astype(np.uint8)
        overlay[heat] = (.3*overlay[heat]+.7*np.array([0, 115, 240])).astype(np.uint8)
        fig, axes = plt.subplots(2, 2, figsize=(12, 6), dpi=150, layout="constrained")
        for i, (x0, y0, x1, y1) in enumerate(boxes):
            for j, im in enumerate((rgb, overlay)):
                axes[i, j].imshow(im[y0:y1, x0:x1], interpolation="nearest"); axes[i, j].axis("off")
                axes[i, j].set_title("原图" if j == 0 else "新标签：红=导线，蓝=tip")
        fig.suptitle(f"{c} · 修正后训练标签（原始标签保留）")
        fig.savefig(review_out/f"{c}_corrected.png"); plt.close(fig)
    print(json.dumps({k: v for k, v in summary.items() if k != "protected_source_sha256"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
