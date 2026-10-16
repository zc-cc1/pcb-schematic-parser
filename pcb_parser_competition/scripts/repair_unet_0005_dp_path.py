"""Recover the inspected 0005 DP path in a separate draft dataset revision."""
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pcb_parser.dataset_converter import load_source_annotation
from pcb_parser.unet_data import annotation_targets

OUTPUT = ROOT / "data/segmentation_unet_v3_local_path_repairs"
CASE = "0005"
EDGE = "E_29"
PATH = [[198, 119], [229, 119]]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def usable(stats):
    return (
        stats["physical_edges"] > 0
        and stats["wire_positive_pixels"] > 0
        and stats["pin_tips"] > 0
        and not any(stats[key] for key in (
            "missing_physical_paths", "out_of_bounds_paths",
            "invalid_pin_tips", "unowned_root_pins",
        ))
    )


def main():
    base_manifest = ROOT / "data/segmentation_unet_v1/manifest.jsonl"
    original_rows = read_rows(base_manifest)
    row = next(value for value in original_rows if value["case_id"] == CASE)
    source_row = next(value for value in read_rows(ROOT / "data/manifest.jsonl") if value["case_id"] == CASE)
    source_path = Path(source_row["raw_annotation_path"])
    config_path = ROOT / "configs/unet_wire_pin.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert row["split"] == "train"
    assert config["label_line_width"] == 3 and config["pin_sigma"] == 2
    protected_paths = [base_manifest, ROOT / "data/manifest.jsonl", config_path,
                       ROOT / "configs/unet_label_reviews.json", source_path]
    protected_paths += [Path(row[key]) for key in ("image", "wire_mask", "pin_heatmap")]
    protected_paths += [Path(row["wire_mask"]).parent / "pin_points.json"]
    checkpoint = ROOT / "weights/unet_wire_pin_v1.pt"
    if checkpoint.exists():
        protected_paths.append(checkpoint)
    before_hashes = {str(path): sha(path) for path in protected_paths}

    original, _ = load_source_annotation(source_path)
    edge = original["edges"][EDGE]
    assert edge["state"] == "active" and edge["category"] == "wire" and edge["isFlywire"] is False
    assert edge["keyPointPaths"] == {"wire": []}
    assert edge["coordinates"] == [[x, 119] for x in range(198, 230)]
    assert original["components"]["C_4"]["pins"]["P_85"]["tip"] == [197.0, 119.0]
    before_wire, before_pin, before_tips, before_stats = annotation_targets(
        original, row["width"], row["height"], 3, 2,
    )
    original_mask = np.asarray(Image.open(row["wire_mask"]))
    assert np.array_equal(np.asarray(before_wire), original_mask)
    assert np.array_equal(np.asarray(before_pin), np.asarray(Image.open(row["pin_heatmap"])))
    rgb = np.asarray(Image.open(row["image"]).convert("RGB"))
    assert not (original_mask[119, 198:230] > 0).any()
    # Confirm continuous green image evidence at every inspected source coordinate.
    pixels = rgb[119, 198:230].astype(int)
    assert np.all(pixels[:, 1] - pixels[:, 0] >= 20)
    assert np.all(pixels[:, 1] - pixels[:, 2] >= 20)

    revised = copy.deepcopy(original)
    revised["edges"][EDGE]["keyPointPaths"] = {"wire": [PATH]}
    wire, pin, tips, stats = annotation_targets(revised, row["width"], row["height"], 3, 2)
    before = np.asarray(before_wire) > 0
    after = np.asarray(wire) > 0
    added = after & ~before
    removed = before & ~after
    allowed = np.zeros_like(before)
    allowed[118:121, 198:230] = True
    assert not removed.any()
    assert np.array_equal(added, allowed), "Unexpected change outside the exact 3px path"
    assert int(added.sum()) == 96
    assert np.array_equal(np.asarray(before_pin), np.asarray(pin)) and before_tips == tips
    assert before_stats["missing_physical_paths"] == 1 and stats["missing_physical_paths"] == 0
    assert usable(stats)
    # The path field is the only source annotation change.
    round_trip = copy.deepcopy(revised)
    round_trip["edges"][EDGE]["keyPointPaths"] = copy.deepcopy(edge["keyPointPaths"])
    assert round_trip == original

    folder = OUTPUT / CASE
    folder.mkdir(parents=True, exist_ok=True)
    source_output = folder / "source_annotation.json"
    write_json(source_output, revised)
    wire.save(folder / "wire_mask.png")
    pin.save(folder / "pin_heatmap.png")
    write_json(folder / "pin_points.json", tips)
    Image.fromarray(added.astype(np.uint8) * 255).save(folder / "added_pixels.png")

    derived_rows = copy.deepcopy(original_rows)
    derived_row = next(value for value in derived_rows if value["case_id"] == CASE)
    derived_row.update(
        wire_mask=str((folder / "wire_mask.png").resolve()),
        pin_heatmap=str((folder / "pin_heatmap.png").resolve()),
        source_annotation_path=str(source_output.resolve()),
        source_annotation_sha256=sha(source_output),
        original_source_annotation_path=str(source_path),
        original_source_annotation_sha256=sha(source_path),
        label_status="draft_source_geometry_with_local_DP_path_repair_not_full_page_certified",
        structurally_usable_draft=True,
        stats=stats,
        local_repair={"edge_id": EDGE, "path": PATH, "added_pixels": 96},
    )
    write_json(folder / "metadata.json", derived_row)
    manifest_output = OUTPUT / "manifest.jsonl"
    manifest_output.write_text("".join(json.dumps(value, ensure_ascii=False) + "\n" for value in derived_rows), encoding="utf-8")
    assert len(derived_rows) == len(original_rows) == 300
    assert all(a == b for a, b in zip(original_rows, derived_rows) if a["case_id"] != CASE)

    # Diagnostic plots show the same image/label window in original coordinates.
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"], "axes.unicode_minus": False})
    x0, y0, x1, y1 = 145, 94, 248, 143
    extent = (x0 - .5, x1 - .5, y1 - .5, y0 - .5)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), dpi=160, layout="constrained")
    axes[0].imshow(rgb[y0:y1, x0:x1], extent=extent, interpolation="nearest")
    axes[0].set_title("原图：DP 引线 / 10 / RPI-DP")
    for ax, mask, title in ((axes[1], before, "补标前：目标段为背景"), (axes[2], after, "补标后：新增 96 个导线像素")):
        ax.imshow(mask[y0:y1, x0:x1], extent=extent, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(title)
    for ax in axes:
        ax.add_patch(Rectangle((197.5, 117.5), 32, 3, fill=False, edgecolor="#e45d23", linewidth=1.6))
        ax.set_xlim(x0 - .5, x1 - .5)
        ax.set_ylim(y1 - .5, y0 - .5)
        ax.set_xlabel("原图 x 坐标")
        ax.set_ylabel("原图 y 坐标")
    fig.suptitle("0005 局部路径修订 · 橙框为补标段 · 白色为导线标签")
    comparison = folder / "comparison.png"
    fig.savefig(comparison, facecolor="white")
    plt.close(fig)

    assert all(sha(path) == digest for path, digest in before_hashes.items())
    summary = dict(
        status="local_draft_path_repair_complete", case_id=CASE, edge_id=EDGE,
        path=PATH, nominal_stroke_width=3, added_pixels=96, removed_pixels=0,
        original_stats=before_stats, revised_stats=stats,
        original_files_sha256=before_hashes,
        original_files_unchanged=True, pin_heatmap_and_points_unchanged=True,
        full_page_certified=False, official_score=None,
        manifest=str(manifest_output.resolve()), revised_source=str(source_output.resolve()),
        comparison=str(comparison.resolve()),
        dataset_cases=300,
        structurally_usable_drafts=sum(value["structurally_usable_draft"] for value in derived_rows),
        usable_by_split=dict(Counter(value["split"] for value in derived_rows if value["structurally_usable_draft"])),
        training_started=False, default_training_manifest_changed=False,
        validation_rows_unchanged=True,
        review_basis="32 consecutive horizontal coordinates, active non-flywire source wire E_29, continuous green image evidence, P_85 pin boundary preserved",
        remaining_scope="Local source-path recovery only; other routes, line widths and wire/pin semantics are still draft supervision.",
        outputs_sha256={str(path): sha(path) for path in (source_output, folder / "wire_mask.png", folder / "pin_heatmap.png", folder / "pin_points.json", manifest_output)},
    )
    write_json(OUTPUT / "repair_summary.json", summary)
    print(json.dumps({key: summary[key] for key in ("status", "added_pixels", "removed_pixels", "usable_by_split", "manifest", "comparison", "original_files_unchanged", "pin_heatmap_and_points_unchanged")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
