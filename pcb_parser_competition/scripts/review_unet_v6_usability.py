"""Verify v6 inference and frozen development metrics without changing defaults."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pcb_parser.dataset_converter import load_source_annotation
from pcb_parser.unet_data import predict_tiled
from pcb_parser.unet_inference import decode_pins
from pcb_parser.unet_model import WirePinUNet
from verify_unet_contrast import path_specs, path_records, pin_metrics

OUT = ROOT / "runs/unet_v6_usability_review_20261015"
RUN = ROOT / "runs/training/unet_wire_pin_v6_corrected_relations_20261015"
DATA = ROOT / "data/segmentation_unet_v6_corrected_relations_20261015"
PRIOR = ROOT / "runs/unet_v6_paper_comparison_20261015"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def metrics(tp, fp, fn):
    return dict(tp=int(tp), fp=int(fp), fn=int(fn), precision=tp/max(1,tp+fp),
                recall=tp/max(1,tp+fn), dice=2*tp/max(1,2*tp+fp+fn), iou=tp/max(1,tp+fp+fn))


def total(values):
    return metrics(*(sum(row[key] for row in values) for key in ("tp", "fp", "fn")))


def render(rgb, old, new):
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"], "axes.unicode_minus": False})
    regions = [(536,31,788,119), (535,157,746,237)]
    fig, axes = plt.subplots(2, 3, figsize=(13,5.6), dpi=160, layout="constrained")
    for row, (x0,y0,x1,y1) in enumerate(regions):
        for col, (name, probability) in enumerate([( "原图", None), ("旧 v1", old), ("新 v6", new)]):
            panel = rgb[y0:y1,x0:x1].copy()
            if probability is not None:
                mask = probability[y0:y1,x0:x1] >= .5
                panel[mask] = (.2*panel[mask]+.8*np.array([0,170,95])).astype(np.uint8)
            axes[row,col].imshow(panel, interpolation="nearest")
            axes[row,col].axis("off")
            axes[row,col].set_title(name + (" · X1两侧仍有断点" if row == 0 and col == 2 else ""))
    fig.suptitle("0036 · 同一原图、原始尺度、阈值0.5；绿色为模型预测\nDOUT固定24px：0→24；X1左侧固定61px：0→46（整条仍未连通）")
    fig.savefig(OUT/"comparison_0036.png", facecolor="white")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    torch.set_num_threads(4)
    manifest = DATA/"training_manifest.jsonl"
    dataset = rows(manifest)
    val = [r for r in dataset if r["split"] == "val"]
    train = [r for r in dataset if r["split"] == "train"]
    assert len(train) == 171 and len(val) == 28
    assert val == rows(DATA/"frozen_validation_manifest.jsonl")
    old_val = [r for r in rows(ROOT/"data/segmentation_unet_v1/manifest.jsonl") if r["split"] == "val" and r["structurally_usable_draft"]]
    assert val == old_val
    weight = RUN/"best.pt"
    protected = {str(p): sha(p) for p in [weight, RUN/"last.pt", manifest,
        ROOT/"weights/unet_wire_pin_v1.pt", ROOT/"configs/unet_inference_v2.json", ROOT/"configs/project.json"]}
    for row in dataset:
        for key in ("image", "wire_mask", "pin_heatmap", "valid_mask"):
            if row.get(key):
                protected[row[key]] = sha(row[key])
    assert sha(weight) == read(RUN/"completion_checks.json")["checkpoint_sha256"]
    ckpt = torch.load(weight, map_location="cpu", weights_only=True)
    assert ckpt["epoch"] == 60 and ckpt["kind"] == "trained_on_draft_labels"
    assert ckpt["manifest_sha256"] == sha(manifest)
    model = WirePinUNet(ckpt["base_channels"])
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()
    assert all(torch.isfinite(v).all() for v in model.state_dict().values())
    split = read(ROOT/"runs/unet_inference_v2/split.json")
    assert split["source_checkpoint_sha256"] == sha(ROOT/"weights/unet_wire_pin_v1.pt")
    unique = set(split["development"] + split["acceptance_holdout"])
    assert len(unique) == 26
    raw = {r["case_id"]:r for r in rows(ROOT/"data/manifest.jsonl")}
    previous = {r["case_id"]:r for r in read(PRIOR/"evaluation.json")["per_page"]}
    # Expanded training set: check exact decoded image overlaps across splits.
    image_groups = defaultdict(list)
    for row in dataset:
        with Image.open(row["image"]) as im:
            rgb = np.asarray(im.convert("RGB"))
        digest = hashlib.sha256(str(rgb.shape).encode()+rgb.tobytes()).hexdigest()
        image_groups[digest].append(dict(case_id=row["case_id"], split=row["split"]))
    cross_split = [g for g in image_groups.values() if len({r["split"] for r in g}) > 1]
    write(OUT/"protocol.json", dict(checkpoint_sha256=sha(weight), device="cpu", dtype="float32",
        tile_size=512, stride=384, threshold=.5, fresh_inference_cases=[r["case_id"] for r in val],
        labels_used_for_scoring_only=True, default_adoption=False, training_performed=False,
        pin_diagnostic=dict(threshold=.8, kernel=5, refinement="quadratic", nms_radius=0,
            limit=500, matching_radius_px=5, status="same_fixed_native_v1_decoder_no_v6_tuning"),
        protected_sha256=protected))
    records = []
    for index, row in enumerate(val, 1):
        case = row["case_id"]
        folder = OUT/case
        folder.mkdir(exist_ok=True)
        with Image.open(row["image"]) as im:
            rgb = np.asarray(im.convert("RGB"))
        probability = predict_tiled(model, rgb, "cpu", 512, 384)
        assert probability.shape == (2, row["height"], row["width"])
        assert np.isfinite(probability).all() and probability.min() >= 0 and probability.max() <= 1
        with np.load(PRIOR/case/"probabilities.npz", allow_pickle=False) as cache:
            cached = cache["wire"]
        assert np.array_equal(probability[0], cached), case
        truth = np.asarray(Image.open(row["wire_mask"])) > 0
        prediction = probability[0] >= .5
        wire = metrics(int((prediction&truth).sum()), int((prediction&~truth).sum()), int((~prediction&truth).sum()))
        assert all(wire[k] == previous[case]["wire"][k] for k in ("tp", "fp", "fn"))
        annotation, _ = load_source_annotation(raw[case]["raw_annotation_path"])
        paths = path_records(prediction, path_specs(annotation, truth.shape))
        assert sum(p["connected_within_reference_tube"] for p in paths) == previous[case]["connected_paths"]
        tips = read(Path(row["wire_mask"]).parent/"pin_points.json")
        pins = pin_metrics(decode_pins(probability[1], .8, 5, "quadratic", 0, 500), tips)
        baseline_pin = None
        if case in unique:
            with np.load(ROOT/f"runs/unet_inference_v2/cache/{case}/scale_1.0.npz", allow_pickle=False) as cache:
                old_wire, old_pin = cache["probability"]
            baseline_pin = pin_metrics(decode_pins(old_pin, .8, 5, "quadratic", 0, 500), tips)
        record = dict(case_id=case, unique_comparison=case in unique, wire=wire,
            connected_paths=sum(p["connected_within_reference_tube"] for p in paths),
            total_paths=len(paths), native_pin_fixed_decoder=pins, baseline_native_pin=baseline_pin,
            existing_wire_cache_bitwise_equal=True)
        if case == "0036":
            probes = {}
            for name, y, x0, x1 in [("X1_left",80,620,681),("X1_right",80,725,765),("DOUT",181,605,629)]:
                probes[name] = dict(y=y, x_half_open=[x0,x1], pixels=x1-x0,
                    v1_detected=int((old_wire[y,x0:x1]>=.5).sum()),
                    v6_detected=int((probability[0,y,x0:x1]>=.5).sum()))
            record["fixed_probes"] = probes
            record["X1_DOUT_paths"] = [p for p in paths if p["edge_id"] in {"E_1","E_8","E_10"}]
            render(rgb, old_wire, probability[0])
        np.savez_compressed(folder/"probabilities.npz", wire=probability[0], pin_tip=probability[1])
        Image.fromarray(prediction.astype(np.uint8)*255).save(folder/"wire_mask.png")
        write(folder/"review.json", record)
        records.append(record)
        print(json.dumps(dict(phase="fresh_v6_cpu_inference", case_id=case, pages=index, total=28,
            elapsed_seconds=round(time.perf_counter()-started,1))), flush=True)
    selected = [r for r in records if r["unique_comparison"]]
    assert all(sha(p)==digest for p,digest in protected.items())
    result = dict(status="verified_candidate_usable_for_wire_segmentation_downstream_migration_pending",
        fresh_inference_pages=28, wire_cache_bitwise_matches=28,
        frozen_original_validation_rows_equal=True, exact_image_cross_split_groups=cross_split,
        source_engineering_overlap="unknown_no_group_ids", weight= str(weight.resolve()), weight_sha256=sha(weight),
        unique26=total([r["wire"] for r in selected]), all28=total([r["wire"] for r in records]),
        connected_paths=sum(r["connected_paths"] for r in selected), total_paths=sum(r["total_paths"] for r in selected),
        pin_fixed_native_decoder_v6=total([r["native_pin_fixed_decoder"] for r in selected]),
        pin_fixed_native_decoder_v1=total([r["baseline_native_pin"] for r in selected]),
        per_page=records, default_files_unchanged=True, model_or_bn_updates=False, independent_test=False,
        label_revision_status="partial_draft_repairs_not_full_page_certification",
        downstream_compatibility="run_unet.py works with explicit weights; old graph runners pin v1 SHA and caches",
        elapsed_seconds=time.perf_counter()-started)
    write(OUT/"verification.json", result)
    print(json.dumps({k:v for k,v in result.items() if k!="per_page"}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
