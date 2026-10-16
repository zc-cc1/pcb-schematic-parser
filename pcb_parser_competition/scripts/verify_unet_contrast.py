"""Offline fixed-gamma U-Net diagnosis; never adopt settings or repair predictions.

Prediction receives image pixels and the frozen checkpoint only. Source labels
are read afterwards for draft pixel metrics and local path connectivity proxies.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pcb_parser.dataset_converter import load_source_annotation
from pcb_parser.unet_data import polylines, predict_tiled
from pcb_parser.unet_inference import decode_pins
from pcb_parser.unet_model import WirePinUNet

GAMMAS = (1.0, 1.5, 2.0)  # Fixed before evaluation; no per-page choice.
MIN_PATH_LENGTH = 10.0
PATH_RADIUS = 2


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metrics(tp, fp, fn):
    return {"tp": int(tp), "fp": int(fp), "fn": int(fn),
            "precision": tp / (tp + fp) if tp + fp else 0.,
            "recall": tp / (tp + fn) if tp + fn else 0.,
            "dice": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.}


def pin_metrics(candidates, tips):
    predicted = np.asarray([c["point"] for c in candidates])
    expected = np.asarray([p["point"] for p in tips])
    if not len(predicted) or not len(expected):
        return metrics(0, len(predicted), len(expected))
    distances = np.linalg.norm(predicted[:, None] - expected[None], axis=2)
    penalty = (min(len(predicted), len(expected)) + 1) * 6
    a, b = linear_sum_assignment(np.where(distances <= 5, distances, penalty))
    tp = int((distances[a, b] <= 5).sum())
    return metrics(tp, len(predicted) - tp, len(expected) - tp)


def disk_components(components, point, radius=PATH_RADIUS):
    x, y = map(lambda v: int(round(v)), point)
    h, w = components.shape
    x0, x1 = max(0, x-radius), min(w, x+radius+1)
    y0, y1 = max(0, y-radius), min(h, y+radius+1)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    values = components[y0:y1, x0:x1][(xx-x)**2 + (yy-y)**2 <= radius**2]
    return set(map(int, values)) - {0}


def path_specs(annotation, shape):
    """Unique source paths >=10px; reference tubes are evaluation-only."""
    h, w = shape
    specs = []
    seen = set()
    for edge_id, edge in annotation.get("edges", {}).items():
        if edge.get("state") == "deleted" or str(edge.get("isFlywire")).lower() == "true":
            continue
        if edge.get("category", "wire") != "wire":
            continue
        raw = edge.get("keyPointPaths", {})
        raw = raw.get("wire", []) if isinstance(raw, dict) else raw
        for path in polylines(raw):
            points = tuple((round(x), round(y)) for x, y in path)
            key = min(points, tuple(reversed(points)))
            length = sum(np.hypot(b[0]-a[0], b[1]-a[1]) for a, b in zip(points, points[1:]))
            if key in seen or length < MIN_PATH_LENGTH:
                continue
            if any(not (0 <= x < w and 0 <= y < h) for x, y in points):
                raise ValueError("Source path outside image")
            seen.add(key)
            x0 = max(0, min(x for x, _ in points)-PATH_RADIUS)
            y0 = max(0, min(y for _, y in points)-PATH_RADIUS)
            x1 = min(w, max(x for x, _ in points)+PATH_RADIUS+1)
            y1 = min(h, max(y for _, y in points)+PATH_RADIUS+1)
            tube = Image.new("1", (x1-x0, y1-y0))
            ImageDraw.Draw(tube).line([(x-x0, y-y0) for x, y in points], fill=1, width=2*PATH_RADIUS+1)
            specs.append({"edge_id": edge_id, "points": points, "length": float(length),
                          "box": (x0, y0, x1, y1), "tube": np.asarray(tube, dtype=bool)})
    return specs


def path_records(mask, specs):
    records = []
    for spec in specs:
        # No closing, bridging or dilation of the prediction. 8-neighbour CC.
        x0, y0, x1, y1 = spec.get("box", (0, 0, mask.shape[1], mask.shape[0]))
        components, _ = ndimage.label(mask[y0:y1, x0:x1] & spec["tube"], np.ones((3, 3)))
        points = [(x-x0, y-y0) for x, y in spec["points"]]
        shared = disk_components(components, points[0])
        for point in points[1:]:
            shared &= disk_components(components, point)
        records.append({k: spec[k] for k in ("edge_id", "points", "length")}
                       | {"connected_within_reference_tube": bool(shared)})
    return records


def self_check():
    spec = {"edge_id": "test", "points": ((3, 5), (16, 5)), "length": 13.,
            "tube": np.zeros((12, 20), bool)}
    spec["tube"][3:8, 3:17] = True
    mask = np.zeros_like(spec["tube"]); mask[5, 3:17] = True
    assert path_records(mask, [spec])[0]["connected_within_reference_tube"]
    mask[5, 10] = False
    assert not path_records(mask, [spec])[0]["connected_within_reference_tube"]
    # An alternate route outside the tube must not certify a missed wire.
    mask[1, 3:17] = True; mask[1:6, 3] = True; mask[1:6, 16] = True
    assert not path_records(mask, [spec])[0]["connected_within_reference_tube"]


def overlay(rgb, mask):
    out = rgb.copy()
    out[mask] = (.25*out[mask] + .75*np.array([0, 130, 235])).astype(np.uint8)
    return out


def render_case(case_id, output, records):
    rgb = np.asarray(Image.open(ROOT/f"data/standardized/{case_id}/{case_id}.png").convert("RGB"))
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
                         "axes.unicode_minus": False})
    rois = [(0, 0, rgb.shape[1], rgb.shape[0])]
    if case_id == "0036":
        rois += [(536, 31, 788, 119), (535, 157, 746, 237)]
    fig, axes = plt.subplots(len(rois), 3, figsize=(16.2, 3.9*len(rois)), dpi=145,
                             squeeze=False, layout="constrained")
    for col, gamma in enumerate(GAMMAS):
        mask = np.asarray(Image.open(output/case_id/f"gamma_{gamma:.1f}"/"wire_mask.png")) > 0
        image = overlay(rgb, mask)
        rec = records[col]
        for row, (x0, y0, x1, y1) in enumerate(rois):
            axes[row, col].imshow(image[y0:y1, x0:x1], interpolation="nearest")
            axes[row, col].axis("off")
            if row == 0:
                title = "原图输入" if gamma == 1 else f"加深输入 γ={gamma:.1f}"
                axes[row, col].set_title(f"{title}\n草稿像素 P={rec['wire']['precision']:.1%} / R={rec['wire']['recall']:.1%}", fontsize=15)
            else:
                edge = "E_1" if row == 1 else "E_10"
                paths = [p for p in rec["paths"] if p["edge_id"] == edge]
                good = sum(p["connected_within_reference_tube"] for p in paths)
                name = "X1 左侧引线" if row == 1 else "DOUT 出口导线"
                axes[row, col].set_title(f"{name} · 参考路径连通 {good}/{len(paths)}", fontsize=14)
    fig.suptitle(f"{case_id} · 蓝色为 U-Net 导线分割；所有面板使用同一原图作背景", fontsize=18)
    fig.savefig(output/f"comparison_{case_id}.png", facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT/"runs/unet_contrast_verification_v1")
    args = parser.parse_args()
    output = args.output.resolve()
    if ROOT not in output.parents or output.exists():
        raise ValueError("Use a new output directory inside the project")
    self_check()
    split_path = ROOT/"runs/unet_inference_v2/split.json"
    split = json.loads(split_path.read_text(encoding="utf-8"))
    inf_path = ROOT/"configs/unet_inference_v2.json"
    inf = json.loads(inf_path.read_text(encoding="utf-8"))
    weight_path = ROOT/"weights/unet_wire_pin_v1.pt"
    weight_hash = sha(weight_path)
    assert weight_hash == inf["checkpoint_sha256"] == split["source_checkpoint_sha256"]
    checkpoint = torch.load(weight_path, map_location="cpu", weights_only=True)
    assert checkpoint["kind"] == "trained_on_draft_labels"
    cfg = checkpoint["config"]
    rows = {row["case_id"]: row for row in split["rows"]}
    raw_rows = {r["case_id"]: r for r in map(json.loads, (ROOT/"data/manifest.jsonl").read_text(encoding="utf-8").splitlines())}
    cases = split["development"] + split["acceptance_holdout"]
    assert len(cases) == len(set(cases)) == 26
    protected = {str(p): sha(p) for p in (split_path, inf_path, weight_path,
                    ROOT/"data/segmentation_unet_v1/manifest.jsonl", ROOT/"data/manifest.jsonl")}
    pixel_hashes = set()
    for case_id in cases:
        row = rows[case_id]
        assert row["split"] == "val"
        image_path = ROOT/f"data/standardized/{case_id}/{case_id}.png"
        assert sha(image_path) == row["image_sha256"]
        rgb = np.asarray(Image.open(image_path).convert("RGB"))
        assert rgb.shape[:2] == (row["height"], row["width"])
        ph = hashlib.sha256(rgb.tobytes()).hexdigest()
        assert ph not in pixel_hashes
        pixel_hashes.add(ph)
        annotation_path = Path(raw_rows[case_id]["raw_annotation_path"])
        assert sha(annotation_path) == row["source_annotation_sha256"]
        for p in (image_path, annotation_path,
                  ROOT/f"data/segmentation_unet_v1/{case_id}/wire_mask.png",
                  ROOT/f"data/segmentation_unet_v1/{case_id}/pin_points.json"):
            protected[str(p)] = sha(p)
    torch.set_num_threads(4)
    device = "cuda:0" if torch.cuda.is_available() and torch.cuda.mem_get_info()[0] > 2*1024**3 else "cpu"
    model = WirePinUNet(checkpoint["base_channels"]).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    output.mkdir(parents=True)
    (output/"protocol.json").write_text(json.dumps({"gammas": GAMMAS, "wire_threshold": .5,
        "pin_settings": inf, "case_groups": {"development": split["development"], "other_validation": split["acceptance_holdout"]},
        "minimum_source_path_length_px": MIN_PATH_LENGTH, "reference_tube_radius_px": PATH_RADIUS,
        "endpoint_disk_radius_px": PATH_RADIUS, "connectivity": "8-neighbour; predictions unchanged",
        "scope": "fixed diagnostic, no tuning/adoption; labels are draft; groups are not independent tests",
        "prediction_reads_labels": False, "source_sha256": protected}, ensure_ascii=False, indent=2), encoding="utf-8")
    records = []
    start = time.perf_counter()
    for index, case_id in enumerate(cases):
        rgb = np.asarray(Image.open(ROOT/f"data/standardized/{case_id}/{case_id}.png").convert("RGB"))
        predictions = []
        # All three predictions are finished before any per-page labels are read.
        for gamma in GAMMAS:
            pixels = rgb if gamma == 1 else np.round(255*(rgb.astype(np.float32)/255)**gamma).astype(np.uint8)
            prob = predict_tiled(model, pixels, device, cfg["tile_size"], cfg["stride"])
            assert prob.shape == (2, *rgb.shape[:2]) and np.isfinite(prob).all()
            candidates = decode_pins(prob[1], inf["pin_threshold"], inf["pin_peak_kernel"],
                                     inf["pin_refinement"], inf["pin_nms_radius"], inf["pin_candidate_limit"])
            predictions.append((prob[0] >= .5, candidates))
            folder = output/case_id/f"gamma_{gamma:.1f}"
            folder.mkdir(parents=True)
            Image.fromarray(predictions[-1][0].astype(np.uint8)*255).save(folder/"wire_mask.png")
            (folder/"pin_candidates.json").write_text(json.dumps(candidates, indent=2), encoding="utf-8")
            if case_id == "0036":
                np.savez_compressed(folder/"probabilities.npz", wire=prob[0], pin_tip=prob[1])
        expected = np.asarray(Image.open(ROOT/f"data/segmentation_unet_v1/{case_id}/wire_mask.png")) > 0
        tips = json.loads((ROOT/f"data/segmentation_unet_v1/{case_id}/pin_points.json").read_text(encoding="utf-8"))
        annotation, _ = load_source_annotation(raw_rows[case_id]["raw_annotation_path"])
        specs = path_specs(annotation, rgb.shape[:2])
        near_reference = ndimage.distance_transform_edt(~expected) <= 2
        baseline = predictions[0][0]
        for gamma, (mask, candidates) in zip(GAMMAS, predictions):
            paths = path_records(mask, specs)
            record = {"case_id": case_id, "group": "development" if case_id in split["development"] else "other_validation",
                      "gamma": gamma, "wire": metrics(int((mask & expected).sum()), int((mask & ~expected).sum()), int((~mask & expected).sum())),
                      "pin_coordinates_5px": pin_metrics(candidates, tips),
                      "extra_pixels_more_than_2px_from_draft": int((mask & ~near_reference).sum()),
                      "added_pixels_more_than_2px_from_draft": int((mask & ~baseline & ~near_reference).sum()),
                      "lost_draft_true_pixels": int((baseline & expected & ~mask).sum()),
                      "recovered_draft_pixels": int((~baseline & expected & mask).sum()),
                      "paths": paths, "path_count": len(paths),
                      "connected_path_count": sum(p["connected_within_reference_tube"] for p in paths)}
            records.append(record)
        print(json.dumps({"case_id": case_id, "completed": index+1, "total": len(cases),
            "dice": [round(r["wire"]["dice"], 4) for r in records[-3:]],
            "connected_paths": [r["connected_path_count"] for r in records[-3:]],
            "paths": len(specs)}, ensure_ascii=False), flush=True)
    groups = {}
    for group in ("all", "development", "other_validation"):
        summary = []
        for gamma in GAMMAS:
            selected = [r for r in records if r["gamma"] == gamma and (group == "all" or r["group"] == group)]
            w = metrics(*(sum(r["wire"][key] for r in selected) for key in ("tp", "fp", "fn")))
            p = metrics(*(sum(r["pin_coordinates_5px"][key] for r in selected) for key in ("tp", "fp", "fn")))
            diffs = [r["wire"]["dice"]-next(b["wire"]["dice"] for b in records if b["case_id"] == r["case_id"] and b["gamma"] == 1.) for r in selected]
            summary.append({"gamma": gamma, "pages": len(selected), "wire_micro": w, "pin_coordinates_micro": p,
                "wire_macro_dice": float(np.mean([r["wire"]["dice"] for r in selected])),
                "pages_dice_improved": sum(d > 1e-6 for d in diffs), "pages_dice_worsened": sum(d < -1e-6 for d in diffs),
                "connected_paths": sum(r["connected_path_count"] for r in selected),
                "paths": sum(r["path_count"] for r in selected),
                "extra_pixels_more_than_2px_from_draft": sum(r["extra_pixels_more_than_2px_from_draft"] for r in selected)})
        groups[group] = summary
    assert all(sha(Path(path)) == digest for path, digest in protected.items())
    result = {"status": "completed_diagnostic_not_adopted", "device": device,
              "wall_seconds": time.perf_counter()-start, "source_files_unchanged": True,
              "model_or_production_settings_changed": False, "training_performed": False,
              "official_score": None, "labels_status": "draft_not_manually_certified",
              "connectivity_scope": "reference tube/path endpoints only; not net correctness",
              "groups": groups, "cases": records}
    (output/"evaluation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    case36 = [r for r in records if r["case_id"] == "0036"]
    render_case("0036", output, case36)
    worst = min((r for r in records if r["gamma"] == 2.), key=lambda r: r["wire"]["dice"]-next(b["wire"]["dice"] for b in records if b["case_id"] == r["case_id"] and b["gamma"] == 1.))
    render_case(worst["case_id"], output, [r for r in records if r["case_id"] == worst["case_id"]])
    lines = ["# U-Net 加深输入验证", "", "固定权重、原尺寸、导线阈值 0.5；γ=1/1.5/2 在验证前固定。未训练、未修改正式设置。",
             "26 张 RGB 去重验证页；0036 是既有开发页。另一组 13 页同样参与过模型选择及后续开发，不是独立测试集。",
             "草稿像素标签尚未人工批准；路径连通按源路径附近 2px 范围内的预测连通分量检查，端点允许 2px 偏差，不代表整网正确。", "",
             "| 组别 | γ | 像素 Precision | Recall | Dice | 连通参考路径 | 离草稿超过2px的额外像素 | 引脚坐标F1 |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for group, stats in groups.items():
        for s in stats:
            w, p = s["wire_micro"], s["pin_coordinates_micro"]
            lines.append(f"| {group} | {s['gamma']} | {w['precision']:.2%} | {w['recall']:.2%} | {w['dice']:.2%} | {s['connected_paths']}/{s['paths']} | {s['extra_pixels_more_than_2px_from_draft']} | {p['dice']:.2%} |")
    lines += ["", "## 每页", "", "| 页 | γ | P | R | Dice | 连通路径 | 新增远离草稿像素 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for r in records:
        w = r["wire"]
        lines.append(f"| {r['case_id']} | {r['gamma']} | {w['precision']:.2%} | {w['recall']:.2%} | {w['dice']:.2%} | {r['connected_path_count']}/{r['path_count']} | {r['added_pixels_more_than_2px_from_draft']} |")
    lines += ["", "## 0036", "", "| γ | 参考路径 | 端点 | 连通 |", "|---|---|---|---|"]
    for r in case36:
        for path in r["paths"]:
            if path["edge_id"] in ("E_1", "E_10"):
                lines.append(f"| {r['gamma']} | {path['edge_id']} | {path['points']} | {path['connected_within_reference_tube']} |")
    lines += ["", "所有输入文件已在运行前后核对 SHA256。原图、权重、标签、冻结配置均保持原样。",
              "额外像素首先是相对草稿的差异；需要在原图上确认，不能全部宣称为真实误检。",
              "加深同时改变文字、元件符号和引线等内容；实验支持对比度敏感性，不能单独隔离像素颜色与周围语境的作用。"]
    (output/"report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "groups": groups, "worst_gamma2_page": worst["case_id"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
