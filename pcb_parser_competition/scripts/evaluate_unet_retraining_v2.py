"""Compare completed retraining arms with v1 on the fixed original-image protocol."""
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
from scipy import ndimage
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from pcb_parser.dataset_converter import load_source_annotation
from pcb_parser.unet_data import predict_tiled
from pcb_parser.unet_inference import decode_pins
from pcb_parser.unet_model import WirePinUNet
from verify_unet_contrast import metrics, path_specs, path_records, pin_metrics, self_check

OUT = ROOT/"runs/unet_retrain_v2"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def aggregate(records, variant, group):
    selected = [r for r in records if r["variant"] == variant and (group == "all" or r["group"] == group)]
    w = metrics(*(sum(r["wire"][key] for r in selected) for key in ("tp", "fp", "fn")))
    p = metrics(*(sum(r["pin_coordinates_5px"][key] for r in selected) for key in ("tp", "fp", "fn")))
    return {"variant": variant, "pages": len(selected), "wire_micro": w, "pin_coordinates_micro": p,
            "connected_paths": sum(r["connected_path_count"] for r in selected),
            "paths": sum(r["path_count"] for r in selected),
            "extra_pixels_more_than_2px_from_draft": sum(r["extra_pixels_more_than_2px_from_draft"] for r in selected)}


def targets_0036(record):
    wanted = {"X1": [[601, 80], [691, 80]], "DOUT_horizontal": [[601, 181], [634, 181]]}
    return {name: next(p["connected_within_reference_tube"] for p in record["paths"] if [list(v) for v in p["points"]] == points)
            for name, points in wanted.items()}


def render_detail(eval_out):
    rgb = np.asarray(Image.open(ROOT/"data/standardized/0036/0036.png").convert("RGB"))
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"], "axes.unicode_minus": False})
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 6.4), dpi=150, layout="constrained")
    for col, (variant, title) in enumerate((("v1", "原模型"), ("control", "修正标签 + 普通微调"), ("contrast", "修正标签 + 对比度增强微调"))):
        mask_path = (ROOT/"runs/unet_contrast_verification_v1/0036/gamma_1.0/wire_mask.png"
                     if variant == "v1" else eval_out/variant/"0036/wire_mask.png")
        mask = np.asarray(Image.open(mask_path)) > 0
        display = rgb.copy(); display[mask] = (.15*display[mask]+.85*np.array([0, 130, 235])).astype(np.uint8)
        for row, (x0, y0, x1, y1) in enumerate(((551, 31, 743, 106), (551, 159, 735, 238))):
            axes[row, col].imshow(display[y0:y1, x0:x1], interpolation="nearest")
            axes[row, col].axis("off")
            axes[row, col].set_title(title+("\nX1 左侧引线" if row == 0 else "\nDOUT 引线"), fontsize=14)
    fig.suptitle("0036 · 三个模型都直接读取原图；蓝色为导线分割", fontsize=17)
    fig.savefig(eval_out/"comparison_0036_detail.png", facecolor="white"); plt.close(fig)


def main():
    self_check()
    cfg = json.loads((ROOT/"configs/unet_retraining_v2.json").read_text(encoding="utf-8"))
    training = json.loads((OUT/"training_summary.json").read_text(encoding="utf-8"))
    assert training["status"] == "both_runs_completed"
    original_protection = json.loads((ROOT/"data/segmentation_unet_v2_reviewed_draft/summary.json").read_text(encoding="utf-8"))["protected_source_sha256"]
    assert all(sha(path) == digest for path, digest in original_protection.items())
    old_root = ROOT/"runs/unet_contrast_verification_v1"
    old_protocol = json.loads((old_root/"protocol.json").read_text(encoding="utf-8"))
    assert all(sha(path) == digest for path, digest in old_protocol["source_sha256"].items())
    old = json.loads((old_root/"evaluation.json").read_text(encoding="utf-8"))
    records = [dict(r, variant="v1") for r in old["cases"] if r["gamma"] == 1.]
    for r in records:
        r.pop("gamma", None)
    assert len(records) == 26
    split = json.loads((ROOT/"runs/unet_inference_v2/split.json").read_text(encoding="utf-8"))
    cases = split["development"] + split["acceptance_holdout"]
    raw = {r["case_id"]: r for r in map(json.loads, (ROOT/"data/manifest.jsonl").read_text(encoding="utf-8").splitlines())}
    inf = json.loads((ROOT/"configs/unet_inference_v2.json").read_text(encoding="utf-8"))
    eval_out = OUT/"evaluation"
    if eval_out.exists():
        raise ValueError("Evaluation already exists; do not overwrite an earlier result")
    eval_out.mkdir()
    torch.set_num_threads(4); device = "cuda:0"
    lineage = {"v1": {"checkpoint_sha256": cfg["initial_sha256"], "baseline_evaluation_sha256": sha(old_root/"evaluation.json")}}
    start = time.perf_counter()
    for variant in cfg["arms"]:
        summary = json.loads((OUT/variant/"summary.json").read_text(encoding="utf-8"))
        assert summary["epochs_completed"] == cfg["epochs"] and summary["output_layer_changed"]
        weight = OUT/variant/"best.pt"
        assert sha(weight) == summary["checkpoint_sha256"]
        checkpoint = torch.load(weight, map_location="cpu", weights_only=True)
        assert checkpoint["kind"] == "trained_on_draft_labels" and checkpoint["initial_weights_sha256"] == cfg["initial_sha256"]
        model = WirePinUNet(checkpoint["base_channels"]).to(device)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        lineage[variant] = {"checkpoint_sha256": sha(weight), "best_epoch": summary["best_epoch"], "training_summary_sha256": sha(OUT/variant/"summary.json")}
        runtime_cfg = {k: inf[k] for k in ("wire_mode", "wire_threshold", "pin_mode", "pin_threshold", "pin_peak_kernel", "pin_refinement", "pin_nms_radius", "pin_candidate_limit", "traditional_image_segmentation")}
        runtime_cfg.update(checkpoint_sha256=sha(weight), version="v2-retraining-candidate", inference_images_augmented=False,
                           settings_status="fixed_v1_settings_used_for_v2_comparison_not_recalibrated", official_score=None)
        (OUT/variant/"inference_config.json").write_text(json.dumps(runtime_cfg, indent=2), encoding="utf-8")
        for index, c in enumerate(cases):
            rgb = np.asarray(Image.open(ROOT/f"data/standardized/{c}/{c}.png").convert("RGB"))
            # No contrast transform, labels, owner boxes, or reference tips enter prediction.
            probability = predict_tiled(model, rgb, device, cfg["tile_size"], cfg["stride"])
            mask = probability[0] >= inf["wire_threshold"]
            candidates = decode_pins(probability[1], inf["pin_threshold"], inf["pin_peak_kernel"], inf["pin_refinement"], inf["pin_nms_radius"], inf["pin_candidate_limit"])
            expected = np.asarray(Image.open(ROOT/f"data/segmentation_unet_v1/{c}/wire_mask.png")) > 0
            tips = json.loads((ROOT/f"data/segmentation_unet_v1/{c}/pin_points.json").read_text(encoding="utf-8"))
            annotation, _ = load_source_annotation(raw[c]["raw_annotation_path"])
            paths = path_records(mask, path_specs(annotation, rgb.shape[:2]))
            for path in paths:
                path["points"] = [list(point) for point in path["points"]]
            record = {"case_id": c, "variant": variant, "group": "development" if c in split["development"] else "other_validation",
                      "wire": metrics(int((mask & expected).sum()), int((mask & ~expected).sum()), int((~mask & expected).sum())),
                      "pin_coordinates_5px": pin_metrics(candidates, tips), "paths": paths, "path_count": len(paths),
                      "connected_path_count": sum(p["connected_within_reference_tube"] for p in paths),
                      "extra_pixels_more_than_2px_from_draft": int((mask & (ndimage.distance_transform_edt(~expected)>2)).sum())}
            baseline = next(r for r in records if r["case_id"] == c and r["variant"] == "v1")
            assert [(p["edge_id"], p["points"]) for p in paths] == [(p["edge_id"], p["points"]) for p in baseline["paths"]]
            record["recovered_connected_paths"] = sum(p["connected_within_reference_tube"] and not b["connected_within_reference_tube"] for b, p in zip(baseline["paths"], paths))
            record["lost_connected_paths"] = sum(b["connected_within_reference_tube"] and not p["connected_within_reference_tube"] for b, p in zip(baseline["paths"], paths))
            if c == "0036":
                record["target_paths_connected"] = targets_0036(record)
                record["target_scores"] = {}
                for name, y, x0, x1 in (("X1", 80, 615, 692), ("DOUT", 181, 604, 632)):
                    profile = probability[0, y-1:y+2, x0:x1+1].max(axis=0)
                    record["target_scores"][name] = {"median": float(np.median(profile)), "minimum": float(profile.min()), "maximum": float(profile.max())}
            records.append(record)
            folder = eval_out/variant/c; folder.mkdir(parents=True)
            Image.fromarray(mask.astype(np.uint8)*255).save(folder/"wire_mask.png")
            (folder/"pin_candidates.json").write_text(json.dumps(candidates, indent=2), encoding="utf-8")
            if c == "0036":
                np.savez_compressed(folder/"probabilities.npz", wire=probability[0], pin_tip=probability[1])
            print(json.dumps({"variant": variant, "case_id": c, "completed": index+1, "total": len(cases), "dice": record["wire"]["dice"], "connected_paths": record["connected_path_count"]}), flush=True)
        del model, checkpoint
        torch.cuda.empty_cache()
    groups = {g: [aggregate(records, v, g) for v in ("v1", "control", "contrast")] for g in ("all", "development", "other_validation")}
    adoption = {}
    for v in cfg["arms"]:
        original, candidate = groups["all"][0], next(r for r in groups["all"] if r["variant"] == v)
        other_original = groups["other_validation"][0]
        other = next(r for r in groups["other_validation"] if r["variant"] == v)
        c36 = next(r for r in records if r["variant"] == v and r["case_id"] == "0036")
        checks = {"all_wire_dice_not_lower": candidate["wire_micro"]["dice"] >= original["wire_micro"]["dice"],
                  "all_connected_paths_not_lower": candidate["connected_paths"] >= original["connected_paths"],
                  "other_validation_wire_dice_not_lower": other["wire_micro"]["dice"] >= other_original["wire_micro"]["dice"],
                  "pin_coordinate_f1_drop_within_1pp": candidate["pin_coordinates_micro"]["dice"] >= original["pin_coordinates_micro"]["dice"]-.01,
                  "0036_target_paths_connected": all(c36["target_paths_connected"].values())}
        adoption[v] = {"checks": checks, "passed_all_checks": all(checks.values()), "default_replaced": False}
    assert all(sha(path) == digest for path, digest in original_protection.items())
    result = {"status": "completed_retraining_comparison", "groups": groups, "adoption_checks": adoption, "lineage": lineage,
              "native_original_image_prediction": True, "fixed_wire_threshold": .5, "reference_used_for_scoring_only": True,
              "original_files_unchanged": True, "production_default_changed": False, "training_label_corrections_only_on_train": True,
              "labels_status": "draft_with_partial_agent_training_review_not_certified", "independent_test_set": False,
              "official_score": None, "wall_seconds": time.perf_counter()-start, "cases": records}
    (eval_out/"evaluation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    render_detail(eval_out)
    lines = ["# U-Net 标签修正与对比度增强重训", "", "从原第57轮模型开始，两组各完成16轮微调；相同训练标签、初始权重、优化器、学习率、随机种子和切片顺序。",
             "103张训练图；26张去重验证图。普通微调作为对照，对比度增强仅用于另一组训练输入。验证与推理始终使用原图。",
             "4张训练图中的9个区域经过agent视觉检查；0073补CC2/VOUT漏标，0088修正两组导线位置及两个电容tip。完整人工批准页仍为0。",
             "损失函数沿用v1；这一轮先验证标签修正和训练增强的作用，没有同时更换结构或损失。", "",
             "| 组别 | 模型 | 导线P | 导线R | Dice | 连通参考路径 | 引脚坐标F1 |", "|---|---|---:|---:|---:|---:|---:|"]
    for group, stats in groups.items():
        for s in stats:
            w, p = s["wire_micro"], s["pin_coordinates_micro"]
            lines.append(f"| {group} | {s['variant']} | {w['precision']:.2%} | {w['recall']:.2%} | {w['dice']:.2%} | {s['connected_paths']}/{s['paths']} | {p['dice']:.2%} |")
    lines += ["", "## 0036目标", "", "| 模型 | X1左引线连通 | DOUT水平引线连通 |", "|---|---|---|"]
    for v in ("v1", "control", "contrast"):
        c36 = next(r for r in records if r["variant"] == v and r["case_id"] == "0036")
        t = targets_0036(c36)
        lines.append(f"| {v} | {t['X1']} | {t['DOUT_horizontal']} |")
    lines += ["", "## 采用检查", ""]
    for v, a in adoption.items():
        lines.append(f"{v}: {json.dumps(a, ensure_ascii=False)}")
    lines += ["", "参考路径连通采用既有规则：源路径长度至少10px、预测限制在路径附近2px范围、端点2px容差、8邻域连通，不做补线。",
              "它只评价局部路径，不能证明整网正确。像素/坐标指标基于未经完整人工认证的草稿，不是正式PinF1/NetF1。",
              "验证图参与过模型选择和后续开发，不是独立工程测试集。原图、原权重、原标签和正式默认配置均保持原样。", "",
              "候选模型可配合对应目录下的inference_config.json运行PNG推理；固定沿用v1阈值，没有在本次验证后搜索新阈值。"]
    (eval_out/"report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    (OUT/"progress.json").write_text(json.dumps({"status": "training_and_evaluation_complete", "groups": groups, "adoption_checks": adoption}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"groups": groups, "adoption_checks": adoption, "output": str(eval_out)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
