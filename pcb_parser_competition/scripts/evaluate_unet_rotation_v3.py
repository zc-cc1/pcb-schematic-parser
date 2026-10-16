"""Original-image evaluation of rotation-only training against v1 and control."""
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
from evaluate_unet_retraining_v2 import aggregate, targets_0036
from retrain_unet_v2 import sha, write_json
from retrain_unet_rotation_v3 import monitor
from verify_unet_contrast import metrics, path_records, path_specs, pin_metrics, self_check
from diagnose_unet_0036_context import right_tile_prediction


def evaluate_phase_stability(rotation_model, cfg, device, output, raw):
    """Fixed 0/1/2px diagnostic; never used for model/epoch selection."""
    rgb = np.asarray(Image.open(ROOT/"data/standardized/0036/0036.png").convert("RGB"))
    records = []
    folder = output/"phase_0036"; folder.mkdir()
    control_weight = ROOT/cfg["reference_control"]/"best.pt"
    source = torch.load(control_weight, map_location="cpu", weights_only=True)
    control_model = WirePinUNet(source["base_channels"]).to(device)
    control_model.load_state_dict(source["state_dict"], strict=True)
    probabilities = []
    baseline_files = {0: ROOT/"runs/unet_context_diagnosis_0036_phase/shift_y_0.npz",
                      1: ROOT/"runs/unet_context_diagnosis_0036_phase/shift_y_1.npz",
                      2: ROOT/"runs/unet_context_diagnosis_0036/right_tile_shift_y_2.npz"}
    hashes = {str(p): sha(p) for p in baseline_files.values()}
    for dy in (0, 1, 2):
        with np.load(baseline_files[dy], allow_pickle=False) as baseline:
            probabilities.append(("v1", dy, np.stack([baseline["wire"], baseline["pin_tip"]])))
        for variant, model in (("control", control_model), ("rotation", rotation_model)):
            p = right_tile_prediction(model, rgb, device, dy)
            probabilities.append((variant, dy, p))
            np.savez_compressed(folder/f"{variant}_shift_y_{dy}.npz", wire=p[0], pin_tip=p[1])
    # Read source geometry only after all phase predictions are complete.
    annotation, _ = load_source_annotation(raw["0036"]["raw_annotation_path"])
    specs = [p for p in path_specs(annotation, rgb.shape[:2]) if p["edge_id"] in ("E_1", "E_8", "E_10")]
    for variant, dy, p in probabilities:
        records.append(dict(variant=variant, shift_y=dy, monitor=monitor(p),
                            target_draft_paths=path_records(p[0] >= .5, specs)))
    assert all(sha(p) == h for p, h in hashes.items())
    result = dict(scope="single-case fixed-position diagnostic, not checkpoint selection or inference setting", shifts=[0, 1, 2],
                  original_pixels_preserved=True, baseline_sha256=hashes, records=records)
    write_json(folder/"evaluation.json", result)
    del control_model, source
    return result


def target_paths(record):
    values = targets_0036(record)
    values["X1_right"] = next(p["connected_within_reference_tube"] for p in record["paths"] if p["edge_id"] == "E_8")
    return values


def render_detail(output):
    rgb = np.asarray(Image.open(ROOT/"data/standardized/0036/0036.png").convert("RGB"))
    sources = [("原模型", ROOT/"runs/unet_contrast_verification_v1/0036/gamma_1.0/wire_mask.png"),
               ("普通微调对照", ROOT/"runs/unet_retrain_v2/evaluation/control/0036/wire_mask.png"),
               ("旋转增强微调", output/"rotation/0036/wire_mask.png")]
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"], "axes.unicode_minus": False})
    fig, axes = plt.subplots(2, 3, figsize=(14, 6.2), dpi=160, layout="constrained")
    for col, (name, source) in enumerate(sources):
        mask = np.asarray(Image.open(source)) > 0
        background = rgb.copy(); background[mask] = (.2*background[mask]+.8*np.array([0, 130, 235])).astype(np.uint8)
        for row, (x0, y0, x1, y1) in enumerate(((551, 31, 788, 119), (551, 158, 747, 237))):
            axes[row, col].imshow(background[y0:y1, x0:x1], interpolation="nearest")
            axes[row, col].axis("off"); axes[row, col].set_title(name+(" · X1" if row == 0 else " · DOUT"), fontsize=14)
    fig.suptitle("0036：三个模型都直接预测原方向图片，阈值0.5；蓝色为导线", fontsize=17)
    fig.savefig(output/"comparison_0036.png", facecolor="white"); plt.close(fig)


def main():
    self_check()
    cfg_path = ROOT/"configs/unet_rotation_retraining_v3.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    root = ROOT/cfg["output"]; output = root/"evaluation"
    if output.exists():
        raise ValueError("Do not overwrite an evaluation")
    summary = json.loads((root/"rotation/summary.json").read_text(encoding="utf-8"))
    assert summary["epochs_completed"] == cfg["epochs"] and summary["output_layer_changed"]
    protocol = json.loads((root/"training_protocol.json").read_text(encoding="utf-8"))
    assert protocol["config_sha256"] == sha(cfg_path)
    assert all(sha(ROOT/p) == h for p, h in protocol["source_code_sha256"].items())
    protected = json.loads((ROOT/cfg["protected_source_summary"]).read_text(encoding="utf-8"))["protected_source_sha256"]
    assert all(sha(p) == h for p, h in protected.items())
    old = json.loads((ROOT/cfg["reference_evaluation"]).read_text(encoding="utf-8"))
    assert old["native_original_image_prediction"] and old["fixed_wire_threshold"] == .5
    assert old["lineage"]["control"]["checkpoint_sha256"] == protocol["reference_control"]["checkpoint_sha256"]
    assert old["lineage"]["v1"]["checkpoint_sha256"] == cfg["initial_sha256"]
    records = [r for r in old["cases"] if r["variant"] in ("v1", "control")]
    assert len(records) == 52
    for variant, cache in (("v1", ROOT/"runs/unet_contrast_verification_v1/0036/gamma_1.0/probabilities.npz"),
                           ("control", ROOT/"runs/unet_retrain_v2/evaluation/control/0036/probabilities.npz")):
        r = next(r for r in records if r["variant"] == variant and r["case_id"] == "0036")
        with np.load(cache, allow_pickle=False) as values:
            r["native_0036_monitor"] = monitor(np.stack([values["wire"], values["pin_tip"]]))
        r["target_paths_connected"] = target_paths(r)
    split = json.loads((ROOT/"runs/unet_inference_v2/split.json").read_text(encoding="utf-8"))
    cases = split["development"] + split["acceptance_holdout"]
    assert len(cases) == 26 and set(cases) == set(summary["validation_cases"])
    raw = {r["case_id"]: r for r in map(json.loads, (ROOT/"data/manifest.jsonl").read_text(encoding="utf-8").splitlines())}
    inf = json.loads((ROOT/"configs/unet_inference_v2.json").read_text(encoding="utf-8"))
    weight = root/"rotation/best.pt"
    assert sha(weight) == summary["checkpoint_sha256"]
    checkpoint = torch.load(weight, map_location="cpu", weights_only=True)
    assert checkpoint["initial_weights_sha256"] == cfg["initial_sha256"] and checkpoint["arm"] == "rotation"
    torch.set_num_threads(4); device = "cuda:0"
    model = WirePinUNet(checkpoint["base_channels"]).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    runtime_cfg = {k: inf[k] for k in ("wire_mode", "wire_threshold", "pin_mode", "pin_threshold", "pin_peak_kernel", "pin_refinement", "pin_nms_radius", "pin_candidate_limit", "traditional_image_segmentation")}
    runtime_cfg.update(checkpoint_sha256=sha(weight), version="v3-rotation-training-candidate",
        inference_images_augmented=False, settings_status="fixed_v1_settings_not_recalibrated", official_score=None)
    write_json(root/"rotation/inference_config.json", runtime_cfg)
    output.mkdir(); start = time.perf_counter()
    for index, c in enumerate(cases):
        rgb = np.asarray(Image.open(ROOT/f"data/standardized/{c}/{c}.png").convert("RGB"))
        probability = predict_tiled(model, rgb, device, cfg["tile_size"], cfg["stride"])
        mask = probability[0] >= inf["wire_threshold"]
        candidates = decode_pins(probability[1], inf["pin_threshold"], inf["pin_peak_kernel"], inf["pin_refinement"], inf["pin_nms_radius"], inf["pin_candidate_limit"])
        # References are loaded after the unchanged original-image prediction.
        expected = np.asarray(Image.open(ROOT/f"data/segmentation_unet_v1/{c}/wire_mask.png")) > 0
        tips = json.loads((ROOT/f"data/segmentation_unet_v1/{c}/pin_points.json").read_text(encoding="utf-8"))
        annotation, _ = load_source_annotation(raw[c]["raw_annotation_path"])
        paths = path_records(mask, path_specs(annotation, rgb.shape[:2]))
        for path in paths:
            path["points"] = [list(point) for point in path["points"]]
        record = dict(case_id=c, variant="rotation", group="development" if c in split["development"] else "other_validation",
            wire=metrics(int((mask & expected).sum()), int((mask & ~expected).sum()), int((~mask & expected).sum())),
            pin_coordinates_5px=pin_metrics(candidates, tips), paths=paths, path_count=len(paths),
            connected_path_count=sum(p["connected_within_reference_tube"] for p in paths),
            extra_pixels_more_than_2px_from_draft=int((mask & (ndimage.distance_transform_edt(~expected)>2)).sum()))
        for baseline_variant in ("v1", "control"):
            baseline = next(r for r in records if r["variant"] == baseline_variant and r["case_id"] == c)
            assert [(p["edge_id"], p["points"]) for p in paths] == [(p["edge_id"], p["points"]) for p in baseline["paths"]]
            record[f"recovered_paths_vs_{baseline_variant}"] = sum(p["connected_within_reference_tube"] and not b["connected_within_reference_tube"] for p, b in zip(paths, baseline["paths"]))
            record[f"lost_paths_vs_{baseline_variant}"] = sum(b["connected_within_reference_tube"] and not p["connected_within_reference_tube"] for p, b in zip(paths, baseline["paths"]))
        if c == "0036":
            record["native_0036_monitor"] = monitor(probability)
            record["target_paths_connected"] = target_paths(record)
        records.append(record)
        folder = output/"rotation"/c; folder.mkdir(parents=True)
        Image.fromarray(mask.astype(np.uint8)*255).save(folder/"wire_mask.png")
        write_json(folder/"pin_candidates.json", candidates)
        if c == "0036":
            np.savez_compressed(folder/"probabilities.npz", wire=probability[0], pin_tip=probability[1])
        print(json.dumps(dict(case_id=c, completed=index+1, total=26, dice=record["wire"]["dice"], connected=record["connected_path_count"])), flush=True)
    phase_diagnosis = evaluate_phase_stability(model, cfg, device, output, raw)
    groups = {g: [aggregate(records, v, g) for v in ("v1", "control", "rotation")] for g in ("all", "development", "other_validation")}
    baseline, _, candidate = groups["all"]
    c36 = next(r for r in records if r["variant"] == "rotation" and r["case_id"] == "0036")
    original36 = next(r for r in records if r["variant"] == "v1" and r["case_id"] == "0036")
    checks = dict(all_wire_dice_not_lower=candidate["wire_micro"]["dice"] >= baseline["wire_micro"]["dice"],
        all_connected_paths_not_lower=candidate["connected_paths"] >= baseline["connected_paths"],
        other_validation_wire_dice_not_lower=groups["other_validation"][2]["wire_micro"]["dice"] >= groups["other_validation"][0]["wire_micro"]["dice"],
        pin_coordinate_f1_drop_within_1pp=candidate["pin_coordinates_micro"]["dice"] >= baseline["pin_coordinates_micro"]["dice"]-.01,
        case_0036_target_paths_connected=all(c36["target_paths_connected"].values()),
        case_0036_IC_border_false_pixels_not_increased=c36["native_0036_monitor"]["manual_nonwire_IC_border"]["pixels_above_050"] <= original36["native_0036_monitor"]["manual_nonwire_IC_border"]["pixels_above_050"])
    assert all(sha(p) == h for p, h in protected.items())
    result = dict(status="rotation_training_and_native_evaluation_complete", groups=groups, cases=records,
        phase_diagnosis=phase_diagnosis,
        adoption=dict(checks=checks, passed_all_checks=all(checks.values()), default_replaced=False),
        lineage=dict(v1_checkpoint_sha256=cfg["initial_sha256"], control_checkpoint_sha256=protocol["reference_control"]["checkpoint_sha256"],
            rotation_checkpoint_sha256=sha(weight), rotation_best_epoch=summary["best_epoch"], baseline_evaluation_sha256=sha(ROOT/cfg["reference_evaluation"])),
        native_original_image_prediction=True, input_rotated_at_inference=False, fixed_wire_threshold=.5,
        labels_status=cfg["label_status"], independent_test_set=False, official_score=None,
        original_files_unchanged=True, production_default_changed=False, wall_seconds=time.perf_counter()-start)
    write_json(output/"evaluation.json", result)
    render_detail(output)
    lines = ["# U-Net 旋转增强训练原方向验证", "", "旋转组从v1第57轮权重完成16轮微调，复用上一轮相同设置的普通微调组作对照。训练103页，验证26个去重页面，0036未进入训练。",
        "训练输入/导线标签/引脚热图/valid区域同步随机旋转0/90/180/270度；不插值、不改变灰度，没有加入位移或对比度增强。验证和PNG推理都使用原方向原图。",
        "模型结构和损失保持一致，按原图验证损失选择checkpoint，不按0036表现选轮次。验证页已经参与过模型选择和后续开发，不是独立工程测试。", "",
        "| 组别 | 模型 | 导线P | 导线R | Dice | 连通参考路径 | 引脚坐标F1 |", "|---|---|---:|---:|---:|---:|---:|"]
    for group, stats in groups.items():
        for s in stats:
            w, p = s["wire_micro"], s["pin_coordinates_micro"]
            lines.append(f"| {group} | {s['variant']} | {w['precision']:.2%} | {w['recall']:.2%} | {w['dice']:.2%} | {s['connected_paths']}/{s['paths']} | {p['dice']:.2%} |")
    lines += ["", "## 0036", "", "| 模型 | X1左线中段分数 | DOUT中段分数 | X1左线连通 | X1右线连通 | DOUT横线连通 | 手工IC边框区域误检像素 |", "|---|---:|---:|---|---|---|---:|"]
    for variant in ("v1", "control", "rotation"):
        r = next(r for r in records if r["variant"] == variant and r["case_id"] == "0036")
        m = r["native_0036_monitor"]; t = r["target_paths_connected"]
        lines.append(f"| {variant} | {m['X1_left']['median']:.5f} | {m['DOUT']['median']:.5f} | {t['X1']} | {t['X1_right']} | {t['DOUT_horizontal']} | {m['manual_nonwire_IC_border']['pixels_above_050']} |")
    lines += ["", "## 固定位置稳定性诊断", "", "另行检查0/1/2px位移，输出还原；仅解释位置敏感性，不参与checkpoint选择或26页原方向指标。", "",
        "| 模型 | 下移px | X1左线中段分数 | DOUT中段分数 |", "|---|---:|---:|---:|"]
    for r in phase_diagnosis["records"]:
        m = r["monitor"]
        lines.append(f"| {r['variant']} | {r['shift_y']} | {m['X1_left']['median']:.5f} | {m['DOUT']['median']:.5f} |")
    lines += ["", "## 采用检查", "", json.dumps(result["adoption"], ensure_ascii=False, indent=2), "",
        "路径连通沿用源路径附近2px管道、端点2px容差、8邻域；不补线，是局部草稿代理，不代表整网正确。",
        "引脚F1是草稿坐标5px匹配；像素标签未完整人工认证，仍非正式PinF1/NetF1。648个原文件SHA256前后相同，正式模型/配置保持原样。"]
    (output/"report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    write_json(root/"progress.json", dict(status="training_and_evaluation_complete", groups=groups, adoption=result["adoption"]))
    print(json.dumps(dict(groups=groups, adoption=result["adoption"], output=str(output)), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
