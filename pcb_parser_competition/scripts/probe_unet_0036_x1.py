"""Frozen X1 sensitivity diagnosis: 2D tile shifts and lossless orientations.

No resize, recoloring, labels in prediction, model edits or deployment. Phase
and orientation selection is post-hoc explanation on one development image.
"""
import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch

from diagnose_unet_0036_context import ROOT, REGIONS, measure, overlay, sha
sys.path.insert(0, str(ROOT/"src"))
from pcb_parser.unet_model import WirePinUNet

X1_RIGHT = dict(id="G", name="X1 右侧横线", axis="h", center=80, start=727, end=760,
                crop=[715, 61, 783, 100])
SAMPLE_REGIONS = [REGIONS[0], REGIONS[1], X1_RIGHT]
ORIENTATIONS = ("flip_lr", "flip_ud", "flip_both", "rotate_90", "rotate_270")


def input_transform(tile, name):
    if name == "flip_lr":
        return np.flip(tile, axis=1).copy()
    if name == "flip_ud":
        return np.flip(tile, axis=0).copy()
    if name == "flip_both":
        return np.flip(tile, axis=(0, 1)).copy()
    return np.rot90(tile, 1 if name == "rotate_90" else 3, axes=(0, 1)).copy()


def restore_orientation(p, name):
    if name == "flip_lr":
        return np.flip(p, axis=2).copy()
    if name == "flip_ud":
        return np.flip(p, axis=1).copy()
    if name == "flip_both":
        return np.flip(p, axis=(1, 2)).copy()
    return np.rot90(p, -1 if name == "rotate_90" else -3, axes=(1, 2)).copy()


@torch.inference_mode()
def predict(model, tile, device):
    batch = torch.from_numpy(tile.transpose(2, 0, 1).copy()).float().unsqueeze(0).to(device)/255
    return model(batch).sigmoid()[0].cpu().numpy()


def global_probability(p, h):
    result = np.full((2, h, 832), np.nan, np.float32)
    result[:, :, 320:] = p[:, :h]
    return result


def scores(rgb, p):
    rows = [measure(rgb, p[0], r) for r in SAMPLE_REGIONS]
    # Full reference span, including the ends omitted by the interior samples.
    profile = p[0, 78:83, 601:692].max(axis=0)
    right = p[0, 78:83, 722:769].max(axis=0)
    return dict(regions=rows, x1_left_reference_columns_above_050=int((profile >= .5).sum()),
        x1_left_reference_columns=91, x1_left_missing_columns=(np.flatnonzero(profile < .5)+601).tolist(),
        x1_right_reference_columns_above_050=int((right >= .5).sum()), x1_right_reference_columns=47)


def compact_score(record):
    a, b, g = record["regions"]
    return dict(key=record["key"], x1_left_median=a["score_median"],
        x1_left_interior_coverage=f"{a['samples_above_050']}/{a['samples']}",
        x1_left_reference_coverage=f"{record['x1_left_reference_columns_above_050']}/91",
        dout_median=b["score_median"], x1_right_median=g["score_median"])


def ranking(record):
    return record["x1_left_reference_columns_above_050"], record["regions"][0]["score_median"]


def render(rgb, records, predictions, best, output):
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"], "axes.unicode_minus": False})
    medians = np.array([[next(r for r in records if r["key"] == f"shift_{dx}_{dy}")["regions"][0]["score_median"]
                          for dx in range(16)] for dy in range(16)])
    fig, ax = plt.subplots(figsize=(7.8, 5.4), dpi=165, layout="constrained")
    heat = ax.imshow(medians, vmin=0, vmax=1, cmap="magma", origin="lower", interpolation="nearest")
    ax.set(xticks=range(16), yticks=range(16), xlabel="右移 dx（px）", ylabel="下移 dy（px）",
           title="X1 左侧细线 · 256 种位置的分数中位数")
    fig.colorbar(heat, ax=ax, label="模型分数"); fig.savefig(output/"x1_phase_grid.png", facecolor="white"); plt.close(fig)
    selected = ["shift_0_0", best["key"], *ORIENTATIONS]
    selected = list(dict.fromkeys(selected))
    titles = {"shift_0_0": "原始位置", best["key"]: f"诊断中选出的最高覆盖位置\n右移{best['dx']} / 下移{best['dy']}px",
              "flip_lr": "左右翻转后还原", "flip_ud": "上下翻转后还原", "flip_both": "双向翻转后还原",
              "rotate_90": "旋转90°后还原", "rotate_270": "旋转270°后还原"}
    # A 3-column layout keeps text and one-pixel wires readable.
    fig, axes = plt.subplots(3, 3, figsize=(14, 8.2), dpi=155, layout="constrained")
    for ax in axes.ravel():
        ax.axis("off")
    for ax, key in zip(axes.ravel(), selected):
        row = next(r for r in records if r["key"] == key)
        ax.imshow(overlay(rgb, predictions[key][0])[55:100, 587:788], interpolation="nearest")
        a = row["regions"][0]
        ax.set_title(f"{titles[key]}\n左线中位数 {a['score_median']:.3f} · 内部 {a['samples_above_050']}/61\n端点附近参考范围覆盖 {row['x1_left_reference_columns_above_050']}/91", fontsize=12)
    fig.suptitle("X1：原图背景 + 还原后的分割；固定原权重及阈值0.5", fontsize=16)
    fig.savefig(output/"x1_transform_comparison.png", facecolor="white"); plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT/"runs/unet_x1_diagnosis_0036")
    args = parser.parse_args()
    output = args.output.resolve()
    if ROOT not in output.parents or output.exists():
        raise ValueError("Use a new output directory inside the project")
    image_path = ROOT/"data/standardized/0036/0036.png"
    weights_path = ROOT/"weights/unet_wire_pin_v1.pt"
    config_path = ROOT/"configs/unet_inference_v2.json"
    native_cache = ROOT/"runs/unet_context_diagnosis_0036_phase/shift_y_0.npz"
    protected = {str(p): sha(p) for p in [image_path, weights_path, config_path, native_cache]}
    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert sha(weights_path) == config["checkpoint_sha256"]
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)
    torch.set_num_threads(4)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = WirePinUNet(checkpoint["base_channels"]).to(device).eval()
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    rgb = np.asarray(Image.open(image_path).convert("RGB"))
    h = rgb.shape[0]
    original = np.full((512, 512, 3), 255, np.uint8)
    original[:h] = rgb[:, 320:]
    assert np.all(original[:, -15:] == 255)  # Shifts discard whitespace only.
    output.mkdir(parents=True)
    protocol = dict(scope="single-case diagnostic, not validated optimization", device=device,
        source_sha256=protected, phase_dx=list(range(16)), phase_dy=list(range(16)),
        orientations=ORIENTATIONS, threshold=.5, prediction_reads_labels=False,
        unchanged="checkpoint, source PNG, line grayscale and width; no resizing",
        caveats=["Phase shifts also alter distance to the tile frame; X1 can depend on frame padding.",
                 "Flips/rotations change symbol/text orientation; restoration preserves evaluation geometry.",
                 "Best phase selected post-hoc on this case for explanation only.",
                 "Reference-column coverage and draft-path connectivity are not complete-net accuracy."],
        best_phase_selection="most left reference columns >=.5, then interior median score")
    (output/"protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    records = []
    predictions = {}
    best = None
    for dy in range(16):
        for dx in range(16):
            tile = np.full_like(original, 255)
            tile[dy:dy+h, dx:] = rgb[:, 320:832-dx]
            # The entire retained source area is pixel-identical after shifting.
            assert np.array_equal(tile[dy:dy+h, dx:], rgb[:, 320:832-dx])
            p = predict(model, tile, device)
            restored = np.full((2, h, 832), np.nan, np.float32)
            restored[:, :, 320:832-dx] = p[:, dy:dy+h, dx:]
            key = f"shift_{dx}_{dy}"
            record = dict(key=key, dx=dx, dy=dy, kind="phase", **scores(rgb, restored))
            records.append(record)
            if dx == dy == 0:
                with np.load(native_cache, allow_pickle=False) as cache:
                    expected = np.stack([cache["wire"], cache["pin_tip"]])
                assert np.array_equal(restored[:, :, 320:], expected[:, :, 320:])
                predictions[key] = restored
            if best is None or ranking(record) > ranking(best):
                if best is not None and best["key"] != "shift_0_0":
                    del predictions[best["key"]]
                best = record
                predictions[key] = restored
        print(json.dumps(dict(completed_dy=dy, best_so_far=compact_score(best))), flush=True)
    for name in ORIENTATIONS:
        tile = input_transform(original, name)
        inverse = restore_orientation(tile.transpose(2, 0, 1), name).transpose(1, 2, 0)
        assert np.array_equal(inverse, original)
        p = restore_orientation(predict(model, tile, device), name)
        restored = global_probability(p, h)
        record = dict(key=name, kind="orientation", **scores(rgb, restored))
        records.append(record); predictions[name] = restored
        print(json.dumps(compact_score(record)), flush=True)
    # Source paths enter only after every pixel-only prediction is complete.
    from verify_unet_contrast import path_specs, path_records
    from pcb_parser.dataset_converter import load_source_annotation
    manifest = [json.loads(line) for line in (ROOT/"data/manifest.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    raw_row = next(row for row in manifest if row["case_id"] == "0036")
    raw_path = Path(raw_row["raw_annotation_path"])
    raw_hash = sha(raw_path)
    annotation, _ = load_source_annotation(raw_path)
    specs = [p for p in path_specs(annotation, rgb.shape[:2]) if p["edge_id"] in ("E_1", "E_8", "E_10")]
    selected_paths = {}
    for key, p in predictions.items():
        selected_paths[key] = path_records(p[0] >= .5, specs)
        np.savez_compressed(output/f"{key}.npz", wire=p[0], pin_tip=p[1])
    assert sha(raw_path) == raw_hash
    render(rgb, records, predictions, best, output)
    assert all(sha(Path(path)) == digest for path, digest in protected.items())
    result = dict(status="completed_diagnostic_not_adopted", source_files_unchanged=True,
        production_settings_changed=False, training=False, best_phase=compact_score(best),
        best_phase_selection_is_post_hoc=True, records=records,
        selected_draft_paths=selected_paths, evaluation_annotation_sha256={str(raw_path): raw_hash})
    (output/"diagnosis.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    lines = ["# 0036 X1 位置及朝向诊断", "", "固定原v1权重、原尺寸、阈值0.5。右tile完成16×16种位移，原内容只移位且只丢弃右侧白色空列；翻转/旋转不缩放或改变灰度。所有输出还原到原坐标。", "",
        "位移同时改变采样相位与边界距离；朝向实验同时改变文字/元件朝向，不能独立归因。最佳位置是本页诊断的事后选择，不作为正式参数。", "",
        "| 变换 | X1 左线分数中位数 | 左线内部检出 | 左线参考范围覆盖 | X1 右线分数中位数 | DOUT 分数中位数 |",
        "|---|---:|---:|---:|---:|---:|"]
    for key in predictions:
        rec = next(r for r in records if r["key"] == key)
        a, b, g = rec["regions"]
        lines.append(f"| {key} | {a['score_median']:.5f} | {a['samples_above_050']}/61 | {rec['x1_left_reference_columns_above_050']}/91 | {g['score_median']:.5f} | {b['score_median']:.5f} |")
    lines += ["", "## 所选变换的草稿路径连通", "", "源路径±2px管道、端点允许2px、8邻域，保持预测原样；不是完整网络正确率。", "",
        "| 变换 | 边 | 路径端点 | 连通 |", "|---|---|---|---|"]
    for key, paths in selected_paths.items():
        for path in paths:
            lines.append(f"| {key} | {path['edge_id']} | {path['points']} | {path['connected_within_reference_tube']} |")
    lines += ["", f"最佳位移：{best['key']}。原图、权重、配置与旧输出SHA256前后相同；原位置概率重放逐值一致。", "所有预测完成后才读取源标注作路径评估，没有标注参与预测、没有图像传统阈值分割或拓扑补线。"]
    (output/"report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps(dict(output=str(output), best_phase=compact_score(best), paths=selected_paths)), flush=True)


if __name__ == "__main__":
    main()
