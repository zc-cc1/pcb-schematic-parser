"""Offline matched-wire and context/translation probes for the frozen v1 U-Net.

All coordinates are manually chosen straight-wire interiors on original 0036.
Inputs are changed only in memory; no labels, topology repair or production
configuration are used. These are local sensitivity probes, not new settings.
"""
import argparse
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
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pcb_parser.unet_data import predict_tiled, tile_starts
from pcb_parser.unet_model import WirePinUNet

REGIONS = [
    dict(id="A", name="X1 左侧横线", axis="h", center=80, start=620, end=680,
         crop=[588, 60, 727, 96]),
    dict(id="B", name="DOUT 上方横线", axis="h", center=181, start=605, end=628,
         crop=[588, 161, 652, 223]),
    dict(id="C", name="ANT 下方竖线", axis="v", center=111, start=58, end=100,
         crop=[94, 46, 130, 111]),
    dict(id="D", name="C4 下方竖线", axis="v", center=448, start=235, end=267,
         crop=[430, 224, 467, 280]),
    dict(id="E", name="CAGC 左侧横线", axis="h", center=181, start=453, end=480,
         crop=[440, 163, 498, 208]),
    dict(id="F", name="RFIN 右侧横线", axis="h", center=113, start=280, end=475,
         crop=[264, 96, 492, 132]),
]
# Preserve the complete visible target horizontals, including their end pixels.
TARGET_BOXES = [[594, 78, 694, 83], [594, 179, 636, 184]]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def band(array, region):
    c, s, e = (region[k] for k in ("center", "start", "end"))
    if region["axis"] == "h":
        return array[c-1:c+2, s:e+1], 0
    return array[s:e+1, c-1:c+2], 1


def measure(rgb, wire, region):
    # These selected interiors are grayscale. Use the actual PNG channel values,
    # not screen appearance; width <200 is an offline descriptive measurement.
    pixels, axis = band(rgb, region)
    assert np.array_equal(pixels[..., 0], pixels[..., 1])
    assert np.array_equal(pixels[..., 0], pixels[..., 2])
    gray = pixels[..., 0]
    scores, _ = band(wire, region)
    core = gray.min(axis=axis)
    profile = scores.max(axis=axis)
    levels, counts = np.unique(core, return_counts=True)
    return dict(region_id=region["id"], name=region["name"],
        core_gray_median=float(np.median(core)),
        core_gray_counts={str(int(v)): int(n) for v, n in zip(levels, counts)},
        median_width_below_gray_200=float(np.median((gray < 200).sum(axis=axis))),
        median_integrated_ink=float(np.median((255-gray.astype(float)).sum(axis=axis))),
        score_median=float(np.median(profile)), score_min=float(profile.min()),
        score_max=float(profile.max()), samples=len(profile),
        samples_above_050=int((profile >= .5).sum()), score_profile=profile.tolist())


def overlay(rgb, wire):
    result = rgb.copy()
    selected = wire >= .5
    result[selected] = (.25*result[selected] + .75*np.array([0, 130, 235])).astype(np.uint8)
    return result


def rect(region):
    c, s, e = (region[k] for k in ("center", "start", "end"))
    return (s-1, c-2, e-s+3, 5) if region["axis"] == "h" else (c-2, s-1, 5, e-s+3)


def render_matched(rgb, wire, rows, output):
    chosen = REGIONS[:4]
    fig = plt.figure(figsize=(14, 10.2), dpi=160, layout="constrained")
    gs = fig.add_gridspec(3, 4, height_ratios=[1.7, 1, 1])
    for col, (background, title) in enumerate([(rgb, "0036 原图：橙框为直线取样段"),
                                               (overlay(rgb, wire), "原模型输出：蓝色为分数 ≥ 0.5")]):
        ax = fig.add_subplot(gs[0, 2*col:2*col+2])
        ax.imshow(background, interpolation="nearest")
        for region in chosen:
            x, y, w, h = rect(region)
            ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor="#df7700", lw=1.4))
            ax.text(x-9, y-7, region["id"], color="#ab5000", fontsize=15, weight="bold",
                    bbox=dict(facecolor="white", edgecolor="none", alpha=.9, pad=1))
        ax.axis("off"); ax.set_title(title, fontsize=14)
    for col, (region, row) in enumerate(zip(chosen, rows)):
        x0, y0, x1, y1 = region["crop"]
        for r, (background, title) in enumerate([(rgb, "原图像素"), (overlay(rgb, wire), "原模型分割")], 1):
            ax = fig.add_subplot(gs[r, col])
            ax.imshow(background[y0:y1, x0:x1], interpolation="nearest")
            ax.axis("off")
            if r == 1:
                ax.set_title(f"{region['id']} · {region['name']}\n线芯灰度 {row['core_gray_median']:.0f} · 宽度 {row['median_width_below_gray_200']:.0f}px", fontsize=13)
            else:
                ax.set_title(f"分数中位数 {row['score_median']:.3f}\n检出 {row['samples_above_050']}/{row['samples']} 个取样截面", fontsize=13)
    fig.suptitle("同样细、灰度相近的导线，模型响应可以相差很大", fontsize=19)
    fig.savefig(output/"matched_gray_comparison.png", facecolor="white")
    plt.close(fig)


@torch.inference_mode()
def right_tile_prediction(model, rgb, device, dy):
    # Translation retains every original pixel in the right tile. White bottom
    # padding is sufficient; no source content is resized, recolored or removed.
    h = rgb.shape[0]
    tile = np.full((512, 512, 3), 255, np.uint8)
    tile[dy:dy+h] = rgb[:, 320:832]
    assert np.array_equal(tile[dy:dy+h], rgb[:, 320:832])
    model.eval()
    batch = torch.from_numpy(tile.transpose(2, 0, 1).copy()).float().unsqueeze(0).to(device)/255
    p = model(batch).sigmoid()[0].cpu().numpy()[:, dy:dy+h]
    restored = np.full((2, h, rgb.shape[1]), np.nan, np.float32)
    restored[:, :, 320:832] = p
    return restored


def render_probes(rgb, variants, results, output):
    names = ["native", "line_only_gamma_1.5", "context_only_gamma_1.5", "whole_gamma_1.5"]
    titles = ["原始输入", "只加深目标横线", "只加深目标线以外", "全图加深"]
    fig, axes = plt.subplots(2, 4, figsize=(15, 6), dpi=160, layout="constrained")
    for row, region in enumerate(REGIONS[:2]):
        x0, y0, x1, y1 = region["crop"]
        for col, (key, title) in enumerate(zip(names, titles)):
            axes[row, col].imshow(overlay(rgb, variants[key][0])[y0:y1, x0:x1], interpolation="nearest")
            r = results[key][row]
            axes[row, col].set_title(f"{title}\n中位数 {r['score_median']:.3f} · {r['samples_above_050']}/{r['samples']}", fontsize=13)
            axes[row, col].axis("off")
    fig.suptitle("控制实验：显示原图背景 + 各输入的分割；权重和阈值相同", fontsize=17)
    fig.savefig(output/"contrast_controls.png", facecolor="white"); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.8), dpi=160, layout="constrained")
    shifts = [0, 1, 2, 4, 8, 16]
    for col, region in enumerate(REGIONS[:2]):
        scores = [results[f"right_tile_shift_y_{dy}"][col]["score_median"] for dy in shifts]
        axes[col].plot(shifts, scores, "o-", color="#2463c5")
        axes[col].axhline(.5, ls="--", color="#ba3b37", label="当前阈值 0.5")
        axes[col].set(ylim=(0, 1.05), xticks=shifts, xlabel="整块原图在白色 tile 内向下平移（px）",
                      ylabel="同一导线取样分数中位数", title=region["name"])
        axes[col].grid(axis="y", alpha=.25); axes[col].legend(frameon=False)
    fig.suptitle("像素、线宽及相对内容保持相同：输出还原到原坐标后测量", fontsize=14)
    fig.savefig(output/"translation_probe.png", facecolor="white"); plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT/"runs/unet_context_diagnosis_0036")
    args = parser.parse_args()
    output = args.output.resolve()
    if ROOT not in output.parents or output.exists():
        raise ValueError("Use a new directory inside the project")
    image_path = ROOT/"data/standardized/0036/0036.png"
    weights_path = ROOT/"weights/unet_wire_pin_v1.pt"
    config_path = ROOT/"configs/unet_inference_v2.json"
    cache_path = ROOT/"runs/unet_contrast_verification_v1/0036/gamma_1.0/probabilities.npz"
    frozen = {str(path): sha(path) for path in [image_path, weights_path, config_path, cache_path]}
    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert sha(weights_path) == config["checkpoint_sha256"]
    rgb = np.asarray(Image.open(image_path).convert("RGB"))
    assert rgb.shape == (376, 832, 3)
    assert tile_starts(832, 512, 384) == [0, 320]
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)
    assert checkpoint["config"]["tile_size"] == 512 and checkpoint["config"]["stride"] == 384
    torch.set_num_threads(4)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = WirePinUNet(checkpoint["base_channels"]).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    output.mkdir(parents=True)
    protocol = dict(scope="offline local sensitivity only; no adoption or training",
        source_sha256=frozen, device=device, gray_scale="0 black / 255 white",
        threshold=.5, profile="maximum within center +/-1px per orthogonal section",
        regions=REGIONS, target_boxes=TARGET_BOXES,
        translations=[0, 1, 2, 4, 8, 16], gamma_values=[1.5, 2.0],
        prediction_reads_labels=False,
        caveats=["Widths below gray 200 are descriptive offline measurements, not segmentation.",
                 "Context-only means every original pixel outside the two protected target strips.",
                 "Tile translation changes sampling phase and distance to padded boundaries together.",
                 "Scores and selected-line coverage are not accuracy or complete-net correctness."])
    (output/"protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    variants = {"native": predict_tiled(model, rgb, device, 512, 384)}
    with np.load(cache_path, allow_pickle=False) as cached:
        replay = dict(max_abs_wire_diff=float(np.abs(variants["native"][0]-cached["wire"]).max()),
                      changed_wire_mask_pixels=int(((variants["native"][0] >= .5) != (cached["wire"] >= .5)).sum()))
    assert replay["max_abs_wire_diff"] < .002 and replay["changed_wire_mask_pixels"] == 0
    results = {"native": [measure(rgb, variants["native"][0], r) for r in REGIONS]}
    print(json.dumps({"variant": "native", "measurements": results["native"]}, ensure_ascii=False), flush=True)
    target = np.zeros(rgb.shape[:2], bool)
    for x0, y0, x1, y1 in TARGET_BOXES:
        target[y0:y1, x0:x1] = True
    intervention_checks = {}
    for gamma in (1.5, 2.0):
        adjusted = np.round(255*(rgb.astype(np.float32)/255)**gamma).astype(np.uint8)
        for mode in ("line_only", "context_only", "whole"):
            pixels = rgb.copy()
            chosen = target if mode == "line_only" else ~target if mode == "context_only" else np.ones_like(target)
            pixels[chosen] = adjusted[chosen]
            assert np.array_equal(pixels[~chosen], rgb[~chosen])
            key = f"{mode}_gamma_{gamma:.1f}"
            intervention_checks[key] = dict(changed_pixels=int((pixels != rgb).any(axis=2).sum()),
                protected_target_unchanged=bool(np.array_equal(pixels[target], rgb[target])),
                outside_target_unchanged=bool(np.array_equal(pixels[~target], rgb[~target])))
            variants[key] = predict_tiled(model, pixels, device, 512, 384)
            results[key] = [measure(pixels, variants[key][0], r) for r in REGIONS]
            print(json.dumps({"variant": key, "target_scores": [r["score_median"] for r in results[key][:2]]}), flush=True)
    for dy in (0, 1, 2, 4, 8, 16):
        key = f"right_tile_shift_y_{dy}"
        variants[key] = right_tile_prediction(model, rgb, device, dy)
        results[key] = [measure(rgb, variants[key][0], r) for r in REGIONS if r["id"] in ("A", "B", "D", "E")]
        if dy == 0:
            assert np.array_equal(variants[key][:, :, 512:], variants["native"][:, :, 512:])
        print(json.dumps({"variant": key, "target_scores": [r["score_median"] for r in results[key][:2]]}), flush=True)
    for key, values in variants.items():
        np.savez_compressed(output/f"{key}.npz", wire=values[0], pin_tip=values[1])
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"], "axes.unicode_minus": False})
    render_matched(rgb, variants["native"][0], results["native"], output)
    render_probes(rgb, variants, results, output)
    assert all(sha(Path(path)) == digest for path, digest in frozen.items())
    record = dict(status="completed_diagnostic_not_adopted", source_files_unchanged=True,
        production_settings_changed=False, training_performed=False, scores_are_not_accuracy=True,
        replay=replay, intervention_checks=intervention_checks, variants=results)
    (output/"diagnosis.json").write_text(json.dumps(record, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    lines = ["# 0036 灰度、上下文和位移诊断", "", "固定原 v1 权重、原尺寸、阈值 0.5；未训练或修改正式设置。", "",
        "灰度 0=黑、255=白；分数为每个截面中心附近 ±1px 的最大导线分数，再取中位数。不是准确率。", "",
        "| 段 | 方向 | 线芯灰度 | 宽度<200 | 累计墨量 | 分数中位数 | 检出截面 |",
        "|---|---|---:|---:|---:|---:|---:|"]
    for region, r in zip(REGIONS, results["native"]):
        lines.append(f"| {r['name']} | {region['axis']} | {r['core_gray_median']:.0f} | {r['median_width_below_gray_200']:.0f} | {r['median_integrated_ink']:.0f} | {r['score_median']:.5f} | {r['samples_above_050']}/{r['samples']} |")
    lines += ["", "## 控制实验", "", "| 输入变化 | X1 分数中位数 | X1 检出截面 | DOUT 分数中位数 | DOUT 检出截面 |", "|---|---:|---:|---:|---:|"]
    for key, rows in results.items():
        a, b = rows[:2]
        lines.append(f"| {key} | {a['score_median']:.5f} | {a['samples_above_050']}/{a['samples']} | {b['score_median']:.5f} | {b['samples_above_050']}/{b['samples']} |")
    lines += ["", "上下文实验仅保护两段目标横线及周围 2px；其余整张图加深，改变了元件、文字及其他导线。",
        "位移实验完整保留右 tile 的所有原始像素和相对位置，向下平移后还原输出坐标；同时改变池化采样相位和边界距离，不能单独确定某一层是原因。",
        "目标位于原 x>512，只有右 tile 覆盖，因此不是 tile 拼接平均直接压低这些位置。",
        "这些实验检测模型敏感性，不代表已确定训练错误、网络结构或标签问题中的唯一根因。",
        "所有原图、权重、配置及旧输出 SHA256 前后相同；原始预测重放掩膜一致。"]
    (output/"report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "replay": replay, "source_files_unchanged": True}), flush=True)


if __name__ == "__main__":
    main()
