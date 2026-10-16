"""Trace the frozen U-Net and intervene on its final skip/decode branches.

Prediction uses pixels and checkpoint only. Coarse-layer comparisons require
interpolation and are descriptive. Final-branch swaps use full-resolution
features and integer coordinate restoration, providing a separate intervention.
"""
import argparse
import json
import math
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from diagnose_unet_0036_context import ROOT, REGIONS, measure, sha
sys.path.insert(0, str(ROOT/"src"))
from pcb_parser.unet_model import WirePinUNet

STAGES = ["inc", "pool1", "down1", "pool2", "down2", "pool3", "down3",
          "pool4", "down4", "up1", "up2", "up3", "up4_concat", "up4", "out"]


def make_tile(rgb, dy):
    tile = np.full((512, 512, 3), 255, np.uint8)
    tile[dy:dy+rgb.shape[0]] = rgb[:, 320:832]
    assert np.array_equal(tile[dy:dy+rgb.shape[0]], rgb[:, 320:832])
    return tile


@torch.inference_mode()
def trace(model, tile, device):
    features = {}
    handles = []
    modules = {"inc": model.inc, **{f"down{i}": getattr(model, f"down{i}") for i in range(1, 5)},
        **{f"pool{i}": getattr(model, f"down{i}").maxpool_conv[0] for i in range(1, 5)},
        **{f"up{i}": getattr(model, f"up{i}") for i in range(1, 5)}, "out": model.out}
    def capture(name):
        def hook(module, inputs, output):
            features[name] = output.detach().cpu().clone()
        return hook
    for name, module in modules.items():
        handles.append(module.register_forward_hook(capture(name)))
    def concat_hook(module, inputs):
        features["up4_concat"] = inputs[0].detach().cpu().clone()
    handles.append(model.up4.conv.register_forward_pre_hook(concat_hook))
    try:
        batch = torch.from_numpy(tile.transpose(2, 0, 1).copy()).float().unsqueeze(0).to(device)/255
        p = model(batch).sigmoid()[0].cpu().numpy()
    finally:
        for handle in handles:
            handle.remove()
    assert set(features) == set(STAGES)
    return p, features


def restored_probability(p, dy, image_shape):
    h, w = image_shape[:2]
    restored = np.full((2, h, w), np.nan, np.float32)
    restored[:, :, 320:832] = p[:, dy:dy+h]
    return restored


def sample_roi(feature, region, dy, radius=6):
    """Sample a coarse feature in original pixel coordinates for description.

    Encoder grids use center (scale-1)/2; decoder grids use the corresponding
    skip's grid as a convention. Alignment at coarse stages is approximate and
    may itself contribute interpolation error. Full-resolution sampling is exact.
    """
    assert region["axis"] == "h"
    x = np.arange(region["start"]-320, region["end"]-320+1)
    y = np.arange(region["center"]-radius, region["center"]+radius+1) + dy
    _, _, h, w = feature.shape
    if h == 512:
        return feature[0, :, y[0]:y[-1]+1, x[0]:x[-1]+1].numpy().copy()
    scale = 512/h
    origin = (scale-1)/2
    yy, xx = np.meshgrid((y-origin)/scale, (x-origin)/scale, indexing="ij")
    grid = np.stack([2*xx/(w-1)-1, 2*yy/(h-1)-1], axis=-1).astype(np.float32)
    return F.grid_sample(feature, torch.from_numpy(grid[None]), align_corners=True,
                         mode="bilinear", padding_mode="border")[0].numpy()


def difference(a, b):
    delta = a.astype(np.float64)-b.astype(np.float64)
    amplitude = math.sqrt(float((a.astype(np.float64)**2).mean()))
    other = math.sqrt(float((b.astype(np.float64)**2).mean()))
    return dict(max_abs=float(np.abs(delta).max()), mean_abs=float(np.abs(delta).mean()),
                normalized_rmse=math.sqrt(float((delta**2).mean()))/max((amplitude+other)/2, 1e-12),
                baseline_rms=amplitude, shifted_rms=other)


def restore_full_feature(donor, baseline, dy=1):
    assert donor.shape == baseline.shape and donor.shape[-2:] == (512, 512)
    result = baseline.clone()
    result[:, :, :-dy] = donor[:, :, dy:]
    return result


@torch.inference_mode()
def decode_final_concat(model, concatenated, device):
    return model.out(model.up4.conv(concatenated.to(device))).sigmoid()[0].cpu().numpy()


@torch.inference_mode()
def change_first_pool_origin(model, original_inc, baseline_concat, dy, device):
    """Same saved shallow features; change the first pooling grid's origin.

    Move the feature within the fixed tensor before pooling, then restore the
    upsampled deep branch to original coordinates before joining the unchanged
    shallow skip. The top strip stays baseline; DOUT does not depend on it.
    This is a diagnostic forward, never a production architecture modification.
    """
    shifted = original_inc.clone()
    if dy:
        shifted[:, :, dy:] = original_inc[:, :, :-dy]
    shifted = shifted.to(device)
    pooled = model.down1.maxpool_conv[0](shifted)
    b = model.down1.maxpool_conv[1](pooled)
    c = model.down2(b); d = model.down3(c); e = model.down4(d)
    u3 = model.up3(model.up2(model.up1(e, d), c), b)
    deep = model.up4.up(u3).cpu()
    channels = original_inc.shape[1]
    if dy:
        deep = restore_full_feature(deep, baseline_concat[:, channels:], dy)
    combined = torch.cat([original_inc, deep], dim=1)
    return decode_final_concat(model, combined, device), pooled.cpu()


def union(a, b):
    return min(a[0], b[0]), max(a[1], b[1]), a[2] or b[2]


def conv_interval(lo, hi, size):
    lo, hi = lo-2, hi+2  # Two 3x3 stride-one convolutions.
    return max(lo, 0), min(hi, size-1), lo < 0 or hi >= size


def encoder_support(level, lo, hi):
    size = 512//2**level
    lo, hi, padding = conv_interval(lo, hi, size)
    if level == 0:
        return lo, hi, padding
    result = encoder_support(level-1, 2*lo, 2*hi+1)
    return result[0], result[1], padding or result[2]


def decoder_support(level, lo, hi):
    size = 512//2**level
    lo, hi, padding = conv_interval(lo, hi, size)
    skip = encoder_support(level, lo, hi)
    source_size = size//2
    # Exact index range for bilinear align_corners=True on a doubled dimension.
    source_lo = math.floor(lo*(source_size-1)/(size-1))
    source_hi = math.ceil(hi*(source_size-1)/(size-1))
    deep = encoder_support(4, source_lo, source_hi) if level == 3 else decoder_support(level+1, source_lo, source_hi)
    result = union(skip, deep)
    return result[0], result[1], padding or result[2]


def receptive_support(region, dy):
    x = decoder_support(0, region["start"]-320, region["end"]-320)
    y = decoder_support(0, region["center"]-1+dy, region["center"]+1+dy)
    return dict(x_range_in_tile=list(x[:2]), y_range_in_tile=list(y[:2]),
                touches_any_network_padding=x[2] or y[2],
                intersects_bottom_image_padding=y[1] >= 376+dy)


def validate_architecture(model):
    assert model.up4.up.align_corners and model.up4.up.scale_factor == 2
    for layer in model.modules():
        if isinstance(layer, torch.nn.Conv2d):
            assert layer.stride == (1, 1) and layer.dilation == (1, 1)
            assert layer.kernel_size in ((3, 3), (1, 1))
        if isinstance(layer, torch.nn.BatchNorm2d):
            assert not layer.training and layer.track_running_stats
        if isinstance(layer, torch.nn.MaxPool2d):
            assert layer.kernel_size == 2 and layer.stride == 2 and layer.padding == 0


def render(features, measurements, swaps, probabilities, output):
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"], "axes.unicode_minus": False})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), dpi=155, layout="constrained")
    labels = ["浅层卷积", "池化1", "卷积1", "池化2", "卷积2", "池化3", "卷积3", "池化4", "卷积4", "解码1", "解码2", "解码3", "最终拼接", "解码4", "输出"]
    for col, region in enumerate(REGIONS[:2]):
        values = [measurements[region["id"]][stage]["normalized_rmse"] for stage in STAGES]
        axes[col].plot(range(len(STAGES)), values, "o-", color="#2463c5", ms=4)
        axes[col].set(xticks=range(len(STAGES)), xticklabels=labels, ylabel="对齐后特征差异 / 平均特征幅度", title=region["name"])
        axes[col].tick_params(axis="x", rotation=60); axes[col].grid(axis="y", alpha=.2)
    fig.suptitle("逐层对照：粗层需要插值对齐，曲线仅描述差异，不能单独判定因果", fontsize=14)
    fig.savefig(output/"layer_differences.png", facecolor="white"); plt.close(fig)
    region = REGIONS[1]
    a = sample_roi(features[0]["inc"], region, 0, 8)
    b = sample_roi(features[1]["inc"], region, 1, 8)
    # Select a channel with a strong local line-vs-background response, for
    # explanation only. Channel choice never affects any prediction or statistic.
    contrast = (a[:, 7:10].mean(axis=(1, 2)) - a[:, [0, 1, 15, 16]].mean(axis=(1, 2)))
    channel = int(np.argmax(contrast))
    pool_a = sample_roi(features[0]["pool1"], region, 0, 8)
    pool_b = sample_roi(features[1]["pool1"], region, 1, 8)
    fig, axes = plt.subplots(3, 2, figsize=(9.3, 6.7), dpi=165, layout="constrained")
    for row, (left, right, title) in enumerate([
            (a[channel], b[channel], f"原尺寸浅层特征 · 通道 {channel}（精确整数对齐）"),
            (pool_a[channel], pool_b[channel], "第一次池化 · 同一通道（插值对齐，仅供观察）"),
            (probabilities["baseline"][0, 173:190, 605:629], probabilities["shifted"][0, 173:190, 605:629], "最终导线分数")]):
        vmax = 1 if row == 2 else max(float(left.max()), float(right.max()), 1e-6)
        for col, values in enumerate((left, right)):
            axes[row, col].imshow(values, vmin=0, vmax=vmax, cmap="magma", interpolation="nearest", aspect="auto")
            axes[row, col].axhline(8, color="#25d5f0", lw=.7)
            axes[row, col].axis("off")
            axes[row, col].set_title(title if col == 0 else ("移 1px 并还原：" + ("浅层特征相同" if row == 0 else "特征形状改变" if row == 1 else "分数升高")), fontsize=11)
    fig.suptitle("DOUT：原图细线的浅层响应保留，后续路径产生位置差异", fontsize=15)
    fig.savefig(output/"dout_feature_trace.png", facecolor="white"); plt.close(fig)
    keys = ["baseline", "fine_from_shifted", "deep_from_shifted", "both_from_shifted", "shifted"]
    names = ["原始输出", "只换浅层特征", "只换深层特征", "两者都换", "真实平移输入"]
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 3.8), dpi=160, layout="constrained")
    for col, region in enumerate(REGIONS[:2]):
        values = [next(r for r in swaps[k] if r["region_id"] == region["id"])["score_median"] for k in keys]
        axes[col].bar(range(len(keys)), values, color=["#858e9f", "#efad53", "#258bd2", "#258bd2", "#56a976"])
        for i, value in enumerate(values):
            axes[col].text(i, value+.025, f"{value:.3f}", ha="center", fontsize=11)
        axes[col].axhline(.5, color="#ba3b37", ls="--")
        axes[col].set(xticks=range(len(keys)), xticklabels=names, ylim=(0, 1.04), title=region["name"], ylabel="导线分数中位数")
        axes[col].tick_params(axis="x", rotation=15)
    fig.suptitle("最终解码前的特征替换对照：原尺寸对齐，权重和阈值相同", fontsize=14)
    fig.savefig(output/"branch_interventions.png", facecolor="white"); plt.close(fig)
    return dict(inc_channel=channel, native_local_contrast=float(contrast[channel]),
                channel_selection="post-hoc explanatory only; no influence on predictions")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT/"runs/unet_feature_diagnosis_0036")
    args = parser.parse_args()
    output = args.output.resolve()
    if ROOT not in output.parents or output.exists():
        raise ValueError("Use a new output directory inside the project")
    image_path = ROOT/"data/standardized/0036/0036.png"
    weights_path = ROOT/"weights/unet_wire_pin_v1.pt"
    config_path = ROOT/"configs/unet_inference_v2.json"
    caches = [ROOT/f"runs/unet_context_diagnosis_0036_phase/shift_y_{dy}.npz" for dy in (0, 1)]
    protected = {str(p): sha(p) for p in [image_path, weights_path, config_path, *caches]}
    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert sha(weights_path) == config["checkpoint_sha256"]
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)
    torch.set_num_threads(4)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = WirePinUNet(checkpoint["base_channels"]).to(device).eval()
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    validate_architecture(model)
    rgb = np.asarray(Image.open(image_path).convert("RGB"))
    assert rgb.shape == (376, 832, 3)
    output.mkdir(parents=True)
    protocol = dict(scope="offline trace and intervention; not an optimized model", device=device,
        source_sha256=protected, shifts=[0, 1], regions=REGIONS[:2], threshold=.5,
        prediction_reads_labels=False, production_code_changed=False, stages=STAGES,
        final_swap="At up4.conv input: first base_channels are full-resolution skip, second are upsampled decoder.",
        restoration="donor y+1 maps to baseline y; final unavailable row remains baseline",
        caveats=["Coarse-stage alignment uses bilinear interpolation; difference magnitudes are descriptive.",
                 "Activation donation creates experimental hybrid features, not a deployable prediction.",
                 "Local line scores and coverage do not measure complete-wire/net correctness."])
    (output/"protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    features = []
    probabilities = {}
    replay = []
    for dy, key, cache_path in zip((0, 1), ("baseline", "shifted"), caches):
        p, captured = trace(model, make_tile(rgb, dy), device)
        features.append(captured)
        probabilities[key] = restored_probability(p, dy, rgb.shape)
        with np.load(cache_path, allow_pickle=False) as cache:
            expected = np.stack([cache["wire"], cache["pin_tip"]])
        diff = float(np.abs(probabilities[key][:, :, 320:]-expected[:, :, 320:]).max())
        assert diff == 0
        replay.append(dict(shift_y=dy, max_abs_probability_diff=diff))
        print(json.dumps(dict(trace=key, replay_difference=diff)), flush=True)
    measurements = {}
    roi_archive = {}
    for region in REGIONS[:2]:
        measurements[region["id"]] = {}
        for stage in STAGES:
            a = sample_roi(features[0][stage], region, 0)
            b = sample_roi(features[1][stage], region, 1)
            measurements[region["id"]][stage] = difference(a, b) | dict(shape=list(features[0][stage].shape),
                alignment="exact integer" if features[0][stage].shape[-1] == 512 else "approximate interpolated")
            roi_archive[f"{region['id']}_{stage}_native"] = a
            roi_archive[f"{region['id']}_{stage}_shifted_restored"] = b
        assert measurements[region["id"]]["inc"]["max_abs"] < 2e-5
    np.savez_compressed(output/"feature_rois.npz", **roi_archive)
    # Full-resolution branch intervention avoids coarse fractional alignment.
    base = features[0]["up4_concat"]
    donor = restore_full_feature(features[1]["up4_concat"], base)
    channels = checkpoint["base_channels"]
    assert base.shape == (1, 2*channels, 512, 512)
    hybrids = {"baseline_reassembled": base, "both_from_shifted": donor}
    fine_only = base.clone(); fine_only[:, :channels] = donor[:, :channels]
    deep_only = base.clone(); deep_only[:, channels:] = donor[:, channels:]
    hybrids["fine_from_shifted"] = fine_only
    hybrids["deep_from_shifted"] = deep_only
    for key, concatenated in hybrids.items():
        p = decode_final_concat(model, concatenated, device)
        probabilities[key] = restored_probability(p, 0, rgb.shape)
    assert np.array_equal(probabilities["baseline_reassembled"][:, :, 320:], probabilities["baseline"][:, :, 320:])
    origin_checks = []
    dout = REGIONS[1]
    py = slice(dout["center"]//2-5, dout["center"]//2+6)
    px = slice((dout["start"]-320)//2, (dout["end"]-320)//2+1)
    for dy in (0, 1, 2):
        p, pooled = change_first_pool_origin(model, features[0]["inc"], base, dy, device)
        key = f"same_inc_pool_origin_{dy}"
        probabilities[key] = restored_probability(p, 0, rgb.shape)
        if dy == 0:
            assert np.array_equal(probabilities[key][:, :, 320:], probabilities["baseline"][:, :, 320:])
            assert torch.equal(pooled, features[0]["pool1"])
        if dy == 1:
            match = float((pooled[:, :, py, px]-features[1]["pool1"][:, :, py, px]).abs().max())
            expected = probabilities["shifted"][0, dout["center"]-1:dout["center"]+2, dout["start"]:dout["end"]+1]
            actual = probabilities[key][0, dout["center"]-1:dout["center"]+2, dout["start"]:dout["end"]+1]
            final_diff = float(np.abs(expected-actual).max())
            assert match == 0 and final_diff < 2e-5
            origin_checks.append(dict(shift_y=dy, actual_shift_pool1_local_max_diff=match,
                                      actual_shift_DOUT_max_score_diff=final_diff))
        if dy == 2:
            shifted_py = slice(py.start+1, py.stop+1)
            match = float((pooled[:, :, shifted_py, px]-features[0]["pool1"][:, :, py, px]).abs().max())
            assert match == 0
            origin_checks.append(dict(shift_y=dy, original_pool1_restored_integer_max_diff=match))
    swaps = {key: [measure(rgb, p[0], r) for r in REGIONS[:2]] for key, p in probabilities.items()}
    supports = {r["id"]: {str(dy): receptive_support(r, dy) for dy in (0, 1)} for r in REGIONS[:2]}
    dout_support = supports["B"]["0"]
    assert not dout_support["touches_any_network_padding"] and not dout_support["intersects_bottom_image_padding"]
    # Empirically check that modifying every pixel outside the computed DOUT
    # dependency rectangle has no effect on these final scores.
    tile = make_tile(rgb, 0)
    outside = np.ones((512, 512), bool)
    x0, x1 = dout_support["x_range_in_tile"]; y0, y1 = dout_support["y_range_in_tile"]
    outside[y0:y1+1, x0:x1+1] = False
    changed = tile.copy(); changed[outside] = 255-changed[outside]
    with torch.inference_mode():
        batch = torch.from_numpy(changed.transpose(2, 0, 1).copy()).float().unsqueeze(0).to(device)/255
        outside_p = restored_probability(model(batch).sigmoid()[0].cpu().numpy(), 0, rgb.shape)
    dout = REGIONS[1]
    yy = slice(dout["center"]-1, dout["center"]+2); xx = slice(dout["start"], dout["end"]+1)
    outside_diff = float(np.abs(outside_p[0, yy, xx]-probabilities["baseline"][0, yy, xx]).max())
    assert outside_diff < 1e-6
    checks = dict(native_probability_replay=replay, reassembled_baseline_exact=True,
        first_pool_origin_controls=origin_checks,
        DOUT_outside_support_changed_pixels=int((changed != tile).any(axis=2).sum()),
        DOUT_outside_support_max_abs_score_diff=outside_diff)
    # Exact correspondence checks in the final full-resolution local features.
    for r in REGIONS[:2]:
        y = slice(r["center"]-3, r["center"]+4); x = slice(r["start"], r["end"]+1)
        expected = probabilities["shifted"][0, y, x]
        actual = probabilities["both_from_shifted"][0, y, x]
        assert np.abs(actual-expected).max() < 2e-5
    for key, p in probabilities.items():
        np.savez_compressed(output/f"{key}.npz", wire=p[0], pin_tip=p[1])
    channel = render(features, measurements, swaps, probabilities, output)
    assert all(sha(Path(p)) == digest for p, digest in protected.items())
    record = dict(status="completed_diagnostic_not_adopted", source_files_unchanged=True,
        training=False, production_settings_changed=False, feature_differences=measurements,
        branch_interventions=swaps, receptive_support=supports, validation=checks,
        explanatory_channel=channel)
    (output/"diagnosis.json").write_text(json.dumps(record, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    lines = ["# 0036 U-Net 特征和分支诊断", "", "原 v1 权重、阈值 0.5；输入只有 PNG，没有标注。未训练、未修改生产设置。", "",
        "## 最终解码前的特征替换", "", "浅层=原尺寸 inc 跳跃连接；深层=up3 上采样到原尺寸的解码特征。供体来自平移1px的同一图，按整数坐标还原。", "",
        "| 对照 | X1 分数中位数 | X1 检出截面 | DOUT 分数中位数 | DOUT 检出截面 |", "|---|---:|---:|---:|---:|"]
    for key, rows in swaps.items():
        a, b = rows
        lines.append(f"| {key} | {a['score_median']:.5f} | {a['samples_above_050']}/{a['samples']} | {b['score_median']:.5f} | {b['samples_above_050']}/{b['samples']} |")
    lines += ["", "## 逐层差异", "", "粗层需要插值对齐，表格只描述相对差异；不能仅凭此数值宣布哪个层造成错误。原尺寸浅层 inc 是整数精确对齐。", "",
        "| 层 | 特征尺寸 | X1 归一化RMSE | DOUT 归一化RMSE |", "|---|---|---:|---:|"]
    for stage in STAGES:
        a = measurements["A"][stage]; b = measurements["B"][stage]
        lines.append(f"| {stage} | {a['shape']} | {a['normalized_rmse']:.5f} | {b['normalized_rmse']:.5f} |")
    lines += ["", "## 同一份浅层特征的池化网格对照", "",
        "直接使用原位置已经计算出的完整 inc 特征。改变进入第一次池化的网格原点，后续深层分支按原模型处理；最终把深层输出还原，再与完全未改的原尺寸 inc 跳跃特征融合。",
        "网格原点偏移0px逐值复现原输出；偏移1px在DOUT复现真实输入平移1px的结果，局部首次池化特征逐值相同；偏移2px的首次池化特征恢复1个粗格后与原池化特征逐值相同。",
        "这确认无需改变浅层提取结果，改变首个采样网格即可通过深层路径复现这次输出翻转。后续层仍可能放大或补偿差异，不能据此宣布只改第一层就能修复整体模型。",
        "X1的依赖碰到边界，该网格对照的上方填充不等同于真实输入平移，不使用X1的模拟值做根因结论。",
        "", "## 排除 DOUT 图块边界解释", "", f"DOUT 原位置输出的输入依赖范围：tile x={x0}～{x1}、y={y0}～{y1}，不碰任何网络补边，也不碰原图下方白色补边。",
        f"将依赖范围以外的 {checks['DOUT_outside_support_changed_pixels']} 个像素反色，DOUT 最终分数最大差异 {outside_diff:.9f}。",
        "范围按当前4次池化、3x3卷积、align_corners=True双线性上采样的完整计算图推导；BatchNorm为eval，使用固定running statistics。",
        "X1 的依赖可能碰到补边，不能用DOUT的排除结论代替X1分析。", "",
        "所有旧概率重放逐值一致；原图、权重、冻结配置和旧输出的SHA256前后相同。",
        "局部特征干预是诊断，不是可部署模型；分数/截面检出率不代表整条线或完整网络正确。"]
    (output/"report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps(dict(output=str(output), inc_DOUT=measurements["B"]["inc"],
        branch_scores={k: [r["score_median"] for r in rows] for k, rows in swaps.items()},
        supports=supports, validation=checks), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
