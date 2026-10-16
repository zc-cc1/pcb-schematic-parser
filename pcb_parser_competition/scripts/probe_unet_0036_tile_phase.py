"""Follow-up phase sweep preserving all original pixels in the right tile."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from diagnose_unet_0036_context import ROOT, REGIONS, measure, overlay, right_tile_prediction, sha, plt
from pcb_parser.unet_model import WirePinUNet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT/"runs/unet_context_diagnosis_0036_phase")
    args = parser.parse_args()
    output = args.output.resolve()
    if ROOT not in output.parents or output.exists():
        raise ValueError("Use a new project output directory")
    image_path = ROOT/"data/standardized/0036/0036.png"
    weights_path = ROOT/"weights/unet_wire_pin_v1.pt"
    cache_path = ROOT/"runs/unet_context_diagnosis_0036/native.npz"
    paths = [image_path, weights_path, cache_path, ROOT/"configs/unet_inference_v2.json"]
    protected = {str(p): sha(p) for p in paths}
    rgb = np.asarray(Image.open(image_path).convert("RGB"))
    config = json.loads(paths[-1].read_text(encoding="utf-8"))
    assert sha(weights_path) == config["checkpoint_sha256"]
    torch.set_num_threads(4)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)
    model = WirePinUNet(checkpoint["base_channels"]).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    shifts = list(range(33))
    output.mkdir(parents=True)
    protocol = dict(shifts=shifts, threshold=.5, regions=REGIONS[:2], device=device,
        source_sha256=protected, prediction_reads_labels=False, training=False,
        production_settings_changed=False,
        original_pixels_preserved="All original right-tile pixels retained exactly; source geometry unchanged.",
        limitations="Joint sensitivity to grid phase and padded boundary distance; selected-case diagnosis only.")
    (output/"protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    rows = []
    for dy in shifts:
        restored = right_tile_prediction(model, rgb, device, dy)
        scores = [measure(rgb, restored[0], r) for r in REGIONS[:2]]
        rows.append(dict(shift_y=dy, regions=scores))
        if dy == 0:
            with np.load(cache_path, allow_pickle=False) as cache:
                assert np.array_equal(restored[:, :, 512:], np.stack([cache["wire"], cache["pin_tip"]])[:, :, 512:])
        if dy in (0, 1, 16, 17, 32):
            np.savez_compressed(output/f"shift_y_{dy}.npz", wire=restored[0], pin_tip=restored[1])
        print(json.dumps(dict(shift_y=dy, medians=[r["score_median"] for r in scores],
                             coverage=[r["samples_above_050"] for r in scores])), flush=True)
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"], "axes.unicode_minus": False})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=160, layout="constrained")
    for col, region in enumerate(REGIONS[:2]):
        medians = [r["regions"][col]["score_median"] for r in rows]
        axes[col].plot(shifts, medians, "o-", color="#2463c5", ms=3)
        axes[col].axhline(.5, color="#ba3b37", ls="--", label="当前阈值 0.5")
        axes[col].set(ylim=(0, 1.03), xticks=[0, 1, 4, 8, 12, 16, 20, 24, 28, 32],
                     title=region["name"], xlabel="右侧图块整体向下平移（px）", ylabel="导线分数中位数")
        axes[col].grid(alpha=.22, axis="y"); axes[col].legend(frameon=False)
    fig.suptitle("像素和相对内容不变，输出还原：细线分数随图块位置明显波动", fontsize=15)
    fig.savefig(output/"phase_sweep.png", facecolor="white"); plt.close(fig)
    with np.load(output/"shift_y_0.npz") as a, np.load(output/"shift_y_1.npz") as b:
        wire0, wire1 = a["wire"].copy(), b["wire"].copy()
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 5.5), dpi=160, layout="constrained")
    for row, region in enumerate(REGIONS[:2]):
        x0, y0, x1, y1 = region["crop"]
        for col, (background, title) in enumerate([(rgb, "原图像素"),
                (overlay(rgb, wire0), "原始位置的分割"),
                (overlay(rgb, wire1), "向下移 1px 后的分割（已还原坐标）")]):
            axes[row, col].imshow(background[y0:y1, x0:x1], interpolation="nearest")
            if col:
                score = rows[col-1]["regions"][row]
                title += f"\n中位数 {score['score_median']:.3f} · 检出 {score['samples_above_050']}/{score['samples']}"
            axes[row, col].set_title(title, fontsize=12); axes[row, col].axis("off")
    fig.suptitle("同一原图、同一权重、同一阈值：蓝色为模型分数 ≥ 0.5", fontsize=16)
    fig.savefig(output/"tile_phase_effect.png", facecolor="white"); plt.close(fig)
    assert all(sha(Path(p)) == h for p, h in protected.items())
    record = dict(status="completed_diagnostic_not_adopted", source_files_unchanged=True,
        coordinates_restored=True, rows=rows, scores_are_not_accuracy=True)
    (output/"phase_sweep.json").write_text(json.dumps(record, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    lines = ["# 0036 右 tile 位移扫描", "", "固定原 v1 权重、阈值 0.5；完整保留右 tile 原图像素，所有输出还原到原始坐标。", "",
        "| 向下平移 px | X1 分数中位数 | X1 检出截面 | DOUT 分数中位数 | DOUT 检出截面 |",
        "|---:|---:|---:|---:|---:|"]
    for row in rows:
        a, b = row["regions"]
        lines.append(f"| {row['shift_y']} | {a['score_median']:.5f} | {a['samples_above_050']}/{a['samples']} | {b['score_median']:.5f} | {b['samples_above_050']}/{b['samples']} |")
    lines += ["", "未重训练、未读取标签、未改变原图/权重/正式推理配置。",
        "平移同时改变网格采样相位及与白色边界的距离；位移敏感性成立，但无法单独锁定具体网络层。",
        "池化层和双线性上采样是后续检查对象，不应仅凭本实验宣布唯一根因。",
        "取样段分数及检出率不代表整条参考路径或完整网络已连通。"]
    (output/"report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps(dict(output=str(output), source_files_unchanged=True)), flush=True)


if __name__ == "__main__":
    main()
