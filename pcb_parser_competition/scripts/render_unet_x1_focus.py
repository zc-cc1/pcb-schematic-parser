"""Compact original / shift / rotation inspection of the completed X1 probe."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from diagnose_unet_0036_context import ROOT, overlay, sha


def main():
    folder = ROOT/"runs/unet_x1_diagnosis_0036"
    protocol = json.loads((folder/"protocol.json").read_text(encoding="utf-8"))
    result = json.loads((folder/"diagnosis.json").read_text(encoding="utf-8"))
    assert all(sha(Path(p)) == h for p, h in protocol["source_sha256"].items())
    rgb = np.asarray(Image.open(ROOT/"data/standardized/0036/0036.png").convert("RGB"))
    best = result["best_phase"]["key"]
    keys = [None, "shift_0_0", best, "rotate_90"]
    names = ["原图 · 导线颜色和线宽不变", "原模型输出", "本页位移中覆盖最多的位置", "旋转90°预测，再还原输出"]
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"], "axes.unicode_minus": False})
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 5.9), dpi=160, layout="constrained")
    for ax, key, title in zip(axes.ravel(), keys, names):
        background = rgb
        if key is not None:
            with np.load(folder/f"{key}.npz", allow_pickle=False) as source:
                background = overlay(rgb, source["wire"])
            record = next(r for r in result["records"] if r["key"] == key)
            score = record["regions"][0]
            if key == best:
                title += f"（右移{record['dx']} / 下移{record['dy']}px）"
            title += f"\n左线中段分数 {score['score_median']:.3f} · 检出 {score['samples_above_050']}/61"
            title += f"\n参考全段像素覆盖 {record['x1_left_reference_columns_above_050']}/91 · 尚未连通"
        ax.imshow(background[54:103, 587:788], interpolation="nearest")
        if key == "rotate_90":
            ax.annotate("边框误检", xy=(5, 36), xytext=(35, 44), fontsize=11, color="#c33328",
                        arrowprops=dict(arrowstyle="->", color="#c33328", lw=1.2))
        ax.set_title(title, fontsize=12); ax.axis("off")
    fig.suptitle("X1：小位移恢复有限，旋转明显恢复中段，端点仍有缺口\n蓝色为导线分数 ≥ 0.5；保持同一原权重", fontsize=16)
    fig.savefig(folder/"x1_focus.png", facecolor="white"); plt.close(fig)
    assert all(sha(Path(p)) == h for p, h in protocol["source_sha256"].items())
    print(str(folder/"x1_focus.png"))


if __name__ == "__main__":
    main()
