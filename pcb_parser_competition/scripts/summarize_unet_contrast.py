"""Inspect completed contrast diagnostics without rerunning/changing inference."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"runs/unet_contrast_verification_v1"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def intervals(xs):
    if not len(xs):
        return []
    return [[int(v[0]), int(v[-1])] for v in np.split(xs, np.flatnonzero(np.diff(xs)>1)+1)]


def main():
    protocol = json.loads((OUT/"protocol.json").read_text(encoding="utf-8"))
    assert all(sha(path) == digest for path, digest in protocol["source_sha256"].items())
    result = json.loads((OUT/"evaluation.json").read_text(encoding="utf-8"))
    records = result["cases"]
    path_changes = []
    for gamma in (1.5, 2.):
        gained = lost = 0
        for r in records:
            if r["gamma"] != gamma:
                continue
            baseline = next(b for b in records if b["case_id"] == r["case_id"] and b["gamma"] == 1.)
            assert [(p["edge_id"], p["points"]) for p in baseline["paths"]] == [(p["edge_id"], p["points"]) for p in r["paths"]]
            gained += sum(q["connected_within_reference_tube"] and not p["connected_within_reference_tube"] for p, q in zip(baseline["paths"], r["paths"]))
            lost += sum(p["connected_within_reference_tube"] and not q["connected_within_reference_tube"] for p, q in zip(baseline["paths"], r["paths"]))
        path_changes.append({"gamma": gamma, "recovered_connected_paths": gained, "lost_connected_paths": lost})
    local = []
    nonwire_rois = {"title": (194, 285, 535, 313), "note": (674, 156, 792, 178),
                   "model_name": (469, 207, 606, 229), "IC_top_frame": (491, 61, 594, 65),
                   "IC_bottom_frame": (491, 203, 594, 207)}
    for gamma in (1., 1.5, 2.):
        mask = np.asarray(Image.open(OUT/"0036"/f"gamma_{gamma:.1f}"/"wire_mask.png")) > 0
        gaps = []
        for name, y, x0, x1 in (("X1_left", 80, 601, 691), ("DOUT_horizontal", 181, 601, 634)):
            supported = mask[y-2:y+3, x0:x1+1].any(axis=0)
            gaps.append({"name": name, "sample_y": y, "sample_x_range": [x0, x1],
                         "unsupported_x_intervals": intervals(np.arange(x0, x1+1)[~supported])})
        counts = {}
        for name, (x0, y0, x1, y1) in nonwire_rois.items():
            counts[name] = int(mask[y0:y1, x0:x1].sum())
        local.append({"gamma": gamma, "gap_sampling": "any predicted pixel within y +/- 2, no bridging",
                      "gaps": gaps, "predicted_pixels_in_selected_nonwire_rois": counts})
    inspection = {"case_id": "0036", "scope": "diagnostic only; human-selected nonwire rectangles and source-path measurements",
                  "nonwire_roi_xyxy_exclusive": nonwire_rois, "path_changes_all_26_pages": path_changes,
                  "local": local, "evaluation_sha256": sha(OUT/"evaluation.json"),
                  "used_in_training": False, "production_settings_changed": False}
    (OUT/"local_inspection.json").write_text(json.dumps(inspection, ensure_ascii=False, indent=2), encoding="utf-8")
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"], "axes.unicode_minus": False})
    rgb = np.asarray(Image.open(ROOT/"data/standardized/0036/0036.png").convert("RGB"))
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 6.4), dpi=150, layout="constrained")
    titles = ["原图", "当前 U-Net 分割", "加深输入 γ=2.0 后的分割"]
    crops = [(551, 31, 743, 106), (551, 159, 735, 238)]
    for col, gamma in enumerate((None, 1., 2.)):
        display = rgb.copy()
        if gamma is not None:
            mask = np.asarray(Image.open(OUT/"0036"/f"gamma_{gamma:.1f}"/"wire_mask.png")) > 0
            display[mask] = (.15*display[mask]+.85*np.array([0, 130, 235])).astype(np.uint8)
        for row, (x0, y0, x1, y1) in enumerate(crops):
            ax = axes[row, col]
            ax.imshow(display[y0:y1, x0:x1], interpolation="nearest")
            ax.axis("off")
            ax.set_title(titles[col]+("\nX1 左侧引线" if row == 0 else "\nDOUT 引线"), fontsize=14)
            if col == 2:
                if row == 0:
                    left, right, y, text = 685, 691, 80, "末端仍缺 7 px"
                else:
                    left, right, y, text = 601, 604, 181, "靠近 IC 端仍缺 4 px"
                ax.add_patch(Rectangle((left-x0-.5, y-y0-3.5), right-left+1, 7,
                                       facecolor="none", edgecolor="#c04e00", linewidth=1.5))
                ax.annotate(text, (left-x0, y-y0), xytext=(14, 16), textcoords="offset points",
                            color="#a83f00", fontsize=12, arrowprops={"arrowstyle": "->", "color": "#a83f00"})
    fig.suptitle("0036 · 蓝色表示预测导线；加深输入恢复了大部分线段，端点仍未连通", fontsize=17)
    fig.savefig(OUT/"comparison_0036_detail.png", facecolor="white")
    plt.close(fig)
    lines = ["# 验证结论", "", "当前 U-Net 对浅灰细线敏感；整图统一加深并不是可采用的修复。", "",
             "26 张去重验证页，固定权重与阈值。草稿标签未人工认证；这些页不是独立测试集。", "",
             "| 设置 | 导线召回率 | 导线 Dice | 连通参考路径 |", "|---|---:|---:|---:|"]
    for r in result["groups"]["all"]:
        lines.append(f"| γ={r['gamma']} | {r['wire_micro']['recall']:.2%} | {r['wire_micro']['dice']:.2%} | {r['connected_paths']}/{r['paths']} |")
    lines += ["", "0036 的浅灰线大部分恢复，但 γ=2.0 时 X1 末端仍有 7 列像素未覆盖，DOUT 靠近 IC 端仍有 4 列未覆盖。",
              "上述缺口按源路径附近纵向 ±2px 内是否存在预测像素测量；X1 和 DOUT 的对应水平参考路径均未通过 2px 端点容差的连通检查。", ""]
    for p in path_changes:
        lines.append(f"γ={p['gamma']}：原来断开的参考路径恢复 {p['recovered_connected_paths']} 条，但原来连通的路径断开 {p['lost_connected_paths']} 条。")
    lines += ["", "0036 所选文字和 IC 上下边框区域没有新增预测：标题额外像素 10→7→2；说明文字、型号文字、上下边框均为 0。",
              "26 页整体相对草稿的额外像素并未增加，主要退化表现为漏线更多。0012 也存在新增远离草稿的像素，因此局部误检变化仍需逐页核查。",
              "0036 像素 Dice 85.16%→89.95%→90.69%；0012 61.46%→56.16%→48.04%。",
              "如把加深图的两个输出头一起使用，引脚坐标 F1 70.17%→69.51%→65.94%；这是坐标匹配，不能代表正式 PinF1。", "",
              "下一步适合针对浅灰细线核对训练覆盖与草稿标签，再做训练增强对照；同步关注元件端点附近的断线。",
              "本次只做验证，正式权重、原图、标签和推理设置保持原样。没有进行训练或将参考路径输入预测。"]
    (OUT/"conclusion.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps(inspection, ensure_ascii=False))


if __name__ == "__main__":
    main()
