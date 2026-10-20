"""画引脚漏检的两张图：结构图 + 最差元件实例图。

    & $py pin_miss_chart.py --tag v33
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(r"C:/Users/Pc/Desktop/新建文件夹")
GT = ROOT / "赛题六公开数据集/赛题六公开数据集/open_GT/open_GT"
PRED = Path(r"D:/pcb_open_gt_v33angle120/final_results")

for candidate in ("C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf"):
    if Path(candidate).is_file():
        font_manager.fontManager.addfont(candidate)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=candidate).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False

COLORS = {"TP": "#4c9f70", "identity": "#e4b363", "name": "#d97b4f",
          "tip_5_20": "#8ab4d8", "tip_20_40": "#4a7fa5", "miss": "#8c2f39"}
LABELS = {"TP": "TP 计分", "identity": "身份错位", "name": "名称不符",
          "tip_5_20": "尖端偏 5–20px", "tip_20_40": "尖端偏 20–40px", "miss": "真漏检"}
ORDER = ["TP", "identity", "name", "tip_5_20", "tip_20_40", "miss"]


def structure(tag: str):
    data = json.loads((ROOT / f"pin_miss_{tag}.json").read_text(encoding="utf-8"))
    rows = data["per_case"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.6), gridspec_kw={"width_ratios": [2.05, 1]})

    cases = [r["case"] for r in rows]
    left = [0] * len(rows)
    for key in ORDER:
        values = [r["tp"] if key == "TP" else r[key] for r in rows]
        ax1.barh(cases, values, left=left, color=COLORS[key], label=LABELS[key], height=0.68)
        left = [a + b for a, b in zip(left, values)]
    for index, row in enumerate(rows):
        ax1.text(row["gt"] + 0.8, index, f"{row['tp']}/{row['gt']}  F1 {row['f1']:.2f}",
                 va="center", fontsize=8.5, color="#333333")
    ax1.set_xlabel("GT IC 引脚数")
    ax1.set_ylabel("case")
    ax1.invert_yaxis()
    ax1.set_title("每个 case 的引脚去向（v33：计分 140 / 191）", fontsize=11)
    ax1.legend(fontsize=8, ncol=3, loc="lower right")
    ax1.grid(axis="x", alpha=0.25)

    buckets = data["buckets"]
    fp = data["fp"]
    cum = data["tp"]
    fp_total = sum(fp.values())
    base_f1 = data["pin_f1_current"]
    steps, tips, f1s = ["现状"], [cum], [base_f1]
    for key in ("identity", "name", "tip_5_20", "tip_20_40", "miss"):
        cum += buckets.get(key, 0)
        steps.append(f"+{LABELS[key]}")
        tips.append(cum)
        f1s.append(2 * cum / (2 * cum + fp_total + data["gt_ic_pins"] - cum))

    ax2.plot(range(len(steps)), f1s, "o-", color="#2f4858", linewidth=2)
    for index, (value, tip) in enumerate(zip(f1s, tips)):
        ax2.annotate(f"{value:.3f}\nTP {tip}", (index, value), textcoords="offset points",
                     xytext=(0, 10), ha="center", fontsize=8)
    ax2.set_xticks(range(len(steps)))
    ax2.set_xticklabels(steps, rotation=28, ha="right", fontsize=8)
    ax2.set_ylim(0.75, 1.02)
    ax2.set_ylabel("PinF1")
    ax2.set_title("逐类修复后的 PinF1（FP 固定 22）", fontsize=11)
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    out = ROOT / f"pin_miss_structure_{tag}.png"
    fig.savefig(out, dpi=170)
    plt.close(fig)
    print("written:", out)


def example(case: str, component: str, margin: int = 70):
    truth = json.loads((GT / case / f"{case}.json").read_text(encoding="utf-8"))
    pred = json.loads((PRED / f"{case}_Pred.json").read_text(encoding="utf-8"))
    image_path = next((GT / case).glob("*.png"))
    with Image.open(image_path) as handle:
        image = handle.convert("RGB")
    height = image.height
    bbox = truth["components"][component]["bbox"]
    x1 = max(0, int(bbox[0]) - margin)
    x2 = min(image.width, int(bbox[2]) + margin)
    y1 = max(0, int(height - bbox[3]) - margin)
    y2 = min(height, int(height - bbox[1]) + margin)
    crop = image.crop((x1, y1, x2, y2))
    scale = 2.2
    crop = crop.resize((int(crop.width * scale), int(crop.height * scale)), Image.LANCZOS)
    draw = ImageDraw.Draw(crop)
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 15)
    except OSError:
        font = ImageFont.load_default()

    ours = (pred.get("pins") or {}).get(component) or {}
    for key, reference in truth["pins"][component].items():
        gx = (reference["point"]["x"] - x1) * scale
        gy = (height - reference["point"]["y"] - y1) * scale
        draw.ellipse([gx - 6, gy - 6, gx + 6, gy + 6], outline="#1f6feb", width=2)
        number = key.replace("pin_", "")
        candidate = ours.get(key)
        if candidate is None:
            draw.text((gx + 9, gy - 8), number, fill="#c1121f", font=font)
            draw.line([gx - 11, gy - 11, gx + 11, gy + 11], fill="#c1121f", width=2)
            continue
        px = (candidate["point"]["x"] - x1) * scale
        py = (height - candidate["point"]["y"] - y1) * scale
        distance = math.hypot(candidate["point"]["x"] - reference["point"]["x"],
                              candidate["point"]["y"] - reference["point"]["y"])
        color = "#2a9d8f" if distance <= 5 else "#e9a23b"
        draw.line([gx, gy, px, py], fill=color, width=2)
        draw.ellipse([px - 4, py - 4, px + 4, py + 4], fill=color)
        draw.text((gx + 9, gy - 8), f"{number} {distance:.0f}px", fill="#333333", font=font)
    crop.save(ROOT / f"pin_miss_example_{case}_{component}.png")
    print("written:", ROOT / f"pin_miss_example_{case}_{component}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="v33")
    parser.add_argument("--case", default="1")
    parser.add_argument("--component", default="USB1")
    args = parser.parse_args()
    structure(args.tag)
    example(args.case, args.component)
