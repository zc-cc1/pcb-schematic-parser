"""Render spatial evidence for missing-path recovery without modifying labels."""
from collections import defaultdict
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/unet_path_recovery_v4/review"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = json.loads((ROOT / "runs/unet_missing_path_audit_20261015/audit.json").read_text(encoding="utf-8"))
    details = json.loads((ROOT / "runs/unet_missing_path_audit_20261015/sparse_source_details.json").read_text(encoding="utf-8"))
    base = {r["case_id"]: r for r in map(json.loads, (ROOT / "data/segmentation_unet_v3_local_path_repairs/manifest.jsonl").read_text(encoding="utf-8").splitlines())}
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"], "axes.unicode_minus": False})
    groups = defaultdict(list)
    for record in details:
        groups[record["case_id"]].append(record)
    for case, records in groups.items():
        row = base[case]
        rgb = np.asarray(Image.open(row["image"]).convert("RGB"))
        n = len(records)
        fig, axes = plt.subplots(n, 2, figsize=(13, max(4, n * 4)), dpi=150, layout="constrained", squeeze=False)
        for axs, record in zip(axes, records):
            coords = np.asarray(record["edge"]["coordinates"], dtype=int)
            x0, y0 = np.maximum(coords.min(axis=0) - 65, 0)
            x1, y1 = np.minimum(coords.max(axis=0) + 66, [row["width"], row["height"]])
            extent = [x0 - .5, x1 - .5, y1 - .5, y0 - .5]
            for ax in axs:
                ax.imshow(rgb[y0:y1, x0:x1], extent=extent, interpolation="nearest")
                ax.set_xlim(x0 - .5, x1 - .5)
                ax.set_ylim(y1 - .5, y0 - .5)
                ax.set_xlabel("原图 x")
                ax.set_ylabel("原图 y")
            axs[0].set_title(f"{case}/{record['edge_id']} 原图")
            axs[1].scatter(coords[:, 0], coords[:, 1], s=65, facecolors="none", edgecolors="#ee6c23", linewidths=1.5)
            for name, (x, y) in zip("AB", coords):
                axs[1].text(x + 5, y - 5, name, color="#b93517", fontsize=12, weight="bold")
            axs[1].set_title("源坐标 A/B（仅定位，尚未补线）")
        fig.suptitle(f"{case} 稀疏坐标记录逐条核查")
        fig.savefig(OUT / f"sparse_{case}.png", facecolor="white")
        plt.close(fig)

    # Sample continuous traces in a contact sheet; draw source-coordinate outlines.
    continuous = [r for r in data["records"] if r["category"] in {"straight_ink_supported", "ordered_nonstraight"}]
    selected = []
    used = set()
    for r in sorted(continuous, key=lambda r: r["valid_coordinate_count"], reverse=True):
        if r["case_id"] not in used:
            selected.append(r); used.add(r["case_id"])
        if len(selected) == 20:
            break
    fig, axes = plt.subplots(5, 4, figsize=(16, 17), dpi=130, layout="constrained")
    from sys import path as import_path
    import_path.insert(0, str(ROOT / "src"))
    from pcb_parser.dataset_converter import load_source_annotation
    for ax, r in zip(axes.flat, selected):
        rgb = np.asarray(Image.open(r["image"]).convert("RGB"))
        annotation, _ = load_source_annotation(r["source"])
        coords = np.asarray(annotation["edges"][r["edge_id"]]["coordinates"], dtype=int)
        x0, y0 = np.maximum(coords.min(axis=0) - 15, 0)
        x1, y1 = np.minimum(coords.max(axis=0) + 16, [rgb.shape[1], rgb.shape[0]])
        ax.imshow(rgb[y0:y1, x0:x1], extent=[x0-.5, x1-.5, y1-.5, y0-.5], interpolation="nearest")
        ax.plot(coords[:, 0], coords[:, 1], color="#ed6621", linewidth=.7, alpha=.65)
        ax.set_title(f"{r['case_id']}/{r['edge_id']} · {len(coords)}点", fontsize=11)
        ax.axis("off")
    fig.suptitle("连续源坐标检查 · 20页最长缺失路径样本 · 橙线为已有坐标")
    fig.savefig(OUT / "continuous20.png", facecolor="white")
    plt.close(fig)
    print(json.dumps(dict(sparse_pages=list(groups), continuous_samples=len(selected), output=str(OUT)), ensure_ascii=False))


if __name__ == "__main__":
    main()
