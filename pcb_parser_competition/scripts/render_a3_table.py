"""A3 experiment log writer: renders the comparison table into markdown.

Kept as a script (not hand-written text) so the numbers always come from the JSON
produced by compare_unet_heads.py, and so ticking a new run only needs one entry.
"""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison", type=Path, default=Path(r"D:/pcb_unet_v2/reports/a3_comparison.json"))
    parser.add_argument("--history-root", type=Path, default=Path(r"D:/pcb_unet_v2/runs"))
    parser.add_argument("--output", type=Path, default=Path(r"D:/pcb_unet_v2/reports/a3_table.md"))
    args = parser.parse_args()

    data = json.loads(args.comparison.read_text(encoding="utf-8"))
    lines = ["| 实验 | 权重 | 引脚目标 | pin 权重 | 轮数 | 导线 F1 | 引脚 F1 (k5/t0.80) | 引脚 F1 (k9/t0.90) |",
             "|---|---|---|---:|---:|---:|---:|---:|"]
    meta = {
        "A2 (v2 data, pin w0.5)": ("A2_full60", "gaussian", "0.5", "60"),
        "A3-pin_l1 (pin w1.0)": ("A3_pin_l1", "gaussian", "1.0", "40"),
        "A3-disk (disk target, w1.0)": ("A3_pin_disk", "5px disk", "1.0", "40"),
        "A3-split (split decoder)": ("A3_pin_split", "gaussian", "1.0", "40 (split)"),
        "A3-split+w2 (split, pin w2.0)": ("A3_split_w2", "gaussian", "2.0", "40 (split)"),
    }
    for label, entry in data["models"].items():
        run, target, weight, epochs = meta.get(label, ("?", "?", "?", "?"))
        lines.append(f"| {label} | {run} | {target} | {weight} | {epochs} | {entry['wire']['f1']:.4f} | "
                     f"{entry['pin']['k5_t0.80']['f1']:.4f} | {entry['pin']['k9_t0.90']['f1']:.4f} |")
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
