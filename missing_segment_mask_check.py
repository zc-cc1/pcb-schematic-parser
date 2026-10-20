"""Do the still-missing reference segments exist in the mask?

Samples the wire probability along every reference segment we do not reproduce, and
reports what share of each segment reaches the decode threshold.  High share means the
mask has it and something later throws it away; low share means the mask never sees it.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, r"C:/Users/Pc/Desktop/新建文件夹/pcb_parser_competition/scripts")
sys.path.insert(0, r"D:/pcb5_submission/src")
from platform_metrics import _segments  # noqa: E402
from pcb_parser.unet_checkpoint import load_wire_pin_checkpoint  # noqa: E402
from pcb_parser.unet_inference import predict_configured  # noqa: E402

GT = Path(r"C:/Users/Pc/Desktop/新建文件夹/赛题六公开数据集/赛题六公开数据集/open_GT/open_GT")
IMAGES = Path(r"D:/pcb_open_gt_in")
PRED = Path(r"D:/pcb_open_gt_v37dots/final_results")
WEIGHTS = Path(r"D:/pcb5_submission/weights/unet_wire_pin_v1.pt")
INFERENCE = json.loads(Path(r"D:/pcb5_submission/configs/unet_inference_v2.json").read_text(encoding="utf-8"))
THRESHOLD = float(INFERENCE.get("wire_threshold", 0.35))


def close(a, b, tol=20.0):
    return ((math.dist(a[0], b[0]) <= tol and math.dist(a[-1], b[-1]) <= tol)
            or (math.dist(a[0], b[-1]) <= tol and math.dist(a[-1], b[0]) <= tol))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", nargs="+", default=["1", "13"])
    args = parser.parse_args()

    model, checkpoint, info = load_wire_pin_checkpoint(WEIGHTS, "cuda:0")
    print(f"checkpoint {info['sha256'][:12]}  阈值 {THRESHOLD}")
    buckets = Counter()

    for case in args.cases:
        truth = json.loads((GT / case / f"{case}.json").read_text(encoding="utf-8"))
        pred = json.loads((PRED / f"{case}_Pred.json").read_text(encoding="utf-8"))
        our = [seg for net in (pred.get("nets") or {}).values() for seg in _segments(net)]
        with Image.open(IMAGES / f"{case}.png") as handle:
            rgb = np.array(handle.convert("RGB"))
            height = rgb.shape[0]
        probability = np.asarray(predict_configured(model, rgb, "cuda:0", checkpoint["config"], INFERENCE)[0])

        shares = []
        for net in (truth.get("nets") or {}).values():
            for seg in _segments(net):
                if any(close(seg, other) for other in our):
                    continue
                steps = max(3, int(math.dist(seg[0], seg[-1])) + 1)
                values = []
                for t in np.linspace(0, 1, steps):
                    x = int(round(seg[0][0] + t * (seg[-1][0] - seg[0][0])))
                    y = int(round(height - (seg[0][1] + t * (seg[-1][1] - seg[0][1]))))
                    if 0 <= y < height and 0 <= x < probability.shape[1]:
                        values.append(float(probability[y, x]))
                if values:
                    shares.append(float((np.asarray(values) >= THRESHOLD).mean()))
        if not shares:
            print(f"  case {case}: 没有缺失段")
            continue
        values = np.asarray(shares)
        print(f"  case {case}: 缺失 {len(values)} 段  "
              f"整段都在阈值上 {int((values >= .9).sum())}  "
              f"大部分在阈值上 {int(((values >= .5) & (values < .9)).sum())}  "
              f"大部分在阈值下 {int(((values > .1) & (values < .5)).sum())}  "
              f"基本没有 {int((values <= .1).sum())}")
        for key, mask in (("mask 里有（≥90% 点过阈值）", values >= .9),
                          ("mask 里有部分（50-90%）", (values >= .5) & (values < .9)),
                          ("mask 里很少（10-50%）", (values > .1) & (values < .5)),
                          ("mask 里没有（≤10%）", values <= .1)):
            buckets[key] += int(mask.sum())

    print("\n合计：")
    for key, value in buckets.items():
        print(f"  {key:<28}{value:>4}")


if __name__ == "__main__":
    main()
