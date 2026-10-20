"""What probability does the wire head give along the gap between two of our fragments?

Low (<0.35) means the model is not confident there, so a local threshold could join them.
High (>=0.35) means the mask already covers it and the graph is what fails to connect.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, r"C:/Users/Pc/Desktop/新建文件夹/pcb_parser_competition/scripts")
sys.path.insert(0, r"D:/pcb5_submission/src")
from platform_metrics import _members, _rails, _vertex_key  # noqa: E402
from pcb_parser.unet_checkpoint import load_wire_pin_checkpoint  # noqa: E402
from pcb_parser.unet_inference import predict_configured  # noqa: E402

GT = Path(r"C:/Users/Pc/Desktop/新建文件夹/赛题六公开数据集/赛题六公开数据集/open_GT/open_GT")
IMAGES = Path(r"D:/pcb_open_gt_in")
WEIGHTS = Path(r"D:/pcb5_submission/weights/unet_wire_pin_v1.pt")
INFERENCE = json.loads(Path(r"D:/pcb5_submission/configs/unet_inference_v2.json").read_text(encoding="utf-8"))
THRESHOLD = float(INFERENCE.get("wire_threshold", 0.35))


def ends(net):
    out = []
    for edge in (net.get("edges") or {}).values():
        if len(edge) >= 2:
            out.append((float(edge[0]["x"]), float(edge[0]["y"])))
            out.append((float(edge[-1]["x"]), float(edge[-1]["y"])))
    return out


def vertices_of(net, components, rail):
    keys = set()
    for token in _members(net):
        component, port = _vertex_key(token, components, rail)
        keys.add(f"{component}.{port}" if port else component)
    return keys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, default=Path(r"D:/pcb_open_gt_v32best/final_results"))
    parser.add_argument("--case", default="13")
    parser.add_argument("--net", default="net_7")
    parser.add_argument("--limit", type=float, default=80.0)
    args = parser.parse_args()

    truth = json.loads((GT / args.case / f"{args.case}.json").read_text(encoding="utf-8"))
    components = truth.get("components") or {}
    rail = _rails(components)
    pred = json.loads((args.predictions / f"{args.case}_Pred.json").read_text(encoding="utf-8"))
    wanted = vertices_of((truth.get("nets") or {})[args.net], components, rail)

    p_components = pred.get("components") or {}
    p_rail = _rails(p_components)
    pieces = []
    for net in (pred.get("nets") or {}).values():
        common = vertices_of(net, p_components, p_rail) & wanted
        if common:
            pieces.append((sorted(common), ends(net)))
    pieces.sort(key=lambda item: -len(item[1]))

    model, checkpoint, info = load_wire_pin_checkpoint(WEIGHTS, "cuda:0")
    with Image.open(IMAGES / f"{args.case}.png") as handle:
        rgb = np.array(handle.convert("RGB"))
        height, width = rgb.shape[:2]
    probability = np.asarray(predict_configured(model, rgb, "cuda:0", checkpoint["config"], INFERENCE)[0])
    print(f"checkpoint {info['sha256'][:12]}  阈值 {THRESHOLD}  概率图 {probability.shape}")

    shown = 0
    for i in range(len(pieces)):
        for j in range(i + 1, len(pieces)):
            a, b = pieces[i][1], pieces[j][1]
            if not a or not b:
                continue
            distance, p, q = min((math.dist(u, v), u, v) for u in a for v in b)
            if distance > args.limit:
                continue
            steps = max(3, int(distance) + 1)
            samples = []
            for t in np.linspace(0, 1, steps):
                x = p[0] + t * (q[0] - p[0])
                y = p[1] + t * (q[1] - p[1])
                ty = height - y                      # target coords are bottom-left
                for offset in (-2, 0, 2):            # sample across the line width
                    xx = int(round(x + offset * 0))
                    yy = int(round(ty + offset))
                    if 0 <= yy < height and 0 <= xx < width:
                        samples.append(float(probability[yy, xx]))
            if not samples:
                continue
            values = np.asarray(samples)
            print(f"  {pieces[i][0][:2]} <-> {pieces[j][0][:2]}  相距 {distance:.0f}px  "
                  f"概率 min {values.min():.2f} 均值 {values.mean():.2f} 中位 {np.median(values):.2f}  "
                  f"≥{THRESHOLD} 的点 {100*(values>=THRESHOLD).mean():.0f}%  ≥0.20 的点 {100*(values>=0.20).mean():.0f}%")
            shown += 1
            if shown >= 8:
                return


if __name__ == "__main__":
    main()
