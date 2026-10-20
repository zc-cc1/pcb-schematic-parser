"""What is actually inside the gap between two fragments of one reference net?

Samples the page image along the shortest line between the two closest endpoints and
reports, for each pixel: is it ink (dark / coloured) and is it inside a component body
(which our body guard masks out)?
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
from platform_metrics import _members, _rails, _vertex_key  # noqa: E402

GT = Path(r"C:/Users/Pc/Desktop/新建文件夹/赛题六公开数据集/赛题六公开数据集/open_GT/open_GT")
IMAGES = Path(r"D:/pcb_open_gt_in")


def ends(net):
    out = []
    for edge in (net.get("edges") or {}).values():
        if len(edge) >= 2:
            out.append((float(edge[0]["x"]), float(edge[0]["y"])))
            out.append((float(edge[-1]["x"]), float(edge[-1]["y"])))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, default=Path(r"D:/pcb_open_gt_v32best/final_results"))
    parser.add_argument("--case", default="13")
    parser.add_argument("--net", default="net_7")
    args = parser.parse_args()

    truth = json.loads((GT / args.case / f"{args.case}.json").read_text(encoding="utf-8"))
    components = truth.get("components") or {}
    rail = _rails(components)
    pred = json.loads((args.predictions / f"{args.case}_Pred.json").read_text(encoding="utf-8"))
    with Image.open(IMAGES / f"{args.case}.png") as handle:
        rgb = np.array(handle.convert("RGB"))
        height = rgb.shape[0]

    wanted = set()
    for token in _members((truth.get("nets") or {})[args.net]):
        component, port = _vertex_key(token, components, rail)
        wanted.add(f"{component}.{port}" if port else component)

    boxes = [(name, item.get("bbox")) for name, item in components.items() if item.get("bbox")]
    def inside_component(x, y):
        # GT boxes are bottom-left; image sampling is top-left
        ty = height - y
        for name, box in boxes:
            if box[0] <= x <= box[2] and box[1] <= ty <= box[3]:
                return name
        return None

    pieces = []
    p_components = pred.get("components") or {}
    p_rail = _rails(p_components)
    for net in (pred.get("nets") or {}).values():
        vertices = set()
        for token in _members(net):
            component, port = _vertex_key(token, p_components, p_rail)
            vertices.add(f"{component}.{port}" if port else component)
        common = vertices & wanted
        if common:
            pieces.append((sorted(common), ends(net)))
    pieces.sort(key=lambda item: -len(item[1]))

    print(f"{args.case} {args.net}: 我们有 {len(pieces)} 个碎片覆盖它")
    checked = 0
    for i in range(len(pieces)):
        for j in range(i + 1, len(pieces)):
            a, b = pieces[i][1], pieces[j][1]
            if not a or not b:
                continue
            best = min(((math.dist(p, q), p, q) for p in a for q in b))
            distance, p, q = best
            if distance > 80:
                continue
            steps = max(3, int(distance) + 1)
            ink = masked = 0
            names = set()
            for t in np.linspace(0, 1, steps):
                x = int(round(p[0] + t * (q[0] - p[0])))
                y = int(round(p[1] + t * (q[1] - p[1])))
                ty = height - y
                if not (0 <= ty < rgb.shape[0] and 0 <= x < rgb.shape[1]):
                    continue
                px = rgb[ty, x].astype(int)
                brightness = float(px.mean())
                if brightness < 160 or (px.max() - px.min()) > 60:
                    ink += 1
                who = inside_component(x, y)
                if who:
                    masked += 1
                    names.add(who)
            print(f"  {pieces[i][0][:2]} <-> {pieces[j][0][:2]}  相距 {distance:.0f}px  "
                  f"沿线 {steps} 点：墨迹 {ink} 点、落在元件框内 {masked} 点 {sorted(names)}")
            checked += 1
            if checked >= 6:
                return


if __name__ == "__main__":
    main()
