"""For one badly fragmented reference net, what separates our pieces?

If the gap between two of our nets is bridged by a component body, the mask guard cut a
wire that passes behind the symbol.  If the gap is plain background, the wire was never
drawn there.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, r"C:/Users/Pc/Desktop/新建文件夹/pcb_parser_competition/scripts")
from platform_metrics import _members, _rails, _vertex_key  # noqa: E402

GT = Path(r"C:/Users/Pc/Desktop/新建文件夹/赛题六公开数据集/赛题六公开数据集/open_GT/open_GT")


def net_polyline_ends(net):
    points = []
    for edge in (net.get("edges") or {}).values():
        if len(edge) >= 2:
            points.append((float(edge[0]["x"]), float(edge[0]["y"])))
            points.append((float(edge[-1]["x"]), float(edge[-1]["y"])))
    return points


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

    members = _members((truth.get("nets") or {})[args.net])
    wanted = set()
    for token in members:
        component, port = _vertex_key(token, components, rail)
        wanted.add(f"{component}.{port}" if port else component)
    print(f"{args.net} 的成员（{len(members)} 个 token → {len(wanted)} 个连通身份）：{sorted(wanted)}")

    p_components = pred.get("components") or {}
    p_rail = _rails(p_components)
    hits = []
    for net in (pred.get("nets") or {}).values():
        vertices = set()
        for token in _members(net):
            component, port = _vertex_key(token, p_components, p_rail)
            vertices.add(f"{component}.{port}" if port else component)
        overlap = vertices & wanted
        if overlap:
            hits.append((net, overlap, net_polyline_ends(net)))
    print(f"我们用了 {len(hits)} 个网覆盖它；两两最小端点距离：")
    for i in range(len(hits)):
        for j in range(i + 1, len(hits)):
            a, b = hits[i][2], hits[j][2]
            if not a or not b:
                continue
            best = min(math.dist(p, q) for p in a for q in b)
            if best <= 60:
                print(f"   {sorted(hits[i][1])[:2]} <-> {sorted(hits[j][1])[:2]}   端点相距 {best:.0f}px")

    print("\n真值里这些成员分别属于哪些元件（看它们之间是否隔着元件本体）：")
    for token in members:
        component, port = _vertex_key(token, components, rail)
        item = components.get(token.split(".")[0]) or {}
        box = item.get("bbox")
        print(f"   {token:<10} type={item.get('type')!s:<12} bbox={[round(v) for v in box] if box else None}")


if __name__ == "__main__":
    main()
