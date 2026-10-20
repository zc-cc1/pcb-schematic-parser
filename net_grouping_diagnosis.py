"""Why is the net grouping wrong on the worst connectivity pages?

For each reference net: how many of our nets does it need to be covered (fragmentation),
and how many reference nets does one of our nets span (over-merge)?  Vertices are the
platform's connectivity identities, so ground/supply collapse onto rails.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, r"C:/Users/Pc/Desktop/新建文件夹/pcb_parser_competition/scripts")
from platform_metrics import _members, _rails, _vertex_key  # noqa: E402

GT = Path(r"C:/Users/Pc/Desktop/新建文件夹/赛题六公开数据集/赛题六公开数据集/open_GT/open_GT")


def net_vertices(container, components):
    rail = _rails(components)
    out = []
    for net in (container.get("nets") or {}).values():
        vertices = {_vertex_key(t, components, rail) for t in _members(net)}
        vertices = {(f"{c}.{p}" if p else c) for c, p in vertices}
        out.append(vertices)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, default=Path(r"D:/pcb_open_gt_v32best/final_results"))
    parser.add_argument("--cases", nargs="+", default=["13", "6", "4", "5", "12"])
    args = parser.parse_args()

    for case in args.cases:
        truth = json.loads((GT / case / f"{case}.json").read_text(encoding="utf-8"))
        path = args.predictions / f"{case}_Pred.json"
        pred = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {"components": {}, "nets": {}}
        gt_nets = net_vertices(truth, truth.get("components") or {})
        our_nets = net_vertices(pred, pred.get("components") or {})
        gt_all = set().union(*gt_nets) if gt_nets else set()
        our_all = set().union(*our_nets) if our_nets else set()

        fragmented = 0
        covered = 0
        detail = []
        for vertices in gt_nets:
            usable = {v for v in vertices if v in our_all}
            if not usable:
                continue
            pieces = [i for i, net in enumerate(our_nets) if net & usable]
            covered += 1
            if len(pieces) > 1:
                fragmented += 1
            detail.append((len(vertices), len(usable), len(pieces)))

        merged = 0
        for net in our_nets:
            hits = sum(1 for vertices in gt_nets if net & vertices)
            if hits > 1:
                merged += 1

        print(f"=== case {case} ===")
        print(f"  真值网 {len(gt_nets)}  我们网 {len(our_nets)}")
        print(f"  我们的网里含多个真值网成分的（并错）: {merged}")
        print(f"  真值网被我们拆成多块才拼齐的（切碎）: {fragmented}"
              f" / 可覆盖的 {covered}")
        print(f"  真值顶点 {len(gt_all)}  我们能命名 {len(our_all)}  交集 {len(gt_all & our_all)}")
        worst = sorted(detail, key=lambda d: d[2], reverse=True)[:5]
        print(f"  切得最碎的 5 个真值网（真值顶点数, 我们能命名的, 用到我们几个网）: {worst}")
        print()


if __name__ == "__main__":
    main()
