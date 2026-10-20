"""Honest connectivity headroom: how good could grouping be, holding our pins fixed?

For every reference net, build a predicted net holding exactly the vertices we can name
(no invented vertices, no empty nets).  If our wire grouping were perfect, that is what
we would submit - so the score of this synthetic prediction is the ceiling that pin
coverage alone allows.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, r"C:/Users/Pc/Desktop/新建文件夹/pcb_parser_competition/scripts")
from platform_metrics import _members, _rails, _vertex_key, f1_block, score_connectivity  # noqa: E402

GT = Path(r"C:/Users/Pc/Desktop/新建文件夹/赛题六公开数据集/赛题六公开数据集/open_GT/open_GT")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, default=Path(r"D:/pcb_open_gt_v40cur/final_results"))
    args = parser.parse_args()

    total = {"tp": 0, "fp": 0, "fn": 0}
    ceiling = {"tp": 0, "fp": 0, "fn": 0}
    rows = []
    for folder in sorted(GT.iterdir(), key=lambda p: int(p.name)):
        case = folder.name
        truth = json.loads(next(folder.glob("*.json")).read_text(encoding="utf-8"))
        path = args.predictions / f"{case}_Pred.json"
        pred = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {"components": {}, "pins": {}, "nets": {}}

        t_components = truth.get("components") or {}
        t_rail = _rails(t_components)
        p_components = pred.get("components") or {}
        p_rail = _rails(p_components)
        mine = set()
        for net in (pred.get("nets") or {}).values():
            for token in _members(net):
                mine.add(_vertex_key(token, p_components, p_rail))

        nets = {}
        for index, (net_id, net) in enumerate((truth.get("nets") or {}).items(), start=1):
            keep = {_vertex_key(t, t_components, t_rail) for t in _members(net)} & mine
            if not keep:
                continue
            tokens = [f"{c}.{p}" if p else c for c, p in sorted(keep)]
            nets[f"oracle_{index:04d}"] = {"hyperGraph": "(" + ",".join(tokens) + ")"}
        oracle = {"components": t_components, "pins": {}, "nets": nets}

        base = score_connectivity(pred, truth)
        top = score_connectivity(oracle, truth)
        for key in total:
            total[key] += base[key]
            ceiling[key] += top[key]
        rows.append((case, base["f1"], top["f1"]))

    base = f1_block(total["tp"], total["fp"], total["fn"])
    top = f1_block(ceiling["tp"], ceiling["fp"], ceiling["fn"])
    print(f"现状             tp {total['tp']:>3} fp {total['fp']:>3} fn {total['fn']:>3}  "
          f"F1 {base['f1']:.4f} = {base['f1']*35:.2f}/35")
    print(f"分组完美的上限   tp {ceiling['tp']:>3} fp {ceiling['fp']:>3} fn {ceiling['fn']:>3}  "
          f"F1 {top['f1']:.4f} = {top['f1']*35:.2f}/35")
    print(f"分组可捡的分: {(top['f1']-base['f1'])*35:+.2f}")
    print("\n各例（现状 -> 上限）:")
    for case, b, t in sorted(rows, key=lambda r: r[2] - r[1]):
        print(f"   case {case:<3} {b:.3f} -> {t:.3f}   可捡 {(t-b)*35:+.2f} 分")


if __name__ == "__main__":
    main()
