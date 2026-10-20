"""Given the pins we actually export, what is the best connectivity we could reach?

Builds an oracle prediction: every reference net is replaced by the intersection of its
vertex set with the vertex set we can form from our own components + pins.  Scoring that
with the platform protocol gives the ceiling that no net-grouping improvement can beat
while the pin set stays as it is.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, r"C:/Users/Pc/Desktop/新建文件夹/pcb_parser_competition/scripts")
from platform_metrics import _members, _rails, _vertex_key, score_connectivity  # noqa: E402

GT = Path(r"C:/Users/Pc/Desktop/新建文件夹/赛题六公开数据集/赛题六公开数据集/open_GT/open_GT")
PRED = Path(r"D:/pcb_open_gt_v32best/final_results")


def our_vertices(pred: dict) -> set:
    components = pred.get("components") or {}
    rail = _rails(components)
    out = set()
    for net in (pred.get("nets") or {}).values():
        for token in _members(net):
            out.add(_vertex_key(token, components, rail))
    return out


def truth_vertices(truth: dict) -> set:
    components = truth.get("components") or {}
    rail = _rails(components)
    out = set()
    for net in (truth.get("nets") or {}).values():
        for token in _members(net):
            out.add(_vertex_key(token, components, rail))
    return out


total_gt = total_ours = total_shared = 0
rows = []
for folder in sorted(GT.iterdir(), key=lambda p: int(p.name)):
    case = folder.name
    truth = json.loads(next(folder.glob("*.json")).read_text(encoding="utf-8"))
    path = PRED / f"{case}_Pred.json"
    pred = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {"components": {}, "pins": {}, "nets": {}}

    truth_components = truth.get("components") or {}
    truth_rail = _rails(truth_components)
    mine = our_vertices(pred)

    # oracle: one predicted net per truth net, holding exactly the vertices we can name
    oracle_nets = {}
    for net_id, net in (truth.get("nets") or {}).items():
        vertices = {_vertex_key(t, truth_components, truth_rail) for t in _members(net)}
        keep = vertices & mine
        oracle_nets[net_id] = {"hyperGraph": "(" + ",".join(f"{c}.{p}" if p else c for c, p in sorted(keep)) + ")"}
    oracle = {"components": truth_components, "pins": {}, "nets": oracle_nets}

    base = score_connectivity(pred, truth)
    ceil = score_connectivity(oracle, truth)
    gt_v = truth_vertices(truth)
    total_gt += len(gt_v); total_ours += len(mine); total_shared += len(gt_v & mine)
    rows.append((case, len(gt_v), len(mine), len(gt_v & mine), base["f1"], ceil["f1"]))

print(f"{'case':>4}{'GT顶点':>8}{'我们能命名':>11}{'交集':>6}{'现状连通':>10}{'引脚不变的上限':>16}")
for case, g, o, s, b, c in rows:
    print(f"{case:>4}{g:>8}{o:>11}{s:>6}{b:>10.3f}{c:>16.3f}")
print(f"\n合计 GT 顶点 {total_gt}，我们能命名 {total_ours}，交集 {total_shared} "
      f"= 顶点召回 {total_shared/total_gt:.1%}")
