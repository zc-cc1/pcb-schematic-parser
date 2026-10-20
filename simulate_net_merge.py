"""Would merging nearby nets fix the fragmentation?  Simulate on the exported JSON.

The platform scores connectivity on each net's vertex set, so a post-hoc merge of nets
whose wire endpoints are close can be evaluated without touching the pipeline: union the
two nets' members and edges, then re-score.  This bounds what any "join the pieces"
change could buy.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, r"C:/Users/Pc/Desktop/新建文件夹/pcb_parser_competition/scripts")
from platform_metrics import _members, score_connectivity, score_wire, f1_block  # noqa: E402

GT = Path(r"C:/Users/Pc/Desktop/新建文件夹/赛题六公开数据集/赛题六公开数据集/open_GT/open_GT")


def endpoints(net):
    points = []
    for edge in (net.get("edges") or {}).values():
        if len(edge) >= 2:
            for end in (edge[0], edge[-1]):
                points.append((float(end["x"]), float(end["y"])))
    return points


def merge_nets(nets: dict, limit: float) -> dict:
    ids = list(nets)
    parent = {i: i for i in ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    points = {i: endpoints(nets[i]) for i in ids}
    for a in range(len(ids)):
        for b in range(a + 1, len(ids)):
            pa, pb = points[ids[a]], points[ids[b]]
            if not pa or not pb:
                continue
            best = min(math.dist(p, q) for p in pa for q in pb)
            if best <= limit:
                ra, rb = find(ids[a]), find(ids[b])
                if ra != rb:
                    parent[rb] = ra
    merged = {}
    for i in ids:
        root = find(i)
        target = merged.setdefault(root, {"hyperGraph": "", "edges": {}, "_members": []})
        target["_members"] += _members(nets[i])
        target["edges"].update(nets[i].get("edges") or {})
    for root, net in merged.items():
        net["hyperGraph"] = "(" + ",".join(sorted(set(net["_members"]))) + ")"
        del net["_members"]
    return merged


def score(pred, truth):
    c = score_connectivity(pred, truth)
    w = score_wire(pred, truth)
    return c, w


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, default=Path(r"D:/pcb_open_gt_v32best/final_results"))
    parser.add_argument("--limits", type=float, nargs="+", default=[0.0, 6.0, 12.0, 20.0])
    args = parser.parse_args()

    cases = []
    for folder in sorted(GT.iterdir(), key=lambda p: int(p.name)):
        case = folder.name
        truth = json.loads(next(folder.glob("*.json")).read_text(encoding="utf-8"))
        path = args.predictions / f"{case}_Pred.json"
        pred = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {"components": {}, "pins": {}, "nets": {}}
        cases.append((case, pred, truth))

    for limit in args.limits:
        ct = {"tp": 0, "fp": 0, "fn": 0}
        wt = {"tp": 0, "fp": 0, "fn": 0}
        for case, pred, truth in cases:
            merged = merge_nets(pred.get("nets") or {}, limit)
            clone = {"components": pred.get("components") or {}, "pins": pred.get("pins") or {}, "nets": merged}
            c, w = score(clone, truth)
            for key in ct:
                ct[key] += c[key]
                wt[key] += w[key]
        c = f1_block(ct["tp"], ct["fp"], ct["fn"])
        w = f1_block(wt["tp"], wt["fp"], wt["fn"])
        print(f"合并距离 ≤{limit:>4.0f}px:  连通 F1 {c['f1']:.4f} ({c['f1']*35:.2f}/35, tp {c['tp']})   "
              f"导线 F1 {w['f1']:.4f} ({w['f1']*10:.2f}/10)   两项合计 {c['f1']*35 + w['f1']*10:.2f} 分")


if __name__ == "__main__":
    main()
