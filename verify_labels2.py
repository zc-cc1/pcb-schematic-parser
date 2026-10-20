"""Verify: is a fragmented reference net held together by repeated printed labels?"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, r"C:/Users/Pc/Desktop/新建文件夹/pcb_parser_competition/scripts")
from platform_metrics import _members, _rails, _vertex_key  # noqa: E402

GT = Path(r"C:/Users/Pc/Desktop/新建文件夹/赛题六公开数据集/赛题六公开数据集/open_GT/open_GT")


def keys(document):
    components = document.get("components") or {}
    rail = _rails(components)

    def one(token):
        component, port = _vertex_key(token, components, rail)
        return f"{component}.{port}" if port else component

    return components, one


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, default=Path(r"D:/pcb_open_gt_v32best/final_results"))
    parser.add_argument("--cases", nargs="+", default=["13", "6", "4"])
    args = parser.parse_args()

    for case in args.cases:
        truth = json.loads((GT / case / f"{case}.json").read_text(encoding="utf-8"))
        t_components, t_key = keys(truth)
        pred = json.loads((args.predictions / f"{case}_Pred.json").read_text(encoding="utf-8"))
        _pc, p_key = keys(pred)
        our_nets = [{p_key(t) for t in _members(net)} for net in (pred.get("nets") or {}).values()]
        our_all = set().union(*our_nets) if our_nets else set()

        print(f"=== case {case} ===")
        rows = []
        for net_id, net in (truth.get("nets") or {}).items():
            members = _members(net)
            names = {t_key(t) for t in members}
            usable = names & our_all
            if not usable:
                continue
            pieces = sum(1 for mine in our_nets if mine & usable)
            labels = Counter(str((t_components.get(token) or {}).get("value"))
                             for token in members
                             if str((t_components.get(token) or {}).get("type")) == "net_bidirection")
            rows.append((pieces, len(names), dict(labels), net_id))
        rows.sort(reverse=True)
        for pieces, size, labels, net_id in rows[:4]:
            if any(value > 1 for value in labels.values()):
                verdict = "同名标签重复 -> 逻辑合并能救"
            elif labels:
                verdict = f"只有一个标签 {labels} -> 合并救不了"
            else:
                verdict = "内部无标签元件 -> 只能靠连线"
            print(f"  {net_id:<10} 真值顶点 {size:>2}, 我们用了 {pieces:>2} 个网   标签 {labels or '{}'}  -> {verdict}")
        print()


if __name__ == "__main__":
    main()
