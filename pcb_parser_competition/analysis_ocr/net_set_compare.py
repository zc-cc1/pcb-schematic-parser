"""Compare predicted nets with the released GT by exact member-set identity.

The official connectivity metric merges nets that share a global power rail and then
does a Hungarian soft-F1 over vertex sets. This script deliberately does the simplest
thing instead - exact set equality over `component.pin` tokens - because it is
transparent and needs no assumptions. It is a *lower* bound on the official number
and is used only to compare two runs of our own pipeline against each other.
"""
import argparse
import json
import re
import sys
from pathlib import Path

MEMBER = re.compile(r"\(([^)]*)\)")


def members(net):
    graph = net.get("hyperGraph") or ""
    inside = MEMBER.search(graph)
    if not inside:
        return frozenset()
    raw = [t.strip() for t in inside.group(1).split(",") if t.strip()]
    return frozenset(raw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--gt-root", required=True)
    ap.add_argument("--glob", default="*_target_named.json")
    args = ap.parse_args()
    pred_dir, gt_root = Path(args.pred), Path(args.gt_root)
    tot = {"gt": 0, "pred": 0, "hit": 0, "multi_gt": 0, "multi_pred": 0}
    print(f"{'case':>4} {'gtNets':>6} {'predNets':>8} {'exactHit':>8} {'gtMulti':>7} {'predMulti':>9}")
    for folder in sorted((p for p in gt_root.iterdir() if p.is_dir()), key=lambda p: (len(p.name), p.name)):
        case = folder.name
        gt_file = next(folder.glob(args.glob), None)
        pred_file = pred_dir / f"{case}_Pred.json"
        if gt_file is None or not pred_file.exists():
            continue
        gt = json.loads(gt_file.read_text(encoding="utf-8"))
        pred = json.loads(pred_file.read_text(encoding="utf-8"))
        gt_sets = [members(n) for n in (gt.get("nets") or {}).values()]
        pr_sets = [members(n) for n in (pred.get("nets") or {}).values()]
        gt_sets = [s for s in gt_sets if len(s) >= 2]
        pr_sets = [s for s in pr_sets if len(s) >= 2]
        hit = len(set(gt_sets) & set(pr_sets))
        print(f"{case:>4} {len(gt_sets):6d} {len(pr_sets):8d} {hit:8d} "
              f"{sum(1 for s in gt_sets if len(s) >= 3):7d} {sum(1 for s in pr_sets if len(s) >= 3):9d}")
        tot["gt"] += len(gt_sets)
        tot["pred"] += len(pr_sets)
        tot["hit"] += hit
    print("\n== TOTAL ==")
    print(f"  GT multi-pin nets      {tot['gt']}")
    print(f"  predicted multi-pin    {tot['pred']}")
    print(f"  exact member-set match {tot['hit']}")
    if tot["hit"] or tot["pred"]:
        p = tot["hit"] / max(1, tot["pred"])
        r = tot["hit"] / max(1, tot["gt"])
        print(f"  P {p:.3f}  R {r:.3f}  F1 {2*p*r/max(1e-9,p+r):.4f}  (strict set identity, not the official soft F1)")


if __name__ == "__main__":
    sys.exit(main())
