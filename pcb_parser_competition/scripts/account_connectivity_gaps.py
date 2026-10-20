"""Where does the connectivity score lose points?

The metric turns every net into a set of vertices (IC: (key, port); other parts: (key, None);
power symbols: (rail, port)), matches pred/GT nets one-to-one by 2|inter|/(|P|+|T|), then
counts vertices: tp = shared, fp = extra in the prediction, fn = reference vertices nobody
covered.

So the losses come from exactly two places, and this separates them:

  * the vertex does not exist in the prediction at all   (component key / pin number missing)
  * the vertex exists but sits in the wrong net          (merge/split)

    python scripts/account_connectivity_gaps.py --predictions D:/pcb_open_gt_v20/final_results
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

GT_ROOT = Path(r"C:/Users/Pc/Desktop/新建文件夹/赛题六公开数据集/赛题六公开数据集/open_GT/open_GT")
IC_TYPES = {"box", "ic"}


def ntype(value):
    text = str(value or "").strip().lower()
    return "box" if text in IC_TYPES else text


def is_ic(value):
    return str(value or "").strip().lower() in IC_TYPES


def members(network):
    body = str(network.get("hyperGraph") or "").strip()
    if body.startswith("(") and body.endswith(")"):
        body = body[1:-1]
    return [token.strip() for token in body.split(",")
            if token.strip() and token.strip() not in {"∅", "null", "None"}]


def rails(components):
    def rail(component, table):
        item = table.get(component) or {}
        kind = ntype(item.get("type"))
        if kind == "gnd":
            return "GND"
        if kind in ("v", "vcc", "vdd"):
            return str(item.get("value") or "").strip() or "VCC"
        return None
    return rail


def vertex(token, components, rail):
    if "." in token:
        component, port = token.rsplit(".", 1)
    else:
        component, port = token, None
    kind = ntype((components.get(component) or {}).get("type"))
    rail_kind = rail(component, components)
    if rail_kind is not None:
        return (rail_kind, port)
    if is_ic(kind):
        return (component, port)
    return (component, None)


def nets_of(document):
    components = document.get("components") or {}
    rail = rails(components)
    out = []
    for net in (document.get("nets") or {}).values():
        vertices = {vertex(token, components, rail) for token in members(net)}
        if vertices:
            out.append(vertices)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--gt-root", default=str(GT_ROOT))
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    gt_root = Path(args.gt_root)
    predictions = Path(args.predictions)

    totals = Counter()
    per_case = []
    why = Counter()
    for folder in sorted((p for p in gt_root.iterdir() if p.is_dir()), key=lambda p: int(p.name)):
        case = folder.name
        truth = json.loads(next(folder.glob("*.json")).read_text(encoding="utf-8"))
        pred = json.loads((predictions / f"{case}_Pred.json").read_text(encoding="utf-8"))
        truth_nets, pred_nets = nets_of(truth), nets_of(pred)
        if not truth_nets or not pred_nets:
            per_case.append((case, len(truth_nets), len(pred_nets), 0, 0, 0, 0.0))
            continue
        score = np.zeros((len(pred_nets), len(truth_nets)))
        for i, p in enumerate(pred_nets):
            for j, t in enumerate(truth_nets):
                inter = len(p & t)
                if inter:
                    score[i, j] = 2 * inter / (len(p) + len(t))
        rows, cols = linear_sum_assignment(-score)
        tp = fp = fn = 0
        matched_pred, matched_truth = set(), set()
        for i, j in zip(rows, cols):
            if score[i, j] <= 0:
                continue
            inter = len(pred_nets[i] & truth_nets[j])
            tp += inter
            fp += len(pred_nets[i]) - inter
            fn += len(truth_nets[j]) - inter
            matched_pred.add(i)
            matched_truth.add(j)
        for i in range(len(pred_nets)):
            if i not in matched_pred:
                fp += len(pred_nets[i])
        for j in range(len(truth_nets)):
            if j not in matched_truth:
                fn += len(truth_nets[j])

        # why: do the reference vertices exist anywhere in the prediction?
        pred_all = set().union(*pred_nets)
        gt_all = set().union(*truth_nets)
        pred_components = pred.get("components") or {}
        pred_pins = pred.get("pins") or {}
        for item in gt_all - pred_all:
            if item[0] in ("GND",) or (item[1] is None and item[0] in pred_components):
                why["电源/两脚顶点身份不符"] += 1
            elif item[0] not in pred_components:
                why["元件 key 缺失"] += 1
            elif item[1] is not None and f"pin_{item[1]}" not in (pred_pins.get(item[0]) or {}):
                why["引脚没导出"] += 1
            else:
                why["顶点存在但没进任何网络"] += 1
        for item in pred_all - gt_all:
            why["多出来的顶点(预测侧)"] += 1

        totals["tp"] += tp
        totals["fp"] += fp
        totals["fn"] += fn
        totals["gt_nets"] += len(truth_nets)
        totals["pred_nets"] += len(pred_nets)
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        per_case.append((case, len(truth_nets), len(pred_nets), tp, fp, fn, f1))

    print(f"{'case':>4}{'gt nets':>9}{'pred nets':>10}{'tp':>6}{'fp':>6}{'fn':>6}{'F1':>8}")
    for row in per_case:
        print(f"{row[0]:>4}{row[1]:>9}{row[2]:>10}{row[3]:>6}{row[4]:>6}{row[5]:>6}{row[6]:>8.3f}")
    tp, fp, fn = totals["tp"], totals["fp"], totals["fn"]
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    print(f"\ntotal tp {tp} fp {fp} fn {fn}  precision {p:.3f} recall {r:.3f} "
          f"F1 {2*p*r/(p+r) if p+r else 0:.4f}")
    print("vertex losses:", dict(why))


if __name__ == "__main__":
    main()
