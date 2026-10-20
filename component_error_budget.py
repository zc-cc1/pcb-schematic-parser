"""Component error budget on the 13 open_GT pages for the current (v17) output.

Replicates scripts/platform_metrics.score_components but records *why* each GT
component fails and what each unmatched prediction is, then prices every fix.
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, r"C:/Users/Pc/Desktop/新建文件夹/pcb_parser_competition/scripts")
from platform_metrics import COMPONENT_TOLERANCE_PX, POWER_KIND, _box_error, f1_block, ntype  # noqa: E402

GT = Path(r"C:/Users/Pc/Desktop/新建文件夹/赛题六公开数据集/赛题六公开数据集/open_GT/open_GT")
PRED = Path(r"D:/pcb_open_gt_v17/final_results")

fn_reasons = Counter()
fp_reasons = Counter()
fn_by_type = Counter()
fp_by_type = Counter()
examples = defaultdict(list)
tp = power_tp = 0
gt_total = pred_total = 0


def kind_of(item):
    raw = ntype(item.get("type"))
    return POWER_KIND.get(raw, raw)


for folder in sorted(GT.iterdir(), key=lambda p: int(p.name)):
    case = folder.name
    truth = json.loads(next(folder.glob("*.json")).read_text(encoding="utf-8")).get("components") or {}
    path = PRED / f"{case}_Pred.json"
    pred = json.loads(path.read_text(encoding="utf-8")).get("components") or {} if path.is_file() else {}
    gt_total += len(truth)
    pred_total += len(pred)

    t_ord, p_ord, t_power, p_power = {}, {}, defaultdict(list), defaultdict(list)
    for key, item in truth.items():
        if ntype(item.get("type")) in POWER_KIND:
            t_power[(POWER_KIND[ntype(item.get("type"))], str(item.get("value")))].append((key, item))
        else:
            t_ord[key] = item
    for key, item in pred.items():
        if ntype(item.get("type")) in POWER_KIND:
            p_power[(POWER_KIND[ntype(item.get("type"))], str(item.get("value")))].append((key, item))
        else:
            p_ord[key] = item

    used_pred = set()
    for key, ref in t_ord.items():
        cand = p_ord.get(key)
        if cand is None:
            fn_reasons["no_prediction_with_this_key"] += 1
            fn_by_type[kind_of(ref)] += 1
            examples["missing"].append((case, key, kind_of(ref)))
            continue
        why = []
        if ntype(cand.get("type")) != ntype(ref.get("type")):
            why.append("type")
        if str(cand.get("Name")) != str(ref.get("Name")):
            why.append("Name")
        if str(cand.get("value")) != str(ref.get("value")):
            why.append("value")
        err = _box_error(cand.get("bbox") or [], ref.get("bbox") or [])
        if err > COMPONENT_TOLERANCE_PX:
            why.append("bbox")
        if why:
            key_ = "+".join(why)
            fn_reasons[key_] += 1
            fn_reasons["_".join(why)] += 1
            fn_by_type[kind_of(ref)] += 1
            examples[key_].append((case, key, kind_of(ref), str(cand.get("Name"))[:18], str(ref.get("Name"))[:18],
                                   str(cand.get("value"))[:10], str(ref.get("value"))[:10], round(err, 1)))
            used_pred.add((key, False))
        else:
            tp += 1
            used_pred.add((key, True))
    # predictions for ordinary keys that failed, and spurious keys
    for key, item in p_ord.items():
        if key not in t_ord:
            fp_reasons["key_not_in_truth"] += 1
            fp_by_type[kind_of(item)] += 1
            examples["spurious"].append((case, key, kind_of(item)))
        elif (key, True) not in used_pred:
            fp_reasons["same_key_but_a_field_wrong"] += 1

    for group, items in t_power.items():
        cands = p_power.get(group) or []
        if not cands:
            for key, ref in items:
                fn_reasons["power_group_absent:" + (group[1] or "NULL")] += 1
                fn_by_type["power:" + (group[1] or "NULL")] += 1
                examples["power_missing"].append((case, key, group[1]))
            continue
        cost = np.full((len(cands), len(items)), 1e6)
        for i, (_pk, pitem) in enumerate(cands):
            for j, (_tk, titem) in enumerate(items):
                err = _box_error(pitem.get("bbox") or [], titem.get("bbox") or [])
                if err <= COMPONENT_TOLERANCE_PX:
                    cost[i, j] = err
        rows, cols = linear_sum_assignment(cost)
        hit_p = {i for i, j in zip(rows, cols) if cost[i, j] < 1e6}
        hit_t = {j for i, j in zip(rows, cols) if cost[i, j] < 1e6}
        power_tp += len(hit_p)
        for i, (pk, pitem) in enumerate(cands):
            if i not in hit_p:
                fp_reasons["power_geometry"] += 1
                fp_by_type["power:" + (group[1] or "NULL")] += 1
        for j, (tk, titem) in enumerate(items):
            if j not in hit_t:
                fn_reasons["power_geometry"] += 1
                fn_by_type["power:" + (group[1] or "NULL")] += 1
                examples["power_geometry"].append((case, tk, group[1]))

total_tp = tp + power_tp
fp = pred_total - total_tp
fn = gt_total - total_tp
f1 = f1_block(total_tp, fp, fn)
print(f"GT {gt_total}  pred {pred_total}  TP {total_tp} (ordinary {tp} + power {power_tp})  FP {fp}  FN {fn}")
print(f"component F1 {f1['f1']:.4f}   (platform reported .8256 for this run)")
print()
print("FN reasons:", dict(fn_reasons.most_common(12)))
print()
print("FP reasons:", dict(fp_reasons.most_common(6)))
print()
print("FN by type:", dict(fn_by_type.most_common(14)))
print("FP by type:", dict(fp_by_type.most_common(14)))
print()
n = len(list(GT.iterdir()))
print("pricing (component points = 0.30 * delta_F1):")
for label, k in (("all FN removed", fn), ("bbox-only FNs", fn_reasons.get("bbox", 0)),
                 ("value-only FNs", fn_reasons.get("value", 0)),
                 ("Name-only FNs", fn_reasons.get("Name", 0)),
                 ("missing-key FNs", fn_reasons["no_prediction_with_this_key"]),
                 ("all FP removed", fp)):
    new_f1 = f1_block(total_tp + (k if "FN" in label else 0),
                      fp - (k if label == "all FP removed" else 0), fn - (k if "FN" in label else 0))["f1"]
    print(f"  {label:<18} +{k:>3} -> component {new_f1:.4f}  ({0.30 * (new_f1 - f1['f1']) * 100:+.2f} 分)")
print()
for name in ("missing", "spurious", "bbox", "value", "Name", "power_missing", "power_geometry"):
    if examples.get(name):
        print(f"-- {name} examples --")
        for e in examples[name][:6]:
            print("   ", e)
