"""A/B the export-time type vocabulary (adapter.refined_type) against GT.

Runs on the pre-`to_target` payloads the pipeline writes to `component_results/`,
so no YOLO/OCR is needed. Scores with the official component rule (key + type +
Name + value and every bbox corner within 20 px; power matched by geometry).
"""
import argparse
import json
import math
import sys
from copy import deepcopy
from pathlib import Path

BASE = Path(r"D:\pcb5_submission")
sys.path.insert(0, str(BASE))
import adapter  # noqa: E402

GT_ROOT = Path(r"C:\Users\Pc\Desktop\新建文件夹\赛题六公开数据集\赛题六公开数据集\200_train_cases")
TOL = 20.0
POWER = {"gnd", "v"}
IC_ALIASES = {"box", "ic"}

SETS = {
    "golden10": ("D:/pcb_v6_check/golden_out/component_results",
                 ["0025", "0030", "0036", "0037", "0042", "0059", "0065", "0070", "0099", "0100"]),
    "big6": ("D:/pcb_v6_check/out_v6_full/component_results",
             ["0001", "0002", "0018", "0029", "0080", "0102"]),
}


def norm_type(t):
    t = str(t or "").strip().lower()
    if t in {"ground", "gnd"}:
        return "gnd"
    if t in {"v", "vcc", "vdd"}:
        return "v"
    if t in IC_ALIASES:
        return "box"
    return t


def legacy_to_target(payload):
    """The behaviour before `refined_type`: type map only, no splits."""
    result = deepcopy(payload)
    for cid, component in result["components"].items():
        component["type"] = adapter.TARGET_TYPE_MAP.get(component["type"], component["type"])
        typ = component["type"]
        if typ in adapter.DERIVED_NAME_TYPES:
            component["Name"] = adapter.derived_name(cid, result["components"])
        if typ in adapter.TEMPLATE_BBOX_DIMS:
            x1, y1, x2, y2 = component["bbox"]
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            short, long = adapter.TEMPLATE_BBOX_DIMS[typ]
            width, height = (long, short) if x2 - x1 > y2 - y1 else (short, long)
            component["bbox"] = [cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2]
    return result


def centre(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def match_power(pred_items, gt_items):
    used, pairs = set(), []
    for pk, pbox in pred_items:
        best = None
        for gk, gbox in gt_items:
            if gk in used:
                continue
            d = math.dist(centre(pbox), centre(gbox))
            if d <= TOL and (best is None or d < best[0]):
                best = (d, gk)
        if best:
            used.add(best[1])
            pairs.append((pk, best[1]))
    return pairs


def score(target, gt):
    tp = fp = fn = 0
    gt_power = [(k, v["bbox"]) for k, v in gt["components"].items() if norm_type(v.get("type")) in POWER]
    gt_ord = {k: v for k, v in gt["components"].items() if norm_type(v.get("type")) not in POWER}
    pr_power = [(k, v["bbox"]) for k, v in target["components"].items() if norm_type(v.get("type")) in POWER]
    pairs = match_power(pr_power, gt_power)
    tp += len(pairs)
    fp += len(pr_power) - len(pairs)
    fn += len(gt_power) - len(pairs)
    detail = []
    for key, comp in target["components"].items():
        if norm_type(comp.get("type")) in POWER:
            continue
        g = gt_ord.get(key)
        if not g:
            fp += 1
            continue
        ok = (norm_type(g.get("type")) == norm_type(comp.get("type"))
              and (comp.get("Name") or None) == (g.get("Name") or None)
              and (comp.get("value") or None) == (g.get("value") or None)
              and all(abs(a - b) <= TOL for a, b in zip(comp["bbox"], g["bbox"])))
        if ok:
            tp += 1
        else:
            fp += 1
            detail.append((key, comp.get("type"), g.get("type")))
    for key, g in gt_ord.items():
        if key not in target["components"]:
            fn += 1
        elif not (norm_type(gt_ord[key].get("type")) == norm_type(target["components"][key].get("type"))
                  and (target["components"][key].get("Name") or None) == (g.get("Name") or None)
                  and (target["components"][key].get("value") or None) == (g.get("value") or None)
                  and all(abs(a - b) <= TOL for a, b in zip(target["components"][key]["bbox"], g["bbox"]))):
            fn += 1
    return tp, fp, fn, detail


def run(set_name):
    folder, cases = SETS[set_name]
    folder = Path(folder)
    tot = {"old": [0, 0, 0], "new": [0, 0, 0]}
    changes = []
    for case in cases:
        path = folder / f"{case}_Pred.json"
        if not path.exists():
            print(f"  (skip {case}: no component_results)")
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        gt = json.loads(next((GT_ROOT / case).glob("*_target_named.json")).read_text(encoding="utf-8"))
        old = legacy_to_target(payload)
        new = adapter.to_target(payload)
        o = score(old, gt)
        n = score(new, gt)
        for index, key in enumerate(("tp", "fp", "fn")):
            tot["old"][index] += o[index]
            tot["new"][index] += n[index]
        if o[0] != n[0]:
            print(f"  {case}: TP {o[0]} -> {n[0]}")
        for key, new_type, gt_type in n[3]:
            old_type = old["components"][key]["type"]
            if old_type != new_type:
                changes.append((case, key, old_type, new_type, gt_type))
    print(f"\n== {set_name} ==")
    for label in ("old", "new"):
        tp, fp, fn = tot[label]
        p, r = tp / max(1, tp + fp), tp / max(1, tp + fn)
        f1 = 2 * p * r / max(1e-9, p + r)
        print(f"  {label}: TP {tp} FP {fp} FN {fn}  P {p:.4f} R {r:.4f} F1 {f1:.4f}")
    print(f"  type changes on mismatching components: {len(changes)}")
    for row in changes[:20]:
        print(f"    {row[0]}/{row[1]}: {row[2]} -> {row[3]} (gt {row[4]})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="all", choices=["all", *SETS])
    args = ap.parse_args()
    for name in (SETS if args.set == "all" else [args.set]):
        run(name)
