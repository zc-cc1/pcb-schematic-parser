"""Offline A/B of binder + adapter changes on the 6 cached pages.

Chain: cached detections+OCR -> component_text.bind_components -> adapter.to_target
-> official-ish component score against the public GT (key+type+Name+value and
every bbox corner within 20px, power symbols matched by geometry ignoring key).

Because detections come from the frozen v5 payload, absolute numbers are a lower
bound (the deployed pipeline also runs identity recovery / ic_geometry). Compare
runs with different --flags on the SAME case set.
"""
import argparse
import json
import math
import sys
from copy import deepcopy
from pathlib import Path

BASE = Path(r"D:\pcb5_submission")
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import adapter  # noqa: E402
from pcb_parser import component_text as CT  # noqa: E402
from replay_binder import build_case, load_configs, inverse_vocabulary, BATCH  # noqa: E402

GT_ROOT = Path(r"C:\Users\Pc\Desktop\新建文件夹\赛题六公开数据集\赛题六公开数据集\200_train_cases")
CASES = ["0001", "0002", "0018", "0029", "0080", "0102"]
TOL = 20.0
POWER = {"gnd", "v"}
IC_ALIASES = {"box", "ic"}


def norm_type(t):
    t = str(t or "").strip().lower()
    if t in {"ground", "gnd"}:
        return "gnd"
    if t in {"v", "vcc", "vdd"}:
        return "v"
    if t in IC_ALIASES:
        return "box"
    return t


def corners_close(a, b):
    return all(abs(x - y) <= TOL for x, y in zip(a, b))


def centre(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def match_power(pred_items, gt_items):
    """One-to-one by centre distance <= TOL, same as the official power rule."""
    used, pairs = set(), []
    for pk, pbox, pname in pred_items:
        best = None
        for gk, gbox, gname in gt_items:
            if gk in used:
                continue
            d = math.dist(centre(pbox), centre(gbox))
            if d <= TOL and (best is None or d < best[0]):
                best = (d, gk)
        if best:
            used.add(best[1])
            pairs.append((pk, best[1]))
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flags", default="{}", help="JSON overrides for component_text.binding")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="binding override, VALUE parsed as JSON when possible")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--legacy-name", action="store_true",
                    help="revert to the v5 rule (Name derived only for r/c/l)")
    ap.add_argument("--explain", action="store_true", help="print every ordinary failure")
    args = ap.parse_args()
    if args.legacy_name:
        adapter.DERIVED_NAME_TYPES = {"r", "c", "l"}
    overrides = json.loads(args.flags) if args.flags.strip().startswith("{") else {}
    for item in args.set:
        key, _, value = item.partition("=")
        try:
            overrides[key.strip()] = json.loads(value)
        except json.JSONDecodeError:
            overrides[key.strip()] = value
    if overrides:
        print("binding overrides:", overrides)
    adopted = load_configs()
    inv = inverse_vocabulary(adopted)
    agg = {"tp": 0, "fp": 0, "fn": 0, "ordinary_tp": 0, "ordinary_matched": 0,
           "resolved": 0, "detections": 0, "power_tp": 0, "power_gt": 0}
    for case in CASES:
        gt = json.loads(next((GT_ROOT / case).glob("*_target_named.json")).read_text(encoding="utf-8"))
        detections, texts, width, height, cfg = build_case(case, adopted, inv)
        cfg["component_text"] = deepcopy(cfg.get("component_text") or {})
        cfg["component_text"].setdefault("binding", {})
        cfg["component_text"]["binding"].update(overrides)
        output, records, diag = CT.bind_components(detections, texts, width, height, cfg)
        pred = adapter.to_target({"components": output["components"], "pins": {}, "nets": {}})

        gtc = gt["components"]
        gt_power = [(k, v["bbox"], v.get("Name")) for k, v in gtc.items() if norm_type(v.get("type")) in POWER]
        gt_ord = {k: v for k, v in gtc.items() if norm_type(v.get("type")) not in POWER}
        pr_power = [(k, v["bbox"], v.get("Name")) for k, v in pred["components"].items()
                    if norm_type(v.get("type")) in POWER]
        pwr_pairs = match_power(pr_power, gt_power)
        tp = fp = fn = 0
        tp += len(pwr_pairs)
        fp += len(pr_power) - len(pwr_pairs)
        fn += len(gt_power) - len(pwr_pairs)
        ord_matched = 0
        for key, comp in pred["components"].items():
            if norm_type(comp.get("type")) in POWER:
                continue
            g = gt_ord.get(key)
            if not g:
                fp += 1
                continue
            ord_matched += 1
            ok = (norm_type(g.get("type")) == norm_type(comp.get("type"))
                  and (comp.get("Name") or None) == (g.get("Name") or None)
                  and (comp.get("value") or None) == (g.get("value") or None)
                  and corners_close(comp["bbox"], g["bbox"]))
            if ok:
                tp += 1
            else:
                fp += 1
                if args.explain:
                    bad = []
                    if norm_type(g.get("type")) != norm_type(comp.get("type")):
                        bad.append(f"type {comp.get('type')}!= {g.get('type')}")
                    if (comp.get("Name") or None) != (g.get("Name") or None):
                        bad.append(f"Name {comp.get('Name')!r}!= {g.get('Name')!r}")
                    if (comp.get("value") or None) != (g.get("value") or None):
                        bad.append(f"value {comp.get('value')!r}!= {g.get('value')!r}")
                    if not corners_close(comp["bbox"], g["bbox"]):
                        bad.append("bbox " + str([round(v, 1) for v in comp["bbox"]]) + " vs "
                                   + str([round(v, 1) for v in g["bbox"]]))
                    print(f"   MISMATCH {case}/{key}: " + "; ".join(bad))
        fn += sum(1 for k in gt_ord if k not in pred["components"])
        for k, v in gt_ord.items():
            if k in pred["components"]:
                g = gt_ord[k]
                if not (norm_type(g.get("type")) == norm_type(pred["components"][k].get("type"))
                        and (pred["components"][k].get("Name") or None) == (g.get("Name") or None)
                        and (pred["components"][k].get("value") or None) == (g.get("value") or None)
                        and corners_close(pred["components"][k]["bbox"], g["bbox"])):
                    fn += 1
        agg["tp"] += tp
        agg["fp"] += fp
        agg["fn"] += fn
        agg["ordinary_tp"] += tp - len(pwr_pairs)
        agg["ordinary_matched"] += ord_matched
        agg["resolved"] += diag["counts"]["resolved_designators"]
        agg["detections"] += diag["counts"]["detections"]
        agg["power_tp"] += len(pwr_pairs)
        agg["power_gt"] += len(gt_power)
        if not args.quiet:
            f1 = 2 * tp / max(1, 2 * tp + fp + fn)
            print(f"{case}: TP {tp:3d} FP {fp:3d} FN {fn:3d}  F1 {f1:.3f}  "
                  f"resolved {diag['counts']['resolved_designators']:3d}/{diag['counts']['detections']:3d}  "
                  f"power {len(pwr_pairs)}/{len(gt_power)}")
    tp, fp, fn = agg["tp"], agg["fp"], agg["fn"]
    p, r = tp / max(1, tp + fp), tp / max(1, tp + fn)
    f1 = 2 * p * r / max(1e-9, p + r)
    print(f"\nTOTAL TP {tp} FP {fp} FN {fn}  P {p:.4f} R {r:.4f} F1 {f1:.4f}  "
          f"(component stage = 30% of final => {30*f1:.2f} pts)")
    print(f"      ordinary TP {agg['ordinary_tp']} of {agg['ordinary_matched']} key-matched; "
          f"resolved {agg['resolved']}/{agg['detections']}; power {agg['power_tp']}/{agg['power_gt']}")


if __name__ == "__main__":
    sys.exit(main())
