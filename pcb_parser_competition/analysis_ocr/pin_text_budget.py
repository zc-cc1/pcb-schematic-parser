"""Where is the remaining OCR money? Pin text vs component text on the 6 pages.

Pin TP (official rule): component key + pin_<number> exact + pinname exact +
point within 5px. Splits the loss into OCR-side (pinname / number characters)
and geometry-side (point), so the next work item is chosen by measurement.
"""
import json
import math
import sys
from collections import Counter
from pathlib import Path

V6 = Path(r"D:\pcb_v6_check\out_v6_full\final_results")
V5 = Path(r"D:\pcb_v5_batch\preds")
GT_ROOT = Path(r"C:\Users\Pc\Desktop\新建文件夹\赛题六公开数据集\赛题六公开数据集\200_train_cases")
CASES = ["0001", "0002", "0018", "0029", "0080", "0102"]
PIN_TOL = 5.0
POWER = {"gnd", "v", "vcc", "vdd", "ground", "voltage_source"}


def norm_type(t):
    t = str(t or "").strip().lower()
    if t in {"ground", "gnd"}:
        return "gnd"
    if t in {"v", "vcc", "vdd"}:
        return "v"
    if t in {"box", "ic"}:
        return "box"
    return t


def pt(p):
    return (float(p["x"]), float(p["y"]))


def main():
    total = Counter()
    for case in CASES:
        gt = json.loads(next((GT_ROOT / case).glob("*_target_named.json")).read_text(encoding="utf-8"))
        pred = json.loads((V6 / f"{case}_Pred.json").read_text(encoding="utf-8"))
        gt_pins = {k: v for k, v in (gt.get("pins") or {}).items()}
        pr_pins = {k: v for k, v in (pred.get("pins") or {}).items()}
        n_gt = sum(len(v) for v in gt_pins.values())
        n_pr = sum(len(v) for v in pr_pins.values())
        # pins of components we did not key-match are unreachable by construction
        key_matched = set(gt["components"]) & set(pred["components"])
        gt_pins_reachable = sum(len(v) for k, v in gt_pins.items() if k in key_matched)
        num_ok = name_ok = point_ok = tp = 0
        missing_owner = 0
        wrong_name_examples = []
        for comp, group in gt_pins.items():
            if comp not in key_matched:
                missing_owner += len(group)
                continue
            mine = pr_pins.get(comp) or {}
            for pin_key, pin in group.items():
                number = pin_key[4:]
                cand = mine.get(pin_key) or mine.get(f"pin_{number}")
                if cand is None:
                    continue
                num_ok += 1
                same_name = (cand.get("pinname") or None) == (pin.get("pinname") or None)
                same_point = math.dist(pt(cand["point"]), pt(pin["point"])) <= PIN_TOL
                name_ok += same_name
                point_ok += same_point
                if same_name and same_point:
                    tp += 1
                elif not same_name and len(wrong_name_examples) < 4:
                    wrong_name_examples.append((comp, number, pin.get("pinname"), cand.get("pinname")))
        # how many GT pinnames are even non-empty
        nonempty = sum(1 for g in gt_pins.values() for p in g.values() if p.get("pinname"))
        print(f"\n=== {case}: GT pins {n_gt} (non-empty pinname {nonempty}), pred pins {n_pr}, "
              f"GT pins of key-matched comps {gt_pins_reachable}")
        print(f"    number matched {num_ok}  pinname exact {name_ok}  point<=5px {point_ok}  "
              f"pin TP {tp}   (owner key missing: {missing_owner })")
        for row in wrong_name_examples:
            print(f"      name mismatch {row[0]}.{row[1]}: GT={row[2]!r} PRED={row[3]!r}")
        total["gt"] += n_gt
        total["pred"] += n_pr
        total["reachable"] += gt_pins_reachable
        total["num"] += num_ok
        total["name"] += name_ok
        total["point"] += point_ok
        total["tp"] += tp
        total["owner_missing"] += missing_owner
        total["nonempty"] += nonempty
    print("\n==== TOTAL ====")
    for k in ("gt", "pred", "reachable", "nonempty", "num", "name", "point", "tp", "owner_missing"):
        print(f"  {k}: {total[k]}")
    if total["reachable"]:
        print(f"  pinname exact / number-matched : {total['name']/max(1,total['num']):.1%}")
        print(f"  point<=5px / number-matched    : {total['point']/max(1,total['num']):.1%}")
        print(f"  pin F1 vs all GT pins          : {2*total['tp']/max(1,2*total['tp']+(total['gt']-total['tp'])*2+0):.4f}"
              f"  (tp {total['tp']} / gt {total['gt']})")


if __name__ == "__main__":
    sys.exit(main())
