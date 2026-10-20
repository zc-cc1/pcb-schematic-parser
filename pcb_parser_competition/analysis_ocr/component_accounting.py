"""Clean component accounting on the 13 released cases, official rule.

For every GT component: emitted-and-correct (TP), emitted-but-a-field-differs, or never
emitted. For every prediction: a TP, a field mismatch, or an extra key. Power symbols
are matched by geometry with type+value compared and key/Name ignored, exactly as the
protocol says.
"""
import json
import math
import sys
from collections import Counter
from pathlib import Path

PRED = Path(r"D:\pcb_v8_check\out\final_results")
GT = Path(r"D:\pcb_v8_check\gt")
TOL = 20.0
POWER = {"gnd", "v"}


def norm_type(value):
    t = str(value or "").strip().lower()
    if t in {"ground", "gnd"}:
        return "gnd"
    if t in {"v", "vcc", "vdd"}:
        return "v"
    if t in {"box", "ic"}:
        return "box"
    return t


def centre(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def main():
    per_class = Counter()
    reasons = Counter()
    for folder in sorted((p for p in GT.iterdir() if p.is_dir()), key=lambda p: (len(p.name), p.name)):
        case = folder.name
        gt = json.loads(next(folder.glob("*_target_named.json")).read_text(encoding="utf-8"))
        pred = json.loads((PRED / f"{case}_Pred.json").read_text(encoding="utf-8"))
        gt_comps = gt["components"]
        our_comps = pred["components"]

        # power first: one-to-one by centre distance, type+value compared, key/Name ignored
        gt_power = [(k, v) for k, v in gt_comps.items() if norm_type(v.get("type")) in POWER]
        our_power = [(k, v) for k, v in our_comps.items() if norm_type(v.get("type")) in POWER]
        used = set()
        pairs = []
        for gk, gv in gt_power:
            best = None
            for ok, ov in our_power:
                if ok in used:
                    continue
                d = math.dist(centre(gv["bbox"]), centre(ov["bbox"]))
                if d <= TOL and (best is None or d < best[0]):
                    best = (d, ok)
            if best:
                used.add(best[1])
                pairs.append((gk, best[1]))
        for gk, ok in pairs:
            gv, ov = gt_comps[gk], our_comps[ok]
            same = (norm_type(gv["type"]) == norm_type(ov["type"])
                    and (gv.get("value") or None) == (ov.get("value") or None))
            if same:
                per_class[("POWER", norm_type(gv["type"]), "TP")] += 1
            else:
                per_class[("POWER", norm_type(gv["type"]), "field_mismatch")] += 1
                reasons["power field: " + ("value" if (gv.get("value") or None) != (ov.get("value") or None) else "type")] += 1
        per_class[("POWER", "all", "GT")] += len(gt_power)
        per_class[("POWER", "all", "never_emitted")] += len(gt_power) - len(pairs)
        per_class[("POWER", "all", "FP_extra")] += len(our_power) - len(pairs)

        # ordinary: key match, then fields
        gt_ord = {k: v for k, v in gt_comps.items() if norm_type(v.get("type")) not in POWER}
        our_ord = {k: v for k, v in our_comps.items() if norm_type(v.get("type")) not in POWER}
        for k, gv in gt_ord.items():
            per_class[("ORDINARY", norm_type(gv["type"]), "GT")] += 1
            ov = our_ord.get(k)
            if ov is None:
                per_class[("ORDINARY", norm_type(gv["type"]), "never_emitted")] += 1
                continue
            failed = []
            if norm_type(gv["type"]) != norm_type(ov["type"]):
                failed.append("type")
            if (gv.get("value") or None) != (ov.get("value") or None):
                failed.append("value")
            if (gv.get("Name") or None) != (ov.get("Name") or None):
                failed.append("Name")
            if not all(abs(a - b) <= TOL for a, b in zip(gv["bbox"], ov["bbox"])):
                failed.append("bbox")
            if failed:
                per_class[("ORDINARY", norm_type(gv["type"]), "field_mismatch")] += 1
                reasons["ordinary field: " + "+".join(failed)] += 1
            else:
                per_class[("ORDINARY", norm_type(gv["type"]), "TP")] += 1
        for k, ov in our_ord.items():
            if k not in gt_ord:
                kind = "UNRESOLVED" if str(k).startswith("UNRESOLVED") else "named"
                per_class[("ORDINARY", norm_type(ov.get("type")), "FP_extra")] += 1
                reasons["extra key: " + kind] += 1

    print(f"{'bucket':10s} {'type':10s} {'GT':>4} {'TP':>4} {'field':>6} {'never':>6} {'FP':>4}")
    for kind in ("POWER", "ORDINARY"):
        agg = Counter()
        for (k, t, field), value in per_class.items():
            if k != kind or t == "all":
                continue
            agg["GT"] += value if field == "GT" else 0
            agg["TP"] += value if field == "TP" else 0
            agg["field"] += value if field == "field_mismatch" else 0
            agg["never"] += value if field == "never_emitted" else 0
            agg["FP"] += value if field == "FP_extra" else 0
            print(f"{kind:10s} {t:10s} {value if field == 'GT' else '':>4} "
                  f"{value if field == 'TP' else '':>4} {value if field == 'field_mismatch' else '':>6} "
                  f"{value if field == 'never_emitted' else '':>6} {value if field == 'FP_extra' else '':>4}")
        print(f"{kind:10s} {'ALL':10s} {agg['GT']:>4} {agg['TP']:>4} {agg['field']:>6} {agg['never']:>6} {agg['FP']:>4}")
        print()
    print("reasons:")
    for key, value in reasons.most_common():
        print(f"  {key:34s} {value}")


if __name__ == "__main__":
    sys.exit(main())
