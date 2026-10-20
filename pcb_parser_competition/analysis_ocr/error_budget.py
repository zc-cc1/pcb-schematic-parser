"""Field-level error budget for the component metric on the 6 cached pages."""
import json
import sys
from collections import Counter
from pathlib import Path

BASE = Path(r"D:\pcb5_submission")
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "src"))
import adapter  # noqa: E402

GT_ROOT = Path(r"C:\Users\Pc\Desktop\新建文件夹\赛题六公开数据集\赛题六公开数据集\200_train_cases")
BATCH = Path(r"D:\pcb_v5_batch")
CASES = ["0001", "0002", "0018", "0029", "0080", "0102"]
TOL = 20.0
POWER = {"gnd", "v"}


def norm_type(t):
    t = str(t or "").strip().lower()
    if t in {"ground", "gnd"}:
        return "gnd"
    if t in {"v", "vcc", "vdd"}:
        return "v"
    if t in {"box", "ic"}:
        return "box"
    return t


def main():
    combos = Counter()
    field_fail = Counter()
    totals = {"pred": 0, "key_matched": 0, "tp": 0, "gt": 0}
    # also count the value situation split by type, which is where the choice matters
    value_stats = Counter()
    for case in CASES:
        payload = json.loads((BATCH / "output" / "component_results" / f"{case}_Pred.json").read_text(encoding="utf-8"))
        gt = json.loads(next((GT_ROOT / case).glob("*_target_named.json")).read_text(encoding="utf-8"))
        pred = adapter.to_target(payload)
        totals["gt"] += len(gt["components"])
        for key, comp in pred["components"].items():
            totals["pred"] += 1
            g = gt["components"].get(key)
            typ = norm_type(comp.get("type"))
            if g and typ in POWER:
                # power ignores key/Name; compare type + geometry only
                fails = []
                if norm_type(g.get("type")) != typ:
                    fails.append("type")
                if not all(abs(a - b) <= TOL for a, b in zip(comp["bbox"], g["bbox"])):
                    fails.append("bbox")
                fset = frozenset(fails) or frozenset({"OK"})
                combos[("POWER",) + tuple(sorted(fset))] += 1
                if not fails:
                    totals["tp"] += 1
                continue
            if not g:
                combos[("no_key",)] += 1
                continue
            totals["key_matched"] += 1
            fails = []
            if norm_type(g.get("type")) != typ:
                fails.append("type")
            if (comp.get("Name") or None) != (g.get("Name") or None):
                fails.append("Name")
            if (comp.get("value") or None) != (g.get("value") or None):
                fails.append("value")
            if not all(abs(a - b) <= TOL for a, b in zip(comp["bbox"], g["bbox"])):
                fails.append("bbox")
            for f in fails:
                field_fail[f] += 1
            combos[tuple(sorted(fails)) or ("OK",)] += 1
            if not fails:
                totals["tp"] += 1
            # value decision quality on passives
            if typ in {"r", "c", "l"}:
                gv = g.get("value")
                pv = comp.get("value")
                if gv is None and pv is not None:
                    value_stats["emitted_but_gt_null"] += 1
                elif gv is not None and pv is None:
                    value_stats["missing_but_gt_has"] += 1
                elif gv is not None and pv is not None and gv != pv:
                    value_stats["wrong_string"] += 1
                elif gv is not None and pv == gv:
                    value_stats["correct"] += 1
                else:
                    value_stats["correct_null"] += 1
    print("totals:", totals)
    print("\nfailing-field counts on key-matched components:", dict(field_fail))
    print("\nfailure combinations (ordinary components):")
    for combo, n in combos.most_common():
        print(f"  {n:4d}  {'+'.join(combo)}")
    print("\npassive value decisions:", dict(value_stats))


if __name__ == "__main__":
    sys.exit(main())
