"""Decompose the v5 -> v6 end-to-end change into field flips vs geometry/detection."""
import json
import sys
from pathlib import Path

V5 = Path(r"D:\pcb_v5_batch\preds")
V6 = Path(r"D:\pcb_v6_check\out_v6_full\final_results")
GT_ROOT = Path(r"C:\Users\Pc\Desktop\新建文件夹\赛题六公开数据集\赛题六公开数据集\200_train_cases")
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


def ok(comp, g):
    return (norm_type(comp.get("type")) == norm_type(g.get("type"))
            and (comp.get("Name") or None) == (g.get("Name") or None)
            and (comp.get("value") or None) == (g.get("value") or None)
            and all(abs(a - b) <= TOL for a, b in zip(comp["bbox"], g["bbox"])))


def main():
    tot = {"field_up": 0, "field_down": 0, "other_up": 0, "other_down": 0,
           "same": 0, "only_v5": 0, "only_v6": 0}
    for case in CASES:
        gt = json.loads(next((GT_ROOT / case).glob("*_target_named.json")).read_text(encoding="utf-8"))
        a = json.loads((V5 / f"{case}_Pred.json").read_text(encoding="utf-8"))["components"]
        b = json.loads((V6 / f"{case}_Pred.json").read_text(encoding="utf-8"))["components"]
        for k in sorted(set(a) | set(b)):
            g = gt["components"].get(k)
            if k not in b:
                tot["only_v5"] += 1
                continue
            if k not in a:
                tot["only_v6"] += 1
                continue
            before, after = (ok(a[k], g) if g else False), (ok(b[k], g) if g else False)
            field_changed = ((a[k].get("Name") or None) != (b[k].get("Name") or None)
                             or (a[k].get("value") or None) != (b[k].get("value") or None))
            box_changed = a[k]["bbox"] != b[k]["bbox"]
            if before and not after:
                key = "field_down" if field_changed else "other_down"
                tot[key] += 1
                print(f"  REGRESS {case}/{k}: {json.dumps(a[k], ensure_ascii=False)} -> {json.dumps(b[k], ensure_ascii=False)}")
            elif after and not before:
                key = "field_up" if field_changed else "other_up"
                tot[key] += 1
                print(f"  FIXED   {case}/{k}: Name/value {a[k].get('Name')!r}/{a[k].get('value')!r} -> "
                      f"{b[k].get('Name')!r}/{b[k].get('value')!r}")
            else:
                tot["same"] += 1
    print("\n", tot)


if __name__ == "__main__":
    sys.exit(main())
