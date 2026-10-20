"""Detail for the components whose key and fields match but whose bbox fails.

Prints GT vs predicted box, the size ratio per axis and the centre offset, so the fix
can be chosen per class: a canonical size (like r/c/l already have), a centre offset,
or "size genuinely varies, leave it alone".
"""
import json
import math
import sys
from collections import Counter, defaultdict
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


def main():
    rows = []
    for folder in sorted((p for p in GT.iterdir() if p.is_dir()), key=lambda p: (len(p.name), p.name)):
        case = folder.name
        gt = json.loads(next(folder.glob("*_target_named.json")).read_text(encoding="utf-8"))
        pred = json.loads((PRED / f"{case}_Pred.json").read_text(encoding="utf-8"))
        for key, gv in gt["components"].items():
            if norm_type(gv.get("type")) in POWER:
                continue
            ov = pred["components"].get(key)
            if ov is None:
                continue
            same_fields = (norm_type(gv.get("type")) == norm_type(ov.get("type"))
                           and (gv.get("value") or None) == (ov.get("value") or None)
                           and (gv.get("Name") or None) == (ov.get("Name") or None))
            if not same_fields:
                continue
            err = max(abs(a - b) for a, b in zip(gv["bbox"], ov["bbox"]))
            if err <= TOL:
                continue
            gx1, gy1, gx2, gy2 = gv["bbox"]
            ox1, oy1, ox2, oy2 = ov["bbox"]
            gw, gh = gx2 - gx1, gy2 - gy1
            ow, oh = ox2 - ox1, oy2 - oy1
            rows.append({
                "case": case, "key": key, "type": norm_type(gv.get("type")),
                "err": round(err, 1),
                "gt": [round(v, 1) for v in gv["bbox"]],
                "our": [round(v, 1) for v in ov["bbox"]],
                "ws": round(ow / gw, 2) if gw else None,
                "hs": round(oh / gh, 2) if gh else None,
                "dx": round((ox1 + ox2) / 2 - (gx1 + gx2) / 2, 1),
                "dy": round((oy1 + oy2) / 2 - (gy1 + gy2) / 2, 1),
            })
    print(f"bbox-only failures: {len(rows)}")
    print(f"{'case':>4} {'key':>8} {'type':>16} {'err':>6} {'w×':>5} {'h×':>5} {'dx':>7} {'dy':>7}   gt / ours")
    for row in rows:
        print(f"{row['case']:>4} {row['key']:>8} {row['type']:>16} {row['err']:>6} "
              f"{str(row['ws']):>5} {str(row['hs']):>5} {row['dx']:>7} {row['dy']:>7}   "
              f"{row['gt']} / {row['our']}")
    by_type = defaultdict(list)
    for row in rows:
        by_type[row["type"]].append(row)
    print("\nsummary by type:")
    for kind, items in sorted(by_type.items(), key=lambda kv: -len(kv[1])):
        ws = [i["ws"] for i in items if i["ws"]]
        hs = [i["hs"] for i in items if i["hs"]]
        dx = [i["dx"] for i in items]
        dy = [i["dy"] for i in items]
        print(f"  {kind:16s} n={len(items):2d}  w× med {sorted(ws)[len(ws)//2] if ws else '-':>5}  "
              f"h× med {sorted(hs)[len(hs)//2] if hs else '-':>5}  "
              f"dx med {sorted(dx)[len(dx)//2]:>7}  dy med {sorted(dy)[len(dy)//2]:>7}")


if __name__ == "__main__":
    sys.exit(main())
