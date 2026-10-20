"""Is there a systematic tip offset? Compares our pin tips with GT IC tips.

If ours sit consistently a few pixels along one axis, a single correction could
move many pins inside the official 5 px tolerance - which gates both pin F1 and,
through `validate_target`, whether a network can reference the pin at all.
"""
import json
import math
import statistics
import sys
from pathlib import Path

GT_ROOT = Path(r"C:\Users\Pc\Desktop\新建文件夹\赛题六公开数据集\赛题六公开数据集\200_train_cases")
REPLAY = Path(r"D:\pcb_v6_check\pin_replay")
CASES = sys.argv[1:] or ["0025", "0030", "0036", "0037", "0042", "0059", "0065", "0070", "0099", "0100"]
POWER = {"gnd", "v"}


def norm_type(t):
    t = str(t or "").strip().lower()
    return "box" if t in {"box", "ic"} else t


def main():
    dxs, dys, dists = [], [], []
    per_side = {}
    for case in CASES:
        f = REPLAY / case / "pin_text_result.json"
        if not f.exists():
            continue
        result = json.loads(f.read_text(encoding="utf-8"))
        gt = json.loads(next((GT_ROOT / case).glob("*_target_named.json")).read_text(encoding="utf-8"))
        ours = [(tuple(p["point"]), p) for p in result["pin_candidates"] if p.get("owner_status") == "assigned"]
        for comp, group in (gt.get("pins") or {}).items():
            gcomp = gt["components"].get(comp) or {}
            if norm_type(gcomp.get("type")) != "box":
                continue
            box = gcomp["bbox"]
            for pin_key, want in group.items():
                point = (float(want["point"]["x"]), float(want["point"]["y"]))
                best = None
                for ours_point, pin in ours:
                    d = math.dist(ours_point, point)
                    if d <= 20 and (best is None or d < best[0]):
                        best = (d, ours_point)
                if best is None:
                    continue
                d, ours_point = best
                dx, dy = ours_point[0] - point[0], ours_point[1] - point[1]
                dxs.append(dx)
                dys.append(dy)
                dists.append(d)
                # which side of the symbol is this pin on?
                side = min(((abs(point[0] - box[0]), "left"), (abs(point[0] - box[2]), "right"),
                            (abs(point[1] - box[1]), "bottom"), (abs(point[1] - box[3]), "top")))[1]
                per_side.setdefault(side, []).append((dx, dy, d))
    if not dists:
        print("no pairs found")
        return 1

    def stats(values):
        return (round(statistics.mean(values), 2), round(statistics.median(values), 2),
                round(min(values), 2), round(max(values), 2))

    print(f"pairs: {len(dists)}")
    print(f"  distance      mean/median/min/max = {stats(dists)}")
    print(f"  dx (ours-gt)  mean/median/min/max = {stats(dxs)}")
    print(f"  dy (ours-gt)  mean/median/min/max = {stats(dys)}")
    for side, rows in sorted(per_side.items()):
        d = [r[2] for r in rows]
        print(f"  side {side:6s} n={len(rows):3d} dx_med={statistics.median([r[0] for r in rows]):6.2f} "
              f"dy_med={statistics.median([r[1] for r in rows]):6.2f} dist_med={statistics.median(d):5.2f}")
    inside = sum(1 for d in dists if d <= 5)
    print(f"  within 5px: {inside}/{len(dists)} = {inside/len(dists):.1%}")
    # what a global (dx,dy) shift would buy
    best = None
    for sx in range(-10, 11):
        for sy in range(-10, 11):
            hit = sum(1 for dx, dy in zip(dxs, dys) if math.hypot(dx + sx, dy + sy) <= 5)
            if best is None or hit > best[0]:
                best = (hit, sx, sy)
    print(f"  best global shift: +({best[1]},{best[2]}) -> {best[0]}/{len(dists)} within 5px "
          f"({best[0]/len(dists):.1%})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
