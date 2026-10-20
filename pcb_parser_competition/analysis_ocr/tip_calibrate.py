"""Calibrate a global tip offset against GT tips.

The previous session calibrated component *boxes* the same way (fixed symbol size +
centre offset, measured on 40 public pages, worth +6.4 points). Pin tips show the
same signature: our tips sit a few pixels below/inside the GT connection point, so
most pairs fall just outside the official 5 px tolerance.

Evaluation is honest about re-pairing: the shift is applied first, then GT tips are
matched one-to-one to the nearest shifted tip within 5 px.
"""
import json
import math
import sys
from pathlib import Path

GT_ROOT = Path(r"C:\Users\Pc\Desktop\新建文件夹\赛题六公开数据集\赛题六公开数据集\200_train_cases")
REPLAY = Path(r"D:\pcb_v6_check\pin_replay")
TOL = 5.0
GOLDEN = ["0025", "0030", "0036", "0037", "0042", "0059", "0065", "0070", "0099", "0100"]
BIG = ["0001", "0002", "0018", "0080", "0102"]


def norm_type(t):
    return "box" if str(t or "").strip().lower() in {"box", "ic"} else str(t or "").strip().lower()


def load(case):
    f = REPLAY / case / "pin_text_result.json"
    if not f.exists():
        # the replays for the big pages used the same path convention
        alt = Path(r"D:\pcb_v6_check\pin_replay_big") / case / "pin_text_result.json"
        if not alt.exists():
            return None
        f = alt
    result = json.loads(f.read_text(encoding="utf-8"))
    gt = json.loads(next((GT_ROOT / case).glob("*_target_named.json")).read_text(encoding="utf-8"))
    gt_points = []
    for comp, group in (gt.get("pins") or {}).items():
        if norm_type((gt["components"].get(comp) or {}).get("type")) != "box":
            continue
        for want in group.values():
            gt_points.append((float(want["point"]["x"]), float(want["point"]["y"])))
    ours = [tuple(p["point"]) for p in result["pin_candidates"] if p.get("owner_status") == "assigned"]
    return gt_points, ours


def hits(gt_points, ours, dx, dy):
    shifted = [(p[0] + dx, p[1] + dy) for p in ours]
    used = set()
    hit = 0
    for gx, gy in gt_points:
        best = None
        for index, (x, y) in enumerate(shifted):
            if index in used:
                continue
            d = math.hypot(x - gx, y - gy)
            if d <= TOL and (best is None or d < best[0]):
                best = (d, index)
        if best:
            used.add(best[1])
            hit += 1
    return hit


def evaluate(cases, shifts):
    data = [(case, load(case)) for case in cases]
    data = [(case, d) for case, d in data if d]
    total_gt = sum(len(d[0]) for _, d in data)
    print(f"cases {len(data)}, GT IC tips {total_gt}")
    rows = []
    for dx, dy in shifts:
        hit = sum(hits(gt, ours, dx, dy) for _, (gt, ours) in data)
        rows.append((hit, dx, dy))
    for case, (gt, ours) in data:
        base = hits(gt, ours, 0, 0)
        best = max((hits(gt, ours, dx, dy), dx, dy) for dx, dy in shifts)
        print(f"  {case}: gt={len(gt):3d} ours={len(ours):3d} base≤5px={base:3d} "
              f"best={best[0]:3d} at ({best[1]},{best[2]})")
    rows.sort(reverse=True)
    print("  top global shifts:")
    for hit, dx, dy in rows[:6]:
        print(f"    ({dx:+.1f},{dy:+.1f}) -> {hit}/{total_gt} = {hit/total_gt:.1%}")
    return rows


if __name__ == "__main__":
    shifts = [(dx / 2, dy / 2) for dx in range(-12, 13) for dy in range(-12, 13)]
    print("=== golden 10 ===")
    evaluate(GOLDEN, shifts)
    print("\n=== big 6 ===")
    evaluate(BIG, shifts)
    sys.exit(0)
