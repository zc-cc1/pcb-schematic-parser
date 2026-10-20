"""Diff component bboxes between the 5th and 7th evaluation outputs."""
from __future__ import annotations

import json
from pathlib import Path

FIFTH = Path(r"C:/Users/Pc/Desktop/新建文件夹/评测结果/第五次评测-34.65/intermediate_files/final_results")
SEVENTH = Path(r"C:/Users/Pc/Desktop/新建文件夹/评测结果/第七次评测-33.525/intermediate_files/final_results")


def load(p: Path) -> dict:
    data = json.loads(p.read_text(encoding="utf-8"))
    return data


for case in ["1", "12", "13"]:
    a = load(FIFTH / f"{case}_Pred.json")
    b = load(SEVENTH / f"{case}_Pred.json")
    ca, cb = a["components"], b["components"]
    print(f"===== case {case}: {len(ca)} vs {len(cb)} components =====")
    moved = []
    for cid in list(ca.keys())[:200]:
        if cid not in cb:
            continue
        ba, bb = ca[cid].get("bbox"), cb[cid].get("bbox")
        if ba and bb and ba != bb:
            d = [round(y - x, 2) for x, y in zip(ba, bb)]
            moved.append((cid, ca[cid].get("type"), ba, bb, d))
    print(f"  moved boxes: {len(moved)}")
    for cid, typ, ba, bb, d in moved[:10]:
        print(f"   {cid:<18} {str(typ):<10} {ba} -> {bb}  delta={d}")
    if moved:
        import statistics
        for i in range(4):
            ds = [m[4][i] for m in moved]
            print(f"   delta[{i}]: mean {statistics.mean(ds):+.2f}  min {min(ds):+.2f}  max {max(ds):+.2f}")
    print()
