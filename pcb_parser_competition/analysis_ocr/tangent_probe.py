"""Detail for the pins lost to the tangent gate: how far outside are they?

Prints, for every GT IC pin whose number text exists but is rejected by
`tangent_outside`, the measured tangent, the current per-pin limit, the pin pitch on
that side, and which config knob would have to move.
"""
import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

BASE = Path(r"D:\pcb5_submission")
sys.path.insert(0, str(BASE / "src"))
from pcb_parser.component_text import center  # noqa: E402
from pcb_parser.pin_text import contexts  # noqa: E402
from pcb_parser.pin_text_v2 import tokens_with_views  # noqa: E402

TIP = 15.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--gt-root", required=True)
    ap.add_argument("--glob", default="*_target_named.json")
    args = ap.parse_args()
    cfg = json.loads((BASE / "configs" / "pin_text_v2.json").read_text(encoding="utf-8"))
    pred_dir, work_root, gt_root = Path(args.pred), Path(args.work), Path(args.gt_root)

    rows = []
    for folder in sorted((p for p in gt_root.iterdir() if p.is_dir()), key=lambda p: (len(p.name), p.name)):
        case = folder.name
        gt_file = next(folder.glob(args.glob), None)
        pred_file = pred_dir / f"{case}_Pred.json"
        png = next(folder.glob("*.png"), None)
        if gt_file is None or not pred_file.exists() or png is None:
            continue
        gt = json.loads(gt_file.read_text(encoding="utf-8"))
        pred = json.loads(pred_file.read_text(encoding="utf-8"))
        import hashlib
        digest = hashlib.sha256(png.read_bytes()).hexdigest()
        work = None
        for candidate in work_root.iterdir():
            owners = sorted(candidate.glob("**/ownership/pin_ownership.json"))
            if owners and json.loads(owners[0].read_text(encoding="utf-8")).get("image_sha256") == digest:
                work = candidate
                break
        if work is None:
            continue
        ownership = json.loads(next(work.glob("**/ownership/pin_ownership.json")).read_text(encoding="utf-8"))
        texts_file = next(work.glob("**/upstream/upstream/texts.json"), None)
        texts = json.loads(texts_file.read_text(encoding="utf-8"))["texts"] if texts_file else []
        ctx = contexts(ownership, cfg)
        tokens = tokens_with_views(texts)
        height = float(ownership["size"][1])
        exported = [(float(p["point"]["x"]), float(p["point"]["y"]))
                    for g in (pred.get("pins") or {}).values() for p in g.values()]

        for comp, group in (gt.get("pins") or {}).items():
            if str((gt["components"].get(comp) or {}).get("type")) not in ("box", "ic"):
                continue
            for pin_key, want in group.items():
                point = (float(want["point"]["x"]), float(want["point"]["y"]))
                if any(math.hypot(x - point[0], y - point[1]) <= 5 for x, y in exported):
                    continue
                near = [(math.hypot(float(c["point"][0]) - point[0], height - float(c["point"][1]) - point[1]), c)
                        for c in ownership["pin_candidates"] if c["candidate_id"] in ctx]
                near = [r for r in near if r[0] <= TIP]
                if not near:
                    continue
                tip = min(near, key=lambda r: r[0])[1]
                context = ctx[tip["candidate_id"]]
                want_text = pin_key[4:]
                matches = [t for t in tokens if t["text"] == want_text]
                if not matches:
                    continue
                # keep only the ones failing on tangent
                axis = 1 if context["side"] in {"left", "right"} else 0
                info = []
                for token in matches:
                    ca = center(token["bbox"])
                    tp = (float(tip["point"][0]), float(tip["point"][1]))
                    tangent = abs(ca[axis] - tp[axis])
                    inward = abs(ca[1 - axis] - tp[1 - axis])
                    if token["confidence"] < cfg["min_confidence"]:
                        continue
                    if tangent > context["tangent_limit"]:
                        info.append((round(tangent, 1), round(inward, 1), round(context["tangent_limit"], 1),
                                     round(token["confidence"], 3)))
                if info:
                    rows.append((case, comp, pin_key[4:], want.get("pinname"), context["side"],
                                 min(info)))

    print(f"pins rejected by the tangent gate: {len(rows)}")
    print(f"{'case':>4} {'pin':>10} {'name':>10} {'side':>6} {'tangent':>8} {'inward':>7} {'limit':>6} {'conf':>6}")
    for case, comp, number, name, side, info in rows:
        print(f"{case:>4} {comp + '.' + number:>10} {str(name)[:10]:>10} {side:>6} "
              f"{info[0]:>8} {info[1]:>7} {info[2]:>6} {info[3]:>6}")
    if rows:
        over = [r[5][0] - r[5][2] for r in rows]
        print(f"\n  tangent exceeds the limit by: min {min(over):.1f}px, median "
              f"{sorted(over)[len(over)//2]:.1f}px, max {max(over):.1f}px")
        sides = Counter(r[4] for r in rows)
        print(f"  sides: {dict(sides)}")
        # what a fixed floor on the limit would buy
        for floor in (8, 10, 12, 15, 22):
            hit = sum(1 for r in rows if r[5][0] <= max(floor, r[5][2]))
            print(f"  limit floor {floor:2d}px -> {hit}/{len(rows)} would pass")


if __name__ == "__main__":
    sys.exit(main())
