"""Score the OCR matching itself against any pred / OCR-log / GT triple.

Keeps three questions apart and never mixes them:
  1. did the recogniser put the GT designator string anywhere in its output?
  2. did the binder turn a visible string into the component key?
  3. for the key-matched components, are type / Name / value / bbox right?
  4. for GT IC pins, does our tip land on them and carry the right number/name?

Both sides use the submission convention (bottom-left origin, y up), so points and
boxes are compared directly.
"""
import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

POWER = {"gnd", "v", "vcc", "vdd", "ground", "voltage_source"}
TOL = 20.0
TIP_TOL = 5.0


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
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--logs", required=True, help="directory with <case>_ocr_texts.json")
    ap.add_argument("--gt-root", required=True)
    ap.add_argument("--glob", default="*_target_named.json")
    ap.add_argument("--cases", nargs="*", default=None)
    args = ap.parse_args()

    pred_dir, log_dir, gt_root = Path(args.pred), Path(args.logs), Path(args.gt_root)
    cases = args.cases or sorted((p.name for p in gt_root.iterdir() if p.is_dir()),
                                 key=lambda s: (len(s), s))
    tally = Counter()
    print(f"{'case':>4} {'gtOrd':>5} {'visible':>7} {'bound':>5} {'fullTP':>6} | "
          f"{'pinGT':>5} {'tip5':>4} {'tip15':>5} {'num':>4} {'name':>4}")
    for case in cases:
        gt_file = next(gt_root.joinpath(case).glob(args.glob), None)
        pred_file = pred_dir / f"{case}_Pred.json"
        if gt_file is None or not pred_file.exists():
            print(f"{case:>4}  (missing)")
            continue
        gt = json.loads(gt_file.read_text(encoding="utf-8"))
        pred = json.loads(pred_file.read_text(encoding="utf-8"))

        seen = set()
        log = log_dir / f"{case}_ocr_texts.json"
        if log.exists():
            for row in json.loads(log.read_text(encoding="utf-8")).get("texts", []):
                for text in [row["text"]] + [o["text"] for o in row.get("observations", [])]:
                    for part in str(text).replace("/", " ").replace("(", " ").replace(")", " ").split():
                        seen.add(part)
                        seen.add(part.upper())

        gt_ord = {k: v for k, v in gt["components"].items() if norm_type(v.get("type")) not in POWER}
        have = pred["components"]
        visible = [k for k in gt_ord if k in seen or k.upper() in seen]
        bound = [k for k in visible if k in have]
        full = 0
        for key in bound:
            g, p = gt_ord[key], have[key]
            if (norm_type(g.get("type")) == norm_type(p.get("type"))
                    and (p.get("Name") or None) == (g.get("Name") or None)
                    and (p.get("value") or None) == (g.get("value") or None)
                    and all(abs(a - b) <= TOL for a, b in zip(p["bbox"], g["bbox"]))):
                full += 1

        gt_pins = [(k, pin, want) for k, group in (gt.get("pins") or {}).items()
                   if norm_type((gt["components"].get(k) or {}).get("type")) == "box"
                   for pin, want in group.items()]
        ours = [(float(pin["point"]["x"]), float(pin["point"]["y"]), pin_key, pin)
                for group in (pred.get("pins") or {}).values()
                for pin_key, pin in group.items()]
        used = set()
        tip5 = tip15 = num_ok = name_ok = 0
        for _, pin_key, want in gt_pins:
            point = (float(want["point"]["x"]), float(want["point"]["y"]))
            best = None
            for index, (x, y, _, _) in enumerate(ours):
                if index in used:
                    continue
                d = math.hypot(x - point[0], y - point[1])
                if d <= 15 and (best is None or d < best[0]):
                    best = (d, index)
            if best is None:
                continue
            used.add(best[1])
            tip15 += 1
            if best[0] <= TIP_TOL:
                tip5 += 1
            _, _, got_key, got = ours[best[1]]
            num_ok += got_key[4:] == pin_key[4:]
            name_ok += (got.get("pinname") or None) == (want.get("pinname") or None)

        print(f"{case:>4} {len(gt_ord):5d} {len(visible):7d} {len(bound):5d} {full:6d} | "
              f"{len(gt_pins):5d} {tip5:4d} {tip15:5d} {num_ok:4d} {name_ok:4d}")
        tally.update(gt_ord=len(gt_ord), visible=len(visible), bound=len(bound), full=full,
                     pinGT=len(gt_pins), tip5=tip5, tip15=tip15, num=num_ok, name=name_ok)

    print("\n== TOTAL ==")
    print(f"  GT ordinary designators      {tally['gt_ord']}")
    print(f"  visible in OCR output        {tally['visible']} "
          f"({tally['visible']/max(1,tally['gt_ord']):.1%})")
    print(f"  bound to a predicted key     {tally['bound']} "
          f"({tally['bound']/max(1,tally['gt_ord']):.1%})")
    print(f"  fully correct component      {tally['full']} "
          f"({tally['full']/max(1,tally['gt_ord']):.1%})")
    print(f"  GT IC pins                   {tally['pinGT']}")
    print(f"  tip within 5px               {tally['tip5']} "
          f"({tally['tip5']/max(1,tally['pinGT']):.1%})")
    print(f"  tip within 15px              {tally['tip15']} "
          f"({tally['tip15']/max(1,tally['pinGT']):.1%})")
    print(f"  number correct on matched    {tally['num']} "
          f"({tally['num']/max(1,tally['tip15']):.1%} of 15px matches)")
    print(f"  pinname correct on matched   {tally['name']} "
          f"({tally['name']/max(1,tally['tip15']):.1%} of 15px matches)")


if __name__ == "__main__":
    sys.exit(main())
