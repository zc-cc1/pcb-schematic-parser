"""Measure how many ordinary components are *reachable* under the official rubric.

The official component metric matches ordinary components by key (designator), and
the designator has to be recovered from the picture. Three failure modes look
identical in the score but need completely different responses:

  (a) not rendered in the image        -> nobody can match it, stop trying
  (b) rendered but not read by OCR     -> an OCR problem, fixable
  (c) read but not bound to a component -> an association problem, fixable

This script separates them: for every GT ordinary component it asks whether a
designator-like OCR token exists near the GT box (the ceiling), and compares that
ceiling with what the pipeline actually resolved. Read-only and offline.

Usage:
    .venv-gpu\\Scripts\\python.exe scripts\\measure_designator_ceiling.py `
        --pred-root D:\\pcb_v5_batch\\output --gt-root "..\\赛题六公开数据集\\赛题六公开数据集\\200_train_cases" `
        --cases 0001 0002 0018 0029 0080 0102
"""
import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from evaluate_official_component import load_gt_case  # noqa: E402

DESIGNATOR = re.compile(r"^(?:R|C|L|D|Q|U|IC|X|Y|J|P|F|SW|TP)\d+$", re.I)
POWER = {"ground", "voltage_source"}


def box_gap(a, b):
    return max(0., max(a[0] - b[2], b[0] - a[2])) ** 2 + max(0., max(a[1] - b[3], b[1] - a[3])) ** 2


def ocr_tokens(pred_root, case):
    """Top-left OCR token boxes from the pipeline's own log artefacts."""
    for pattern in (f"logs/{case}_ocr_texts.json", f"logs/{case}_summary.json"):
        path = pred_root / case / ".." / pattern
        candidate = (pred_root / pattern).resolve()
        if candidate.exists() and "ocr" in pattern:
            raw = json.loads(candidate.read_text(encoding="utf-8"))
            out = []
            for row in raw.get("texts") or []:
                polygon = row.get("polygon")
                if not polygon:
                    continue
                xs = [p[0] for p in polygon]
                ys = [p[1] for p in polygon]
                out.append({"text": str(row.get("text", "")).strip(),
                            "bbox": [min(xs), min(ys), max(xs), max(ys)]})
            return out
    return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-root", type=Path, required=True,
                    help="run output root holding logs/<case>_ocr_texts.json and final_results/")
    ap.add_argument("--gt-root", type=Path, required=True)
    ap.add_argument("--cases", nargs="+", required=True)
    ap.add_argument("--radius", type=float, default=60.,
                    help="how far from the GT box a designator may sit to count as printed nearby")
    args = ap.parse_args()

    print(f"{'case':>6} {'GT ord':>7} {'printed':>8} {'ceil%':>6} {'resolved':>9} {'act%':>6} "
          f"{'not printed':>12} {'read,unbound':>13}")
    totals = Counter()
    for case in args.cases:
        gt_files = list((args.gt_root / case).glob("*_target_named.json"))
        if not gt_files:
            print(f"{case:>6}  (no GT)")
            continue
        pngs = list((args.gt_root / case).glob("*.png"))
        from PIL import Image
        height = Image.open(pngs[0]).size[1]
        gt = load_gt_case(gt_files[0], height)
        tokens = ocr_tokens(args.pred_root, case)
        pred_path = args.pred_root / "final_results" / f"{case}_Pred.json"
        pred = json.loads(pred_path.read_text(encoding="utf-8")) if pred_path.exists() else {"components": {}}
        resolved = {k for k in (pred.get("components") or {}) if not str(k).startswith("UNRESOLVED")}

        ordinary = [g for g in gt if g["class_name"] not in POWER]
        printed = unbound = 0
        details = []
        for g in ordinary:
            near = [t for t in tokens
                    if DESIGNATOR.match(t["text"]) and box_gap(g["bbox"], t["bbox"]) <= args.radius ** 2]
            if near:
                printed += 1
                # was that designator actually bound to *some* component in the output?
                if not any(t["text"].upper() in resolved for t in near):
                    unbound += 1
                details.append((g["key"], [t["text"] for t in near]))
            else:
                details.append((g["key"], None))
        ceil = printed / len(ordinary) if ordinary else 0.
        actual = sum(1 for g in ordinary if g["key"] in resolved) / len(ordinary) if ordinary else 0.
        print(f"{case:>6} {len(ordinary):>7} {printed:>8} {ceil:>6.0%} {len([g for g in ordinary if g['key'] in resolved]):>9} "
              f"{actual:>6.0%} {len(ordinary)-printed:>12} {unbound:>13}")
        totals["gt"] += len(ordinary)
        totals["printed"] += printed
        totals["actual"] += sum(1 for g in ordinary if g["key"] in resolved)
        if len(sys.argv) > 99:  # debug: only with an absurd arg list
            for row in details[:10]:
                print("     ", row)

    if totals["gt"]:
        print(f"\nTOTAL ordinary GT={totals['gt']}  printed designs={totals['printed']} "
              f"({totals['printed']/totals['gt']:.0%} ceiling)  actually resolved={totals['actual']} "
              f"({totals['actual']/totals['gt']:.0%})")
        print(f"gap to ceiling = {totals['printed']-totals['actual']} components that were printed and readable "
              f"but did not reach the output")


if __name__ == "__main__":
    main()
