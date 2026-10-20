"""Is the fixed canonical box wrong on pages drawn at another scale? Compare three
strategies for the templated classes (r/c/l/d/led):

  raw       keep the detected box
  fixed     current behaviour: canonical (short, long) in absolute pixels
  scaled    canonical shape x a per-page scale estimated from the detected boxes

The per-page scale is the median, over this page's r/c/l components, of
`detected long side / canonical long side`. That keeps the calibrated *shape* but lets
the *scale* follow the page.
"""
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

PRED_RAW = Path(r"D:\pcb_v8_check\out\component_results")
GT = Path(r"D:\pcb_v8_check\gt")
TOL = 20.0
TEMPLATE = {"r": (8.0, 20.0), "c": (16.0, 20.0), "l": (6.0, 34.0),
            "d": (14.0, 20.0), "led": (14.0, 20.0)}
SCALE_CLASSES = ("r", "c", "l")


def norm_type(value):
    t = str(value or "").strip().lower()
    if t in {"ground", "gnd"}:
        return "gnd"
    if t in {"v", "vcc", "vdd"}:
        return "v"
    if t in {"box", "ic"}:
        return "box"
    return t


def templated(bbox, kind, scale=1.0):
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    short, long = TEMPLATE[kind]
    short, long = short * scale, long * scale
    width, height = (long, short) if x2 - x1 > y2 - y1 else (short, long)
    return [cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2]


def err(a, b):
    return max(abs(x - y) for x, y in zip(a, b))


def main():
    totals = Counter()
    per_case = defaultdict(Counter)
    for folder in sorted((p for p in GT.iterdir() if p.is_dir()), key=lambda p: (len(p.name), p.name)):
        case = folder.name
        gt = json.loads(next(folder.glob("*_target_named.json")).read_text(encoding="utf-8"))
        raw = json.loads((PRED_RAW / f"{case}_Pred.json").read_text(encoding="utf-8"))
        # per-page scale from r/c/l detected boxes against the canonical long side
        ratios = []
        for key, comp in raw["components"].items():
            kind = {"resistor": "r", "capacitor": "c", "inductor": "l"}.get(str(comp.get("type")))
            if not kind:
                continue
            x1, y1, x2, y2 = comp["bbox"]
            long_side = max(x2 - x1, y2 - y1)
            ratios.append(long_side / max(TEMPLATE[kind]))
        scale = statistics.median(ratios) if ratios else 1.0
        for key, gv in gt["components"].items():
            kind = norm_type(gv.get("type"))
            if kind not in TEMPLATE:
                continue
            ov = raw["components"].get(key)
            if ov is None:
                continue
            raw_box = [float(v) for v in ov["bbox"]]
            fixed = templated(raw_box, kind, 1.0)
            scaled = templated(raw_box, kind, scale)
            for name, box in (("raw", raw_box), ("fixed", fixed), ("scaled", scaled)):
                per_case[case][f"{name}_n"] += 1
                per_case[case][f"{name}_ok"] += err(box, gv["bbox"]) <= TOL
        totals["pages"] += 1
        totals["scale_x10"] += round(scale * 10)

    print(f"{'case':>4} {'scale':>6} {'raw ok':>8} {'fixed ok':>9} {'scaled ok':>10} {'n':>4}")
    grand = Counter()
    for case, counts in per_case.items():
        n = counts["fixed_n"]
        scale = None
        print(f"{case:>4} {'':>6} {counts['raw_ok']:>8} {counts['fixed_ok']:>9} "
              f"{counts['scaled_ok']:>10} {n:>4}")
        for name in ("raw", "fixed", "scaled"):
            grand[f"{name}_ok"] += counts[f"{name}_ok"]
            grand[f"{name}_n"] += counts[f"{name}_n"]
    print("\n== totals over templated classes ==")
    for name in ("raw", "fixed", "scaled"):
        ok, n = grand[f"{name}_ok"], grand[f"{name}_n"]
        print(f"  {name:7s} {ok}/{n} = {ok/max(1,n):.1%} within 20px")


if __name__ == "__main__":
    sys.exit(main())
