"""How many passive components actually carry a printed designator?

Uses the CAD annotation layer bundled with Dataset300 (`*_CVJsonStd.json` -> texts,
each with `content` and `location`). For every resistor/capacitor/inductor/diode we
ask whether a reference-shaped string (R12, C7, ...) sits next to it. Same
coordinate space on both sides, so the origin convention does not matter.

Then, for the pages that also have the published `_target_named.json`, it compares
the GT keys with the strings the drawing actually contains.
"""
import json
import math
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

DATASET = Path(r"C:\Users\Pc\Desktop\新建文件夹\Dataset300")
PUBLIC = Path(r"C:\Users\Pc\Desktop\新建文件夹\赛题六公开数据集\赛题六公开数据集\200_train_cases")
REF = re.compile(r"^(?:IC|SW|TP|FB|FET|LED|JP|[RCLDQUXYJPFS])\d+$", re.I)
PASSIVE_CATEGORIES = {"R", "C", "L", "D", "Led", "LED", "Diode", "Inductor",
                      "Resistor", "Capacitor"}
NEAR = 60.0


def centre(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def gap(a, b):
    return math.hypot(max(a[0] - b[2], b[0] - a[2], 0), max(a[1] - b[3], b[1] - a[3], 0))


def main():
    rows = []
    for folder in sorted(p for p in DATASET.iterdir() if p.is_dir()):
        std = next(folder.glob("*_CVJsonStd.json"), None)
        if std is None:
            continue
        doc = json.loads(std.read_text(encoding="utf-8"))
        texts = []
        for item in (doc.get("texts") or {}).values():
            content = str(item.get("content") or "").strip()
            loc = item.get("location") or {}
            if not content or not all(k in loc for k in ("x1", "y1", "x2", "y2")):
                continue
            texts.append((content, [float(loc["x1"]), float(loc["y1"]),
                                    float(loc["x2"]), float(loc["y2"])]))
        printed = [t for t in texts if REF.match(t[0])]
        passives = []
        for comp in (doc.get("components") or {}).values():
            category = str(comp.get("category") or "")
            if category not in PASSIVE_CATEGORIES:
                continue
            loc = comp.get("location") or {}
            if not all(k in loc for k in ("x1", "y1", "x2", "y2")):
                continue
            passives.append([float(loc["x1"]), float(loc["y1"]),
                             float(loc["x2"]), float(loc["y2"])])
        local = 0
        for box in passives:
            if any(gap(box, t[1]) <= NEAR for t in printed):
                local += 1
        # GT comparison where the published answer exists
        gt = None
        named = PUBLIC / folder.name
        if named.is_dir():
            f = next(named.glob("*_target_named.json"), None)
            if f:
                gt = json.loads(f.read_text(encoding="utf-8"))
        gt_passives = 0
        gt_in_printed = 0
        if gt:
            printed_strings = {t[0].upper() for t in printed}
            for key, comp in gt["components"].items():
                if str(comp.get("type")) not in {"r", "c", "l", "d", "led"}:
                    continue
                gt_passives += 1
                if key.upper() in printed_strings:
                    gt_in_printed += 1
        rows.append({"case": folder.name, "passives": len(passives), "printed_refs": len(printed),
                     "passives_with_local_ref": local, "gt_passives": gt_passives,
                     "gt_in_drawing": gt_in_printed})

    total_p = sum(r["passives"] for r in rows)
    total_local = sum(r["passives_with_local_ref"] for r in rows)
    total_gt = sum(r["gt_passives"] for r in rows)
    total_gt_in = sum(r["gt_in_drawing"] for r in rows)
    print(f"cases: {len(rows)}")
    print(f"passive symbols (R/C/L/D/LED): {total_p}")
    print(f"  with a reference-shaped string within {NEAR:.0f}px: {total_local} "
          f"({total_local/max(1,total_p):.1%})")
    print(f"published GT passives: {total_gt}")
    print(f"  whose key appears verbatim among the drawing's texts: {total_gt_in} "
          f"({total_gt_in/max(1,total_gt):.1%})")
    ratios = [r["passives_with_local_ref"] / r["passives"] for r in rows if r["passives"] >= 5]
    buckets = Counter()
    for value in ratios:
        buckets["0%" if value == 0 else "1-25%" if value <= .25 else "26-50%" if value <= .5
                else "51-75%" if value <= .75 else "76-100%"] += 1
    print(f"\npages with >=5 passives: {len(ratios)}; median fraction with a local reference: "
          f"{statistics.median(ratios):.1%}")
    for label in ("0%", "1-25%", "26-50%", "51-75%", "76-100%"):
        print(f"  {label:8s} {buckets[label]:3d} pages")
    print("\nworst pages (fewest printed references):")
    for row in sorted([r for r in rows if r["passives"] >= 10],
                      key=lambda r: r["passives_with_local_ref"] / r["passives"])[:8]:
        print(f"  {row['case']}: {row['passives_with_local_ref']}/{row['passives']} passives "
              f"with a local ref, {row['printed_refs']} ref-shaped texts on the whole drawing")


if __name__ == "__main__":
    sys.exit(main())
