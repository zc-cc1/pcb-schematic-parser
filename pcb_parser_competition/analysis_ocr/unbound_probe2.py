"""Refined triage of every GT ordinary designator that never became a pred key.

Classes per GT key:
  BOUND_OTHER      a component key exists (shouldn't happen - filtered
  LABEL_ON_SYMBOL  exact GT string sits within NEAR px of the GT box -> our binder lost it
  NEAR_OTHER_REF   another reference-shaped token sits on the symbol (digit/prefix
                   disagreement with GT, or the page is renumbered)
  NEAR_TEXT_ONLY   only non-reference text on the symbol (glyph never recognised)
  NO_TEXT          nothing at all within NEAR px
"""
import json
import math
import re
import sys
from pathlib import Path
from PIL import Image

GT_ROOT = Path(r"C:\Users\Pc\Desktop\新建文件夹\赛题六公开数据集\赛题六公开数据集\200_train_cases")
LOG = Path(r"D:\pcb_v5_batch\output\logs")
PRED = Path(r"D:\pcb_v5_batch\preds")
CASES = ["0001", "0002", "0018", "0029", "0080", "0102"]
POWER = {"gnd", "v", "vcc", "vdd", "voltage_source", "net_input", "net_output",
         "net_short", "net_bidirection"}
REF = re.compile(r"^(?:IC|SW|TP|FB|FET|LED|JP|[RCLDQUXYJPFS])\d+$", re.I)
NEAR = 30.0


def gap(a, b):
    return math.hypot(max(a[0] - b[2], b[0] - a[2], 0), max(a[1] - b[3], b[1] - a[3], 0))


def main():
    tally = {}
    detail = {}
    for case in CASES:
        gt = json.loads(next((GT_ROOT / case).glob("*_target_named.json")).read_text(encoding="utf-8"))
        pred = json.loads((PRED / f"{case}_Pred.json").read_text(encoding="utf-8"))
        ocr = json.loads((LOG / f"{case}_ocr_texts.json").read_text(encoding="utf-8"))
        H = Image.open(next((GT_ROOT / case).glob("*.png"))).size[1]
        texts = ocr["texts"]
        have = set(pred["components"])
        rows = []
        for key, comp in gt["components"].items():
            if str(comp.get("type")) in POWER or key.startswith("∅") or key in have:
                continue
            x1, y1, x2, y2 = comp["bbox"]
            box = [x1, H - y2, x2, H - y1]
            near = [(gap(box, t["bbox"]), t) for t in texts]
            near = [p for p in near if p[0] <= NEAR]
            exact = [t for d, t in near if t["text"].strip().upper() == key.upper()]
            refs = [(d, t["text"], t["confidence"]) for d, t in near
                    if REF.match(t["text"].strip())]
            if exact:
                cls = "LABEL_ON_SYMBOL"
            elif refs:
                cls = "NEAR_OTHER_REF"
            elif near:
                cls = "NEAR_TEXT_ONLY"
            else:
                cls = "NO_TEXT"
            tally[cls] = tally.get(cls, 0) + 1
            ordered = sorted(near, key=lambda p: p[0])
            detail.setdefault(cls, []).append((case, key, comp.get("type"),
                                               sorted(refs)[:3] if refs else
                                               [(round(d, 1), t["text"]) for d, t in ordered[:3]]))
            rows.append((key, cls, sorted(refs)[:2] if refs else [(round(d, 1), t["text"]) for d, t in ordered[:2]]))
        print(f"\n=== {case} ({len(rows)} unbound GT keys)")
        for key, cls, ev in rows:
            print(f"  {key:8s} {cls:16s} {ev}")
    print("\n==== TALLY ====")
    for k, v in sorted(tally.items(), key=lambda kv: -kv[1]):
        print(f"  {k:16s} {v}")
    print("\n==== grouped evidence ====")
    for cls, items in detail.items():
        print(f"\n-- {cls} ({len(items)})")
        for case, key, typ, ev in items:
            print(f"  {case}/{key:8s} {str(typ):12s} {ev}")


if __name__ == "__main__":
    sys.exit(main())
