"""Ceiling check: is the GT designator string even present in the OCR output?

If a GT key never appears among the OCR tokens of that page (any view), then no
binding improvement can ever score it - the page's GT numbering differs from the
printed text. This separates "OCR recall work" from "dataset numbering mismatch".
"""
import json
import sys
from pathlib import Path

GT_ROOT = Path(r"C:\Users\Pc\Desktop\新建文件夹\赛题六公开数据集\赛题六公开数据集\200_train_cases")
LOG = Path(r"D:\pcb_v5_batch\output\logs")
PRED = Path(r"D:\pcb_v5_batch\preds")
CASES = ["0001", "0002", "0018", "0029", "0080", "0102"]
POWER = {"gnd", "v", "vcc", "vdd", "voltage_source", "net_input", "net_output",
         "net_short", "net_bidirection"}


def main():
    tot = {"gt": 0, "visible": 0, "hit": 0}
    for case in CASES:
        gt = json.loads(next((GT_ROOT / case).glob("*_target_named.json")).read_text(encoding="utf-8"))
        ocr = json.loads((LOG / f"{case}_ocr_texts.json").read_text(encoding="utf-8"))
        pred = json.loads((PRED / f"{case}_Pred.json").read_text(encoding="utf-8"))
        # every spelling ever observed, any view
        seen = set()
        for row in ocr["texts"]:
            for t in [row["text"]] + [o["text"] for o in row.get("observations", [])]:
                for part in str(t).replace("/", " ").replace("(", " ").replace(")", " ").split():
                    seen.add(part)
                    seen.add(part.upper())
        comps = {k: v for k, v in gt["components"].items()
                 if str(v.get("type")) not in POWER and not k.startswith("∅")}
        visible = [k for k in comps if k in seen or k.upper() in seen]
        hit = [k for k in visible if k in pred["components"]]
        tot["gt"] += len(comps)
        tot["visible"] += len(visible)
        tot["hit"] += len(hit)
        print(f"\n=== {case}: GT ordinary {len(comps)}, designator string visible in OCR {len(visible)} "
              f"({len(visible)/max(1,len(comps)):.0%}), of those bound to a component {len(hit)}")
        invisible = [k for k in comps if k not in visible]
        if invisible:
            print("   NOT visible:", " ".join(sorted(invisible)[:30]))
        unbound = [k for k in visible if k not in pred["components"]]
        if unbound:
            print("   visible but unbound:", " ".join(sorted(unbound)[:30]))
    print("\n==== TOTAL ====")
    print(f"  GT ordinary designators: {tot['gt']}")
    print(f"  visible in OCR output: {tot['visible']} ({tot['visible']/tot['gt']:.1%})")
    print(f"  bound to a predicted component: {tot['hit']} ({tot['hit']/tot['gt']:.1%})")
    print(f"  ceiling if every visible designator were bound: {tot['visible']/tot['gt']:.1%}")


if __name__ == "__main__":
    sys.exit(main())
