"""Offline replay of the designator/model/value binder on cached v5 artefacts.

Why: the 6-case v5 batch already contains every detection box and every OCR row
for those pages, so a binder change can be measured in ~1s instead of re-running
YOLO + OCR (~200-600s per page).

Input reconstruction
  detections  <- component_results/<case>_Pred.json   (pre-geometry payload, all
                 detections incl. UNRESOLVED_DET_*, type already in official words)
  class_name  <- inverted configs/export_vocabulary.json so bind_components sees
                 the detector class it expects for `class_mapping`
  texts       <- logs/<case>_ocr_texts.json
  image H     <- the public PNG

Caveat printed by --report: replay cannot reproduce detections that the deployed
pipeline dropped upstream (0029 exports no ordinary component at all), so absolute
scores are a lower bound; use it for A/B on the same case set only.
"""
import argparse
import json
import sys
from pathlib import Path

BASE = Path(r"D:\pcb5_submission")
sys.path.insert(0, str(BASE / "src"))

from pcb_parser import component_text  # noqa: E402
from PIL import Image  # noqa: E402

GT_ROOT = Path(r"C:\Users\Pc\Desktop\新建文件夹\赛题六公开数据集\赛题六公开数据集\200_train_cases")
BATCH = Path(r"D:\pcb_v5_batch")
CASES = ["0001", "0002", "0018", "0029", "0080", "0102"]
POWER = {"gnd", "v", "vcc", "vdd", "voltage_source"}


def load_configs():
    return json.loads((BASE / "configs" / "baseline_v3_adopted.json").read_text(encoding="utf-8"))


# Exported word -> internal kind, straight from adapter.TARGET_TYPE_MAP plus the
# power words component_export writes. Anything else is assumed to be already an
# internal kind or the detector class itself.
EXPORTED_TO_KIND = {"r": "resistor", "c": "capacitor", "l": "inductor", "d": "diode",
                    "box": "ic", "led": "led", "v": "vcc", "gnd": "gnd"}


def inverse_vocabulary(adopted):
    """internal kind -> a detector class that class_mapping sends to that kind."""
    mapping = adopted.get("class_mapping") or {}
    inv = {}
    for detector_class, kind in mapping.items():
        inv.setdefault(kind, detector_class)
    return inv


def build_case(case, adopted, inv):
    img_path = next((GT_ROOT / case).glob("*.png"))
    width, height = Image.open(img_path).size
    payload = json.loads((BATCH / "output" / "component_results" / f"{case}_Pred.json").read_text(encoding="utf-8"))
    mapping = adopted.get("class_mapping") or {}
    detections = []
    for key, comp in payload["components"].items():
        bbox = comp["bbox"]
        exported = comp.get("type")
        if exported in EXPORTED_TO_KIND:
            kind = EXPORTED_TO_KIND[exported]
            class_name = inv.get(kind, kind)
        elif exported in mapping:            # already a detector class
            class_name = exported
        else:                                # an internal kind such as 'vbus_in'
            class_name = inv.get(exported, exported)
        detections.append({
            "class_name": class_name,
            "confidence": 1.0,
            "bbox": [bbox[0], height - bbox[3], bbox[2], height - bbox[1]],
            "original_key": key,
        })
    texts = json.loads((BATCH / "output" / "logs" / f"{case}_ocr_texts.json").read_text(encoding="utf-8"))["texts"]
    cfg = dict(adopted)
    cfg["class_mapping"] = mapping
    return detections, texts, width, height, cfg


def keys_of(output):
    return {k for k in output["components"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="*", default=CASES)
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    adopted = load_configs()
    inv = inverse_vocabulary(adopted)
    print("inverse vocabulary sample:", dict(list(inv.items())[:8]))
    total = {"replay": 0, "actual": 0, "both": 0, "unresolved": 0}
    for case in args.cases:
        detections, texts, width, height, cfg = build_case(case, adopted, inv)
        output, records, diag = component_text.bind_components(detections, texts, width, height, cfg)
        replay = keys_of(output)
        actual_payload = json.loads((BATCH / "preds" / f"{case}_Pred.json").read_text(encoding="utf-8"))
        actual = set(actual_payload["components"])
        both = replay & actual
        unresolved = sum(1 for k in replay if k.startswith("UNRESOLVED"))
        total["replay"] += len(replay)
        total["actual"] += len(actual)
        total["both"] += len(both)
        total["unresolved"] += unresolved
        print(f"\n=== {case}: replay {len(replay)} keys ({unresolved} unresolved), actual {len(actual)}, "
              f"overlap {len(both)}")
        if args.report:
            only_actual = sorted(actual - replay)
            only_replay = sorted(replay - actual)
            print("   only in actual :", " ".join(only_actual[:20]))
            print("   only in replay :", " ".join(only_replay[:20]))
            print("   counts:", diag["counts"])
    print("\n==== totals ====", total)
    print("replay recall vs actual:", f"{total['both']/max(1,total['actual']):.1%}")


if __name__ == "__main__":
    sys.exit(main())
