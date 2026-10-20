"""Separate OCR recall from binding-rule quality for pin names.

The annotations ship a labelled pin-name source: `pin.pinNameByOCR` points at a text
object whose `content` is that pin's name, and the public reference files use exactly
that string as `pinname` (verified 20/20 on case 0042).  Feeding those texts into our
binder as if they were our own OCR isolates the two failure modes:

  * low "produced"  -> the binding rule rejects text that is provably the right answer
  * produced >> matched -> the rule binds the wrong text

Both are rule problems, not OCR problems, because the input text here is perfect.
Component geometry and pin ownership come from the annotation, so the geometric gates
are exercised as they are in production.

Run:  python scripts/probe_pin_name_binding.py [page_count]
"""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from pcb_parser.pin_text import bind_pin_text  # noqa: E402

MANIFEST = ROOT / "data" / "manifest.jsonl"
PAGES = int(sys.argv[1]) if len(sys.argv) > 1 else 40

CATEGORY_TYPES = {
    "gnd": "gnd", "c": "capacitor", "r": "resistor", "box": "ic", "v": "vcc",
    "pin": "pin_port", "net_input": "net_input", "d": "diode", "testpoint": "testpoint",
    "other": "other", "l": "inductor", "led": "led", "crystal": "crystal",
    "switch": "switch", "mosfet": "mosfet", "battary": "battery", "bjt": "bjt",
    "fuse": "fuse", "amp": "opamp", "net_bidirection": "net_bidirectional",
    "net_output": "net_output", "circle_header": "circle_header", "jumper": "jumper",
    "antenna": "antenna", "opt": "optocoupler", "motor": "motor",
}


def annotation_texts(raw):
    out = []
    for item in (raw.get("texts") or {}).values():
        content = str(item.get("content") or "").strip()
        box = item.get("location") or {}
        if not content or not all(k in box for k in ("x1", "y1", "x2", "y2")):
            continue
        bbox = [float(box["x1"]), float(box["y1"]), float(box["x2"]), float(box["y2"])]
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            continue
        out.append({"text": content, "confidence": 1.0, "bbox": bbox, "view": "annotation",
                    "polygon": [[bbox[0], bbox[1]], [bbox[2], bbox[1]],
                                [bbox[2], bbox[3]], [bbox[0], bbox[3]]]})
    return out


def build_ownership(raw):
    texts = raw.get("texts") or {}
    components, pins, expected = [], [], {}
    for index, component in enumerate((raw.get("components") or {}).values()):
        box = component.get("realBBox") or component.get("location") or {}
        if not all(k in box for k in ("x1", "y1", "x2", "y2")):
            continue
        kind = CATEGORY_TYPES.get(str(component.get("category") or "").strip().lower(), "other")
        components.append({"component_id": f"DET_{index:04d}", "detection_index": index,
                           "type": kind, "class_name": kind, "confidence": 1.0,
                           "bbox_top_left": [float(box["x1"]), float(box["y1"]),
                                             float(box["x2"]), float(box["y2"])]})
        for pin_id, pin in (component.get("pins") or {}).items():
            tip = pin.get("tip")
            if not tip:
                continue
            pins.append({"candidate_id": pin_id, "point": [float(tip[0]), float(tip[1])],
                         "confidence": 1.0, "owner": f"DET_{index:04d}",
                         "owner_status": "assigned", "owner_detection_index": index,
                         "number": None, "pinname": None, "number_status": "none",
                         "pinname_status": "none", "text_evidence": {}, "text_candidates": {}})
            pointer = pin.get("pinNameByOCR")
            item = texts.get(pointer) if pointer else None
            if item and str(item.get("content") or "").strip():
                expected[pin_id] = str(item["content"]).strip()
    return components, pins, expected


def main():
    rows = [json.loads(line) for line in MANIFEST.read_text(encoding="utf-8").splitlines()][:PAGES]
    cfg = json.loads((ROOT / "configs/pin_text_v1.json").read_text(encoding="utf-8"))
    totals = Counter()
    for row in rows:
        raw = json.loads(Path(row["raw_annotation_path"]).read_text(encoding="utf-8"))
        components, pins, expected = build_ownership(raw)
        if not expected:
            continue
        texts = annotation_texts(raw)
        ownership = {"size": [raw["canvas"]["width"], raw["canvas"]["height"]],
                     "components": components, "pin_candidates": pins}
        result, _ = bind_pin_text(ownership, texts, cfg, [])
        got = {pin["candidate_id"]: str(pin.get("pinname") or "").strip()
               for pin in result["pin_candidates"]}
        for pin_id, name in expected.items():
            totals["expected"] += 1
            produced = got.get(pin_id)
            if produced:
                totals["produced"] += 1
                if produced.upper() == name.upper():
                    totals["matched"] += 1
    total = totals["expected"]
    print(f"pages probed: {PAGES}")
    print(f"pins with a reference text name: {total}")
    print(f"  binder produced some name   : {totals['produced']} "
          f"({100 * totals['produced'] / max(1, total):.1f}%)")
    print(f"  and it equalled the reference: {totals['matched']} "
          f"({100 * totals['matched'] / max(1, total):.1f}%)")
    print()
    print("Input text is perfect here, so both numbers measure the RULE, not OCR.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
