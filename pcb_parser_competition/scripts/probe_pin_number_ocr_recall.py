"""Feasibility probe: does our own OCR already read the printed pin numbers?

The annotations record, per pin, a pointer to the text object that IS its number
(`pinNumberByOCR` -> texts[tid]).  That makes it possible to ask two different
questions about the same page:

  read   - did our OCR produce the correct digit string near that text object?
           If not, no downstream binder can ever recover the number.
  bindable - did our OCR produce it near the pin tip, i.e. inside the band a binder
           would search?  If `read` is high but `bindable` is low, the fix is cheap.

Runs on the cached OCR output of the public Golden pages, so it costs seconds.

Run:  python scripts/probe_pin_number_ocr_recall.py
"""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from evaluate_golden_cases import GT_DIR  # noqa: E402

BASE = Path(r"D:/pcb_unet_v2/golden_eval")
MANIFEST = ROOT / "data" / "manifest.jsonl"


def centre(box):
    return (float(box[0]) + float(box[2])) / 2.0, (float(box[1]) + float(box[3])) / 2.0


def distance(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def has_text_near(tokens, target, expected, radius):
    for token in tokens:
        if str(token["text"]).strip() != expected:
            continue
        if distance(centre(token["bbox"]), target) <= radius:
            return True
    return False


def main():
    manifest = {json.loads(line)["case_id"]: json.loads(line)
                for line in MANIFEST.read_text(encoding="utf-8").splitlines()}
    totals = Counter()
    per_case = []
    for folder in sorted(GT_DIR.iterdir()):
        if not folder.is_dir():
            continue
        case = folder.name.split("-")[0]
        texts_path = BASE / case / "pin_text" / "texts.json"
        if not texts_path.exists() or case not in manifest:
            continue
        tokens = json.loads(texts_path.read_text(encoding="utf-8"))["texts"]
        raw = json.loads(Path(manifest[case]["raw_annotation_path"]).read_text(encoding="utf-8"))
        text_objects = raw.get("texts") or {}
        labels = 0
        read = 0
        bindable = 0
        not_read_examples = []
        for component in (raw.get("components") or {}).values():
            for pin in (component.get("pins") or {}).values():
                pointer = pin.get("pinNumberByOCR")
                item = text_objects.get(pointer) if pointer else None
                if not item:
                    continue
                expected = str(item.get("content") or "").strip()
                box = item.get("location") or {}
                if not expected or not all(k in box for k in ("x1", "y1", "x2", "y2")):
                    continue
                tip = pin.get("tip")
                if not tip:
                    continue
                labels += 1
                glyph = centre([box["x1"], box["y1"], box["x2"], box["y2"]])
                pin_point = (float(tip[0]), float(tip[1]))
                # our OCR token geometry lives in image coordinates, same as tips
                glyph_point = (glyph[0], glyph[1])
                got_read = has_text_near(tokens, glyph_point, expected, 18)
                got_bind = has_text_near(tokens, pin_point, expected, 45)
                read += got_read
                bindable += got_bind
                if not got_read and len(not_read_examples) < 3:
                    nearby = sorted(((distance(centre(t["bbox"]), glyph_point), str(t["text"]))
                                     for t in tokens))[:4]
                    not_read_examples.append((case, expected, [(round(d, 1), t) for d, t in nearby]))
        totals["labels"] += labels
        totals["read"] += read
        totals["bindable"] += bindable
        per_case.append((case, labels, read, bindable, not_read_examples))

    print(f"{'case':<8}{'labelled':>9}{'our OCR read it':>17}{'findable at pin':>17}")
    for case, labels, read, bindable, _ in per_case:
        share = 100 * read / labels if labels else 0.0
        bind_share = 100 * bindable / labels if labels else 0.0
        print(f"{case:<8}{labels:>9}{read:>10} {share:5.1f}%{bindable:>10} {bind_share:5.1f}%")
    labels = totals["labels"]
    print()
    print(f"pins with a labelled number : {labels}")
    print(f"  our OCR produced it near the glyph : {totals['read']} "
          f"({100 * totals['read'] / max(1, labels):.1f}%)")
    print(f"  and it lands inside the pin band   : {totals['bindable']} "
          f"({100 * totals['bindable'] / max(1, labels):.1f}%)")
    print()
    print("examples our OCR missed (expected vs nearest tokens we did produce):")
    for case, expected, nearby in (per_case[0][4] if per_case else []):
        print(f"  {case}: expected {expected!r} nearest {nearby}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
