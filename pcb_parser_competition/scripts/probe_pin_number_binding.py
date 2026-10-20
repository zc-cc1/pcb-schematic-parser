"""Measure pin-number binding quality against the labelled annotation numbers.

`pin.pinNumberByOCR` points at a text object whose `content` is that pin's number, and
the reference uses the same strings, so this is a clean gold standard over ~7700 pins
across the 300 pages.

Two numbers are reported, and the gap between them is the point:
  produced - the rule found *some* number for the pin
  matched  - that number equalled the labelled one

Input text is the annotation's own, i.e. perfect OCR, so a low `produced` means the
geometric bands reject legitimate text and a low `matched` means the global assignment
picks the wrong neighbour.

Run:  python scripts/probe_pin_number_binding.py [page_count]
"""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from pcb_parser.pin_text import bind_pin_text  # noqa: E402
from probe_pin_name_binding import build_ownership, annotation_texts  # noqa: E402

MANIFEST = ROOT / "data" / "manifest.jsonl"
PAGES = int(sys.argv[1]) if len(sys.argv) > 1 else 40


def expected_pin_numbers(raw):
    texts = raw.get("texts") or {}
    out = {}
    for component in (raw.get("components") or {}).values():
        for pin_id, pin in (component.get("pins") or {}).items():
            pointer = pin.get("pinNumberByOCR")
            item = texts.get(pointer) if pointer else None
            if item and str(item.get("content") or "").strip():
                out[pin_id] = str(item["content"]).strip()
    return out


def main():
    rows = [json.loads(line) for line in MANIFEST.read_text(encoding="utf-8").splitlines()][:PAGES]
    cfg = json.loads((ROOT / "configs/pin_text_v1.json").read_text(encoding="utf-8"))
    totals = Counter()
    per_case = []
    for row in rows:
        raw = json.loads(Path(row["raw_annotation_path"]).read_text(encoding="utf-8"))
        expected = expected_pin_numbers(raw)
        if not expected:
            continue
        components, pins, _ = build_ownership(raw)
        texts = annotation_texts(raw)
        ownership = {"size": [raw["canvas"]["width"], raw["canvas"]["height"]],
                     "components": components, "pin_candidates": pins}
        result, _ = bind_pin_text(ownership, texts, cfg, [])
        got = {pin["candidate_id"]: str(pin.get("number") or "").strip()
               for pin in result["pin_candidates"]}
        produced = matched = 0
        for pin_id, number in expected.items():
            totals["expected"] += 1
            if got.get(pin_id):
                produced += 1
                totals["produced"] += 1
                if got[pin_id].upper() == number.upper():
                    matched += 1
                    totals["matched"] += 1
        per_case.append((row["case_id"], len(expected), produced, matched))
    total = totals["expected"]
    print(f"pages probed: {PAGES}")
    print(f"pins with a labelled number: {total}")
    print(f"  rule produced a number : {totals['produced']} "
          f"({100 * totals['produced'] / max(1, total):.1f}%)")
    print(f"  and it was correct     : {totals['matched']} "
          f"({100 * totals['matched'] / max(1, total):.1f}%)")
    print()
    print("input text is perfect here, so both numbers measure the RULE")
    print()
    worst = sorted(per_case, key=lambda r: (r[3] / r[1] if r[1] else 1))[:8]
    print(f"  {'case':<8}{'labelled':>9}{'produced':>9}{'correct':>9}")
    for case, labelled, produced, matched in worst:
        print(f"  {case:<8}{labelled:>9}{produced:>9}{matched:>9}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
