"""Pre-upload gate for submission outputs.

The 6th official evaluation scored 0.0 although the container exited 0 and wrote
13 valid-looking files. The scorer's report showed why: every component carried
``Name = null`` and ``value = null``, ``pins`` was empty, and only 17 of ~62
designators were resolved. All four metrics collapsed to zero while the schema,
the type vocabulary and the bbox geometry were fine.

So schema validity is not the bar. This script checks the things that actually
decide the score, on any ``final_results`` directory:

  * every case file exists and has the three accepted top-level sections;
  * ``pins`` is non-empty (an empty pin block also makes the pin metric zero);
  * designators are resolved for a reasonable share of components - the official
    rubric matches ordinary components by key, so ``UNRESOLVED_DET_*`` can never
    be a true positive;
  * ordinary components carry ``Name`` or ``value`` (the rubric compares them
    exactly, and null against a real value is a mismatch);
  * bboxes are ordered, finite and inside the image.

Run it before every upload:

    .venv-gpu\\Scripts\\python.exe scripts\\check_submission_outputs.py `
        --pred <dir with *_Pred.json> --image-root data/standardized
"""
import argparse
import json
import sys
from pathlib import Path

POWER_TYPES = {"gnd", "v", "vcc", "vdd"}
LABEL_TYPES = {"vbus_in", "vbus_out", "vbus_bidirectional", "net_input", "net_output",
               "net_bidirection", "net_short", "pin", "testpoint"}


def image_size(image_root, case):
    for name in (f"{case}.png", f"{case}.jpg"):
        candidate = Path(image_root) / case / name
        if candidate.exists():
            from PIL import Image
            return Image.open(candidate).size
    return None


def check_case(path, image_root, thresholds):
    case = path.name[: -len("_Pred.json")]
    problems, notes = [], {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if set(payload) != {"components", "pins", "nets"}:
        return [f"{case}: top-level keys {sorted(payload)}"], notes

    comps = payload.get("components") or {}
    pins = payload.get("pins") or {}
    if not comps:
        return [f"{case}: zero components"], notes
    if not pins:
        problems.append(f"{case}: pins is empty (pin metric is 0 by construction)")

    unresolved = [k for k in comps if str(k).startswith("UNRESOLVED")]
    resolved_ratio = 1. - len(unresolved) / len(comps)
    notes["resolved_ratio"] = round(resolved_ratio, 3)
    if resolved_ratio < thresholds["min_resolved"]:
        problems.append(f"{case}: only {resolved_ratio:.0%} designators resolved "
                        f"(< {thresholds['min_resolved']:.0%}); unresolved keys cannot match")

    ordinary = {k: v for k, v in comps.items() if str(v.get("type")) not in POWER_TYPES}
    if ordinary:
        filled = sum(1 for v in ordinary.values() if v.get("Name") or v.get("value"))
        ratio = filled / len(ordinary)
        notes["field_ratio"] = round(ratio, 3)
        if ratio < thresholds["min_fields"]:
            problems.append(f"{case}: only {ratio:.0%} ordinary components carry Name or value "
                            f"(< {thresholds['min_fields']:.0%}); the rubric compares them exactly")
        label_like = sum(1 for v in ordinary.values() if str(v.get("type")) in LABEL_TYPES)
        notes["label_like"] = label_like

    size = image_size(image_root, case)
    for key, comp in comps.items():
        bbox = comp.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            problems.append(f"{case}/{key}: malformed bbox {bbox}")
            continue
        try:
            x1, y1, x2, y2 = (float(v) for v in bbox)
        except (TypeError, ValueError):
            problems.append(f"{case}/{key}: non-numeric bbox {bbox}")
            continue
        if not all(map(lambda v: v == v and abs(v) != float("inf"), (x1, y1, x2, y2))):
            problems.append(f"{case}/{key}: non-finite bbox {bbox}")
        elif not (x1 < x2 and y1 < y2):
            problems.append(f"{case}/{key}: unordered bbox {bbox}")
        elif size and not (0 <= x1 and 0 <= y1 and x2 <= size[0] and y2 <= size[1]):
            problems.append(f"{case}/{key}: bbox {bbox} outside image {size}")
    return problems, notes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", type=Path, required=True, help="directory holding *_Pred.json")
    ap.add_argument("--image-root", type=Path, help="directory of <case>/<case>.png for bounds checks")
    ap.add_argument("--min-resolved", type=float, default=0.50)
    ap.add_argument("--min-fields", type=float, default=0.80)
    args = ap.parse_args()

    thresholds = {"min_resolved": args.min_resolved, "min_fields": args.min_fields}
    files = sorted(args.pred.glob("*_Pred.json"))
    if not files:
        print(f"FAIL: no *_Pred.json in {args.pred}")
        return 1

    print(f"{'case':>6} {'comps':>6} {'pins':>5} {'resolved':>9} {'fields':>7} {'label':>6}")
    all_problems = []
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        problems, notes = check_case(path, args.image_root, thresholds)
        all_problems += problems
        case = path.name[: -len("_Pred.json")]
        print(f"{case:>6} {len(payload.get('components') or {}):>6} "
              f"{len(payload.get('pins') or {}):>5} "
              f"{notes.get('resolved_ratio', float('nan')):>9.0%} "
              f"{notes.get('field_ratio', float('nan')):>7.0%} "
              f"{notes.get('label_like', 0):>6}")

    print()
    if all_problems:
        for problem in all_problems[:40]:
            print(f"FAIL {problem}")
        if len(all_problems) > 40:
            print(f"... and {len(all_problems) - 40} more")
        print(f"\n{len(all_problems)} problem(s) across {len(files)} case(s)")
        return 1
    print(f"PASS: {len(files)} case(s) clear the pre-upload gates "
          f"(pins present, designators ≥{thresholds['min_resolved']:.0%}, "
          f"fields ≥{thresholds['min_fields']:.0%})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
