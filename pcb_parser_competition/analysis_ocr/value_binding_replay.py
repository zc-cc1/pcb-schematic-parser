"""Replay the component binder on the 13 released cases and score its `value` field.

Runs `component_text.bind_components` on the cached detections + OCR rows of each kept
work directory, so a binding rule can be A/B'd in seconds instead of a 10 minute run.
Score = value-exact on components whose key matches GT.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

WORK = Path(r"D:\pcb_final\out\work")
GT = Path(r"D:\pcb_final\gt")
BASE_CONFIG = Path(r"D:\pcb5_submission\configs\baseline_v3_adopted.json")

sys.path.insert(0, r"D:\pcb5_submission\src")
from pcb_parser import component_text as CT  # noqa: E402


def case_work(case, digest):
    for folder in WORK.iterdir():
        for owner in folder.glob("**/ownership/pin_ownership.json"):
            if json.loads(owner.read_text(encoding="utf-8")).get("image_sha256") == digest:
                return folder
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=BASE_CONFIG)
    ap.add_argument("--set", action="append", default=[], help="binding option overrides, key=value")
    ap.add_argument("--label", default="run")
    args = ap.parse_args()

    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    binding = cfg.setdefault("component_text", {}).setdefault("binding", {})
    for item in args.set:
        key, _, raw = item.partition("=")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        binding[key] = value

    total_gt = total_ok = total_bound = total_wrong = 0
    per_case = []
    for folder in sorted((p for p in GT.iterdir() if p.is_dir()), key=lambda p: (len(p.name), p.name)):
        case = folder.name
        gt = json.loads(next(folder.glob("*_target_named.json")).read_text(encoding="utf-8"))
        png = next(folder.glob("*.png"))
        work = case_work(case, hashlib.sha256(png.read_bytes()).hexdigest())
        if work is None:
            continue
        ownership = json.loads(next(work.glob("**/ownership/pin_ownership.json")).read_text(encoding="utf-8"))
        texts = json.loads(next(work.glob("**/upstream/upstream/texts.json")).read_text(encoding="utf-8"))["texts"]
        detections = [{"class_name": c["class_name"], "confidence": c.get("confidence", 1.0),
                       "bbox": c["bbox_top_left"]} for c in ownership["components"]]
        _, records, _ = CT.bind_components(detections, texts, ownership["size"][0], ownership["size"][1], cfg)

        gt_by_key = {k: v for k, v in gt["components"].items()}
        exact = bound = wrong = gt_with_value = 0
        for record in records:
            key = record.get("designator") or record.get("component_id")
            want = gt_by_key.get(key)
            if want is None:
                continue
            want_value = want.get("value")
            if want_value is None:
                continue
            gt_with_value += 1
            got = record.get("value")
            if got is None:
                continue
            bound += 1
            if str(got) == str(want_value):
                exact += 1
            else:
                wrong += 1
        per_case.append((case, exact, bound, wrong, gt_with_value))
        total_gt += gt_with_value
        total_ok += exact
        total_bound += bound
        total_wrong += wrong

    print(f"{'case':>4} {'exact':>6} {'bound':>6} {'wrong':>6} {'gt':>4}")
    for case, exact, bound, wrong, gt_count in per_case:
        print(f"{case:>4} {exact:>6} {bound:>6} {wrong:>6} {gt_count:>4}")
    print(f"\n[{args.label}] value exact {total_ok}/{total_gt} = {total_ok/max(1,total_gt):.1%}"
          f"   bound {total_bound}, wrong {total_wrong},"
          f"  precision {total_ok/max(1,total_bound):.1%}")


if __name__ == "__main__":
    sys.exit(main())
