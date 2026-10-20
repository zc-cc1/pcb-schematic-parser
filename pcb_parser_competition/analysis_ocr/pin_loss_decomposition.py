"""Where does each missing pin die? Decomposes the pin loss on the 13 real cases.

For every GT pin of an IC-scope component, walk the same chain the pipeline walks and
assign exactly one cause:

  KEY_MISS          the GT component key never appeared in our components, so the pin
                    item cannot match at all (designator problem, not text)
  NO_TIP            no learned tip within 15 px of the GT point (U-Net recall)
  GATED_<status>    a tip exists, but that pin was not exported; the status says why
                    (number_status from pin_text: no_reliable_text / ambiguous_text /
                     text_used_by_other_pin / owner_unresolved / duplicate_...)
  WRONG_NUMBER      exported on the right tip, wrong pin_<number>
  WRONG_NAME        right tip and number, pinname differs
  OK                every check passes

Inputs: final preds, the kept per-case work dir, and the released open_GT.
"""
import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

BASE = Path(r"D:\pcb5_submission")
sys.path.insert(0, str(BASE / "src"))

TIP = 5.0
NEAR = 15.0
IC_TYPES = {"box", "ic"}


def norm_type(value):
    t = str(value or "").strip().lower()
    return "box" if t in IC_TYPES else t


def load_work(case_dir):
    owners = sorted(case_dir.glob("**/ownership/pin_ownership.json"))
    texts = sorted(case_dir.glob("**/upstream/pin_text/pin_text_result.json"))
    ownership = json.loads(owners[0].read_text(encoding="utf-8")) if owners else None
    pin_text = json.loads(texts[0].read_text(encoding="utf-8")) if texts else ownership
    return ownership, pin_text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--gt-root", required=True)
    ap.add_argument("--glob", default="*_target_named.json")
    args = ap.parse_args()
    pred_dir, work_root, gt_root = Path(args.pred), Path(args.work), Path(args.gt_root)

    tally = Counter()
    rows = []
    for folder in sorted((p for p in gt_root.iterdir() if p.is_dir()), key=lambda p: (len(p.name), p.name)):
        case = folder.name
        gt_file = next(folder.glob(args.glob), None)
        pred_file = pred_dir / f"{case}_Pred.json"
        if gt_file is None or not pred_file.exists():
            continue
        gt = json.loads(gt_file.read_text(encoding="utf-8"))
        pred = json.loads(pred_file.read_text(encoding="utf-8"))

        gt_pins = [(comp, pin, want) for comp, group in (gt.get("pins") or {}).items()
                   if norm_type((gt["components"].get(comp) or {}).get("type")) == "box"
                   for pin, want in group.items()]
        if not gt_pins:
            continue
        # per-case work dir: the one whose ownership image matches this case's GT png
        work = None
        for candidate in work_root.iterdir():
            if not candidate.is_dir():
                continue
            owners = sorted(candidate.glob("**/ownership/pin_ownership.json"))
            if not owners:
                continue
            try:
                doc = json.loads(owners[0].read_text(encoding="utf-8"))
            except Exception:
                continue
            png = next(folder.glob("*.png"), None)
            if png is None:
                continue
            import hashlib
            if doc.get("image_sha256") == hashlib.sha256(png.read_bytes()).hexdigest():
                work = candidate
                break
        if work is None:
            print(f"  (no work dir for case {case})")
            continue
        ownership, pin_text = load_work(work)
        candidates = ownership["pin_candidates"] if ownership else []
        # ownership is recorded in top-left pixel coordinates; the submission (and the
        # GT) use bottom-left. Convert once, here, or every distance is meaningless.
        height = float((ownership or {}).get("size", [0, 0])[1])
        for candidate in candidates:
            x, y = candidate["point"]
            candidate["point"] = [float(x), height - float(y)]
        status_of = {p["candidate_id"]: (p.get("number_status"), p.get("pinname_status"))
                     for p in (pin_text or {}).get("pin_candidates", [])}

        ours = []
        for owner, group in (pred.get("pins") or {}).items():
            for pin_key, pin in group.items():
                ours.append((float(pin["point"]["x"]), float(pin["point"]["y"]),
                             owner, pin_key[4:], pin.get("pinname")))
        used = set()
        case_counts = Counter()
        for comp, pin_key, want in gt_pins:
            point = (float(want["point"]["x"]), float(want["point"]["y"]))
            if comp not in pred["components"]:
                case_counts["KEY_MISS"] += 1
                continue
            if not any(math.hypot(float(c["point"][0]) - point[0], float(c["point"][1]) - point[1]) <= NEAR
                       for c in candidates):
                # distinguish "no tip at all" from "a tip exists but sits too far away"
                nearest = min((math.hypot(float(c["point"][0]) - point[0],
                                          float(c["point"][1]) - point[1]) for c in candidates),
                              default=1e9)
                bucket = ("NONE" if nearest > 1e8 else
                          "15-30" if nearest <= 30 else
                          "30-60" if nearest <= 60 else ">60")
                case_counts[f"NO_TIP({bucket})"] += 1
                continue
            best = None
            for index, (x, y, _, _, _) in enumerate(ours):
                if index in used:
                    continue
                d = math.hypot(x - point[0], y - point[1])
                if d <= TIP and (best is None or d < best[0]):
                    best = (d, index)
            if best is None:
                # a tip exists but nothing was exported on it; ask the text stage why
                near = min((c for c in candidates),
                           key=lambda c: math.hypot(float(c["point"][0]) - point[0],
                                                    float(c["point"][1]) - point[1]))
                status = (status_of.get(near["candidate_id"]) or ("unknown", "unknown"))[0]
                case_counts[f"GATED_{status}"] += 1
                continue
            used.add(best[1])
            _, _, owner, number, pinname = ours[best[1]]
            if number != pin_key[4:]:
                case_counts["WRONG_NUMBER"] += 1
            elif (pinname or None) != (want.get("pinname") or None):
                case_counts["WRONG_NAME"] += 1
            else:
                case_counts["OK"] += 1
        rows.append((case, len(gt_pins), case_counts))
        tally.update(case_counts)
        print(f"{case:>3}: {len(gt_pins):3d} pins -> " +
              ", ".join(f"{k}={v}" for k, v in sorted(case_counts.items(), key=lambda kv: -kv[1])))

    print("\n== TOTAL over the 13 cases ==")
    total = sum(tally.values())
    for key, value in sorted(tally.items(), key=lambda kv: -kv[1]):
        print(f"  {key:34s} {value:4d}  {value/max(1,total):6.1%}")
    print(f"  {'TOTAL':34s} {total:4d}")
    text_like = sum(v for k, v in tally.items() if k.startswith("GATED_")) + \
        tally["WRONG_NUMBER"] + tally["WRONG_NAME"]
    no_tip = sum(v for k, v in tally.items() if k.startswith("NO_TIP"))
    print("\n  pin loss attributed to")
    print(f"    geometry / learned tips   {no_tip:4d}  {no_tip/max(1,total):6.1%}")
    print(f"      of which no tip within 60px        {sum(v for k, v in tally.items() if '>60' in k):4d}")
    print(f"      of which a tip exists at 15-60px   {sum(v for k, v in tally.items() if '15-30' in k or '30-60' in k):4d}")
    print(f"    component key never matched {tally['KEY_MISS']:4d}  {tally['KEY_MISS']/max(1,total):6.1%}")
    print(f"    pin text / OCR              {text_like:4d}  {text_like/max(1,total):6.1%}")
    print(f"    fully correct               {tally['OK']:4d}  {tally['OK']/max(1,total):6.1%}")


if __name__ == "__main__":
    sys.exit(main())
