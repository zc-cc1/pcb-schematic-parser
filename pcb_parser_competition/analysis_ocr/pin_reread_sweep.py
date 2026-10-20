"""Sweep all 13 cases: how many of the still-missing pin numbers does a 2x ROI re-read recover?

For every GT IC pin that has no exported pin nearby, crop the small band just outside
the symbol edge on that pin's row, run the recogniser on the crop at 1x / 2x / 4x, and
compare against the GT number. Reports the recovery rate, so the value of a targeted
re-OCR pass is known before any code changes.
"""
import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

BASE = Path(r"D:\pcb5_submission")
sys.path.insert(0, str(BASE / "src"))
from pcb_parser.pin_text_ocr import create_engine  # noqa: E402

DIGITS = re.compile(r"(\d{1,4})")


def clean(text, side):
    """Strip the pin lead / stray punctuation that OCR glues onto the digit."""
    text = str(text or "").strip()
    if side in ("left", "right"):
        match = DIGITS.search(text)
        return match.group(1) if match and match.end() - match.start() <= 4 else ""
    match = DIGITS.search(text)
    return match.group(1) if match else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--gt-root", required=True)
    ap.add_argument("--pad", type=float, default=14.0)
    args = ap.parse_args()
    pred_dir, gt_root = Path(args.pred), Path(args.gt_root)
    engine, models = create_engine({"onnx_threads": 4})
    print("engine:", models["Rec"]["file"])

    tally = Counter()
    per_case = []
    for folder in sorted((p for p in gt_root.iterdir() if p.is_dir()), key=lambda p: (len(p.name), p.name)):
        case = folder.name
        pred_file = pred_dir / f"{case}_Pred.json"
        gt_file = next(folder.glob("*_target_named.json"), None)
        png = next(folder.glob("*.png"), None)
        if gt_file is None or not pred_file.exists() or png is None:
            continue
        gt = json.loads(gt_file.read_text(encoding="utf-8"))
        pred = json.loads(pred_file.read_text(encoding="utf-8"))
        rgb = cv2.cvtColor(cv2.imread(str(png)), cv2.COLOR_BGR2RGB)
        height = rgb.shape[0]
        exported = [(float(p["point"]["x"]), float(p["point"]["y"]))
                    for g in (pred.get("pins") or {}).values() for p in g.values()]
        case_stats = Counter()
        for comp, group in (gt.get("pins") or {}).items():
            if str((gt["components"].get(comp) or {}).get("type")) not in ("box", "ic"):
                continue
            box = gt["components"][comp]["bbox"]
            for pin_key, want in group.items():
                px, py = float(want["point"]["x"]), float(want["point"]["y"])
                if any(math.hypot(x - px, y - py) <= 5 for x, y in exported):
                    continue
                case_stats["missing"] += 1
                ty = height - py
                # which edge is this pin on?
                distances = {
                    "left": abs(px - box[0]), "right": abs(px - box[2]),
                    "bottom": abs(py - box[1]), "top": abs(py - box[3]),
                }
                side = min(distances, key=distances.get)
                if side in ("left", "right"):
                    x_edge = box[0] if side == "left" else box[2]
                    crop = [int(x_edge - args.pad * 2.2), int(ty - args.pad),
                            int(x_edge + 4), int(ty + args.pad)]
                else:
                    y_edge = height - box[1] if side == "bottom" else height - box[3]
                    crop = [int(px - args.pad), int(y_edge + args.pad * 2.2),
                            int(px + args.pad), int(y_edge - 4)]
                x1, y1, x2, y2 = (min(crop[0], crop[2]), min(crop[1], crop[3]),
                                  max(crop[0], crop[2]), max(crop[1], crop[3]))
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(rgb.shape[1], x2), min(rgb.shape[0], y2)
                if x2 - x1 < 4 or y2 - y1 < 6:
                    case_stats["crop_invalid"] += 1
                    continue
                patch = rgb[y1:y2, x1:x2]
                best = {}
                for scale in (1, 2, 4):
                    scaled = cv2.resize(patch, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                    result = engine(np.ascontiguousarray(scaled[:, :, ::-1]), use_det=False,
                                    use_cls=False, use_rec=True)
                    pairs = list(zip(result.txts or [], result.scores or []))
                    text, score = max(pairs, key=lambda r: r[1], default=("", 0.))
                    best[scale] = (text, float(score))
                want_number = pin_key[4:]
                case_stats["hit_1x_raw"] += best[1][0].strip() == want_number
                case_stats["hit_2x_raw"] += best[2][0].strip() == want_number
                case_stats["hit_2x_clean"] += clean(best[2][0], side) == want_number
                case_stats["hit_4x_clean"] += clean(best[4][0], side) == want_number
        per_case.append((case, case_stats))
        tally.update(case_stats)
        print(f"{case:>3}: missing={case_stats['missing']:3d}  "
              f"1x={case_stats['hit_1x_raw']:3d}  2x={case_stats['hit_2x_raw']:3d}  "
              f"2x+clean={case_stats['hit_2x_clean']:3d}  4x+clean={case_stats['hit_4x_clean']:3d}")

    missing = tally["missing"]
    print(f"\n== TOTAL ==")
    print(f"  pins missing from the export: {missing}")
    for key in ("hit_1x_raw", "hit_2x_raw", "hit_2x_clean", "hit_4x_clean"):
        print(f"  recovered by {key:12s}: {tally[key]:3d}  ({tally[key]/max(1,missing):.1%})")


if __name__ == "__main__":
    sys.exit(main())
