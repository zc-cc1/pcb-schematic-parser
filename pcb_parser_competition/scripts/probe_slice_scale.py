"""Slice-size sweep for detection recall (CPU, no GPU needed).

Where the misses are: on the 59-page validation set, 89% of the missed components are
smaller than 64 px (138 below 32 px), and 40% of them are pure misses - nothing
detected at all. The current production setting slices at 1024 px with imgsz 1536,
which magnifies a 3300 px page only ~1.5x; a 512 px slice magnifies ~3x.

This probe sweeps the slice size at a fixed confidence and counts, under the official
20 px corner rule, how many components are found - overall and by size bucket. The
existing evaluation entry point hardcodes cuda:0, hence a separate probe that runs on
CPU while the GPU is busy training the U-Net.

Reads ground truth; inference never does.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
from PIL import Image

from evaluate_official_component import corner_error, load_gt_case

POWER = {"ground", "voltage_source"}


def size_bucket(box):
    side = max(box[2] - box[0], box[3] - box[1])
    if side < 32:
        return "tiny"
    if side < 64:
        return "small"
    if side < 128:
        return "medium"
    return "large"


def match(gts, preds):
    """One-to-one, same class, every corner within 20 px (the official rule)."""
    cand = []
    for gi, g in enumerate(gts):
        for pi, p in enumerate(preds):
            if p["class_name"] != g["class_name"]:
                continue
            err = corner_error(p["bbox"], g["bbox"])
            if err <= 20.0:
                cand.append((err, gi, pi))
    cand.sort()
    ug, up = set(), set()
    for err, gi, pi in cand:
        if gi in ug or pi in up:
            continue
        ug.add(gi)
        up.add(pi)
    return ug, up


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, default=ROOT / "weights/detector_dataset300_v3_nano.pt")
    ap.add_argument("--config", type=Path, default=ROOT / "configs/baseline_v3.json")
    ap.add_argument("--manifest", type=Path, default=ROOT / "data/detection_v1/manifest.jsonl")
    ap.add_argument("--gt-root", type=Path,
                    default=ROOT.parent / "赛题六公开数据集" / "赛题六公开数据集" / "200_train_cases")
    ap.add_argument("--slices", nargs="+", type=int, default=[1024, 768, 512])
    ap.add_argument("--limit", type=int, default=20, help="how many val pages to use")
    ap.add_argument("--step", type=int, default=1, help="take every Nth val page")
    ap.add_argument("--overlap", type=float, default=0.25)
    args = ap.parse_args()

    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction

    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    imgsz = cfg["detector"]["imgsz"]
    conf = cfg["detector"]["confidence"]

    rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines()]
    val = [r["case_id"] for r in rows if r.get("split") == "val"]
    have_gt = {p.name for p in args.gt_root.iterdir() if p.is_dir()}
    cases = [c for c in val if c in have_gt][::args.step][:args.limit]
    print(f"cases={len(cases)} imgsz={imgsz} conf={conf} overlap={args.overlap}")

    model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics", model_path=str(args.weights),
        confidence_threshold=conf, device="cpu", image_size=imgsz)

    results = {}
    for case in cases:
        image_path = ROOT / "data/standardized" / case / f"{case}.png"
        image = Image.open(image_path).convert("RGB")
        rgb = np.asarray(image)
        gt_file = next((args.gt_root / case).glob("*_target_named.json"))
        gts = load_gt_case(gt_file, image.height)
        for slice_size in args.slices:
            out = get_sliced_prediction(
                rgb, model, slice_height=slice_size, slice_width=slice_size,
                overlap_height_ratio=args.overlap, overlap_width_ratio=args.overlap,
                postprocess_match_metric="IOS", postprocess_match_threshold=0.45,
                postprocess_type="GREEDYNMM", verbose=0)
            preds = [{"class_name": p.category.name,
                      "bbox": [float(v) for v in p.bbox.to_xyxy()]}
                     for p in out.object_prediction_list]
            matched, _ = match(gts, preds)
            stat = results.setdefault(slice_size, defaultdict(int))
            stat["gt"] += len(gts)
            stat["tp"] += len(matched)
            stat["pred"] += len(preds)
            for gi, g in enumerate(gts):
                bucket = size_bucket(g["bbox"])
                stat[bucket + "_gt"] += 1
                if gi in matched:
                    stat[bucket + "_tp"] += 1
        print(f"  {case}: done")

    print(f"\n{'slice':>6} {'GT':>5} {'pred':>5} {'TP':>5} {'P':>6} {'R':>6} {'F1':>6} "
          f"{'tiny R':>7} {'small R':>8} {'med R':>6} {'large R':>8}")
    for slice_size in args.slices:
        s = results[slice_size]
        p = s["tp"] / s["pred"] if s["pred"] else 0.
        r = s["tp"] / s["gt"] if s["gt"] else 0.
        f1 = 2 * p * r / (p + r) if p + r else 0.
        def rec(bucket):
            den = s.get(bucket + "_gt", 0)
            return s.get(bucket + "_tp", 0) / den if den else 0.
        print(f"{slice_size:>6} {s['gt']:>5} {s['pred']:>5} {s['tp']:>5} {p:>6.3f} {r:>6.3f} {f1:>6.3f} "
              f"{rec('tiny'):>7.2f} {rec('small'):>8.2f} {rec('medium'):>6.2f} {rec('large'):>8.2f}")


if __name__ == "__main__":
    main()
