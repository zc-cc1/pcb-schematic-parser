"""Attribute component false negatives / false positives to actionable causes.

Detector work keeps stalling on "where are the errors", so this script answers it
mechanically. It reads a *raw* pipeline run (``run_baseline.py`` output, i.e.
``detections.json`` + ``ocr.json``, both top-left pixel coordinates) and the
published GT, then classifies every error:

FN buckets
  ``pure_miss``      no prediction of any class overlaps the GT at IoU>=0.3
  ``confusion``      another class covers the GT at IoU>=0.5 (a labelling win, not detection)
  ``localization``   same class present at 0.3<=IoU<0.5 (the box is not tight enough)
  ``near_other``     another class at 0.3<=IoU<0.5

FP buckets
  ``duplicate``      same class as an accepted prediction at IoU>=0.5
  ``fragment``       same class at 0.3<=IoU<0.5 with an accepted prediction
  ``wrong_class``    covers a GT that was already matched by another class
  ``background``     no GT overlap >= 0.3 (text, title block, graphics)

Also reports FN size buckets and the FP confidence distribution, so a pruning
threshold can be chosen from data instead of taste.

Reads ground truth; inference never does.
"""
import argparse
import glob
import json
import os
import statistics as stats
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from PIL import Image

from evaluate_official_component import (canonical_bbox, load_gt_case, pred_class_name,
                                         iou, corner_error, shrink_bbox)

POWER = {"ground", "voltage_source"}


def bbox_gap(a, b):
    return ((max(a[0] - b[2], b[0] - a[2], 0) ** 2) + (max(a[1] - b[3], b[1] - a[3], 0) ** 2)) ** 0.5


def size_bucket(box):
    s = max(box[2] - box[0], box[3] - box[1])
    if s < 32:
        return "tiny(<32)"
    if s < 64:
        return "small(32-64)"
    if s < 128:
        return "medium(64-128)"
    return "large(>=128)"


def load_run(run_dir, case_id):
    """Returns (detections, texts) for one case from a run_baseline output tree."""
    files = glob.glob(os.path.join(run_dir, case_id, "*", "*", "detections.json"))
    if not files:
        return None, None
    det = json.loads(Path(files[0]).read_text(encoding="utf-8"))
    dets = [{"class_name": d["class_name"], "bbox": [float(v) for v in d["bbox"]],
             "confidence": float(d.get("confidence", 0.))} for d in det.get("detections", [])]
    texts = []
    ocr = Path(run_dir) / case_id / "ocr.json"
    if ocr.exists():
        raw = json.loads(ocr.read_text(encoding="utf-8"))
        for t in raw.get("texts") or raw.get("results") or []:
            poly = t.get("polygon")
            if not poly:
                continue
            xs = [p[0] for p in poly]
            ys = [p[1] for p in poly]
            texts.append({"text": t.get("text", ""), "bbox": [min(xs), min(ys), max(xs), max(ys)],
                          "confidence": float(t.get("confidence", 0.))})
    return dets, texts


def apply_geometry(dets, geom):
    """Rewrite detection boxes the way the exporter will, so the diagnosis talks
    about the boxes that actually ship (raw detector boxes are 2-3x too large and
    make every localisation error look like a miss)."""
    if not geom:
        return dets
    mode, table = geom
    for d in dets:
        spec = table.get(d["class_name"])
        if not spec:
            continue
        if mode == "canonical" and "w" in spec:
            d["bbox"] = canonical_bbox(d["bbox"], spec)
        else:
            d["bbox"] = shrink_bbox(d["bbox"], spec.get("rw", 1.), spec.get("rh", 1.),
                                    spec.get("dxf", 0.), spec.get("dyf", 0.))
    return dets


def diagnose(run_dir, gt_root, cases, tol=20., geom=None):
    fn_causes = Counter()
    fp_causes = Counter()
    fn_by_class = Counter()
    fn_by_size = Counter()
    fp_conf = []
    fn_examples = defaultdict(list)
    fp_examples = defaultdict(list)
    total_gt = total_pred = tp_total = 0
    dup_iou = []

    for cid in cases:
        gdir = Path(gt_root) / cid
        pngs = glob.glob(str(gdir / "*.png"))
        gts = glob.glob(str(gdir / "*_target_named.json"))
        if not (pngs and gts):
            continue
        height = Image.open(pngs[0]).size[1]
        gt = load_gt_case(gts[0], height)
        dets, texts = load_run(run_dir, cid)
        if dets is None:
            continue
        for d in dets:
            d["class_name"] = pred_class_name(d["class_name"])
        dets = apply_geometry(dets, geom)
        total_gt += len(gt)
        total_pred += len(dets)

        # accepted = one-to-one, same class, all corners within tolerance
        cand = []
        for gi, g in enumerate(gt):
            for pi, p in enumerate(dets):
                if p["class_name"] != g["class_name"]:
                    continue
                err = corner_error(p["bbox"], g["bbox"])
                if err <= tol:
                    cand.append((err, gi, pi))
        cand.sort()
        ug, up = set(), set()
        for err, gi, pi in cand:
            if gi in ug or pi in up:
                continue
            ug.add(gi)
            up.add(pi)
        tp_total += len(ug)

        for gi, g in enumerate(gt):
            if gi in ug:
                continue
            fn_by_class[g["class_name"]] += 1
            fn_by_size[size_bucket(g["bbox"])] += 1
            best, best_i, best_pi = 0., None, None
            for pi, p in enumerate(dets):
                v = iou(p["bbox"], g["bbox"])
                if v > best:
                    best, best_i, best_pi = v, pi, pi
            if best < 0.3:
                cause = "pure_miss"
            elif dets[best_i]["class_name"] == g["class_name"]:
                cause = "localization"
            elif best >= 0.5:
                cause = "confusion"
            else:
                cause = "near_other"
            fn_causes[cause] += 1
            if len(fn_examples[cause]) < 6:
                fn_examples[cause].append({
                    "case": cid, "key": g["key"], "class": g["class_name"],
                    "size": size_bucket(g["bbox"]), "best_iou": round(best, 2),
                    "best_class": dets[best_i]["class_name"] if best_i is not None else None})

        for pi, p in enumerate(dets):
            if pi in up:
                continue
            fp_conf.append(p["confidence"])
            same = [qi for qi, q in enumerate(dets) if qi != pi and q["class_name"] == p["class_name"]]
            best_dup = max(((iou(p["bbox"], dets[qi]["bbox"]), qi) for qi in same), default=(0., None))
            covers = max(((iou(p["bbox"], g["bbox"]), gi) for gi, g in enumerate(gt)), default=(0., None))
            if best_dup[0] >= 0.5 and best_dup[1] in up:
                cause = "duplicate"
                dup_iou.append(best_dup[0])
            elif best_dup[0] >= 0.3 and best_dup[1] in up:
                cause = "fragment"
            elif covers[0] >= 0.3 and covers[1] in ug:
                cause = "wrong_class"
            else:
                cause = "background"
            fp_causes[cause] += 1
            if len(fp_examples[cause]) < 6:
                text_near = min((bbox_gap(p["bbox"], t["bbox"]) for t in texts), default=None)
                fp_examples[cause].append({
                    "case": cid, "class": p["class_name"], "conf": round(p["confidence"], 3),
                    "size": size_bucket(p["bbox"]),
                    "nearest_text_px": None if text_near is None else round(text_near, 1)})

    return {
        "cases": len(cases), "gt": total_gt, "pred": total_pred, "tp": tp_total,
        "fn": total_gt - tp_total, "fp": total_pred - tp_total,
        "fn_causes": dict(fn_causes), "fp_causes": dict(fp_causes),
        "fn_by_class": dict(fn_by_class.most_common()),
        "fn_by_size": dict(fn_by_size),
        "fp_confidence": {
            "n": len(fp_conf),
            "median": round(stats.median(fp_conf), 3) if fp_conf else None,
            "lt_0.2": sum(1 for c in fp_conf if c < 0.2),
            "lt_0.3": sum(1 for c in fp_conf if c < 0.3),
            "lt_0.4": sum(1 for c in fp_conf if c < 0.4),
        },
        "duplicate_iou_median": round(stats.median(dup_iou), 3) if dup_iou else None,
        "fn_examples": {k: v for k, v in fn_examples.items()},
        "fp_examples": {k: v for k, v in fp_examples.items()},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True, help="run_baseline output directory")
    ap.add_argument("--gt-root", type=Path, required=True)
    ap.add_argument("--case-id", action="append")
    ap.add_argument("--cases-file", type=Path)
    ap.add_argument("--tol", type=float, default=20.)
    ap.add_argument("--shrink", type=Path, help="bbox_convention_v1.json to score shipped boxes")
    ap.add_argument("--geometry", choices=["affine", "canonical"], default="canonical")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    cases = set(args.case_id or [])
    if args.cases_file:
        cases |= set(args.cases_file.read_text(encoding="utf-8").split())
    if not cases:
        cases = {p.name for p in args.run.iterdir() if p.is_dir() and p.name[:4].isdigit()}
    geom = None
    if args.shrink:
        raw = json.loads(args.shrink.read_text(encoding="utf-8"))
        table = raw.get(args.geometry) or raw.get("transform") or {}
        table = {k: v for k, v in table.items() if isinstance(v, dict)}
        geom = (args.geometry, table)
    result = diagnose(args.run, args.gt_root, sorted(cases), args.tol, geom)
    result["run"] = str(args.run)
    result["tolerance_px"] = args.tol
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    p = result["tp"] / result["pred"] if result["pred"] else 0
    r = result["tp"] / result["gt"] if result["gt"] else 0
    print(f"cases={result['cases']} GT={result['gt']} pred={result['pred']} TP={result['tp']} "
          f"FP={result['fp']} FN={result['fn']} P={p:.4f} R={r:.4f} F1={2*p*r/(p+r):.4f}")
    print(f"\nFN causes: {result['fn_causes']}")
    print(f"FN by size:  {result['fn_by_size']}")
    print(f"FN by class: {result['fn_by_class']}")
    print(f"\nFP causes: {result['fp_causes']}")
    print(f"FP confidence: {result['fp_confidence']}")
    print(f"duplicate IoU median: {result['duplicate_iou_median']}")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
