"""Scale-sensitivity probe for the export geometry, with the estimated-scale variant.

The 6th evaluation showed bbox accuracy collapsing on 5 of 13 pages. Resampling a
public page reproduces the failure exactly - and shows it is a *scale* effect:

    scale   raw   canonical(absolute)   oracle(known scale)
    1.0     0.14  1.00                  1.00
    2.0     0.00  0.86                  0.86
    3.0     0.00  0.00                  0.86
    4.0     0.00  0.00                  0.86

Absolute canonical sizes collapse at 3x because the 20 px tolerance can no longer
absorb a table measured on 1x pages. ``auto`` estimates the page scale from the
detections themselves (median detected ruler width / reference width), so it is the
shippable version of ``oracle``: if auto tracks oracle, the fix works without GT.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
from PIL import Image

from pcb_parser.export_geometry import estimate_page_scale, load_scale_reference
from evaluate_official_component import (canonical_bbox, corner_error, load_gt_case,
                                         shrink_bbox)

POWER = {"ground", "voltage_source"}


def scaled_table(table, factor):
    out = {}
    for name, spec in table.items():
        copied = dict(spec)
        for key in ("w", "h", "w_v", "h_v"):
            if key in copied:
                copied[key] = round(copied[key] * factor, 2)
        out[name] = copied
    return out


def detect(model, image, imgsz, conf, iou, max_det):
    result = model.predict(source=np.asarray(image), imgsz=imgsz, conf=conf, iou=iou,
                           max_det=max_det, device="cpu", verbose=False)[0]
    names = model.names
    return [{"class_name": names[int(c)], "bbox": [float(v) for v in xyxy],
             "confidence": float(s)}
            for xyxy, c, s in zip(result.boxes.xyxy.tolist(), result.boxes.cls.tolist(),
                                  result.boxes.conf.tolist())]


def match_ratio(gts, preds, key):
    cand = []
    for gi, g in enumerate(gts):
        for pi, p in enumerate(preds):
            if p["class_name"] != g["class_name"]:
                continue
            err = corner_error(p[key], g["bbox"])
            if err <= 20.0:
                cand.append((err, gi, pi))
    cand.sort()
    ug, up = set(), set()
    for err, gi, pi in cand:
        if gi in ug or pi in up:
            continue
        ug.add(gi)
        up.add(pi)
    return len(ug) / len(gts) if gts else 0.


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="+", default=["0002"])
    ap.add_argument("--scales", nargs="+", type=float, default=[1.0, 2.0, 3.0, 4.0])
    ap.add_argument("--weights", type=Path, default=ROOT / "weights/detector_dataset300_v3_nano.pt")
    ap.add_argument("--config", type=Path, default=ROOT / "configs/baseline_v3.json")
    ap.add_argument("--geometry", type=Path, default=ROOT / "configs/bbox_convention_v1.json")
    ap.add_argument("--gt-root", type=Path,
                    default=ROOT.parent / "赛题六公开数据集" / "赛题六公开数据集" / "200_train_cases")
    args = ap.parse_args()

    from ultralytics import YOLO
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    det_cfg = cfg["detector"]
    geometry_doc = json.loads(args.geometry.read_text(encoding="utf-8"))
    table = geometry_doc["canonical"]
    affine = geometry_doc["transform"]
    reference, limits = load_scale_reference(
        {"bbox_convention": {"enabled": True, "path": str(args.geometry)}})
    model = YOLO(str(args.weights))

    print(f"{'case':>6} {'scale':>5} {'GT':>3} {'raw':>5} {'canon':>6} {'oracle':>7} {'auto':>6} "
          f"{'affine':>6} {'est':>6} {'med capW':>9}")
    for case in args.cases:
        image = Image.open(ROOT / "data/standardized" / case / f"{case}.png").convert("RGB")
        gt_files = list((args.gt_root / case).glob("*_target_named.json"))
        gt_base = load_gt_case(gt_files[0], image.height)
        fixed = [g for g in gt_base if g["class_name"] in table and g["class_name"] not in POWER]
        for scale in args.scales:
            big = image if scale == 1.0 else image.resize(
                (round(image.width * scale), round(image.height * scale)), Image.LANCZOS)
            gts = [dict(g, bbox=[v * scale for v in g["bbox"]]) for g in fixed]
            preds = detect(model, big, det_cfg["imgsz"], det_cfg["confidence"],
                           det_cfg["iou"], det_cfg["max_det"])

            widths = {}
            for p in preds:
                widths.setdefault(p["class_name"], []).append(p["bbox"][2] - p["bbox"][0])
            est, _ = estimate_page_scale(widths, reference, limits)
            oracle_table = scaled_table(table, scale)
            auto_table = scaled_table(table, est)

            for p in preds:
                p["bbox_raw"] = p["bbox"]
                spec = table.get(p["class_name"])
                p["bbox_canon"] = canonical_bbox(p["bbox"], spec) if spec and "w" in spec else p["bbox"]
                spec_o = oracle_table.get(p["class_name"])
                p["bbox_oracle"] = canonical_bbox(p["bbox"], spec_o) if spec_o and "w" in spec_o else p["bbox"]
                spec_a = auto_table.get(p["class_name"])
                p["bbox_auto"] = canonical_bbox(p["bbox"], spec_a) if spec_a and "w" in spec_a else p["bbox"]
                spec_f = affine.get(p["class_name"])
                p["bbox_affine"] = (shrink_bbox(p["bbox"], spec_f["rw"], spec_f["rh"],
                                                spec_f.get("dxf", 0.), spec_f.get("dyf", 0.))
                                    if spec_f else p["bbox"])

            n = len(gts)
            print(f"{case:>6} {scale:>5.1f} {n:>3} "
                  f"{match_ratio(gts, preds, 'bbox_raw'):>5.2f} "
                  f"{match_ratio(gts, preds, 'bbox_canon'):>6.2f} "
                  f"{match_ratio(gts, preds, 'bbox_oracle'):>7.2f} "
                  f"{match_ratio(gts, preds, 'bbox_auto'):>6.2f} "
                  f"{match_ratio(gts, preds, 'bbox_affine'):>6.2f} {est:>6.2f} "
                  f"{np.median(widths.get('capacitor', [float('nan')])):>9.1f}")


if __name__ == "__main__":
    main()
