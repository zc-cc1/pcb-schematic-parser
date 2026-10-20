"""Calibrate the bbox convention gap between our boxes and the official GT.

Background: our detector is trained on ``Dataset300/*_CVJsonStd.json``
``location`` boxes, which are CAD *cells* (glyph + pin leads + a little margin).
The published ``*_target_named.json`` boxes are the drawn symbol only. The gap is
a per-class scale, not a bug in the model: measured medians run from 0.95
(ground) down to 0.50 (resistor) in each dimension, so a fixed pixel nudge
cannot fix it and a per-class affine can.

The script fits two independent tables:

``label_space``
    CVJsonStd ``location`` vs official GT, all 200 published cases. Large samples
    (thousands per class), no detector noise. This is the *convention*.
``detector_space``
    our detector output vs official GT. Smaller and noisier, but it measures what
    we actually have to correct. Use ``--fit-cases`` / ``--eval-cases`` to keep
    the two disjoint when you intend to validate rather than fit.

Both are reported as ``{rw, rh, dxf, dyf}``: rescale about the centre by rw/rh,
then shift by a fraction of the box size (so the correction scales with the
symbol; the ground symbol's box is bottom-anchored, which is exactly a size
dependent shift).

This entrypoint may read ground truth; inference does not.
"""
import argparse
import glob
import json
import math
import os
import statistics as stats
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from PIL import Image

from evaluate_official_component import (OFFICIAL_TYPE_MAP, official_type, flip, iou,
                                         load_eval_json, load_gt_case)

# Extra raw aliases only present in the CVJsonStd source.
CAND_MAP = {
    "box": "ic", "c": "capacitor", "r": "resistor", "l": "inductor",
    "crystal": "crystal", "switch": "switch", "gnd": "ground", "v": "voltage_source",
    "mosfet": "mosfet", "bjt": "transistor", "d": "diode", "led": "led",
    "pin": "pin_port", "testpoint": "testpoint", "net_input": "net_input",
    "net_output": "net_output", "net_bidirection": "net_bidirectional",
    "fuse": "fuse", "buzzer": "buzzer", "motor": "motor", "battary": "battery",
    "other": "other", "transfomer": "transformer", "jumper": "jumper",
    "block": "raw_block", "seg": "other", "amp": "opamp",
    "circle_header": "circle_header", "ant": "antenna", "esd": "esd",
    "m": "raw_m", "net_short": "net_short", "opt": "optocoupler", "dc": "raw_dc",
    "raw_m": "raw_m",
}

CLAMP = {"rw": (0.35, 1.20), "rh": (0.30, 1.20), "dxf": (-0.5, 0.5), "dyf": (-0.6, 0.6)}
MIN_SAMPLES = 8

# Power symbols are deliberately excluded from the affine table.
# Their source boxes are bimodal in the published data: on some pages the
# CVJsonStd ``location`` is the pin-tip square (14x14) while the official GT is
# the drawn triangle (20x19) sitting below it; on others it is the full cell
# (40x40). A single median therefore fits neither population - measured A/B put
# power F1 at .9369 -> .9067 when the affine was applied. Power geometry is fixed
# instead by anchoring on the glyph ink (see docs/detection_v4_plan.md WP1).
POWER_CLASSES = {"ground", "voltage_source"}

# Mechanism, not taste: for passives our ``location`` box is the *cell* (glyph +
# pin leads), so the gap is a scale. For ``Box``/IC the cell already is the drawn
# rectangle and the source also carries ``realBBox``, which equals the official GT
# exactly on the verified sample (0002/IC2) - the gap there is a small constant
# margin, and a ratio shrink over-corrects large ICs (measured: 181 -> 177 TP).
# The IC fix therefore belongs to the realBBox/retrain path (WP3-B), not here.
CONSTANT_MARGIN_CLASSES = {"ic"}

# Symbol sizes vary per page for ICs and connectors; canonicalise only symbols
# whose published GT size is stable. Threshold on a MAD-based coefficient of
# variation, so a class is "fixed size" only if 68% of its boxes stay within
# +/-30% of the median in both dimensions.
CANONICAL_MAX_CV = 0.30


def clamp(value, lo, hi):
    return max(lo, min(hi, value))


def robust_cv(values):
    """MAD-based coefficient of variation: scale-free, insensitive to outliers."""
    med = stats.median(values)
    if med == 0:
        return 0.
    mad = stats.median(abs(v - med) for v in values)
    return 1.4826 * mad / abs(med)


def box_stats(match, dest):
    """Accumulate per-class statistics for (source box, official box) pairs."""
    for cls, (src, dst) in match:
        sw, sh = src[2] - src[0], src[3] - src[1]
        dw, dh = dst[2] - dst[0], dst[3] - dst[1]
        if sw <= 0 or sh <= 0 or dw <= 0 or dh <= 0:
            continue
        scx, scy = (src[0] + src[2]) / 2., (src[1] + src[3]) / 2.
        dcx, dcy = (dst[0] + dst[2]) / 2., (dst[1] + dst[3]) / 2.
        dest[cls].append({"rw": dw / sw, "rh": dh / sh,
                          "dxf": (dcx - scx) / sw, "dyf": (dcy - scy) / sh,
                          "src_w": sw, "src_h": sh, "dst_w": dw, "dst_h": dh})


def fit_table(samples, fallback=None):
    table = {}
    for cls, rows in samples.items():
        if len(rows) < MIN_SAMPLES:
            if fallback and cls in fallback:
                table[cls] = dict(fallback[cls], n=len(rows), source="fallback")
            continue
        entry = {"n": len(rows)}
        clamped = []
        for key in ("rw", "rh", "dxf", "dyf"):
            raw = stats.median(r[key] for r in rows)
            value = clamp(raw, *CLAMP[key])
            if key in ("rw", "rh") and abs(value - raw) > 1e-9:
                clamped.append(key)
            entry[key] = round(value, 4)
        entry["median_src_w"] = round(stats.median(r["src_w"] for r in rows), 1)
        entry["median_src_h"] = round(stats.median(r["src_h"] for r in rows), 1)
        entry["median_gt_w"] = round(stats.median(r["dst_w"] for r in rows), 1)
        entry["median_gt_h"] = round(stats.median(r["dst_h"] for r in rows), 1)
        entry["source"] = "fit"
        if clamped:
            # A ratio pinned at the clamp bound means the population is not a
            # single scale; do not ship a correction we cannot justify.
            entry["source"] = "excluded_clamp_hit:" + ",".join(clamped)
        table[cls] = entry
    return table


def deployable_transform(table):
    """The subset of the fitted table that is safe to apply at export time."""
    out, excluded = {}, {}
    for cls, entry in table.items():
        if cls in POWER_CLASSES:
            excluded[cls] = "power_geometry_needs_glyph_anchor_not_affine"
            continue
        if cls in CONSTANT_MARGIN_CLASSES:
            excluded[cls] = "constant_margin_not_scale_use_realbbox_path"
            continue
        if str(entry.get("source", "")).startswith("excluded_"):
            excluded[cls] = entry["source"]
            continue
        out[cls] = {"rw": entry["rw"], "rh": entry["rh"],
                    "dxf": entry["dxf"], "dyf": entry["dyf"]}
    return out, excluded


def canonical_table(gt_root, offsets, cases=None):
    """Per-class natural symbol size from the published GT, split by orientation.

    The organiser boxes the drawn symbol, and a symbol has one natural size per
    orientation, so this table is read straight off the GT distribution - no
    model and no pairing involved. ``offsets`` (from the label-space fit) supply
    the centre correction, which is the part that actually went wrong in the 5th
    evaluation: sizes were canonical already, centres were 20-34 px off.
    """
    horiz, vert = defaultdict(list), defaultdict(list)
    for cid, gdir in case_dirs(gt_root).items():
        if cases and cid not in cases:
            continue
        pngs = glob.glob(os.path.join(gdir, "*.png"))
        gts = glob.glob(os.path.join(gdir, "*_target_named.json"))
        if not (pngs and gts):
            continue
        height = Image.open(pngs[0]).size[1]
        for g in load_gt_case(gts[0], height):
            cls = g["class_name"]
            if not cls:
                continue
            w = g["bbox"][2] - g["bbox"][0]
            h = g["bbox"][3] - g["bbox"][1]
            if w <= 0 or h <= 0:
                continue
            (vert if h > w else horiz)[cls].append((w, h))
    out = {}
    excluded = {}
    for cls in set(horiz) | set(vert):
        hz, vt = horiz.get(cls, []), vert.get(cls, [])
        if len(hz) + len(vt) < MIN_SAMPLES:
            continue
        base = hz if len(hz) >= len(vt) else vt
        alt = vt if base is hz else hz
        cv_w, cv_h = robust_cv([b[0] for b in base]), robust_cv([b[1] for b in base])
        if max(cv_w, cv_h) > CANONICAL_MAX_CV:
            # A variable-size symbol (an IC is 51x58 on one page and 247x372 on
            # another) must keep the detected extent; faking a canonical size
            # would be worse than useless.
            excluded[cls] = {"cv_w": round(cv_w, 3), "cv_h": round(cv_h, 3),
                             "reason": "size_not_fixed_use_detected_extent"}
            continue
        entry = {"w": round(stats.median(b[0] for b in base), 1),
                 "h": round(stats.median(b[1] for b in base), 1),
                 "cv_w": round(cv_w, 3), "cv_h": round(cv_h, 3), "n": len(base)}
        if len(alt) >= MIN_SAMPLES:
            entry["w_v"] = round(stats.median(b[0] for b in alt), 1)
            entry["h_v"] = round(stats.median(b[1] for b in alt), 1)
            entry["n_v"] = len(alt)
        off = offsets.get(cls) or {}
        entry["dxf"] = round(float(off.get("dxf", 0.)), 4)
        entry["dyf"] = round(float(off.get("dyf", 0.)), 4)
        out[cls] = entry
    return out, excluded


def case_dirs(gt_root):
    return {os.path.basename(d): d for d in glob.glob(str(Path(gt_root) / "*")) if os.path.isdir(d)}


def label_space_pairs(gt_root, dataset_root, cases=None):
    """CVJsonStd location boxes paired with official GT by class + centre distance."""
    samples = defaultdict(list)
    pairs = []
    for cid, gdir in case_dirs(gt_root).items():
        if cases and cid not in cases:
            continue
        pngs = glob.glob(os.path.join(gdir, "*.png"))
        gts = glob.glob(os.path.join(gdir, "*_target_named.json"))
        cands = glob.glob(os.path.join(str(dataset_root), cid, "*_CVJsonStd.json"))
        if not (pngs and gts and cands):
            continue
        height = Image.open(pngs[0]).size[1]
        gt = load_gt_case(gts[0], height)
        raw = json.loads(Path(cands[0]).read_text(encoding="utf-8-sig"))
        locs = []
        for key, comp in (raw.get("components") or {}).items():
            loc = comp.get("location")
            cls = CAND_MAP.get(str(comp.get("category") or "").strip().lower())
            if not isinstance(loc, dict) or not cls:
                continue
            locs.append((cls, [float(loc["x1"]), float(loc["y1"]), float(loc["x2"]), float(loc["y2"])]))
        used = set()
        for g in gt:
            gc = g["class_name"]
            gcx, gcy = (g["bbox"][0] + g["bbox"][2]) / 2., (g["bbox"][1] + g["bbox"][3]) / 2.
            best, best_d = None, 1e9
            for i, (cls, box) in enumerate(locs):
                if cls != gc or i in used:
                    continue
                d = ((box[0] + box[2]) / 2. - gcx) ** 2 + ((box[1] + box[3]) / 2. - gcy) ** 2
                if d < best_d:
                    best, best_d = i, d
            if best is None or best_d ** .5 > 15:
                continue
            used.add(best)
            pairs.append((gc, (locs[best][1], g["bbox"])))
    box_stats(pairs, samples)
    return samples


def detector_offsets(pred_path, gt_root, cases=None, max_centre_px=40.):
    """Centre offset per class, fitted on our detector output.

    Matching is nearest-GT-centre rather than IoU: an IoU gate would drop exactly
    the badly-placed components whose offset we are trying to measure, which
    biases the estimate toward zero.

    The offset is fitted on *detections* because that is what the transform is
    applied to. Fitting it on CVJsonStd labels instead mixes two source regimes
    (on some pages a ground's location is a 14x14 pin tip, on others a 40x40
    cell) and over-corrects by roughly 7 px - measured on 0002.
    """
    samples = defaultdict(list)
    pairs = []
    preds_by_case, _ = load_eval_json(pred_path)
    for cid, preds in preds_by_case.items():
        if cases and cid not in cases:
            continue
        gdir = Path(gt_root) / cid
        pngs = glob.glob(str(gdir / "*.png"))
        gts = glob.glob(str(gdir / "*_target_named.json"))
        if not (pngs and gts):
            continue
        height = Image.open(pngs[0]).size[1]
        gt = load_gt_case(gts[0], height)
        used = set()
        for p in preds:
            pw = p["bbox"][2] - p["bbox"][0]
            ph = p["bbox"][3] - p["bbox"][1]
            if pw <= 0 or ph <= 0:
                continue
            pcx, pcy = (p["bbox"][0] + p["bbox"][2]) / 2., (p["bbox"][1] + p["bbox"][3]) / 2.
            best, best_d = None, max_centre_px
            for gi, g in enumerate(gt):
                if g["class_name"] != p["class_name"] or gi in used:
                    continue
                gcx, gcy = (g["bbox"][0] + g["bbox"][2]) / 2., (g["bbox"][1] + g["bbox"][3]) / 2.
                d = math.hypot(gcx - pcx, gcy - pcy)
                if d < best_d:
                    best, best_d = gi, d
            if best is None:
                continue
            used.add(best)
            g = gt[best]
            gcx, gcy = (g["bbox"][0] + g["bbox"][2]) / 2., (g["bbox"][1] + g["bbox"][3]) / 2.
            samples[p["class_name"]].append({
                "dxf": (gcx - pcx) / pw, "dyf": (gcy - pcy) / ph,
                "rw": (g["bbox"][2] - g["bbox"][0]) / pw, "rh": (g["bbox"][3] - g["bbox"][1]) / ph,
                "src_w": pw, "src_h": ph,
                "dst_w": g["bbox"][2] - g["bbox"][0], "dst_h": g["bbox"][3] - g["bbox"][1]})
    out = {}
    for cls, rows in samples.items():
        if len(rows) < MIN_SAMPLES:
            continue
        out[cls] = {"n": len(rows),
                    "dxf": round(clamp(stats.median(r["dxf"] for r in rows), *CLAMP["dxf"]), 4),
                    "dyf": round(clamp(stats.median(r["dyf"] for r in rows), *CLAMP["dyf"]), 4),
                    "rw": round(stats.median(r["rw"] for r in rows), 4),
                    "rh": round(stats.median(r["rh"] for r in rows), 4)}
    return out


def detector_space_pairs(pred_path, gt_root, cases=None):
    """Detector boxes paired with official GT at IoU>=0.3 (loose: fitting only)."""
    samples = defaultdict(list)
    pairs = []
    preds_by_case, _ = load_eval_json(pred_path)
    for cid, preds in preds_by_case.items():
        if cases and cid not in cases:
            continue
        gdir = Path(gt_root) / cid
        pngs = glob.glob(str(gdir / "*.png"))
        gts = glob.glob(str(gdir / "*_target_named.json"))
        if not (pngs and gts):
            continue
        height = Image.open(pngs[0]).size[1]
        gt = load_gt_case(gts[0], height)
        cand = []
        for gi, g in enumerate(gt):
            for pi, p in enumerate(preds):
                if p["class_name"] != g["class_name"]:
                    continue
                v = iou(p["bbox"], g["bbox"])
                if v >= 0.3:
                    cand.append((v, gi, pi))
        cand.sort(reverse=True)
        ug, up = set(), set()
        for v, gi, pi in cand:
            if gi in ug or pi in up:
                continue
            ug.add(gi)
            up.add(pi)
            pairs.append((gt[gi]["class_name"], (preds[pi]["bbox"], gt[gi]["bbox"])))
    box_stats(pairs, samples)
    return samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-root", type=Path, required=True)
    ap.add_argument("--dataset-root", type=Path, default=ROOT.parent / "Dataset300")
    ap.add_argument("--pred", type=Path, help="eval JSON; enables the detector-space table")
    ap.add_argument("--fit-cases", nargs="*", help="restrict the detector-space fit to these case ids")
    ap.add_argument("--transform-source", choices=["label", "detector"], default="label",
                    help="which table becomes the deployable 'transform'. 'detector' fits on our own "
                         "detector output, which is what the transform is actually applied to")
    ap.add_argument("--offset-source", choices=["label", "detector"], default="label",
                    help="where the canonical centre offset comes from. 'detector' (recommended when "
                         "--pred is given) avoids mixing the two CVJsonStd regimes for power symbols")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    label_samples = label_space_pairs(args.gt_root, args.dataset_root,
                                      set(args.fit_cases) if args.fit_cases else None)
    label_table = fit_table(label_samples)
    transform, excluded = deployable_transform(label_table)

    # The canonical table needs sizes (read straight off the GT) and a centre
    # offset. Offsets fitted on CVJsonStd labels mix two source regimes for power
    # symbols, so when a pred file is available they are fitted on detections.
    offset_table = label_table
    if args.pred and args.offset_source == "detector":
        offset_table = detector_offsets(args.pred, args.gt_root,
                                        set(args.fit_cases) if args.fit_cases else None)
    canonical, canonical_excluded = canonical_table(
        args.gt_root, offset_table, set(args.fit_cases) if args.fit_cases else None)

    result = {
        "version": "bbox-convention-v1",
        "status": "fitted_from_published_gt_offline",
        "note": ("two tables. 'canonical': replace the detected extent with the symbol's "
                 "natural size read off the published GT and shift the centre by the fitted "
                 "offset. 'transform': affine rescale about the centre, the fallback for "
                 "classes whose size is not fixed. Both are a convention difference, not "
                 "detector error."),
        "min_samples": MIN_SAMPLES,
        "clamp": CLAMP,
        "offset_source": args.offset_source,
        "canonical_max_cv": CANONICAL_MAX_CV,
        "transform": transform,
        "canonical": canonical,
        "canonical_excluded": canonical_excluded,
        "excluded_classes": excluded,
        "label_space": label_table,
    }

    if args.pred:
        det_samples = detector_space_pairs(args.pred, args.gt_root, set(args.fit_cases) if args.fit_cases else None)
        det_table = fit_table(det_samples, fallback=label_table)
        result["detector_space"] = det_table
        result["detector_space_fit_cases"] = sorted(args.fit_cases) if args.fit_cases else "all"
        if args.transform_source == "detector":
            transform, excluded = deployable_transform(det_table)
            result["transform"] = transform
            result["excluded_classes"] = excluded
            result["applies_to"] = "ordinary components only (fitted in detector space); power symbols use glyph anchoring"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"{'class':>16} {'n':>5} {'rw':>6} {'rh':>6} {'dxf':>7} {'dyf':>7} {'src WxH':>12} {'gt WxH':>11}  deployed")
    for cls, v in sorted(label_table.items(), key=lambda kv: -kv[1]["n"]):
        flag = "yes" if cls in transform else "-"
        print(f"{cls:>16} {v['n']:>5} {v['rw']:>6.3f} {v['rh']:>6.3f} {v['dxf']:>7.3f} {v['dyf']:>7.3f} "
              f"{v['median_src_w']:>6.1f}x{v['median_src_h']:<5.1f} {v['median_gt_w']:>5.1f}x{v['median_gt_h']:<5.1f}  {flag}")
    print(f"\ndeployed classes: {len(transform)}  excluded: {len(excluded)}")
    for cls, why in sorted(excluded.items()):
        print(f"   excluded {cls:>16}: {why}")
    if "detector_space" in result:
        print(f"\ndetector-space table ({len(result['detector_space'])} classes) written for A/B comparison")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
