"""Official-rubric component evaluation against the published 200-case GT.

Why this exists: ``evaluate_detector.py`` scores class-name + IoU>=0.5 on our own
converted labels. The organiser scores something different:

* ordinary components: key + type + Name + value must match, and *every* bbox
  coordinate must land within 20 px (not an IoU test);
* power symbols (gnd / v/vcc/vdd): key and Name are ignored, type and value must
  match, and the geometry is matched one-to-one.

The two rubrics disagree by several points and can even disagree in sign, so
detector work must be judged with this script, not with the IoU proxy.

This entrypoint may read ground truth; inference does not.

Usage (project root)::

    .\\.venv-gpu\\Scripts\\python.exe scripts/evaluate_official_component.py `
        --pred runs/eval_v3_fused_v2thr.json `
        --gt-root "..\\赛题六公开数据集\\赛题六公开数据集\\200_train_cases" `
        --output runs/official_eval_baseline.json

Prediction sources
------------------
``--pred`` accepts two shapes:

1. an offline evaluation JSON written by ``evaluate_detector.py``
   (``cases[].predictions[].{class_name,bbox}``, bbox in top-left pixels);
2. a directory holding submission-format ``*_Pred.json`` files (``components``
   dict, bbox already exported bottom-left and therefore flipped back here).

Only shape (1) can measure pure detector geometry, because it carries no
component keys. Shape (2) additionally enables ``--match key``.

Rubric limits (stated, not hidden)
----------------------------------
* ``--match class`` (default): class/type + geometry only. This is a *upper
  bound* on the official score - it credits a component whose designator OCR
  would have failed, which the official rubric would not.
* ``--match key``: also requires the component key to be equal (ordinary
  components) or the power geometry to be assigned (power symbols).
* value/Name equality is checked and reported as a separate breakdown so a
  detection-only run does not get blamed for OCR mistakes.
"""
import argparse
import glob
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from PIL import Image

# Official raw type -> detector class name. Mirrors configs/conversion.json plus
# the variants actually found in the published GT (crystal_2pin/3pin/4pin,
# net_bidirection, amp_3pin/5pin, bjt_pnp, mosfet_pnp, battary, transfomer, ...).
OFFICIAL_TYPE_MAP = {
    "box": "ic", "c": "capacitor", "r": "resistor", "l": "inductor",
    "crystal": "crystal", "crystal_2pin": "crystal", "crystal_3pin": "crystal",
    "crystal_4pin": "crystal", "switch": "switch", "gnd": "ground",
    "v": "voltage_source", "mosfet": "mosfet", "mosfet_pnp": "mosfet",
    "bjt": "transistor", "bjt_pnp": "transistor", "d": "diode", "led": "led",
    "pin": "pin_port", "testpoint": "testpoint", "net_input": "net_input",
    "net_output": "net_output", "net_bidirection": "net_bidirectional",
    "net_short": "net_short", "fuse": "fuse", "buzzer": "buzzer",
    "motor": "motor", "battary": "battery", "other": "other",
    "transfomer": "transformer", "jumper": "jumper", "block": "raw_block",
    "seg": "other", "amp": "opamp", "amp_3pin": "opamp", "amp_5pin": "opamp",
    "circle_header": "circle_header", "ant": "antenna", "esd": "esd",
    "m": "raw_m", "opt": "optocoupler", "dc": "raw_dc",
}

POWER_CLASSES = {"ground", "voltage_source"}
# Documented leniency only: ICs and power symbols. Everything else is compared
# as spelled, which is what makes the type vocabulary a scored property.
OFFICIAL_TYPE_ALIASES = (frozenset({"box", "ic"}), frozenset({"gnd", "ground"}),
                         frozenset({"v", "vcc", "vdd", "supply"}))
# Detector class name -> accepted official type aliases, for runs where the
# prediction side is still using detector-style names instead of raw types.
CLASS_TO_OFFICIAL = {
    "ground": {"gnd"}, "voltage_source": {"v", "vcc", "vdd"},
    "ic": {"box", "ic"}, "capacitor": {"c"}, "resistor": {"r"},
    "inductor": {"l"}, "diode": {"d"}, "transistor": {"bjt", "bjt_pnp"},
}

MAX_COORD_PX = 20.0


def flip(bbox, height):
    """Bottom-left origin bbox -> top-left origin bbox."""
    x1, y1, x2, y2 = (float(v) for v in bbox)
    return [x1, height - y2, x2, height - y1]


def corner_error(a, b):
    return max(abs(a[i] - b[i]) for i in range(4))


def iou(a, b):
    ov = max(0., min(a[2], b[2]) - max(a[0], b[0])) * max(0., min(a[3], b[3]) - max(a[1], b[1]))
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ov
    return ov / ua if ua > 0 else 0.


def shrink_bbox(bbox, rw, rh, dxf=0., dyf=0.):
    """Rescale about the centre and shift by a fraction of the box size.

    ``rw``/``rh`` are width/height multipliers; ``dxf``/``dyf`` are offsets
    expressed as a fraction of the box width/height so the correction scales
    with the symbol instead of being a fixed pixel nudge.
    """
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    cx, cy = (bbox[0] + bbox[2]) / 2., (bbox[1] + bbox[3]) / 2.
    cx, cy = cx + dxf * w, cy + dyf * h
    hw, hh = w * rw / 2., h * rh / 2.
    return [cx - hw, cy - hh, cx + hw, cy + hh]


def canonical_bbox(bbox, spec):
    """Replace the box size with a canonical symbol size, keep the (shifted) centre.

    ``spec`` keys: ``w``/``h`` for the upright variant, optional ``w_v``/``h_v``
    for the variant whose detection box is taller than wide, plus ``dxf``/``dyf``
    as a fraction of the *detection* box size. This models the organiser GT
    definition, whose boxes are the drawn symbol at its natural size.

    ``clamp_to_detection`` (default on) keeps the symbol box inside the detected
    cell. The glyph is physically drawn inside the cell, so an offset that would
    push it outside is an artefact of fitting the offset on a differently-sized
    source box - without the clamp a ground symbol lands ~7 px low on dense pages.
    """
    x1, y1, x2, y2 = (float(v) for v in bbox)
    w, h = x2 - x1, y2 - y1
    cx = (x1 + x2) / 2. + spec.get("dxf", 0.) * w
    cy = (y1 + y2) / 2. + spec.get("dyf", 0.) * h
    if h > w and "w_v" in spec:
        nw, nh = spec["w_v"], spec["h_v"]
    else:
        nw, nh = spec["w"], spec["h"]
    if spec.get("clamp_to_detection", True):
        if nw <= w:
            cx = min(max(cx, x1 + nw / 2.), x2 - nw / 2.)
        if nh <= h:
            cy = min(max(cy, y1 + nh / 2.), y2 - nh / 2.)
    return [cx - nw / 2., cy - nh / 2., cx + nw / 2., cy + nh / 2.]


def pred_class_name(raw):
    """Normalise a *predicted* type string to a detector class name.

    Prediction files may carry either vocabulary: offline evaluation JSON uses
    detector names (``voltage_source``), while a submission exported with the
    official raw mapping uses ``vcc``/``gnd``/``box``/``r``. Mixing the two
    silently turns every VCC symbol into one unmatched GT plus one false positive
    (and files that false positive under the wrong bucket), so the normalisation
    happens once, at load time.
    """
    t = str(raw or "").strip().lower()
    if t in ("vcc", "vdd"):
        return "voltage_source"
    if t == "gnd":
        return "ground"
    return official_type(t) or t


def class_matches(pred_type, gt_type):
    """Do two *raw* type strings name the same component under the official rule?

    The organiser compares the type string, and the only documented leniency is
    an alias table for ICs (``box``/``ic``) and power symbols
    (``gnd``/``ground``, ``v``/``vcc``/``vdd``). Everything else must be spelled
    the same - so exporting ``resistor`` where the GT says ``r`` is a type
    failure, not a cosmetic difference. Earlier versions of this script normalised
    both sides to detector class names first, which silently hid that (and made
    the reported score optimistic).
    """
    p = str(pred_type).strip().lower()
    g = str(gt_type).strip().lower()
    if p == g:
        return True
    for group in OFFICIAL_TYPE_ALIASES:
        if p in group and g in group:
            return True
    return False


def official_type(raw_type):
    """Normalise an official raw type to a detector class name.

    Prefix rules cover the variants whose suffix is a localisation or a typo in
    the published GT (``m3螺丝`` = M3 silkscreen -> raw_m, ``crystal_*pin``).
    """
    t = str(raw_type or "").strip().lower().replace(" ", "")
    if t in OFFICIAL_TYPE_MAP:
        return OFFICIAL_TYPE_MAP[t]
    if t.startswith("m3") or t.startswith("m "):
        return "raw_m"
    if t.startswith("crystal"):
        return "crystal"
    if t.startswith("amp"):
        return "opamp"
    if t.startswith("net_bidir"):
        return "net_bidirectional"
    return None


def load_gt_case(path, height):
    raw = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    out = []
    for key, comp in (raw.get("components") or {}).items():
        bbox = comp.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        raw_type = str(comp.get("type") or "").strip().lower()
        out.append({
            "key": key,
            "raw_type": raw_type,
            "class_name": official_type(raw_type),
            "Name": comp.get("Name"),
            "value": comp.get("value"),
            "bbox": flip(bbox, height),
        })
    return out


def load_eval_json(path):
    """evaluate_detector.py output -> {case_id: [{class_name, bbox, conf}]}."""
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    cases = {}
    for case in data.get("cases") or []:
        cases[case["case_id"]] = [
            {"key": None, "class_name": pred_class_name(p["class_name"]),
             "declared_type": p["class_name"], "from_eval": True,
             "bbox": [float(v) for v in p["bbox"]], "confidence": p.get("confidence")}
            for p in case.get("predictions") or []
        ]
    return cases, data


def load_pred_dir(path, height_by_case, image_root, pattern="*_Pred.json"):
    """Submission-format directory -> {case_id: [{key, class_name, bbox}]}.

    ``pattern`` may be recursive (``**/components_partial.json``) so that raw
    pipeline run directories can be scored without copying files around; the case
    id is then taken from the nearest parent directory name.
    """
    cases = {}
    files = glob.glob(os.path.join(path, pattern), recursive=True)
    for file in sorted(files):
        cid = os.path.basename(file).split("_")[0] if file.endswith("_Pred.json") else None
        if cid is None:
            parts = Path(file).resolve().parts
            cid = next((p for p in reversed(parts) if p[:4].isdigit() or (p.isdigit() and len(p) <= 4)), None)
        if cid is None:
            continue
        height = height_by_case(cid, image_root)
        if height is None:
            continue
        raw = json.loads(Path(file).read_text(encoding="utf-8-sig"))
        items = []
        for key, comp in (raw.get("components") or {}).items():
            bbox = comp.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            items.append({"key": key, "class_name": pred_class_name(comp.get("type")),
                          "declared_type": str(comp.get("type") or ""),
                          "Name": comp.get("Name"), "value": comp.get("value"),
                          "bbox": flip(bbox, height)})
        cases.setdefault(cid, []).extend(items)
    return cases


def match_case(gts, preds, match_mode, tol, shrink=None, ignore_fields=False):
    """Return (tp, fp, fn, details). One-to-one, geometry first, then key/value.

    In ``key`` mode the official rubric also requires type/Name/value equality, so
    a pair that matches on key+geometry but disagrees on a field is *not* a true
    positive - it costs one FP and one FN, exactly as the organiser counts it.
    ``ignore_fields`` relaxes that for pipelines which do not emit Name/value yet
    (the quick-dev path); use it only when you mean "geometry and designator
    only", never to make a number look better.
    """
    for p in preds:
        spec = (shrink or {}).get(p["class_name"])
        if not spec:
            p["bbox_shrunk"] = p["bbox"]
        elif "w" in spec:  # canonical symbol size
            p["bbox_shrunk"] = canonical_bbox(p["bbox"], spec)
        else:              # affine rescale about the centre
            p["bbox_shrunk"] = shrink_bbox(p["bbox"], spec["rw"], spec["rh"],
                                           spec.get("dxf", 0.), spec.get("dyf", 0.))

    def type_ok(g, p):
        declared = p.get("declared_type")
        if declared is None or p.get("from_eval"):
            # offline evaluation JSON carries detector class names, not the type
            # string we would export, so vocabulary cannot be scored there
            return g["class_name"] == p["class_name"]
        return class_matches(declared, g["raw_type"]) and g["class_name"] == p["class_name"]

    def geom_ok(g, p):
        if not type_ok(g, p):
            return None
        err = corner_error(p["bbox_shrunk"], g["bbox"])
        return err if err <= tol else None

    cand = []
    for gi, g in enumerate(gts):
        for pi, p in enumerate(preds):
            err = geom_ok(g, p)
            if err is None:
                continue
            if match_mode == "key" and g["class_name"] not in POWER_CLASSES:
                if not p["key"] or str(p["key"]).strip().lower() != str(g["key"]).strip().lower():
                    continue
            cand.append((err, gi, pi))
    cand.sort()
    used_g, used_p = set(), set()
    details = {"missing": [], "mismatch": [], "extra": [], "geom_matched": 0}
    matched_pairs = {}
    for err, gi, pi in cand:
        if gi in used_g or pi in used_p:
            continue
        g, p = gts[gi], preds[pi]
        failed = []
        if not type_ok(g, p):
            failed.append("type")
        if not ignore_fields:
            # Power symbols are matched on type + value + one-to-one geometry, with
            # key and Name ignored (protocol: "power: type/value and one-to-one
            # geometry, ignore key/Name"). Its `value` is the net caption (GND, VCC,
            # +3V3), so skipping the value check here made the local score ~9 points
            # optimistic on the released 13-case reference: measured .8388 vs the
            # organiser's .7482 before this fix.
            if g["class_name"] in POWER_CLASSES:
                if (g.get("value") or None) != (p.get("value") or None):
                    failed.append("value")
            else:
                if (g.get("value") or None) != (p.get("value") or None):
                    failed.append("value")
                if (g.get("Name") or None) != (p.get("Name") or None):
                    failed.append("Name")
        details["geom_matched"] += 1
        used_g.add(gi)
        used_p.add(pi)
        if failed:
            details["mismatch"].append({"key": g["key"], "pred_key": p.get("key"),
                                        "failed_checks": failed, "corner_error_px": round(err, 2)})
            continue  # not a true positive: counts as one FP and one FN
        matched_pairs[gi] = pi

    tp = len(matched_pairs)
    for gi, g in enumerate(gts):
        if gi not in matched_pairs:
            details["missing"].append({"key": g["key"], "class_name": g["class_name"]})
    for pi, p in enumerate(preds):
        if pi not in used_p:
            details["extra"].append({"class_name": p["class_name"], "key": p.get("key")})
    return tp, len(preds) - tp, len(gts) - tp, details


def metric(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.
    r = tp / (tp + fn) if tp + fn else 0.
    return {"tp": tp, "fp": fp, "fn": fn, "precision": round(p, 4), "recall": round(r, 4),
            "f1": round(2 * p * r / (p + r), 4) if p + r else 0.}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", type=Path, required=True, help="eval JSON file or submission-format directory")
    ap.add_argument("--gt-root", type=Path, required=True, help="published 200_train_cases directory")
    ap.add_argument("--case-id", action="append", help="restrict to these cases")
    ap.add_argument("--cases-file", type=Path, help="whitespace-separated case ids to restrict to")
    ap.add_argument("--match", choices=["class", "key"], default="class")
    ap.add_argument("--require-fields", action="store_true",
                    help="enforce Name/value equality in key mode (official behaviour). "
                         "Leave off only for pipelines that do not emit those fields yet.")
    ap.add_argument("--pred-glob", default="*_Pred.json",
                    help="file pattern inside --pred when it is a directory "
                         "(e.g. '**/components_partial.json' for run_baseline output)")
    ap.add_argument("--tol", type=float, default=MAX_COORD_PX)
    ap.add_argument("--shrink", type=Path, help="per-class geometry JSON (bbox_convention_v1.json)")
    ap.add_argument("--geometry", choices=["affine", "canonical"], default="affine",
                    help="which table in the config to apply: affine rescale, or canonical symbol size "
                         "with a calibrated centre offset")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    shrink = None
    if args.shrink:
        raw = json.loads(args.shrink.read_text(encoding="utf-8"))
        table = raw.get(args.geometry) or raw.get("transform") or raw.get("canonical") or raw
        shrink = {}
        for name, spec in table.items():
            if not isinstance(spec, (list, tuple, dict)) or name.startswith("_"):
                continue
            if isinstance(spec, dict):
                if "w" in spec:
                    shrink[name] = spec
                else:
                    shrink[name] = {"rw": float(spec.get("rw", 1.)), "rh": float(spec.get("rh", 1.)),
                                    "dxf": float(spec.get("dxf", 0.)), "dyf": float(spec.get("dyf", 0.))}
            else:
                shrink[name] = {"rw": float(spec[0]), "rh": float(spec[1]),
                                "dxf": float(spec[2]) if len(spec) > 2 else 0.,
                                "dyf": float(spec[3]) if len(spec) > 3 else 0.}

    def gt_case_dir(cid):
        return args.gt_root / cid

    def image_height(cid):
        pngs = glob.glob(str(gt_case_dir(cid) / "*.png"))
        if not pngs:
            return None
        return Image.open(pngs[0]).size[1]

    if args.pred.is_dir():
        cache = {}

        def height_for(cid, _root):
            if cid not in cache:
                cache[cid] = image_height(cid)
            return cache[cid]
        preds_by_case = load_pred_dir(args.pred, height_for, None, args.pred_glob)
    else:
        preds_by_case, _raw = load_eval_json(args.pred)

    totals = Counter()
    by_class = defaultdict(lambda: Counter())
    details_all = defaultdict(list)
    cases_used = 0
    skipped = []
    only = set(args.case_id or [])
    if args.cases_file:
        only |= set(args.cases_file.read_text(encoding="utf-8").split())
    for cid in sorted(preds_by_case):
        if only and cid not in only:
            continue
        height = image_height(cid)
        gts = glob.glob(str(gt_case_dir(cid) / "*_target_named.json"))
        if height is None or not gts:
            skipped.append(cid)
            continue
        gts = load_gt_case(gts[0], height)
        preds = [dict(p) for p in preds_by_case[cid]]
        tp, fp, fn, det = match_case(gts, preds, args.match, args.tol, shrink,
                                    ignore_fields=not args.require_fields)
        cases_used += 1
        totals.update({"tp": tp, "fp": fp, "fn": fn})
        # Per-class accounting: on the GT side, every GT key that is absent from
        # det["missing"] was matched; on the prediction side, det["extra"] is the
        # unmatched remainder.
        matched_keys = {g["key"] for g in gts} - {m["key"] for m in det["missing"]}
        for g in gts:
            name = g["class_name"] or g["raw_type"]
            by_class[name]["gt"] += 1
            if g["key"] in matched_keys:
                by_class[name]["tp"] += 1
            else:
                by_class[name]["fn"] += 1
        for e in det["extra"]:
            by_class[e["class_name"]]["fp"] += 1
        for m in det["missing"]:
            details_all["missing"].append({"case": cid, **m})
        for m in det["mismatch"]:
            details_all["mismatch"].append({"case": cid, **m})

    overall = metric(totals["tp"], totals["fp"], totals["fn"])
    power_gt = sum(c["gt"] for k, c in by_class.items() if k in POWER_CLASSES)
    result = {
        "status": "official_rubric_offline_evaluation",
        "gt_root": str(args.gt_root),
        "pred": str(args.pred),
        "match_mode": args.match,
        "tolerance_px": args.tol,
        "shrink": str(args.shrink) if args.shrink else None,
        "cases_used": cases_used,
        "cases_skipped": skipped,
        "note": ("class-mode is an upper bound on the official component F1: it credits components "
                 "whose designator OCR would have failed"),
        "overall": overall,
        "by_class": {k: {**metric(v["tp"], v["fp"], v["fn"]), "gt": v["gt"]}
                     for k, v in sorted(by_class.items(), key=lambda kv: -kv[1]["gt"])},
        "power_gt_count": power_gt,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"cases={cases_used} skipped={len(skipped)} match={args.match} tol={args.tol}px"
          + (f" shrink={args.shrink}" if args.shrink else ""))
    print(f"OVERALL  TP={overall['tp']} FP={overall['fp']} FN={overall['fn']} "
          f"P={overall['precision']:.4f} R={overall['recall']:.4f} F1={overall['f1']:.4f}")
    power = Counter()
    for k, v in by_class.items():
        if k in POWER_CLASSES:
            for f in ("tp", "fp", "fn", "gt"):
                power[f] += v[f]
    m = metric(power["tp"], power["fp"], power["fn"])
    print(f"POWER    TP={m['tp']} FP={m['fp']} FN={m['fn']} P={m['precision']:.4f} "
          f"R={m['recall']:.4f} F1={m['f1']:.4f}  (gt={power['gt']})")
    ordn = metric(totals["tp"] - power["tp"], totals["fp"] - power["fp"], totals["fn"] - power["fn"])
    print(f"ORDINARY TP={ordn['tp']} FP={ordn['fp']} FN={ordn['fn']} P={ordn['precision']:.4f} "
          f"R={ordn['recall']:.4f} F1={ordn['f1']:.4f}")
    print(f"\n{'class':>16} {'gt':>5} {'tp':>5} {'fp':>4} {'fn':>4} {'R':>6}")
    for k, v in list(result["by_class"].items())[:20]:
        print(f"{k:>16} {v['gt']:>5} {v['tp']:>5} {v['fp']:>4} {v['fn']:>4} {v['recall']:>6.2f}")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
