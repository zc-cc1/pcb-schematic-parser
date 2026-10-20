"""Export-time component geometry: map detector boxes to the official GT convention.

Two conventions are supported, because the organiser's GT and our detector do not
describe the same rectangle (see docs/detection_v4_plan.md section 1.3):

``canonical``
    Replace the detected extent with the symbol's natural size and keep a
    calibrated centre. This matches how the published GT boxes are drawn (a
    resistor is always 8x20 or 20x8) and is what the 5th official evaluation
    implies: the submitted sizes were already canonical, only the centres were
    20-34 px off. Measured on the 40-page public subset: component F1
    .8641 -> .8999, power F1 .9369 -> .9616.

``affine``
    Rescale about the centre (rw/rh) with a size-relative shift (dxf/dyf). This
    is the safe fallback for classes whose size is not fixed, and it is what the
    first calibration pass produced.

Both tables are produced by ``scripts/calibrate_bbox_convention.py``. Nothing
here is allowed to invent a size for a variable-size symbol (an IC is 51x58 on
one page and 247x372 on another), so the config lists those classes as excluded
and this module leaves their detected extent untouched.

Disabled by default: a config without a ``bbox_convention`` block behaves exactly
as before.
"""
from copy import deepcopy
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Raw official type -> detector class name. Export may run with either
# vocabulary depending on the class_mapping in the run config.
RAW_TO_CLASS = {
    "r": "resistor", "c": "capacitor", "l": "inductor", "box": "ic", "d": "diode",
    "led": "led", "gnd": "ground", "v": "voltage_source", "vcc": "voltage_source",
    "bjt": "transistor", "pin": "pin_port", "net_bidirection": "net_bidirectional",
    "battary": "battery", "transfomer": "transformer", "block": "raw_block",
    "opt": "optocoupler", "dc": "raw_dc", "ant": "antenna", "m": "raw_m",
}


def class_of(kind):
    """Resolve a possibly-raw type string to the internal detector class name."""
    if not kind:
        return kind
    key = str(kind).strip().lower()
    return RAW_TO_CLASS.get(key, kind)


def load_type_map(cfg, root=None):
    """Return the export type table, or ``None`` when the block is absent/off.

    Separate from ``class_mapping`` on purpose: ``class_mapping`` also feeds the
    network/label stages (``vbus_in`` and friends), so changing it to the official
    raw vocabulary would break them. Only the exported string is remapped here.
    """
    block = (cfg or {}).get("export_vocabulary") or {}
    if not block.get("enabled"):
        return None
    if "types" in block:
        return block["types"]
    path = block.get("path")
    if not path:
        return None
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = Path(root or ROOT) / candidate
    if not candidate.exists():
        return None
    return json.loads(candidate.read_text(encoding="utf-8")).get("types") or None


def export_type(detector_class, fallback, table):
    """Official raw type string for a component.

    Keyed on the *detector* class, not on the mapped stage type, because the stage
    type collapses several detector classes into one label type
    (``pin_port``/``testpoint``/``net_input`` -> ``vbus_in``) and the official GT
    treats those as three different types. Collapsing them makes a correct
    detection uncountable no matter how good the box is.
    """
    if not table:
        return fallback
    if detector_class in table:
        return table[detector_class]
    return table.get(class_of(fallback), fallback)


def resolve_spec(table, kind):
    """Look a class up by detector name, tolerating the raw official vocabulary."""
    if not table or not kind:
        return None
    if kind in table:
        return table[kind]
    key = str(kind).strip().lower()
    if key in table:
        return table[key]
    alias = RAW_TO_CLASS.get(key)
    return table.get(alias) if alias else None


def load_table(cfg, root=None):
    """Return ``(mode, table)`` from a run config, or ``(None, None)`` when off.

    The config block accepts either an inline table or a path to the JSON written
    by ``scripts/calibrate_bbox_convention.py``.
    """
    block = (cfg or {}).get("bbox_convention") or {}
    if not block.get("enabled"):
        return None, None
    mode = block.get("mode", "canonical")
    if "table" in block:
        return mode, block["table"]
    path = block.get("path")
    if not path:
        return None, None
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = Path(root or ROOT) / candidate
    if not candidate.exists():
        return None, None
    data = json.loads(candidate.read_text(encoding="utf-8"))
    return mode, data.get(mode) or data.get("transform") or {}


def load_scale_reference(cfg, root=None):
    """Return ``(reference_widths, (lo, hi))`` for page-scale estimation.

    The canonical table holds sizes measured on pages drawn at the usual scale. A
    page drawn at 3x makes every canonical box a third of the size it should be,
    and the 20 px tolerance is no longer enough to absorb it: measured bbox accuracy
    on a resampled page goes 1.00 -> 0.86 -> 0.00 at 1x/2x/3x, while scaling the
    table by the known factor holds at 0.86. The detector's own boxes carry that
    factor (median capacitor width 58.5 -> 176.7 between 1x and 3x), so the scale is
    estimated from the detections and no GT is needed.
    """
    block = (cfg or {}).get("bbox_convention") or {}
    if not block.get("enabled"):
        return None, None
    data = {}
    path = block.get("path")
    if path:
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = Path(root or ROOT) / candidate
        if candidate.exists():
            data = json.loads(candidate.read_text(encoding="utf-8"))
    if not data and isinstance(block.get("table"), dict):
        data = block["table"]
    page_scale = data.get("page_scale") or {}
    reference = page_scale.get("reference_detected_widths") or {}
    limits = page_scale.get("clamp") or [0.5, 8.0]
    return (reference or None), (float(limits[0]), float(limits[1]))


def estimate_page_scale(boxes_by_class, reference, limits=(0.5, 8.0), min_samples=3):
    """Median ratio of detected ruler widths to the reference widths.

    ``boxes_by_class`` maps a class name to a list of detected widths on this page.
    Returns ``(scale, provenance)``; ``scale`` is 1.0 when nothing usable was found,
    which reproduces the old absolute behaviour rather than guessing.
    """
    if not reference or not boxes_by_class:
        return 1.0, None
    ratios = {}
    for name, ref in reference.items():
        widths = [w for w in (boxes_by_class.get(name) or []) if w > 0]
        if len(widths) < min_samples or not ref:
            continue
        ratios[name] = round(float(statistics.median(widths)) / float(ref), 3)
    if not ratios:
        return 1.0, None
    raw = float(statistics.median(ratios.values()))
    scale = min(max(raw, limits[0]), limits[1])
    return round(scale, 3), {"raw": round(raw, 3), "clamped": scale != round(raw, 3),
                             "rulers": ratios}


def apply_to_box(box, kind, mode, table, scale=1.0):
    """Transform a top-left pixel box. Returns ``(box, provenance)``.

    ``provenance`` is ``None`` when nothing was applied, so callers can record
    exactly which components were rewritten at export time.
    """
    spec = resolve_spec(table, kind)
    if not spec:
        return list(box), None
    x1, y1, x2, y2 = (float(v) for v in box)
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        return list(box), None
    cx = (x1 + x2) / 2. + float(spec.get("dxf", 0.)) * w
    cy = (y1 + y2) / 2. + float(spec.get("dyf", 0.)) * h
    if mode == "canonical" and "w" in spec:
        # `w/h` hold the orientation that is common in the reference, `w_v/h_v` the
        # other one (calibrate_bbox_convention groups GT boxes by `h > w`).  Pick by
        # comparing the DETECTION's orientation with the base entry's, never by the
        # detection alone: testing only `h > w` transposed every landscape symbol of
        # a portrait-base class, so a 20x8 resistor came out 8x20 and its corners
        # moved 6 px - that is what cost case 12/13 in the 7th/8th evaluation.
        # A near-square detection keeps the base entry.
        nw, nh = float(spec["w"]), float(spec["h"])
        if "w_v" in spec and (h > w) != (float(spec["h"]) > float(spec["w"])):
            nw, nh = float(spec["w_v"]), float(spec["h_v"])
        nw, nh = nw * scale, nh * scale
    elif mode == "canonical":
        # canonical table entry without a size: fall back to the affine numbers
        nw, nh = w * float(spec.get("rw", 1.)), h * float(spec.get("rh", 1.))
    else:
        nw, nh = w * float(spec.get("rw", 1.)), h * float(spec.get("rh", 1.))
    clamped = False
    if mode == "canonical" and spec.get("clamp_to_detection", True):
        # The glyph is drawn inside the detected cell, so a centre shift that
        # pushes the symbol box outside the cell is an artefact of fitting the
        # offset on a differently-sized source box. Measured: without the clamp a
        # ground symbol lands ~7 px low on dense pages.
        if nw <= w:
            clamped_cx = min(max(cx, x1 + nw / 2.), x2 - nw / 2.)
            clamped = clamped or clamped_cx != cx
            cx = clamped_cx
        if nh <= h:
            clamped_cy = min(max(cy, y1 + nh / 2.), y2 - nh / 2.)
            clamped = clamped or clamped_cy != cy
            cy = clamped_cy
    out = [cx - nw / 2., cy - nh / 2., cx + nw / 2., cy + nh / 2.]
    return out, {"mode": mode, "size": [round(nw, 2), round(nh, 2)], "scale": round(scale, 3),
                 "offset_frac": [spec.get("dxf", 0.), spec.get("dyf", 0.)],
                 "clamped_to_detection": clamped}


def apply_if_enabled(box, kind, cfg, root=None):
    """Convenience wrapper for callers that hold the whole run config."""
    mode, table = cfg if isinstance(cfg, tuple) else load_table(cfg, root)
    if not mode:
        return list(box), None
    return apply_to_box(box, kind, mode, table)


def summarise(enabled, applied):
    """Small provenance blob for the run summary."""
    return {"enabled": bool(enabled), "components_rewritten": len(applied),
            "classes": sorted({a["class"] for a in applied}) if applied else []}
