"""Clean up pin numbers the tiled OCR pass left unusable, by re-reading their own box.

Why this exists: on dense IC rows the tiled pass often glues the pin lead onto the
digit, producing "—1" / "一4", which fails the `^\\d{1,4}$` test and drops the pin.
Re-reading *that glyph's own box* at 2x turns it into a clean digit: measured on the
13 released cases, 24 of the 64 missing IC pins recover this way, and 4x adds nothing
over 2x.

Deliberately narrow:
  * only tokens already sitting in the pin's own number band are considered, so the
    geometry decision stays with `pin_text`;
  * only fills a number that is still None, never overwrites observed text;
  * accepts only a cleaned `^\\d{1,4}$` reading above the confidence floor;
  * refuses a number another pin of the same component already uses;
  * records every crop, raw reading and decision for review.
"""
import math
import time

import cv2
import numpy as np

DEFAULTS = {
    "enabled": True,
    "scale": 2,               # 2x measured equal to 4x, at half the cost
    "expand_px": 2.0,         # grow the glyph box slightly so lead pixels do not clip it
    "min_confidence": 0.90,
    "band_scale": 1.5,        # allow the token a little outside the strict band
    "max_pins": 400,
}


def _clean(text):
    """Keep only digits; reject empty or longer than a pin number."""
    digits = "".join(ch for ch in str(text or "") if ch.isdigit())
    if not digits or len(digits) > 4:
        return None
    return digits


def _reader(engine, rgb, box, scale):
    x1, y1, x2, y2 = (max(0, int(math.floor(box[0]))), max(0, int(math.floor(box[1]))),
                      min(rgb.shape[1], int(math.ceil(box[2]))), min(rgb.shape[0], int(math.ceil(box[3]))))
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None, 0., [x1, y1, x2, y2]
    patch = rgb[y1:y2, x1:x2]
    scaled = cv2.resize(patch, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    detected = engine(np.ascontiguousarray(scaled[:, :, ::-1]),
                      use_det=False, use_cls=False, use_rec=True)
    readings = list(zip(detected.txts or [], detected.scores or []))
    if not readings:
        return None, 0., [x1, y1, x2, y2]
    text, score = max(readings, key=lambda row: float(row[1]))
    return str(text), float(score), [x1, y1, x2, y2]


def reread_pin_numbers(rgb, result, texts, cfg, engine=None):
    """Fill missing pin numbers from a 2x re-read of their own glyph box.

    ``result`` is the pin_text document (pin_candidates with top-left ``point`` and a
    designator in ``owner`` once identities are attached); ``texts`` are the merged
    OCR rows. Modifies ``result`` in place and returns an audit document.
    """
    from .pin_text import contexts, local_coordinates
    from .pin_text_v2 import tokens_with_views

    options = dict(DEFAULTS)
    options.update(cfg.get("number_reread") or {})
    audit = {"policy": "pin_number_reread_glyphbox_2x_v1", "enabled": bool(options["enabled"]),
             "scale": options["scale"], "min_confidence": options["min_confidence"],
             "considered": 0, "added": [], "rejected": [], "wall_seconds": 0.0}
    if not options["enabled"]:
        return audit

    if engine is None:
        from .pin_text_ocr import create_engine
        engine, models = create_engine(cfg)
        audit["OCR_models"] = models

    context_map = contexts(result, cfg)
    tokens = tokens_with_views(texts)
    taken = {}
    for pin in result.get("pin_candidates", []):
        if pin.get("number") is not None:
            taken.setdefault(pin.get("owner"), set()).add(str(pin["number"]))

    begin = time.perf_counter()
    for pin in result.get("pin_candidates", []):
        if audit["considered"] >= options["max_pins"]:
            break
        if pin.get("owner_status") != "assigned" or pin.get("number") is not None:
            continue
        owner = pin.get("owner")
        if not owner or str(owner).startswith("UNRESOLVED"):
            continue
        context = context_map.get(pin["candidate_id"])
        if context is None:
            continue
        audit["considered"] += 1
        outer = cfg["number_outer_band_px"]
        inner = cfg["number_inner_band_px"]
        if context["component"].get("type") not in cfg["name_component_types"]:
            outer = cfg["passive_number_outer_band_px"]
            inner = cfg["passive_number_inner_band_px"]
        candidates = []
        for token in tokens:
            tangent, inward = local_coordinates(context, token["bbox"])
            if abs(tangent) > context["tangent_limit"] * options["band_scale"]:
                continue
            if not -outer <= inward <= inner:
                continue
            if token["confidence"] < cfg["min_confidence"] * 0.8:
                continue
            candidates.append((abs(tangent) + .1 * abs(inward), token))
        if not candidates:
            audit["rejected"].append({"candidate_id": pin["candidate_id"], "reason": "no_token_in_band"})
            continue
        candidates.sort(key=lambda row: row[0])
        _, token = candidates[0]
        box = list(token["bbox"])
        grow = options["expand_px"]
        box = [box[0] - grow, box[1] - grow, box[2] + grow, box[3] + grow]
        raw, score, crop = _reader(engine, rgb, box, options["scale"])
        number = _clean(raw)
        record = {"candidate_id": pin["candidate_id"], "owner": owner, "crop": crop,
                  "source_text": token["text"], "raw": raw,
                  "confidence": round(score, 4), "number": number}
        if number is None:
            record["reason"] = "not_a_small_integer"
            audit["rejected"].append(record)
            continue
        if score < options["min_confidence"]:
            record["reason"] = "low_confidence"
            audit["rejected"].append(record)
            continue
        if number in taken.get(owner, set()):
            record["reason"] = "number_already_used_on_this_component"
            audit["rejected"].append(record)
            continue
        if number == str(token["text"]):
            record["reason"] = "already_readable_no_change"
            audit["rejected"].append(record)
            continue
        # a different number than the one the pipeline had to work with: keep it, with evidence
        pin["number"] = number
        pin["number_status"] = "observed_OCR_text"
        pin.setdefault("text_evidence", {})["number"] = {
            "text": number, "raw": raw, "confidence": score, "bbox": token["bbox"],
            "crop": crop, "source_view": f"reread_{options['scale']}x",
            "original_source_text": token["text"]}
        taken.setdefault(owner, set()).add(number)
        audit["added"].append(record)

    audit["wall_seconds"] = round(time.perf_counter() - begin, 3)
    audit["added_count"] = len(audit["added"])
    audit["rejected_count"] = len(audit["rejected"])
    summary = result.get("pin_text_summary")
    if summary is not None:
        summary["number_observed"] = sum(p.get("number") is not None
                                         for p in result.get("pin_candidates", []))
        summary["number_and_name_observed"] = sum(
            p.get("number") is not None and p.get("pinname") is not None
            for p in result.get("pin_candidates", []))
        summary["numbers_from_reread"] = audit["added_count"]
    return audit
