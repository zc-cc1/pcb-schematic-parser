"""Read a pin's printed NUMBER from the tiny strip beside its lead.

On these schematics the pin **name** is printed inside the body and the pin **number** is
printed just outside the body edge, a few pixels above the lead line.  The global OCR reads
the names reliably and the numbers only sometimes: on the 13 released pages 35 of the 40
"missing number" pins do have ink in that strip, but the glyph is ~8 px tall and usually
touches the body outline, so the page-level pass never segments it.

The number does NOT sit at the body edge: it is printed above the **outer end of the lead**,
where the wire arrives - 30-50 px from the body on these pages.  That is why the page-level
OCR band (26 px outside the body) never saw it.  This pass therefore centres the window on the
pin's own TIP (which the colour scan already measures) and reads a 44x20 px strip above it at
3x/4x/6x in both orientations, accepting a reading only when it is a bare 1-3 digit number,
is seen by at least two views (or one very confident view that the part's library entry also
lists) and does not collide with a number already used by the same component.
"""
import math
import re

import cv2
import numpy as np

NUMBER = re.compile(r"^\d{1,3}$")
BALL = re.compile(r"^[A-Z]{1,2}\d{1,3}$")
DEFAULT = {
    "scales": [3.0, 4.0, 6.0],
    # window centred on the pin tip: +-along along the lead, -up..+down above it
    "along_px": 22,
    "up_px": 18,
    "down_px": 2,
    "min_confidence": 0.9,
    "single_view_confidence": 0.99,
    "min_views": 2,
    "component_types": ["ic", "connector", "block"],
    "ball_component_types": ["ic", "connector", "block"],
}


def _side(pin_point, box):
    x, y = pin_point
    x1, y1, x2, y2 = box
    return min([("left", abs(x - x1)), ("right", abs(x - x2)),
                ("top", abs(y - y1)), ("bottom", abs(y - y2))], key=lambda row: row[1])[0]


def strip_crop(point, cfg):
    """Axis-aligned window centred on the pin tip; the number is printed just above it."""
    x, y = point
    return (int(x - cfg["along_px"]), int(y - cfg["up_px"]),
            int(x + cfg["along_px"]), int(y + cfg["down_px"]))


def inside_crop(point, box, side, cfg):
    """Window INSIDE the body on the pin's own row.

    A header prints its pins as bare digits inside the body ("1 2 3"), and the reference uses
    those digits as the pin number too.  Only used when the outside strip found nothing.
    """
    depth = float(cfg.get("inside_depth_px", 34))
    half = float(cfg.get("inside_half_px", 8))
    x, y = point
    x1, y1, x2, y2 = box
    if side == "left":
        return (int(x1 + 3), int(y - half), int(x1 + depth), int(y + half))
    if side == "right":
        return (int(x2 - depth), int(y - half), int(x2 - 3), int(y + half))
    if side == "top":
        return (int(x - half), int(y1 + 3), int(x + half), int(y1 + depth))
    return (int(x - half), int(y2 - depth), int(x + half), int(y2 - 3))


def choose(readings, library_numbers, cfg):
    """Majority spelling, with a single-view escape hatch only when the part corroborates it."""
    counts, best = {}, {}
    for text, confidence in readings:
        counts[text] = counts.get(text, 0) + 1
        best[text] = max(best.get(text, 0.0), confidence)
    if not counts:
        return None, "no_reading"
    ranked = sorted(counts.items(), key=lambda row: (-row[1], -best[row[0]]))
    text, votes = ranked[0]
    if len(ranked) > 1 and votes == ranked[1][1] and abs(best[text] - best[ranked[1][0]]) < 0.02:
        return None, "ambiguous"
    if votes >= cfg["min_views"]:
        return text, "multi_view"
    if best[text] >= cfg["single_view_confidence"]:
        # one view, but nothing contradicts it: no other spelling was produced at all
        if library_numbers and text in library_numbers:
            return text, "single_view_library_confirmed"
        if len(counts) == 1 and cfg.get("single_view_no_conflict", True):
            return text, "single_view_no_conflict"
    return None, "insufficient_evidence"


def local_pitch(pins, pin, record):
    """Distance to the nearest neighbouring pin on the same side of the same component."""
    x, y = pin["point"]
    box = record.get("bbox_top_left") or [0, 0, 0, 0]
    edges = {"left": abs(x - box[0]), "right": abs(x - box[2]),
             "top": abs(y - box[1]), "bottom": abs(y - box[3])}
    side = min(edges, key=edges.get)
    best = None
    for other in pins:
        if other is pin or other.get("owner_detection_index") != record.get("detection_index"):
            continue
        ox, oy = other["point"]
        o_edges = {"left": abs(ox - box[0]), "right": abs(ox - box[2]),
                   "top": abs(oy - box[1]), "bottom": abs(oy - box[3])}
        if min(o_edges, key=o_edges.get) != side:
            continue
        delta = abs(oy - y) if side in ("left", "right") else abs(ox - x)
        if delta > 2 and (best is None or delta < best):
            best = delta
    return best


def read_numbers(rgb, pins_doc, records, cfg, engine, library_numbers=None, graph=None,
                 scope="unattached"):
    """Add `number` to IC pins that have none, from the printed strip. Returns an audit."""
    cfg = {**DEFAULT, **(cfg or {})}
    height = rgb.shape[0]
    by_index = {r["detection_index"]: r for r in records}
    # key collision is checked on the same key the exporter uses (the component id string),
    # otherwise a duplicate hyperGraph member reaches the export and aborts the whole page
    def owner_key(pin):
        return pin.get("owner") or pin.get("owner_detection_index")
    used = {}
    for pin in pins_doc["pin_candidates"]:
        if pin.get("owner_status") == "assigned" and str(pin.get("number") or "").strip():
            used.setdefault(owner_key(pin), set()).add(str(pin["number"]).strip())
    # scope='unattached' only numbers pins that are in NO network yet: those add score without
    # reshaping a net that already exists, which is what costs connectivity precision.
    busy = set()
    if graph and str(scope).lower() in {"unattached", "new", "only_unattached"}:
        for net in graph.get("networks") or []:
            busy.update(net.get("pin_candidate_ids") or [])
    adopted, rejected = [], []
    for pin in pins_doc["pin_candidates"]:
        if pin.get("owner_status") != "assigned" or str(pin.get("number") or "").strip():
            continue
        if pin.get("candidate_id") in busy:
            continue
        record = by_index.get(pin.get("owner_detection_index")) or {}
        if str(record.get("type")) not in cfg["component_types"]:
            continue
        box = record.get("bbox_top_left")
        if not box or len(box) != 4:
            continue
        x, y = pin["point"]
        image_point = (float(x), float(y))
        side = _side(image_point, [float(v) for v in box])
        # A dense row (connector, BGA) has a pitch close to the default window height, so the
        # window would also cover the neighbour's number and the reading becomes ambiguous.
        # Scale the height with the local pitch instead of using one value for every part.
        local = {**cfg}
        pitch = local_pitch(pins_doc["pin_candidates"], pin, record)
        if pitch and cfg.get("pitch_aware_window", True):
            share = float(cfg.get("pitch_window_fraction", 0.45))
            local["up_px"] = max(float(cfg.get("up_px_min", 6)), min(float(cfg["up_px"]), pitch * share))
        x1, y1, x2, y2 = strip_crop(image_point, local)
        patch = rgb[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
        if patch.size == 0:
            rejected.append({"candidate_id": pin["candidate_id"], "reason": "empty_crop"})
            continue
        def ocr_readings(region):
            collected = []
            if region.size == 0:
                return collected
            for rotation in cfg.get("rotations", [0, 90]):
                upright = region if rotation == 0 else cv2.rotate(region, cv2.ROTATE_90_CLOCKWISE)
                for scale in cfg["scales"]:
                    image = cv2.resize(upright, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                    result = engine(np.ascontiguousarray(image[:, :, ::-1]), use_det=True, use_cls=True, use_rec=True)
                    for text, confidence in zip(result.txts or [], result.scores or []):
                        text = str(text).strip()
                        if not (NUMBER.match(text) or BALL.match(text)):
                            continue
                        if float(confidence) < cfg["min_confidence"]:
                            continue
                        collected.append((text, float(confidence)))
            return collected

        readings = ocr_readings(patch)
        source = "strip"
        # One window cannot cover every symbol: a header prints its digits inside the body,
        # a connector prints them to the left of the lead.  Measured on 15 pins, a single
        # rectangle read 5/10 connector pins, three windows read 8/10 while keeping 5/5 of the
        # control pins.  Each probe is tried in order and the first admissible reading wins.
        if cfg.get("probe_directions") and not any(text == str(pin.get("pinname") or "").strip()
                                                   for text, _c in readings):
            for offsets in cfg["probe_directions"]:
                dx0, dy0, dx1, dy1 = offsets
                region = rgb[max(0, int(y) + dy0):max(0, int(y) + dy1),
                             max(0, int(x) + dx0):max(0, int(x) + dx1)]
                extra = ocr_readings(region)
                if extra:
                    trial, trial_why = choose(extra, (library_numbers or {}).get(record.get("component_id")) or set(), cfg)
                    if trial is not None:
                        readings = extra
                        source = f"probe{offsets}"
                        break
        numbers = (library_numbers or {}).get(record.get("component_id")) or set()
        text, why = choose(readings, numbers, cfg)
        probe_text = text
        if text is None and source != "strip" and readings:
            text, why = choose(readings, numbers, cfg)
        if text is None and cfg.get("inside_digit_fallback", True):
            inner = inside_crop(image_point, [float(v) for v in box], side, cfg)
            region = rgb[max(0, inner[1]):max(0, inner[3]), max(0, inner[0]):max(0, inner[2])]
            inside_readings = ocr_readings(region)
            text, why = choose(inside_readings, numbers, cfg)
            if text is not None:
                source = "inside_body"
                readings = inside_readings
        if text is None:
            rejected.append({"candidate_id": pin["candidate_id"], "reason": why,
                             "readings": readings[:4]})
            continue
        # A package cannot have a pin number far beyond its own pin count: on a 4-pin header a
        # stray "11" was adopted and became a false positive.  Numeric reads are bounded by the
        # number of candidates that component actually has.
        if text.isdigit():
            siblings = sum(1 for other in pins_doc["pin_candidates"]
                           if other.get("owner") == pin.get("owner")
                           and other.get("owner_status") == "assigned")
            limit = max(4, int(round(1.5 * siblings)))
            if int(text) > limit:
                rejected.append({"candidate_id": pin["candidate_id"],
                                 "reason": f"number_beyond_pin_count({text}>{limit})",
                                 "readings": readings[:4]})
                continue
        taken = used.setdefault(owner_key(pin), set())
        if text in taken:
            rejected.append({"candidate_id": pin["candidate_id"], "reason": "number_already_used",
                             "readings": readings[:4]})
            continue
        taken.add(text)
        pin["number"] = text
        pin["number_status"] = "targeted_OCR_number_strip"
        pin["number_source"] = "pin_number_strip_OCR"
        # the graph keeps its own copy of every pin's number and the stage asserts that the
        # two never disagree, so the attachment is updated with the reading
        if graph:
            for attachment in graph.get("pin_attachments") or []:
                if attachment.get("candidate_id") == pin["candidate_id"]:
                    attachment["number"] = text
                    break
        adopted.append({"candidate_id": pin["candidate_id"], "component_id": record.get("component_id"),
                        "number": text, "how": why, "source": source, "readings": readings[:4]})
    # A header prints bare digits and the reference uses them as BOTH name and number, so a
    # component whose pins produced no name at all gets its number copied into the name.  When
    # the reference name is something else this costs nothing: a blank name and a wrong name
    # both miss, and the pin is exported either way.
    if cfg.get("digit_name_from_number", True):
        by_component = {}
        for row in adopted:
            by_component.setdefault(row.get("component_id"), []).append(row)
        for component, rows in by_component.items():
            owned = [pin for pin in pins_doc["pin_candidates"] if pin.get("owner") == component]
            if any(str(pin.get("pinname") or "").strip() for pin in owned):
                continue
            for row in rows:
                pin = next(item for item in owned if item["candidate_id"] == row["candidate_id"])
                pin["pinname"] = str(pin.get("number"))
                pin["pinname_status"] = "digit_name_from_number"
                row["name_from_number"] = True

    # Final safety net: the export keys a pin by (component id, number), so a number that
    # collides with another pin of the same component would abort the whole page.  Drop our
    # own reading rather than risk that.
    from collections import Counter
    counts = Counter((str(pin.get("owner")), str(pin.get("number")).strip())
                     for pin in pins_doc["pin_candidates"]
                     if pin.get("owner_status") == "assigned" and str(pin.get("number") or "").strip())
    dropped = []
    for row in reversed(adopted):
        pin = next(item for item in pins_doc["pin_candidates"] if item["candidate_id"] == row["candidate_id"])
        key = (str(pin.get("owner")), str(pin.get("number")).strip())
        if counts.get(key, 0) > 1:
            counts[key] -= 1
            pin["number"] = None
            pin["number_status"] = "no_reliable_text"
            pin.pop("number_source", None)
            adopted.remove(row)
            dropped.append({**row, "reason": "duplicate_pin_key_dropped"})
    rejected.extend(dropped)
    return {"adopted": adopted, "rejected": rejected,
            "summary": {"pins_numbered_from_strip": len(adopted), "pins_rejected": len(rejected)}}
