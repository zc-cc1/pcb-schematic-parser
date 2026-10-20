"""Complete pins the heatmap missed, using only geometry the component already shows.

The U-Net pin head misses some pins (measured on the public cases: R4 has 1 of 2 pins,
0036's U1 has 6 of 8).  When the component's designator is known, its pins are scored, so
a missing pin is a guaranteed false negative.

Two rules, both derived from the symbol's own geometry:

  two-pin part  - mirror the detected pin across the body centre; a 2-pin passive is
                  symmetric, so the second pin sits at the opposite end.
  multi-pin IC  - on each side, take the detected pins' spacing and extend the column by
                  one pitch above and below while the extension stays inside a sane
                  multiple of the body size.

Nothing is invented where the symbol gives no evidence: a component with fewer than two
detected pins and no clear axis is left alone.
"""
import math

MIN_PITCH = 4.0
MAX_EXTENSION = 2.0      # do not extend further than this many pitches from the known span


def _centre(box):
    return (float(box[0]) + float(box[2])) / 2.0, (float(box[1]) + float(box[3])) / 2.0


def _clone(pin, point, suffix):
    clone = dict(pin)
    clone["candidate_id"] = f"{pin['candidate_id']}_{suffix}"
    clone["point"] = [float(point[0]), float(point[1])]
    clone["confidence"] = round(float(pin.get("confidence") or 0.0) * 0.5, 4)
    clone["point_source"] = "geometric_completion"
    clone["number"] = None
    clone["pinname"] = None
    clone["number_status"] = "completed"
    clone["pinname_status"] = "completed"
    clone["text_evidence"] = {}
    clone["text_candidates"] = {}
    return clone


def complete_pins(pin_result, two_pin_types=None, ic_types=None):
    """Add geometrically implied pins.  Returns an audit dict."""
    two_pin_types = set(two_pin_types or ["resistor", "capacitor", "inductor"])
    ic_types = set(ic_types or ["ic"])
    components = {c.get("detection_index"): c for c in (pin_result.get("components") or [])}
    by_owner = {}
    for pin in pin_result.get("pin_candidates") or []:
        owner = pin.get("owner_detection_index")
        if owner is None:
            owner = pin.get("owner")
        if owner is None or pin.get("owner_status") != "assigned":
            continue
        by_owner.setdefault(owner, []).append(pin)
    added = []
    audit = {"completed_two_pin": 0, "completed_ic_column": 0, "skipped": 0}
    for owner, pins in by_owner.items():
        component = components.get(owner) or {}
        box = component.get("bbox_top_left")
        kind = str(component.get("type") or "")
        if not box:
            audit["skipped"] += 1
            continue
        cx, cy = _centre(box)
        if kind in two_pin_types and len(pins) == 1:
            source = pins[0]
            mirrored = (2 * cx - source["point"][0], 2 * cy - source["point"][1])
            added.append(_clone(source, mirrored, "mirror"))
            audit["completed_two_pin"] += 1
            continue
        if kind in ic_types and len(pins) >= 2:
            # group by side (left/right of the box centre), then extend along y
            for side in (-1, 1):
                column = [p for p in pins if (p["point"][0] - cx) * side > 0]
                if len(column) < 2:
                    continue
                column.sort(key=lambda p: p["point"][1])
                pitches = [column[i + 1]["point"][1] - column[i]["point"][1]
                           for i in range(len(column) - 1)]
                pitches = [p for p in pitches if p >= MIN_PITCH]
                if not pitches:
                    continue
                pitch = sum(pitches) / len(pitches)
                top = column[0]
                bottom = column[-1]
                limit = MAX_EXTENSION * pitch
                # extend one pitch beyond each end while inside the limit
                candidate_top = top["point"][1] - pitch
                if top["point"][1] - candidate_top <= limit and candidate_top >= float(box[1]) - pitch:
                    if not any(abs(p["point"][1] - candidate_top) < MIN_PITCH and
                               (p["point"][0] - cx) * side > 0 for p in pins):
                        added.append(_clone(top, (top["point"][0], candidate_top), "extend_top"))
                        audit["completed_ic_column"] += 1
                candidate_bottom = bottom["point"][1] + pitch
                if candidate_bottom - bottom["point"][1] <= limit and candidate_bottom <= float(box[3]) + pitch:
                    if not any(abs(p["point"][1] - candidate_bottom) < MIN_PITCH and
                               (p["point"][0] - cx) * side > 0 for p in pins):
                        added.append(_clone(bottom, (bottom["point"][0], candidate_bottom), "extend_bottom"))
                        audit["completed_ic_column"] += 1
    if added:
        pin_result.setdefault("pin_candidates", []).extend(added)
    audit["pins_added"] = len(added)
    return audit
