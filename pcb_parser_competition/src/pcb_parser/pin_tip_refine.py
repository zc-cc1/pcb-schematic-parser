"""Refine pin points towards the contest convention: body edge + 10 px.

Why: the reference places a pin point exactly 10.00 px outside its component's BODY box
(217 of 378 reference pins are exactly 10.00 px out; the per-case median is 10.00 for all
ten public cases).  Our detector box is an OUTER box that already contains the leads, so
our points sit at the outer end of each lead and overshoot by (lead length - 10).

`refine_points()` walks each pin's lead inward until the ink thickens, measures that
lead's length, and moves the point to `lead end - (length - 10)`.  Measured effect on the
public cases: 35 -> 42 matched pins out of 89 evaluable, so it is applied only when the
configuration enables it.
"""
import math

import numpy as np

STUB = 10.0
THICK = 6
MAX_WALK = 45


def _run_width(ink, x, y, axis):
    if not (0 <= y < ink.shape[0] and 0 <= x < ink.shape[1]) or not ink[y, x]:
        return 0
    total = 1
    if axis == 0:
        for step in (1, -1):
            yy = y + step
            while 0 <= yy < ink.shape[0] and ink[yy, x]:
                total += 1
                yy += step
    else:
        for step in (1, -1):
            xx = x + step
            while 0 <= xx < ink.shape[1] and ink[y, xx]:
                total += 1
                xx += step
    return total


def _body_edge(ink, tip, centre):
    dx, dy = centre[0] - tip[0], centre[1] - tip[1]
    norm = math.hypot(dx, dy)
    if norm < 1e-6:
        return None
    ux, uy = dx / norm, dy / norm
    axis = 0 if abs(ux) >= abs(uy) else 1
    for step in range(2, MAX_WALK):
        x = int(round(tip[0] + ux * step))
        y = int(round(tip[1] + uy * step))
        if not (0 <= y < ink.shape[0] and 0 <= x < ink.shape[1]):
            return None
        if _run_width(ink, x, y, axis) >= THICK:
            return (x, y)
    return None


def refine_points(pin_result, gray, stub=STUB, enabled_types=None):
    """Move pin points to `body edge + stub`, in place.  Returns an audit dict.

    `pin_result['size']` is [width, height]; `pin_result` pin points are top-left image
    coordinates, the same convention as `gray`, so no flip is applied here.  The exported
    bottom-left conversion stays in one place (the exporter).
    """
    height = gray.shape[0]
    ink = gray < 150
    components = {c.get("detection_index"): c for c in (pin_result.get("components") or [])}
    moved = kept = skipped = 0
    for pin in pin_result.get("pin_candidates") or []:
        component = components.get(pin.get("owner_detection_index")) or components.get(pin.get("owner"))
        box = (component or {}).get("bbox_top_left")
        if not box or (enabled_types and str(component.get("type")) not in enabled_types):
            skipped += 1
            continue
        cx = (float(box[0]) + float(box[2])) / 2.0
        cy = (float(box[1]) + float(box[3])) / 2.0
        tip = (float(pin["point"][0]), float(pin["point"][1]))
        edge = _body_edge(ink, tip, (cx, cy))
        if edge is None:
            skipped += 1
            continue
        length = math.hypot(edge[0] - tip[0], edge[1] - tip[1])
        reach = math.hypot(cx - tip[0], cy - tip[1])
        if reach < 1e-6 or length <= 0:
            skipped += 1
            continue
        delta = length - stub
        if abs(delta) < 1.0:
            kept += 1
            continue
        ux, uy = (cx - tip[0]) / reach, (cy - tip[1]) / reach
        pin["point"] = [tip[0] + ux * delta, tip[1] + uy * delta]
        pin["point_source"] = "body_edge_plus_stub"
        pin["lead_length_px"] = round(length, 2)
        moved += 1
    return {"points_moved": moved, "points_within_tolerance_of_stub": kept,
            "points_without_lead_evidence": skipped, "stub_px": stub}
