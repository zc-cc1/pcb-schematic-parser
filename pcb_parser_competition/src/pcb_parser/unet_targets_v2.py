"""Wire/pin supervision targets, v2.

Differences from `pcb_parser.unet_data.annotation_targets` (v1, kept unchanged):

1. Edge geometry fallback.  v1 reads only `edge.keyPointPaths.wire`.  692 of the
   14,895 physical wire edges have that dict empty but do carry a complete
   pixel-level chain in `edge.coordinates`; v1 silently treated those wires as
   background and rejected every page that contained one.  v2 falls back to
   `coordinates` **only** for edges whose keyPointPaths are missing.

   It must not fall back for edges that do have keyPointPaths: on those edges
   `coordinates` additionally carries filled net polygons, and stroking them
   would paint solid blobs that do not exist in the image.

2. Per-channel ignore masks instead of page-level rejection.  An unowned root
   pin, an out-of-bounds tip or an edge with no usable geometry marks the
   surrounding pixels as "do not score" instead of discarding the whole page.

3. Pin targets use a disk-flavoured Gaussian so the decay length matches the
   official 5 px pin tolerance rather than a fixed sigma.

Nothing here reads the rendered PNG.  Geometry comes from the annotation only.
"""
import math

import numpy as np
from PIL import Image, ImageDraw

from .unet_data import point, polylines

DEFAULT_LINE_WIDTH = 3
DEFAULT_PIN_SIGMA = 2.5
IGNORE_HALO = 5
UNGCOREABLE_HALO = 8


def _blank(width, height):
    return Image.new("L", (width, height), 0)


def _draw_path(draw, path, width):
    draw.line([(round(x), round(y)) for x, y in path], fill=255, width=width)


def build_targets(annotation, width, height, line_width=DEFAULT_LINE_WIDTH, pin_sigma=DEFAULT_PIN_SIGMA):
    """Return (wire_mask, pin_heatmap, wire_ignore, pin_ignore, tips, stats)."""
    if annotation["canvas"]["width"] != width or annotation["canvas"]["height"] != height:
        raise ValueError("Annotation canvas and PNG dimensions differ")

    wire = _blank(width, height)
    wire_ignore = _blank(width, height)
    pin = np.zeros((height, width), dtype=np.float32)
    pin_ignore = _blank(width, height)
    wire_draw = ImageDraw.Draw(wire)
    wire_ignore_draw = ImageDraw.Draw(wire_ignore)
    pin_ignore_draw = ImageDraw.Draw(pin_ignore)

    stats = {
        "physical_edges": 0,
        "flywires_excluded": 0,
        "edges_using_key_point_paths": 0,
        "edges_recovered_from_coordinates": 0,
        "edges_without_geometry": 0,
        "edges_skipped_out_of_bounds": 0,
        "wire_positive_pixels": 0,
        "pin_tips": 0,
        "unowned_root_pins_ignored": 0,
        "invalid_pin_tips_ignored": 0,
    }

    for edge in annotation.get("edges", {}).values():
        if edge.get("state") == "deleted":
            continue
        if edge.get("isFlywire") is True or str(edge.get("isFlywire")).lower() == "true":
            stats["flywires_excluded"] += 1
            continue
        if (edge.get("category") or "wire") != "wire":
            continue
        stats["physical_edges"] += 1

        key_paths = list(polylines((edge.get("keyPointPaths") or {}).get("wire") or []))
        if key_paths:
            stats["edges_using_key_point_paths"] += 1
            paths = key_paths
        else:
            coordinates = edge.get("coordinates") or []
            if len(coordinates) < 2:
                stats["edges_without_geometry"] += 1
                continue
            paths = [[(float(p[0]), float(p[1])) for p in coordinates]]
            stats["edges_recovered_from_coordinates"] += 1

        for path in paths:
            if any(not (0 <= x < width and 0 <= y < height) for x, y in path):
                # Never guess the clipped part, but do not score the neighbourhood
                # either: 0229/E_8 ends exactly on the canvas edge, 0259/E_12 runs
                # 4 px off it.  Without the ignore ring these become unlearnable FNs.
                stats["edges_skipped_out_of_bounds"] += 1
                _draw_path(wire_ignore_draw, path, line_width + 2 * IGNORE_HALO)
                continue
            _draw_path(wire_draw, path, line_width)

    # --- pins -------------------------------------------------------------
    tips, owned = [], set()
    for component_id, component in annotation.get("components", {}).items():
        if component.get("state") == "deleted":
            continue
        for pin_id, local in (component.get("pins") or {}).items():
            if local.get("state") == "deleted":
                continue
            owned.add(pin_id)
            merged = {**annotation.get("pins", {}).get(pin_id, {}), **local}
            position = point(merged.get("tip"))
            if position is None or not (0 <= position[0] < width and 0 <= position[1] < height):
                stats["invalid_pin_tips_ignored"] += 1
                if position is not None:
                    _disk(pin_ignore_draw, position, UNGCOREABLE_HALO)
                continue
            tips.append({"component_source_id": component_id, "pin_source_id": pin_id,
                         "point": [position[0], position[1]]})

    for pin_id, root in annotation.get("pins", {}).items():
        if root.get("state") == "deleted" or pin_id in owned:
            continue
        stats["unowned_root_pins_ignored"] += 1
        position = point(root.get("tip"))
        if position is not None:
            _disk(pin_ignore_draw, position, UNGCOREABLE_HALO)
            # the wire that used to attach here is genuinely unknown
            _disk(wire_ignore_draw, position, IGNORE_HALO)

    _splat_pins(pin, tips, pin_sigma)
    stats["pin_tips"] = len(tips)
    stats["wire_positive_pixels"] = int(np.count_nonzero(np.asarray(wire)))
    return wire, Image.fromarray(np.round(pin * 255).astype(np.uint8)), wire_ignore, pin_ignore, tips, stats


def _disk(draw, center, radius):
    x, y = center
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=255)


def _splat_pins(heat, tips, sigma):
    radius = max(1, int(math.ceil(sigma * 3)))
    for tip in tips:
        x, y = tip["point"]
        x1, x2 = max(0, math.floor(x) - radius), min(heat.shape[1], math.floor(x) + radius + 1)
        y1, y2 = max(0, math.floor(y) - radius), min(heat.shape[0], math.floor(y) + radius + 1)
        yy, xx = np.mgrid[y1:y2, x1:x2]
        gaussian = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))
        heat[y1:y2, x1:x2] = np.maximum(heat[y1:y2, x1:x2], gaussian)


def render_review(image_path, wire, pin_heatmap, wire_ignore, pin_ignore, tips, max_side=1400):
    """Composite label review: green = wire supervision, blue = pin tips,
    yellow = ignored.  Diagnostic only; never a model input."""
    with Image.open(image_path) as source:
        rgb = source.convert("RGB")
        scale = min(1.0, max_side / max(rgb.size))
        rgb = rgb.resize((max(1, round(rgb.width * scale)), max(1, round(rgb.height * scale))),
                         Image.Resampling.LANCZOS) if scale < 1.0 else rgb
        size = rgb.size
        canvas = np.asarray(rgb).astype(np.float32)
        pin_binary = Image.fromarray((np.asarray(pin_heatmap) >= 128).astype(np.uint8) * 255)
        for mask, colour in ((wire, (40, 190, 90)), (pin_binary, (40, 120, 230))):
            selected = np.asarray(mask.resize(size, Image.Resampling.NEAREST)) > 0
            canvas[selected] = 0.25 * canvas[selected] + 0.75 * np.array(colour, dtype=np.float32)
        combined = Image.fromarray(np.maximum(np.asarray(wire_ignore), np.asarray(pin_ignore)))
        ignored = np.asarray(combined.resize(size, Image.Resampling.NEAREST)) > 0
        canvas[ignored] = 0.35 * canvas[ignored] + 0.65 * np.array((245, 200, 40), dtype=np.float32)
    return Image.fromarray(canvas.astype(np.uint8))
