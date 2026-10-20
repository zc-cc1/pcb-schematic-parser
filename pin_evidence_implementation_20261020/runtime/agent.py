#!/usr/bin/env python3
"""Competition batch entrypoint. Inference reads PNGs and bundled assets only."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import traceback

from adapter import COMPONENT_OUTPUT_POLICY, to_sample, to_target, validate_target

ROOT = Path(__file__).resolve().parent
EMPTY = {'components': {}, 'pins': {}, 'nets': {}}


def write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temporary.replace(path)


def finalize_power_wire_export(target, stage, image_sha256):
    """Finish target-format wires using only this image's saved model artifacts."""
    if os.environ.get('PCB_POWER_WIRE_EXPORT', '1').strip().lower() in {'0', 'off', 'false'}:
        return target, {'enabled': False}
    from pcb_parser.power_wire_export import restore_power_wires
    graph = json.loads((stage/'export/engineering_graph_proposals.json').read_text(encoding='utf-8'))
    pins = json.loads((stage/'export/engineering_pin_text_result.json').read_text(encoding='utf-8'))
    if graph.get('image_sha256') != image_sha256:
        raise ValueError('Power wire graph differs from the current image')
    cleanup = cleanup_wire_edges if os.environ.get('PCB_WIRE_CLEANUP', '1').strip().lower() \
        not in {'0', 'off', 'false'} else None
    restored, audit = restore_power_wires(target, graph, pins, cleanup)
    validate_target(restored)
    return restored, {**audit, 'enabled': True, 'image_sha256': image_sha256}


def finalize_pin_field_recovery(target, stage, image_rgb, image_sha256,
                                raw_pins, lead_px=18, tip_mode="color", reader=None):
    """Recover observed numeric pin fields, then sync proven existing nets."""
    if os.environ.get('PCB_PIN_FIELD_RECOVERY', '1').strip().lower() in {'0', 'off', 'false'}:
        return target, {'enabled': False, 'changes': [], 'network_changes': [], 'ocr_calls': 0}
    from pcb_parser.pin_field_recovery import recover_pin_fields, recognize
    pins = json.loads((stage/'export/engineering_pin_text_result.json').read_text(encoding='utf-8'))
    graph = json.loads((stage/'export/engineering_graph_proposals.json').read_text(encoding='utf-8'))
    records = json.loads((stage/'export/completed_component_records.json').read_text(encoding='utf-8'))['records']
    if graph.get('image_sha256') != image_sha256 or pins.get('image_sha256', image_sha256) != image_sha256:
        raise ValueError('Pin field recovery artifacts differ from the current image')
    if graph.get('coordinate_origin') != 'top_left' or graph.get('size') != [image_rgb.shape[1], image_rgb.shape[0]]:
        raise ValueError('Pin field recovery image dimensions or coordinates differ')
    config_path = ROOT/'configs/pin_field_recovery_v1.json'
    config = json.loads(config_path.read_text(encoding='utf-8'))
    engine = None
    fingerprints = None
    if reader is None:
        def reader(rgb, box, scale, rotation, recognition_only, view):
            nonlocal engine, fingerprints
            if engine is None:
                from pcb_parser.pin_text_ocr import create_engine
                engine, fingerprints = create_engine({'onnx_threads': 4})
            return recognize(engine, rgb, box, scale, rotation, recognition_only, view)
    def correct_positions(snapshot):
        if lead_px:
            pin_lead_snap(snapshot, float(lead_px), image_rgb, tip_mode)
    recovered, audit = recover_pin_fields(target, pins, graph, records, image_rgb,
        reader=reader, position_corrector=correct_positions, raw_pins=raw_pins, cfg=config)
    validate_target(recovered)
    audit['image_sha256'] = image_sha256
    audit['config_sha256'] = hashlib.sha256(config_path.read_bytes()).hexdigest()
    audit['ocr_models'] = fingerprints
    return recovered, audit


def finalize_pin_geometry_recovery(target, stage, image_rgb, image_sha256, reader=None):
    """Recover visible missing leads without changing wire geometry."""
    if os.environ.get('PCB_PIN_GEOMETRY_RECOVERY', '1').strip().lower() in {'0', 'off', 'false'}:
        return target, {'enabled': False, 'changes': [], 'network_changes': [], 'ocr_calls': 0}
    from pcb_parser.pin_geometry_recovery import recover_image_leads
    from pcb_parser.pin_field_recovery import recognize
    pins = json.loads((stage/'export/engineering_pin_text_result.json').read_text(encoding='utf-8'))
    graph = json.loads((stage/'export/engineering_graph_proposals.json').read_text(encoding='utf-8'))
    records = json.loads((stage/'export/completed_component_records.json').read_text(encoding='utf-8'))['records']
    if graph.get('image_sha256') != image_sha256 or pins.get('image_sha256', image_sha256) != image_sha256:
        raise ValueError('Pin geometry artifacts differ from the current image')
    if graph.get('coordinate_origin') != 'top_left' or graph.get('size') != [image_rgb.shape[1], image_rgb.shape[0]]:
        raise ValueError('Pin geometry image dimensions or coordinates differ')
    config_path = ROOT/'configs/pin_geometry_recovery_v1.json'
    config = json.loads(config_path.read_text(encoding='utf-8'))
    engine = None
    fingerprints = None
    if reader is None:
        def reader(rgb, box, scale, rotation, recognition_only, view):
            nonlocal engine, fingerprints
            if engine is None:
                from pcb_parser.pin_text_ocr import create_engine
                engine, fingerprints = create_engine({'onnx_threads': 4})
            return recognize(engine, rgb, box, scale, rotation, recognition_only, view)
    recovered, audit = recover_image_leads(target, pins, graph, records, image_rgb, reader=reader, cfg=config)
    validate_target(recovered)
    audit['image_sha256'] = image_sha256
    audit['config_sha256'] = hashlib.sha256(config_path.read_bytes()).hexdigest()
    audit['ocr_models'] = fingerprints
    return recovered, audit


def finalize_pin_relocation(target, stage, image_rgb, image_sha256, reader=None):
    """Recheck unsupported elimination-only pins against current-image evidence."""
    if os.environ.get('PCB_PIN_RELOCATION', '1').strip().lower() in {'0', 'off', 'false'}:
        return target, {'enabled': False, 'changes': [], 'network_changes': [], 'ocr_calls': 0}
    from pcb_parser.pin_relocation import relocate_unsupported_pins
    from pcb_parser.pin_field_recovery import recognize
    pins = json.loads((stage/'export/engineering_pin_text_result.json').read_text(encoding='utf-8'))
    graph = json.loads((stage/'export/engineering_graph_proposals.json').read_text(encoding='utf-8'))
    records = json.loads((stage/'export/completed_component_records.json').read_text(encoding='utf-8'))['records']
    if graph.get('image_sha256') != image_sha256 or pins.get('image_sha256') != image_sha256:
        raise ValueError('Pin relocation artifacts differ from the current image')
    for document in [pins, graph]:
        if document.get('coordinate_origin') != 'top_left' or document.get('size') != [image_rgb.shape[1], image_rgb.shape[0]]:
            raise ValueError('Pin relocation dimensions or coordinate system differ')
    config_path = ROOT/'configs/pin_relocation_v1.json'
    config = json.loads(config_path.read_text(encoding='utf-8'))
    engine = None
    fingerprints = None
    if reader is None:
        def reader(rgb, box, scale, rotation, recognition_only, view):
            nonlocal engine, fingerprints
            if engine is None:
                from pcb_parser.pin_text_ocr import create_engine
                engine, fingerprints = create_engine({'onnx_threads': 4})
            return recognize(engine, rgb, box, scale, rotation, recognition_only, view)
    recovered, audit = relocate_unsupported_pins(target, pins, graph, records, image_rgb, reader=reader, cfg=config)
    validate_target(recovered)
    audit.update(image_sha256=image_sha256, config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(), ocr_models=fingerprints)
    return recovered, audit


def finalize_pin_pairing(target, stage, image_rgb, image_sha256, reader=None):
    """Resolve shared-number conflicts as atomic pin/network transactions."""
    if os.environ.get('PCB_PIN_PAIRING', '1').strip().lower() in {'0', 'off', 'false'}:
        return target, {'enabled': False, 'changes': [], 'network_changes': [], 'restored_networks': [], 'ocr_calls': 0}
    from pcb_parser.pin_pairing import reconcile_pin_pairs
    from pcb_parser.pin_field_recovery import recognize
    pins = json.loads((stage/'export/engineering_pin_text_result.json').read_text(encoding='utf-8'))
    graph = json.loads((stage/'export/engineering_graph_proposals.json').read_text(encoding='utf-8'))
    records = json.loads((stage/'export/completed_component_records.json').read_text(encoding='utf-8'))['records']
    for document in [pins, graph]:
        if document.get('image_sha256') != image_sha256 or not document.get('prediction_only'):
            raise ValueError('Pin pairing artifacts differ from current-image predictions')
        if document.get('coordinate_origin') != 'top_left' or document.get('size') != [image_rgb.shape[1], image_rgb.shape[0]]:
            raise ValueError('Pin pairing dimensions or coordinate system differ')
    config_path = ROOT/'configs/pin_pairing_v1.json'
    config = json.loads(config_path.read_text(encoding='utf-8'))
    engine = None
    fingerprints = None
    if reader is None:
        def reader(rgb, box, scale, rotation, recognition_only, view):
            nonlocal engine, fingerprints
            if engine is None:
                from pcb_parser.pin_text_ocr import create_engine
                engine, fingerprints = create_engine({'onnx_threads': 4})
            return recognize(engine, rgb, box, scale, rotation, recognition_only, view)
    cleanup = cleanup_wire_edges if os.environ.get('PCB_WIRE_CLEANUP', '1').strip().lower() not in {'0', 'off', 'false'} else None
    recovered, audit = reconcile_pin_pairs(target, pins, graph, records, image_rgb, reader=reader, cleanup=cleanup, cfg=config)
    validate_target(recovered)
    audit.update(image_sha256=image_sha256, config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(), ocr_models=fingerprints)
    return recovered, audit


def finalize_pin_evidence_repair(target, stage, image_rgb, image_sha256, reader=None):
    """Restore corroborated printed labels and isolated collision points."""
    if os.environ.get('PCB_PIN_EVIDENCE_REPAIR', '1').strip().lower() in {'0', 'off', 'false'}:
        return target, {'enabled': False, 'changes': [], 'network_changes': [], 'ocr_calls': 0}
    from pcb_parser.pin_evidence_repair import repair_observed_pin_evidence
    from pcb_parser.pin_field_recovery import recognize
    pins = json.loads((stage/'export/engineering_pin_text_result.json').read_text(encoding='utf-8'))
    graph = json.loads((stage/'export/engineering_graph_proposals.json').read_text(encoding='utf-8'))
    records = json.loads((stage/'export/completed_component_records.json').read_text(encoding='utf-8'))['records']
    for document in [pins, graph]:
        if document.get('image_sha256') != image_sha256 or not document.get('prediction_only'):
            raise ValueError('Pin evidence artifacts differ from current-image predictions')
        if document.get('coordinate_origin') != 'top_left' or document.get('size') != [image_rgb.shape[1], image_rgb.shape[0]]:
            raise ValueError('Pin evidence dimensions or coordinate system differ')
    config_path = ROOT/'configs/pin_evidence_repair_v1.json'
    config = json.loads(config_path.read_text(encoding='utf-8'))
    engine = None
    fingerprints = None
    if reader is None:
        def reader(rgb, box, scale, rotation, recognition_only, view):
            nonlocal engine, fingerprints
            if engine is None:
                from pcb_parser.pin_text_ocr import create_engine
                engine, fingerprints = create_engine({'onnx_threads': 4})
            return recognize(engine, rgb, box, scale, rotation, recognition_only, view)
    recovered, audit = repair_observed_pin_evidence(target, pins, graph, records, image_rgb, reader=reader, cfg=config)
    validate_target(recovered)
    audit.update(image_sha256=image_sha256, config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(), ocr_models=fingerprints)
    return recovered, audit


def images_in(directory):
    images = sorted((p for p in directory.rglob('*') if p.is_file() and p.suffix.lower() == '.png'),
                    key=lambda p: p.relative_to(directory).as_posix())
    ids = [p.stem.casefold() for p in images]
    if len(ids) != len(set(ids)):
        raise ValueError('Input contains repeated image stems; case_id output files would collide')
    if not images:
        raise ValueError(f'No PNG images found under {directory}')
    return images


def configure_environment(output):
    os.environ['PYTHONPATH'] = os.pathsep.join([str(ROOT/'src'), str(ROOT/'scripts'), str(ROOT)])
    for path in [str(ROOT/'src'), str(ROOT/'scripts')]:
        if path not in sys.path:
            sys.path.insert(0, path)
    os.environ['PYTHONUTF8'] = '1'
    os.environ['PYTHONUNBUFFERED'] = '1'
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    os.environ['YOLO_CONFIG_DIR'] = str(output/'runtime/ultralytics')
    os.environ['YOLO_AUTOINSTALL'] = 'false'
    os.environ['MPLCONFIGDIR'] = str(output/'runtime/matplotlib')
    os.environ['MPLBACKEND'] = 'Agg'
    os.environ.setdefault('OMP_NUM_THREADS', '4')
    os.environ.setdefault('MKL_NUM_THREADS', '4')


def probe_runtime_device(torch, requested):
    """Prefer CUDA only after an actual kernel succeeds, and retain its failure."""
    probe = {'attempted': requested != 'cpu', 'success': False}
    if requested == 'cpu':
        return 'cpu', probe
    candidate = 'cuda:0' if requested == 'auto' else requested
    try:
        if not torch.cuda.is_available():
            # is_available() alone loses the useful driver/visibility error.
            torch.cuda.init()
            if not torch.cuda.is_available():
                raise RuntimeError('CUDA initialization exposed no usable GPU')
        tensor = torch.ones((4, 4), device=candidate)
        value = (tensor @ tensor).sum().item()
        torch.cuda.synchronize(candidate)
        if value != 64:
            raise RuntimeError('CUDA matrix probe returned an unexpected result')
        probe.update(success=True, device=candidate, name=torch.cuda.get_device_name(candidate),
                     capability=list(torch.cuda.get_device_capability(candidate)),
                     compiled_arches=torch.cuda.get_arch_list())
    except Exception as error:
        probe['error'] = f'{type(error).__name__}: {error}'
    return ('cuda:0' if probe['success'] else 'cpu') if requested == 'auto' else requested, probe


def preflight(device, diagnostics_path=None):
    import torch
    manifest = json.loads((ROOT/'package_manifest.json').read_text(encoding='utf-8'))
    for relative in manifest['runtime_assets']:
        path = ROOT/relative
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != manifest['files'][relative]['sha256']:
            raise ValueError(f'Bundled asset hash differs: {relative}')
    cuda_available = torch.cuda.is_available()
    diagnostics = {'requested_device':device, 'torch':torch.__version__, 'torch_cuda_build':torch.version.cuda,
                   'cuda_available':cuda_available, 'cuda_device_count':torch.cuda.device_count(),
                   'environment':{key:os.environ.get(key) for key in ['PCB_DEVICE','CUDA_VISIBLE_DEVICES','NVIDIA_VISIBLE_DEVICES','NVIDIA_DRIVER_CAPABILITIES']}}
    device, diagnostics['cuda_probe'] = probe_runtime_device(torch, device)
    diagnostics['selected_device'] = device
    if diagnostics_path:
        write(diagnostics_path, diagnostics)
    print('[Submission] runtime=' + json.dumps(diagnostics, ensure_ascii=True), flush=True)
    if device.startswith('cuda') and not diagnostics['cuda_probe']['success']:
        raise RuntimeError('CUDA requested but unusable: ' + diagnostics['cuda_probe'].get('error', 'unknown error'))
    if device == 'cpu' and diagnostics['cuda_probe']['attempted']:
        print('[Submission] CUDA unavailable; using CPU. Reason: ' + diagnostics['cuda_probe']['error'], flush=True)
    print(f'[Submission] device={device}; output_format={os.environ.get("PCB_OUTPUT_FORMAT", "target")}', flush=True)
    return device


def choose_workers(requested, device, case_count, gpu_memory_bytes=0):
    if requested < 0 or requested > 3:
        raise ValueError('workers must be 0 (auto), 1, 2 or 3')
    selected = requested or (3 if device.startswith('cuda') and gpu_memory_bytes >= 24*1024**3 else 1)
    return min(selected, case_count)


def _shape_passive(box, detected, pins, grow, orient=True):
    """Orient a passive by its pin tips, then grow the canonical box toward the detected cell.

    Two measured facts, both from the 13 released pages:
      * the detector cell can be taller than wide for a symbol drawn horizontally (case 12
        resistors), so the canonical table picks the wrong orientation - the two pin tips
        give the axis directly;
      * the reference box often spans the drawn leads (case 13: 23x58) while the canonical
        table writes the tight body (8x20), which misses "every coordinate within 20 px".
        Growing toward the detected cell, but by at most `grow` x the canonical size per
        side, recovers that without swallowing a neighbouring symbol (cap 1.0 already
        breaks case 10, 0.5-0.75 sits on the plateau).
    """
    x1, y1, x2, y2 = (float(v) for v in box)
    w, h = x2 - x1, y2 - y1
    entries = list(pins.values()) if isinstance(pins, dict) else list(pins or [])
    points = [(entry or {}).get("point") or {} for entry in entries]
    if orient and len(points) == 2:
        dx = abs(float(points[0].get("x", 0)) - float(points[1].get("x", 0)))
        dy = abs(float(points[0].get("y", 0)) - float(points[1].get("y", 0)))
        horizontal = dx >= dy
        if horizontal != (w >= h):
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            x1, x2 = cx - h / 2.0, cx + h / 2.0
            y1, y2 = cy - w / 2.0, cy + w / 2.0
            w, h = x2 - x1, y2 - y1
    if grow and detected and len(detected) == 4:
        dx1, dy1, dx2, dy2 = (float(v) for v in detected)
        x1 = min(x1, max(dx1, x1 - w * grow))
        y1 = min(y1, max(dy1, y1 - h * grow))
        x2 = max(x2, min(dx2, x2 + w * grow))
        y2 = max(y2, min(dy2, y2 + h * grow))
    return [x1, y1, x2, y2]


def _is_wire_pixel(rgb):
    r, g, b = int(rgb[0]), int(rgb[1]), int(rgb[2])
    return g > r + 20 and g > b + 20 and g > 60


def _is_ink_pixel(rgb):
    r, g, b = int(rgb[0]), int(rgb[1]), int(rgb[2])
    return (r + g + b) / 3.0 < 205 and not _is_wire_pixel(rgb)


def _hit(pixels, x, y, test, width, height):
    x, y = int(round(x)), int(round(y))
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            xx, yy = x + dx, y + dy
            if 0 <= xx < width and 0 <= yy < height and test(pixels[yy, xx]):
                return True
    return False


def measure_lead(pixels, width, height, box, side, point, limit=60):
    """Distance from the body edge to the lead/wire boundary, in image pixels.

    The reference rule (appendix 3) is that the pin tip is where the lead meets the green
    wire.  That boundary is directly visible: symbol ink is dark red, wires are green.  Walk
    outward from the body edge along the pin's normal and take the last ink pixel before the
    first green one.  Returns None when no boundary is found.
    """
    x1, y1, x2, y2 = box
    x, y = point
    if side == "left":
        start, step, fixed = x1, -1, y
        at = lambda d: (start + step * d, fixed)
    elif side == "right":
        start, step, fixed = x2, 1, y
        at = lambda d: (start + step * d, fixed)
    elif side == "top":
        start, step, fixed = y1, -1, x
        at = lambda d: (fixed, start + step * d)
    else:
        start, step, fixed = y2, 1, x
        at = lambda d: (fixed, start + step * d)
    ink_end = None
    gap = 0
    for distance in range(0, limit):
        sx, sy = at(distance)
        if _hit(pixels, sx, sy, _is_wire_pixel, width, height):
            # the wire has to be contiguous with the lead; a green blob further down the row
            # belongs to another net, not to this pin
            return max(distance - 1, 0)
        if _hit(pixels, sx, sy, _is_ink_pixel, width, height):
            ink_end = distance
            gap = 0
        else:
            gap += 1
            if ink_end is not None and gap > 2:
                # the lead ended and the row is empty: that end IS the tip
                return ink_end
    return ink_end


def cleanup_wire_edges(target, merge_degrees=4.0, min_length_px=8.0):
    """Tidy the exported segments before scoring.

    Two measured facts on the released pages (20 px rule, 13 pages, +0.51 points):

      * our skeleton adds a vertex in the middle of a straight run, so one reference segment
        becomes two of ours and neither matches.  Merging only at DEGREE-2 joints fixes that
        without touching real junctions (GT splits there on purpose).  A blanket merge made
        things worse: 0.448 -> 0.29.
      * segments shorter than ~8 px are skeleton noise that can never be a reference segment;
        dropping them cut 205 false positives and cost 17 true ones.
    """
    import math
    from collections import Counter

    def angle(a, b):
        ux, uy = a[1][0] - a[0][0], a[1][1] - a[0][1]
        vx, vy = b[1][0] - b[0][0], b[1][1] - b[0][1]
        nu, nv = math.hypot(ux, uy), math.hypot(vx, vy)
        if nu < 1e-6 or nv < 1e-6:
            return 0.0
        cos = max(-1.0, min(1.0, (ux * vx + uy * vy) / (nu * nv)))
        return math.degrees(math.acos(cos))

    merged_total = dropped_total = 0
    for net in (target.get('nets') or {}).values():
        segs = []
        for points in (net.get('edges') or {}).values():
            if len(points) == 2:
                segs.append([[float(points[0]['x']), float(points[0]['y'])],
                             [float(points[1]['x']), float(points[1]['y'])]])
        if not segs:
            continue
        # 1. merge at degree-2, near-collinear joints
        merging = True
        while merging:
            merging = False
            degree = Counter()
            for a, b in segs:
                degree[tuple(a)] += 1
                degree[tuple(b)] += 1
            for i in range(len(segs)):
                for j in range(len(segs)):
                    if i == j:
                        continue
                    a, b = segs[i], segs[j]
                    shared = joined = None
                    if math.dist(a[1], b[0]) < 1e-6:
                        shared, joined = tuple(a[1]), [a[0], b[1]]
                    elif math.dist(a[1], b[1]) < 1e-6:
                        shared, joined = tuple(a[1]), [a[0], b[0]]
                    elif math.dist(a[0], b[0]) < 1e-6:
                        shared, joined = tuple(a[0]), [b[1], a[1]]
                    elif math.dist(a[0], b[1]) < 1e-6:
                        shared, joined = tuple(a[0]), [b[0], a[1]]
                    if shared is None or degree[shared] != 2:
                        continue
                    if angle(a, joined) > merge_degrees:
                        continue
                    segs[i] = joined
                    segs.pop(j)
                    merged_total += 1
                    merging = True
                    break
                if merging:
                    break
        # 2. drop the stubs
        kept = [s for s in segs if math.dist(tuple(s[0]), tuple(s[1])) >= min_length_px]
        dropped_total += len(segs) - len(kept)
        if not kept:
            net['edges'] = {}
            continue
        net['edges'] = {f'edge_{index+1}': [{'x': s[0][0], 'y': s[0][1]}, {'x': s[1][0], 'y': s[1][1]}]
                        for index, s in enumerate(kept)}
    cleanup_wire_edges.merged = merged_total
    cleanup_wire_edges.dropped = dropped_total
    return merged_total, dropped_total


def pin_lead_snap(target, lead_px, image_rgb=None, mode="color", side_audit=None):
    """Push each IC pin point out to its real tip.

    Two decisions, both measured on the 13 released pages:

    * the tip is where the pin lead meets the green wire (appendix 3), so it is measured from
      the image instead of assumed - the reference lead is 20, 29 or 40 px depending on the
      symbol, and a fixed 18 px is wrong on two thirds of those pages;
    * one symbol has one lead length, so a component's own measurements are reduced to their
      median.  Individual pins land on a neighbouring wire or on the body outline, and a
      single bad reading must not move only that one pin.

    A measured length outside ``[min_px, max_px]`` is treated as a failed reading and falls
    back to ``lead_px``, which is the old fixed behaviour.
    """
    try:
        min_px = float(os.environ.get('PCB_PIN_TIP_MIN', '12'))
    except ValueError:
        min_px = 12.0
    try:
        max_px = float(os.environ.get('PCB_PIN_TIP_MAX', '60'))
    except ValueError:
        max_px = 60.0
    unify = os.environ.get('PCB_PIN_TIP_UNIFY', '1').strip().lower() not in {'0', 'off', 'false'}

    # Keep a pre-snap snapshot so corrected coordinates cannot vote for themselves.
    side_enabled = (image_rgb is not None and mode == "color" and
                    os.environ.get('PCB_PIN_SIDE_CONSENSUS', '1').strip().lower()
                    not in {'0', 'off', 'false'})
    original_pins = None
    if side_enabled:
        from copy import deepcopy
        original_pins = deepcopy(target.get('pins') or {})
    if side_audit is not None:
        side_audit.clear()
        side_audit.update(enabled=False, modified_pin_count=0, groups=[])
    moved = measured = unified = 0
    components = target.get('components') or {}
    width = height = None
    pixels = None
    if image_rgb is not None and mode == "color":
        height, width = image_rgb.shape[0], image_rgb.shape[1]
        pixels = image_rgb
    names = {'x1': 'left', 'x2': 'right', 'y1': 'top', 'y2': 'bottom'}
    for component_id, pins in (target.get('pins') or {}).items():
        component = components.get(component_id) or {}
        if str(component.get('type') or '') not in {'ic', 'box', 'connector', 'raw_block', 'block'}:
            continue
        box = component.get('bbox') or []
        if len(box) != 4:
            continue
        x1, y1, x2, y2 = (float(value) for value in box)
        # the submitted box is bottom-left origin; the image scan works top-left
        top, bottom = (height - y2, height - y1) if height else (y1, y2)
        edges = {}
        readings = []
        for pin_key, pin in pins.items():
            point = pin.get('point') or {}
            px, py = float(point.get('x', 0.0)), float(point.get('y', 0.0))
            sides = {'x1': x1 - px if px < x1 else None, 'x2': px - x2 if px > x2 else None,
                     'y1': y1 - py if py < y1 else None, 'y2': py - y2 if py > y2 else None}
            sides = {k: v for k, v in sides.items() if v is not None}
            if not sides:
                continue
            edge = max(sides, key=sides.get)
            if sides[edge] > 60:
                continue
            edges[pin_key] = edge
            if pixels is None:
                continue
            try:
                lead = measure_lead(pixels, width, height, (x1, top, x2, bottom), names[edge],
                                    (px, height - py))
            except Exception:
                lead = None
            readings.append(lead)
        valid = sorted(value for value in readings if value is not None and min_px <= value <= max_px)
        component_lead = None
        if valid and unify and len(valid) >= 2:
            middle = len(valid) // 2
            component_lead = valid[middle] if len(valid) % 2 else (valid[middle - 1] + valid[middle]) / 2.0
            unified += 1
        for pin_key, edge in edges.items():
            pin = pins[pin_key]
            point = pin.get('point') or {}
            px, py = float(point.get('x', 0.0)), float(point.get('y', 0.0))
            lead = component_lead
            if lead is None and pixels is not None:
                try:
                    lead = measure_lead(pixels, width, height, (x1, top, x2, bottom), names[edge],
                                        (px, height - py))
                except Exception:
                    lead = None
                if lead is not None and min_px <= lead <= max_px:
                    measured += 1
                else:
                    lead = None
            if lead is None:
                lead = lead_px
            if edge == 'x1':
                point['x'] = x1 - lead
            elif edge == 'x2':
                point['x'] = x2 + lead
            elif edge == 'y1':
                point['y'] = y1 - lead
            else:
                point['y'] = y2 + lead
            moved += 1
    if side_enabled:
        from pcb_parser.pin_position_consensus import correct_pin_positions_by_side
        correction = correct_pin_positions_by_side(target, original_pins, image_rgb, measure_lead)
        if side_audit is not None:
            side_audit.update(correction)
    pin_lead_snap.measured = measured
    pin_lead_snap.unified = unified
    return moved


def run_case(image, output, device, output_format, keep_work):
    case_id = image.stem
    row = {'case_id': case_id, 'input': str(image), 'success': False}
    start = time.perf_counter()
    log = output/'logs'/f'{case_id}.log'
    log.parent.mkdir(parents=True, exist_ok=True)
    work_parent = output/'work' if keep_work else None
    if work_parent:
        work_parent.mkdir(parents=True, exist_ok=True)
        work = Path(tempfile.mkdtemp(prefix='pcb_', dir=work_parent))
    else:
        work = Path(tempfile.mkdtemp(prefix='pcb_'))
    try:
        stage = work/'p'
        command = [sys.executable, str(ROOT/'scripts/run_observed_net_fields.py'),
                   '--image', str(image), '--output', str(stage), '--device', device]
        print(f'[Submission] Start {case_id}; log={log}', flush=True)
        # Stream child output to a durable file; no unbounded in-memory log buffer.
        child_environment = os.environ.copy()
        # Parallel cases must not race on Ultralytics/Matplotlib settings files.
        child_environment['YOLO_CONFIG_DIR'] = str(output/'runtime/ultralytics'/case_id)
        child_environment['MPLCONFIGDIR'] = str(output/'runtime/matplotlib'/case_id)
        with log.open('w', encoding='utf-8') as stream:
            process = subprocess.run(command, cwd=ROOT, env=child_environment,
                                     stdout=stream, stderr=subprocess.STDOUT)
        if process.returncode:
            raise RuntimeError(f'Image pipeline exited with code {process.returncode}; inspect {log.name}')
        paper = json.loads((stage/'export/engineering_submission.json').read_text(encoding='utf-8'))
        from component_export import export_power_components, missing_field_diagnostics
        records_document = json.loads((stage/'export/completed_component_records.json').read_text(encoding='utf-8'))
        summary = json.loads((stage/'summary.json').read_text(encoding='utf-8'))
        ocr_paths = [Path(p) for p in summary['original_S23_input_sha256'] if p.endswith('texts.json')]
        if len(ocr_paths) != 1:
            raise ValueError('Expected one current-image OCR provenance file')
        ocr_document = json.loads(ocr_paths[0].read_text(encoding='utf-8'))
        component_payload, component_audit = export_power_components(paper, records_document, ocr_document)
        target = to_target(component_payload)
        from pcb_parser.ic_name_selection import select_ic_names
        target, name_audit = select_ic_names(target, records_document, ocr_document)
        validate_target(target)
        write(output/'logs'/f'{case_id}_ic_name_selection.json', name_audit)
        write(output/'logs'/f'{case_id}_ocr_texts.json', ocr_document)
        rgb = None
        if output_format == 'target' and any(c['type'] in {'ic','box'} for c in target['components'].values()):
            ic_start = time.perf_counter()
            from PIL import Image
            import numpy as np
            from pcb_parser.ic_geometry import refine_ic_boxes
            rgb = np.array(Image.open(image).convert('RGB'))
            target, ic_audit = refine_ic_boxes(rgb, target)
            validate_target(target)
            ic_audit['image_sha256'] = summary['image_sha256']
            ic_audit['wall_seconds'] = round(time.perf_counter()-ic_start, 3)
            row['ic_geometry_seconds'] = ic_audit['wall_seconds']
            write(output/'logs'/f'{case_id}_ic_geometry.json', ic_audit)
        if output_format == 'target' and any(c['type']=='led' for c in component_payload['components'].values()):
            led_start = time.perf_counter()
            from PIL import Image
            import numpy as np
            from pcb_parser.led_geometry import refine_led_boxes, yolo_predictor
            if rgb is None:
                rgb = np.array(Image.open(image).convert('RGB'))
            target, led_audit = refine_led_boxes(rgb, component_payload, target,
                yolo_predictor(ROOT/'weights/detector_dataset300_v3_nano.pt', device))
            validate_target(target)
            led_audit['image_sha256'] = summary['image_sha256']
            led_audit['wall_seconds'] = round(time.perf_counter()-led_start, 3)
            row['led_geometry_seconds'] = led_audit['wall_seconds']
            write(output/'logs'/f'{case_id}_led_geometry.json', led_audit)
        # --- canonical export geometry (v4 overlay) --------------------------------
        # The organiser's GT boxes the drawn symbol at its natural size; the detector
        # boxes the CAD cell, which is 1.05x-2x larger. Rescaling to the canonical
        # size and shifting the centre by the calibrated offset raised bbox accuracy
        # on the public subset from .905 to 1.000 (official 20 px rubric).
        # 'ic' is excluded (this pipeline refines IC boxes itself, and IC size is not
        # fixed); 'led' is excluded for the same reason (refine_led_boxes).
        if target.get('components') and os.environ.get('PCB_BBOX_CONVENTION', '1') != '0':
            try:
                from pcb_parser import export_geometry as _geom
                from PIL import Image as _Image
                _mode, _table = _geom.load_table(
                    {'bbox_convention': {'enabled': True, 'mode': 'canonical',
                                         'path': 'configs/bbox_convention_v1.json'}}, root=ROOT)
                # PCB_BBOX_SCALE_PASSIVE applies a fraction of the detected box instead of the
                # fixed canonical size.  OFF by default: the offline scan said 0.7 would add
                # +0.02 component F1, but end to end it cost 0.011 (0.706 -> 0.695), because the
                # box the pipeline holds at this point is the fused detection, not the raw one.
                try:
                    _scale = float(os.environ.get('PCB_BBOX_SCALE_PASSIVE', '0'))
                except ValueError:
                    _scale = 0
                # Growth toward the detected cell (cap 0.75 sits on the measured plateau;
                # 0 disables) and tip-based orientation.  Both default on.
                try:
                    _shape = float(os.environ.get('PCB_BBOX_GROW_TO_DETECTED', '0.75'))
                except ValueError:
                    _shape = 0.75
                _orient = os.environ.get('PCB_BBOX_ORIENT_FROM_TIPS', '1').strip().lower() not in {'0', 'off', 'false'}
                # Only the three two-lead passives: the scan that produced 0.7 covered r/c/l,
                # and folding diodes in on a hunch made the end-to-end score worse.
                _scalable = {'r', 'c', 'l', 'resistor', 'capacitor', 'inductor'}
                if _mode:
                    _height = _Image.open(image).size[1]
                    _applied = 0
                    for _key, _comp in target['components'].items():
                        if _comp.get('type') in {'ic', 'box', 'led', 'net_bidirection', 'net_input',
                                                 'net_output', 'net_short'}:
                            # A net label's annotation box is the drawn flag, whose width
                            # follows its text (`VOUT` 82 px vs `LED-` 41 px); the fixed
                            # canonical label box fits short flags by luck and misses a
                            # long one by 23 px, while the detected box is within 4-8 px
                            # on every released label.  Passives only get the convention.
                            continue
                        _bbox = _comp.get('bbox')
                        if not _bbox or len(_bbox) != 4:
                            continue
                        _tl = [_bbox[0], _height - _bbox[3], _bbox[2], _height - _bbox[1]]
                        if _scale and _comp.get('type') in _scalable:
                            # The reference draws the same passive at 23x58 on one page and
                            # 40x16 on another, so a fixed canonical size can only fit one
                            # convention.  The detector box already tracks the drawing, so a
                            # fraction of it adapts per page; 0.7 sits on the plateau that
                            # helps both the 13 released pages and the 10-page Golden set.
                            _cx = (_tl[0] + _tl[2]) / 2.0
                            _cy = (_tl[1] + _tl[3]) / 2.0
                            _w = (_tl[2] - _tl[0]) * _scale
                            _h = (_tl[3] - _tl[1]) * _scale
                            _new = [_cx - _w / 2.0, _cy - _h / 2.0, _cx + _w / 2.0, _cy + _h / 2.0]
                            _proof = True
                        else:
                            _new, _proof = _geom.apply_to_box(_tl, _comp.get('type'), _mode, _table)
                        if not _proof:
                            continue
                        if (_shape or _orient) and str(_comp.get('type')) in {'r', 'c', 'l', 'resistor', 'capacitor', 'inductor'}:
                            _new = _shape_passive(_new, _tl, (target.get('pins') or {}).get(_key), _shape, _orient)
                        _comp['bbox'] = [_new[0], _height - _new[3], _new[2], _height - _new[1]]
                        _applied += 1
                    row['bbox_convention_applied'] = _applied
            except Exception as _error:
                row['bbox_convention_error'] = f'{type(_error).__name__}: {_error}'
        write(output/'project_results'/f'{case_id}_Pred.json', paper)
        write(output/'component_results'/f'{case_id}_Pred.json', component_payload)
        write(output/'logs'/f'{case_id}_component_export_audit.json', component_audit)
        write(output/'logs'/f'{case_id}_component_fields_debug.json', missing_field_diagnostics(records_document, ocr_document))
        prediction = to_sample(paper) if output_format == 'sample' else (paper if output_format == 'project' else target)
        # Pin tips of ICs sit on the END of the symbol lead; the model's centroid stops
        # ~7 px short of it, which costs every IC pin under the 5 px rule.  On by default
        # (18 px is the flat part of the measured 16-22 px plateau); PCB_PIN_LEAD_PX=0 or
        # "off" restores the old behaviour.  Only pins outside the body are moved.
        _field_raw_pins = None
        if output_format == 'target':
            from copy import deepcopy as _field_deepcopy
            _field_raw_pins = _field_deepcopy(target.get('pins') or {})
        _lead = os.environ.get('PCB_PIN_LEAD_PX', '18')
        if _lead.strip().lower() in {'off', 'false', ''}:
            _lead = None
        elif _lead:
            _lead = float(_lead)
        if _lead and output_format == 'target':
            try:
                _tip_mode = os.environ.get('PCB_PIN_TIP_MODE', 'color').strip().lower()
                _rgb = None
                if _tip_mode == 'color':
                    from PIL import Image as _TipImage
                    import numpy as _tip_np
                    with _TipImage.open(image) as _tip_handle:
                        _rgb = _tip_np.asarray(_tip_handle.convert('RGB'))
                _side_audit = {}
                _moved = pin_lead_snap(target, float(_lead), _rgb, _tip_mode, side_audit=_side_audit)
                row['pin_side_consensus'] = {key: _side_audit[key] for key in
                                              ['enabled', 'modified_pin_count']}
                write(output/'logs'/f'{case_id}_pin_side_consensus.json', _side_audit)
                row['pin_tip_mode'] = _tip_mode
                row['pin_tip_measured'] = getattr(pin_lead_snap, 'measured', 0)
                row['pin_tip_unified'] = getattr(pin_lead_snap, 'unified', 0)
                row['pin_lead_snap'] = _moved
            except Exception as _error:
                row['pin_lead_snap_error'] = f'{type(_error).__name__}: {_error}'
        if os.environ.get('PCB_WIRE_CLEANUP', '1').strip().lower() not in {'0', 'off', 'false'}:
            try:
                _merged, _dropped = cleanup_wire_edges(target)
                row['wire_cleanup'] = {'merged': _merged, 'dropped': _dropped}
            except Exception as _error:
                row['wire_cleanup_error'] = f'{type(_error).__name__}: {_error}'
        if output_format == 'target':
            try:
                target, power_wire_audit = finalize_power_wire_export(target, stage, summary['image_sha256'])
                prediction = target
                row['power_wire_export'] = {key: power_wire_audit[key] for key in
                    ['enabled', 'restored_net_count', 'primary_contact_count'] if key in power_wire_audit}
                write(output/'logs'/f'{case_id}_power_wire_export.json', power_wire_audit)
            except Exception as _error:
                row['power_wire_export_error'] = f'{type(_error).__name__}: {_error}'
        if output_format == 'target':
            try:
                if rgb is None:
                    from PIL import Image as _FieldImage
                    import numpy as _field_np
                    with _FieldImage.open(image) as _field_image:
                        rgb = _field_np.asarray(_field_image.convert('RGB'))
                target, _field_audit = finalize_pin_field_recovery(
                    target, stage, rgb, summary['image_sha256'], _field_raw_pins,
                    _lead, os.environ.get('PCB_PIN_TIP_MODE', 'color').strip().lower())
                prediction = target
                row['pin_field_recovery'] = {'enabled': _field_audit['enabled'],
                    'changed_pins': len(_field_audit['changes']),
                    'changed_networks': sum('after' in item for item in _field_audit['network_changes']),
                    'ocr_calls': _field_audit['ocr_calls']}
                write(output/'logs'/f'{case_id}_pin_field_recovery.json', _field_audit)
            except Exception as _error:
                row['pin_field_recovery_error'] = f'{type(_error).__name__}: {_error}'
        if output_format == 'target':
            try:
                if rgb is None:
                    from PIL import Image as _GeometryImage
                    import numpy as _geometry_np
                    with _GeometryImage.open(image) as _geometry_image:
                        rgb = _geometry_np.asarray(_geometry_image.convert('RGB'))
                target, _geometry_audit = finalize_pin_geometry_recovery(
                    target, stage, rgb, summary['image_sha256'])
                prediction = target
                row['pin_geometry_recovery'] = {'enabled': _geometry_audit['enabled'],
                    'changed_pins': len(_geometry_audit['changes']),
                    'changed_networks': sum('after' in item for item in _geometry_audit['network_changes']),
                    'ocr_calls': _geometry_audit['ocr_calls']}
                write(output/'logs'/f'{case_id}_pin_geometry_recovery.json', _geometry_audit)
            except Exception as _error:
                row['pin_geometry_recovery_error'] = f'{type(_error).__name__}: {_error}'
        if output_format == 'target':
            try:
                if rgb is None:
                    from PIL import Image as _RelocationImage
                    import numpy as _relocation_np
                    with _RelocationImage.open(image) as _relocation_image:
                        rgb = _relocation_np.asarray(_relocation_image.convert('RGB'))
                target, _relocation_audit = finalize_pin_relocation(target, stage, rgb, summary['image_sha256'])
                prediction = target
                row['pin_relocation'] = {'enabled': _relocation_audit['enabled'],
                    'changed_pins': len(_relocation_audit['changes']),
                    'changed_networks': sum('after' in item for item in _relocation_audit['network_changes']),
                    'ocr_calls': _relocation_audit['ocr_calls']}
                write(output/'logs'/f'{case_id}_pin_relocation.json', _relocation_audit)
            except Exception as _error:
                row['pin_relocation_error'] = f'{type(_error).__name__}: {_error}'
        if output_format == 'target':
            try:
                if rgb is None:
                    from PIL import Image as _PairingImage
                    import numpy as _pairing_np
                    with _PairingImage.open(image) as _pairing_image:
                        rgb = _pairing_np.asarray(_pairing_image.convert('RGB'))
                target, _pairing_audit = finalize_pin_pairing(target, stage, rgb, summary['image_sha256'])
                prediction = target
                row['pin_pairing'] = {'enabled': _pairing_audit['enabled'],
                    'changed_pins': len(_pairing_audit['changes']),
                    'changed_networks': sum('after' in item for item in _pairing_audit['network_changes']),
                    'restored_networks': len(_pairing_audit['restored_networks']),
                    'ocr_calls': _pairing_audit['ocr_calls']}
                write(output/'logs'/f'{case_id}_pin_pairing.json', _pairing_audit)
            except Exception as _error:
                row['pin_pairing_error'] = f'{type(_error).__name__}: {_error}'
        if output_format == 'target':
            try:
                if rgb is None:
                    from PIL import Image as _EvidenceImage
                    import numpy as _evidence_np
                    with _EvidenceImage.open(image) as _evidence_image:
                        rgb = _evidence_np.asarray(_evidence_image.convert('RGB'))
                target, _evidence_audit = finalize_pin_evidence_repair(target, stage, rgb, summary['image_sha256'])
                prediction = target
                row['pin_evidence_repair'] = {'enabled': _evidence_audit['enabled'],
                    'changed_pins': len(_evidence_audit['changes']),
                    'changed_networks': sum('after' in item for item in _evidence_audit['network_changes']),
                    'ocr_calls': _evidence_audit['ocr_calls']}
                write(output/'logs'/f'{case_id}_pin_evidence_repair.json', _evidence_audit)
            except Exception as _error:
                row['pin_evidence_repair_error'] = f'{type(_error).__name__}: {_error}'
        write(output/'final_results'/f'{case_id}_Pred.json', prediction)
        for name in ['engineering_export_coverage.json', 'field_provenance.json']:
            shutil.copy2(stage/'export'/name, output/'logs'/f'{case_id}_{name}')
        shutil.copy2(stage/'summary.json', output/'logs'/f'{case_id}_summary.json')
        row.update(success=True, components=len(target['components']),
                   pins=sum(len(pins) for pins in target['pins'].values()), nets=len(target['nets']))
    except Exception as error:
        row['error'] = f'{type(error).__name__}: {error}'
        with log.open('a', encoding='utf-8') as stream:
            traceback.print_exc(file=stream)
        write(output/'final_results'/f'{case_id}_Pred.json', EMPTY)
        print(f'[Submission] FAILED {case_id}: {row["error"]}', flush=True)
        if log.exists():
            print('\n'.join(log.read_text(encoding='utf-8', errors='replace').splitlines()[-24:]), flush=True)
    finally:
        row['seconds'] = round(time.perf_counter()-start, 3)
        if not keep_work:
            shutil.rmtree(work)
    print(f'[Submission] End {case_id}: {json.dumps(row, ensure_ascii=True)}', flush=True)
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-input', '--input', type=Path, required=True)
    parser.add_argument('-output', '--output', type=Path, required=True)
    parser.add_argument('--device', default=os.environ.get('PCB_DEVICE', 'auto'))
    parser.add_argument('--output-format', choices=['target', 'sample', 'project'], default=os.environ.get('PCB_OUTPUT_FORMAT', 'target'))
    parser.add_argument('--workers', type=int, choices=[0, 1, 2, 3], default=int(os.environ.get('PCB_WORKERS', '0')))
    parser.add_argument('--keep-work', action='store_true')
    args = parser.parse_args()
    args.input, args.output = args.input.resolve(), args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.input == args.output or args.output.is_relative_to(args.input) or args.input.is_relative_to(args.output):
        parser.error('Input and output directories must not overlap')
    configure_environment(args.output)
    start = time.perf_counter()
    decision = {'method': 'yolov8_v3_rapidocr_unet_v1_S24_engineering',
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'processed_cases': 0, 'successful_cases': 0, 'failed_cases': 0,
                'output_format': args.output_format, 'coordinate_origin': 'bottom_left_H_minus_y',
                'token_usage': {'total_tokens': 0, 'prompt_tokens': 0, 'completion_tokens': 0, 'api_calls': 0},
                'cases': []}
    fatal = False
    try:
        if not args.input.is_dir():
            raise FileNotFoundError(args.input)
        images = images_in(args.input)
        device = preflight(args.device, args.output/'runtime_environment.json')
        gpu_memory = 0
        if device.startswith('cuda'):
            import torch
            gpu_memory = torch.cuda.get_device_properties(device).total_memory
        workers = choose_workers(args.workers, device, len(images), gpu_memory)
        decision.update(device=device, input_cases=len(images), workers=workers,
                        gpu_memory_bytes=gpu_memory, protocol='2026-10-08-global-power-rails',
                        type_mapping_status='six aliases verified in submission 8180 and public labels',
                        component_output_policy=COMPONENT_OUTPUT_POLICY)
        print(f'[Submission] workers={workers}; gpu_memory_bytes={gpu_memory}', flush=True)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(run_case, image, args.output, device, args.output_format, args.keep_work)
                       for image in images]
            completed = (future.result() for future in as_completed(futures))
            for row in completed:
                decision['cases'].append(row)
                decision['processed_cases'] += 1
                decision['successful_cases' if row['success'] else 'failed_cases'] += 1
                decision['wall_seconds'] = round(time.perf_counter()-start, 3)
                write(args.output/'final_results/decision_log.json', decision)
                write(args.output/'decision_log.json', decision)
    except Exception as error:
        fatal = True
        decision['fatal_error'] = f'{type(error).__name__}: {error}'
        traceback.print_exc()
    finally:
        decision['wall_seconds'] = round(time.perf_counter()-start, 3)
        write(args.output/'final_results/decision_log.json', decision)
        write(args.output/'decision_log.json', decision)
    return 1 if fatal or not decision['successful_cases'] else 0


if __name__ == '__main__':
    sys.exit(main())
