"""Localized RGB OCR for unresolved, uniquely attached power symbols.

Adds actual detected glyph boxes and retains conflicting spellings. No
recognition-only crop is represented as a word box; topology and pin fields
are untouched. Repeated views of one model are not independent evidence.
"""
from copy import deepcopy
import math
import time

import cv2
import numpy as np

from .component_text import bbox, center, iou, restore_polygon
from .network_labels import symbol_matches, within
from .power_networks import qualified_name


def recovery_rois(pins, graph, records, baseline_audit, cfg):
    width, height = pins['size']
    by_index = {r['detection_index']: r for r in records}
    attached = {p['candidate_id']: p for p in graph['pin_attachments']}
    rows = []
    for row in baseline_audit['component_review_queue']:
        if row['bound_site_id'] is not None or len(row['owned_tip_ids']) != 1:
            continue
        pin = attached[row['owned_tip_ids'][0]]
        if pin['attachment_status'] != 'attached' or pin['network_id'] is None:
            continue
        record = by_index[row['detection_index']]
        box = record['bbox_top_left']
        crop = [max(0, math.floor(box[0])-cfg['padding_x_px']),
                max(0, math.floor(box[1])-cfg['padding_y_px']),
                min(width, math.ceil(box[2])+cfg['padding_x_px']),
                min(height, math.ceil(box[3])+cfg['padding_y_px'])]
        rows.append({'detection_index': row['detection_index'], 'component_type': record['type'],
                     'candidate_id': pin['candidate_id'], 'crop': crop})
    return rows[:cfg['max_rois']], len(rows)-min(len(rows), cfg['max_rois'])


def recover_power_captions(rgb, pins, graph, records, texts, baseline_audit,
                           engine, cfg, power_cfg, label_cfg):
    if list(rgb.shape[:2][::-1]) != pins['size']:
        raise ValueError('RGB dimensions differ from power input')
    scales, rotations = cfg['scales'], cfg['rotations']
    if (not scales or len(set(scales)) != len(scales)
            or any(not math.isfinite(s) or s <= 0 for s in scales)
            or not rotations or len(set(rotations)) != len(rotations)
            or any(r not in (0, 90) for r in rotations)
            or any(cfg[k] < 0 for k in ['padding_x_px', 'padding_y_px', 'boundary_margin_px', 'max_rois'])
            or not isinstance(cfg['max_rois'], int) or not 0 <= cfg['min_confidence'] <= 1):
        raise ValueError('Invalid power OCR configuration')
    rois, skipped = recovery_rois(pins, graph, records, baseline_audit, cfg)
    blocked = [r['fields'][f]['bbox'] for r in records for f in ['designator', 'Name', 'value']
               if f in r.get('fields', {})]
    blocked += [p['text_evidence'][f]['bbox'] for p in pins['pin_candidates']
                for f in ['number', 'pinname'] if 'bbox' in p.get('text_evidence', {}).get(f, {})]
    recovered, rejected, timings = [], [], []
    for roi in rois:
        x1, y1, x2, y2 = roi['crop']
        patch = rgb[y1:y2, x1:x2]
        for rotation in rotations:
            rotated = cv2.rotate(patch, cv2.ROTATE_90_CLOCKWISE) if rotation == 90 else patch
            for scale in scales:
                scaled = cv2.resize(rotated, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                start = time.perf_counter()
                result = engine(np.ascontiguousarray(scaled[:, :, ::-1]), use_det=True, use_cls=True, use_rec=True)
                view = f"power_roi_{roi['detection_index']}_r{rotation}_s{scale}"
                count = 0
                if result.boxes is not None:
                    for polygon, text, score in zip(result.boxes, result.txts, result.scores):
                        score = float(score)
                        if not math.isfinite(score) or not 0 <= score <= 1:
                            raise ValueError('Invalid localized OCR confidence')
                        if score < cfg['min_confidence']:
                            continue
                        local = np.asarray(polygon, dtype=float)
                        if local.shape != (4, 2) or not np.isfinite(local).all():
                            raise ValueError('Invalid localized OCR polygon')
                        points = restore_polygon(local, roi['crop'], scale, rotation)
                        box = bbox(points)
                        margin = cfg['boundary_margin_px']
                        row = {'text': str(text), 'confidence': score, 'bbox': box, 'polygon': points,
                               'view': view, 'target_detection_index': roi['detection_index']}
                        if (box[0] <= x1+margin or box[1] <= y1+margin
                                or box[2] >= x2-margin or box[3] >= y2-margin):
                            rejected.append({**row, 'reason': 'possible_truncated_glyph_at_crop_boundary'})
                            continue
                        # Do not filter out conflicting captions here. Both recognized
                        # GND and a supported alternative must reach the spelling gate.
                        row['text_id'] = f'POWER_ROI_TEXT_{len(recovered)+1:05d}'
                        recovered.append(row)
                        count += 1
                timings.append({'view': view, 'crop': roi['crop'], 'localized_observations': count,
                                'wall_seconds': time.perf_counter()-start})
    sites = []
    target_indexes = {r['detection_index'] for r in rois}
    for row in recovered:
        box = row['bbox']
        if any(iou(box, b) >= label_cfg['text_role_overlap_iou'] for b in blocked):
            continue
        if any(c['type'] not in label_cfg['label_component_types']+label_cfg['power_component_types']
               and within(center(box), c['bbox_top_left']) for c in pins['components']):
            continue
        options = sorted(symbol_matches(box, pins, graph, label_cfg), key=lambda o: o['cost'])
        if not options:
            continue
        best = options[0]
        # Distinct symbols on the same physical net still constitute ambiguous
        # caption ownership; do not collapse alternatives by network ID here.
        if len(options) > 1 and options[1]['cost']-best['cost'] < label_cfg['network_attachment_margin_px']:
            continue
        if (best['detection_index'] != row['target_detection_index']
                or best['detection_index'] not in target_indexes
                or not qualified_name(row['text'], best['component_type'], power_cfg)):
            continue
        if any(s['detection_index'] == best['detection_index'] and s['label'] == row['text']
               and iou(s['bbox'], box) >= power_cfg['same_glyph_iou'] for s in sites):
            continue
        sites.append({'site_id': f'POWER_ROI_SITE_{len(sites)+1:05d}', **best,
                      'label': row['text'], 'bbox': box, 'confidence': row['confidence'],
                      'status': 'power_semantics_pending', 'merge_eligible': False,
                      'coordinate_origin': 'top_left', 'image_sha256': pins['image_sha256'],
                      'source': 'localized_RGB_detection_and_recognition'})
    updated = deepcopy(graph)
    updated['label_sites'].extend(sites)
    return updated, deepcopy(texts)+recovered, {
        'roi_count': len(rois), 'skipped_by_budget': skipped, 'rois': rois, 'views': timings,
        'localized_observations': len(recovered), 'new_power_caption_sites': len(sites),
        'rejected_boundary_observations': rejected,
        'wall_seconds': sum(t['wall_seconds'] for t in timings),
        'same_model_views_not_independent': True, 'spelling_replacements': 0,
        'pin_fields_changed': 0, 'source_annotations_used': False}
