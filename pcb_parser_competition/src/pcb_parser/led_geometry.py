"""Image-only LED lead-axis evidence, independent of text and reference labels."""
from copy import deepcopy
import math

import cv2
import numpy as np

POLICY = 'led_image_lead_axis_local_consensus_v1'


def lead_axis(rgb, box):
    """Find two opposite, aligned leads; near-square detector boxes are not axes.

    Returns no decision when either lead is missing or both axes are supported.
    The pixel evidence determines orientation only: annotation centres need not
    coincide with the diode body centre, so this function never moves a box.
    """
    height, width = rgb.shape[:2]
    x1, y1, x2, y2 = map(float, box)
    if not all(math.isfinite(x) for x in box) or x2 <= x1 or y2 <= y1:
        raise ValueError('Invalid LED detection box')
    w, h = x2-x1, y2-y1
    cx, cy = (x1+x2)/2, (y1+y2)/2
    left, top = max(0, math.floor(cx-1.15*w)), max(0, math.floor(cy-1.15*h))
    right, bottom = min(width, math.ceil(cx+1.15*w)), min(height, math.ceil(cy+1.15*h))
    if right-left < 8 or bottom-top < 8:
        return {'axis': None, 'reason': 'insufficient_image_context'}
    patch = rgb[top:bottom, left:right]
    ink = (patch.min(axis=2) < 180).astype(np.uint8)
    # Long axial strokes survive; short diagonal light arrows and text do not
    # provide two continuous, opposite leads through the candidate centre.
    def evidence(horizontal):
        a = ink if horizontal else ink.T
        along = w if horizontal else h
        across = h if horizontal else w
        ac = cx-left if horizontal else cy-top
        bc = cy-top if horizontal else cx-left
        kernel = np.ones((1, max(7, round(along*.65))), np.uint8)
        opened = cv2.morphologyEx(a, cv2.MORPH_OPEN, kernel)
        lo, hi = max(0, round(bc-across*.32)), min(a.shape[0], round(bc+across*.32)+1)
        ranges = [(max(0, round(ac-along*1.05)), max(0, round(ac-along*.25))),
                  (min(a.shape[1], round(ac+along*.25)), min(a.shape[1], round(ac+along*1.05)))]
        if hi <= lo or any(b-a < max(4, along*.25) for a,b in ranges):
            return 0.
        sides = [opened[lo:hi, a:b].mean(axis=1) for a,b in ranges]
        return float(np.minimum(*sides).max())
    horizontal, vertical = evidence(True), evidence(False)
    best, other = max(horizontal, vertical), min(horizontal, vertical)
    axis = ('horizontal' if horizontal > vertical else 'vertical') if best >= .65 and best-other >= .3 else None
    return {'axis': axis, 'reason': 'opposite_aligned_image_leads' if axis else 'lead_axis_ambiguous',
            'horizontal_support': round(horizontal, 4), 'vertical_support': round(vertical, 4),
            'roi_top_left': [left, top, right, bottom]}


def _center(box):
    return ((box[0]+box[2])/2, (box[1]+box[3])/2)


def _local_center(rgb, box, others, predict):
    """Two independent contexts must support the same unambiguous LED centre."""
    height, width = rgb.shape[:2]
    original = _center(box)
    span = max(box[2]-box[0], box[3]-box[1])
    radius = max(18., min(60., .9*span))
    views, selections, regions = [], [], []
    for side in (384, 640):
        x = max(0, min(max(0, width-side), round(original[0]-side/2)))
        y = max(0, min(max(0, height-side), round(original[1]-side/2)))
        region = [x, y, min(width, x+side), min(height, y+side)]
        if region in regions:
            return None, {'reason':'repeated_context_is_not_independent', 'views':views}
        regions.append(region)
        candidates = []
        for row in predict(rgb[y:region[3], x:region[2]]):
            if row['class_name'] != 'led' or row['confidence'] < .7:
                continue
            a,b,c,d = row['bbox']
            q = [a+x,b+y,c+x,d+y]
            if not all(math.isfinite(v) for v in q) or c <= a or d <= b:
                continue
            if not (.35*span <= max(c-a,d-b) <= 2.*span):
                continue
            point = _center(q)
            distance = math.dist(point, original)
            if distance > radius:
                continue
            candidates.append({'bbox_top_left':q, 'confidence':float(row['confidence']),
                               'distance_from_original_px':distance})
        candidates.sort(key=lambda row:row['distance_from_original_px'])
        views.append({'roi_top_left':region, 'candidates':candidates[:3]})
        if not candidates:
            return None, {'reason':'local_detection_not_supported', 'views':views}
        if len(candidates)>1 and candidates[1]['distance_from_original_px']-candidates[0]['distance_from_original_px'] < max(5.,.15*span):
            return None, {'reason':'neighboring_LED_competes', 'views':views}
        selections.append(candidates[0])
    disagreement = math.dist(*[_center(row['bbox_top_left']) for row in selections])
    if disagreement > min(5., max(3., .06*span)):
        return None, {'reason':'local_views_disagree', 'views':views, 'disagreement_px':disagreement}
    weights = np.array([row['confidence'] for row in selections])
    candidate = np.average([row['bbox_top_left'] for row in selections], axis=0, weights=weights).tolist()
    point = _center(candidate)
    shift = math.dist(point, original)
    audit = {'reason':'local_views_agree', 'views':views, 'disagreement_px':disagreement, 'shift_px':shift}
    if any(math.dist(point,_center(other)) <= shift+3 for other in others):
        return None, {**audit, 'reason':'would_cross_neighboring_LED_ownership'}
    # Preserve already stable centres instead of adding subpixel detector noise.
    if shift <= 2.5:
        return None, {**audit, 'reason':'original_center_already_stable'}
    return candidate, audit


def refine_led_boxes(rgb, raw_payload, target_payload, predict=None):
    """Refine only Target LED boxes. Text, keys, other parts, pins/nets stay put.

    ``predict`` consumes an RGB crop and returns model detections in crop-local
    top-left coordinates. With no predictor, only supported orientation changes
    are made. No reference labels, case identities or score errors are inputs.
    """
    target = deepcopy(target_payload)
    height, width = rgb.shape[:2]
    leds = {}
    for key, component in raw_payload['components'].items():
        if component['type'] != 'led':
            continue
        b = component['bbox']
        leds[key] = [b[0],height-b[3],b[2],height-b[1]]
    entries = []
    for key, original in leds.items():
        box = original
        local = {'reason':'local_detector_not_requested'}
        if predict is not None:
            candidate, local = _local_center(rgb, original, [b for k,b in leds.items() if k!=key], predict)
            if candidate is not None:
                box = candidate
        axis = lead_axis(rgb, box)
        before = target['components'][key]['bbox'][:]
        cx, cy = _center(box)
        cy = height-cy
        # The public annotation size stays fixed; resolution alone is not
        # evidence that a different online reference uses a scaled box size.
        w,h = (20.,14.) if axis['axis']=='horizontal' else (14.,20.) if axis['axis']=='vertical' else (before[2]-before[0],before[3]-before[1])
        proposed = [cx-w/2,cy-h/2,cx+w/2,cy+h/2]
        if not (0<=proposed[0]<proposed[2]<=width and 0<=proposed[1]<proposed[3]<=height):
            proposed = before
            local = {**local, 'reason':'proposed_box_outside_image'}
        target['components'][key]['bbox'] = proposed
        entries.append({'component_id':key,'raw_bbox_bottom_left':raw_payload['components'][key]['bbox'],
                        'before_bbox_bottom_left':before,'after_bbox_bottom_left':proposed,
                        'axis_evidence':axis,'local_detection':local,
                        'center_changed':math.dist(_center(before),_center(proposed))>1e-6,
                        'bbox_changed':any(abs(a-b)>1e-6 for a,b in zip(before,proposed))})
    return target, {'policy':POLICY,'size':[width,height], 'coordinate_origin':'mixed_explicit_field_names',
                    'entries':entries, 'changed_LED_boxes':sum(row['bbox_changed'] for row in entries),
                    'changed_LED_centers':sum(row['center_changed'] for row in entries),
                    'input_contract':'current image and model predictions only; no references or reported errors'}


def yolo_predictor(weights, device):
    """Reuse the frozen detector checkpoint; initialize once for the page."""
    from ultralytics import YOLO
    model = YOLO(str(weights))

    def predict(rgb):
        result = model.predict(np.ascontiguousarray(rgb[:,:,::-1]), imgsz=640, conf=.7, iou=.5,
                               device=device, verbose=False, save=False)[0]
        return [{'bbox':box,'confidence':score,'class_name':result.names[int(cls)]}
                for box,score,cls in zip(result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist(),
                                         result.boxes.cls.cpu().tolist())]
    return predict
