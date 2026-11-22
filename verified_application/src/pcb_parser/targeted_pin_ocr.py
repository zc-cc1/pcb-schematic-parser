"""Fill missing pin fields from localized RGB OCR, preserving established fields.

Observed text only: no numbering order, pin-name convention, title identity,
library lookup, annotation access, or modification of learned pin locations.
"""
from collections import Counter, defaultdict
from copy import deepcopy
import math
import time

import cv2
import numpy as np

from .component_text import bbox, iou, restore_polygon
from .pin_text import local_views, contexts
from .pin_text_v2 import bind_pin_text_v2, tokens_with_views, cost_v2
from .pin_output_rules import sync_fields
from .submission_fields import validate_fields


def select_rois(pins, records, queue, cfg, text_cfg):
    records_by_index = {r['detection_index']: r for r in records}
    priority = {}
    for row in queue['networks']:
        rank = (bool(row['non_field_export_blockers']), row['completion_task_count'], row['missing_field_total'])
        for pin in row['pin_candidates']:
            key = pin['candidate_id']
            priority[key] = min(priority.get(key, rank), rank)
    local_cfg = {**text_cfg, 'local_ocr': {
        'tangent_padding_px': cfg['tangent_padding_px'], 'max_rois': len(pins['pin_candidates'])}}
    views, _ = local_views(pins, pins, local_cfg)
    pmap = {p['candidate_id']: p for p in pins['pin_candidates']}
    selected = []
    for row in views:
        pin = pmap[row['candidate_id']]
        record = records_by_index[pin['owner_detection_index']]
        if record['type'] not in cfg['component_types'] or not record.get('designator'):
            continue
        if pin.get('text_hypothesis_primary_candidate_id'):
            continue
        visible_evidence = any(p.get('confidence',0)>=cfg['adoption_min_confidence']
                               and not p.get('approximate_split_geometry')
                               for field in ['number','pinname'] if pin.get(field) is None
                               for p in pin.get('text_candidates',{}).get(field,[]))
        selected.append({**row, 'priority': priority.get(pin['candidate_id'], (True, 999, 999)),
                         'existing_unassigned_glyph_evidence': visible_evidence})
    selected.sort(key=lambda r: (not r['existing_unassigned_glyph_evidence'],r['priority'],r['candidate_id']))
    return selected[:cfg['max_rois']], max(0, len(selected)-cfg['max_rois'])


def existing_readings(pins):
    """Retain unsplit original glyph candidates without inventing word boxes."""
    output=[];seen=set()
    for pin in pins['pin_candidates']:
        for field in ['number','pinname']:
            for reading in pin.get('text_candidates',{}).get(field,[]):
                if (reading.get('approximate_split_geometry') or reading.get('source_text')!=reading.get('text')
                        or not reading.get('source_view') or not reading.get('bbox')):continue
                key=(reading['text'],tuple(reading['bbox']),reading['source_view'])
                if key in seen:continue
                seen.add(key);x1,y1,x2,y2=reading['bbox']
                output.append({'text_id':f'EXISTING_PIN_GLYPH_{len(output)+1:05d}','text':reading['text'],
                    'bbox':reading['bbox'],'polygon':[[x1,y1],[x2,y1],[x2,y2],[x1,y2]],
                    'confidence':reading['confidence'],'view':reading['source_view'],
                    'source':'original_unassigned_model_glyph_candidate'})
    return output


def collect_readings(rgb, rois, engine, cfg):
    if (not cfg['scales'] or len(set(cfg['scales'])) != len(cfg['scales'])
            or any(not math.isfinite(s) or s <= 0 for s in cfg['scales'])
            or not cfg['rotations'] or len(set(cfg['rotations'])) != len(cfg['rotations'])
            or any(r not in (0, 90) for r in cfg['rotations'])):
        raise ValueError('Invalid targeted OCR views')
    texts, rejected, timings = [], [], []
    for roi in rois:
        x1,y1,x2,y2 = roi['crop']
        patch = rgb[y1:y2,x1:x2]
        for rotation in cfg['rotations']:
            rotated = cv2.rotate(patch, cv2.ROTATE_90_CLOCKWISE) if rotation == 90 else patch
            for scale in cfg['scales']:
                image = cv2.resize(rotated,None,fx=scale,fy=scale,interpolation=cv2.INTER_CUBIC)
                start = time.perf_counter()
                result = engine(np.ascontiguousarray(image[:,:,::-1]),use_det=True,use_cls=True,use_rec=True)
                view = f"missing_pin_{roi['candidate_id']}_s{scale}_r{rotation}"
                count = 0
                if result.boxes is not None:
                    for polygon,text,confidence in zip(result.boxes,result.txts,result.scores):
                        confidence = float(confidence)
                        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
                            raise ValueError('Invalid localized OCR confidence')
                        if confidence < cfg['min_observation_confidence']:continue
                        polygon = np.asarray(polygon,dtype=float)
                        if polygon.shape != (4,2) or not np.isfinite(polygon).all():
                            raise ValueError('Invalid localized OCR polygon')
                        restored = restore_polygon(polygon,roi['crop'],scale,rotation)
                        box = bbox(restored);guard = cfg['crop_boundary_guard_px']
                        row = {'text':str(text),'confidence':confidence,'bbox':box,'polygon':restored,'view':view,
                               'target_candidate_id':roi['candidate_id']}
                        if box[0] <= x1+guard or box[1] <= y1+guard or box[2] >= x2-guard or box[3] >= y2-guard:
                            rejected.append({**row,'reason':'possible_truncated_glyph_at_crop_boundary'});continue
                        row['text_id'] = f'TARGET_PIN_TEXT_{len(texts)+1:05d}'
                        texts.append(row);count += 1
                timings.append({'view':view,'crop':roi['crop'],'observations':count,'wall_seconds':time.perf_counter()-start})
    return texts, {'rois':rois,'views':timings,'localized_observations':len(texts),
                   'rejected_boundary_observations':rejected,'same_model_views_not_independent':True}


def adopt_missing_fields(pins, graph, records, texts, target_ids, cfg, text_cfg, passive_cfg):
    validate_fields(pins,graph,records)
    result = deepcopy(pins)
    # Exclude every established field glyph from the new assignment. Derived
    # library-number evidence shares a name box and is also protected.
    occupied = [e['bbox'] for p in pins['pin_candidates'] for field,e in p.get('text_evidence',{}).items()
                if field in {'number','pinname'} and p.get(field) is not None and e.get('bbox')]
    blocked_records = deepcopy(records)+[{'fields':{'Name':{'bbox':b}}} for b in occupied]
    work = deepcopy(pins)
    cindex = {c['detection_index']:c for c in pins['components']}
    work['pin_candidates'] = [p for p in work['pin_candidates'] if p['owner_status']=='assigned'
                              and cindex[p['owner_detection_index']]['type'] in cfg['component_types']]
    experimental, _ = bind_pin_text_v2(work,texts,text_cfg,blocked_records)
    context_map = contexts(work,text_cfg)
    observed_tokens = tokens_with_views(texts)
    proposals = {p['candidate_id']:p for p in experimental['pin_candidates']}
    claimed = {(p['owner'],p['number']) for p in pins['pin_candidates'] if p.get('number') is not None}
    occupied_boxes = occupied.copy()
    changes, rejected = [], []
    for pin in result['pin_candidates']:
        if pin['candidate_id'] not in target_ids or pin['candidate_id'] not in proposals:continue
        proposal = proposals[pin['candidate_id']]
        for field in ['number','pinname']:
            if pin.get(field) is not None or proposal.get(field) is None:continue
            evidence = proposal['text_evidence'][field]
            text = proposal[field];box = evidence['bbox'];reasons = []
            readings = [r for r in evidence.get('alternatives',[]) if r.get('confidence',0)>=cfg['adoption_min_confidence']
                        and not r.get('approximate_split_geometry') and r.get('source_text')==r.get('text')
                        and iou(r['bbox'],box)>=.4]
            votes = defaultdict(set)
            for reading in readings:votes[reading['text']].add(reading['source_view'])
            if evidence.get('confidence',0)<cfg['adoption_min_confidence']:reasons.append('insufficient_confidence')
            if len(votes[text])<cfg['adoption_min_views']:reasons.append('insufficient_same_glyph_views')
            local_prefix=f"missing_pin_{pin['candidate_id']}_"
            if not any(v.startswith(local_prefix) for v in votes[text]):reasons.append('no_matching_local_ROI_observation')
            if any(other != text and len(views)>=cfg['conflicting_spelling_min_views'] for other,views in votes.items()):
                reasons.append('conflicting_supported_spelling')
            if evidence.get('global_assignment_margin',0)<cfg['adoption_min_assignment_margin']:reasons.append('ambiguous_assignment')
            if any(iou(box,b)>=.45 for b in occupied_boxes):reasons.append('glyph_already_used_by_existing_field')
            # Repeated rotated crops can repeatedly drop a narrow letter (IN
            # -> N). One strong whole spelling at the same glyph is enough to
            # withhold a new fragment; it is not enough to fill its replacement.
            if field=='pinname' and any(other['text'] != text and text in other['text']
                    and len(other['text'])>len(text) and other['confidence']>=cfg['adoption_min_confidence']
                    and not other.get('approximate_split_geometry') and other['source_text']==other['text']
                    and iou(other['bbox'],box)>=.4
                    and cost_v2(other,context_map[pin['candidate_id']],field,text_cfg,occupied) is not None
                    for other in observed_tokens):
                reasons.append('supported_longer_observed_spelling_requires_review')
            kind = cindex[pin['owner_detection_index']]['type']
            if field=='number':
                if (pin['owner'],text) in claimed:reasons.append('number_already_claimed_on_component')
                if kind in passive_cfg['passive_candidate_numbers'] and text not in passive_cfg['passive_candidate_numbers'][kind]:
                    reasons.append('passive_number_outside_supported_literal_range')
            row = {'candidate_id':pin['candidate_id'],'owner':pin['owner'],'field':field,'text':text,
                   'evidence':evidence,'withheld_reasons':reasons}
            if reasons:rejected.append(row);continue
            pin[field] = text;pin[field+'_status'] = 'observed_targeted_local_OCR_missing_field_fill'
            pin.setdefault('text_evidence',{})[field] = {**evidence,'source':'localized_RGB_pin_field_recovery',
                'observed_support_views':sorted(votes[text]),'source_annotation_used':False}
            occupied_boxes.append(box)
            if field=='number':claimed.add((pin['owner'],text))
            changes.append(row)
    updated_graph = sync_fields(result,graph)
    by_pin = {p['candidate_id']:p for p in result['pin_candidates']}
    reference_additions=[]
    for change in changes:
        if change['field']!='number':continue
        pin=by_pin[change['candidate_id']]
        for net in updated_graph['networks']:
            if pin['candidate_id'] not in net['pin_candidate_ids'] or 'reference_pin_references' not in net:continue
            ref=f"{pin['owner']}.{pin['number']}"
            if ref not in net['reference_pin_references']:
                net['reference_pin_references'].append(ref)
                reference_additions.append({'network_id':net['network_id'],'reference':ref,'candidate_id':pin['candidate_id']})
    validate_fields(result,updated_graph,records)
    topology=lambda g:[{k:v for k,v in n.items() if k!='reference_pin_references'} for n in g['networks']]
    if graph['edges']!=updated_graph['edges'] or topology(graph)!=topology(updated_graph):
        raise ValueError('Localized field recovery changed topology')
    for old,new in zip(pins['pin_candidates'],result['pin_candidates']):
        if any(old.get(k)!=new.get(k) for k in ['candidate_id','point','owner','owner_detection_index','owner_status']):
            raise ValueError('Localized field recovery changed localization/ownership')
        if any(old.get(field) is not None and old.get(field)!=new.get(field) for field in ['number','pinname']):
            raise ValueError('Localized field recovery overwrote established fields')
    return result,updated_graph,{'changes':changes,'withheld':rejected,'derived_reference_additions':reference_additions,'summary':{
        'missing_numbers_added':sum(r['field']=='number' for r in changes),
        'missing_names_added':sum(r['field']=='pinname' for r in changes),
        'existing_fields_overwritten':0,'tip_coordinates_changed':0,'network_members_changed':0,
        'withheld_reason_counts':dict(Counter(r for row in rejected for r in row['withheld_reasons']))}}
