"""Reconcile complementary OCR fields on two hypotheses of the same lead.

Uses prediction geometry, physical wire attachments and literal OCR only.
Never merges/deletes tips, transfers numbers, or resolves network membership.
"""
from collections import Counter
from copy import deepcopy
import math

from .component_text import iou
from .pin_output_rules import sync_fields
from .submission_fields import validate_fields


OBSERVED_STATUSES = {'observed_OCR_text', 'observed_OCR_v2_missing_name_fill',
                     'observed_targeted_local_OCR_missing_field_fill'}


def support(pin, field, cfg):
    evidence = pin.get('text_evidence', {}).get(field, {})
    value = pin.get(field)
    if (pin.get(field + '_status') not in OBSERVED_STATUSES
            or evidence.get('text') != value or not evidence.get('bbox')
            or evidence.get('source_text') != value or evidence.get('approximate_split_geometry')
            or evidence.get('confidence', 0) < cfg['min_ocr_confidence']
            or evidence.get('global_assignment_margin', 0) < cfg['min_assignment_margin']):
        return [], evidence
    views = set()
    for row in [evidence] + evidence.get('alternatives', []):
        if (row.get('text') == value == row.get('source_text') and row.get('source_view')
                and not row.get('approximate_split_geometry')
                and row.get('confidence', 0) >= cfg['min_ocr_confidence']
                and row.get('bbox') and iou(row['bbox'], evidence['bbox']) >= cfg['glyph_box_iou']):
            views.add(row['source_view'])
    return sorted(views), evidence


def segment_distance(point, a, b):
    dx, dy = b[0]-a[0], b[1]-a[1]
    length2 = dx*dx+dy*dy
    if length2 == 0:
        return math.dist(point, a)
    t = max(0, min(1, ((point[0]-a[0])*dx+(point[1]-a[1])*dy)/length2))
    return math.dist(point, [a[0]+t*dx, a[1]+t*dy])


def reconcile_nearby_tip_fields(pins, graph, records, cfg):
    validate_fields(pins, graph, records)
    if cfg['policy'] != 'fill_missing_name_only_preserve_donor_fields_points_and_members':
        raise ValueError('Unsupported nearby-tip field policy')
    for key in ['max_tip_distance_px', 'max_projection_distance_px', 'max_wire_gap_px',
                'edge_projection_tolerance_px', 'third_tip_clearance_px', 'min_assignment_margin',
                'glyph_row_padding_px', 'side_ambiguity_margin_px']:
        if not math.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError('Invalid nearby-tip distance/margin')
    for key in ['min_tip_confidence', 'min_edge_probability', 'min_ocr_confidence', 'glyph_box_iou']:
        if not math.isfinite(cfg[key]) or not 0 < cfg[key] <= 1:
            raise ValueError('Invalid confidence/IoU')
    if cfg['min_distinct_views'] < 2:
        raise ValueError('At least two distinct OCR views required')
    if cfg['max_tip_distance_px'] > 5:
        raise ValueError('This pilot supports at most 5px between tip hypotheses')
    result = deepcopy(pins)
    pmap = {p['candidate_id']: p for p in pins['pin_candidates']}
    attachments = {a['candidate_id']: a for a in graph['pin_attachments']}
    record_map = {r['detection_index']: r for r in records}
    physical = {n['network_id']: n for n in graph.get('physical_networks', [])}
    logical = {n['network_id']: n for n in graph['networks']}
    edges = {e['edge_id']: e for e in graph['edges']}
    gap_nets = {n for gap in graph.get('gap_candidates', []) for n in gap.get('network_ids', [])}
    proposals = []
    for target in pins['pin_candidates']:
        if (target['owner_status'] != 'assigned' or target.get('number') is None
                or target.get('pinname') is not None):
            continue
        record = record_map[target['owner_detection_index']]
        if record['type'] not in cfg['component_types'] or not record.get('designator'):
            continue
        for donor in pins['pin_candidates']:
            if (donor['candidate_id'] == target['candidate_id'] or donor['owner_status'] != 'assigned'
                    or donor['owner_detection_index'] != target['owner_detection_index']
                    or donor.get('number') is not None or donor.get('pinname') is None):
                continue
            distance = math.dist(target['point'], donor['point'])
            if distance > cfg['max_tip_distance_px']:
                continue
            reasons = []
            a, b = attachments[target['candidate_id']], attachments[donor['candidate_id']]
            if any(not math.isfinite(p['confidence']) or not cfg['min_tip_confidence'] <= p['confidence'] <= 1 for p in [target, donor]):
                reasons.append('weak_model_tip')
            if any(p['candidate_id'] != target['candidate_id'] and p.get('owner') == target['owner']
                   and p.get('number') == target['number'] for p in pins['pin_candidates']):
                reasons.append('number_already_claimed_by_another_tip')
            # Secondary hypotheses may have no text_side metadata in fresh
            # inference. Derive the side from immutable model geometry instead.
            x1, y1, x2, y2 = record['bbox_top_left']
            sides = []
            ambiguous_side = False
            for pin in [target, donor]:
                x, y = pin['point']
                ranked = sorted([('left', abs(x-x1)), ('right', abs(x-x2)),
                                 ('top', abs(y-y1)), ('bottom', abs(y-y2))], key=lambda p: p[1])
                sides.append(ranked[0][0])
                if ranked[1][1]-ranked[0][1] < cfg['side_ambiguity_margin_px']:
                    ambiguous_side = True
            side = sides[0]
            if ambiguous_side or sides[0] != sides[1] or any(p.get('text_side') is not None and p['text_side'] != s for p, s in zip([target, donor], sides)):
                reasons.append('different_or_unresolved_component_side')
            third = [math.dist(p['point'], t['point']) for p in pins['pin_candidates']
                     if p['candidate_id'] not in {target['candidate_id'], donor['candidate_id']}
                     for t in [target, donor]]
            nearest_third = min(third, default=None)
            if nearest_third is not None and nearest_third < distance + cfg['third_tip_clearance_px']:
                reasons.append('competing_nearby_tip')
            pid = a.get('physical_network_id')
            network = physical.get(pid)
            lid = a.get('network_id')
            lnet = logical.get(lid)
            if (a['attachment_status'] != 'attached' or b['attachment_status'] != 'attached'
                    or not pid or pid != b.get('physical_network_id') or network is None
                    or not lid or lid != b.get('network_id') or lnet is None
                    or not {target['candidate_id'], donor['candidate_id']}.issubset(network.get('pin_candidate_ids', []))
                    or not {target['candidate_id'], donor['candidate_id']}.issubset(lnet['pin_candidate_ids'])):
                reasons.append('different_or_unverified_physical_attachment')
            if ((network and network.get('unresolved_junction_ids'))
                    or (lnet and (lnet.get('unresolved_junction_ids') or lnet.get('power_logical_merge') or lnet.get('flywire')))
                    or pid in gap_nets or lid in gap_nets):
                reasons.append('uncertain_network_topology')
            if any(x.get('wire_gap_px') is None or x['wire_gap_px'] > cfg['max_wire_gap_px'] for x in [a, b]):
                reasons.append('distant_wire_attachment')
            if any(other.get('physical_network_id', other.get('network_id')) not in {pid, lid}
                   and other.get('distance_px', math.inf) < (x.get('wire_gap_px') or 0) + 2
                   for x in [a, b] for other in x.get('alternatives', [])):
                reasons.append('competing_wire_attachment')
            projections = [a.get('wire_projection'), b.get('wire_projection')]
            shared = []
            if all(projections):
                if math.dist(*projections) > cfg['max_projection_distance_px']:
                    reasons.append('separated_wire_projections')
                for eid in network.get('edge_ids', []) if network else []:
                    edge = edges.get(eid)
                    probability = edge.get('mean_probability', 0) if edge else 0
                    if not edge or not math.isfinite(probability) or not cfg['min_edge_probability'] <= probability <= 1:
                        continue
                    for index, (v, w) in enumerate(zip(edge['polyline'], edge['polyline'][1:])):
                        if v != w and all(segment_distance(p, v, w) <= cfg['edge_projection_tolerance_px'] for p in projections):
                            shared.append({'edge_id': eid, 'segment_index': index})
                if len(shared) != 1:
                    reasons.append('no_unique_shared_physical_wire_segment')
            else:
                reasons.append('missing_wire_projection')
            number_views, number_evidence = support(target, 'number', cfg)
            name_views, name_evidence = support(donor, 'pinname', cfg)
            if len(number_views) < cfg['min_distinct_views']:
                reasons.append('number_not_supported_by_repeated_literal_OCR')
            if len(name_views) < cfg['min_distinct_views']:
                reasons.append('name_not_supported_by_repeated_literal_OCR')
            if name_evidence.get('bbox') and number_evidence.get('bbox') and all(projections) and side:
                axis = 1 if side in {'left', 'right'} else 0
                row = sum(p[axis] for p in projections)/2
                for evidence in [name_evidence, number_evidence]:
                    box = evidence['bbox'];pad = cfg['glyph_row_padding_px']
                    if not box[axis]-pad <= row <= box[axis+2]+pad:
                        reasons.append('glyph_not_on_shared_lead_row')
                        break
            # A longer same-glyph reading prevents copying a truncated name.
            for field in ['number', 'pinname']:
                owner_pin = target if field == 'number' else donor
                evidence = number_evidence if field == 'number' else name_evidence
                box = evidence.get('bbox')
                competing = ([evidence] + evidence.get('alternatives', [])
                             + [r for p in pins['pin_candidates'] for r in p.get('text_candidates', {}).get(field, [])])
                for row in competing:
                    if (box and row.get('bbox') and row.get('text') != owner_pin[field]
                            and row.get('text') == row.get('source_text') and not row.get('approximate_split_geometry')
                            and row.get('confidence', 0) >= cfg['min_ocr_confidence']
                            and iou(row['bbox'], box) >= cfg['glyph_box_iou']):
                        reasons.append('conflicting_observed_glyph_spelling')
                        break
            # No second numbered tip may already own the name glyph.
            if name_evidence.get('bbox') and any(p['candidate_id'] not in {target['candidate_id'], donor['candidate_id']}
                    and p.get('pinname') is not None and p.get('text_evidence', {}).get('pinname', {}).get('bbox')
                    and iou(p['text_evidence']['pinname']['bbox'], name_evidence['bbox']) >= cfg['glyph_box_iou']
                    for p in pins['pin_candidates']):
                reasons.append('glyph_claimed_by_another_pin')
            proposals.append({'candidate_id': target['candidate_id'], 'donor_candidate_id': donor['candidate_id'],
                              'owner': target['owner'], 'number': target['number'], 'pinname': donor['pinname'],
                              'tip_distance_px': distance, 'nearest_third_tip_distance_px': nearest_third,
                              'physical_network_id': pid, 'network_id': lid, 'shared_wire_segments': shared,
                              'component_side': side, 'side_evidence': 'nearest_model_component_boundary',
                              'number_support_views': number_views, 'name_support_views': name_views,
                              'withheld_reasons': sorted(set(reasons))})
    accepted = [p for p in proposals if not p['withheld_reasons']]
    use_counts = Counter(cid for row in accepted for cid in [row['candidate_id'], row['donor_candidate_id']])
    changes = []
    for proposal in proposals:
        if not proposal['withheld_reasons'] and any(use_counts[cid] > 1 for cid in [proposal['candidate_id'], proposal['donor_candidate_id']]):
            proposal['withheld_reasons'].append('multiple_competing_field_links')
        if proposal['withheld_reasons']:
            continue
        target = next(p for p in result['pin_candidates'] if p['candidate_id'] == proposal['candidate_id'])
        donor = pmap[proposal['donor_candidate_id']]
        target['pinname'] = donor['pinname']
        target['pinname_status'] = 'observed_OCR_name_reconciled_same_physical_lead'
        target.setdefault('text_evidence', {})['pinname'] = {
            **deepcopy(donor['text_evidence']['pinname']), 'source': 'nearby_tip_same_physical_lead_reconciliation',
            'observed_at_candidate_id': donor['candidate_id'], 'field_link': deepcopy(proposal),
            'new_OCR_observation': False, 'source_annotation_used': False}
        changes.append(deepcopy(proposal))
    updated = sync_fields(result, graph)
    validate_fields(result, updated, records)
    # Semantic fields are the only permitted graph changes.
    protected_graph = deepcopy(updated)
    for a, old in zip(protected_graph['pin_attachments'], graph['pin_attachments']):
        a['pinname'] = old.get('pinname')
    if protected_graph != graph or any(old.get(k) != new.get(k) for old, new in zip(pins['pin_candidates'], result['pin_candidates'])
            for k in ['candidate_id', 'point', 'owner', 'owner_status', 'owner_detection_index', 'number']):
        raise ValueError('Nearby-tip reconciliation changed protected fields')
    if any(old.get('pinname') is not None and old['pinname'] != new.get('pinname')
           for old, new in zip(pins['pin_candidates'], result['pin_candidates'])):
        raise ValueError('Nearby-tip reconciliation overwrote a name')
    return result, updated, {'changes': changes, 'withheld': [p for p in proposals if p['withheld_reasons']],
                             'summary': {'names_reconciled': len(changes), 'numbers_added': 0,
                                         'tips_removed_or_moved': 0, 'members_removed_or_merged': 0,
                                         'withheld_reason_counts': dict(Counter(r for p in proposals for r in p['withheld_reasons']))}}
