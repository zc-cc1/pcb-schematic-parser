"""Provisional two-terminal formatting, kept separate from recognition.

Only known R/C/L identities with two opposed, attached model contacts qualify.
No point, edge or network membership is created or removed here.
"""
from collections import Counter, defaultdict
from copy import deepcopy
import math
import re

from .pin_output_rules import point_segment_distance, sync_fields
from .submission_fields import validate_fields


def wire_axis(pin, graph, cfg):
    attachment = next(a for a in graph['pin_attachments'] if a['candidate_id'] == pin['candidate_id'])
    reasons = []
    if attachment.get('attachment_status') != 'attached':reasons.append('wire_not_uniquely_attached')
    if attachment.get('wire_gap_px') is None or attachment['wire_gap_px'] > cfg['maximum_wire_gap_px']:reasons.append('wire_gap_too_large')
    physical = attachment.get('physical_network_id')
    alternatives = {a.get('physical_network_id') for a in attachment.get('alternatives', [])}
    if not physical or alternatives - {physical}:reasons.append('ambiguous_physical_attachment')
    projection = attachment.get('wire_projection')
    if reasons or projection is None:return None, {'reasons': reasons or ['wire_projection_missing']}
    net = next((n for n in graph.get('physical_networks', []) if n['network_id'] == physical), None)
    if not net:return None, {'reasons': ['physical_network_missing']}
    edges = {e['edge_id']: e for e in graph['edges']};segments = []
    for eid in net['edge_ids']:
        edge = edges[eid]
        if edge.get('mean_probability', 0) < cfg['minimum_wire_probability']:continue
        for a, b in zip(edge['polyline'], edge['polyline'][1:]):
            dx, dy = abs(b[0]-a[0]), abs(b[1]-a[1]);major = max(dx, dy)
            if (major < cfg['minimum_axis_segment_length_px']
                    or min(dx, dy) > cfg['maximum_axis_slope_ratio']*major
                    or point_segment_distance(projection, a, b) > cfg['projection_segment_tolerance_px']):continue
            segments.append({'edge_id': eid, 'axis': 0 if dx > dy else 1, 'segment': [a, b],
                             'mean_probability': edge['mean_probability']})
    axes = {s['axis'] for s in segments}
    proof = {'physical_network_id': physical, 'wire_projection': projection,
             'wire_gap_px': attachment['wire_gap_px'], 'local_model_segments': segments, 'reasons': []}
    if len(axes) != 1:proof['reasons'].append('local_wire_axis_ambiguous_or_missing');return None, proof
    return next(iter(axes)), proof


def alignment(pin, record, axis, cfg):
    box = record['bbox_top_left'];tangent = 1-axis
    offset = abs(pin['point'][tangent]-(box[tangent]+box[tangent+2])/2)
    tolerance = max(cfg['alignment_min_px'], cfg['alignment_body_fraction']*(box[tangent+2]-box[tangent]))
    return {'axis': axis, 'tangential_offset_px': offset, 'tangential_tolerance_px': tolerance,
            'aligned': offset <= tolerance}


def rebuild_references(pins, graph, records):
    graph = sync_fields(pins, graph);pmap = {p['candidate_id']: p for p in pins['pin_candidates']}
    known = {r['detection_index'] for r in records if r.get('designator')}
    for net in graph['networks']:
        net['official_pin_references'] = sorted({f"{p['owner']}.{p['number']}" for cid in net['pin_candidate_ids']
            for p in [pmap[cid]] if p['owner_status'] == 'assigned' and p['owner_detection_index'] in known and p.get('number') is not None})
    return graph


def complete_engineering_fields(pins, graph, records, cfg):
    validate_fields(pins, graph, records)
    if (cfg['profile'] != 'engineering_only_provisional_numbering'
            or cfg['number_policy'] != 'left_or_top_1_right_or_bottom_2' or cfg['name_policy'] != 'number_as_name'
            or cfg['official_numbering_verified']):raise ValueError('Unsupported engineering convention')
    updated = deepcopy(pins);rmap = {r['detection_index']: r for r in records}
    withdrawals, protected, changes, proposals = [], [], [], []
    for pin in updated['pin_candidates']:
        if pin['owner_status'] != 'assigned':continue
        record = rmap[pin['owner_detection_index']];box = record['bbox_top_left']
        if record['type'] not in cfg['ownership_withdrawal_types']:continue
        width, height = box[2]-box[0], box[3]-box[1];dominant = 0 if width > height else 1
        if min(width, height) <= 0 or max(width, height)/min(width, height) < cfg['ownership_min_body_aspect_ratio']:continue
        axis, wire = wire_axis(pin, graph, cfg)
        if axis is None or axis != dominant:continue
        align = alignment(pin, record, axis, cfg)
        if align['aligned']:continue
        proof = {'candidate_id': pin['candidate_id'], 'old_owner': pin['owner'],
                 'old_owner_detection_index': pin['owner_detection_index'], 'wire': wire, **align,
                 'reason': 'blank_tip_off_passive_body_wire_axis'}
        if pin.get('number') is not None or pin.get('pinname') is not None:protected.append(proof);continue
        withdrawals.append(proof)
        pin.update(owner=None, owner_detection_index=None, owner_status='engineering_geometry_withheld',
                   engineering_owner_evidence=proof)
    owned = defaultdict(list)
    for pin in updated['pin_candidates']:
        if pin['owner_status'] == 'assigned':owned[pin['owner_detection_index']].append(pin)
    for record in records:
        if record['type'] not in cfg['passive_types'] or not record.get('designator'):continue
        members = owned[record['detection_index']];reasons = [];proofs = []
        if (record.get('duplicate_of_detection_index') is not None
                or record.get('detection_confidence', 0) < cfg['minimum_detection_confidence']):reasons.append('component_detection_unqualified')
        if len(members) != 2:reasons.append('requires_exactly_two_owned_model_tips')
        for pin in members:
            axis, wire = wire_axis(pin, graph, cfg)
            proof = {'candidate_id': pin['candidate_id'], 'wire': wire, 'axis': axis}
            if pin.get('confidence', 0) < cfg['minimum_tip_confidence']:reasons.append('tip_confidence_unqualified')
            if axis is None:reasons.extend(wire['reasons'])
            else:
                proof.update(alignment(pin, record, axis, cfg))
                if not proof['aligned']:reasons.append('tip_off_passive_body_wire_axis')
            proofs.append(proof)
        axis = proofs[0]['axis'] if len(proofs) == 2 else None
        if len(proofs) == 2 and axis is not None and proofs[1]['axis'] == axis:
            a, b = sorted(members, key=lambda p: p['point'][axis]);box = record['bbox_top_left']
            span = box[axis+2]-box[axis];center = (box[axis]+box[axis+2])/2
            if (b['point'][axis]-a['point'][axis] < cfg['minimum_pair_span_body_fraction']*span
                    or a['point'][axis] > center-cfg['minimum_end_offset_body_fraction']*span
                    or b['point'][axis] < center+cfg['minimum_end_offset_body_fraction']*span):reasons.append('tips_not_on_opposite_body_ends')
            if proofs[0]['wire'].get('physical_network_id') == proofs[1]['wire'].get('physical_network_id'):reasons.append('both_tips_on_same_physical_wire')
            for number, pin in zip(['1', '2'], [a, b]):
                if pin.get('number') not in {None, number}:reasons.append('existing_number_conflicts_with_fixed_convention')
                if pin.get('pinname') not in {None, number}:reasons.append('existing_name_conflicts_with_fixed_convention')
                if pin.get('text_candidates', {}).get('pinname'):reasons.append('name_candidates_require_review')
                if pin.get('number') is None and any(c.get('cost', math.inf) <= cfg['number_candidate_veto_max_cost']
                        and re.fullmatch(r'[1-9]\d*', c.get('text', '')) for c in pin.get('text_candidates', {}).get('number', [])):
                    reasons.append('nearby_number_candidates_require_review')
        elif len(proofs) == 2:reasons.append('tip_wire_axes_disagree_or_missing')
        proposal = {'component_id': record['component_id'], 'detection_index': record['detection_index'],
                    'original_owned_member_ids': [p['candidate_id'] for p in members], 'contact_geometry': proofs,
                    'withheld_reasons': sorted(set(reasons))}
        proposals.append(proposal)
        if reasons:continue
        for number, pin in zip(['1', '2'], [a, b]):
            for field, value in [('number', number), ('pinname', number)]:
                if pin.get(field) is not None:continue
                evidence = {'source': 'engineering_two_terminal_fixed_convention_v1', 'profile': 'engineering',
                            'convention': cfg['number_policy'] if field == 'number' else cfg['name_policy'],
                            'observed_OCR_glyph': False, 'official_numbering_verified': False,
                            'component_id': record['component_id'], 'contact_geometry': proofs,
                            'original_owned_member_ids': proposal['original_owned_member_ids']}
                pin[field] = value;pin[field+'_status'] = 'engineering_convention_unverified'
                pin.setdefault('text_evidence', {})[field] = evidence
                changes.append({'candidate_id': pin['candidate_id'], 'owner': pin['owner'], 'field': field, 'value': value, **evidence})
    updated_graph = rebuild_references(updated, graph, records)
    validate_fields(updated, updated_graph, records)
    return updated, updated_graph, {'changes': changes, 'ownership_withdrawals': withdrawals,
        'existing_fields_protected': protected, 'component_proposals': proposals,
        'summary': {'generated_pin_numbers': sum(c['field'] == 'number' for c in changes),
                    'generated_pin_names': sum(c['field'] == 'pinname' for c in changes),
                    'qualified_two_terminal_components': sum(not p['withheld_reasons'] for p in proposals),
                    'blank_owner_assignments_withdrawn': len(withdrawals),
                    'withheld_component_reason_counts': dict(Counter(r for p in proposals for r in p['withheld_reasons']))},
        'original_tip_points_edges_and_members_preserved': True, 'new_OCR_observations': 0,
        'official_numbering_verified': False}
