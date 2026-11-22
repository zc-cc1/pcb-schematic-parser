"""Explicit output conventions, separate from OCR recognition and topology.

PDF p25 names a label pin after its label and names the example resistor pin
after its number. Extension of the latter to other passives is a development
candidate profile, never silently mixed into the conservative profile.
"""
from collections import Counter, defaultdict
from copy import deepcopy
import math

from .submission_fields import validate_fields


def box_iou(a, b):
    area = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - area
    return area / union if union > 0 else 0


def number_views(pin, cfg):
    """Only repeated readings of the same literal, unsplit local number glyph."""
    evidence = pin.get('text_evidence', {}).get('number', {})
    number = pin.get('number')
    if (pin.get('number_status') not in {'observed_OCR_text','observed_targeted_local_OCR_missing_field_fill'} or evidence.get('text') != number
            or evidence.get('global_assignment_margin', 0) < cfg['passive_min_assignment_margin']
            or not evidence.get('bbox')):
        return []
    readings = [evidence] + evidence.get('alternatives', [])
    views = set()
    for reading in readings:
        if (reading.get('text') == number and reading.get('source_text') == number
                and not reading.get('approximate_split_geometry')
                and reading.get('confidence', 0) >= cfg['passive_min_confidence']
                and reading.get('bbox')
                and box_iou(evidence['bbox'], reading['bbox']) >= cfg['number_view_box_iou']
                and reading.get('source_view')):
            views.add(reading['source_view'])
    return sorted(views)


def sync_fields(pins, graph):
    graph = deepcopy(graph)
    by_id = {p['candidate_id']: p for p in pins['pin_candidates']}
    for attachment in graph['pin_attachments']:
        source = by_id[attachment['candidate_id']]
        for field in ['owner', 'owner_status', 'number', 'pinname']:
            attachment[field] = source.get(field)
    return graph


def complete_pin_output_names(pins, graph, records, cfg):
    """No new pin numbers, IDs, tip points, components or network memberships."""
    validate_fields(pins, graph, records)
    if (cfg['unobserved_two_terminal_number_policy'] != 'keep_unresolved'
            or cfg['unobserved_power_identity_policy'] != 'keep_unresolved'
            or cfg['blank_pinname_policy'] != 'keep_unresolved'
            or cfg['passive_name_policy'] != 'candidate_only_number_as_name_example_extension'):
        raise ValueError('Unsupported numbering, identity or name policy')
    conservative = deepcopy(pins)
    by_index = {r['detection_index']: r for r in records}
    owned = defaultdict(list)
    for pin in conservative['pin_candidates']:
        if pin['owner_status'] == 'assigned':
            owned[pin['owner_detection_index']].append(pin)
    sites = defaultdict(list)
    for site in graph.get('label_sites', []):
        if site.get('status') == 'bound_ordinary_label' and site.get('evidence_kind') == 'detected_symbol_and_owned_pin':
            sites[site.get('detection_index')].append(site)
    changes, proposals = [], []
    for index, members in owned.items():
        record = by_index[index]
        if record['type'] not in cfg['ordinary_label_types'] or len(members) != 1:
            continue
        pin = members[0]
        matching = [s for s in sites[index] if s.get('candidate_id') == pin['candidate_id']
                    and s.get('label') == record.get('designator') == record.get('Name')
                    and s.get('confidence', 0) >= cfg['label_min_confidence']
                    and s.get('distinct_view_votes', {}).get(s['label'], 0) >= cfg['label_min_views']]
        if (not matching or not record.get('designator') or pin.get('pinname') is not None
                or pin.get('number') != '1'
                or pin.get('text_candidates', {}).get('pinname')):
            continue
        site = max(matching, key=lambda s: s['confidence'])
        pin.update(pinname=site['label'], pinname_status='PDF_example_observed_ordinary_label_name')
        evidence = {'source': 'PDF_0803Rev1_page25_RUN_1_pinname_example', 'site_id': site['site_id'],
                    'observed_label_text': site['label'], 'observed_pinname_role_glyph': False,
                    'distinct_OCR_views': site['distinct_view_votes'][site['label']],
                    'OCR_recognition_improved': False, 'canonical_component_order_inferred': False}
        pin.setdefault('text_evidence', {})['pinname'] = evidence
        changes.append({'candidate_id': pin['candidate_id'], 'owner': pin['owner'], 'number': pin['number'],
                        'pinname': pin['pinname'], 'profile': 'conservative', **evidence})
    conservative_graph = sync_fields(conservative, graph)
    candidate = deepcopy(conservative)
    for pin in candidate['pin_candidates']:
        if pin['owner_status'] != 'assigned':
            continue
        record = by_index[pin['owner_detection_index']]
        if record['type'] not in cfg['passive_candidate_numbers'] or pin.get('pinname') is not None:
            continue
        reasons = []
        if not record.get('designator'):
            reasons.append('component_identity_unresolved')
        if pin.get('number') not in cfg['passive_candidate_numbers'][record['type']]:
            reasons.append('number_not_in_supported_literal_pin_range')
        views = number_views(pin, cfg)
        if len(views) < cfg['passive_min_views']:
            reasons.append('insufficient_local_number_glyph_evidence')
        if pin.get('text_candidates', {}).get('pinname'):
            reasons.append('existing_name_candidates_require_review')
        evidence = {'source': 'reference_pdf_pinname_number_example_extension',
                    'example_direct_component_type': 'resistor', 'predicted_component_type': record['type'],
                    'direct_rule_for_all_passives_in_PDF': False, 'golden_verified': False,
                    'observed_number_views': views, 'observed_pinname_role_glyph': False,
                    'OCR_recognition_improved': False, 'provisional': True}
        accepted = not reasons
        proposals.append({'candidate_id': pin['candidate_id'], 'owner': pin['owner'], 'number': pin.get('number'),
                          'proposed_pinname': pin.get('number'), 'accepted_in_candidate_profile': accepted,
                          'withheld_reasons': reasons, **evidence})
        if accepted:
            pin.update(pinname=pin['number'], pinname_status='candidate_PDF_example_number_as_passive_name')
            pin.setdefault('text_evidence', {})['pinname'] = evidence
            changes.append({'candidate_id': pin['candidate_id'], 'owner': pin['owner'], 'number': pin['number'],
                            'pinname': pin['pinname'], 'profile': 'candidate', **evidence})
    candidate_graph = sync_fields(candidate, graph)
    for updated, updated_graph in [(conservative, conservative_graph), (candidate, candidate_graph)]:
        validate_fields(updated, updated_graph, records)
        if graph['edges'] != updated_graph['edges'] or graph['networks'] != updated_graph['networks']:
            raise ValueError('Output field completion altered topology')
        for old, new in zip(pins['pin_candidates'], updated['pin_candidates']):
            if any(old.get(k) != new.get(k) for k in ['candidate_id', 'point', 'owner', 'owner_detection_index', 'number']):
                raise ValueError('Output name completion changed identity, number or geometry')
    summary = {'ordinary_label_names_added': sum(c['profile']=='conservative' for c in changes),
               'candidate_passive_names_added': sum(c['profile']=='candidate' for c in changes),
               'candidate_passive_names_by_type': dict(Counter(c['predicted_component_type'] for c in changes if c['profile']=='candidate')),
               'new_observed_number_glyphs': 0, 'automatic_two_terminal_numbers': 0,
               'power_component_identities_invented': 0, 'blank_names_filled': 0,
               'tip_coordinates_changed': 0, 'network_members_changed': 0, 'recognition_accuracy_gain_claimed': False}
    return conservative, conservative_graph, candidate, candidate_graph, {'summary': summary, 'changes': changes, 'passive_proposals': proposals}


def power_network_guard(pins, graph, records, cfg):
    """Detect omitted model-owned power tips near an existing model wire.

This is a conservative export gate, not a raw-image segmentation operation.
It proposes neither missing pin coordinates nor power network merges.
"""
    owned = defaultdict(list)
    attachments = {p['candidate_id']: p for p in graph['pin_attachments']}
    for pin in pins['pin_candidates']:
        if pin['owner_status'] == 'assigned':
            owned[pin['owner_detection_index']].append(pin)
    edges = {e['edge_id']: e for e in graph['edges']}
    blocks, reviews = [], []
    padding = cfg['power_edge_guard_px']
    for record in records:
        if record['type'] not in cfg['power_types']:
            continue
        members = owned[record['detection_index']]
        x1, y1, x2, y2 = record['bbox_top_left']
        nearby = []
        for net in graph['networks']:
            if not net['pin_candidate_ids']:
                continue
            represented = any(p['candidate_id'] in net['pin_candidate_ids']
                              and attachments[p['candidate_id']].get('attachment_status') == 'attached' for p in members)
            if represented:
                continue
            close = any(x1-padding <= point[0] <= x2+padding and y1-padding <= point[1] <= y2+padding
                        for eid in net['edge_ids'] for point in edges[eid]['polyline'])
            if close:
                nearby.append(net['network_id'])
                blocks.append({'network_id': net['network_id'], 'detection_index': record['detection_index'],
                               'component_id': record['component_id'], 'component_type': record['type'],
                               'bbox_top_left': record['bbox_top_left'],
                               'reasons': ['detected_power_symbol_near_wire_without_attached_member'],
                               'heuristic_gate_not_certified_topology': True})
        reviews.append({'detection_index': record['detection_index'], 'component_id': record['component_id'],
                        'component_type': record['type'], 'bbox_top_left': record['bbox_top_left'],
                        'designator': record.get('designator'), 'assigned_tip_count': len(members),
                        'attached_tip_count': sum(attachments[p['candidate_id']].get('attachment_status')=='attached' for p in members),
                        'candidate_ids': [p['candidate_id'] for p in members],
                        'nearby_network_ids_with_missing_power_member': nearby,
                        'identity_status': 'recognized' if record.get('designator') else 'unresolved',
                        'power_logical_merges_applied': False})
    return {'blocks': blocks, 'components': reviews,
            'summary': {'power_components': len(reviews), 'power_without_owned_tip': sum(r['assigned_tip_count']==0 for r in reviews),
                        'networks_with_possible_omitted_power_member': len({b['network_id'] for b in blocks}),
                        'power_merges_applied': 0, 'pin_points_created': 0}}


def apply_power_export_gate(submission, coverage, guard):
    out, updated = deepcopy(submission), deepcopy(coverage)
    excluded = []
    blocked_ids = sorted({b['network_id'] for b in guard['blocks']})
    for network_id in blocked_ids:
        if network_id in out['nets']:
            del out['nets'][network_id]
            excluded.append(network_id)
        # Record the gate even if another field already withheld the network.
        # Otherwise strict triage incorrectly advertises a field-only target.
        rows = [row for row in updated['withheld_networks'] if row['network_id'] == network_id]
        if not rows:
            row = {'network_id': network_id, 'reasons': []}
            updated['withheld_networks'].append(row)
            rows = [row]
        reason = 'detected_power_symbol_near_wire_without_attached_member'
        if not any(reason in row['reasons'] for row in rows):
            rows[0]['reasons'].append(reason)
    updated['network_reason_counts'] = dict(Counter(r for n in updated['withheld_networks'] for r in n['reasons']))
    updated.update(exported_nets=len(out['nets']), exported_network_ids=sorted(out['nets']),
                   power_guard_excluded_network_ids=excluded, power_guard_is_heuristic=True)
    updated['power_guard_blocked_network_ids'] = blocked_ids
    return out, updated


def point_segment_distance(point, start, end):
    dx, dy = end[0]-start[0], end[1]-start[1]
    denominator = dx*dx + dy*dy
    t = max(0, min(1, ((point[0]-start[0])*dx+(point[1]-start[1])*dy)/denominator)) if denominator else 0
    return math.hypot(point[0]-start[0]-t*dx, point[1]-start[1]-t*dy)


def simplify_model_polyline(points, tolerance):
    """Compress an existing graph path with bounded deviation and fixed ends."""
    if not math.isfinite(tolerance) or not 0 <= tolerance <= 1:
        raise ValueError('Candidate edge tolerance must be within 0..1 original pixels')
    if len(points) < 2:
        return list(points), 0.0
    keep={0,len(points)-1};pending=[(0,len(points)-1)]
    while pending:
        start,end=pending.pop()
        if end-start<=1:continue
        index,dist=max(((i,point_segment_distance(points[i],points[start],points[end])) for i in range(start+1,end)),key=lambda x:x[1])
        if dist>tolerance:
            keep.add(index);pending.extend([(start,index),(index,end)])
    indices=sorted(keep)
    maximum=max((point_segment_distance(points[i],points[start],points[end])
                 for start,end in zip(indices,indices[1:]) for i in range(start,end+1)),default=0.0)
    if maximum>tolerance+1e-8:raise ValueError('Polyline compression exceeded bound')
    return [points[i] for i in indices],maximum


def compress_candidate_edges(submission, graph, height, tolerance):
    """Candidate JSON only; never bridge paths, alter graph or move pin tips."""
    out=deepcopy(submission);edges={e['edge_id']:e for e in graph['edges']};rows=[]
    for net in graph['networks']:
        if net['network_id'] not in out['nets'] or net.get('flywire'):continue
        segments={};input_segments=0;maximum=0.0;paths=[]
        for eid in net['edge_ids']:
            original=edges[eid]['polyline'];points,deviation=simplify_model_polyline(original,tolerance)
            input_segments+=max(0,len(original)-1);maximum=max(maximum,deviation)
            paths.append({'edge_id':eid,'original_endpoints':[original[0],original[-1]] if original else [],
                          'simplified_polyline_top_left':points,'max_deviation_px':deviation})
            for a,b in zip(points,points[1:]):
                segments[f'edge_{len(segments)+1}']=[{'x':a[0],'y':height-a[1]},{'x':b[0],'y':height-b[1]}]
        out['nets'][net['network_id']]['edges']=segments
        rows.append({'network_id':net['network_id'],'input_pixel_segments':input_segments,'output_segments':len(segments),
                     'max_deviation_px':maximum,'path_endpoints_preserved':True,'paths':paths})
    return out,{'policy':'candidate_only_bounded_model_polyline_compression','tolerance_original_px':tolerance,
                'input_pixel_segments':sum(r['input_pixel_segments'] for r in rows),
                'output_segments':sum(r['output_segments'] for r in rows),'networks':rows,
                'new_connections_created':0,'graph_or_pin_geometry_changed':False,'reference_NetLineF1':None}
