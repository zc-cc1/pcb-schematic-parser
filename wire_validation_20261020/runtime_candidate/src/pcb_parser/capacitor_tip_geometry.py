"""Model-only capacitor coordinate correction with explicit local edge changes.

Coordinates come from 2x U-Net peaks. Native/1.5x/2x peaks must move monotonically
inward on both opposed leads. Local learned wire pixels must support the entire
extension; no reference, box-edge snapping or raw-pixel segmentation is used.
"""
from collections import Counter, defaultdict, deque
from copy import deepcopy
import math

import numpy as np

from .engineering_network_export import wire_axis
from .pin_output_rules import point_segment_distance
from .submission_fields import validate_fields
from .unet_inference import peak_candidates
from .wire_graph import polyline


def supported_extension(start, tip, probability, cfg):
    height, width = probability.shape;pad = cfg['path_padding_px'];radius = cfg['new_tip_wire_radius_px']
    x1 = max(0, math.floor(min(start[0], tip[0])-pad));x2 = min(width-1, math.ceil(max(start[0], tip[0])+pad))
    y1 = max(0, math.floor(min(start[1], tip[1])-pad));y2 = min(height-1, math.ceil(max(start[1], tip[1])+pad))
    seed = (round(start[0]), round(start[1]))
    if not (0 <= seed[0] < width and 0 <= seed[1] < height) or probability[seed[1], seed[0]] < cfg['wire_threshold']:return None
    parents = {seed: None};pending = deque([seed])
    while pending:
        x, y = pending.popleft()
        for dx, dy in [(0, -1), (-1, 0), (1, 0), (0, 1), (-1, -1), (1, -1), (-1, 1), (1, 1)]:
            q = (x+dx, y+dy)
            if (q in parents or not x1 <= q[0] <= x2 or not y1 <= q[1] <= y2
                    or probability[q[1], q[0]] < cfg['wire_threshold']):continue
            parents[q] = (x, y);pending.append(q)
    eligible = [q for q in parents if math.dist(q, tip) <= radius]
    if not eligible:return None
    end = min(eligible, key=lambda q: (math.dist(q, tip), math.dist(q, seed), q))
    path = [];q = end
    while q is not None:path.append(list(q));q = parents[q]
    path.reverse()
    if len(path) < 2 or sum(math.dist(a, b) for a, b in zip(path, path[1:])) > 2*cfg['max_shift_px']:return None
    return path


def refine_capacitor_geometry(pins, graph, records, native, zoom15, zoom20, cfg, engineering_cfg):
    validate_fields(pins, graph, records)
    if (cfg['policy'] != 'opposed_capacitor_triplet_scale_peaks_and_supported_endpoint_extension'
            or cfg['scales'] != [1.0, 1.5, 2.0] or cfg['component_type'] != 'capacitor'):
        raise ValueError('Unsupported capacitor geometry policy')
    width, height = pins['size']
    for array in [native, zoom15, zoom20]:
        if array.shape != (2, height, width) or not np.isfinite(array).all() or array.min() < 0 or array.max() > 1:
            raise ValueError('Invalid model probability map')
    updated, new_graph = deepcopy(pins), deepcopy(graph)
    pmap = {p['candidate_id']: p for p in updated['pin_candidates']}
    attachments = {a['candidate_id']: a for a in new_graph['pin_attachments']}
    nodes = {n['node_id']: n for n in new_graph['nodes']};edges = {e['edge_id']: e for e in new_graph['edges']}
    degree = Counter(n for e in new_graph['edges'] for n in [e['source'], e['target']])
    owned = defaultdict(list)
    for p in pins['pin_candidates']:
        if p['owner_status'] == 'assigned':owned[p['owner_detection_index']].append(p)
    peaks = [peak_candidates(a[1], threshold, cfg['peak_kernel'], cfg['peak_refinement'])
             for a, threshold in [(zoom15, cfg['min_zoom15_confidence']), (zoom20, cfg['min_zoom20_confidence'])]]
    wire = np.maximum(native[0], zoom20[0]);changes, proposals = [], []
    for record in records:
        if record['type'] != cfg['component_type'] or not record.get('designator'):continue
        members = owned[record['detection_index']];reasons = [];pair = []
        if len(members) != 2:reasons.append('requires_exactly_two_owned_model_tips')
        if record.get('detection_confidence', 0) < cfg['min_detection_confidence']:reasons.append('weak_component_detection')
        for pin in members:
            axis, wire_proof = wire_axis(pin, graph, engineering_cfg)
            if axis is None:reasons.append('original_wire_axis_unqualified');continue
            if pin.get('number') is not None or pin.get('pinname') is not None:reasons.append('recognized_fields_protected');continue
            if pin['confidence'] < cfg['min_native_confidence']:reasons.append('weak_native_tip');continue
            box = record['bbox_top_left'];center = (box[axis]+box[axis+2])/2
            sign = 1 if pin['point'][axis] < center else -1
            matches = []
            for candidates in peaks:
                near = [p for p in candidates if math.dist(p['point'], pin['point']) <= cfg['max_shift_px']
                        and abs(p['point'][1-axis]-pin['point'][1-axis]) <= cfg['max_tangential_shift_px']
                        and sign*(p['point'][axis]-pin['point'][axis]) >= cfg['min_each_scale_inward_shift_px']]
                if len(near) != 1:reasons.append('zoom_peak_missing_or_competing');break
                matches.append(near[0])
            if len(matches) != 2:continue
            if sign*(matches[1]['point'][axis]-matches[0]['point'][axis]) < cfg['min_each_scale_inward_shift_px']:
                reasons.append('scale_motion_not_monotonic_inward');continue
            q = matches[1]['point']
            if sign*(center-q[axis]) <= 0:reasons.append('zoom_tip_crosses_body_center');continue
            attachment = attachments[pin['candidate_id']]
            net = next(n for n in graph['physical_networks'] if n['network_id'] == attachment['physical_network_id'])
            endpoints = [n for n in nodes.values() if degree[n['node_id']] == 1
                         and any(n['node_id'] in [edges[e]['source'], edges[e]['target']] for e in net['edge_ids'])
                         and math.dist(n['point'], pin['point']) <= cfg['endpoint_search_radius_px']]
            if len(endpoints) != 1:reasons.append('original_endpoint_missing_or_competing');continue
            node = endpoints[0];edge = next(edges[e] for e in net['edge_ids'] if node['node_id'] in [edges[e]['source'], edges[e]['target']])
            path = supported_extension(node['point'], q, wire, cfg)
            if path is None:reasons.append('learned_wire_does_not_support_extension');continue
            if sign*(path[-1][axis]-node['point'][axis]) <= 0:reasons.append('wire_extension_not_toward_body');continue
            if any(p['candidate_id'] != pin['candidate_id'] and math.dist(p['point'], node['point']) <= cfg['endpoint_search_radius_px'] for p in pins['pin_candidates']):
                reasons.append('another_tip_uses_endpoint');continue
            if any(point_segment_distance(pt, a, b) < cfg['other_edge_clearance_px']
                   for other in graph['edges'] if other['edge_id'] != edge['edge_id']
                   for a, b in zip(other['polyline'], other['polyline'][1:]) for pt in path[1:]):
                reasons.append('extension_near_other_wire');continue
            if any(r['detection_index'] != record['detection_index']
                   and any(r['bbox_top_left'][0] <= pt[0] <= r['bbox_top_left'][2] and r['bbox_top_left'][1] <= pt[1] <= r['bbox_top_left'][3] for pt in path[1:]) for r in records):
                reasons.append('extension_enters_other_component');continue
            pair.append({'candidate_id': pin['candidate_id'], 'axis': axis, 'old_point': pin['point'],
                'new_point': q, 'native_confidence': pin['confidence'], 'zoom15_peak': matches[0], 'zoom20_peak': matches[1],
                'node_id': node['node_id'], 'edge_id': edge['edge_id'], 'old_node_point': node['point'],
                'physical_network_id': attachment['physical_network_id'],
                'supported_wire_path': path, 'min_path_probability': float(min(wire[y, x] for x, y in path)),
                'confidence': float(np.mean([pin['confidence'], matches[0]['confidence'], matches[1]['confidence']]))})
        if len(pair) == 2:
            axis = pair[0]['axis'];box = record['bbox_top_left']
            if pair[1]['axis'] != axis or abs(pair[0]['new_point'][axis]-pair[1]['new_point'][axis]) < cfg['min_pair_span_body_fraction']*(box[axis+2]-box[axis]):
                reasons.append('refined_pair_not_opposed')
            center = (box[axis]+box[axis+2])/2
            if (pair[0]['new_point'][axis]-center)*(pair[1]['new_point'][axis]-center) >= 0:reasons.append('both_refined_tips_on_same_body_side')
            if pair[0]['node_id'] == pair[1]['node_id'] or pair[0]['edge_id'] == pair[1]['edge_id']:reasons.append('both_tips_share_endpoint_or_wire')
            if pair[0]['physical_network_id'] == pair[1]['physical_network_id']:reasons.append('both_leads_on_same_physical_network')
        else:reasons.append('both_leads_must_qualify')
        proposal = {'component_id': record['component_id'], 'detection_index': record['detection_index'],
                    'tip_proposals': pair, 'withheld_reasons': sorted(set(reasons))};proposals.append(proposal)
        if reasons:continue
        for proof in pair:
            pin = pmap[proof['candidate_id']];pin.update(point=proof['new_point'], confidence=proof['confidence'],
                native_model_point=proof['old_point'], native_model_confidence=proof['native_confidence'],
                geometry_source='capacitor_triplet_scale_U-Net_peak_v1', geometry_evidence=deepcopy(proof),
                confidence_aggregation='mean_of_correlated_model_scale_scores_not_calibrated')
            att = attachments[proof['candidate_id']];att.update(point=pin['point'], wire_projection=proof['supported_wire_path'][-1],
                wire_gap_px=math.dist(pin['point'], proof['supported_wire_path'][-1]))
            for alt in att.get('alternatives', []):alt.update(wire_projection=att['wire_projection'], distance_px=att['wire_gap_px'])
            edge = edges[proof['edge_id']];path = proof['supported_wire_path']
            extension = polyline(path, cfg['path_simplification_tolerance_px'])
            if math.dist(edge['polyline'][0], proof['old_node_point']) < math.dist(edge['polyline'][-1], proof['old_node_point']):
                edge['polyline'] = list(reversed(extension[1:]))+edge['polyline']
            else:edge['polyline'] += extension[1:]
            added_length = sum(math.dist(a, b) for a, b in zip(path, path[1:]))
            old_length = edge['length_px'];extension_mean = float(np.mean([wire[y, x] for x, y in path[1:]]))
            edge['mean_probability'] = (edge['mean_probability']*old_length+extension_mean*added_length)/(old_length+added_length)
            edge['length_px'] += added_length;edge['local_model_extension'] = deepcopy(proof)
            nodes[proof['node_id']]['point'] = path[-1];changes.append({**deepcopy(proof), 'component_id': record['component_id']})
    validate_fields(updated, new_graph, records)
    if changes:
        new_graph.update(original_physical_edge_geometry_preserved=False,
            geometry_derivation='native_model_graph_with_local_native_zoom20_supported_wire_extensions')
    if ([n['pin_candidate_ids'] for n in graph['networks']] != [n['pin_candidate_ids'] for n in new_graph['networks']]
            or graph['physical_networks'] != new_graph['physical_networks']):raise ValueError('Geometry stage changed membership')
    return updated, new_graph, {'changes': changes, 'component_proposals': proposals,
        'summary': {'refined_tips': len(changes), 'refined_capacitors': sum(not p['withheld_reasons'] for p in proposals),
                    'changed_wire_edges': len({p['edge_id'] for p in changes}),
                    'withheld_reason_counts': dict(Counter(r for p in proposals for r in p['withheld_reasons']))},
        'model_tip_count_and_raw_membership_preserved': True, 'new_numbers_or_names': 0,
        'points_snapped_to_reference_or_bbox': 0, 'raw_image_segmentation_used': False}
