"""Image evidence shared by independent connectivity repair rules."""
from collections import defaultdict
from copy import deepcopy
import math
import numpy as np

NC_NAMES = {'NC', 'N.C.', 'N/C', 'DNC'}
POWER = {'gnd', 'v', 'vcc', 'vdd'}

def _members(net):
    return {r.strip() for r in str(net.get('hyperGraph') or '').strip('()').split(',') if r.strip()}

def _project(point, a, b):
    dx, dy = b[0]-a[0], b[1]-a[1]
    norm = dx*dx + dy*dy
    t = max(0., min(1., ((point[0]-a[0])*dx+(point[1]-a[1])*dy)/norm)) if norm else 0.
    q = [a[0]+t*dx, a[1]+t*dy]
    return math.dist(point, q), q

def _vertex(ref, components):
    owner, number = ref.rsplit('.', 1)
    comp = components[owner]
    kind = str(comp['type']).lower()
    if kind in {'ic', 'box'}:
        return ('ic', owner, number)
    if kind in POWER:
        return ('rail', 'ground' if kind == 'gnd' else 'supply',
                'GND' if kind == 'gnd' else str(comp.get('value') or 'VCC'))
    return ('ordinary', owner)

def collect_evidence(target, graph, pin_document, rgb, config, audit, cleanup):
    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError('Expected an original-size uint8 RGB image')
    height, width = rgb.shape[:2]
    digest = graph.get('image_sha256')
    if (not digest or digest != pin_document.get('image_sha256')
            or graph.get('size') != [width, height] or pin_document.get('size') != [width, height]
            or graph.get('coordinate_origin') != 'top_left'
            or pin_document.get('coordinate_origin') != 'top_left'
            or not graph.get('prediction_only') or not pin_document.get('prediction_only')):
        raise ValueError('Connectivity artifacts must describe the same current-image predictions')
    for key in ['maximum_distance_px', 'ambiguity_margin_px', 'crossing_guard_px']:
        if not isinstance(config[key], (int, float)) or not math.isfinite(config[key]) or config[key] < 0:
            raise ValueError('Invalid nonnegative distance configuration: '+key)
    colors = rgb.astype(np.int16)
    red, green, blue = [colors[:, :, i] for i in range(3)]
    red_mask = (red > green+35) & (red > blue+35) & (green < 150) & (blue < 150)
    green_mask = (green > red+20) & (green > blue+20) & (green > 60)
    audit['color_evidence'] = {'green_pixels': int(green_mask.sum()), 'red_pixels': int(red_mask.sum())}
    if green_mask.sum() < config['minimum_colored_wire_pixels'] or red_mask.sum() < config['minimum_red_pixels']:
        audit['status'] = 'unsupported_color_profile'
        return {'ready': False, 'audit': audit}
    conductive = green_mask | red_mask
    edges = {e['edge_id']: e for e in graph.get('edges', [])}
    physical = {n['network_id']: n for n in graph.get('physical_networks', [])}
    if not physical:
        audit['status'] = 'no_physical_model_networks'
        return {'ready': False, 'audit': audit}
    segments = {}
    for nid, net in physical.items():
        segments[nid] = []
        for eid in net.get('edge_ids', []):
            if eid not in edges:
                raise ValueError('Physical model network references an absent edge')
            points = edges[eid].get('polyline', [])
            if any(len(p) != 2 or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in p)
                   or not 0 <= p[0] < width or not 0 <= p[1] < height for p in points):
                raise ValueError('Model wire point outside the current image')
            segments[nid].extend((a, b, eid) for a, b in zip(points, points[1:]))
    logical_for = defaultdict(list)
    for n in graph.get('networks', []):
        for pid in n.get('physical_network_ids', []):
            logical_for[pid].append(n['network_id'])
    existing = set().union(*(_members(n) for n in target['nets'].values())) if target['nets'] else set()
    def destination(pid):
        restored = 'RESTORED_'+pid
        if restored in target['nets']:
            return restored
        candidates = sorted(nid for nid in logical_for[pid] if nid in target['nets'])
        if len(candidates) == 1:
            return candidates[0]
        return 'RECOVERED_'+pid
    contacts = []
    for owner, owner_pins in sorted(target['pins'].items()):
        if owner not in target['components']:
            raise ValueError('Known pin owner absent from current components')
        for key, pin in sorted(owner_pins.items()):
            ref = owner+'.'+key.removeprefix('pin_')
            if ref in existing or str(pin.get('pinname', '')).strip().upper() in NC_NAMES:
                continue
            p = pin['point']
            point = [p['x'], height-p['y']]
            if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in point) or not 0 <= point[0] < width or not 0 <= point[1] < height:
                raise ValueError('Known pin outside current image')
            contacts.append({'member': ref, 'point': point, 'new_pin': False})
    by_owner = defaultdict(list)
    for p in pin_document.get('pin_candidates', []):
        if p.get('owner_status') == 'assigned':
            by_owner[p.get('owner')].append(p)
    for owner, comp in sorted(target['components'].items()):
        if comp.get('type') != 'gnd' or comp.get('value') != 'GND':
            continue
        owned = by_owner[owner]
        primary = [p for p in owned if not p.get('text_hypothesis_primary_candidate_id')]
        if len(primary) != 1:
            continue
        p = primary[0]
        if (not p.get('number') or p.get('confidence', 0) < config['minimum_power_confidence']
                or any(q is not p and q.get('text_hypothesis_primary_candidate_id') != p['candidate_id'] for q in owned)):
            continue
        ref, key = owner+'.'+str(p['number']), 'pin_'+str(p['number'])
        if key in target['pins'].get(owner, {}) or ref in existing:
            continue
        point = p['point']
        if len(point) != 2 or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in point) or not 0 <= point[0] < width or not 0 <= point[1] < height:
            raise ValueError('Power source candidate outside current image')
        contacts.append({'member': ref, 'point': list(point), 'new_pin': True,
                         'source_candidate_id': p['candidate_id'],
                         'pin': {'pinname': '', 'point': {'x': float(point[0]), 'y': float(height-point[1])}}})
    junctions = graph.get('junction_candidates', [])
    grouped = defaultdict(list)
    for contact in contacts:
        choices = []
        for pid, lines in segments.items():
            if not lines:
                continue
            distance, projection, eid = min((*_project(contact['point'], a, b), eid) for a, b, eid in lines)
            choices.append({'physical_id': pid, 'distance_px': distance, 'projection': projection,
                            'edge_id': eid, 'destination': destination(pid)})
        choices.sort(key=lambda v: (v['distance_px'], v['physical_id']))
        proposal = {k: deepcopy(v) for k, v in contact.items() if k != 'pin'}
        proposal['alternatives'] = choices[:3]
        reason = None
        if not choices or choices[0]['distance_px'] > config['maximum_distance_px']:
            reason = 'no_model_wire_within_radius'
        elif len(choices) > 1 and choices[1]['distance_px']-choices[0]['distance_px'] < config['ambiguity_margin_px']:
            reason = 'ambiguous_physical_networks'
        elif any(math.dist(contact['point'], j['point']) < config['crossing_guard_px'] for j in junctions if not j.get('resolved')):
            reason = 'contact_near_unresolved_crossing'
        else:
            a, b = contact['point'], choices[0]['projection']
            points = np.rint(np.linspace(a, b, max(2, math.ceil(math.dist(a, b))+1))).astype(int)
            if not all(conductive[max(0,y-1):min(height,y+2),max(0,x-1):min(width,x+2)].any() for x,y in points):
                reason = 'no_continuous_image_support'
        proposal['status'] = reason or 'eligible'
        audit['proposals'].append(proposal)
        if reason is None:
            grouped[choices[0]['physical_id']].append(contact)
    return {'ready': True, 'audit': audit, 'grouped': dict(grouped),
            'segments': segments, 'physical': physical, 'green_mask': green_mask,
            'destinations': {pid: destination(pid) for pid in grouped},
            'config': config, 'cleanup': cleanup}
