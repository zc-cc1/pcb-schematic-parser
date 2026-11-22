"""Join exported network fragments only at observed same-image conductors.

This stage uses existing predicted geometry and red junctions from the current
image. It does not read reference annotations, page identifiers or designators.
"""
from copy import deepcopy
import math

import numpy as np


DEFAULT = {'enabled': False, 'minimum_overlap_px': 60., 'maximum_dot_gap_px': 30.,
           'minimum_ink_support': .95, 'maximum_groups': 8}


def _edge_points(edge):
    return [np.asarray([point['x'], point['y']], float) for point in edge]


def _collinear_overlap(first, second, minimum):
    a, b = _edge_points(first)
    c, d = _edge_points(second)
    v, w = b-a, d-c
    lv, lw = float(np.linalg.norm(v)), float(np.linalg.norm(w))
    if min(lv, lw) < minimum:
        return 0.
    u, z = v/lv, w/lw
    if abs(float(u@z)) < math.cos(math.radians(3)):
        return 0.
    if max(abs(u[0]*(c-a)[1]-u[1]*(c-a)[0]),
           abs(u[0]*(d-a)[1]-u[1]*(d-a)[0])) > 2:
        return 0.
    lo = max(0., min(float((c-a)@u), float((d-a)@u)))
    hi = min(lv, max(float((c-a)@u), float((d-a)@u)))
    return max(0., hi-lo)


def _endpoint_at_dot(net, dot, ink, height, width, maximum_gap, minimum_support):
    for edge in net['edges'].values():
        points = _edge_points(edge)
        for q, r in (points, points[::-1]):
            v = q-dot
            distance = float(np.linalg.norm(v))
            if distance > maximum_gap:
                continue
            direction = r-q
            length = float(np.linalg.norm(direction))
            if distance > 3:
                if distance < 4 or length < 5:
                    continue
                if float(v@direction)/(distance*length) < .95:
                    continue
            samples = np.rint(np.linspace(q, dot, max(2, math.ceil(distance)+1))).astype(int)
            supported = 0
            for x, y in samples:
                image_y = height-y
                if (2 <= x < width-2 and 2 <= image_y < height-2
                        and ink[image_y-2:image_y+3, x-2:x+3].any()):
                    supported += 1
            if supported/len(samples) >= minimum_support:
                return True
    return False


def _signature(edge):
    return frozenset((round(float(point['x']), 1), round(float(point['y']), 1)) for point in edge)


def _near_duplicate(first, second):
    a, b = _edge_points(first)
    c, d = _edge_points(second)
    return min(max(float(np.linalg.norm(a-c)), float(np.linalg.norm(b-d))),
               max(float(np.linalg.norm(a-d)), float(np.linalg.norm(b-c)))) <= 2


def reconcile_network_junctions(target, rgb, graph, image_sha256, observed_junctions, *, cfg=None):
    config = {**DEFAULT, **(cfg or {})}
    if set(config) != set(DEFAULT):
        raise ValueError('unknown_network_reconciliation_setting')
    if type(config['enabled']) is not bool:
        raise ValueError('network_reconciliation_enabled_must_be_boolean')
    if type(config['maximum_groups']) is not int or not 0 <= config['maximum_groups'] <= 64:
        raise ValueError('invalid_network_reconciliation_budget')
    for key in ('minimum_overlap_px', 'maximum_dot_gap_px', 'minimum_ink_support'):
        if type(config[key]) not in (int, float) or not math.isfinite(config[key]):
            raise ValueError('invalid_network_reconciliation_threshold')
    if not (20 <= config['minimum_overlap_px'] <= 500
            and 3 <= config['maximum_dot_gap_px'] <= 50
            and 0 < config['minimum_ink_support'] <= 1):
        raise ValueError('network_reconciliation_threshold_out_of_range')
    if not config['enabled']:
        return deepcopy(target), {'enabled': False, 'status': 'disabled', 'groups': [], 'proposals': []}
    height, width = rgb.shape[:2]
    if (graph.get('image_sha256') != image_sha256 or graph.get('size') != [width, height]
            or graph.get('coordinate_origin') != 'top_left' or graph.get('prediction_only') is not True):
        raise ValueError('network_reconciliation_evidence_differs_from_image')
    rgb = np.asarray(rgb)
    red, green, blue = [rgb[:, :, i].astype(np.int16) for i in range(3)]
    ink = ((red > green+35) & (red > blue+35) & (green < 150) & (blue < 150)) | (
        (green > red+20) & (green > blue+20) & (green > 60))
    out = deepcopy(target)
    ids = [nid for nid, net in out['nets'].items() if net.get('edges')]
    parent = {nid: nid for nid in ids}

    def find(nid):
        while nid != parent[nid]:
            parent[nid] = parent[parent[nid]]
            nid = parent[nid]
        return nid

    def union(first, second):
        first, second = find(first), find(second)
        if first != second:
            parent[max(first, second)] = min(first, second)

    proposals = []
    for index, first in enumerate(ids):
        for second in ids[index+1:]:
            length = max((_collinear_overlap(e, f, config['minimum_overlap_px'])
                          for e in out['nets'][first]['edges'].values()
                          for f in out['nets'][second]['edges'].values()), default=0.)
            if length >= config['minimum_overlap_px']:
                proposals.append({'kind': 'long_collinear_overlap', 'nets': [first, second],
                                  'overlap_px': round(length, 2)})
                union(first, second)
    for point in observed_junctions:
        if not (isinstance(point, (list, tuple)) and len(point) == 2
                and all(type(value) in (int, float) and math.isfinite(value) for value in point)):
            raise ValueError('invalid_observed_junction')
        dot = np.asarray(point, float)
        touching = [nid for nid in ids if _endpoint_at_dot(out['nets'][nid], dot, ink,
            height, width, config['maximum_dot_gap_px'], config['minimum_ink_support'])]
        if len(touching) < 2:
            continue
        proposals.append({'kind': 'observed_red_junction', 'nets': touching,
                          'point': [round(float(value), 2) for value in dot]})
        for nid in touching[1:]:
            union(touching[0], nid)
    groups = {}
    for nid in ids:
        groups.setdefault(find(nid), []).append(nid)
    groups = [group for group in groups.values() if len(group) > 1]
    if len(groups) > config['maximum_groups']:
        return deepcopy(target), {'enabled': True, 'status': 'withheld_budget', 'groups': [],
                                  'proposals': proposals, 'proposed_groups': len(groups)}
    committed = []
    for group in groups:
        if any(set(out['nets'][nid]) != {'hyperGraph', 'edges'} for nid in group):
            continue
        destination = max(group, key=lambda nid: (
            len(out['nets'][nid]['hyperGraph'].split(',')), len(out['nets'][nid]['edges'])))
        members, edges = set(), []
        for nid in group:
            net = out['nets'][nid]
            members.update(net['hyperGraph'].strip('()').split(','))
            edges.extend(net['edges'].values())
        for nid in group:
            if nid != destination:
                del out['nets'][nid]
        unique, signatures = [], set()
        for edge in edges:
            key = _signature(edge)
            if key in signatures or any(_near_duplicate(edge, prior) for prior in unique):
                continue
            signatures.add(key)
            unique.append(edge)
        out['nets'][destination]['hyperGraph'] = '('+','.join(sorted(members))+')'
        out['nets'][destination]['edges'] = {'edge_'+str(i+1): edge for i, edge in enumerate(unique)}
        committed.append({'retained_net': destination, 'merged_nets': group,
                          'members': len(members), 'edges': len(unique)})
    if out['components'] != target['components'] or out['pins'] != target['pins']:
        raise ValueError('network_reconciliation_modified_components_or_pins')
    return out, {'enabled': True, 'status': 'applied' if committed else 'no_effect',
                 'groups': committed, 'proposals': proposals, 'reference_read': False}
