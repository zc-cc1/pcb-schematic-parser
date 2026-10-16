"""Resolve physical contacts jointly per component after learned wire attachment.

Only predicted geometry, observed OCR and a packaged symbol library enter here.
No label reads, point motion, synthetic contacts or library-array pin ordering.
"""
from collections import Counter, defaultdict
from copy import deepcopy
import math
import re

import numpy as np
from scipy.optimize import linear_sum_assignment

from .component_text import center, iou
from .pin_text_v2 import tokens_with_views, glyph_groups
from .engineering_network_export import rebuild_references
from .submission_fields import validate_fields


INFERRED = {'engineering_convention_unverified'}


def _side(pin, box):
    x, y = pin['point']
    return min([('left', abs(x-box[0])), ('right', abs(x-box[2])),
                ('top', abs(y-box[1])), ('bottom', abs(y-box[3]))], key=lambda p: p[1])[0]


def _local(point, box, side):
    x, y = point
    if side == 'left':
        return y, x-box[0]
    if side == 'right':
        return y, box[2]-x
    if side == 'top':
        return x, y-box[1]
    return x, box[3]-y


def _physical(attachment):
    return attachment.get('physical_network_id') or attachment.get('network_id')


def physical_contacts(pins, graph, records, cfg):
    """Collapse collinear same-wire hypotheses, not adjacent true pin rows."""
    rmap = {r['detection_index']: r for r in records}
    amap = {a['candidate_id']: a for a in graph['pin_attachments']}
    grouped = defaultdict(list)
    for pin in pins['pin_candidates']:
        if pin['owner_status'] == 'assigned':
            grouped[pin['owner_detection_index']].append(pin)
    contacts, aliases = [], {}
    for index, candidates in sorted(grouped.items()):
        box = rmap[index]['bbox_top_left']
        clusters = []
        for pin in sorted(candidates, key=lambda p: p['candidate_id']):
            a = amap[pin['candidate_id']]
            side = _side(pin, box)
            tangent, normal = _local(pin['point'], box, side)
            for cluster in clusters:
                compatible = True
                for other in cluster:
                    b = amap[other['candidate_id']]
                    ot, on = _local(other['point'], box, _side(other, box))
                    # Every alias needs the same observed physical wire and row.
                    if (not _physical(a) or _physical(a) != _physical(b) or side != _side(other, box)
                            or abs(tangent-ot) > cfg['contact_row_tolerance_px']
                            or abs(normal-on) > cfg['contact_normal_tolerance_px']):
                        compatible = False
                    for field in ['number', 'pinname']:
                        if (pin.get(field) is not None and other.get(field) is not None
                                and pin[field] != other[field]):
                            compatible = False
                if compatible:
                    cluster.append(pin)
                    break
            else:
                clusters.append([pin])
        for cluster in clusters:
            representative = min(cluster, key=lambda p: (
                -sum(p.get(f) is not None for f in ['number', 'pinname']),
                abs(_local(p['point'], box, _side(p, box))[1]), -p.get('confidence', 0), p['candidate_id']))
            cid = representative['candidate_id']
            for member in cluster:
                aliases[member['candidate_id']] = cid
            contacts.append({'candidate_id': cid, 'member_ids': [p['candidate_id'] for p in cluster],
                             'index': index, 'side': _side(representative, box),
                             'physical_network_id': _physical(amap[cid]),
                             'wire_attached': amap[cid].get('attachment_status') == 'attached'})
    return contacts, aliases


def _assign(costs, unmatched, margin):
    """One glyph instance per contact/field, with global alternative checks."""
    n, m = costs.shape
    if not n or not m:
        return {}
    matrix = np.concatenate([costs, np.full((n, n), unmatched)], axis=1)
    rows, cols = linear_sum_assignment(matrix)
    base = float(matrix[rows, cols].sum())
    result = {}
    for i, j in zip(rows, cols):
        if j >= m or matrix[i, j] >= unmatched:
            continue
        alternative = matrix.copy()
        alternative[i, j] = 1e6
        ar, ac = linear_sum_assignment(alternative)
        stability = float(alternative[ar, ac].sum()-base)
        if stability >= margin:
            result[int(i)] = (int(j), float(matrix[i, j]), stability)
    return result


def _model_part(record, texts, library):
    parts = library.get('parts', {})
    if record.get('Name') in parts:
        return record['Name'], parts[record['Name']]
    if record.get('Name'):
        return None, None
    box = record['bbox_top_left']
    observed = {t['text'] for t in texts if t['text'] in parts and len(t['text']) >= 5
                and re.search(r'[A-Za-z]', t['text']) and re.search(r'\d', t['text'])
                and t.get('confidence', 0) >= .9
                and box[0]-20 <= center(t['bbox'])[0] <= box[2]+20
                and box[1]-30 <= center(t['bbox'])[1] <= box[3]+30}
    if len(observed) == 1:
        model = next(iter(observed))
        return model, parts[model]
    return None, None


def resolve_component_pins(pins, graph, records, texts, library, cfg):
    validate_fields(pins, graph, records)
    data = deepcopy(pins)
    pmap = {p['candidate_id']: p for p in data['pin_candidates']}
    rmap = {r['detection_index']: r for r in records}
    changes, decisions = [], []
    # Old top=1 numbering conflicts with the public bottom-left convention.
    for pin in data['pin_candidates']:
        for field in ['number', 'pinname']:
            if pin.get(field+'_status') in INFERRED:
                pin.setdefault('joint_previous_fields', {})[field] = pin.get(field)
                pin[field] = None
                pin[field+'_status'] = 'joint_revalidation_pending'
    contacts, aliases = physical_contacts(data, graph, records, cfg)
    by_component, by_side = defaultdict(list), defaultdict(list)
    for contact in contacts:
        by_component[contact['index']].append(contact)
        by_side[contact['index'], contact['side']].append(contact)
    parts = {i: _model_part(r, texts, library) for i, r in rmap.items()}
    blocked = [r['fields'][f]['bbox'] for r in records for f in ['designator', 'Name', 'value']
               if f in r.get('fields', {}) and r['fields'][f].get('bbox')]
    tokens = [t for t in tokens_with_views(texts) if t['confidence'] >= cfg['minimum_text_confidence']
              and t['text'].strip() and len(t['text']) <= 100
              and not any(iou(t['bbox'], b) >= .45 for b in blocked)]
    groups = glyph_groups(tokens)
    occupied = set()
    for pin in data['pin_candidates']:
        for field in ['number', 'pinname']:
            evidence = pin.get('text_evidence', {}).get(field, {}) if pin.get(field) is not None else {}
            if evidence.get('bbox'):
                occupied.update(j for j, group in enumerate(groups)
                                if any(iou(t['bbox'], evidence['bbox']) >= .45 for t in group))

    def fill(pin, field, value, source, evidence):
        if pin.get(field) is not None:
            return False
        pin[field] = str(value)
        pin[field+'_status'] = source
        pin.setdefault('text_evidence', {})[field] = {'source': source, **evidence}
        changes.append({'candidate_id': pin['candidate_id'], 'owner': pin['owner'], 'field': field,
                        'value': str(value), 'source': source,
                        'previous_inferred_value': pin.get('joint_previous_fields', {}).get(field)})
        return True

    # Pair contacts using wire support, so repeated detections do not veto a pair.
    for index, cs in by_component.items():
        record = rmap[index]
        if record['type'] not in cfg['two_terminal_types'] or len(cs) != 2:
            continue
        a, b = [pmap[c['candidate_id']] for c in cs]
        dx, dy = abs(a['point'][0]-b['point'][0]), abs(a['point'][1]-b['point'][1])
        axis = 0 if dx > dy else 1
        box = record['bbox_top_left']
        span = max(dx, dy)
        aligned = min(dx, dy) <= max(4, cfg['two_terminal_max_alignment_fraction']*(box[3-axis]-box[1-axis]))
        opposite = min(a['point'][axis], b['point'][axis]) < (box[axis]+box[axis+2])/2 < max(a['point'][axis], b['point'][axis])
        connected = all(c['wire_attached'] for c in cs) and len({c['physical_network_id'] for c in cs}) == 2
        if not (aligned and opposite and connected and span >= cfg['two_terminal_min_span_fraction']*(box[axis+2]-box[axis])):
            continue
        # Original PNG uses top-left coordinates; bottom has the larger y.
        ordered = sorted([a, b], key=lambda p: p['point'][axis], reverse=axis == 1)
        if any(p.get('number') not in {None, str(i+1)} for i, p in enumerate(ordered)):
            continue
        for number, pin in enumerate(ordered, 1):
            evidence = {'convention': cfg['two_terminal_order'], 'observed_OCR_glyph': False,
                        'contact_ids': [c['candidate_id'] for c in cs], 'wire_supported': True}
            fill(pin, 'number', number, 'public_two_terminal_contact_convention', evidence)
            fill(pin, 'pinname', number, 'public_two_terminal_contact_convention', evidence)

    # Each side learns its column offset from the complete component's anchors.
    geometry = {}
    for (index, side), cs in by_side.items():
        box = rmap[index]['bbox_top_left']
        rows = sorted(_local(pmap[c['candidate_id']]['point'], box, side)[0] for c in cs)
        spacings = [b-a for a, b in zip(rows, rows[1:]) if b-a > cfg['contact_row_tolerance_px']]
        pitch = float(np.median(spacings)) if spacings else 20.
        offsets = []
        for c in cs:
            p = pmap[c['candidate_id']]
            ev = p.get('text_evidence', {}).get('number', {})
            if p.get('number') is not None and ev.get('bbox') and p.get('number_status') not in INFERRED:
                offsets.append(_local(center(ev['bbox']), box, side)[0]-_local(p['point'], box, side)[0])
        offset = float(np.median(offsets)) if len(offsets) >= cfg['number_offset_min_anchors'] else None
        anchors = [(_local(pmap[c['candidate_id']]['point'], box, side)[0],
                    int(pmap[c['candidate_id']]['number'])) for c in cs
                   if str(pmap[c['candidate_id']].get('number', '')).isdigit()
                   and pmap[c['candidate_id']].get('number_status') != 'public_two_terminal_contact_convention']
        pattern = None
        if len(anchors) >= 2:
            fitted = []
            for ai, (ta, na) in enumerate(anchors):
                for tb, nb in anchors[ai+1:]:
                    if abs(tb-ta) < 1:
                        continue
                    slope = (nb-na)/(tb-ta)
                    if not .6 <= abs(slope*pitch) <= 2.4:
                        continue
                    intercept = na-slope*ta
                    errors = [abs(slope*t+intercept-n) for t, n in anchors]
                    fitted.append((sum(e < .3 for e in errors), sum(min(e, 2) for e in errors), slope, intercept))
            if fitted:
                best = min(fitted, key=lambda x: (-x[0], x[1]))
                if best[0] >= max(2, math.ceil(len(anchors)*.8)):
                    pattern = best[2:]
        geometry[index, side] = (max(cfg['minimum_row_scale_px'], pitch*.42), offset, pattern)

    for field in ['pinname', 'number']:
        targets = [c for c in contacts if pmap[c['candidate_id']].get(field) is None
                   and rmap[c['index']]['type'] not in cfg['two_terminal_types']]
        costs = np.full((len(targets), len(groups)), 1e6)
        options = {}
        for i, c in enumerate(targets):
            pin, record = pmap[c['candidate_id']], rmap[c['index']]
            box, side = record['bbox_top_left'], c['side']
            tangent, _ = _local(pin['point'], box, side)
            row_scale, offset, pattern = geometry[c['index'], side]
            model, part = parts[c['index']]
            width = box[2]-box[0] if side in {'left', 'right'} else box[3]-box[1]
            for j, group in enumerate(groups):
                if j in occupied:
                    continue
                ranked = []
                votes = defaultdict(set)
                for token in group:
                    if token['confidence'] >= .9:
                        votes[token['text']].add(token.get('source_view', 'unknown'))
                for token in group:
                    text = token['text']
                    t, inward = _local(center(token['bbox']), box, side)
                    b = token['bbox']
                    height = b[3]-b[1] if side in {'left', 'right'} else b[2]-b[0]
                    expected_offset = offset if field == 'number' and offset is not None else 0.
                    residual = abs(t-tangent-expected_offset)
                    row_cost = max(0., residual-(height*.45 if expected_offset == 0 else 0.))/row_scale
                    if field == 'number':
                        lexical = 0. if re.fullmatch(r'\d+|[A-Za-z]{1,3}\d+(?:_\d+)*', text) else 2.6
                        if part and text in part['number_to_names']:
                            lexical = 0.
                        normal_cost = abs(inward+4)/max(18., width*.12)
                        # Internal signal labels remain possible, but score as names.
                        normal_cost += max(0., inward-8)/max(10., width*.12)
                        if pattern and text.isdecimal():
                            lexical += min(4., abs(pattern[0]*tangent+pattern[1]-int(text))*.8)
                    else:
                        lexical = .9 if text.isdecimal() else 0.
                        normal_cost = abs(inward-min(width*.22, 45.))/max(25., width*.3)
                        normal_cost += max(0., -inward)/max(15., width*.15)
                    confidence_cost = 2*(1-token['confidence'])-.22*min(3, len(votes[text]))
                    # Prefer an observed full spelling over a cropped suffix.
                    if any(text != full and text in full and len(votes[full]) >= max(2, len(votes[text]))
                           for full in votes):
                        confidence_cost += 1.2
                    cost = row_cost+normal_cost+lexical+confidence_cost+(0 if c['wire_attached'] else .4)
                    cost += .4 if token.get('approximate_split_geometry') else 0.
                    if field == 'number' and any(p.get('number') == text and p['owner'] == pin['owner']
                                                  and aliases.get(p['candidate_id']) != c['candidate_id'] for p in data['pin_candidates']):
                        continue
                    ranked.append((cost, token))
                if ranked:
                    ranked.sort(key=lambda v: (v[0], -v[1]['confidence'], v[1]['text']))
                    cost, token = ranked[0]
                    competing = next((v for v in ranked[1:] if v[1]['text'] != token['text']), None)
                    spelling_clear = competing is None or competing[0]-cost >= .18
                    costs[i, j], options[i, j] = cost, (token, spelling_clear)
        assignments = _assign(costs, cfg['unmatched_cost'], cfg['assignment_margin'])
        for i, (j, cost, margin) in assignments.items():
            c = targets[i]
            token, spelling_clear = options[i, j]
            if not spelling_clear:
                continue
            pin = pmap[c['candidate_id']]
            if field == 'number' and any(p.get('number') == token['text'] and p['owner'] == pin['owner']
                                         and aliases.get(p['candidate_id']) != c['candidate_id'] for p in data['pin_candidates']):
                continue
            fill(pin, field, token['text'], 'joint_component_observed_OCR',
                 {**token, 'cost': cost, 'global_assignment_margin': margin,
                  'side': c['side'], 'wire_attached': c['wire_attached']})
            occupied.add(j)

    # Aggregate libraries are relations: only unique mappings may fill a field.
    for c in contacts:
        pin = pmap[c['candidate_id']]
        model, part = parts[c['index']]
        if not part:
            continue
        for field, other, mapping in [('number', 'pinname', 'name_to_numbers'),
                                      ('pinname', 'number', 'number_to_names')]:
            choices = part[mapping].get(str(pin.get(other)), [])
            if len(choices) != 1:
                continue
            if field == 'number' and any(p['owner'] == pin['owner'] and p.get('number') == choices[0]
                                        and aliases.get(p['candidate_id']) != c['candidate_id'] for p in data['pin_candidates']):
                continue
            fill(pin, field, choices[0], 'joint_component_unique_library_relation',
                 {'model': model, 'observed_field': other, 'observed_value': pin.get(other),
                  'library_source_sha256': library.get('source_sha256')})

    # Interpolate only an observed, consistent numbering progression on a side.
    if cfg.get('sequence_fill_enabled'):
        for (index, side), cs in by_side.items():
            box = rmap[index]['bbox_top_left']
            cs = sorted(cs, key=lambda c: _local(pmap[c['candidate_id']]['point'], box, side)[0])
            anchors = [(i, int(pmap[c['candidate_id']]['number'])) for i, c in enumerate(cs)
                       if str(pmap[c['candidate_id']].get('number', '')).isdigit()
                       and pmap[c['candidate_id']].get('number_status') not in {'public_two_terminal_contact_convention'}]
            if len(anchors) < cfg['sequence_min_anchors']:
                continue
            slopes = {(b[1]-a[1])/(b[0]-a[0]) for a, b in zip(anchors, anchors[1:])}
            if len(slopes) != 1 or next(iter(slopes)) not in {-2., -1., 1., 2.}:
                continue
            slope = next(iter(slopes)); intercept = anchors[0][1]-slope*anchors[0][0]
            positions = [_local(pmap[c['candidate_id']]['point'], box, side)[0] for c in cs]
            if len(positions) > 2:
                pitch = float(np.median(np.diff(positions)))
                if pitch <= 0 or max(abs((v-positions[0])/pitch-i) for i, v in enumerate(positions)) > cfg['sequence_max_residual_rows']:
                    continue
            for i in range(anchors[0][0]+1, anchors[-1][0]):
                pin = pmap[cs[i]['candidate_id']]; number = str(int(intercept+slope*i))
                if int(number) <= 0 or any(p['owner'] == pin['owner'] and p.get('number') == number for p in data['pin_candidates']):
                    continue
                fill(pin, 'number', number, 'joint_component_anchored_sequence',
                     {'side': side, 'anchors': anchors, 'stride': int(slope), 'interpolation_only': True})

    for contact in contacts:
        pin = pmap[contact['candidate_id']]
        decisions.append({**contact, 'owner': pin['owner'], 'number': pin.get('number'), 'pinname': pin.get('pinname'),
                          'model': parts[contact['index']][0]})
    # Keep every raw hypothesis; aliases resolve only at physical-contact export.
    data['joint_contact_aliases'] = aliases
    data['joint_contact_decisions'] = decisions
    updated_graph = rebuild_references(data, graph, records)
    validate_fields(data, updated_graph, records)
    summary = {'raw_candidates': len(data['pin_candidates']), 'physical_contacts': len(contacts),
               'duplicate_contact_hypotheses': sum(len(c['member_ids'])-1 for c in contacts),
               'resolved_contacts': sum(pmap[c['candidate_id']].get('number') is not None
                                        and pmap[c['candidate_id']].get('pinname') is not None for c in contacts),
               'fields_added_by_source': dict(Counter(c['source'] for c in changes)),
               'component_library_matches': sum(part is not None for _, part in parts.values()),
               'points_changed': 0, 'wire_edges_changed': 0}
    return data, updated_graph, {'summary': summary, 'changes': changes, 'contacts': decisions}


def export_joint_contacts(pins, graph, records, baseline, cfg):
    """Export actual segments after resolving contact aliases and all members."""
    validate_fields(pins, graph, records)
    out = {'components': deepcopy(baseline['components']), 'pins': {}, 'nets': {}}
    pmap = {p['candidate_id']: p for p in pins['pin_candidates']}
    aliases = pins['joint_contact_aliases']
    refs, pending = {}, []
    for c in pins['joint_contact_decisions']:
        pin = pmap[c['candidate_id']]
        if pin['owner'] not in out['components'] or pin.get('number') is None or pin.get('pinname') is None:
            pending.append(c)
            continue
        key = 'pin_'+str(pin['number'])
        target = out['pins'].setdefault(pin['owner'], {})
        if key in target:
            raise ValueError('Two distinct contacts claim the same physical pin')
        target[key] = {'pinname': pin['pinname'], 'point': {'x': pin['point'][0], 'y': pins['size'][1]-pin['point'][1]}}
        for cid in c['member_ids']:
            refs[cid] = pin['owner']+'.'+str(pin['number'])
    edges = {e['edge_id']: e for e in graph['edges']}
    withheld = []
    for net in graph['networks']:
        members = net['pin_candidate_ids']
        if not members:
            continue
        reasons = []
        if any(cid not in refs for cid in members):
            reasons.append('contact_identity_unresolved')
        if net.get('unresolved_junction_ids'):
            reasons.append('junction_semantics_unresolved')
        if len(set(refs.get(cid) for cid in members)) < 2 and not cfg['allow_single_contact_net']:
            reasons.append('single_contact_net_disabled')
        if reasons:
            withheld.append({'network_id': net['network_id'], 'reasons': reasons})
            continue
        segments = {}
        for eid in net['edge_ids']:
            for a, b in zip(edges[eid]['polyline'], edges[eid]['polyline'][1:]):
                segments[f'edge_{len(segments)+1}'] = [{'x': a[0], 'y': pins['size'][1]-a[1]},
                                                    {'x': b[0], 'y': pins['size'][1]-b[1]}]
        if not segments:
            continue
        out['nets'][net['network_id']] = {'hyperGraph': '('+','.join(sorted({refs[cid] for cid in members}))+')', 'edges': segments}
    return out, {'exported_components': len(out['components']), 'exported_pins': sum(map(len, out['pins'].values())),
                 'exported_nets': len(out['nets']), 'exported_segments': sum(len(n['edges']) for n in out['nets'].values()),
                 'pending_contacts': pending, 'withheld_networks': withheld,
                 'network_reason_counts': dict(Counter(r for n in withheld for r in n['reasons'])),
                 'raw_candidates_preserved': len(aliases), 'points_changed': 0}
