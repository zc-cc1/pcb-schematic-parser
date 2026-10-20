"""Export observed power contacts and physical wires withheld by field gates.

Consumes current-image model artifacts only. Ordinary pin fields, component
records and existing wire geometry are preserved. No long-distance rail lines
are drawn: same-caption branches remain separate physical nets in the JSON.
"""
from collections import defaultdict
from copy import deepcopy

POWER_TYPES = {'gnd', 'v', 'vcc'}
IC_TYPES = {'box', 'ic', 'connector', 'raw_block', 'block'}


def _members(net):
    return {member.strip() for member in net['hyperGraph'].strip('()').split(',') if member.strip()}


def _contact(pin, height):
    x, y = pin['point']
    return {'pinname': pin.get('pinname') or '', 'point': {'x': float(x), 'y': float(height-y)}}


def _restore_primary_contacts(out, graph, pmap, attachments, accepted, height):
    """Recover a primary tip only when every other owned tip explicitly aliases it.

    A duplicate secondary hypothesis can make the upstream single-tip gate
    withhold a real power contact. Its primary reference is existing evidence;
    neither closest-point guessing nor a newly assigned pin number is needed.
    """
    owners = defaultdict(list)
    for pin in pmap.values():
        if pin.get('owner_status') == 'assigned':
            owners[pin.get('owner')].append(pin)
    audit = []
    net_map = {net['network_id']: net for net in graph['networks']}
    for owner, pins in owners.items():
        component = out['components'].get(owner, {})
        label = str(component.get('value') or '').upper()
        if component.get('type') not in POWER_TYPES or label not in accepted or len(pins) < 2:
            continue
        primaries = [p for p in pins if not p.get('text_hypothesis_primary_candidate_id')]
        if len(primaries) != 1:
            continue
        primary = primaries[0]
        cid = primary['candidate_id']
        if any(p is not primary and (p.get('pinname_status') != 'duplicate_tip_hypothesis'
               or p.get('text_hypothesis_primary_candidate_id') != cid) for p in pins):
            continue
        attachment = attachments.get(cid, {})
        nid = attachment.get('network_id')
        source = net_map.get(nid, {})
        if (primary.get('number') is None or attachment.get('attachment_status') != 'attached'
                or nid not in out['nets'] or cid not in source.get('pin_candidate_ids', [])
                or attachment.get('point') != primary['point']):
            continue
        key = 'pin_' + str(primary['number'])
        ref = f"{owner}.{primary['number']}"
        members = _members(out['nets'][nid])
        if ref in members:
            continue
        # Do not attach a known exported contact to a second network.
        if any(ref in _members(net) for net in out['nets'].values()):
            continue
        out['pins'].setdefault(owner, {}).setdefault(key, _contact(primary, height))
        out['nets'][nid]['hyperGraph'] = '(' + ','.join(sorted(members | {ref})) + ')'
        audit.append({'network_id': nid, 'candidate_id': cid, 'member': ref,
                      'observed_rail': label, 'reason': 'explicit_primary_of_duplicate_power_tips'})
    return audit


def restore_power_wires(base, graph, pins, cleanup):
    """Return a new target and audit; never modify the input artifacts.

    cleanup is the same per-net wire cleanup used by the batch entrypoint.
    Provenance disagreement fails before any change, allowing the caller to
    retain its valid base result. The caller validates the final target schema.
    """
    if (not graph.get('image_sha256') or graph['image_sha256'] != pins.get('image_sha256')
            or graph.get('size') != pins.get('size')
            or graph.get('coordinate_origin') != 'top_left'
            or pins.get('coordinate_origin') != 'top_left'
            or not graph.get('prediction_only') or not pins.get('prediction_only')):
        raise ValueError('Power wire artifacts must refer to the same current-image predictions')
    out = deepcopy(base)
    pmap = {p['candidate_id']: p for p in pins['pin_candidates']}
    emap = {e['edge_id']: e for e in graph['edges']}
    sites = {s['site_id']: s for s in graph.get('power_sites', [])}
    attachments = {a['candidate_id']: a for a in graph['pin_attachments']}
    accepted = {d['label'] for d in graph.get('power_merge_decisions', [])
                if not d['withheld_reasons'] and d['status'] == 'power_logical_connectivity_proposal'}
    height = graph['size'][1]
    primary_audit = _restore_primary_contacts(out, graph, pmap, attachments,
                                              {str(v).upper() for v in accepted}, height)
    candidates = {}
    for net in graph['networks']:
        names = net.get('power_names', [])
        if (not net.get('power_logical_merge') or net['network_id'] in out['nets']
                or len(names) != 1 or names[0] not in accepted):
            continue
        edge_ids = set(net['edge_ids'])
        for physical in graph.get('physical_networks', []):
            if not physical['edge_ids'] or not set(physical['edge_ids']) <= edge_ids:
                continue
            site_ids = [sid for sid in net.get('power_site_ids', [])
                        if sites[sid]['candidate_id'] in physical['pin_candidate_ids']]
            if site_ids:
                nid = 'RESTORED_' + physical['network_id']
                candidates[nid] = {**physical, 'network_id': nid,
                                   'power_names': names, 'power_site_ids': site_ids}
    audit = []
    for nid, net in sorted(candidates.items()):
        if nid in out['nets'] or net['unresolved_junction_ids']:
            continue
        name = net['power_names'][0]
        pending = {}
        qualified_sites = []
        for sid in net['power_site_ids']:
            site = sites[sid]
            pin = pmap[site['candidate_id']]
            owner = pin.get('owner')
            component = out['components'].get(owner, {})
            if (site['withheld_reasons'] or site['label'] != name
                    or pin.get('owner_status') != 'assigned' or site['component_id'] != owner
                    or component.get('type') not in POWER_TYPES
                    or str(component.get('value') or '').upper() != name.upper()
                    or pin.get('number') is None
                    or attachments.get(pin['candidate_id'], {}).get('attachment_status') != 'attached'):
                continue
            qualified_sites.append(sid)
            key = 'pin_' + str(pin['number'])
            if key not in out['pins'].get(owner, {}):
                pending[(owner, key)] = _contact(pin, height)
        if not qualified_sites:
            continue
        refs, vertices = set(), set()
        for cid in net['pin_candidate_ids']:
            pin = pmap[cid]
            owner, number = pin.get('owner'), pin.get('number')
            if owner not in out['components'] or number is None:
                continue
            key = 'pin_' + str(number)
            if key not in out['pins'].get(owner, {}) and (owner, key) not in pending:
                continue
            refs.add(f'{owner}.{number}')
            component = out['components'][owner]
            kind = component['type']
            if kind in IC_TYPES:
                vertices.add((owner, str(number)))
            elif kind in POWER_TYPES:
                vertices.add((kind, str(component.get('value'))))
            else:
                vertices.add((owner,))
        if len(vertices) < 2:
            continue
        segments = {}
        for eid in net['edge_ids']:
            points = emap[eid]['polyline']
            for a, b in zip(points, points[1:]):
                segments[f'edge_{len(segments)+1}'] = [
                    {'x': float(a[0]), 'y': float(height-a[1])},
                    {'x': float(b[0]), 'y': float(height-b[1])}]
        restored = {'hyperGraph': '(' + ','.join(sorted(refs)) + ')', 'edges': segments}
        local = {'nets': {nid: restored}}
        if cleanup is not None:
            cleanup(local)
        if not local['nets'][nid]['edges']:
            continue
        for (owner, key), pin in pending.items():
            out['pins'].setdefault(owner, {})[key] = pin
        out['nets'][nid] = local['nets'][nid]
        audit.append({'network_id': nid, 'observed_rail': name, 'members': len(refs),
                      'added_power_contacts': len(pending), 'edges': len(out['nets'][nid]['edges']),
                      'source_power_sites': net['power_site_ids']})
    return out, {'version': 'observed-physical-power-wires-v1', 'restored_networks': audit,
                 'primary_contacts': primary_audit, 'restored_net_count': len(audit),
                 'primary_contact_count': len(primary_audit), 'inference_calls': 0}
