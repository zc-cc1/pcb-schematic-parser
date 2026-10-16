"""Convert the existing paper-format predictions to the sample README format."""
from copy import deepcopy
import math


def point(value):
    if isinstance(value, dict):
        value = [value['x'], value['y']]
    if len(value) != 2 or not all(isinstance(v, (int, float)) and not isinstance(v, bool)
                                  and math.isfinite(v) for v in value):
        raise ValueError(f'Invalid point: {value!r}')
    return list(value)


def to_sample(payload):
    """Preserve types, text, geometry and membership; change field layout only.

    Coordinates retain the project's bottom-left, H-y convention. The sample
    README does not specify a different coordinate origin or a type enum.
    """
    result = {'components': {}, 'pins': {}, 'nets': {}}
    references = {}
    for cid, component in payload['components'].items():
        box = component['bbox']
        if len(box) != 4:
            raise ValueError(f'Invalid component box: {cid}')
        point(box[:2]); point(box[2:])
        if box[0] >= box[2] or box[1] >= box[3]:
            raise ValueError(f'Unordered component box: {cid}')
        result['components'][cid] = {
            **deepcopy(component),
            'center': [(box[0]+box[2])/2, (box[1]+box[3])/2],
        }
    for cid, pins in payload['pins'].items():
        if cid not in result['components']:
            raise ValueError(f'Pin owner missing: {cid}')
        for key, pin in pins.items():
            if not key.startswith('pin_') or not key[4:]:
                raise ValueError(f'Invalid paper pin ID: {cid}.{key}')
            number = key[4:]
            pid = f'{cid}_{number}'
            if pid in result['pins']:
                raise ValueError(f'Flattened pin ID collision: {pid}')
            result['pins'][pid] = {
                'component': cid, 'position': point(pin['point']),
                'pin_number': number, 'pinname': pin['pinname'],
            }
            references[f'{cid}.{number}'] = pid
    for nid, net in payload['nets'].items():
        hypergraph = net['hyperGraph']
        if not isinstance(hypergraph, str) or not (hypergraph.startswith('(') and hypergraph.endswith(')')):
            raise ValueError(f'Invalid hyperGraph: {nid}')
        members = [p.strip() for p in hypergraph[1:-1].split(',') if p.strip()]
        if len(set(members)) != len(members):
            raise ValueError(f'Duplicate net members: {nid}')
        try:
            pin_ids = [references[member] for member in members]
        except KeyError as error:
            raise ValueError(f'Unresolved net member in {nid}: {error.args[0]}') from error
        segments = []
        for eid, endpoints in net['edges'].items():
            if len(endpoints) != 2:
                raise ValueError(f'Expected two-point edge: {nid}/{eid}')
            segments.append({'points': [point(p) for p in endpoints], 'type': 'polyline'})
        result['nets'][nid] = {'pins': pin_ids, 'wire_segments': segments}
    return result
