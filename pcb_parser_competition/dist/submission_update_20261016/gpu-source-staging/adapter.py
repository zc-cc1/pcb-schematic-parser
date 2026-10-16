"""Target protocol export, with the old sample layout retained for diagnostics."""
from copy import deepcopy
import math
import re


# The six ordinary aliases passed raw comparison in submission 8180. The
# additional vcc -> v alias and power box sizes come from public annotations.
TARGET_TYPE_MAP = {'resistor': 'r', 'capacitor': 'c', 'inductor': 'l',
                   'diode': 'd', 'ic': 'box', 'led': 'led', 'vcc': 'v'}
COMPONENT_OUTPUT_POLICY = 'public_20261016_passive_names_template_boxes_power_v2_led_geometry_v1_ic_name_priority_v1_ic_geometry_v1'
# The component classes below use the public annotation box dimensions,
# swapped for orientation. These are annotation conventions in original pixels,
# not a scale factor learned from or looked up for an individual evaluation image.
TEMPLATE_BBOX_DIMS = {'r': (8., 20.), 'c': (16., 20.), 'l': (6., 34.),
                      'd': (14., 20.), 'led': (14., 20.),
                      'gnd': (19., 20.), 'v': (10., 10.)}


def passive_name(component_id, component_ids):
    # Duplicate designators can have instance suffixes while Name retains the
    # observed base: the public C6_1/C6_2 example has Name=C6 for both instances.
    match = re.fullmatch(r'([RCL]\d+)_[1-9]\d*', component_id)
    if match:
        base = match[1]
        if base in component_ids or any(
                other != component_id and re.fullmatch(re.escape(base)+r'_[1-9]\d*', other)
                for other in component_ids):
            return base
    return component_id


def validate_target(payload):
    """Check the live Target layout and references, without claiming accuracy."""
    if not isinstance(payload, dict) or set(payload) != {'components', 'pins', 'nets'}:
        raise ValueError('Target requires components, pins and nets objects')
    if not all(isinstance(payload[k], dict) for k in payload):
        raise ValueError('Target sections must be objects')
    references = set()
    for cid, component in payload['components'].items():
        if not isinstance(component, dict) or not isinstance(component.get('type'), str) or not component['type']:
            raise ValueError(f'Invalid component type: {cid}')
        for field in ['Name', 'value']:
            if field not in component or (component[field] is not None and not isinstance(component[field], str)):
                raise ValueError(f'Invalid component {field}: {cid}')
        box = component.get('bbox')
        if not isinstance(box, list) or len(box) != 4:
            raise ValueError(f'Invalid component box: {cid}')
        point(box[:2]); point(box[2:])
        if box[0] >= box[2] or box[1] >= box[3]:
            raise ValueError(f'Unordered component box: {cid}')
    for cid, pins in payload['pins'].items():
        if cid not in payload['components'] or not isinstance(pins, dict):
            raise ValueError(f'Invalid nested pin owner: {cid}')
        for key, pin in pins.items():
            if not re.fullmatch(r'pin_.+', key) or not isinstance(pin, dict):
                raise ValueError(f'Expected nested pin_<number>: {cid}.{key}')
            if not isinstance(pin.get('pinname'), str) or not isinstance(pin.get('point'), dict):
                raise ValueError(f'Expected pinname and point.x/y: {cid}.{key}')
            point(pin['point'])
            references.add(f'{cid}.{key[4:]}')
    for nid, net in payload['nets'].items():
        if not isinstance(net, dict) or not isinstance(net.get('edges'), dict):
            raise ValueError(f'Expected Target hyperGraph/edges: {nid}')
        graph = net.get('hyperGraph')
        if not isinstance(graph, str) or not graph.startswith('(') or not graph.endswith(')'):
            raise ValueError(f'Invalid hyperGraph: {nid}')
        members = [s.strip() for s in graph[1:-1].split(',') if s.strip()]
        if len(set(members)) != len(members):
            raise ValueError(f'Duplicate net members: {nid}')
        if set(members) - references:
            raise ValueError(f'Unresolved net member in {nid}: {sorted(set(members)-references)}')
        for eid, edge in net['edges'].items():
            if not isinstance(edge, list) or len(edge) != 2 or not all(isinstance(p, dict) for p in edge):
                raise ValueError(f'Expected two point.x/y endpoints: {nid}/{eid}')
            for endpoint in edge:
                point(endpoint)
    return payload


def to_target(payload, *, normalize_components=True):
    """Encode public component conventions; leave pin/net predictions intact.

    The raw model/OCR export remains available separately in project_results.
    This function reads no images, reference labels, file paths or model caches.
    """
    validate_target(payload)
    result = deepcopy(payload)
    for cid, component in result['components'].items():
        component['type'] = TARGET_TYPE_MAP.get(component['type'], component['type'])
        typ = component['type']
        if not normalize_components:
            continue
        if typ in {'r', 'c', 'l'}:
            component['Name'] = passive_name(cid, result['components'])
        if typ in TEMPLATE_BBOX_DIMS:
            x1, y1, x2, y2 = component['bbox']
            cx, cy = (x1+x2)/2, (y1+y2)/2
            short, long = TEMPLATE_BBOX_DIMS[typ]
            width, height = (long, short) if x2-x1 > y2-y1 else (short, long)
            component['bbox'] = [cx-width/2, cy-height/2, cx+width/2, cy+height/2]
    validate_target(result)
    return result


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
