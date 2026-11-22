"""Keep valid prediction objects when a local export object is malformed."""
from copy import deepcopy
import json
import math
from pathlib import Path
import re


def valid_point(value, size=None):
    if isinstance(value, dict):
        if set(value) != {'x', 'y'}: return False
        value = [value['x'], value['y']]
    if not isinstance(value, (list, tuple)) or len(value) != 2: return False
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in value): return False
    return size is None or (0 <= value[0] <= size[0] and 0 <= value[1] <= size[1])


def sanitize_export(payload, size=None):
    """Remove only invalid objects/references. Never infer text, coordinates or joins."""
    if not isinstance(payload, dict) or any(not isinstance(payload.get(k), dict) for k in ('components', 'pins', 'nets')):
        raise ValueError('Invalid top-level prediction')
    out = {'components': {}, 'pins': {}, 'nets': {}}
    removed = []
    def note(kind, key, reason): removed.append({'kind': kind, 'id': key, 'reason': reason})
    for cid, c in payload['components'].items():
        box = c.get('bbox') if isinstance(c, dict) else None
        if (not isinstance(cid, str) or not cid or not isinstance(c, dict)
            or not isinstance(c.get('type'), str) or not c['type']
            or any(c.get(k) is not None and not isinstance(c.get(k), str) for k in ('Name','value'))
            or not isinstance(box, list) or len(box) != 4 or not valid_point(box[:2], size)
            or not valid_point(box[2:], size) or box[0] >= box[2] or box[1] >= box[3]):
            note('component', cid, 'invalid_component');continue
        out['components'][cid] = {k: deepcopy(c.get(k)) for k in ('type','Name','value','bbox')}
    refs = set()
    for cid, pins in payload['pins'].items():
        if cid not in out['components'] or not isinstance(pins, dict):
            note('pin_owner', cid, 'invalid_owner');continue
        for key, pin in pins.items():
            if (not isinstance(key, str) or not re.fullmatch(r'pin_[^.,()\s]+', key)
                or not isinstance(pin, dict) or not isinstance(pin.get('pinname'), str)
                or not isinstance(pin.get('point'), dict) or not valid_point(pin['point'], size)):
                note('pin', f'{cid}.{key}', 'invalid_pin');continue
            out['pins'].setdefault(cid, {})[key] = deepcopy(pin)
            refs.add(f'{cid}.{key[4:]}')
        if not pins:out['pins'][cid] = {}
    used = set()
    for nid, net in payload['nets'].items():
        graph = net.get('hyperGraph') if isinstance(net, dict) else None
        if (not isinstance(nid,str) or not nid or not isinstance(graph,str)
            or not graph.startswith('(') or not graph.endswith(')') or not isinstance(net.get('edges'),dict)):
            note('net',nid,'invalid_net');continue
        members = []
        for token in graph[1:-1].split(','):
            token = token.strip()
            if not token or token not in refs or token in members or token in used:
                note('reference',f'{nid}/{token}','invalid_duplicate_or_conflicting_reference');continue
            members.append(token)
        if not members:
            note('net',nid,'no_valid_member');continue
        edges = {}
        for eid, edge in net['edges'].items():
            if (not isinstance(eid,str) or not eid or not isinstance(edge,list) or len(edge)!=2
                or not all(isinstance(p,dict) and valid_point(p,size) for p in edge)):
                note('edge',f'{nid}/{eid}','invalid_edge');continue
            edges[eid] = deepcopy(edge)
        used.update(members)
        out['nets'][nid] = {'hyperGraph':'('+','.join(members)+')','edges':edges}
    return out, {'removed':removed, 'removed_count':len(removed)}


def recover_current_run(work, output_format):
    """Inspect only the unique current case work directory, never old final outputs."""
    from adapter import to_sample, to_target, validate_target
    work = Path(work).resolve()
    candidates = [p for p in work.rglob('*.json') if p.name in {'engineering_submission.json','partial_submission.json'}]
    # Later pipeline stages have shorter paths (p/export vs nested upstream stages).
    candidates.sort(key=lambda p:(len(p.relative_to(work).parts),len(str(p.relative_to(work))),str(p)))
    for p in candidates:
        if not p.resolve().is_relative_to(work):continue
        try:
            payload,_ = sanitize_export(json.loads(p.read_text(encoding='utf-8-sig')))
            if not payload['components']:continue
            validate_target(payload)
            result = to_sample(payload) if output_format=='sample' else to_target(payload) if output_format=='target' else payload
            json.dumps(result,allow_nan=False)
            return result
        except (ValueError, TypeError, KeyError, OSError):
            continue
    return None
