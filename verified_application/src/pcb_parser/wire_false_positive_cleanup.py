"""Remove redundant or unsupported wire geometry using only current-image evidence.

This runs after junction-based network reconciliation. It does not alter net
membership, components, or pins, and never reads reference annotations.
"""
from copy import deepcopy
import math

import cv2
import numpy as np


DEFAULT = {'enabled': False, 'maximum_changes': 128}


def _points(edge):
    return np.asarray([[p['x'], p['y']] for p in edge], float)


def _length(edge):
    a, b = _points(edge)
    return float(np.linalg.norm(b-a))


def _coverage(edge, others, tolerance=3.):
    a, b = _points(edge)
    vector = b-a
    length = float(np.linalg.norm(vector))
    if length < 1e-6:
        return 0.
    unit = vector/length
    intervals = []
    for other in others:
        c, d = _points(other)-a
        if max(abs(c[0]*unit[1]-c[1]*unit[0]),
               abs(d[0]*unit[1]-d[1]*unit[0])) > tolerance:
            continue
        low, high = sorted((float(c@unit), float(d@unit)))
        low, high = max(0., low), min(length, high)
        if high > low:
            intervals.append((low, high))
    end = total = 0.
    for low, high in sorted(intervals):
        if high > end:
            total += high-max(end, low)
            end = high
    return total/length


class Cleanup:
    def __init__(self, target, rgb, observed_junctions):
        self.out = deepcopy(target)
        self.height, self.width = rgb.shape[:2]
        red, green, blue = [np.asarray(rgb)[:, :, i].astype(np.int16) for i in range(3)]
        self.red = (red > green+35) & (red > blue+35) & (green < 150) & (blue < 150)
        self.green = (green > red+20) & (green > blue+20) & (green > 60)
        self.color = self.red | self.green
        self.junctions = [np.asarray(point, float) for point in observed_junctions]
        self.events = []

    def support(self, edge, mask, radius=2):
        a, b = _points(edge)
        count = max(2, math.ceil(float(np.linalg.norm(b-a)))+1)
        samples = np.rint(np.linspace(a, b, count)).astype(int)
        h, w = self.height, self.width
        return sum(radius <= x < w-radius and radius <= h-y < h-radius and
                   mask[h-y-radius:h-y+radius+1, x-radius:x+radius+1].any()
                   for x, y in samples)/len(samples)

    def trim_box_pin_overruns(self):
        """Stop a green wire at a measured box/IC or capacitor pin's red lead."""
        for nid, net in self.out['nets'].items():
            refs=sorted(set(str(net.get('hyperGraph','')).strip('()').split(',')))
            for ref in refs:
                owner,separator,number=ref.rpartition('.')
                comp=self.out['components'].get(owner,{})
                if not separator or comp.get('type') not in {'box','ic','c'}:
                    continue
                pin=self.out['pins'].get(owner,{}).get('pin_'+number,{}).get('point')
                if not pin:
                    continue
                point=np.asarray([pin['x'],pin['y']],float)
                left,bottom,right,top=comp['bbox']
                center=np.asarray([(left+right)/2,(bottom+top)/2],float)
                for eid,edge in net['edges'].items():
                    positions=_points(edge)
                    for index in (0,1):
                        endpoint,other=positions[index],positions[1-index]
                        vector=endpoint-other;den=float(vector@vector)
                        if den<=0:
                            continue
                        fraction=float((point-other)@vector/den)
                        nearest=other+fraction*vector
                        gap=float(np.linalg.norm(endpoint-point))
                        if not (.4<fraction<.95 and 8<=gap<=45
                                and np.linalg.norm(point-nearest)<=3):
                            continue
                        if float((endpoint-point)@(center-point))<=0 or np.linalg.norm(other-point)<15:
                            continue
                        direction=(endpoint-point)/gap
                        def segment(a,b):
                            return [{'x':float(a[0]),'y':float(a[1])},
                                    {'x':float(b[0]),'y':float(b[1])}]
                        red=segment(point+direction*4,endpoint-direction*2)
                        green=segment(point-direction*12,point-direction*4)
                        if (self.support(red,self.red,radius=1)<.9
                                or self.support(green,self.green,radius=1)<.9):
                            continue
                        before=deepcopy(edge[index])
                        edge[index]={'x':float(point[0]),'y':float(point[1])}
                        self.events.append({'rule':'capacitor_pin_red_lead_trim' if comp['type']=='c' else 'box_pin_red_lead_trim','network_id':nid,
                                            'edge_id':eid,'pin':ref,'endpoint':index,
                                            'before':before,'after':deepcopy(edge[index])})
                        positions[index]=point

    def discard(self, rule, net_id, edge_id):
        self.events.append({'rule': rule, 'network_id': net_id, 'edge_id': edge_id,
                            'edge': deepcopy(self.out['nets'][net_id]['edges'][edge_id])})
        del self.out['nets'][net_id]['edges'][edge_id]

    def unsupported_long_diagonals(self):
        for nid, net in self.out['nets'].items():
            for eid, edge in list(net['edges'].items()):
                if _length(edge) <= 250:
                    continue
                a, b = _points(edge)
                angle = math.degrees(math.atan2(abs(b[1]-a[1]), abs(b[0]-a[0])))
                if 3 < angle < 87 and self.support(edge, self.color) < .5:
                    self.discard('unsupported_long_diagonal', nid, eid)

    def red_resistor_symbol_leads(self):
        for nid, net in self.out['nets'].items():
            pins = []
            for ref in str(net.get('hyperGraph', '')).strip('()').split(','):
                owner, separator, number = ref.rpartition('.')
                if not separator or self.out['components'].get(owner, {}).get('type') != 'r':
                    continue
                point = self.out['pins'].get(owner, {}).get('pin_'+number, {}).get('point')
                if point:
                    pins.append(np.asarray([point['x'], point['y']], float))
            if not pins:
                continue
            for eid, edge in list(net['edges'].items()):
                if _length(edge) > 25:
                    continue
                a, b = _points(edge)
                if min((float(np.linalg.norm(q-p)) for q in (a, b) for p in pins), default=math.inf) > 3.5:
                    continue
                # Junction splitting is needed at an observed multi-arm dot.
                if any(min(float(np.linalg.norm(a-dot)), float(np.linalg.norm(b-dot))) <= 5
                       for dot in self.junctions):
                    continue
                if self.support(edge, self.red) >= .95 and self.support(edge, self.green) <= .25:
                    self.discard('red_resistor_symbol_lead', nid, eid)

    def tiny_fragments(self):
        for nid, net in self.out['nets'].items():
            for eid, edge in list(net['edges'].items()):
                if _length(edge) <= 4.1:
                    self.discard('tiny_fragment', nid, eid)

    def short_red_symbol_fragments(self):
        """Remove pin-adjacent red lead fragments, keeping actual green branches.

        Raster pin markers can leave isolated green pixels on a red symbol.
        A connected green component larger than 3x3 is conductor evidence and
        prevents this rule, even when only its endpoint touches the fragment.
        """
        labels = stats = None
        for nid, net in self.out['nets'].items():
            pins = []
            for ref in str(net.get('hyperGraph', '')).strip('()').split(','):
                owner, separator, number = ref.strip().rpartition('.')
                if (not separator or self.out['components'].get(owner, {}).get('type')
                        not in {'r', 'c', 'l', 'box', 'ic', 'gnd'}):
                    continue
                point = self.out['pins'].get(owner, {}).get('pin_'+number, {}).get('point')
                if point:
                    pins.append(np.asarray([point['x'], point['y']], float))
            if not pins:
                continue
            for eid, edge in list(net['edges'].items()):
                if not 4.1 < _length(edge) <= 25:
                    continue
                a, b = _points(edge)
                if min(float(np.linalg.norm(q-p)) for q in (a, b) for p in pins) > 8:
                    continue
                if any(min(float(np.linalg.norm(a-dot)), float(np.linalg.norm(b-dot))) <= 5
                       for dot in self.junctions):
                    continue
                if self.support(edge, self.red, radius=1) < .9 or self.support(edge, self.green, radius=1) > .4:
                    continue
                if labels is None:
                    _, labels, stats, _ = cv2.connectedComponentsWithStats(self.green.astype('uint8'), 8)
                green_ids = set()
                samples = np.rint(np.linspace(a, b, math.ceil(_length(edge))+1)).astype(int)
                for x, y in samples:
                    row = self.height-y
                    if 1 <= x < self.width-1 and 1 <= row < self.height-1:
                        green_ids.update(labels[row-1:row+2, x-1:x+2].ravel())
                if any(stats[index, cv2.CC_STAT_WIDTH] > 3 or stats[index, cv2.CC_STAT_HEIGHT] > 3
                       for index in green_ids if index):
                    continue
                self.discard('short_red_symbol_fragment', nid, eid)

    def within_net_duplicates(self):
        for nid, net in self.out['nets'].items():
            retained = []
            for eid, edge in list(net['edges'].items()):
                if _coverage(edge, retained) >= .9:
                    self.discard('within_net_covered', nid, eid)
                else:
                    retained.append(edge)

    def cross_net_duplicates(self):
        for nid, net in self.out['nets'].items():
            for eid, edge in list(net['edges'].items()):
                if _length(edge) <= 25:
                    continue
                other = [candidate for other_id, other_net in self.out['nets'].items()
                         if other_id != nid for candidate in other_net['edges'].values()
                         if _length(candidate) > _length(edge)+10]
                if _coverage(edge, other, tolerance=5.) >= .95:
                    self.discard('cross_net_covered', nid, eid)

    def long_red_resistor_leads_at_junctions(self):
        """At a junction, only a longer pure-symbol resistor lead is disposable."""
        for nid, net in self.out['nets'].items():
            pins = []
            for ref in str(net.get('hyperGraph', '')).strip('()').split(','):
                owner, separator, number = ref.rpartition('.')
                if not separator or self.out['components'].get(owner, {}).get('type') != 'r':
                    continue
                point = self.out['pins'].get(owner, {}).get('pin_'+number, {}).get('point')
                if point:
                    pins.append(np.asarray([point['x'], point['y']], float))
            if not pins:
                continue
            for eid, edge in list(net['edges'].items()):
                if not 18 <= _length(edge) <= 25:
                    continue
                a, b = _points(edge)
                if min((float(np.linalg.norm(q-p)) for q in (a, b) for p in pins), default=math.inf) > 3.5:
                    continue
                if not any(min(float(np.linalg.norm(a-dot)), float(np.linalg.norm(b-dot))) <= 5
                           for dot in self.junctions):
                    continue
                if self.support(edge, self.red) >= .95 and self.support(edge, self.green) <= .25:
                    self.discard('long_red_resistor_symbol_lead_at_junction', nid, eid)

    def extend_long_wire_to_observed_junction(self):
        """Complete an almost-finished long conductor along visible ink to one dot."""
        for nid, net in self.out['nets'].items():
            for eid, edge in net['edges'].items():
                if _length(edge) < 200:
                    continue
                for index in (0, 1):
                    q, other = _points(edge)[index], _points(edge)[1-index]
                    direction = q-other
                    wire_length = float(np.linalg.norm(direction))
                    candidates = []
                    for dot in self.junctions:
                        continuation = dot-q
                        gap = float(np.linalg.norm(continuation))
                        if not 20 <= gap <= 30:
                            continue
                        if float(continuation@direction)/(gap*wire_length) < .98:
                            continue
                        if abs(direction[0]*continuation[1]-direction[1]*continuation[0])/wire_length > 2:
                            continue
                        bridge = [{'x':float(q[0]),'y':float(q[1])},
                                  {'x':float(dot[0]),'y':float(dot[1])}]
                        if self.support(bridge, self.color) >= .98:
                            candidates.append(dot)
                    if len(candidates) == 1:
                        before = deepcopy(edge[index])
                        dot = candidates[0]
                        edge[index] = {'x':float(dot[0]),'y':float(dot[1])}
                        self.events.append({'rule':'long_wire_endpoint_to_observed_junction',
                                            'network_id':nid,'edge_id':eid,'endpoint':index,
                                            'before':before,'after':deepcopy(edge[index])})

    def run(self):
        self.unsupported_long_diagonals()
        self.red_resistor_symbol_leads()
        self.tiny_fragments()
        self.within_net_duplicates()
        self.cross_net_duplicates()
        self.long_red_resistor_leads_at_junctions()
        self.extend_long_wire_to_observed_junction()
        self.trim_box_pin_overruns()
        self.short_red_symbol_fragments()
        for net in self.out['nets'].values():
            edges = net['edges']
            if list(edges) != ['edge_'+str(index) for index in range(1, len(edges)+1)]:
                net['edges'] = {'edge_'+str(index): edge for index, edge in enumerate(edges.values(), 1)}
        return self.out


def cleanup_wire_false_positives(target, rgb, graph, image_sha256, observed_junctions, *, cfg=None):
    config = {**DEFAULT, **(cfg or {})}
    if set(config) != set(DEFAULT) or type(config['enabled']) is not bool:
        raise ValueError('invalid_wire_false_positive_cleanup_setting')
    if type(config['maximum_changes']) is not int or not 0 <= config['maximum_changes'] <= 512:
        raise ValueError('invalid_wire_false_positive_cleanup_budget')
    if not config['enabled']:
        return deepcopy(target), {'enabled': False, 'status': 'disabled', 'changes': []}
    height, width = rgb.shape[:2]
    if (graph.get('image_sha256') != image_sha256 or graph.get('size') != [width, height]
            or graph.get('coordinate_origin') != 'top_left' or graph.get('prediction_only') is not True):
        raise ValueError('wire_false_positive_evidence_differs_from_image')
    if any(not isinstance(point, (list, tuple)) or len(point) != 2 or
           any(type(value) not in (int, float) or not math.isfinite(value) for value in point)
           for point in observed_junctions):
        raise ValueError('invalid_observed_junction')
    draft = Cleanup(target, rgb, observed_junctions)
    out = draft.run()
    if len(draft.events) > config['maximum_changes']:
        return deepcopy(target), {'enabled': True, 'status': 'withheld_budget', 'changes': [],
                                  'proposed_changes': len(draft.events)}
    if out['components'] != target['components'] or out['pins'] != target['pins'] or set(out['nets']) != set(target['nets']):
        raise ValueError('wire_false_positive_cleanup_modified_structure')
    for nid, net in out['nets'].items():
        if {key: value for key, value in net.items() if key != 'edges'} != {
                key: value for key, value in target['nets'][nid].items() if key != 'edges'}:
            raise ValueError('wire_false_positive_cleanup_modified_net_members')
    return out, {'enabled': True, 'status': 'applied' if draft.events else 'no_effect',
                 'changes': draft.events, 'reference_read': False}
