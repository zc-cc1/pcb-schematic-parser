"""Image-derived patterns from the connectivity pilot, without named owners.

This private draft is never exported directly. The rule executor validates and
commits the complete pin/identity/topology delta as one transaction.
"""
from collections import defaultdict
from copy import deepcopy
import math
import re

import cv2
import numpy as np

from .terminal_identity_recovery import recover_capacitors, recover_grounds
from .terminal_topology import recover_ground_branches, prune_no_connect_nets
from .color_trace import make_graph
from .pin_numeric_connector import observations
from .wire_graph import detect_junction_dots
from .numeric_power_caption import read_numeric_supply, numeric_voltage, outward_wire
from .red_resistor_contacts import recover_resistor_contacts
from .single_led_identity import recover_single_leds

NC = {'NC', 'N.C.', 'N/C', 'DNC'}
RAILS = {'gnd', 'v', 'vcc', 'vdd'}
NAMED_SUPPLIES = {'VCC','VDD','VBAT','VBUS','VIN','VOUT'}


def guarded_named_supply(words, rgb, box, point):
    """Two OCR views can confirm a model supply only with its T and outward ink."""
    a,b,c,d=box;cx,cy=(a+c)/2,(b+d)/2
    nearby=[token for token in words if str(token.get('text','')).upper() in NAMED_SUPPLIES
            and token.get('confidence',0)>=.85 and token.get('view')
            and abs((token['bbox'][0]+token['bbox'][2])/2-cx)<45
            and abs((token['bbox'][1]+token['bbox'][3])/2-cy)<25]
    names={token['text'].upper() for token in nearby}
    if len(names)!=1:return None
    caption=next(iter(names))
    views={token['view'] for token in nearby if token['text'].upper()==caption}
    if len(views)<2 or max(token['confidence'] for token in nearby)<.93:return None
    wire=outward_wire(rgb,box,point)
    if wire is None:return None
    return {'caption':caption,'views':sorted(views),'outward_wire':wire}


def members(net):
    return {s.strip() for s in str(net.get('hyperGraph') or '').strip('()').split(',') if s.strip()}


class StructuralDraft:
    def __init__(self, target, rgb, records, pins, ocr, wire_cfg, cfg, reader=None):
        self.out = deepcopy(target)
        self.rgb = np.asarray(rgb)
        self.h, self.w = self.rgb.shape[:2]
        self.records = deepcopy(records['records'])
        self.rmap = {r['component_id']: r for r in self.records}
        self.pin_document = pins
        self.source_points = defaultdict(list)
        for p in pins['pin_candidates']:
            if p.get('owner_status') == 'assigned' and not p.get('text_hypothesis_primary_candidate_id'):
                self.source_points[p.get('owner')].append(p['point'])
        self.words = observations(ocr)
        self.wire_cfg, self.cfg, self.reader = wire_cfg, cfg, reader
        self.events, self.withheld = [], []
        self.ocr_calls = 0
        c = self.rgb.astype(np.int16)
        r, g, b = [c[:, :, i] for i in range(3)]
        self.red = (r > g+35) & (r > b+35) & (g < 150) & (b < 150)
        self.green = (g > r+20) & (g > b+20) & (g > 60)
        self.dots = detect_junction_dots(self.rgb, wire_cfg)
        thick = cv2.distanceTransform(self.red.astype('uint8'), cv2.DIST_L2, 5) >= 3.2
        _, _, _, centres = cv2.connectedComponentsWithStats(thick.astype('uint8'), 8)
        for p in centres[1:]:
            x, y = map(round, p)
            arms = sum(bool(self.green[max(0,y+dy-2):min(self.h,y+dy+3),
                                       max(0,x+dx-2):min(self.w,x+dx+3)].any())
                       for dx, dy in [(8,0),(-8,0),(0,8),(0,-8)])
            if all(math.dist(p, q) > 5 for q in self.dots) and arms >= 2:
                self.dots.append(p.tolist())

    def enabled(self, name):
        return self.cfg.get('patterns', {}).get(name, True)

    def point(self, pin):
        return [pin['point']['x'], self.h-pin['point']['y']]

    def export_pin(self, p, name=''):
        return {'pinname': str(name), 'point': {'x': float(p[0]), 'y': float(self.h-p[1])}}

    def box(self, comp):
        a, b, c, d = comp['bbox']
        return [a, self.h-d, c, self.h-b]

    def forget_owner_memberships(self, owner):
        for nid, net in list(self.out['nets'].items()):
            before = members(net)
            after = {s for s in before if s.rsplit('.', 1)[0] != owner}
            if before == after:
                continue
            if after:
                net['hyperGraph'] = '('+','.join(sorted(after))+')'
            else:
                del self.out['nets'][nid]

    def terminal_pair(self, owner, record=None, axis_override=None):
        comp = self.out['components'][owner]
        record = record or self.rmap.get(owner)
        if not record:
            return None
        box = record['bbox_top_left']
        axis = int(box[3]-box[1] >= box[2]-box[0])
        center = [(box[0]+box[2])/2, (box[1]+box[3])/2]
        source_points = self.source_points.get(owner, [])
        if comp['type'] == 'c':
            if source_points:
                spans = [max(p[i] for p in source_points)-min(p[i] for p in source_points) for i in (0,1)]
                axis = int(spans[1] > spans[0]) if max(spans) > 10 else int(
                    abs(source_points[0][1]-center[1]) > abs(source_points[0][0]-center[0]))
            else:
                cx, cy = map(round, center)
                patch = self.red[max(0,cy-30):cy+31, max(0,cx-30):cx+31].astype('uint8')
                axis = int(bool(cv2.morphologyEx(patch, cv2.MORPH_OPEN, np.ones((1,14),np.uint8)).sum()))
        if axis_override is not None:
            axis = axis_override
        elif comp['type']=='c':
            from .capacitor_plate_axis import capacitor_axis_proof
            plates=capacitor_axis_proof(self.rgb,record)
            if plates is not None:axis=plates['axis']
        tangent = 1-axis
        if source_points:
            center[tangent] = float(np.median([p[tangent] for p in source_points]))
        terminals = []
        for direction in (-1, 1):
            found = None
            for distance in range(5, 86):
                p = center.copy()
                p[axis] = round(center[axis])+direction*distance
                x, y = map(round, p)
                if min(x,y) < 3 or x >= self.w-3 or y >= self.h-3:
                    break
                def hit(k):
                    q = p.copy();q[axis] += direction*k
                    xx, yy = map(round, q)
                    return self.green[max(0,yy-2):min(self.h,yy+3),max(0,xx-2):min(self.w,xx+3)].any()
                if hit(0) and (all(hit(k) for k in range(5)) or self.green[y-5:y+6,x-5:x+6].sum() >= 7):
                    p[axis] -= direction
                    found = p
                    break
            nearby = [p for p in self.dots if abs(p[tangent]-center[tangent]) <= 3
                      and 10 < direction*(p[axis]-center[axis]) < 85]
            if nearby:
                dot = min(nearby, key=lambda p: abs(p[axis]-center[axis]))
                if found is None or abs(dot[axis]-center[axis]) < abs(found[axis]-center[axis]):
                    found = dot
            if found is None:
                return None
            terminals.append(found)
        if terminals[1][axis]-terminals[0][axis] < .7*(box[axis+2]-box[axis]):
            return None
        return axis, terminals

    def normalize_passive(self, owner, axis=None):
        pair = self.terminal_pair(owner, axis_override=axis)
        if pair is None:
            self.withheld.append({'rule': 'visible_terminal_pair', 'owner': owner,
                                  'reason': 'no_two_visible_external_stems'})
            return False
        axis, terminals = pair
        before = deepcopy(self.out['pins'].get(owner, {}))
        # Symmetric devices retain their existing spatial numbering. Capacitors
        # use the exporter's established left/top=1 convention, as in the pilot.
        order = ['1', '2']
        if self.out['components'][owner]['type'] != 'c':
            if not all('pin_'+k in before for k in order):
                return False  # No identity inferred from a sequence.
            order.sort(key=lambda k: self.point(before['pin_'+k])[axis])
        after = {'pin_'+k: self.export_pin(p, before.get('pin_'+k, {}).get('pinname', k))
                 for k, p in zip(order, terminals)}
        if after == before:
            return True
        old_order=sorted(['1','2'],key=lambda k:self.point(before['pin_'+k])[axis]) if all('pin_'+k in before for k in ('1','2')) else []
        if self.out['components'][owner]['type']=='c' and old_order!=order:
            self.forget_owner_memberships(owner)
        else:
            obsolete={owner+'.'+k[4:] for k in before if k not in after}
            for nid,net in list(self.out['nets'].items()):
                kept=members(net)-obsolete
                if not kept:del self.out['nets'][nid]
                elif kept!=members(net):net['hyperGraph']='('+','.join(sorted(kept))+')'
        self.out['pins'][owner] = after
        self.events.append({'rule': 'visible_terminal_pair', 'owner': owner,
                            'axis': axis, 'terminals': terminals, 'numbering': order,
                            'before': before, 'after': deepcopy(after)})
        return True

    def recover_inductors(self):
        for owner, comp in sorted(self.out['components'].items()):
            if comp['type'] != 'l' or len(self.out['pins'].get(owner, {})) != 1 or owner not in self.rmap:
                continue
            record = self.rmap[owner]
            box = record['bbox_top_left'];axis = int(box[3]-box[1] >= box[2]-box[0])
            known = [self.point(p) for p in self.out['pins'][owner].values()]
            candidates = []
            for p in self.pin_document['pin_candidates']:
                if not any(c['component_id'] == owner for c in p.get('owner_candidates', [])):
                    continue
                if min(math.dist(p['point'], k) for k in known) < 8:
                    continue
                opponents = [r for r in self.records if r['detection_index'] in
                             [v.get('detection_index') for v in p.get('owner_candidates', [])]
                             and r['component_id'] != owner]
                if any(r.get('designator') or r.get('detection_confidence', 0) > record['detection_confidence']-.2 for r in opponents):
                    continue
                if abs(p['point'][1-axis]-known[0][1-axis]) > 4:
                    continue
                center = (box[axis]+box[axis+2])/2
                if (p['point'][axis]-center)*(known[0][axis]-center) >= 0:
                    continue
                candidates.append(p)
            if len(candidates) != 1:
                continue
            p = candidates[0];ends = sorted([known[0], p['point']], key=lambda v: v[axis])
            self.forget_owner_memberships(owner)
            self.out['pins'][owner] = {'pin_1': self.export_pin(ends[0], '1'), 'pin_2': self.export_pin(ends[1], '2')}
            self.events.append({'rule': 'opposed_inductor_endpoint', 'owner': owner,
                                'source_candidate': p['candidate_id'], 'terminals': ends})

    def arrow_labels(self):
        candidates = []
        for record in self.records:
            if record['type'] not in {'vbus_in','vbus_out','vbus_bidirectional'} or record.get('designator'):
                continue
            a,b,c,d = record['bbox_top_left'];support = defaultdict(list)
            for t in self.words:
                x1,y1,x2,y2 = t['bbox'];cy = (y1+y2)/2
                if c-10 <= x1 <= c+35 and b-8 <= cy <= d+8 and t['confidence'] >= .98 and re.fullmatch(r'[A-Za-z][A-Za-z0-9_+\-]{1,15}', t['text']):
                    support[t['text']].append(t)
            accepted = [(text, ts) for text, ts in support.items() if len({t['view'] for t in ts}) >= 2]
            if len(accepted) != 1:
                continue
            text, ts = accepted[0]
            aa,bb,cc,dd = max(0,math.floor(a)),max(0,math.floor(b)),min(self.w,math.ceil(c)),min(self.h,math.ceil(d))
            _,_,stats,_ = cv2.connectedComponentsWithStats(self.red[bb:dd,aa:cc].astype('uint8'), 8)
            shapes = [v for v in stats[1:] if v[2] >= 25 and v[3] >= 12 and v[4] >= 40]
            if len(shapes) != 1:
                continue
            x,y,bw,bh,area = map(int, shapes[0]);frame = [aa+x,bb+y,aa+x+bw-1,bb+y+bh-1]
            candidates.append({'text': text, 'point': [frame[0],(frame[1]+frame[3])/2],
                               'frame': frame, 'views': sorted({t['view'] for t in ts}),
                               'source_record': record['component_id']})
        counts = defaultdict(int)
        for row in sorted(candidates, key=lambda r: (r['point'][1],r['point'][0])):
            text = row['text'];counts[text] += 1
            key = text if counts[text] == 1 else text+'_'+str(counts[text])
            if key in self.out['components']:
                continue  # Identity collisions require separate adjudication.
            a,b,c,d = row['frame']
            self.out['components'][key] = {'type': 'net_bidirection', 'Name': key, 'value': text,
                                            'bbox': [a,self.h-d,c,self.h-b]}
            self.out['pins'][key] = {'pin_1': self.export_pin(row['point'], '1')}
            self.events.append({'rule': 'arrow_caption_identity', 'owner': key, **row})

    def led_bbox(self, vertices, apex_top):
        x,y = vertices.mean(0);left,top = vertices.min(0)-2;right,bottom = vertices.max(0)+2
        ax1,ay1 = max(0,round(x)-60),max(0,round(y)-45)
        ax2,ay2 = min(self.w,round(x)+61),min(self.h,round(y)+46)
        _,_,stats,centres = cv2.connectedComponentsWithStats(self.red[ay1:ay2,ax1:ax2].astype('uint8'),8)
        arrows = []
        for s,p in zip(stats[1:],centres[1:]):
            xx,yy = p+[ax1,ay1];a,b,c,d,area = map(int,s)
            if 10 <= c <= 20 and 10 <= d <= 20 and 35 <= area <= 100 and 8 < abs(xx-x) < 55 and ((xx<x and yy<y) if apex_top else (xx>x and yy>y)):
                arrows.append([ax1+a,ay1+b,ax1+a+c-1,ay1+b+d-1])
        if len(arrows) == 2:
            left=min(left,*(a[0] for a in arrows));top=min(top,*(a[1] for a in arrows))
            right=max(right,*(a[2] for a in arrows));bottom=max(bottom,*(a[3] for a in arrows))
        return list(map(float,[left,top,right,bottom]))

    def split_led_clusters(self):
        claimed = set()
        for record in self.records:
            owner = record['component_id']
            if record['type'] != 'led' or owner not in self.out['components']:
                continue
            a,b,c,d = record['bbox_top_left']
            x1,y1=max(0,int(a)-8),max(0,int(b)-12);x2,y2=min(self.w,int(c)+8),min(self.h,int(d)+8)
            contours,hierarchy=cv2.findContours(self.red[y1:y2,x1:x2].astype('uint8'),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            if hierarchy is None:
                continue
            triangles=[]
            for contour,info in zip(contours,hierarchy[0]):
                vertices=cv2.approxPolyDP(contour,2,True).reshape(-1,2)+[x1,y1]
                if info[3] >= 0 and len(vertices) == 3 and 80 <= cv2.contourArea(contour) <= 300:
                    triangles.append(vertices)
            if len(triangles) < 2 or len(triangles) > 4:
                continue
            supports=defaultdict(list)
            for t in self.words:
                cx,cy=(t['bbox'][0]+t['bbox'][2])/2,(t['bbox'][1]+t['bbox'][3])/2
                if re.fullmatch(r'LED\d+', t['text']) and t['confidence'] >= .98 and x1-15 < cx < x2+90 and y1-15 < cy < y2+25:
                    supports[t['text']].append(t)
            names=[(name,float(np.median([t['bbox'][0] for t in ts])),sorted({t['view'] for t in ts}))
                   for name,ts in supports.items() if len({t['view'] for t in ts}) >= 2]
            names=[n for n in names if any(5 < n[1]-v[:,0].mean() < 90 for v in triangles)]
            if len(names) != len(triangles) or owner not in {n[0] for n in names}:
                continue
            assignments=[]
            for vertices,(name,textx,views) in zip(sorted(triangles,key=lambda v:v[:,0].mean()),sorted(names,key=lambda v:v[1])):
                if name in claimed or not 5 < textx-vertices[:,0].mean() < 90:
                    break
                if name in self.out['components'] and self.out['components'][name]['type'] != 'led':
                    break
                ys=sorted(vertices[:,1]);apex_top=ys[1]-ys[0] > ys[2]-ys[1]
                assignments.append((name,vertices,apex_top,views))
            if len(assignments) != len(triangles):
                continue
            saved=deepcopy(self.out);event_count=len(self.events);record_map=deepcopy(self.rmap)
            valid=True
            for name,vertices,apex_top,views in assignments:
                left,top,right,bottom=self.led_bbox(vertices,apex_top)
                self.forget_owner_memberships(name)
                self.out['components'][name]={'type':'led','Name':name,'value':None,
                    'bbox':[left,self.h-bottom,right,self.h-top]}
                self.rmap[name]={'bbox_top_left':[left,top,right,bottom]}
                self.out['pins'][name]={}
                pair=self.terminal_pair(name,axis_override=1)
                if pair is None:
                    valid=False;break
                _,terminals=pair
                order=['2','1'] if apex_top else ['1','2']
                self.out['pins'][name]={'pin_'+k:self.export_pin(p,k) for k,p in zip(order,terminals)}
                self.events.append({'rule':'independent_led_symbols','owner':name,'triangle':vertices.tolist(),
                                    'views':views,'terminals':terminals,'cathode_at_top':bool(apex_top)})
            if not valid:
                self.out=saved;self.rmap=record_map;del self.events[event_count:]
            else:
                claimed.update(n[0] for n in assignments)
        return claimed

    def recover_power(self):
        by_owner=defaultdict(list)
        for p in self.pin_document['pin_candidates']:
            if p.get('owner_status') == 'assigned':
                by_owner[p['owner']].append(p)
        for owner,comp in sorted(self.out['components'].items()):
            if comp['type'] not in RAILS or (comp['type']=='gnd' and self.out['pins'].get(owner)):
                continue
            a,b,c,d=self.box(comp);cx,cy=(a+c)/2,(b+d)/2
            candidates=[p for p in by_owner[owner] if not p.get('text_hypothesis_primary_candidate_id') and p.get('confidence',0)>=.8]
            if len(candidates)>1:
                candidates=[p for p in candidates if min(abs(p['point'][0]-cx),abs(p['point'][1]-cy))<=3]
            if len(candidates)!=1:
                continue
            if (comp['type']!='gnd' and self.enabled('numeric_power_caption')
                    and not self.out['pins'].get(owner) and owner in self.rmap
                    and candidates[0].get('confidence',0)>=.9
                    and (not comp.get('value') or numeric_voltage(comp['value']))):
                numeric=read_numeric_supply(self.rgb,self.rmap[owner]['bbox_top_left'],candidates[0]['point'],
                    self.words,reader=self.reader,
                    remaining_budget=max(0,self.cfg.get('maximum_caption_views',8)-self.ocr_calls))
                self.ocr_calls+=numeric['calls']
                if numeric['status']=='accepted':
                    p=candidates[0]
                    self.forget_owner_memberships(owner)
                    comp['value']=numeric['caption']
                    self.out['pins'][owner]={'pin_1':self.export_pin(p['point'])}
                    self.events.append({'rule':'typed_power_contact','owner':owner,
                        'candidate':p['candidate_id'],'caption':numeric['caption'],
                        'views':sorted({t['view'] for t in numeric['observations']}),'point':p['point'],
                        'caption_kind':'numeric_voltage','caption_observations':numeric['observations'],
                        'source_confidence':p['confidence'],'outward_wire':numeric['wire']})
                    continue
                if numeric['reason']!='named_supply_handled_by_existing_rule':
                    self.withheld.append({'rule':'numeric_power_caption','owner':owner,
                        'reason':numeric['reason'],'ocr_calls':numeric['calls'],
                        'observations':numeric['observations']})
            if not self.enabled('typed_power_contact'):
                continue
            text=comp.get('value');before_value=text;views=[]
            if comp['type']!='gnd':
                ts=[t for t in self.words if t['text'].upper() in {'VCC','VDD','VBAT','VBUS','VIN','VOUT'} and t['confidence']>=.90
                    and abs((t['bbox'][0]+t['bbox'][2])/2-cx)<45 and abs((t['bbox'][1]+t['bbox'][3])/2-cy)<25]
                support=defaultdict(set)
                for t in ts:support[t['text'].upper()].add(t['view'])
                if len(support)!=1:
                    continue
                text=next(iter(support))
                if len(support[text])<3 and self.reader is not None and ts:
                    source=max(ts,key=lambda t:t['confidence'])
                    for scale in (3.,5.):
                        if self.ocr_calls>=self.cfg.get('maximum_caption_views',8):break
                        self.ocr_calls+=1
                        results=self.reader(self.rgb,source['bbox'],scale,'supply_caption')
                        for t in results:
                            if str(t.get('text','')).upper()==text and t.get('confidence',0)>=.9 and t.get('view'):
                                support[text].add(t['view'])
                guarded=None
                if len(support[text])<3:
                    guarded=guarded_named_supply(self.words,self.rgb,self.rmap[owner]['bbox_top_left'],candidates[0]['point'])
                    if guarded is None or guarded['caption']!=text:
                        self.withheld.append({'rule':'typed_power_contact','owner':owner,'reason':'insufficient_independent_caption_views'})
                        continue
                views=guarded['views'] if guarded else sorted(support[text]);comp['value']=text
            p=candidates[0]
            after={'pin_1':self.export_pin(p['point'])}
            if after==self.out['pins'].get(owner) and comp.get('value')==before_value:continue
            self.forget_owner_memberships(owner)
            self.out['pins'][owner]=after
            self.events.append({'rule':'typed_power_contact','owner':owner,'candidate':p['candidate_id'],
                                'caption':text,'views':views,'point':p['point'],
                                **({'caption_kind':'guarded_named_supply','outward_wire':guarded['outward_wire']}
                                   if comp['type']!='gnd' and guarded else {})})

    def rebuild_color_networks(self):
        pp={o+'.'+k[4:]:self.point(p) for o,ps in self.out['pins'].items() for k,p in ps.items()
            if str(p.get('pinname','')).strip().upper() not in NC}
        keepouts=np.zeros((self.h,self.w),bool)
        for comp in self.out['components'].values():
            a,b,c,d=self.box(comp)
            if comp['type'] in RAILS|{'box','ic','switch'}:
                keepouts[max(0,math.ceil(b+2)):min(self.h,math.floor(d-2)),max(0,math.ceil(a+2)):min(self.w,math.floor(c-2))]=True
        nets,bind,evidence,mask=make_graph(self.rgb,pp,self.wire_cfg,keepouts)
        def bind_contact(ref,p):
            cid=bind(p);comp=self.out['components'][ref.rsplit('.',1)[0]]
            if cid or comp['type'] not in RAILS:return cid
            cid=bind(p,12)
            if not cid:return None
            q=min((v for e in nets[cid]['edges'] for v in e['path']),key=lambda v:math.dist(v,p))
            a,b,c,d=self.box(comp);center=[(a+c)/2,(b+d)/2]
            if np.dot(np.subtract(p,center),np.subtract(q,p))<0:return None
            samples=np.rint(np.linspace(p,q,max(2,math.ceil(math.dist(p,q))+1))).astype(int)
            if not all(self.red[max(0,y-2):min(self.h,y+3),max(0,x-2):min(self.w,x+3)].any() for x,y in samples):return None
            return cid
        grouped=defaultdict(set)
        for ref,p in pp.items():
            cid=bind_contact(ref,p)
            if cid:grouped[cid].add(ref)
        count=0
        for cid,refs in sorted(grouped.items()):
            owners={r.rsplit('.',1)[0] for r in refs}
            if len(refs)<2 or len(owners)<2:continue
            kinds={self.out['components'][o]['type'] for o in owners}
            rails={('GND' if self.out['components'][o]['type']=='gnd' else str(self.out['components'][o].get('value')))
                   for o in owners if self.out['components'][o]['type'] in RAILS}
            if len(rails)>1:
                self.withheld.append({'rule':'colored_wire_network','color_network':cid,'reason':'conflicting_rail_identities','members':sorted(refs)})
                continue
            if any(members(net)==refs and net.get('edges') for net in self.out['nets'].values()):
                continue  # Preserve a complete existing branch and its geometry.
            if count>=self.cfg.get('maximum_color_networks',128):
                raise ValueError('color_network_budget_exhausted')
            affected=[]
            for nid,net in list(self.out['nets'].items()):
                before=members(net)
                if not before&refs:continue
                affected.append(nid);remain=before-refs
                if remain:net['hyperGraph']='('+','.join(sorted(remain))+')'
                else:del self.out['nets'][nid]
            nid=next((n for n in sorted(affected) if n not in self.out['nets']),None) or 'IMAGE_'+cid
            if nid in self.out['nets']:raise ValueError('image_network_identity_collision')
            edges={}
            for e in nets[cid]['edges']:
                for a,b in zip(e['polyline'],e['polyline'][1:]):
                    if math.dist(a,b)<2:continue
                    edges['edge_'+str(len(edges)+1)]=[{'x':float(p[0]),'y':float(self.h-p[1])} for p in (a,b)]
            if not edges:raise ValueError('empty_color_branch')
            self.out['nets'][nid]={'hyperGraph':'('+','.join(sorted(refs))+')','edges':edges}
            self.events.append({'rule':'colored_wire_network','color_network':cid,'network_id':nid,
                                'members':sorted(refs),'replaced_networks':affected,'edge_count':len(edges)})
            count+=1
        self.graph_evidence=evidence

    def direct_red_pair(self,first,second):
        def get(ref):
            o,k=ref.rsplit('.',1);return self.point(self.out['pins'][o]['pin_'+k])
        a,b=get(first),get(second)
        if math.dist(a,b)>85 or min(abs(a[0]-b[0]),abs(a[1]-b[1]))>4:return False
        for ref,p,q in [(first,a,b),(second,b,a)]:
            owner=ref.rsplit('.',1)[0];aa,bb,cc,dd=self.box(self.out['components'][owner])
            center=[(aa+cc)/2,(bb+dd)/2]
            if np.dot(np.subtract(p,center),np.subtract(q,p))<0:return False
        samples=np.rint(np.linspace(a,b,max(2,math.ceil(math.dist(a,b))+1))).astype(int)
        if any(not(0<=x<self.w and 0<=y<self.h) for x,y in samples):return False
        if sum(self.red[max(0,y-2):min(self.h,y+3),max(0,x-2):min(self.w,x+3)].any() for x,y in samples)/len(samples)<.96:return False
        ids=[nid for nid,n in self.out['nets'].items() if {first,second}&members(n)]
        combined={first,second}|set().union(*(members(self.out['nets'][nid]) for nid in ids))
        identities={('GND' if self.out['components'][r.rsplit('.',1)[0]]['type']=='gnd'
                     else str(self.out['components'][r.rsplit('.',1)[0]].get('value')))
                    for r in combined if self.out['components'][r.rsplit('.',1)[0]]['type'] in RAILS}
        if len(identities)>1:return False
        refs={first,second};edges={}
        for nid in ids:
            net=self.out['nets'].pop(nid);refs|=members(net)
            for edge in net['edges'].values():edges['edge_'+str(len(edges)+1)]=edge
        edges['edge_'+str(len(edges)+1)]=[{'x':float(p[0]),'y':float(self.h-p[1])} for p in (a,b)]
        nid=ids[0] if ids else 'RED_'+first.replace('.','_')
        self.out['nets'][nid]={'hyperGraph':'('+','.join(sorted(refs))+')','edges':edges}
        self.events.append({'rule':'outward_red_stem','members':[first,second],'points':[a,b]})
        return True

    def connect_red_stems(self):
        # Arrow captions can have a red-only connection to a two-terminal part.
        labels=[e['owner'] for e in self.events if e['rule']=='arrow_caption_identity']
        for owner in labels:
            first=owner+'.1';a=self.point(self.out['pins'][owner]['pin_1']);choices=[]
            for o,ps in self.out['pins'].items():
                if self.out['components'][o]['type'] not in {'r','c','l'}:continue
                for k,p in ps.items():
                    q=self.point(p)
                    if math.dist(a,q)<=85 and min(abs(a[0]-q[0]),abs(a[1]-q[1]))<=4:choices.append((math.dist(a,q),o+'.'+k[4:]))
            for _,second in sorted(choices):
                if self.direct_red_pair(first,second):break
        refs=[o+'.'+k[4:] for o,ps in self.out['pins'].items() for k,p in ps.items()
              if self.out['components'][o]['type'] not in RAILS and str(p.get('pinname','')).strip().upper() not in NC]
        for ref in refs:
            owner,key=ref.rsplit('.',1);p=self.point(self.out['pins'][owner]['pin_'+key]);choices=[]
            if any(ref in members(n) and any(self.out['components'][s.rsplit('.',1)[0]]['type']=='gnd' for s in members(n)) for n in self.out['nets'].values()):continue
            for go,gc in self.out['components'].items():
                if gc['type']!='gnd':continue
                a,b,c,d=self.box(gc);center=[(a+c)/2,(b+d)/2]
                axis=int(abs(p[1]-center[1])>=abs(p[0]-center[0]))
                if abs(p[1-axis]-center[1-axis])>4 or math.dist(p,center)>85:continue
                anchor=p.copy();anchor[axis]=[a,b][axis]-4 if p[axis]<center[axis] else [c,d][axis]+4
                if (p[axis]-center[axis])*(anchor[axis]-center[axis])<0:continue
                q=np.rint(np.linspace(p,anchor,max(2,math.ceil(math.dist(p,anchor))+1))).astype(int)
                if any(not(0<=x<self.w and 0<=y<self.h) for x,y in q):continue
                if sum(self.red[max(0,y-2):min(self.h,y+3),max(0,x-2):min(self.w,x+3)].any() for x,y in q)/len(q)<.96:continue
                choices.append((math.dist(p,center),go,anchor))
            choices.sort()
            if not choices or len(choices)>1 and choices[1][0]-choices[0][0]<8:continue
            _,go,anchor=choices[0]
            saved=deepcopy(self.out);event_count=len(self.events)
            if not self.out['pins'].get(go):self.out['pins'][go]={'pin_1':self.export_pin(anchor)}
            if not any(go+'.1' in members(n) for n in self.out['nets'].values()):self.out['pins'][go]['pin_1']=self.export_pin(anchor)
            if not self.direct_red_pair(ref,go+'.1'):
                self.out=saved;del self.events[event_count:]
            else:self.events.append({'rule':'ground_outside_body_contact','owner':go,'point':anchor})

    def prune_fragments(self):
        for nid,net in list(self.out['nets'].items()):
            refs=members(net);owners={r.rsplit('.',1)[0] for r in refs}
            if net['edges'] or len(owners)!=1:continue
            owner=next(iter(owners));comp=self.out['components'][owner];a,b,c,d=self.box(comp)
            ps=[self.point(self.out['pins'][owner]['pin_'+r.rsplit('.',1)[1]]) for r in refs]
            no_green=all(not self.green[max(0,round(y)-7):min(self.h,round(y)+8),max(0,round(x)-7):min(self.w,round(x)+8)].any() for x,y in ps)
            inside=all(a-3<=x<=c+3 and b-3<=y<=d+3 for x,y in ps)
            if comp['type'] in {'box','ic'} and len(refs)==1 and no_green or comp['type'] in {'r','c','l'} and inside:
                del self.out['nets'][nid]
                self.events.append({'rule':'zero_wire_fragment','network_id':nid,'members':sorted(refs)})

    def run(self):
        if self.enabled('opposed_inductor_endpoint'):self.recover_inductors()
        if self.enabled('arrow_caption_identity'):self.arrow_labels()
        claimed=self.split_led_clusters() if self.enabled('independent_led_symbols') else set()
        if self.enabled('unidentified_capacitor_identity'):claimed |= recover_capacitors(self)
        if self.enabled('single_filled_led_identity'):claimed |= recover_single_leds(self)
        if self.enabled('ground_owner_arbitration'):recover_grounds(self)
        if self.enabled('observed_resistor_terminals'):recover_resistor_contacts(self)
        if self.enabled('visible_terminal_pair'):
            for owner,comp in sorted(self.out['components'].items()):
                if owner not in claimed and comp['type'] in {'r','c','l','led'} and owner in self.rmap:
                    self.normalize_passive(owner,axis=1 if comp['type']=='led' else None)
        if self.enabled('typed_power_contact') or self.enabled('numeric_power_caption'):self.recover_power()
        if self.enabled('colored_wire_network'):self.rebuild_color_networks()
        if self.enabled('outward_red_stem'):self.connect_red_stems()
        if self.enabled('mixed_color_ground_branch'):recover_ground_branches(self)
        if self.enabled('image_no_connect_marker'):prune_no_connect_nets(self)
        if self.enabled('zero_wire_fragment'):self.prune_fragments()
        return self.out
