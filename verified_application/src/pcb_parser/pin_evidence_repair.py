"""Restore observed full labels and replace isolated duplicate-row identities.

No expected device fields, case IDs or reference coordinates are encoded.
"""
from copy import deepcopy
import math
import re

from .pin_geometry_recovery import DEFAULT as GEOMETRY_DEFAULT
from .pin_geometry_recovery import IC_TYPES, LeadEvidence, _members, recover_image_leads, side_of, vote

DEFAULT={'max_components':3,'max_ocr_calls':24,'min_confidence':.98,'min_views':2,
         'max_collision_confidence':.8,'min_anchor_confidence':.9,'collision_radius_px':3}


def full_label_extends(full, short):
    match=re.fullmatch(r'([A-Za-z0-9_+./-]+)\(([A-Za-z0-9_+./-]+)\)',full)
    return bool(match and short.casefold() in {part.casefold() for part in match.groups()})


def _frame(target, owner, height, evidence):
    a,b,c,d=target['components'][owner]['bbox']
    found=evidence.frame([a,height-d,c,height-b])
    return found[0] if found else None


def _located_texts(tokens, point, frame, side, threshold):
    """Match text on the near interior half, preserving the original spelling."""
    if side not in {'left','right'}:return []
    selected=[]
    for token in tokens:
        if token.get('confidence',0)<threshold or token.get('approximate_split_geometry'):continue
        a,b,c,d=token['bbox'];x,y=(a+c)/2,(b+d)/2
        inward=x-frame[0] if side=='left' else frame[2]-x
        if frame[1]<y<frame[3] and 3<=inward<=.55*(frame[2]-frame[0]) and abs(y-point[1])<=10:
            selected.append(token)
    return selected


def _empty_singleton(candidate, target, graph):
    ref=candidate['owner']+'.'+str(candidate['number'])
    memberships=[nid for nid,net in target['nets'].items() if ref in _members(net)]
    attachments=[a for a in graph.get('pin_attachments',[]) if a['candidate_id']==candidate['candidate_id']]
    if len(memberships)!=1 or len(attachments)!=1:return None
    nid=memberships[0];attachment=attachments[0]
    source=next((n for n in graph['networks'] if n['network_id']==nid),None)
    if (attachment.get('attachment_status')!='attached' or attachment.get('network_id')!=nid
            or attachment.get('point')!=candidate['point'] or source is None
            or source.get('unresolved_junction_ids') or source.get('edge_ids')
            or set(source.get('pin_candidate_ids',[]))!={candidate['candidate_id']}
            or target['nets'][nid].get('edges') or _members(target['nets'][nid])!={ref}):return None
    return nid


def repair_observed_pin_evidence(target, document, graph, records, rgb, *, reader, cfg=None):
    cfg={**DEFAULT,**(cfg or {})};height,width=rgb.shape[:2]
    out=deepcopy(target);pins=document['pin_candidates'];by_owner={r['component_id']:r for r in records}
    evidence=LeadEvidence(rgb,GEOMETRY_DEFAULT)
    audit={'enabled':True,'policy':'observed_pin_evidence_v1','changes':[],'network_changes':[],
           'proposals':[],'ocr_calls':0,'budget_exhausted':False}
    selected=set();whole_cache={}
    def reserve(owner):
        if owner in selected:return True
        if len(selected)>=cfg['max_components']:
            audit['budget_exhausted']=True;return False
        selected.add(owner);return True
    def read(image,box,scale,rotation,recognition_only,view):
        if audit['ocr_calls']>=cfg['max_ocr_calls']:
            audit['budget_exhausted']=True;return []
        audit['ocr_calls']+=1
        return reader(image,box,scale,rotation,recognition_only,view)
    def whole(owner):
        if owner in whole_cache:return whole_cache[owner]
        box=by_owner[owner]['bbox_top_left']
        crop=[max(0,math.floor(box[0]-32)),max(0,math.floor(box[1]-32)),min(width,math.ceil(box[2]+32)),min(height,math.ceil(box[3]+32))]
        tokens=[]
        for scale in [1.,3.]:
            for rotation in [0,90]:tokens.extend(read(rgb,crop,scale,rotation,False,f'evidence_component_{owner}_s{scale}_r{rotation}'))
        whole_cache[owner]=tokens
        return tokens

    # A field at the same exported point as a stronger pin is suspect only
    # when its number came from a library and its sole net has no wire/member.
    for candidate in sorted(pins,key=lambda p:p['candidate_id']):
        owner=candidate.get('owner');number=str(candidate.get('number'));name=candidate.get('pinname')
        if (candidate.get('owner_status')!='assigned' or not number.isdigit() or not name
                or candidate.get('number_source')!='library_pin_name'
                or candidate.get('confidence',1)>=cfg['max_collision_confidence']
                or candidate.get('text_hypothesis_primary_candidate_id')
                or out['components'].get(owner,{}).get('type') not in IC_TYPES or owner not in by_owner):continue
        old=out['pins'].get(owner,{}).get('pin_'+number)
        if old is None or old['pinname']!=name:continue
        old_point=[old['point']['x'],height-old['point']['y']]
        if abs(old_point[1]-candidate['point'][1])>3 or math.dist(old_point,candidate['point'])>50:continue
        nid=_empty_singleton(candidate,out,graph)
        if nid is None:continue
        anchors=[]
        for key,pin in out['pins'].get(owner,{}).items():
            point=[pin['point']['x'],height-pin['point']['y']]
            if key=='pin_'+number or pin['pinname']==name or math.dist(point,old_point)>cfg['collision_radius_px']:continue
            sources=[p for p in pins if p.get('owner')==owner and str(p.get('number'))==key[4:]
                     and p.get('pinname')==pin['pinname'] and p.get('owner_status')=='assigned'
                     and not p.get('text_hypothesis_primary_candidate_id') and p.get('confidence',0)>=cfg['min_anchor_confidence']]
            if len(sources)==1 and abs(sources[0]['point'][1]-point[1])<=3:anchors.append(key)
        if len(anchors)!=1:continue
        frame=_frame(out,owner,height,evidence)
        if frame is None:continue
        old_side=side_of(old_point,frame)
        if old_side not in {'left','right'}:continue
        if not reserve(owner):continue
        focus={'components':{owner:deepcopy(out['components'][owner])},'pins':{owner:deepcopy(out['pins'][owner])},'nets':{}}
        _,proof=recover_image_leads(focus,document,graph,records,rgb,reader=read,cfg={
            'max_components':1,'max_ocr_calls':max(0,cfg['max_ocr_calls']-audit['ocr_calls']),
            'min_confidence':cfg['min_confidence'],'min_views':cfg['min_views']})
        if proof['budget_exhausted']:audit['budget_exhausted']=True
        proposals=[p for p in proof['proposals'] if p.get('status')=='existing_number'
                   and p.get('number')==number and p.get('name')==name
                   and p['side']!=old_side and math.dist(p['point'],old_point)>=frame[2]-frame[0]]
        entry={'owner':owner,'number':number,'name':name,'source_candidate_id':candidate['candidate_id'],
               'collision_with':anchors[0],'old_singleton_network':nid,'proof':proposals}
        audit['proposals'].append(entry)
        if len(proposals)!=1:entry['status']='no_unique_observed_counterpart';continue
        proposal=proposals[0];x,y=proposal['point'];entry['before']=deepcopy(old['point'])
        old['point']={'x':float(x),'y':float(height-y)};entry['after']=deepcopy(old['point'])
        entry['status']='replaced_collision_by_observed_opposite_lead';audit['changes'].append(entry)
        # A singleton with no wire contains no electrical connection to move.
        # Keep it isolated; never turn an ambiguous ground stroke into a net.
        audit['network_changes'].append({'member':owner+'.'+number,'status':'withheld',
            'reason':'isolated_membership_kept_until_counterpart_network_is_proven','network_id':nid})

    # Library-shortened labels may be expanded only to text actually observed
    # in this row, corroborated by fresh component crops at the same threshold.
    for candidate in sorted(pins,key=lambda p:p['candidate_id']):
        owner=candidate.get('owner');number=str(candidate.get('number'));name=candidate.get('pinname')
        if (candidate.get('owner_status')!='assigned' or not number.isdigit() or not name
                or candidate.get('pinname_source')!='library_pin_name' or candidate.get('text_hypothesis_primary_candidate_id')
                or out['components'].get(owner,{}).get('type') not in IC_TYPES or owner not in by_owner):continue
        old=out['pins'].get(owner,{}).get('pin_'+number)
        if old is None or old['pinname']!=name:continue
        point=[old['point']['x'],height-old['point']['y']]
        if math.dist(point,candidate['point'])>25:continue
        if sum(p.get('owner')==owner and str(p.get('number'))==number and not p.get('text_hypothesis_primary_candidate_id') for p in pins)!=1:continue
        frame=_frame(out,owner,height,evidence)
        if frame is None:continue
        side=side_of(point,frame)
        observed=_located_texts(candidate.get('text_candidates',{}).get('pinname',[]),point,frame,side,cfg['min_confidence'])
        observed=[{**t,'view':t.get('source_view','')} for t in observed if t.get('source_view') and full_label_extends(t['text'],name)]
        expected,support=vote(observed,cfg)
        if expected is None:continue
        if not reserve(owner):continue
        tokens=_located_texts(whole(owner),point,frame,side,cfg['min_confidence'])
        full,support=vote(tokens,cfg)
        entry={'owner':owner,'number':number,'source_candidate_id':candidate['candidate_id'],
               'before_name':name,'expected_observed_name':expected,'name_support':support}
        audit['proposals'].append(entry)
        if full!=expected:entry['status']='full_label_not_corroborated';continue
        old['pinname']=full;entry['after_name']=full;entry['status']='restored_observed_full_label';audit['changes'].append(entry)
    return out,audit
