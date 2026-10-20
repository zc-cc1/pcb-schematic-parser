"""Bind observed text to owned learned pin points; never infer a numbering order."""
from collections import Counter, defaultdict
from copy import deepcopy
import math
import re

import numpy as np

from .component_text import bbox, center, choose_global, iou


def contexts(ownership, cfg):
    components={c['detection_index']:c for c in ownership['components']}
    output={}
    for pin in ownership['pin_candidates']:
        if pin['owner_status']!='assigned':
            continue
        component=components[pin['owner_detection_index']]
        x,y=pin['point']
        x1,y1,x2,y2=component['bbox_top_left']
        side=min([('left',abs(x-x1)),('right',abs(x-x2)),('top',abs(y-y1)),('bottom',abs(y-y2))],key=lambda r:r[1])[0]
        output[pin['candidate_id']]={'side':side,'point':pin['point'],'component':component,
                                     'tangent_limit':float(cfg['tangent_limit_px'])}
    # Dense rows get narrower matching gates; no numbering is derived from order.
    for cid,context in output.items():
        axis=1 if context['side'] in {'left','right'} else 0
        nearby=[abs(context['point'][axis]-other['point'][axis]) for oid,other in output.items()
                if oid!=cid and other['component']['detection_index']==context['component']['detection_index']
                and other['side']==context['side'] and abs(context['point'][axis]-other['point'][axis])>2]
        if nearby:
            context['tangent_limit']=max(cfg['minimum_tangent_limit_px'],min(context['tangent_limit'],min(nearby)*cfg['dense_pin_spacing_fraction']))
    return output


def text_tokens(texts):
    """Keep observed spelling alternatives and their actual source geometry."""
    tokens=[]
    seen=set()
    for row in texts:
        rb=row.get('bbox') or bbox(row['polygon'])
        observations=[{'text':row['text'],'confidence':row['confidence'],'bbox':rb,'view':row.get('view','external')}]+row.get('observations',[])
        for observation in observations:
            source=str(observation['text'])
            box=list(map(float,observation.get('bbox',rb)))
            confidence=float(observation['confidence'])
            if len(box)!=4 or not all(math.isfinite(v) for v in box+[confidence]) or box[2]<=box[0] or box[3]<=box[1]:
                raise ValueError('Invalid OCR geometry/confidence')
            words=list(re.finditer(r'\S+',source))
            for word in words:
                tb=box.copy()
                approximate=len(words)>1
                if approximate:
                    axis=1 if box[3]-box[1]>1.8*(box[2]-box[0]) else 0
                    length=box[axis+2]-box[axis]
                    tb[axis],tb[axis+2]=box[axis]+length*word.start()/len(source),box[axis]+length*word.end()/len(source)
                key=(word.group(),tuple(round(v,2) for v in tb),round(confidence,5))
                if key in seen:
                    continue
                seen.add(key)
                tokens.append({'token_id':f'PT_{len(tokens)+1:05d}','text':word.group(),'bbox':tb,
                               'confidence':confidence,'source_text':source,'source_view':observation.get('view',row.get('view','external')),
                               'approximate_split_geometry':approximate})
    return tokens


def local_coordinates(context,box):
    cx,cy=center(box)
    x,y=context['point']
    x1,y1,x2,y2=context['component']['bbox_top_left']
    side=context['side']
    if side=='left':
        return cy-y,cx-x1
    if side=='right':
        return cy-y,x2-cx
    if side=='top':
        return cx-x,cy-y1
    return cx-x,y2-cy


def candidate_cost(token,context,field,cfg,blocked):
    if token['confidence']<cfg['min_confidence']:
        return None
    if any(iou(token['bbox'],b)>=.45 for b in blocked):
        return None
    text=token['text']
    tangent,inward=local_coordinates(context,token['bbox'])
    if abs(tangent)>context['tangent_limit']:
        return None
    component=context['component']
    if field=='number':
        decimal=re.fullmatch(r'\d{1,4}',text)
        ball=component['type'] in cfg['ball_number_component_types'] and re.fullmatch(r'[A-Z]{1,2}\d{1,3}',text)
        if not (decimal or ball) or not -cfg['number_outer_band_px']<=inward<=cfg['number_inner_band_px']:
            return None
    else:
        # A header prints its pins as bare digits inside the body ("1 2 3"), and the reference
        # uses those digits as both the name and the number.  The letter requirement would drop
        # them, so a pure 1-3 digit token is allowed as a name when that is switched on, with a
        # tighter inward band so it cannot pick up a value such as "1.5uH" elsewhere.
        digit_name = bool(cfg.get('allow_digit_pin_names')) and re.fullmatch(r'\d{1,3}', text)
        if component['type'] not in cfg['name_component_types'] or not (re.search(r'[A-Za-z]',text) or digit_name):
            return None
        if digit_name:
            cfg = {**cfg, 'name_max_inward_px': float(cfg.get('digit_name_max_inward_px', 40))}
        if not re.fullmatch(r'[A-Za-z0-9_+./~!#()\[\]-]+',text) or not cfg['name_min_inward_px']<=inward<=cfg['name_max_inward_px']:
            return None
        cx,cy=center(token['bbox'])
        x1,y1,x2,y2=component['bbox_top_left']
        if not x1<=cx<=x2 or not y1<=cy<=y2:
            return None
    return abs(tangent)+cfg['distance_cost_weight']*abs(inward)+cfg['confidence_cost_px']*(1-token['confidence'])


def spatial_groups(tokens):
    groups=[]
    for token in tokens:
        for group in groups:
            if any(iou(token['bbox'],other['bbox'])>=.4 for other in group):
                group.append(token)
                break
        else:
            groups.append([token])
    return groups


def bind_pin_text(ownership,texts,cfg,component_records=None):
    result=deepcopy(ownership)
    context_map=contexts(result,cfg)
    tokens=text_tokens(texts)
    blocked=[record['fields'][field]['bbox'] for record in (component_records or [])
             for field in ('designator','Name','value') if field in record.get('fields',{})]
    groups=spatial_groups(tokens)
    pins=[p for p in result['pin_candidates'] if p['candidate_id'] in context_map]
    debug={'tokens':tokens,'rejected_assignments':[],'policy':'Independent observed number/name evidence; unique spatial text assignment; ambiguous fields remain null'}
    for pin in result['pin_candidates']:
        pin.update(number=None,pinname=None,number_status='owner_unresolved' if pin['owner_status']!='assigned' else 'no_reliable_text',
                   pinname_status='owner_unresolved' if pin['owner_status']!='assigned' else 'no_reliable_text',text_evidence={},text_candidates={})
    for field in ('number','pinname'):
        costs=np.full((len(pins),len(groups)),1e6)
        selected={}
        alternatives={}
        for i,pin in enumerate(pins):
            choices=[]
            context=context_map[pin['candidate_id']]
            pin['text_side']=context['side']
            for j,group in enumerate(groups):
                ranked=sorted([(score,token) for token in group
                               if (score:=candidate_cost(token,context,field,cfg,blocked)) is not None],key=lambda row:row[0])
                if not ranked:
                    continue
                score,chosen=ranked[0]
                different=[row for row in ranked[1:] if row[1]['text']!=chosen['text']]
                if different and different[0][0]-score<cfg['spelling_margin']:
                    choices.extend([{**token,'cost':s,'status':'conflicting_observed_spelling'} for s,token in ranked[:4]])
                    continue
                costs[i,j],selected[i,j]=score,chosen
                alternatives[i,j]=[{**token,'cost':s} for s,token in ranked[:4]]
                choices.extend(alternatives[i,j])
            pin['text_candidates'][field]=sorted(choices,key=lambda t:t['cost'])[:5]
        rejected=[]
        assignments=choose_global(costs,cfg['unmatched_cost'],cfg['ambiguity_margin'],rejected,field)
        debug['rejected_assignments'].extend(rejected)
        rejected_indices={r['detection_index'] for r in rejected}
        for i,pin in enumerate(pins):
            if i not in assignments:
                if pin['text_candidates'][field]:
                    pin[field+'_status']='ambiguous_text' if i in rejected_indices or any(t.get('status') for t in pin['text_candidates'][field]) else 'text_used_by_other_pin'
                continue
            j,score,stability=assignments[i]
            token=selected[i,j]
            pin[field]=token['text']
            pin[field+'_status']='observed_OCR_text'
            pin['text_evidence'][field]={**token,'cost':score,'global_assignment_margin':stability,'alternatives':alternatives[i,j]}
    for pin in pins:
        evidence=pin['text_evidence']
        if 'number' in evidence and 'pinname' in evidence and iou(evidence['number']['bbox'],evidence['pinname']['bbox'])>=.5:
            pin['number']=pin['pinname']=None
            pin['number_status']=pin['pinname_status']='same_glyph_number_name_role_ambiguous'
    # Multiple coordinates claiming the same physical number are unresolved.
    numbered=defaultdict(list)
    for pin in pins:
        if pin['number'] is not None:
            numbered[pin['owner'],pin['number']].append(pin)
    for group in numbered.values():
        if len(group)>1:
            for pin in group:
                pin['number']=None
                pin['number_status']='duplicate_number_on_same_component'
    counts=Counter({'number_observed':0,'name_observed':0,'number_and_name_observed':0})
    for pin in result['pin_candidates']:
        counts['number_observed']+=pin['number'] is not None
        counts['name_observed']+=pin['pinname'] is not None
        counts['number_and_name_observed']+=pin['number'] is not None and pin['pinname'] is not None
    result['pin_text_summary']={'candidate_count':len(result['pin_candidates']),'owned_candidates':len(pins),**dict(counts),
                                'number_status_counts':dict(Counter(p['number_status'] for p in result['pin_candidates'])),
                                'name_status_counts':dict(Counter(p['pinname_status'] for p in result['pin_candidates'])),
                                'coordinate_changes':0,'numbers_guessed':0,'nets_constructed':False}
    result['status']='pin_text_proposals_topology_pending'
    return result,debug


def local_views(ownership,binding,cfg):
    width,height=ownership['size']
    context_map=contexts(ownership,cfg)
    views=[]
    for pin in binding['pin_candidates']:
        if pin['candidate_id'] not in context_map:
            continue
        context=context_map[pin['candidate_id']]
        needs_name=context['component']['type'] in cfg['name_component_types'] and pin['pinname'] is None
        if pin['number'] is not None and not needs_name:
            continue
        x,y=context['point']
        x1,y1,x2,y2=context['component']['bbox_top_left']
        tangent=cfg['local_ocr']['tangent_padding_px']
        inward=cfg['name_max_inward_px'] if needs_name else cfg['number_inner_band_px']
        outward=cfg['number_outer_band_px']
        side=context['side']
        if side=='left': crop=[x1-outward,y-tangent,x1+inward,y+tangent]
        elif side=='right': crop=[x2-inward,y-tangent,x2+outward,y+tangent]
        elif side=='top': crop=[x-tangent,y1-outward,x+tangent,y1+inward]
        else: crop=[x-tangent,y2-inward,x+tangent,y2+outward]
        crop=[max(0,math.floor(crop[0])),max(0,math.floor(crop[1])),min(width,math.ceil(crop[2])),min(height,math.ceil(crop[3]))]
        if crop[2]>crop[0] and crop[3]>crop[1]:
            views.append({'candidate_id':pin['candidate_id'],'crop':crop,'side':side})
    return views[:cfg['local_ocr']['max_rois']],max(0,len(views)-cfg['local_ocr']['max_rois'])
