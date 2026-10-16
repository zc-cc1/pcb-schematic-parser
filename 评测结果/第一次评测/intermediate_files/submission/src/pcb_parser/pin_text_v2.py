"""Observed pin text with dense-row gates and competing tip hypotheses."""
from collections import Counter,defaultdict
from copy import deepcopy
import math
import re

import numpy as np

from .component_text import center,iou,choose_global
from .pin_text import contexts,text_tokens,candidate_cost,local_coordinates


def tokens_with_views(texts):
    tokens=[];seen=set()
    for row in texts:
        for obs in [row]+row.get('observations',[]):
            text=str(obs['text']);box=obs.get('bbox',row['bbox']);view=obs.get('view',row.get('view','external'))
            score=float(obs['confidence']);key=(text,tuple(round(float(v),2) for v in box),view,round(score,5))
            if key in seen:continue
            seen.add(key)
            source={'text':text,'bbox':box,'polygon':[[box[0],box[1]],[box[2],box[1]],[box[2],box[3]],[box[0],box[3]]],
                    'confidence':score,'view':view}
            parts=text_tokens([source])
            if len(text.split())>1 and re.search(r'[A-Za-z]',text) and re.fullmatch(r'[A-Za-z][A-Za-z0-9_+./~!#()\[\] \-]*',text):
                parts.append({'text':text,'bbox':list(map(float,box)),'confidence':score,'source_text':text,'source_view':view,
                              'approximate_split_geometry':False,'whole_multiword_name':True})
            for token in parts:
                tokens.append({**token,'token_id':f'PV2_{len(tokens)+1:05d}'})
    return tokens


def glyph_groups(tokens):
    """Complete-link spatial grouping prevents a wide crop bridging rows."""
    def same(a,b):
        ba,bb=a['bbox'],b['bbox'];ca,cb=center(ba),center(bb)
        return (iou(ba,bb)>=.4 and abs(ca[0]-cb[0])<=.55*min(ba[2]-ba[0],bb[2]-bb[0])+2
                and abs(ca[1]-cb[1])<=.55*min(ba[3]-ba[1],bb[3]-bb[1])+2)
    groups=[]
    for token in sorted(tokens,key=lambda t:-t['confidence']):
        for group in groups:
            if all(same(token,other) for other in group):group.append(token);break
        else:groups.append([token])
    return groups


def tip_hypotheses(ownership,cfg):
    context=contexts(ownership,cfg);groups=[]
    owned=[p for p in ownership['pin_candidates'] if p['candidate_id'] in context]
    def aligned(a,b):
        ca,cb=context[a['candidate_id']],context[b['candidate_id']]
        if a['owner_detection_index']!=b['owner_detection_index'] or ca['side']!=cb['side']:return False
        axis=1 if ca['side'] in {'left','right'} else 0
        return (abs(a['point'][axis]-b['point'][axis])<=cfg['tip_hypothesis_cluster_px']
                and abs(a['point'][1-axis]-b['point'][1-axis])<=cfg['tip_hypothesis_normal_px'])
    for pin in sorted(owned,key=lambda p:(-p['confidence'],p['candidate_id'])):
        for group in groups:
            if all(aligned(pin,p) for p in group):
                group.append(pin);break
        else:groups.append([pin])
    def primary_cost(pin):
        ctx=context[pin['candidate_id']];b=ctx['component']['bbox_top_left'];x,y=pin['point']
        boundary={'left':abs(x-b[0]),'right':abs(x-b[2]),'top':abs(y-b[1]),'bottom':abs(y-b[3])}[ctx['side']]
        return boundary,-pin['confidence'],pin['candidate_id']
    for group in groups:group.sort(key=primary_cost)
    secondary={p['candidate_id']:g[0]['candidate_id'] for g in groups for p in g[1:]}
    representatives=deepcopy(ownership)
    representatives['pin_candidates']=[p for p in representatives['pin_candidates'] if p['candidate_id'] not in secondary]
    return representatives,secondary


def cost_v2(token,context,field,cfg,blocked):
    options=cfg
    if field=='number' and context['component']['type'] not in cfg['name_component_types']:
        options={**cfg,'number_outer_band_px':cfg['passive_number_outer_band_px'],'number_inner_band_px':cfg['passive_number_inner_band_px']}
    score=candidate_cost(token,context,field,options,blocked)
    if score is None and field=='pinname' and token.get('whole_multiword_name'):
        # Whole observed line names must fit inside the body, so a line such as
        # "GND 15" straddling the external number column cannot become pinname.
        box=token['bbox'];db=context['component']['bbox_top_left'];tangent,inward=local_coordinates(context,box)
        if (context['component']['type'] in cfg['name_component_types'] and token['confidence']>=cfg['min_confidence']
            and db[0]<=box[0]<=box[2]<=db[2] and db[1]<=box[1]<=box[3]<=db[3]
            and abs(tangent)<=context['tangent_limit'] and cfg['name_min_inward_px']<=inward<=cfg['name_max_inward_px']
            and not any(iou(box,b)>=.45 for b in blocked)):
            score=abs(tangent)+cfg['distance_cost_weight']*abs(inward)+cfg['confidence_cost_px']*(1-token['confidence'])
    if score is None:return None
    _,inward=local_coordinates(context,token['bbox'])
    if field=='number' and re.fullmatch(r'[A-Z]{1,2}\d{1,3}',token['text']) and inward>cfg['ball_number_max_inward_px']:return None
    if field=='pinname' and token.get('approximate_split_geometry') and ' ' in token['source_text']:
        if re.fullmatch(r'[A-Za-z][A-Za-z0-9_+./~!#()\[\] \-]*',token['source_text']) and len(token['source_text'].split())>1:
            return None  # retain the complete observed name candidate instead
    b=token['bbox'];axis=1 if context['side'] in {'left','right'} else 0
    span=b[axis+2]-b[axis]
    if span>max(14,4*context['tangent_limit']):return None
    return score


def resolve_spelling(ranked,cfg):
    """Select only observed complete spellings; retain disagreements."""
    votes=defaultdict(set)
    for score,token in ranked:
        if token['confidence']>=cfg['min_vote_confidence']:votes[token['text']].add(token['source_view'])
    removed=[];filtered=[]
    for score,token in ranked:
        partial=next((full for full,views in votes.items() if full!=token['text'] and token['text'] in full
                      and len(views)>=cfg['full_spelling_min_views']
                      and len(votes.get(token['text'],set()))<len(views)
                      and len(token['text'])/len(full)>=.5),None)
        if partial:removed.append({'text':token['text'],'preferred_observed_complete_spelling':partial});continue
        filtered.append((score,token))
    if not filtered:return None,removed
    counts=sorted([(len(v),k) for k,v in votes.items()],reverse=True)
    dominant=counts[0][1] if counts and counts[0][0]>=cfg['vote_min_views'] and (len(counts)==1 or counts[0][0]-counts[1][0]>=cfg['vote_min_lead']) else None
    if dominant:
        supported=[r for r in filtered if r[1]['text']==dominant]
        if supported:return supported[0],removed
    best=filtered[0]
    if any(token['text']!=best[1]['text'] and score-best[0]<cfg['spelling_margin'] for score,token in filtered[1:]):return None,removed
    return best,removed


def bind_pin_text_v2(ownership,texts,cfg,component_records=None):
    result=deepcopy(ownership);working,secondary=tip_hypotheses(ownership,cfg);context=contexts(working,cfg)
    blocked=[r['fields'][f]['bbox'] for r in (component_records or []) for f in ['designator','Name','value'] if f in r.get('fields',{})]
    # Field boxes are global evidence exclusions; evaluate once per OCR token,
    # rather than repeating the same overlap checks for every pin and field.
    tokens=[t for t in tokens_with_views(texts) if not any(iou(t['bbox'],b)>=.45 for b in blocked)]
    groups=glyph_groups(tokens);blocked=[]
    selected_pins=[p for p in result['pin_candidates'] if p['candidate_id'] in context]
    debug={'tip_hypothesis_secondary_to_primary':secondary,'spelling_decisions':[],'rejected_assignments':[],
           'policy':'All input hypotheses retained; OCR allocated to strongest close-tip hypothesis; no field guessing'}
    for pin in result['pin_candidates']:
        status='owner_unresolved' if pin['owner_status']!='assigned' else 'duplicate_tip_hypothesis' if pin['candidate_id'] in secondary else 'no_reliable_text'
        pin.update(number=None,pinname=None,number_status=status,pinname_status=status,text_evidence={},text_candidates={})
        if pin['candidate_id'] in secondary:pin['text_hypothesis_primary_candidate_id']=secondary[pin['candidate_id']]
    for field in ['number','pinname']:
        costs=np.full((len(selected_pins),len(groups)),1e6);selected={};alternatives={};support={}
        for i,pin in enumerate(selected_pins):
            ctx=context[pin['candidate_id']];pin['text_side']=ctx['side'];choices=[]
            for j,group in enumerate(groups):
                ranked=sorted([(score,t) for t in group if (score:=cost_v2(t,ctx,field,cfg,blocked)) is not None],key=lambda r:r[0])
                if not ranked:continue
                chosen,discarded=resolve_spelling(ranked,cfg)
                if discarded:debug['spelling_decisions'].append({'candidate_id':pin['candidate_id'],'field':field,'partial_observations':discarded})
                alternatives[i,j]=[{**token,'cost':score} for score,token in ranked[:6]]
                choices.extend(alternatives[i,j])
                if chosen is None:
                    for t in alternatives[i,j]:t['status']='conflicting_observed_spelling'
                    continue
                costs[i,j],selected[i,j]=chosen
                support[i,j]=sorted({t['source_view'] for _,t in ranked if t['text']==chosen[1]['text'] and t['confidence']>=cfg['min_vote_confidence']})
            pin['text_candidates'][field]=sorted(choices,key=lambda r:r['cost'])[:6]
        rejected=[];assignments=choose_global(costs,cfg['unmatched_cost'],cfg['ambiguity_margin'],rejected,field)
        debug['rejected_assignments'].extend(rejected);rejected_indices={r['detection_index'] for r in rejected}
        for i,pin in enumerate(selected_pins):
            if i not in assignments:
                if pin['text_candidates'][field]:pin[field+'_status']='ambiguous_text' if i in rejected_indices or any(t.get('status') for t in pin['text_candidates'][field]) else 'text_used_by_other_pin'
                continue
            j,score,stability=assignments[i];token=selected[i,j]
            pin[field]=token['text'];pin[field+'_status']='observed_OCR_text'
            pin['text_evidence'][field]={**token,'cost':score,'global_assignment_margin':stability,'alternatives':alternatives[i,j],
                                        'observed_support_views':support[i,j]}
    numbered=defaultdict(list)
    for pin in selected_pins:
        evidence=pin['text_evidence']
        if 'number' in evidence and 'pinname' in evidence and iou(evidence['number']['bbox'],evidence['pinname']['bbox'])>=.5:
            pin['number']=pin['pinname']=None;pin['number_status']=pin['pinname_status']='same_glyph_number_name_role_ambiguous'
        if pin['number'] is not None:numbered[pin['owner'],pin['number']].append(pin)
    for group in numbered.values():
        if len(group)>1:
            for pin in group:pin['number']=None;pin['number_status']='duplicate_number_on_same_component'
    summary={'candidate_count':len(result['pin_candidates']),'owned_candidates':sum(p['owner_status']=='assigned' for p in result['pin_candidates']),
             'number_observed':sum(p['number'] is not None for p in result['pin_candidates']),
             'name_observed':sum(p['pinname'] is not None for p in result['pin_candidates']),
             'number_and_name_observed':sum(p['number'] is not None and p['pinname'] is not None for p in result['pin_candidates']),
             'secondary_tip_hypotheses_text_withheld':len(secondary),'coordinate_changes':0,'numbers_guessed':0,
             'pinname_guesses':0,'nets_constructed':False,
             'number_status_counts':dict(Counter(p['number_status'] for p in result['pin_candidates'])),
             'name_status_counts':dict(Counter(p['pinname_status'] for p in result['pin_candidates']))}
    result['pin_text_summary']=summary;result['status']='pin_text_v2_proposals_topology_pending'
    return result,debug


def adopt_pin_fields(baseline,experimental,cfg):
    """Keep v1 numbers/names; add only well-supported, unclaimed new names."""
    if cfg['number_policy']!='preserve_v1_observed_numbers' or cfg['name_policy']!='preserve_v1_then_fill_unclaimed_glyphs':
        raise ValueError('Unsupported adoption policy')
    before={p['candidate_id']:p for p in baseline['pin_candidates']};result=deepcopy(experimental)
    used=[(p['candidate_id'],p['text_evidence']['pinname']['bbox']) for p in baseline['pin_candidates'] if p.get('pinname') is not None]
    number_boxes=[p['text_evidence']['number']['bbox'] for p in baseline['pin_candidates'] if p.get('number') is not None]
    additions=[];withheld=[]
    for pin in result['pin_candidates']:
        old=before[pin['candidate_id']]
        if old['point']!=pin['point'] or old['owner_status']!=pin['owner_status']:raise ValueError('Baseline localization/ownership differs')
        for field in ['number','pinname']:
            if field=='number' or old.get('pinname') is not None:
                pin[field]=old.get(field);pin[field+'_status']=old[field+'_status']
                pin['text_candidates'][field]=deepcopy(old['text_candidates'].get(field,[]))
                if field in old['text_evidence']:pin['text_evidence'][field]=deepcopy(old['text_evidence'][field])
                else:pin['text_evidence'].pop(field,None)
        if old.get('pinname') is not None or pin.get('pinname') is None:continue
        evidence=pin['text_evidence']['pinname'];box=evidence['bbox'];reasons=[]
        if evidence['confidence']<cfg['new_name_min_confidence']:reasons.append('new_name_confidence_below_adoption_gate')
        if len(evidence.get('observed_support_views',[]))<cfg['new_name_min_views']:reasons.append('new_name_insufficient_distinct_views')
        if any(cid!=pin['candidate_id'] and iou(box,b)>=.45 for cid,b in used):reasons.append('glyph_already_used_by_v1_pin_name')
        if any(iou(box,b)>=.5 for b in number_boxes):reasons.append('glyph_already_used_as_pin_number')
        if reasons:
            withheld.append({'candidate_id':pin['candidate_id'],'text':pin['pinname'],'reasons':reasons,'evidence':evidence})
            pin['pinname']=None;pin['pinname_status']='v2_name_withheld_at_adoption';pin['text_evidence'].pop('pinname',None)
        else:
            pin['pinname_status']='observed_OCR_v2_missing_name_fill'
            additions.append({'candidate_id':pin['candidate_id'],'text':pin['pinname'],'evidence':evidence})
            used.append((pin['candidate_id'],box))
    summary=result['pin_text_summary']
    summary.update(number_observed=sum(p['number'] is not None for p in result['pin_candidates']),
                   name_observed=sum(p['pinname'] is not None for p in result['pin_candidates']),
                   number_and_name_observed=sum(p['number'] is not None and p['pinname'] is not None for p in result['pin_candidates']),
                   missing_names_added=len(additions),
                   number_status_counts=dict(Counter(p['number_status'] for p in result['pin_candidates'])),
                   name_status_counts=dict(Counter(p['pinname_status'] for p in result['pin_candidates'])))
    return result,{'number_policy':cfg['number_policy'],'name_policy':cfg['name_policy'],
                   'adopted_name_additions':additions,'withheld_name_additions':withheld,'old_numbers_changed':0,'old_nonempty_names_changed':0}
