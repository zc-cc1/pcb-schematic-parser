"""Resolve a shared numeral token as one pin-and-network transaction."""
from copy import deepcopy
import math
import statistics

from .pin_geometry_recovery import DEFAULT as GEOMETRY_DEFAULT
from .pin_geometry_recovery import IC_TYPES, LeadEvidence, _members, side_of, vote
from .pin_relocation import isolate_number
from .power_wire_export import IC_TYPES as POWER_IC_TYPES, restore_power_wires

DEFAULT = {'max_groups': 2, 'max_ocr_calls': 16, 'min_confidence': .98,
           'min_views': 2, 'min_row_gap_px': 18}


def shared_number_pairs(document, threshold):
    pins=document['pin_candidates']
    for missing in sorted(pins,key=lambda p:p['candidate_id']):
        if (missing.get('owner_status')!='assigned' or missing.get('number') is not None
                or missing.get('number_status')!='text_used_by_other_pin'
                or not missing.get('pinname') or missing.get('text_hypothesis_primary_candidate_id')):continue
        tokens={t['token_id']:t for t in missing.get('text_candidates',{}).get('number',[])
                if t.get('token_id') and str(t.get('text','')).isdigit() and t.get('confidence',0)>=threshold
                and not t.get('approximate_split_geometry')}
        rivals=[]
        for other in pins:
            if (other['candidate_id']==missing['candidate_id'] or other.get('owner')!=missing.get('owner')
                    or other.get('owner_status')!='assigned' or not str(other.get('number','')).isdigit()
                    or not other.get('pinname') or other.get('text_hypothesis_primary_candidate_id')):continue
            shared=[tokens[t['token_id']] for t in other.get('text_candidates',{}).get('number',[])
                    if t.get('token_id') in tokens and t.get('confidence',0)>=threshold
                    and str(t.get('text'))==str(other['number']) and not t.get('approximate_split_geometry')]
            shared=[t for t in shared if str(t['text'])==str(other['number'])]
            if shared:rivals.append((other,shared))
        if len(rivals)==1:yield missing,*rivals[0]


def _attachment(candidate, graph):
    matches=[a for a in graph.get('pin_attachments',[]) if a['candidate_id']==candidate['candidate_id']]
    if len(matches)!=1:return None
    attachment=matches[0]
    source=next((n for n in graph['networks'] if n['network_id']==attachment.get('network_id')),None)
    if (attachment.get('attachment_status')!='attached' or attachment.get('point')!=candidate['point']
            or source is None or candidate['candidate_id'] not in source.get('pin_candidate_ids',[])
            or source.get('unresolved_junction_ids')):return None
    return attachment


def _restore_changed_power_branches(out, graph, model, recovered, cleanup):
    """Copy only qualified two-contact branches involving a recovered candidate."""
    restored,power_audit=restore_power_wires(out,graph,model,cleanup)
    physical={'RESTORED_'+n['network_id']:n for n in graph.get('physical_networks',[])}
    pmap={p['candidate_id']:p for p in model['pin_candidates']};accepted=[]
    for entry in power_audit['restored_networks']:
        nid=entry['network_id'];source=physical.get(nid,{})
        ids=source.get('pin_candidate_ids',[])
        relevant=[cid for cid in ids if cid in recovered]
        if len(ids)!=2 or len(relevant)!=1 or source.get('unresolved_junction_ids'):continue
        cid=relevant[0];p=pmap[cid];attachment=_attachment(p,graph)
        if not attachment or attachment.get('physical_network_id')!=source['network_id']:continue
        net=restored['nets'][nid];members=_members(net);ref=p['owner']+'.'+str(p['number'])
        if len(members)!=2 or ref not in members:continue
        # A new physical branch must not steal an existing IC membership.
        if any(member in _members(n) and out['components'].get(member.rsplit('.',1)[0],{}).get('type') in POWER_IC_TYPES
               for member in members for n in out['nets'].values()):continue
        missing_pins=[];valid=True
        for member in members:
            owner,number=member.rsplit('.',1);key='pin_'+number
            pin=restored['pins'].get(owner,{}).get(key)
            if pin is None:valid=False;break
            existing=out['pins'].get(owner,{}).get(key)
            if existing is not None and existing!=pin:valid=False;break
            if existing is None:missing_pins.append((owner,key,pin))
        if not valid:continue
        for owner,key,pin in missing_pins:out['pins'].setdefault(owner,{})[key]=deepcopy(pin)
        out['nets'][nid]=deepcopy(net);accepted.append(entry)
    return accepted


def reconcile_pin_pairs(target, document, graph, records, rgb, *, reader, cleanup=None, cfg=None):
    cfg={**DEFAULT,**(cfg or {})};height,width=rgb.shape[:2]
    out=deepcopy(target);model=deepcopy(document);pmap={p['candidate_id']:p for p in model['pin_candidates']}
    evidence=LeadEvidence(rgb,GEOMETRY_DEFAULT);by_owner={r['component_id']:r for r in records}
    audit={'enabled':True,'policy':'shared_number_joint_pairing_v1','groups':[],'changes':[],
           'network_changes':[],'restored_networks':[],'ocr_calls':0,'budget_exhausted':False}
    recovered=set();touched=set()
    def read(image,box,scale,rotation,recognition_only,view):
        if audit['ocr_calls']>=cfg['max_ocr_calls']:
            audit['budget_exhausted']=True;return []
        audit['ocr_calls']+=1
        return reader(image,box,scale,rotation,recognition_only,view)
    for missing,other,shared in shared_number_pairs(document,cfg['min_confidence']):
        owner=missing['owner'];old_number=str(other['number'])
        if (owner not in by_owner or out['components'].get(owner,{}).get('type') not in IC_TYPES
                or any(p['candidate_id'] in touched for p in [missing,other])):continue
        same=[p for p in document['pin_candidates'] if p.get('owner')==owner and p.get('owner_status')=='assigned']
        if sum(str(p.get('number'))==old_number for p in same)!=1:continue
        old=out['pins'].get(owner,{}).get('pin_'+old_number)
        if old is None or old['pinname']!=other['pinname']:continue
        old_point=[old['point']['x'],height-old['point']['y']]
        if math.dist(old_point,other['point'])>25 or abs(old_point[1]-other['point'][1])>3:continue
        a,b,c,d=out['components'][owner]['bbox'];found=evidence.frame([a,height-d,c,height-b])
        if not found:continue
        frame=found[0]
        if any(side_of(p['point'],frame)!='left' for p in [other,missing]):continue
        gap=missing['point'][1]-other['point'][1]
        if gap<cfg['min_row_gap_px']:continue
        # The disputed glyph lies above the missing row and below the wrong row.
        valid_tokens=[t for t in shared if -20<=(t['bbox'][1]+t['bbox'][3])/2-missing['point'][1]<=-2
                      and -30<=(t['bbox'][0]+t['bbox'][2])/2-frame[0]<=2
                      and 2<=(t['bbox'][1]+t['bbox'][3])/2-other['point'][1]<=20]
        if not valid_tokens:continue
        contacts=[_attachment(p,graph) for p in [other,missing]]
        if any(a is None for a in contacts):continue
        old_ref=owner+'.'+old_number;old_nid=contacts[0]['network_id']
        memberships=[nid for nid,n in out['nets'].items() if old_ref in _members(n)]
        if memberships!=[old_nid]:continue
        leads=evidence.leads(frame,'left');points=[]
        for p in [other,missing]:
            matches=[v for v in leads if abs(v['row']-p['point'][1])<=3]
            if len(matches)!=1:break
            points.append(matches[0]['point'])
        if len(points)!=2 or any(math.dist(p['point'],point)>25 for p,point in zip([other,missing],points)):continue
        if any(math.dist(points[1],[pin['point']['x'],height-pin['point']['y']])<=6
               for key,pin in out['pins'].get(owner,{}).items() if key!='pin_'+old_number):continue
        if len(audit['groups'])>=cfg['max_groups']:
            audit['budget_exhausted']=True;break
        group={'owner':owner,'shared_token_ids':sorted({t['token_id'] for t in valid_tokens}),'rows':[]}
        audit['groups'].append(group)
        column=statistics.median((t['bbox'][0]+t['bbox'][2])/2 for t in valid_tokens)
        offset=statistics.median((t['bbox'][1]+t['bbox'][3])/2-missing['point'][1] for t in valid_tokens)
        record=by_owner[owner]['bbox_top_left']
        crop=[max(0,math.floor(record[0]-32)),max(0,math.floor(record[1]-32)),min(width,math.ceil(record[2]+32)),min(height,math.ceil(record[3]+32))]
        whole=[]
        for scale in [1.,3.]:
            for rotation in [0,90]:whole.extend(read(rgb,crop,scale,rotation,False,f'pair_component_{owner}_s{scale}_r{rotation}'))
        for p,point in zip([other,missing],points):
            row=point[1];expected=row+offset
            number_box=[max(0,math.floor(column-10)),max(0,math.floor(expected-10)),min(width,math.ceil(column+10)),min(height,math.ceil(expected+10))]
            isolated=isolate_number(rgb,number_box);texts=[]
            for scale in [3.,5.]:texts.extend(read(isolated,[0,0,isolated.shape[1],isolated.shape[0]],scale,0,True,f"pair_glyph_{p['candidate_id']}_s{scale}"))
            number,support=vote([t for t in texts if t['text'].isdigit()],cfg)
            names=[t for t in whole if 3<(t['bbox'][0]+t['bbox'][2])/2-frame[0]<.55*(frame[2]-frame[0])
                   and frame[1]<(t['bbox'][1]+t['bbox'][3])/2<frame[3]
                   and abs((t['bbox'][1]+t['bbox'][3])/2-row)<=10 and any(ch.isalpha() for ch in t['text'])]
            name,name_support=vote(names,cfg)
            group['rows'].append({'candidate_id':p['candidate_id'],'old_number':p.get('number'),'old_name':p.get('pinname'),
                                  'number':number,'name':name,'number_support':support,'name_support':name_support,
                                  'number_readings':texts,'number_crop':number_box,'point':point})
        rows=group['rows']
        if any(not r['number'] or r['name']!=r['old_name'] for r in rows) or len({r['number'] for r in rows})!=2:
            group['status']='insufficient_joint_evidence';continue
        new_number=rows[0]['number'];new_ref=owner+'.'+new_number
        if (rows[1]['number']!=old_number or new_number==old_number
                or 'pin_'+new_number in out['pins'].get(owner,{})
                or any(str(p.get('number'))==new_number for p in same)
                or any(new_ref in _members(n) for n in out['nets'].values())):
            group['status']='incomplete_or_reserved_rekey';continue
        # No mutation occurs until both rows and the existing network are proven.
        out['pins'][owner].pop('pin_'+old_number);out['pins'][owner]['pin_'+new_number]=deepcopy(old)
        before=sorted(_members(out['nets'][old_nid]));after=sorted((set(before)-{old_ref})|{new_ref})
        out['nets'][old_nid]['hyperGraph']='('+','.join(after)+')'
        audit['network_changes'].append({'network_id':old_nid,'before':before,'after':after,'reason':'observed_number_rekey'})
        x,y=points[1];out['pins'][owner]['pin_'+old_number]={'pinname':rows[1]['name'],'point':{'x':float(x),'y':float(height-y)}}
        for row in rows:
            row['owner']=owner;p=pmap[row['candidate_id']];p['number']=row['number'];p['number_source']='observed_joint_pair_OCR'
            touched.add(row['candidate_id']);audit['changes'].append(row)
        recovered.add(missing['candidate_id']);group['status']='complete_observed_rekey_and_restore'
        new_nid=contacts[1]['network_id']
        if new_nid in out['nets']:
            before=sorted(_members(out['nets'][new_nid]));after=sorted(set(before)|{old_ref})
            out['nets'][new_nid]['hyperGraph']='('+','.join(after)+')'
            audit['network_changes'].append({'network_id':new_nid,'before':before,'after':after,'reason':'confirmed_missing_pin'})
    if recovered:
        audit['restored_networks']=_restore_changed_power_branches(out,graph,model,recovered,cleanup)
        for cid in recovered:
            p=pmap[cid];ref=p['owner']+'.'+str(p['number'])
            if not any(ref in _members(n) for n in out['nets'].values()):
                audit['network_changes'].append({'member':ref,'status':'withheld','reason':'no_qualified_exported_or_physical_power_branch'})
    return out,audit
