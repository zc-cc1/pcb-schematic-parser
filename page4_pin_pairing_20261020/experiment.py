"""Jointly resolve a shared numeral token before reading reference labels."""
import copy
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parent;WORK=ROOT.parent;LIVE=Path('D:/pcb5_submission')
sys.dont_write_bytecode=True;sys.path.insert(0,str(LIVE));sys.path.insert(0,str(LIVE/'src'))
import agent
from pcb_parser.pin_geometry_recovery import DEFAULT, LeadEvidence, _members, side_of, vote
from pcb_parser.pin_relocation import isolate_number
from pcb_parser.pin_field_recovery import recognize
from pcb_parser.pin_text_ocr import create_engine
from pcb_parser.power_wire_export import restore_power_wires

read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
case='4';image=Path(f'D:/pcb_open_gt_in/{case}.png');rgb=np.asarray(Image.open(image).convert('RGB'));height,width=rgb.shape[:2]
digest=hashlib.sha256(image.read_bytes()).hexdigest()
item=next(r for r in read(WORK/'pin_detection_audit_20261020/audit.json')['per_case'] if r['case']==case)
stage=Path(item['model_cache'])/'p';document=read(stage/'export/engineering_pin_text_result.json')
graph=read(stage/'export/engineering_graph_proposals.json');assert graph['image_sha256']==digest==document['image_sha256']
records={r['component_id']:r for r in read(stage/'export/completed_component_records.json')['records']}
base=read(WORK/f'pin_relocation_implementation_20261020/deployed/{case}_Pred.json')
out=copy.deepcopy(base);model=copy.deepcopy(document);pmap={p['candidate_id']:p for p in model['pin_candidates']}
attachments={a['candidate_id']:a for a in graph['pin_attachments']};evidence=LeadEvidence(rgb,DEFAULT)
audit={'image_sha256':digest,'groups':[],'changes':[],'network_changes':[],'new_ocr_calls':0,'reused_ocr_calls':0}
cache=ROOT/'ocr';cache.mkdir(exist_ok=True);engine=None
def ocr(image,box,scale,rotation,rec,view):
    global engine
    patch=np.ascontiguousarray(image[box[1]:box[3],box[0]:box[2]])
    key=hashlib.sha256(patch.tobytes()+str((patch.shape,box,scale,rotation,rec)).encode()).hexdigest();dest=cache/(key+'.json')
    old=WORK/'pin_geometry_implementation_20261020/ocr_cache'/(key+'.json')
    if not dest.exists() and old.exists():write(dest,read(old))
    if dest.exists():audit['reused_ocr_calls']+=1;return [{**t,'view':view} for t in read(dest)]
    if engine is None:engine,_=create_engine({'onnx_threads':4})
    texts=recognize(engine,image,box,scale,rotation,rec,view);write(dest,texts);audit['new_ocr_calls']+=1;return texts

for missing in document['pin_candidates']:
    if missing.get('owner_status')!='assigned' or missing.get('number') is not None or missing.get('number_status')!='text_used_by_other_pin':continue
    owner=missing['owner']
    if owner not in records:continue
    same=[p for p in document['pin_candidates'] if p.get('owner')==owner and p.get('owner_status')=='assigned']
    missing_tokens={t['token_id'] for t in missing.get('text_candidates',{}).get('number',[]) if t.get('confidence',0)>=.98}
    rivals=[p for p in same if p['candidate_id']!=missing['candidate_id'] and p.get('number') is not None
            and missing_tokens&{t['token_id'] for t in p.get('text_candidates',{}).get('number',[]) if t.get('confidence',0)>=.98}]
    if len(rivals)!=1:continue
    other=rivals[0];a,b,c,d=base['components'][owner]['bbox'];found=evidence.frame([a,height-d,c,height-b])
    if not found:continue
    frame=found[0]
    if side_of(missing['point'],frame)!='left' or side_of(other['point'],frame)!='left':continue
    pair=[other,missing];observed=[]
    for p in same:
        if side_of(p['point'],frame)!='left':continue
        for t in p.get('text_candidates',{}).get('number',[]):
            cx=(t['bbox'][0]+t['bbox'][2])/2;cy=(t['bbox'][1]+t['bbox'][3])/2;delta=cy-p['point'][1]
            if t['confidence']>=.98 and -20<=delta<=-2:observed.append((cx,delta))
    if not observed:continue
    column=statistics.median(v[0] for v in observed);offset=statistics.median(v[1] for v in observed)
    box=records[owner]['bbox_top_left'];crop=[max(0,math.floor(box[0]-32)),max(0,math.floor(box[1]-32)),min(width,math.ceil(box[2]+32)),min(height,math.ceil(box[3]+32))]
    whole=[]
    for scale in [1.,3.]:
        for rotation in [0,90]:whole.extend(ocr(rgb,crop,scale,rotation,False,f'pair_component_{owner}_s{scale}_r{rotation}'))
    group={'owner':owner,'shared_token_ids':sorted(missing_tokens),'column':column,'number_offset':offset,'rows':[]}
    audit['groups'].append(group)
    for p in pair:
        lead=[v for v in evidence.leads(frame,'left') if abs(v['row']-p['point'][1])<=3]
        if len(lead)!=1:break
        row=lead[0]['row'];expected=row+offset
        number_box=[max(0,math.floor(column-10)),max(0,math.floor(expected-10)),min(width,math.ceil(column+10)),min(height,math.ceil(expected+10))]
        isolated=isolate_number(rgb,number_box);Image.fromarray(isolated).save(ROOT/f"{p['candidate_id']}_number.png")
        texts=[]
        for scale in [3.,5.]:texts.extend(ocr(isolated,[0,0,isolated.shape[1],isolated.shape[0]],scale,0,True,f"pair_glyph_{p['candidate_id']}_s{scale}"))
        number,support=vote([t for t in texts if t['text'].isdigit()],DEFAULT)
        names=[t for t in whole if 3<(t['bbox'][0]+t['bbox'][2])/2-frame[0]<.55*(frame[2]-frame[0])
               and abs((t['bbox'][1]+t['bbox'][3])/2-row)<=10 and any(ch.isalpha() for ch in t['text'])]
        name,name_support=vote(names,DEFAULT)
        group['rows'].append({'candidate_id':p['candidate_id'],'old_number':p.get('number'),'old_name':p.get('pinname'),
                              'number':number,'name':name,'number_support':support,'name_support':name_support,
                              'number_readings':texts,'number_crop':number_box,'point':lead[0]['point']})
    rows=group['rows']
    if len(rows)!=2 or any(not r['number'] or r['name']!=r['old_name'] for r in rows) or len({r['number'] for r in rows})!=2:
        group['status']='insufficient_joint_evidence';continue
    old_number=str(other['number'])
    if rows[1]['number']!=old_number or rows[0]['number']==old_number or 'pin_'+rows[0]['number'] in out['pins'].get(owner,{}):
        group['status']='not_a_complete_rekey_transaction';continue
    old=out['pins'].get(owner,{}).get('pin_'+old_number)
    if old is None or old['pinname']!=other['pinname']:group['status']='old_record_changed';continue
    old_ref=owner+'.'+old_number;new_ref=owner+'.'+rows[0]['number']
    a=attachments.get(other['candidate_id'],{});nid=a.get('network_id')
    current=[n for n,v in out['nets'].items() if old_ref in _members(v)]
    source=next((n for n in graph['networks'] if n['network_id']==nid),{})
    if (a.get('attachment_status')!='attached' or a.get('point')!=other['point'] or current!=[nid]
            or other['candidate_id'] not in source.get('pin_candidate_ids',[])
            or any(new_ref in _members(v) for v in out['nets'].values())):
        group['status']='unproven_existing_network_rekey';continue
    # Preserve the old VOUT coordinate and change the reference atomically.
    out['pins'][owner].pop('pin_'+old_number);out['pins'][owner]['pin_'+rows[0]['number']]=copy.deepcopy(old)
    before=sorted(_members(out['nets'][nid]));after=sorted((set(before)-{old_ref})|{new_ref})
    out['nets'][nid]['hyperGraph']='('+','.join(after)+')'
    audit['network_changes'].append({'network_id':nid,'before':before,'after':after})
    vin=rows[1];x,y=vin['point'];out['pins'][owner]['pin_'+vin['number']]={'pinname':vin['name'],'point':{'x':float(x),'y':float(height-y)}}
    for r in rows:
        pmap[r['candidate_id']]['number']=r['number'];pmap[r['candidate_id']]['number_source']='observed_joint_pair_OCR'
        audit['changes'].append(r)
    group['status']='complete_observed_rekey_and_restore'

# Re-run the existing tested power-branch exporter against the corrected field
# identities. It uses unchanged model edges/attachments, not synthetic wires.
out,power_audit=restore_power_wires(out,graph,model,agent.cleanup_wire_edges)
audit['power_export']=power_audit
agent.validate_target(out)
assert out['components']==base['components']
for nid,n in base['nets'].items():assert out['nets'][nid]['edges']==n['edges']
write(ROOT/'4_Pred.json',out);write(ROOT/'generation_audit.json',audit)
# Evaluation starts only after the image-only export has been frozen.
sys.path.insert(0,str(WORK/'pcb_parser_competition/scripts'))
import platform_metrics as metrics
from platform_wire_metrics import logical_nets,_assign_nets,score_wire
truth=read(next((WORK/'赛题六公开数据集/赛题六公开数据集/open_GT/open_GT/4').glob('*.json')))
def score(doc):
    pn,tn=logical_nets(doc),logical_nets(truth)
    conn=2*sum(s for _,_,s in _assign_nets(pn,tn))/max(1,len(pn)+len(tn))
    pins=metrics.score_pins(doc,truth,20)[0];wire=score_wire(doc,truth)
    comp=metrics.score_components(doc,truth)[0]
    return {'pin':pins,'connectivity':conn,'wire':wire,'score_100':100*(.3*comp['f1']+.25*pins['f1']+.35*conn+.1*wire['f1'])}
result={'before':score(base),'after':score(out),'new_ocr_calls':audit['new_ocr_calls'],'reused_ocr_calls':audit['reused_ocr_calls'],
        'new_nets':sorted(set(out['nets'])-set(base['nets'])),'external_api_calls':0,'full_model_runs':0,'production_modified':False}
write(ROOT/'results.json',result)
print(json.dumps(audit,ensure_ascii=False));print(json.dumps(result,ensure_ascii=False))
