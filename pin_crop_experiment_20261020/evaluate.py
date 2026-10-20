"""Prediction-only crop OCR adoption, followed by separate GT scoring."""
from collections import Counter
import copy
import inspect
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parent
WORK=ROOT.parent
sys.path.insert(0,'D:/pcb5_submission')
sys.path.insert(0,'D:/pcb5_submission/src')
sys.path.insert(0,str(WORK/'pcb_parser_competition/scripts'))
import agent
from adapter import to_target,validate_target
from pcb_parser import targeted_pin_ocr as recovery
from platform_wire_metrics import score_wire,logical_nets,_assign_nets
import platform_metrics_v3 as metrics

read=lambda p:json.loads(p.read_text(encoding='utf-8'))
save=lambda p,d:p.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
# Reuse all original adoption gates, changing only the component crop view prefix.
source=inspect.getsource(recovery.adopt_missing_fields)
old='local_prefix=f"missing_pin_{pin[\'candidate_id\']}_"'
new='local_prefix=f"component_crop_{pin[\'owner_detection_index\']}_"'
assert source.count(old)==1
namespace=dict(recovery.__dict__)
exec(compile(source.replace(old,new),'<isolated-component-crop-adoption>','exec'),namespace)
adopt=namespace['adopt_missing_fields']


def old_readings(row,pins):
    document=read(WORK/f"wire_validation_20261020/candidate_path/logs/{row['case']}_ocr_texts.json")
    observations=[]
    for text in document['texts']:
        observations.append(text)
        for other in text.get('observations',[]):
            box=other.get('bbox',text['bbox'])
            observations.append({**other,'bbox':box})
    observations+=recovery.existing_readings(pins)
    output=[];seen=set()
    for component in row['components']:
        x1,y1,x2,y2=component['bbox']
        for text in observations:
            box=text['bbox'];x=(box[0]+box[2])/2;y=(box[1]+box[3])/2
            if not x1-32<x<x2+32 or not y1-32<y<y2+32:continue
            view=str(text.get('view','external'))
            key=(component['index'],text['text'],tuple(box),view)
            if key in seen:continue
            seen.add(key)
            output.append({'text':text['text'],'bbox':box,'confidence':float(text['confidence']),
                           'view':f"component_crop_{component['index']}_existing_{view}"})
    return output


def recover_digit_header_names(base,pins,updated):
    """Use a literal digit as name only when existing pins establish that schema."""
    result=copy.deepcopy(updated)
    old={p['candidate_id']:p for p in pins['pin_candidates']}
    changes=[]
    for pin in result['pin_candidates']:
        if pin.get('pinname') is not None or not str(pin.get('number') or '').isdigit():continue
        if pin.get('number')==old[pin['candidate_id']].get('number'):continue
        known=base['pins'].get(pin.get('owner'),{})
        if len(known)<2 or any(not k[4:].isdigit() or p['pinname']!=k[4:] for k,p in known.items()):continue
        # The new number is directly OCR-observed; the established component uses
        # that same printed literal as its pin name. No pin ordering is inferred.
        pin['pinname']=str(pin['number'])
        pin['pinname_status']='crop_OCR_digit_with_existing_header_schema'
        changes.append({'candidate_id':pin['candidate_id'],'owner':pin['owner'],'field':'pinname',
                        'text':pin['pinname'],'evidence':'existing_exported_names_equal_numbers'})
    return result,changes


def export_recovered(base,pins,updated,graph,row):
    out=copy.deepcopy(base)
    original={p['candidate_id']:p for p in pins['pin_candidates']}
    additions={};audit=[]
    height=updated['size'][1]
    for pin in updated['pin_candidates']:
        old=original[pin['candidate_id']]
        if pin.get('number') is None or pin.get('pinname') is None or pin.get('owner_status')!='assigned':continue
        if pin.get('number')==old.get('number') and pin.get('pinname')==old.get('pinname'):continue
        owner=pin['owner'];key='pin_'+str(pin['number'])
        if owner not in out['components'] or key in out['pins'].get(owner,{}):continue
        if out['components'][owner]['type'] not in {'ic','box'}:continue
        x,y=pin['point']
        value={'pinname':pin['pinname'],'point':{'x':float(x),'y':float(height-y)}}
        additions[(owner,key)]=(pin,value)
    raw=to_target(read(WORK/f"wire_validation_20261020/candidate_path/component_results/{row['case']}_Pred.json"))
    snap={'components':copy.deepcopy(base['components']),'pins':raw['pins'],'nets':{}}
    for (owner,key),(_,value) in additions.items():snap['pins'].setdefault(owner,{})[key]=copy.deepcopy(value)
    bgr=cv2.imdecode(np.fromfile(row['image'],np.uint8),cv2.IMREAD_COLOR)
    agent.pin_lead_snap(snap,18,cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB),'color')
    for (owner,key),(pin,value) in additions.items():
        out['pins'].setdefault(owner,{})[key]=snap['pins'][owner][key]
        audit.append({'candidate_id':pin['candidate_id'],'owner':owner,'key':key,
                      'pinname':pin['pinname'],'point':out['pins'][owner][key]['point']})
    changed_ids={pin['candidate_id'] for pin,_ in additions.values()}
    physical={n['network_id']:n for n in graph.get('physical_networks',[])}
    restored={nid:physical[nid.removeprefix('RESTORED_')] for nid in out['nets']
              if nid.startswith('RESTORED_') and nid.removeprefix('RESTORED_') in physical}
    pmap={p['candidate_id']:p for p in updated['pin_candidates']}
    emap={e['edge_id']:e for e in graph['edges']}
    for net in graph['networks']:
        if not changed_ids.intersection(net['pin_candidate_ids']):continue
        destinations=[(net['network_id'],net)] if net['network_id'] in out['nets'] else []
        destinations.extend((nid,p) for nid,p in restored.items()
                            if changed_ids.intersection(p['pin_candidate_ids']))
        if not destinations and not net.get('power_logical_merge'):
            destinations=[(net['network_id'],net)]
        for nid,physical_net in destinations:
            refs=set()
            for cid in physical_net['pin_candidate_ids']:
                pin=pmap[cid];owner=pin.get('owner');number=pin.get('number')
                if number is not None and 'pin_'+str(number) in out['pins'].get(owner,{}):refs.add(f'{owner}.{number}')
            if not refs:continue
            if nid in out['nets']:
                previous={v.strip() for v in out['nets'][nid]['hyperGraph'].strip('()').split(',') if v.strip()}
                out['nets'][nid]['hyperGraph']='('+','.join(sorted(previous|refs))+')'
            else:
                edges={}
                for eid in physical_net['edge_ids']:
                    points=emap[eid]['polyline']
                    for a,b in zip(points,points[1:]):
                        edges[f'edge_{len(edges)+1}']=[{'x':float(a[0]),'y':float(height-a[1])},
                                                     {'x':float(b[0]),'y':float(height-b[1])}]
                local={'nets':{nid:{'hyperGraph':'('+','.join(sorted(refs))+')','edges':edges}}}
                agent.cleanup_wire_edges(local)
                out['nets'][nid]=local['nets'][nid]
    validate_target(out)
    assert out['components']==base['components']
    for owner,owner_pins in base['pins'].items():
        assert all(out['pins'][owner][key]==value for key,value in owner_pins.items())
    for nid,net in base['nets'].items():assert out['nets'][nid]['edges']==net['edges']
    return out,audit


def score(pred,truth):
    components,_=metrics.old.score_components(pred,truth)
    pins=metrics.old.score_pins(pred,truth,20)[0]
    pn,tn=logical_nets(pred),logical_nets(truth)
    connectivity=2*sum(s for _,_,s in _assign_nets(pn,tn))/max(1,len(pn)+len(tn))
    wire=score_wire(pred,truth)
    total=100*(.30*components['f1']+.25*pins['f1']+.35*connectivity+.10*wire['f1'])
    return {'component':components,'pin':pins,'connectivity':connectivity,'wire':wire,'score_100':total}


def main():
    inventory=read(ROOT/'inventory.json')
    collection=read(ROOT/'ocr_collection.json')
    cfg=read(Path('D:/pcb5_submission/configs/targeted_pin_ocr_v1.json'))
    cfg['component_types']=['ic','connector','block','opamp']
    binding=read(Path('D:/pcb5_submission/configs/pin_text_v2.json'))
    passive=read(Path('D:/pcb5_submission/configs/pin_output_rules_v1.json'))
    rows=[]
    for row in inventory:
        case=row['case'];folder=Path(row['work'])
        base=read(WORK/f'wire_contact_implementation_20261020/predictions/{case}_Pred.json')
        pins=read(folder/'p/export/engineering_pin_text_result.json')
        graph=read(folder/'p/export/engineering_graph_proposals.json')
        records=read(folder/'p/export/completed_component_records.json')['records']
        original=copy.deepcopy((base,pins,graph,records))
        old=old_readings(row,pins)
        views=[v for v in collection['views'] if v['case']==case]
        variants={'baseline':(base,{'changes':[],'withheld':[]},[])}
        target_ids={cid for c in row['components'] for cid in c['candidate_ids']}
        for name,scales in [('old_ocr_rebind',[]),('crop_1x',[1.0]),('crop_3x',[3.0]),('crop_combined',[1.0,3.0]),
                            ('old_ocr_header_schema',[]),('crop_header_schema',[1.0,3.0]),
                            ('old_ocr_single99_header',[]),('crop_single99_header',[1.0,3.0])]:
            texts=old+[text for view in views if view['scale'] in scales for text in view['texts']]
            options={**cfg,'adoption_min_views':1,'adoption_min_confidence':.99} if 'single99' in name else cfg
            updated,newgraph,audit=adopt(pins,graph,records,texts,target_ids,options,binding,passive)
            if 'header' in name:
                updated,header_audit=recover_digit_header_names(base,pins,updated)
                audit['changes'].extend(header_audit)
            prediction,export_audit=export_recovered(base,pins,updated,newgraph,row)
            variants[name]=(prediction,audit,export_audit)
        assert (base,pins,graph,records)==original
        # Annotations enter only after all predictions and export decisions are frozen.
        truth=read(next((WORK/'赛题六公开数据集/赛题六公开数据集/open_GT/open_GT'/case).glob('*.json')))
        result={'case':case,'variants':{}}
        for name,(prediction,audit,export_audit) in variants.items():
            scored=score(prediction,truth)
            assert scored['component']==score(base,truth)['component']
            destination=ROOT/name
            destination.mkdir(exist_ok=True)
            save(destination/f'{case}_Pred.json',prediction)
            result['variants'][name]={'metrics':scored,'adoption_audit':audit,'exported_new_pins':export_audit}
        rows.append(result)
    summary={name:{'pin_f1':sum(r['variants'][name]['metrics']['pin']['f1'] for r in rows)/len(rows),
                    'connectivity_f1':sum(r['variants'][name]['metrics']['connectivity'] for r in rows)/len(rows),
                    'wire_f1':sum(r['variants'][name]['metrics']['wire']['f1'] for r in rows)/len(rows),
                    'score_100':sum(r['variants'][name]['metrics']['score_100'] for r in rows)/len(rows),
                    'fields_added':sum(len(r['variants'][name]['adoption_audit']['changes']) for r in rows),
                    'pins_exported':sum(len(r['variants'][name]['exported_new_pins']) for r in rows)}
             for name in rows[0]['variants']}
    save(ROOT/'results.json',{'summary':summary,'per_case':rows,'local_ocr_calls':len(collection['views']),'external_api_calls':0,
                              'gt_used_for_crop_binding_or_export':False,'production_modified':False,
                              'binding_config':binding,'adoption_config':cfg})
    print(json.dumps({'summary':summary,'changed_cases':[{'case':r['case'],'variants':{n:{'pin':v['metrics']['pin'],
        'score':v['metrics']['score_100'],'fields':len(v['adoption_audit']['changes']),
        'new_pins':v['exported_new_pins']} for n,v in r['variants'].items()
        if v['metrics']!=r['variants']['baseline']['metrics']}} for r in rows
        if any(v['metrics']!=r['variants']['baseline']['metrics'] for v in r['variants'].values())]},ensure_ascii=False))


if __name__=='__main__':main()
