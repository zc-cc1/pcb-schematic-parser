"""Trace existing artifacts for cases 1 and 11; read-only on production."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
WORK=ROOT.parent
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
audit=read(WORK/'pin_detection_audit_20261020/audit.json')
cases={r['case']:r for r in audit['per_case'] if r['case'] in {'1','11'}}
out=[]
for case,info in cases.items():
    folder=Path(info['model_cache'])
    model=read(folder/'p/export/engineering_pin_text_result.json')
    records=read(folder/'p/export/completed_component_records.json')['records']
    recordmap={r['component_id']:r for r in records}
    candidate_map={p['candidate_id']:p for p in model['pin_candidates']}
    files={}
    for pattern in ['*audit*.json','*pin*result.json']:
        for file in folder.rglob(pattern):
            if 'probabilities' not in file.parts:files[str(file.relative_to(folder))]=file
    strips=[p for name,p in files.items() if p.name=='pin_number_strip_audit.json']
    stage_pins=[(name,read(file)) for name,file in files.items() if 'pin' in file.name and 'result.json' in file.name]
    pred=read(Path(f'D:/pcb_open_gt_v37dots_b/final_results/{case}_Pred.json'))
    targets=[r for r in audit['misses'] if r['case']==case]
    component_ids={r['component'] for r in targets}
    details=[]
    for target in targets:
        ids={item['id'] for item in [target['candidate'],target['nearest_candidate']] if item}
        ids.update(p['candidate_id'] for p in model['pin_candidates'] if p.get('owner')==target['component']
                   and (str(p.get('number'))==target['key'][4:] or str(p.get('pinname'))==target['name']))
        evidence=[]
        for cid in sorted(ids):
            pin=candidate_map[cid]
            stages=[]
            for name,document in stage_pins:
                stage=next((p for p in document.get('pin_candidates',[]) if p.get('candidate_id')==cid),None)
                if stage:
                    stages.append({'file':name,'owner':stage.get('owner'),'number':stage.get('number'),
                                   'pinname':stage.get('pinname'),'number_status':stage.get('number_status'),
                                   'name_status':stage.get('pinname_status'),'point':stage.get('point')})
            strip=[]
            for file in strips:
                payload=read(file)
                for key,values in payload.items():
                    if isinstance(values,list):
                        strip.extend({'file':str(file.relative_to(folder)),'list':key,'item':v}
                                     for v in values if isinstance(v,dict) and v.get('candidate_id')==cid)
            evidence.append({'candidate':pin,'stages':stages,'number_strip':strip})
        details.append({'failure':target,'evidence':evidence})
    out.append({'case':case,'folder':str(folder),'stage_files':sorted(files),
        'components':{c:{'record':recordmap.get(c),'predicted':pred['components'].get(c),
                         'predicted_pins':pred['pins'].get(c)} for c in component_ids},
        'details':details})
(ROOT/'trace.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
for row in out:
    print('CASE',row['case'],'FILES',len(row['stage_files']))
    print('\n'.join(n for n in row['stage_files'] if 'strip' in n or 'library' in n or 'ownership' in n))
    for detail in row['details']:
        f=detail['failure']
        print(json.dumps({'pin':f"{f['component']}.{f['key'][4:]} {f['name']}",
                         'evidence':[{'id':e['candidate']['candidate_id'],'number':e['candidate'].get('number'),
                          'name':e['candidate'].get('pinname'),'number_source':e['candidate'].get('number_source'),
                          'library':e['candidate'].get('library_evidence'),
                          'strip':[{k:v for k,v in s['item'].items() if k not in ['readings']} for s in e['number_strip']]} for e in detail['evidence']]},ensure_ascii=False))
