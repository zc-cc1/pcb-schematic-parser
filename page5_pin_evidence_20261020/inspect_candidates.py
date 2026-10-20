import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent;WORK=ROOT.parent
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
item=next(r for r in read(WORK/'pin_detection_audit_20261020/audit.json')['per_case'] if r['case']=='5')
stage=Path(item['model_cache'])/'p';model=read(stage/'export/engineering_pin_text_result.json');graph=read(stage/'export/engineering_graph_proposals.json')
records=read(stage/'export/completed_component_records.json')['records'];pred=read(WORK/'pin_pairing_implementation_20261020/deployed/5_Pred.json')
print('STAGE',stage)
for owner in ['U1','U3']:
    print('RECORD',json.dumps(next(r for r in records if r['component_id']==owner),ensure_ascii=False))
    for p in model['pin_candidates']:
        if p.get('owner')!=owner:continue
        print('CANDIDATE',json.dumps({k:v for k,v in p.items() if k not in {'owner_candidates','text_candidates','text_evidence'}},ensure_ascii=False))
        print('TEXT',json.dumps({k:[{a:b for a,b in t.items() if a not in {'alternatives','polygon'}} for t in v] for k,v in p.get('text_candidates',{}).items()},ensure_ascii=False))
        print('ATTACH',json.dumps([a for a in graph['pin_attachments'] if a['candidate_id']==p['candidate_id']],ensure_ascii=False))
    print('NETS',json.dumps({n:v for n,v in pred['nets'].items() if owner+'.' in v['hyperGraph']},ensure_ascii=False))
