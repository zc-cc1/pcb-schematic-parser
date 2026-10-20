import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent;WORK=ROOT.parent
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
item=next(r for r in read(WORK/'pin_detection_audit_20261020/audit.json')['per_case'] if r['case']=='5')
stage=Path(item['model_cache'])/'p';model=read(stage/'export/engineering_pin_text_result.json');graph=read(stage/'export/engineering_graph_proposals.json')
pred=read(WORK/'pin_pairing_implementation_20261020/deployed/5_Pred.json');h=graph['size'][1]
for e in graph['edges']:
    if e['edge_id'].startswith('W00019'):print('EDGE',json.dumps(e))
for owner,comp in pred['components'].items():
    if comp['type']!='gnd':continue
    for k,p in pred['pins'].get(owner,{}).items():
        x,y=p['point']['x'],h-p['point']['y']
        if 700<x<770 and 330<y<450:print('GND',owner,k,json.dumps(comp),json.dumps(p))
for p in model['pin_candidates']:
    x,y=p['point']
    if 700<x<770 and 330<y<450:
        print('CANDIDATE',json.dumps(p))
        print('ATTACH',json.dumps([a for a in graph['pin_attachments'] if a['candidate_id']==p['candidate_id']]))
print('OLD',json.dumps([a for a in graph['pin_attachments'] if a['candidate_id']=='PIN_CAND_0067']))
