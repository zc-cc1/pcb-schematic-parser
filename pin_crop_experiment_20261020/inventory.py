"""Select missing-field component crops from predictions only."""
from collections import Counter
import argparse
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
WORK=ROOT.parent
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
parser=argparse.ArgumentParser()
parser.add_argument('--cases',nargs='+',default=['1','10','11','13'])
parser.add_argument('--output',default='inventory.json')
args=parser.parse_args()
by_hash={}
for folder in (WORK/'wire_validation_20261020/candidate_path/work').iterdir():
    graph=read(folder/'p/export/engineering_graph_proposals.json')
    by_hash[graph['image_sha256']]=folder
rows=[]
for case in args.cases:
    image=Path(f'D:/pcb_open_gt_in/{case}.png')
    digest=hashlib.sha256(image.read_bytes()).hexdigest()
    folder=by_hash[digest]
    pins=read(folder/'p/export/engineering_pin_text_result.json')
    records=read(folder/'p/export/completed_component_records.json')['records']
    base=read(WORK/f'wire_contact_implementation_20261020/predictions/{case}_Pred.json')
    components=[]
    for record in records:
        owner=record['component_id']
        if base['components'].get(owner,{}).get('type') not in {'box','ic'}:continue
        candidates=[p for p in pins['pin_candidates'] if p.get('owner')==owner
                    and p.get('owner_status')=='assigned' and not p.get('text_hypothesis_primary_candidate_id')]
        missing=[p for p in candidates if p.get('number') is None or p.get('pinname') is None]
        unexported=[p for p in candidates if p.get('number') is None or
                    'pin_'+str(p['number']) not in base['pins'].get(owner,{})]
        if not missing and not unexported:continue
        components.append({'owner':owner,'index':record['detection_index'],'bbox':record['bbox_top_left'],
            'type':record['type'],'candidate_count':len(candidates),'exported_pins':len(base['pins'].get(owner,{})),
            'missing_number':sum(p.get('number') is None for p in candidates),
            'missing_name':sum(p.get('pinname') is None for p in candidates),
            'candidate_ids':[p['candidate_id'] for p in candidates],
            'missing':[{'id':p['candidate_id'],'point':p['point'],'number':p.get('number'),
                        'name':p.get('pinname'),'number_status':p.get('number_status'),
                        'name_status':p.get('pinname_status')} for p in missing]})
    rows.append({'case':case,'image':str(image),'work':str(folder),'image_sha256':digest,
                 'components':components})
(ROOT/args.output).write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
for row in rows:
    print(json.dumps({'case':row['case'],'components':[{k:v for k,v in c.items() if k not in {'candidate_ids','missing'}}
                                                    for c in row['components']]},ensure_ascii=False))
