from collections import Counter
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
row=next(row for row in read(ROOT/'miss_diagnosis.json') if row['case']=='13')
folder=Path(row['work'])
graph=read(folder/'p/export/engineering_graph_proposals.json')
pins=read(folder/'p/export/engineering_pin_text_result.json')
base=read(ROOT.parent/'wire_implementation_20261020/cached_predictions/13_Pred.json')
attachments={a['candidate_id']:a for a in graph['pin_attachments']}
power=[]
for pin in pins['pin_candidates']:
    owner=pin.get('owner')
    component=base['components'].get(owner,{})
    if component.get('type') in {'gnd','v','vcc'}:
        a=attachments[pin['candidate_id']]
        power.append({'candidate':pin['candidate_id'],'owner':owner,'type':component['type'],
                      'value':component.get('value'),'number':pin.get('number'),
                      'duplicate_primary':pin.get('text_hypothesis_primary_candidate_id'),
                      'attachment':a['attachment_status'],'network':a.get('network_id'),
                      'existing_net':a.get('network_id') in base['nets']})
result={'case':'13','pin_owner_status':dict(Counter(p.get('owner_status') for p in pins['pin_candidates'])),
        'power_pins':power,'numbered_candidates':sum(p.get('number') is not None for p in pins['pin_candidates']),
        'pin_candidates':len(pins['pin_candidates'])}
(ROOT/'case13_bottleneck.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result,ensure_ascii=False))
