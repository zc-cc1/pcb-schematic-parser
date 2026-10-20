import json
from pathlib import Path
import hashlib
ROOT=Path(__file__).resolve().parents[1]
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
image=hashlib.sha256(Path('D:/pcb_open_gt_in/1.png').read_bytes()).hexdigest()
for work in (ROOT/'wire_validation_20261020/candidate_path/work').iterdir():
    path=work/'p/export/engineering_graph_proposals.json'
    graph=read(path)
    if graph['image_sha256']==image:break
pins={p['candidate_id']:p for p in read(work/'p/export/engineering_pin_text_result.json')['pin_candidates']}
base=read(ROOT/'wire_validation_small_20261020/baseline/1_Pred.json')
print('primary attachment',next(a for a in graph['pin_attachments'] if a['candidate_id']=='PIN_CAND_0102'))
print('LOG9',base['nets']['LOG_NET_CAND_00009'])
print('decisions',graph.get('power_merge_decisions'))
for site in graph.get('power_sites',[]):
    if site.get('input_network_id')=='LOG_NET_CAND_00009':
        print('existing rail site',site)
for n in graph['networks']:
    if n['network_id']=='LOG_NET_CAND_00009' or n.get('power_logical_merge'):
        print('logical',n)
        if n['network_id']=='LOG_NET_CAND_00009':
            print('existing power pins',[(c,pins[c],base['components'].get(pins[c].get('owner'))) for c in n['pin_candidate_ids'] if (base['components'].get(pins[c].get('owner')) or {}).get('type') in {'gnd','v','vcc'}])
for n in graph['physical_networks']:
    if n['network_id'] in {'NET_CAND_00028','NET_CAND_00052','NET_CAND_00065','NET_CAND_00066'}:
        print('physical',n)
        print('pins',[(c,pins[c].get('owner'),pins[c].get('number')) for c in n['pin_candidate_ids']])
