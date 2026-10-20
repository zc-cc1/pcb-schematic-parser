import json
from pathlib import Path
import sys
sys.path.insert(0,'D:/pcb5_submission/src')
from pcb_parser.pin_geometry_recovery import _segment_distance
ROOT=Path(__file__).resolve().parent;WORK=ROOT.parent
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
item=next(r for r in read(WORK/'pin_detection_audit_20261020/audit.json')['per_case'] if r['case']=='2')
graph=read(Path(item['model_cache'])/'p/export/engineering_graph_proposals.json')
base=read(WORK/'pin_geometry_implementation_20261020/deployed/2_Pred.json')
change=read(ROOT/'generation_audit.json')['changes'][0]
point=change['proposal']['point'];cid=change['source_candidate_id'];edges={e['edge_id']:e for e in graph['edges']}
print('OLD_ATTACHMENT',json.dumps([a for a in graph['pin_attachments'] if a['candidate_id']==cid]))
for net in graph['networks']:
    ds=[]
    for eid in net.get('edge_ids',[]):
        poly=edges[eid]['polyline']
        ds.extend(_segment_distance(point,a,b) for a,b in zip(poly,poly[1:]))
    if ds and min(ds)<6:
        nid=net['network_id']
        print('NEAR_NET',json.dumps(net,ensure_ascii=False))
        print('DISTANCE',min(ds),'EXPORTED',json.dumps(base['nets'].get(nid),ensure_ascii=False))
print('EXPORT_MEMBERSHIPS',json.dumps({nid:n['hyperGraph'] for nid,n in base['nets'].items() if 'U2.' in n['hyperGraph']}))
