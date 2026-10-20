"""Attribute current wire misses from frozen predictions; never performs inference."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parent
WORK=ROOT.parent
sys.path.insert(0,str(WORK/'pcb_parser_competition/scripts'))
from platform_wire_metrics import logical_nets,_assign_nets,_match_edges,score_wire
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
work_by_hash={}
for folder in (WORK/'wire_validation_20261020/candidate_path/work').iterdir():
    graph=read(folder/'p/export/engineering_graph_proposals.json')
    work_by_hash[graph['image_sha256']]=folder
result=[]
for case in sys.argv[1:] or ['5','9','10','13']:
    prediction=read(WORK/f'wire_implementation_20261020/cached_predictions/{case}_Pred.json')
    truth=read(next((WORK/'赛题六公开数据集/赛题六公开数据集/open_GT/open_GT'/case).glob('*.json')))
    pn,tn=logical_nets(prediction),logical_nets(truth)
    pairs={i:(j,s) for i,j,s in _assign_nets(pn,tn)}
    all_pred=tuple(e for n in pn for e in n.edges)
    all_truth=tuple(e for n in tn for e in n.edges)
    rows=[]
    for i,net in enumerate(tn):
        j,s=pairs.get(i,(None,0))
        matched=pn[j] if j is not None else None
        tp=_match_edges(matched.edges,net.edges,20) if matched else 0
        geometry=_match_edges(all_pred,net.edges,20)
        missing=net.vertices-(matched.vertices if matched else set())
        rows.append({'gt':net.source_ids,'gt_edges':len(net.edges),'tp':tp,'fn':len(net.edges)-tp,
                     'geometry_in_any_net':geometry,'pred':matched.source_ids if matched else [],
                     'pred_edges':len(matched.edges) if matched else 0,'pin_set_f1':round(s,4),
                     'missing_vertices':sorted(missing),
                     'extra_vertices':sorted(matched.vertices-net.vertices) if matched else []})
    digest=hashlib.sha256(Path(f'D:/pcb_open_gt_in/{case}.png').read_bytes()).hexdigest()
    folder=work_by_hash[digest]
    graph=read(folder/'p/export/engineering_graph_proposals.json')
    pins=read(folder/'p/export/engineering_pin_text_result.json')
    coverage=read(folder/'p/export/engineering_export_coverage.json')
    pmap={p['candidate_id']:p for p in pins['pin_candidates']}
    omitted=[]
    for net in graph['networks']:
        if net['network_id'] in prediction['nets']:continue
        omitted.append({'id':net['network_id'],'edges':len(net['edge_ids']),
            'unresolved_junctions':net['unresolved_junction_ids'],
            'power_names':net.get('power_names',[]),
            'members':[{'owner':pmap[c].get('owner'),'number':pmap[c].get('number'),
                        'pinname':pmap[c].get('pinname'),'status':pmap[c].get('owner_status')}
                       for c in net['pin_candidate_ids']]})
    block=score_wire(prediction,truth)
    row={'case':case,'wire':block,'pure_geometry_tp':_match_edges(all_pred,all_truth,20),
         'missing_nets':sorted(rows,key=lambda n:-n['fn']),
         'omitted_model_nets':omitted,'work':str(folder),
         'power_decisions':graph.get('power_merge_decisions',[]),
         'withheld_reasons':coverage.get('network_reason_counts')}
    result.append(row)
(ROOT/'miss_diagnosis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
for row in result:
    print(json.dumps({k:v for k,v in row.items() if k not in {'omitted_model_nets','power_decisions','missing_nets','work'}}))
    print(json.dumps({'case':row['case'],'top_losses':row['missing_nets'][:4]},ensure_ascii=False))
