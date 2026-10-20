"""Check integrated production output against the previously verified candidate."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import sys

ROOT=Path(__file__).resolve().parent
WORK=ROOT.parent
parser=argparse.ArgumentParser()
parser.add_argument('--runtime',type=Path,default=ROOT/'runtime')
parser.add_argument('--report',default='staged_cached_check.json')
args=parser.parse_args()
sys.path.insert(0,str(args.runtime))
sys.path.insert(0,str(args.runtime/'src'))
sys.path.insert(0,str(WORK/'pcb_parser_competition/scripts'))
import agent
from adapter import validate_target
from pcb_parser.power_wire_export import restore_power_wires,POWER_TYPES,_members
from platform_wire_metrics import logical_nets,_assign_nets,score_wire
import platform_metrics_v3 as metrics

read=lambda p:json.loads(p.read_text(encoding='utf-8'))
def connectivity(pred,truth):
    pn,tn=logical_nets(pred),logical_nets(truth)
    return 2*sum(score for _,_,score in _assign_nets(pn,tn))/max(1,len(pn)+len(tn))

by_hash={}
for folder in (WORK/'wire_validation_20261020/candidate_path/work').iterdir():
    graph=read(folder/'p/export/engineering_graph_proposals.json')
    by_hash[graph['image_sha256']]=folder
rng=random.Random(20261020)
rows=[]
for case in map(str,range(1,14)):
    base=read(WORK/f'wire_validation_small_20261020/baseline/{case}_Pred.json')
    previous=read(WORK/f'wire_implementation_20261020/cached_predictions/{case}_Pred.json')
    expected=read(WORK/f'wire_next_20261020/unique_or_explicit_primary/{case}_Pred.json')
    digest=hashlib.sha256(Path(f'D:/pcb_open_gt_in/{case}.png').read_bytes()).hexdigest()
    folder=by_hash[digest]
    graph=read(folder/'p/export/engineering_graph_proposals.json')
    pins=read(folder/'p/export/engineering_pin_text_result.json')
    original=copy.deepcopy((base,graph,pins))
    actual,audit=agent.finalize_power_wire_export(base,folder/'p',digest)
    assert actual==expected,('Integrated output differs from candidate',case)
    validate_target(actual)
    assert actual['components']==previous['components']
    assert actual['nets'].keys()==previous['nets'].keys()
    for nid,net in previous['nets'].items():
        assert actual['nets'][nid]['edges']==net['edges']
        assert _members(net)<=_members(actual['nets'][nid])
    for owner,owner_pins in previous['pins'].items():
        assert all(actual['pins'][owner][key]==pin for key,pin in owner_pins.items())
    for owner,owner_pins in actual['pins'].items():
        if any(key not in previous['pins'].get(owner,{}) for key in owner_pins):
            assert actual['components'][owner]['type'] in POWER_TYPES
    assert restore_power_wires(actual,graph,pins,agent.cleanup_wire_edges)[0]==actual
    for _ in range(5):
        changed_graph,changed_pins=copy.deepcopy((graph,pins))
        for key in ['networks','physical_networks','pin_attachments','edges']:
            rng.shuffle(changed_graph[key])
        rng.shuffle(changed_pins['pin_candidates'])
        assert restore_power_wires(base,changed_graph,changed_pins,agent.cleanup_wire_edges)[0]==actual
    assert (base,graph,pins)==original
    truth=read(next((WORK/'赛题六公开数据集/赛题六公开数据集/open_GT/open_GT'/case).glob('*.json')))
    before,after=metrics.score_case(previous,truth),metrics.score_case(actual,truth)
    assert all(before[k]==after[k] for k in ['component_f1','pin_f1'])
    rows.append({'case':case,'wire_before':score_wire(previous,truth),'wire_after':score_wire(actual,truth),
                 'connectivity_before':connectivity(previous,truth),'connectivity_after':connectivity(actual,truth),
                 'candidate_identical':True,'audit':audit})
    destination=ROOT/'predictions'
    destination.mkdir(exist_ok=True)
    (destination/f'{case}_Pred.json').write_text(json.dumps(actual,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
summary={'cases':len(rows),'identical_to_candidate':len(rows),'order_checks_passed':len(rows)*5,
         'idempotent_cases':len(rows),'component_and_pin_unchanged':len(rows),
         'existing_geometry_unchanged':len(rows),'neural_inference_calls':0,
         'wire_before':sum(r['wire_before']['f1'] for r in rows)/len(rows),
         'wire_after':sum(r['wire_after']['f1'] for r in rows)/len(rows),
         'connectivity_before':sum(r['connectivity_before'] for r in rows)/len(rows),
         'connectivity_after':sum(r['connectivity_after'] for r in rows)/len(rows)}
(ROOT/args.report).write_text(json.dumps({'summary':summary,'per_case':rows},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False))
