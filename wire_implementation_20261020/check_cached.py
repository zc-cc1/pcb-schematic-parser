"""Validate staged production export against frozen inputs; no inference imports."""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
parser = argparse.ArgumentParser()
parser.add_argument('--cases', nargs='+', default=['1','5','6','13'])
parser.add_argument('--runtime', type=Path, default=ROOT/'runtime')
parser.add_argument('--report', default='small_cached_check.json')
args = parser.parse_args()
sys.path.insert(0, str(args.runtime))
sys.path.insert(0, str(args.runtime/'src'))
sys.path.insert(0, str(WORKSPACE/'pcb_parser_competition/scripts'))
import agent
from adapter import validate_target
from platform_wire_metrics import score_wire
import platform_metrics_v3 as metrics
spec = importlib.util.spec_from_file_location('power_export', args.runtime/'src/pcb_parser/power_wire_export.py')
power = importlib.util.module_from_spec(spec)
spec.loader.exec_module(power)
read = lambda p: json.loads(p.read_text(encoding='utf-8'))
work_by_hash = {}
for work in (WORKSPACE/'wire_validation_20261020/candidate_path/work').iterdir():
    graph = read(work/'p/export/engineering_graph_proposals.json')
    work_by_hash[graph['image_sha256']] = work
rows = []
for case in args.cases:
    digest = hashlib.sha256(Path(f'D:/pcb_open_gt_in/{case}.png').read_bytes()).hexdigest()
    work = work_by_hash[digest]
    graph = read(work/'p/export/engineering_graph_proposals.json')
    pins = read(work/'p/export/engineering_pin_text_result.json')
    base = read(WORKSPACE/f'wire_validation_small_20261020/baseline/{case}_Pred.json')
    untouched = copy.deepcopy((base,graph,pins))
    fixed, audit = agent.finalize_power_wire_export(base, work/'p', digest)
    validate_target(fixed)
    assert (base,graph,pins) == untouched
    assert fixed['components'] == base['components']
    for owner, owner_pins in base['pins'].items():
        assert all(fixed['pins'][owner][key] == pin for key, pin in owner_pins.items())
    for owner, owner_pins in fixed['pins'].items():
        if any(key not in base['pins'].get(owner,{}) for key in owner_pins):
            assert fixed['components'][owner]['type'] in power.POWER_TYPES
    for nid, net in base['nets'].items():
        assert fixed['nets'][nid]['edges'] == net['edges']
        assert power._members(net) <= power._members(fixed['nets'][nid])
    repeated, _ = power.restore_power_wires(fixed, graph, pins, agent.cleanup_wire_edges)
    assert repeated == fixed
    truth = read(next((WORKSPACE/'赛题六公开数据集/赛题六公开数据集/open_GT/open_GT'/case).glob('*.json')))
    before_metrics,after_metrics=metrics.score_case(base,truth),metrics.score_case(fixed,truth)
    assert all(before_metrics[key]==after_metrics[key] for key in ['component_f1','pin_f1'])
    destination = ROOT/'cached_predictions'
    destination.mkdir(exist_ok=True)
    (destination/f'{case}_Pred.json').write_text(json.dumps(fixed,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    rows.append({'case':case,'baseline_wire':score_wire(base,truth),'fixed_wire':score_wire(fixed,truth),'audit':audit})
summary = {'cases':len(rows),'baseline_wire_f1':sum(r['baseline_wire']['f1'] for r in rows)/len(rows),
           'fixed_wire_f1':sum(r['fixed_wire']['f1'] for r in rows)/len(rows),
           'improved_equal_worse':[sum(r['fixed_wire']['f1']>r['baseline_wire']['f1'] for r in rows),
                                  sum(r['fixed_wire']['f1']==r['baseline_wire']['f1'] for r in rows),
                                  sum(r['fixed_wire']['f1']<r['baseline_wire']['f1'] for r in rows)],
           'neural_inference_calls':0}
(ROOT/args.report).write_text(json.dumps({'summary':summary,'per_case':rows},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'summary':summary,'per_case':[{'case':r['case'],'before':r['baseline_wire']['f1'],
                  'after':r['fixed_wire']['f1'],'primary_contacts':r['audit']['primary_contact_count']} for r in rows]},ensure_ascii=False))
