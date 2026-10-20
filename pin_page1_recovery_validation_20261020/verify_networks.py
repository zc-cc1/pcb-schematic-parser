"""Check field synchronization on existing, model-attached exported nets only."""
import copy
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent;WORK=ROOT.parent
sys.dont_write_bytecode=True
sys.path.insert(0,str(WORK/'pcb_parser_competition/scripts'))
from platform_wire_metrics import logical_nets,_assign_nets,score_wire
import platform_metrics as metrics
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
trace=next(r for r in read(WORK/'pin_cases_1_11_20261020/trace.json') if r['case']=='1')
graph=read(Path(trace['folder'])/'p/export/engineering_graph_proposals.json')
model=read(Path(trace['folder'])/'p/export/engineering_pin_text_result.json')
baseline=read(WORK/'pin_position_implementation_20261020/deployed/1_Pred.json')
output=read(ROOT/'orientation_checked_binding/1_Pred.json')
groups=read(ROOT/'orientation_checked_binding/binding_audit.json')
contacts={a['candidate_id']:a for a in graph['pin_attachments']}
audit=[]
for net in graph['networks']:
    nid=net['network_id']
    changed=[g for g in groups if g['id'] in net['pin_candidate_ids'] and
             g['number'] is not None and g['name'] is not None]
    if not changed:continue
    if nid not in output['nets']:
        audit.append({'network_id':nid,'status':'withheld_new_net','reason':'no_existing_exported_network'})
        continue
    refs={v.strip() for v in output['nets'][nid]['hyperGraph'].strip('()').split(',') if v.strip()}
    before=sorted(refs)
    for g in changed:
        a=contacts.get(g['id'],{})
        if a.get('attachment_status')!='attached' or a.get('network_id')!=nid:continue
        ref=f"{g['owner']}.{g['number']}"
        if any(ref in {v.strip() for v in other['hyperGraph'].strip('()').split(',')}
               for other_id,other in output['nets'].items() if other_id!=nid):continue
        refs.add(ref)
    output['nets'][nid]['hyperGraph']='('+','.join(sorted(refs))+')'
    audit.append({'network_id':nid,'before':before,'after':sorted(refs)})
for nid,net in baseline['nets'].items():assert output['nets'][nid]['edges']==net['edges']
assert output['nets'].keys()==baseline['nets'].keys()
dest=ROOT/'network_synced';dest.mkdir(exist_ok=True)
(dest/'1_Pred.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
(dest/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
truth=read(next((WORK/'赛题六公开数据集/赛题六公开数据集/open_GT/open_GT/1').glob('*.json')))
def score(p):
    pn,tn=logical_nets(p),logical_nets(truth)
    conn=2*sum(s for _,_,s in _assign_nets(pn,tn))/max(1,len(pn)+len(tn))
    return {'pin':metrics.score_pins(p,truth,20)[0],'connectivity':conn,'wire':score_wire(p,truth)}
result={'before':score(baseline),'after':score(output),'audit':audit}
(ROOT/'network_sync_results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result,ensure_ascii=False))
