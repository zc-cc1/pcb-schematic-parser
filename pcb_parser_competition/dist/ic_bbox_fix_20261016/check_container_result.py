"""Verify full PNG entrypoint and the declared Target-only changes."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'submission/platform_adapter'))
from adapter import COMPONENT_OUTPUT_POLICY,to_target,validate_target

def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

output=HERE/'container-output';case='0036'
decision=read(output/'decision_log.json')
assert decision['successful_cases']==1 and decision['failed_cases']==0
assert decision['device']=='cpu' and decision['output_format']=='target'
assert decision['component_output_policy']==COMPONENT_OUTPUT_POLICY
before=read(output/'project_results'/f'{case}_Pred.json')
augmented=read(output/'component_results'/f'{case}_Pred.json')
assert before['nets']==augmented['nets']
assert all(augmented['components'][k]==v for k,v in before['components'].items())
assert all(augmented['pins'][k]==v for k,v in before['pins'].items())
expected=to_target(augmented)
name_audit=read(output/'logs'/f'{case}_ic_name_selection.json')
for entry in name_audit['components']:
    c=expected['components'][entry['component_id']]
    assert c['Name']==entry['before_Name']
    c['Name']=entry['after_Name']
pre_geometry=deepcopy(expected)
audit=read(output/'logs'/f'{case}_ic_geometry.json')
for entry in audit['entries']:
    c=expected['components'][entry['component_id']]
    assert c['type'] in {'box','ic'}
    assert c['bbox']==entry['before_bbox_bottom_left']
    c['bbox']=entry['after_bbox_bottom_left']
actual=read(output/'final_results'/f'{case}_Pred.json')
assert actual==expected
validate_target(actual)
control=deepcopy(actual)
for key,c in pre_geometry['components'].items():
    if c['type'] in {'box','ic'}:control['components'][key]['bbox']=c['bbox']
assert control==pre_geometry
summary=read(output/'logs'/f'{case}_summary.json')
assert audit['image_sha256']==summary['image_sha256']==sha(HERE/'container-input'/f'{case}.png')
sources=read(HERE/'container_source_sha256.json')
for relative,expected_hash in sources.items():assert sha(ROOT/relative)==expected_hash,relative
compatibility=read(HERE/'compatibility-output/compatibility_validation.json')
assert compatibility['ic_geometry_sha256']==sha(ROOT/'src/pcb_parser/ic_geometry.py')
report={'fresh_PNG_case':case,'device':decision['device'],'network_disabled':True,
        'successful_cases':decision['successful_cases'],'full_output_matches_declared_changes':True,
        'only_IC_bbox_changed_in_geometry_step':True,'source_hashes_unchanged':True,
        'IC_Name_after':{k:c['Name'] for k,c in actual['components'].items() if c['type']=='box'},
        'IC_boxes':audit['entries'],'IC_geometry_seconds':audit['wall_seconds'],
        'compatibility_cases':compatibility['cases'],
        'source_sha256':sources,'scope':'Fresh PNG CPU inference plus 3 cached-proposal image-only compatibility checks. No official evaluation.'}
(HERE/'container_validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(report,ensure_ascii=False,indent=2))
