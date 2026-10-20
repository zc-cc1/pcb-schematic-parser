import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
DEPLOY=Path('D:/pcb5_submission')
FILES=['src/pcb_parser/power_wire_export.py','tests/test_wire_export_regression.py',
       'docs/wire_export_update_20261020.md','package_manifest.json']
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
validation=json.loads((ROOT/'staged_cached_check.json').read_text(encoding='utf-8'))['summary']
assert validation['identical_to_candidate']==13 and validation['order_checks_passed']==65
manifest_path=ROOT/'runtime/package_manifest.json'
manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
for name in FILES:
    if name=='package_manifest.json':continue
    path=ROOT/'runtime'/name
    manifest['files'][name]={'bytes':path.stat().st_size,'sha256':sha(path)}
manifest['wire_export_update'].update({'version':'observed-physical-power-wires-v2',
    'existing_contact_recovery':'unique_observed_or_explicit_duplicate_primary',
    'cached_cases_validated':13,'regression_tests_passed':15,'input_order_checks_passed':65,
    'new_model_inference_calls':0,'docker_build_verified':False})
manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
plan=[]
for name in FILES:
    source=ROOT/'runtime'/name
    backup=ROOT/'backup'/name
    assert sha(backup)==sha(DEPLOY/name),'Production file changed since backup: '+name
    if source.suffix=='.py':compile(source.read_text(encoding='utf-8'),str(source),'exec')
    plan.append({'path':name,'bytes':source.stat().st_size,'sha256':sha(source),
                 'previous_sha256':sha(backup)})
(ROOT/'release_plan.json').write_text(json.dumps({'target':str(DEPLOY),'files':plan},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'target':str(DEPLOY),'files':len(plan),'paths':FILES},ensure_ascii=False))
