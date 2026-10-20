import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
RUNTIME=ROOT/'runtime'
DEPLOY=Path('D:/pcb5_submission')
FILES=['agent.py','src/pcb_parser/wire_graph.py','src/pcb_parser/power_wire_export.py',
       'Dockerfile.overlay','overlay_check.py','package_manifest.json',
       'tests/test_wire_export_regression.py','docs/wire_export_update_20261020.md']
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
manifest=json.loads((RUNTIME/'package_manifest.json').read_text(encoding='utf-8'))
for name in FILES:
    if name=='package_manifest.json':continue
    path=RUNTIME/name
    manifest['files'][name]={'bytes':path.stat().st_size,'sha256':sha(path)}
manifest['wire_export_update']={'version':'observed-physical-power-wires-v1','date':'2026-10-20',
    'default_enabled':True,'disable_env':'PCB_POWER_WIRE_EXPORT=0',
    'path_splitting':'preserve_raster_path','cached_cases_validated':13,
    'regression_tests_passed':11,'new_model_inference_calls':0,'docker_build_verified':False}
(RUNTIME/'package_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
plan=[]
for name in FILES:
    source=RUNTIME/name
    target=DEPLOY/name
    backup=ROOT/'backup'/name
    if target.exists():
        if not backup.exists() or sha(backup)!=sha(target):
            raise ValueError('Deployment changed since backup: '+name)
    elif backup.exists():
        raise ValueError('Unexpected backup of missing target: '+name)
    if source.suffix=='.py':compile(source.read_text(encoding='utf-8'),str(source),'exec')
    plan.append({'path':name,'bytes':source.stat().st_size,'sha256':sha(source),
                 'previous_sha256':sha(backup) if backup.exists() else None})
(ROOT/'release_plan.json').write_text(json.dumps({'target':str(DEPLOY),'files':plan},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'target':str(DEPLOY),'files':len(plan),'changed':[p['path'] for p in plan]},ensure_ascii=False))
