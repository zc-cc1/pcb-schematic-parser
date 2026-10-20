import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
DEPLOY=Path('D:/pcb5_submission')
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
before=read(ROOT/'deployment_before.json')
plan=read(ROOT/'release_plan.json')
after={str(p.relative_to(DEPLOY)).replace('\\','/'):sha(p) for p in DEPLOY.rglob('*')
       if p.is_file() and p.suffix in {'.py','.json','.pt','.onnx'} and '__pycache__' not in p.parts}
changed=sorted(k for k in before.keys()|after.keys() if before.get(k)!=after.get(k))
allowed={item['path'] for item in plan['files']}
assert set(changed)<=allowed,changed
assert all(sha(DEPLOY/item['path'])==item['sha256'] for item in plan['files'])
manifest=read(DEPLOY/'package_manifest.json')
assert all(sha(DEPLOY/name)==manifest['files'][name]['sha256'] for name in manifest['runtime_assets'])
assert all(manifest['files'][item['path']]['sha256']==item['sha256']
           for item in plan['files'] if item['path']!='package_manifest.json')
result={'verified':True,'planned_files':len(allowed),'protected_files_before':len(before),
        'protected_files_after':len(after),'changed':changed,'unexpected_changes':[],
        'runtime_asset_hashes_valid':True,'release_hashes_valid':True}
(ROOT/'deployment_integrity.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result,ensure_ascii=False))
