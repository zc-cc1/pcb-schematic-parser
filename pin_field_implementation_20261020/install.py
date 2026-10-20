"""Apply or roll back the exact reviewed release; no recursive file operations."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

ROOT=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None
parser=argparse.ArgumentParser()
mode=parser.add_mutually_exclusive_group(required=True)
mode.add_argument('--apply',action='store_true')
mode.add_argument('--rollback',action='store_true')
args=parser.parse_args()
plan=json.loads((ROOT/'release_plan.json').read_text(encoding='utf-8'))
target_root=Path(plan['target']).resolve()
assert target_root==Path('D:/pcb5_submission').resolve()


def write_atomic(path,data):
    assert path.resolve().is_relative_to(target_root)
    path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix='.pcb-pin-fields-',suffix='.tmp',dir=path.parent,delete=False) as handle:
        temporary=Path(handle.name)
        handle.write(data)
    try:
        os.replace(temporary,path)
    finally:
        if temporary.exists():temporary.unlink()


def restore(item):
    path=target_root/item['path']
    assert path.resolve().is_relative_to(target_root)
    if sha(path)!=item['sha256']:
        raise ValueError('Refusing to roll back a file changed after this release: '+item['path'])
    if item['previous_sha256'] is None:
        path.unlink()
    else:
        backup=ROOT/'backup'/item['path']
        assert sha(backup)==item['previous_sha256']
        write_atomic(path,backup.read_bytes())


for item in plan['files']:
    path=target_root/item['path']
    assert path.resolve().is_relative_to(target_root)
    expected=item['previous_sha256'] if args.apply else item['sha256']
    if sha(path)!=expected:raise ValueError('Target differs from expected state: '+item['path'])
    source=ROOT/'runtime'/item['path']
    if sha(source)!=item['sha256']:raise ValueError('Staged source differs: '+item['path'])
    if item['previous_sha256'] is not None:
        assert sha(ROOT/'backup'/item['path'])==item['previous_sha256']

applied=[]
try:
    for item in plan['files']:
        if args.apply:
            write_atomic(target_root/item['path'],(ROOT/'runtime'/item['path']).read_bytes())
            applied.append(item)
        else:
            restore(item)
except Exception:
    if args.apply:
        for item in reversed(applied):restore(item)
    raise
expected_key='sha256' if args.apply else 'previous_sha256'
assert all(sha(target_root/i['path'])==i[expected_key] for i in plan['files'])
status={'status':'applied' if args.apply else 'rolled_back','target':str(target_root),
        'files':len(plan['files']),'hashes_verified':True}
(ROOT/'deployment_status.json').write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(status,ensure_ascii=False))
