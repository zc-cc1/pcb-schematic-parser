"""Stage current production sources and immutable backups of this release."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parent
DEPLOY=Path('D:/pcb5_submission')
FILES=['src/pcb_parser/power_wire_export.py','tests/test_wire_export_regression.py',
       'docs/wire_export_update_20261020.md','package_manifest.json']
for relative in FILES:
    backup=ROOT/'backup'/relative
    if backup.exists():raise FileExistsError(backup)
    backup.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(DEPLOY/relative,backup)
for source in DEPLOY.rglob('*'):
    if (not source.is_file() or '__pycache__' in source.parts
            or source.suffix not in {'.py','.json'} and source.name not in {'run.sh','Dockerfile.overlay'}):
        continue
    destination=ROOT/'runtime'/source.relative_to(DEPLOY)
    destination.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(source,destination)
doc=ROOT/'runtime/docs/wire_export_update_20261020.md'
doc.parent.mkdir(parents=True,exist_ok=True)
shutil.copy2(DEPLOY/'docs/wire_export_update_20261020.md',doc)
snapshot={str(p.relative_to(DEPLOY)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest()
          for p in DEPLOY.rglob('*') if p.is_file() and p.suffix in {'.py','.json','.pt','.onnx'}
          and '__pycache__' not in p.parts}
(ROOT/'deployment_before.json').write_text(json.dumps(snapshot,indent=2)+'\n',encoding='utf-8')
shutil.copy2(ROOT.parent/'wire_implementation_20261020/install.py',ROOT/'install.py')
shutil.copy2(ROOT.parent/'wire_implementation_20261020/verify_deployment.py',ROOT/'verify_deployment.py')
print('Staged production sources;',len(snapshot),'protected files recorded; four files backed up.')
