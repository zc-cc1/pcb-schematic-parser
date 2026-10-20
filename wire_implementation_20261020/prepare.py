"""Stage only production sources; inference caches stay in validation directories."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent
DEPLOY = Path('D:/pcb5_submission')
runtime = ROOT / 'runtime'
backup = ROOT / 'backup'
for relative in ['agent.py', 'adapter.py', 'src/pcb_parser/wire_graph.py']:
    for folder in [runtime, backup]:
        target = folder / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise FileExistsError(target)
        shutil.copy2(DEPLOY / relative, target)

source = (runtime / 'src/pcb_parser/wire_graph.py').read_text(encoding='utf-8')
candidate = (ROOT.parent / 'wire_validation_20261020/runtime_candidate/src/pcb_parser/wire_graph.py').read_text(encoding='utf-8')
begin = source.index('    pin_splits=0\n')
end = source.index('    nets,attachments=network_membership(', begin)
cbegin = candidate.index('    # Validation variant: cut on the actual raster path')
cend = candidate.index('    nets,attachments=network_membership(', cbegin)
replacement = candidate[cbegin:cend].replace('# Validation variant: cut', '# Cut')
(runtime / 'src/pcb_parser/wire_graph.py').write_text(source[:begin]+replacement+source[end:], encoding='utf-8')
snapshot = {str(p.relative_to(DEPLOY)).replace('\\', '/'): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in DEPLOY.rglob('*') if p.is_file() and p.suffix in {'.py','.json','.pt','.onnx'}
            and '__pycache__' not in p.parts}
(ROOT / 'deployment_before.json').write_text(json.dumps(snapshot, indent=2)+'\n', encoding='utf-8')
print('Staged source files and recorded',len(snapshot),'deployment hashes.')
