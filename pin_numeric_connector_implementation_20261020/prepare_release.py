"""Freeze five staged files and preserve the exact previous production state."""
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent;LIVE=Path('D:/pcb5_submission')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None
files=['agent.py','src/pcb_parser/pin_numeric_connector.py','configs/pin_numeric_connector_v1.json','tests/test_pin_numeric_connector.py','docs/pin_numeric_connector.md']
items=[]
for name in files:
    source=ROOT/'runtime'/name;assert source.is_file()
    previous=sha(LIVE/name)
    if name=='agent.py':assert previous==sha(ROOT/'backup/agent.py')
    else:assert previous is None, 'Unexpected existing file: '+name
    items.append({'path':name,'sha256':sha(source),'previous_sha256':previous})
(ROOT/'release_plan.json').write_text(json.dumps({'target':str(LIVE),'files':items},indent=2)+'\n',encoding='utf-8')
installer=(ROOT.parent/'pin_row_library_implementation_20261020/install.py').read_text(encoding='utf-8').replace('.pcb-pin-row-library-','.pcb-pin-numeric-connector-')
(ROOT/'install.py').write_text(installer,encoding='utf-8')
print(json.dumps({'files':files,'previous_agent_sha256':items[0]['previous_sha256'],'new_agent_sha256':items[0]['sha256']}))
