"""Export-only compatibility check using frozen image OCR/records, no model rerun."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from adapter import validate_target
import agent
from pcb_parser.ic_name_selection import select_ic_names

out=Path('/output');agent.configure_environment(out)
rows=[]
for path in sorted(Path('/fixtures').glob('*.json')):
    fixture=json.loads(path.read_text(encoding='utf-8'))
    actual,audit=select_ic_names(fixture['before'],fixture['records'],fixture['ocr'])
    assert actual==fixture['expected'],path.name
    validate_target(actual)
    control=deepcopy(actual)
    for key,c in fixture['before']['components'].items():
        if c['type'] in {'box','ic'}:control['components'][key]['Name']=c['Name']
    assert control==fixture['before']
    case=fixture['case']
    agent.write(out/'final_results'/f'{case}_Pred.json',actual)
    agent.write(out/'logs'/f'{case}_ic_name_selection.json',audit)
    agent.write(out/'logs'/f'{case}_ocr_texts.json',fixture['ocr'])
    assert json.loads((out/'logs'/f'{case}_ocr_texts.json').read_text())==fixture['ocr']
    rows.append({'case':case,'ICs':len(audit['components']),'changed_Name_count':audit['changed_Name_count'],
                 'all_OCR_rows_and_observations_preserved':True})
report={'cases':rows,'case_count':len(rows),'network_disabled':True,
        'only_IC_Name_changed':True,'all_OCR_roundtrip_verified':True,
        'PNG_model_inference_rerun':False,'scope':'Frozen current-image prediction export only; not full image inference or scoring.',
        'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [
            Path('/submission/agent.py'),Path('/submission/adapter.py'),
            Path('/submission/src/pcb_parser/ic_name_selection.py'),Path('/submission/src/pcb_parser/component_text.py')]}}
agent.write(out/'container_export_validation.json',report)
print(json.dumps(report))
