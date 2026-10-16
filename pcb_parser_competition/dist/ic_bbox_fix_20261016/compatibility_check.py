"""Official-runtime check of image-only fixtures; no reference labels mounted."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from pcb_parser.ic_geometry import refine_ic_boxes

output=Path('/output');output.mkdir(parents=True,exist_ok=True)
rows=[]
for path in sorted(Path('/fixtures').glob('*.json')):
    fixture=json.loads(path.read_text(encoding='utf-8'))
    image=path.with_suffix('.png')
    assert hashlib.sha256(image.read_bytes()).hexdigest()==fixture['image_sha256']
    actual,audit=refine_ic_boxes(np.array(Image.open(image).convert('RGB')),fixture['before'])
    assert actual==fixture['expected'],path.name
    control=deepcopy(actual)
    for key,c in fixture['before']['components'].items():
        control['components'][key]['bbox']=c['bbox']
    assert control==fixture['before']
    (output/(path.stem+'_ic_geometry.json')).write_text(json.dumps(audit,indent=2),encoding='utf-8')
    rows.append({'case':path.stem,'ICs':len(audit['entries']),'changed':audit['changed_IC_boxes'],
                 'host_expected_matches':True,'all_non_bbox_fields_preserved':True})
report={'cases':rows,'network_disabled':True,'scope':'Image-only runtime compatibility; cached detector proposals, no model rerun.',
        'ic_geometry_sha256':hashlib.sha256(Path('/submission/src/pcb_parser/ic_geometry.py').read_bytes()).hexdigest()}
(output/'compatibility_validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report))
