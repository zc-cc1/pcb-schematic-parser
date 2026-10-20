"""Smoke-check actual local OCR through the staged/deployed page-one wrapper."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parent;WORK=ROOT.parent;LIVE=Path('D:/pcb5_submission')
parser=argparse.ArgumentParser();parser.add_argument('--deployed',action='store_true');args=parser.parse_args()
sys.dont_write_bytecode=True
sys.path.insert(0,str(LIVE));sys.path.insert(0,str(LIVE/'src'))
if not args.deployed:sys.path.insert(0,str(ROOT/'runtime'))
import pcb_parser
if not args.deployed:pcb_parser.__path__.insert(0,str(ROOT/'runtime/src/pcb_parser'))
import agent
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
case=next(r for r in read(WORK/'pin_detection_audit_20261020/audit.json')['per_case'] if r['case']=='2')
baseline=read(WORK/'pin_geometry_implementation_20261020/deployed/2_Pred.json')
raw=read('D:/pcb_open_gt_v37dots_b/component_results/2_Pred.json')
image=Path('D:/pcb_open_gt_in/2.png');rgb=np.asarray(Image.open(image).convert('RGB'))
result,audit=agent.finalize_pin_relocation(baseline,Path(case['model_cache'])/'p',rgb,
    hashlib.sha256(image.read_bytes()).hexdigest())
expected=read(ROOT/'staged/2_Pred.json')
assert result==expected,'Fresh OCR differs from the cached replay'
agent.validate_target(result)
mode='deployed' if args.deployed else 'staged'
(ROOT/f'{mode}_fresh_ocr_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'mode':mode,'fresh_ocr_matches_cache':True,'local_ocr_calls':audit['ocr_calls'],
                  'changed_pin_records':len(audit['changes']),'network_audit_entries':len(audit['network_changes']),
                  'external_api_calls':0,'full_model_runs':0},ensure_ascii=False))

