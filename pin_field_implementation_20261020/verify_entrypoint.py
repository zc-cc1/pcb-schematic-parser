"""Run the live case entrypoint with retained model output and cached OCR.

The detector/U-Net subprocess and LED detector are replaced by this same
image's retained outputs. The entrypoint, field hook, logs and export are real.
"""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from unittest.mock import patch

import numpy as np

ROOT=Path(__file__).resolve().parent;WORK=ROOT.parent;LIVE=Path('D:/pcb5_submission')
sys.dont_write_bytecode=True;sys.path.insert(0,str(LIVE));sys.path.insert(0,str(LIVE/'src'))
import agent
from pcb_parser import led_geometry
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
case=next(r for r in read(WORK/'pin_detection_audit_20261020/audit.json')['per_case'] if r['case']=='1')
folder=Path(case['model_cache']);output=ROOT/'cached_entrypoint';output.mkdir(exist_ok=True)
for name in ['final_results','project_results','component_results','logs']:(output/name).mkdir(exist_ok=True)
baseline=read(WORK/'pin_position_implementation_20261020/deployed/1_Pred.json')
expected=read(ROOT/'staged/1_Pred.json')
ocr_path=Path('D:/pcb_open_gt_v37dots_b/logs/1_ocr_texts.json')
original_recovery=agent.finalize_pin_field_recovery
original_subprocess_run=subprocess.run
calls=[]
def fake_pipeline(command,**kwargs):
    if not isinstance(command,list) or '--output' not in command:
        return original_subprocess_run(command,**kwargs)
    destination=Path(command[command.index('--output')+1]).resolve()
    assert destination.is_relative_to(output.resolve())
    shutil.copytree(folder/'p',destination,dirs_exist_ok=True)
    summary=read(destination/'summary.json')
    summary['original_S23_input_sha256']={str(ocr_path):hashlib.sha256(ocr_path.read_bytes()).hexdigest()}
    (destination/'summary.json').write_text(json.dumps(summary),encoding='utf-8')
    return subprocess.CompletedProcess(command,0)
def frozen_led(rgb,payload,target,predictor):
    target=copy.deepcopy(target)
    for owner,comp in target['components'].items():
        if comp['type']=='led':comp['bbox']=copy.deepcopy(baseline['components'][owner]['bbox'])
    return target,{'policy':'cached_model_smoke_test','entries':[]}
def cached_reader(image,box,scale,rotation,recognition_only,view):
    crop=np.ascontiguousarray(image[box[1]:box[3],box[0]:box[2]])
    key=hashlib.sha256(crop.tobytes()+str((crop.shape,box,scale,rotation,recognition_only)).encode()).hexdigest()
    source=ROOT/'ocr_cache'/(key+'.json')
    assert source.exists(),f'OCR replay observation missing: {view}'
    return [{**t,'view':view} for t in read(source)]
def recovery_hook(target,stage,rgb,digest,raw_pins,lead_px,tip_mode):
    before=copy.deepcopy(target)
    result,audit=original_recovery(target,stage,rgb,digest,raw_pins,lead_px,tip_mode,reader=cached_reader)
    assert target==before
    assert result['components']==before['components']
    for nid,net in before['nets'].items():assert result['nets'][nid]['edges']==net['edges']
    calls.append(audit)
    return result,audit
with patch.object(agent.subprocess,'run',side_effect=fake_pipeline),patch.object(led_geometry,'yolo_predictor',return_value=None),patch.object(led_geometry,'refine_led_boxes',side_effect=frozen_led),patch.object(agent,'finalize_pin_field_recovery',side_effect=recovery_hook):
    row=agent.run_case(Path('D:/pcb_open_gt_in/1.png'),output,'cpu','target',True)
assert row['success'] and 'pin_field_recovery_error' not in row,row
assert len(calls)==1
actual=read(output/'final_results/1_Pred.json')
for owner,comp in expected['components'].items():
    if comp['type'] in {'box','ic'}:assert actual['pins'].get(owner)==expected['pins'].get(owner),owner
assert row['pin_field_recovery']=={'enabled':True,'changed_pins':8,'changed_networks':3,'ocr_calls':52}
assert (output/'logs/1_pin_field_recovery.json').is_file()
result={'success':True,'live_case_hook_executed':True,'ic_pins_match_verified_output':True,
        'field_log_written':True,'summary':row['pin_field_recovery'],'new_model_calls':0,'new_ocr_calls':0,
        'mode':'cached_pipeline_entrypoint_smoke','kept_work':True}
(ROOT/'entrypoint_check.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result,ensure_ascii=False))
