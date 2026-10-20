"""Exercise all live export hooks with retained same-image model/OCR outputs."""
import copy
import argparse
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
parser=argparse.ArgumentParser();parser.add_argument('--deployed',action='store_true');args=parser.parse_args()
if not args.deployed:sys.path.insert(0,str(ROOT/'runtime'))
import pcb_parser
if not args.deployed:pcb_parser.__path__.insert(0,str(ROOT/'runtime/src/pcb_parser'))
import agent
from pcb_parser import led_geometry
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
inventory={r['case']:r for r in read(WORK/'pin_detection_audit_20261020/audit.json')['per_case']}
original_run=subprocess.run
original_field=agent.finalize_pin_field_recovery
original_geometry=agent.finalize_pin_geometry_recovery
original_relocation=agent.finalize_pin_relocation
original_pairing=agent.finalize_pin_pairing
original_evidence=agent.finalize_pin_evidence_repair
original_cycle=agent.finalize_pin_number_cycle
original_nc=agent.finalize_pin_nc_recovery
original_vertical=agent.finalize_pin_vertical_repair
cache_roots=[ROOT/'ocr_cache',WORK/'pin_nc_recovery_implementation_20261020/ocr_cache',WORK/'pin_number_cycle_implementation_20261020/ocr_cache',WORK/'pin_evidence_implementation_20261020/ocr_cache',WORK/'pin_pairing_implementation_20261020/ocr_cache',WORK/'pin_relocation_implementation_20261020/ocr_cache',WORK/'pin_geometry_implementation_20261020/ocr_cache',
             WORK/'pin_field_implementation_20261020/ocr_cache']
def reader(image,box,scale,rotation,recognition_only,view):
    crop=np.ascontiguousarray(image[box[1]:box[3],box[0]:box[2]])
    key=hashlib.sha256(crop.tobytes()+str((crop.shape,box,scale,rotation,recognition_only)).encode()).hexdigest()
    sources=[p/(key+'.json') for p in cache_roots if (p/(key+'.json')).exists()]
    assert sources,'OCR observation unavailable: '+view
    return [{**t,'view':view} for t in read(sources[0])]

results=[]
for case in ['1','2','4','5','6','8','10']:
    folder=Path(inventory[case]['model_cache']);output=ROOT/'cached_entrypoint'/case;output.mkdir(parents=True,exist_ok=True)
    for name in ['final_results','project_results','component_results','logs']:(output/name).mkdir(exist_ok=True)
    baseline=read(WORK/f'pin_nc_recovery_implementation_20261020/deployed/{case}_Pred.json')
    expected=read(ROOT/f'staged/{case}_Pred.json')
    ocr_path=Path(f'D:/pcb_open_gt_v37dots_b/logs/{case}_ocr_texts.json')
    calls=[]
    def fake_pipeline(command,**kwargs):
        if not isinstance(command,list) or '--output' not in command:return original_run(command,**kwargs)
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
        return target,{'policy':'cached_entrypoint_smoke','entries':[]}
    def checked_call(function,target,*args):
        before=copy.deepcopy(target);result,audit=function(target,*args,reader=reader)
        assert target==before and result['components']==before['components']
        for nid,net in before['nets'].items():assert result['nets'][nid]['edges']==net['edges']
        return result,audit
    def field_hook(*args):return checked_call(original_field,*args)
    def geometry_hook(*args):return checked_call(original_geometry,*args)
    def relocation_hook(*args):return checked_call(original_relocation,*args)
    def pairing_hook(*args):return checked_call(original_pairing,*args)
    def evidence_hook(*args):
        before=copy.deepcopy(args[0])
        result,audit=checked_call(original_evidence,*args)
        assert result['nets']==before['nets']
        return result,audit
    def cycle_hook(*args):
        result,audit=checked_call(original_cycle,*args)
        return result,audit
    def nc_hook(*args):
        before=copy.deepcopy(args[0])
        result,audit=checked_call(original_nc,*args)
        assert result['nets']==before['nets']
        return result,audit
    def vertical_hook(*args):
        result,audit=checked_call(original_vertical,*args)
        calls.append(audit);return result,audit
    with patch.object(agent.subprocess,'run',side_effect=fake_pipeline),patch.object(led_geometry,'yolo_predictor',return_value=None),patch.object(led_geometry,'refine_led_boxes',side_effect=frozen_led),patch.object(agent,'finalize_pin_field_recovery',side_effect=field_hook),patch.object(agent,'finalize_pin_geometry_recovery',side_effect=geometry_hook),patch.object(agent,'finalize_pin_relocation',side_effect=relocation_hook),patch.object(agent,'finalize_pin_pairing',side_effect=pairing_hook),patch.object(agent,'finalize_pin_evidence_repair',side_effect=evidence_hook),patch.object(agent,'finalize_pin_number_cycle',side_effect=cycle_hook),patch.object(agent,'finalize_pin_nc_recovery',side_effect=nc_hook),patch.object(agent,'finalize_pin_vertical_repair',side_effect=vertical_hook):
        row=agent.run_case(Path(f'D:/pcb_open_gt_in/{case}.png'),output,'cpu','target',True)
    assert row['success'],row
    assert not any(key.startswith('pin_') and key.endswith('_error') for key in row),row
    assert len(calls)==1
    actual=read(output/f'final_results/{case}_Pred.json')
    for owner,comp in expected['components'].items():
        if comp['type'] in {'box','ic'}:assert actual['pins'].get(owner)==expected['pins'].get(owner),owner
    verified=read(ROOT/f'staged/{case}_audit.json')
    expected_summary={'enabled':True,'changed_pins':len(verified['changes']),
                      'changed_networks':sum('after' in v for v in verified['network_changes']),
                      'ocr_calls':verified['ocr_calls'],'restored_networks':len(verified['restored_networks'])}
    assert row['pin_vertical_repair']==expected_summary
    assert (output/f'logs/{case}_pin_vertical_repair.json').is_file()
    prior_pairing=read(WORK/f'pin_pairing_implementation_20261020/deployed/{case}_audit.json')
    for restored in prior_pairing['restored_networks']+verified['restored_networks']:
        nid=restored['network_id']
        assert actual['nets'][nid]==expected['nets'][nid], 'Restored branch differs from verified output'
    prior_cycle=read(WORK/f'pin_number_cycle_implementation_20261020/deployed/{case}_audit.json')
    for change in prior_pairing['network_changes']+prior_cycle['network_changes']+verified['network_changes']:
        if 'after' in change:
            nid=change['network_id']
            members=sorted(v.strip() for v in actual['nets'][nid]['hyperGraph'].strip('()').split(',') if v.strip())
            assert members==change['after'], 'Rekeyed membership differs from verified output'
    results.append({'case':case,'success':True,'ic_pins_match_verified_output':True,
                    'vertical_repair_log_written':True,'summary':row['pin_vertical_repair'],
                    'new_model_calls':0,'new_ocr_calls':0})
result={'mode':'cached_model_and_ocr_live_entrypoint','cases':results,'kept_work':True,'runtime':'deployed' if args.deployed else 'staged'}
(ROOT/('entrypoint_deployed_check.json' if args.deployed else 'entrypoint_check.json')).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result,ensure_ascii=False))
