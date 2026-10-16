"""PNG -> learned ownership/OCR -> header recovery and dense pin text v2."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from PIL import Image

from run_pin_text import ROOT,read,save,sha,validate_image,attach_identities,overlay
from pcb_parser.component_text import bind_components,run_ocr
from pcb_parser.component_identity_recovery import recover_component_identities
from pcb_parser.pin_text_v2 import bind_pin_text_v2,adopt_pin_fields
from pcb_parser.pin_text import bind_pin_text
from pcb_parser.pin_text_ocr import create_engine


def process(image_path,ownership_path,texts_path,output,cfg,engine=None,models=None):
    begin=time.perf_counter()
    with Image.open(image_path) as source:rgb=np.asarray(source.convert('RGB'))
    size=[rgb.shape[1],rgb.shape[0]];image_hash=sha(image_path);ownership=read(ownership_path)
    validate_image(ownership,image_hash,size)
    component_cfg=read(ROOT/'configs/pin_text_component_v2.json')
    if texts_path:
        ocr=read(texts_path);validate_image(ocr,image_hash,size)
        if not ocr.get('prediction_only'):raise ValueError('Expected image-model OCR predictions')
        texts=ocr['texts'];models=ocr.get('OCR_models');timing={'source':'frozen_image_model_OCR','sha256':sha(texts_path)}
    else:
        if engine is None:engine,models=create_engine(cfg)
        texts,timing=run_ocr(rgb,engine,component_cfg['component_text']['ocr'])
        ocr={'image_sha256':image_hash,'size':size,'coordinate_origin':'top_left','texts':texts,'OCR_models':models,
             'prediction_only':True,'timing':timing}
    detections=[{'bbox':c['bbox_top_left'],'confidence':c['confidence'],'class_name':c['class_name']} for c in ownership['components']]
    _,base_records,component_debug=bind_components(detections,texts,*size,component_cfg)
    records,recovery=recover_component_identities(ownership,texts,base_records,cfg['identity_recovery'])
    joined=attach_identities(ownership,records,component_cfg)
    experimental,debug=bind_pin_text_v2(joined,texts,cfg,records)
    baseline,_=bind_pin_text(joined,texts,read(ROOT/'configs/pin_text_v1.json'),base_records)
    result,adoption=adopt_pin_fields(baseline,experimental,cfg['adoption'])
    if [(p['candidate_id'],p['point']) for p in ownership['pin_candidates']]!=[(p['candidate_id'],p['point']) for p in result['pin_candidates']]:
        raise ValueError('Pin points/IDs changed')
    result.update(image_sha256=image_hash,ownership_input_sha256=sha(ownership_path),
                  pin_text_config_sha256=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest(),
                  component_text_config_sha256=sha(ROOT/'configs/pin_text_component_v2.json'))
    result['pin_text_summary']['component_identity_resolved']=sum(bool(r['designator']) for r in records)
    result['pin_text_summary']['new_header_identities']=recovery['new_identities']
    result['pin_text_summary']['new_model_names']=recovery['new_model_names']
    save(output/'pin_text_result.json',result);save(output/'component_records.json',{'image_sha256':image_hash,'records':records})
    save(output/'texts.json',ocr);save(output/'text_binding_debug.json',debug);save(output/'field_adoption_debug.json',adoption)
    experimental.update(image_sha256=image_hash,pin_text_config_sha256=result['pin_text_config_sha256'])
    save(output/'experimental_pin_text_result.json',experimental)
    save(output/'component_binding_debug.json',component_debug);save(output/'identity_recovery_debug.json',recovery)
    overlay(rgb,result,output/'pin_text_overlay.png')
    summary={'image_sha256':image_hash,'size':size,'summary':result['pin_text_summary'],'config_sha256':result['pin_text_config_sha256'],
             'ownership_input_sha256':sha(ownership_path),'used_ownership_cache':True,'used_external_OCR_cache':bool(texts_path),
             'input_contract':'PNG/image-model predictions only; no reference annotation, CAD or Golden reads',
             'OCR_models':models,'timing':timing,'wall_seconds':time.perf_counter()-begin,'official_PinF1':None}
    save(output/'summary.json',summary);return summary


def main():
    p=argparse.ArgumentParser();p.add_argument('--image',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--ownership',type=Path);p.add_argument('--texts',type=Path);p.add_argument('--device',default='cpu')
    p.add_argument('--config',type=Path,default=ROOT/'configs/pin_text_v2.json');a=p.parse_args()
    ownership=a.ownership
    if ownership is None:
        ownership=a.output/'ownership/pin_ownership.json'
        subprocess.run([sys.executable,str(ROOT/'scripts/run_pin_ownership.py'),'--image',str(a.image),'--device',a.device,
                        '--output',str(ownership.parent)],check=True)
    summary=process(a.image,ownership,a.texts,a.output,read(a.config))
    summary['used_ownership_cache']=a.ownership is not None;save(a.output/'summary.json',summary)
    print(json.dumps({k:summary[k] for k in ['summary','used_ownership_cache','used_external_OCR_cache']},ensure_ascii=True),flush=True)


if __name__=='__main__':main()
