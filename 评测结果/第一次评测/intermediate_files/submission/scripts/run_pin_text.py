"""PNG -> model-owned pins -> observed OCR numbers/names. No paired labels read."""
import argparse
from copy import deepcopy
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from pcb_parser.component_text import bind_components,run_ocr
from pcb_parser.pin_ownership import normalized_components
from pcb_parser.pin_text import bind_pin_text
from pcb_parser.pin_text_ocr import create_engine,run_local_ocr


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path,payload):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(payload,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def validate_image(payload,image_sha,size):
    if payload.get('image_sha256')!=image_sha or payload.get('coordinate_origin')!='top_left' or payload.get('size',size)!=size:
        raise ValueError('Prediction cache does not match input PNG/hash/coordinate origin')


def attach_identities(ownership,records,cfg):
    detections=[{'bbox':c['bbox_top_left'],'confidence':c['confidence'],'class_name':c['class_name']} for c in ownership['components']]
    if [c['detection_index'] for c in ownership['components']]!=list(range(len(detections))):
        raise ValueError('Ownership component indices must match detector order')
    result=deepcopy(ownership)
    result['components']=normalized_components(detections,cfg['class_mapping'],records)
    by_index={c['detection_index']:c for c in result['components']}
    for pin in result['pin_candidates']:
        if pin['owner_status']=='assigned':
            pin['owner']=by_index[pin['owner_detection_index']]['component_id']
        for candidate in pin.get('owner_candidates',[]):
            candidate['component_id']=by_index[candidate['detection_index']]['component_id']
    return result


def overlay(rgb,result,path):
    image=Image.fromarray(rgb)
    draw=ImageDraw.Draw(image)
    font=ImageFont.truetype(str(Path(__file__).resolve().parents[1] / 'assets/fonts/DejaVuSansMono.ttf'),12)
    for component in result['components']:
        box=component['bbox_top_left']
        draw.rectangle(box,outline='#999999')
        draw.text((box[0],max(0,box[1]-15)),component['component_id'],font=font,fill='#444444',stroke_width=1,stroke_fill='white')
    for pin in result['pin_candidates']:
        x,y=pin['point']
        color='#087b44' if pin['number'] is not None else '#c46800'
        draw.ellipse((x-3,y-3,x+3,y+3),outline=color,width=2)
        label=pin['number'] or '?'
        if pin['pinname']:
            label+=' '+pin['pinname']
        if pin['number'] is not None or pin['pinname'] is not None:
            draw.text((x+5,y+3),label,font=font,fill=color,stroke_width=1,stroke_fill='white')
    image.thumbnail((1700,1700))
    canvas=Image.new('RGB',(max(680,image.width),image.height+44),'white')
    canvas.paste(image)
    legend=ImageDraw.Draw(canvas)
    legend.text((8,image.height+3),'Green: observed number | Orange: number unresolved | Text: OCR proposals',font=font,fill='#444444')
    legend.text((8,image.height+23),'Colours indicate field status, not correctness. Original pin points are unchanged.',font=font,fill='#444444')
    canvas.save(path)


def process(image_path,ownership_path,output,cfg,component_cfg,engine=None,model_fingerprint=None,texts_path=None,local=True):
    start=time.perf_counter()
    with Image.open(image_path) as image:
        rgb=np.asarray(image.convert('RGB'))
    size=[rgb.shape[1],rgb.shape[0]]
    image_sha=sha(image_path)
    ownership=read(ownership_path)
    validate_image(ownership,image_sha,size)
    if texts_path:
        cache=read(texts_path)
        validate_image(cache,image_sha,size)
        texts=cache['texts']
        base_timing={'source':'external_model_OCR_cache','cache_sha256':sha(texts_path),
                     'source_ocr_config_sha256':cache.get('ocr_config_sha256'),
                     'source_OCR_models':cache.get('OCR_models'),'wall_seconds':0}
    else:
        if engine is None:
            engine,model_fingerprint=create_engine(cfg)
        texts,base_timing=run_ocr(rgb,engine,component_cfg['component_text']['ocr'])
    detections=[{'bbox':c['bbox_top_left'],'confidence':c['confidence'],'class_name':c['class_name']} for c in ownership['components']]
    _,records,component_debug=bind_components(detections,texts,size[0],size[1],component_cfg)
    ownership=attach_identities(ownership,records,component_cfg)
    before,_=bind_pin_text(ownership,texts,cfg,records)
    local_timing={'roi_count':0,'wall_seconds':0,'disabled':True}
    if local and cfg['local_ocr']['enabled']:
        if engine is None:
            engine,model_fingerprint=create_engine(cfg)
        texts,local_timing=run_local_ocr(rgb,engine,ownership,before,texts,cfg)
        # Pin evidence never changes the already frozen component identities.
    result,debug=bind_pin_text(ownership,texts,cfg,records)
    result.update(image_sha256=image_sha,ownership_input_sha256=sha(ownership_path),
                  pin_text_config_sha256=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest(),
                  component_text_config_sha256=hashlib.sha256(json.dumps(component_cfg,sort_keys=True).encode()).hexdigest())
    result['pin_text_summary']['component_identity_resolved']=sum(bool(r['designator']) for r in records)
    timing={'base_OCR':base_timing,'local_OCR':local_timing,'wall_seconds_to_binding':time.perf_counter()-start}
    save(output/'pin_text_result.json',result)
    save(output/'component_records.json',{'image_sha256':image_sha,'records':records})
    save(output/'component_binding_debug.json',component_debug)
    save(output/'text_binding_debug.json',debug)
    save(output/'texts.json',{'image_sha256':image_sha,'size':size,'coordinate_origin':'top_left','texts':texts,
                             'OCR_models':model_fingerprint,'timing':timing,'prediction_only':True})
    overlay(rgb,result,output/'pin_text_overlay.png')
    summary={'image':str(image_path),'image_sha256':image_sha,'ownership_input_sha256':sha(ownership_path),
             'summary':result['pin_text_summary'],'before_local':before['pin_text_summary'],'timing':timing,
             'rapidocr_version':importlib.metadata.version('rapidocr'),'OCR_models':model_fingerprint,
             'used_ownership_cache':True,'used_external_OCR_cache':bool(texts_path),
             'input_contract':'PNG plus image-model predictions only; no CAD/annotation/Golden reads','official_score':None}
    save(output/'summary.json',summary)
    return summary


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--image',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--ownership',type=Path,help='Optional frozen image-model ownership prediction')
    parser.add_argument('--texts',type=Path,help='Optional hash-validated upstream model OCR cache')
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--config',type=Path,default=ROOT/'configs/pin_text_v1.json')
    parser.add_argument('--component-config',type=Path,default=ROOT/'configs/pin_text_component_v2.json')
    parser.add_argument('--skip-local',action='store_true',help='For comparing global OCR binding with local rereading')
    parser.add_argument('--enable-local',action='store_true',help='Experimental local rereading; default disabled after validation')
    args=parser.parse_args()
    if args.enable_local and args.skip_local:
        parser.error('--enable-local and --skip-local are mutually exclusive')
    ownership_path=args.ownership
    if ownership_path is None:
        folder=args.output/'ownership'
        subprocess.run([sys.executable,str(ROOT/'scripts/run_pin_ownership.py'),'--image',str(args.image),'--output',str(folder),'--device',args.device],check=True)
        ownership_path=folder/'pin_ownership.json'
    cfg=read(args.config)
    if args.enable_local:
        cfg['local_ocr']['enabled']=True
    summary=process(args.image,ownership_path,args.output,cfg,read(args.component_config),texts_path=args.texts,local=not args.skip_local)
    summary['used_ownership_cache']=args.ownership is not None
    save(args.output/'summary.json',summary)
    print(json.dumps({k:summary[k] for k in ['summary','used_ownership_cache','used_external_OCR_cache','official_score']},ensure_ascii=False))


if __name__=='__main__':
    main()
