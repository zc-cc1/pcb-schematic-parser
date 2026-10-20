"""PNG -> frozen image models -> provisional wire graph; no reference reads."""
import argparse
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
from PIL import Image
from pcb_parser.wire_graph import build_wire_graph, detect_junction_dots
from pcb_parser.wire_graph_artifacts import render_graph,partial_submission


def read(path):return json.loads(path.read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path,payload):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(payload,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def validate_cache(metadata,path,image_hash,size,model_hash,inference_hash):
    if metadata.get('image_sha256')!=image_hash or metadata.get('size')!=size or metadata.get('coordinate_origin')!='top_left':
        raise ValueError('Wire cache does not match image hash, size or coordinate origin')
    if metadata.get('weights_sha256')!=model_hash or metadata.get('inference_config_sha256')!=inference_hash or not metadata.get('prediction_only'):
        raise ValueError('Wire cache does not match the adopted model or inference configuration')
    if metadata.get('probabilities_sha256')!=sha(path):raise ValueError('Wire probability cache bytes differ')


def predict_wire(rgb,device,inference):
    import torch
    from pcb_parser.unet_checkpoint import load_wire_pin_checkpoint
    from pcb_parser.unet_inference import predict_configured
    torch.set_num_threads(4)
    weights=ROOT/'weights/unet_wire_pin_v1.pt'
    if not weights.is_file():weights=ROOT/'runs/training/unet_wire_pin_v1/best.pt'
    if sha(weights)!=inference['checkpoint_sha256']:raise ValueError('U-Net checkpoint changed')
    model,checkpoint,_unet_info=load_wire_pin_checkpoint(weights,device)
    return predict_configured(model,rgb,device,checkpoint['config'],inference)[0]


def process(image_path,pin_path,wire_path,output,cfg,metadata_path=None,device='cpu',junction_predictor=None):
    start=time.perf_counter()
    with Image.open(image_path) as image:rgb=np.asarray(image.convert('RGB'))
    image_hash=sha(image_path);size=[rgb.shape[1],rgb.shape[0]]
    pin_result=read(pin_path)
    if pin_result.get('image_sha256')!=image_hash or pin_result.get('size')!=size or pin_result.get('coordinate_origin')!='top_left':
        raise ValueError('Pin text predictions do not match input image')
    inference_path=ROOT/'configs/unet_inference_v2.json';inference=read(inference_path)
    if wire_path:
        metadata=read(metadata_path or wire_path.with_suffix('.json'))
        validate_cache(metadata,wire_path,image_hash,size,inference['checkpoint_sha256'],sha(inference_path))
        with np.load(wire_path,allow_pickle=False) as cached:wire=cached['wire']
    else:
        wire=predict_wire(rgb,device,inference)
        output.mkdir(parents=True,exist_ok=True)
        wire_path=output/'wire_probability.npz';np.savez_compressed(wire_path,wire=wire)
        metadata={'image_sha256':image_hash,'size':size,'coordinate_origin':'top_left','weights_sha256':inference['checkpoint_sha256'],
                  'inference_config_sha256':sha(inference_path),'probabilities_sha256':sha(wire_path),'prediction_only':True,'source':'PNG_U-Net_native_model_inference'}
        save(wire_path.with_suffix('.json'),metadata)
    dot_points=detect_junction_dots(rgb,cfg) if float(cfg.get('dot_split_max_distance_px') or 0)>0 else []
    result,masks=build_wire_graph(wire,pin_result,cfg,dot_points=dot_points)
    if cfg.get('junction_model_available'):
        if junction_predictor is None:
            from pcb_parser.junction_classifier import JunctionPredictor
            junction_predictor=JunctionPredictor(ROOT/cfg['junction_checkpoint'],device,cfg['junction_checkpoint_sha256'])
        if junction_predictor.sha256!=cfg['junction_checkpoint_sha256']:raise ValueError('Unexpected junction model')
        decisions=junction_predictor.predict(rgb,result['junction_candidates'],cfg['junction_confidence_threshold'])
        result,masks=build_wire_graph(wire,pin_result,cfg,decisions,dot_points=dot_points)
        result['junction_model_sha256']=junction_predictor.sha256
        save(output/'junction_predictions.json',{'decisions':decisions,'image_sha256':image_hash,
             'model_sha256':junction_predictor.sha256,'source':'RGB model inference, no reviewed labels read'})
    result.update(image_sha256=image_hash,pin_input_sha256=sha(pin_path),wire_cache_sha256=sha(wire_path),
                  unet_weights_sha256=inference['checkpoint_sha256'],wire_graph_config_sha256=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest())
    output.mkdir(parents=True,exist_ok=True)
    save(output/'graph_proposals.json',result)
    for name,mask in masks.items():Image.fromarray(mask.astype(np.uint8)*255).save(output/f'{name}.png')
    render_graph(rgb,result,output/'wire_graph_overlay.png')
    records_path=pin_path.parent/'component_records.json'
    records=read(records_path)
    if records.get('image_sha256')!=image_hash:raise ValueError('Component identity records do not match image')
    components={c['detection_index']:c for c in pin_result['components']}
    for record in records['records']:
        component=components.get(record['detection_index'])
        if not component or record['component_id']!=component['component_id'] or not np.allclose(record['bbox_top_left'],component['bbox_top_left'],atol=.05,rtol=0):
            raise ValueError('Component identities do not match the frozen pin predictions')
    submission,coverage=partial_submission(pin_result,result,records['records'])
    save(output/'partial_submission.json',submission);save(output/'export_coverage.json',coverage)
    review=[{'kind':'junction',**j} for j in result['junction_candidates'] if not j.get('resolved',False)]+[{'kind':'gap',**g} for g in result['gap_candidates']]
    review += [{'kind':'pin_attachment',**p} for p in result['pin_attachments'] if p['attachment_status']!='attached']
    save(output/'review_queue.json',{'items':review,'count':len(review),'automatic_label_changes':0})
    summary={'summary':result['summary'],'image_sha256':image_hash,'pin_input_sha256':sha(pin_path),'wire_input_metadata':metadata,
             'wall_seconds':time.perf_counter()-start,'graph_config':cfg,'networkx_version':importlib.metadata.version('networkx'),
             'scikit_image_version':importlib.metadata.version('scikit-image'),'exported_components':coverage['exported_components'],
             'exported_pins':coverage['exported_pins'],'exported_nets':coverage['exported_nets'],
             'input_contract':'PNG plus image-model predictions only; no raw annotations/CAD/SPICE reads','official_NetF1':None}
    save(output/'summary.json',summary)
    return summary


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--image',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--pin-text',type=Path,help='Optional hash-validated frozen model pin-text prediction')
    parser.add_argument('--wire-cache',type=Path,help='Optional U-Net wire probability cache, with its .json metadata')
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--config',type=Path,default=ROOT/'configs/wire_graph_v1.json')
    args=parser.parse_args()
    pin_path=args.pin_text;wire_path=args.wire_cache;metadata_path=None
    if pin_path is None:
        upstream=args.output/'upstream'
        subprocess.run([sys.executable,str(ROOT/'scripts/run_pin_text.py'),'--image',str(args.image),'--output',str(upstream),'--device',args.device],check=True)
        pin_path=upstream/'pin_text_result.json'
        if wire_path is None:
            wire_path=upstream/'ownership/probabilities.npz'
            origin=read(upstream/'ownership/summary.json')
            inf=ROOT/'configs/unet_inference_v2.json'
            metadata_path=args.output/'wire_input_metadata.json'
            save(metadata_path,{'image_sha256':origin['image_sha256'],'size':origin['size'],'coordinate_origin':'top_left',
                 'weights_sha256':origin['unet_weights_sha256'],'inference_config_sha256':sha(inf),
                 'probabilities_sha256':sha(wire_path),'prediction_only':True,'source':'Current_PNG_upstream_U-Net_inference'})
    summary=process(args.image,pin_path,wire_path,args.output,read(args.config),metadata_path,args.device)
    summary['used_frozen_pin_input']=args.pin_text is not None;summary['used_external_wire_cache']=args.wire_cache is not None
    save(args.output/'summary.json',summary)
    print(json.dumps({k:summary[k] for k in ['summary','exported_components','exported_pins','exported_nets','used_frozen_pin_input','used_external_wire_cache','official_NetF1']},ensure_ascii=False))


if __name__=='__main__':main()
