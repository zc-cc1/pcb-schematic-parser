"""PNG -> frozen YOLO/SAHI + U-Net -> component-owned pin proposals.

Optional caches contain model predictions with image hashes. This entrypoint
never reads raw annotations, paired reference JSON, CAD, or SPICE netlists.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from PIL import Image
import torch

from pcb_parser.pin_ownership import normalized_components,associate_pins
from pcb_parser.pin_ownership_visual import draw_ownership
from pcb_parser.unet_model import WirePinUNet
from pcb_parser.unet_inference import predict_configured,decode_pins


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def validate_cache(payload,image_hash,size):
    if payload.get('image_sha256')!=image_hash or payload.get('size')!=size or payload.get('coordinate_origin')!='top_left':
        raise ValueError('Model prediction cache does not match image hash, size, or coordinate origin')


def predict_components(rgb,cfg,device):
    # Reuse the adopted detector implementation and all per-class thresholds.
    from run_baseline import YOLO,fused_predictions,whole_predictions,apply_class_confidence
    from sahi import AutoDetectionModel
    weights=ROOT/cfg['weights']
    whole=YOLO(str(weights))
    mode=cfg.get('recommended_mode','fused')
    if mode=='whole':
        predicted=whole_predictions(whole,np.ascontiguousarray(rgb[:,:,::-1]),cfg['detector'],device)
    elif mode=='fused':
        sliced=AutoDetectionModel.from_pretrained(model_type='ultralytics',model_path=str(weights),
              confidence_threshold=cfg.get('fusion',{}).get('sliced_confidence',cfg['detector']['confidence']),
              device=device,image_size=cfg['detector']['imgsz'])
        sliced.model.overrides.update({'iou':cfg['detector']['iou'],'max_det':cfg['detector']['max_det'],'half':cfg['detector']['half']})
        predicted=fused_predictions(whole,sliced,rgb,device,cfg)
    else:
        raise ValueError('Ownership runner supports the adopted whole/fused detector modes')
    return apply_class_confidence(predicted,cfg['detector'].get('class_confidence'))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--image',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--config',type=Path,default=ROOT/'configs/pin_ownership_v1.json')
    parser.add_argument('--detector-config',type=Path,default=ROOT/'configs/baseline_y11s.json')
    parser.add_argument('--unet-inference-config',type=Path,default=ROOT/'configs/unet_inference_v2.json')
    parser.add_argument('--unet-weights',type=Path)
    parser.add_argument('--detections',type=Path,help='Optional frozen image-model prediction cache')
    parser.add_argument('--pin-predictions',type=Path,help='Optional frozen image-model pin prediction cache')
    parser.add_argument('--component-records',type=Path,help='Optional image-validated OCR component identities')
    args=parser.parse_args()
    torch.set_num_threads(4)
    start=time.perf_counter()
    with Image.open(args.image) as image:
        rgb=np.asarray(image.convert('RGB'))
    height,width=rgb.shape[:2]
    size=[width,height]
    image_hash=sha(args.image)
    cfg=read(args.config)
    detector_cfg=read(args.detector_config)
    unet_cfg=read(args.unet_inference_config)
    detector_hash=sha(ROOT/detector_cfg['weights'])
    if detector_cfg.get('weights_sha256') and detector_hash!=detector_cfg['weights_sha256']:
        raise ValueError('Detector checkpoint does not match the frozen detector configuration')
    if args.detections:
        detector_payload=read(args.detections)
        validate_cache(detector_payload,image_hash,size)
        if detector_payload.get('weights_sha256')!=detector_hash:
            raise ValueError('Detector cache uses a different checkpoint')
    else:
        detections=predict_components(rgb,detector_cfg,args.device)
        detector_payload={'image_sha256':image_hash,'size':size,'coordinate_origin':'top_left',
                          'weights_sha256':detector_hash,'detections':detections,'source':'PNG_model_inference'}
    detections=detector_payload['detections']
    wire=None
    if args.pin_predictions:
        pin_payload=read(args.pin_predictions)
        validate_cache(pin_payload,image_hash,size)
        if pin_payload.get('weights_sha256')!=unet_cfg['checkpoint_sha256'] or pin_payload.get('inference_config_sha256')!=sha(args.unet_inference_config):
            raise ValueError('Pin cache does not match adopted checkpoint/inference settings')
        candidates=pin_payload['pin_candidates']
    else:
        weights=args.unet_weights or ROOT/'weights/unet_wire_pin_v1.pt'
        if not weights.is_file() and args.unet_weights is None:
            weights=ROOT/'runs/training/unet_wire_pin_v1/best.pt'
        if sha(weights)!=unet_cfg['checkpoint_sha256']:
            raise ValueError('U-Net checkpoint does not match calibrated inference settings')
        checkpoint=torch.load(weights,map_location='cpu',weights_only=True)
        if checkpoint['kind'] in {'cpu_smoke_only','partial_training_debug'}:
            raise ValueError('Debug checkpoint cannot be used for pin ownership inference')
        model=WirePinUNet(checkpoint['base_channels'])
        model.load_state_dict(checkpoint['state_dict'],strict=True)
        model.to(args.device).eval()
        probability=predict_configured(model,rgb,args.device,checkpoint['config'],unet_cfg)
        wire=probability[0]
        candidates=decode_pins(probability[1],unet_cfg['pin_threshold'],unet_cfg['pin_peak_kernel'],
                               unet_cfg['pin_refinement'],unet_cfg['pin_nms_radius'],unet_cfg['pin_candidate_limit'])
        pin_payload={'image_sha256':image_hash,'size':size,'coordinate_origin':'top_left',
                     'weights_sha256':sha(weights),'inference_config_sha256':sha(args.unet_inference_config),
                     'pin_candidates':candidates,'source':'PNG_model_inference'}
        args.output.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(args.output/'probabilities.npz',wire=wire,pin_tip=probability[1])
        Image.fromarray((wire>=unet_cfg['wire_threshold']).astype(np.uint8)*255).save(args.output/'wire_mask.png')
    records=None
    if args.component_records:
        payload=read(args.component_records)
        if payload.get('image_sha256')!=image_hash:
            raise ValueError('OCR component records do not match the input image')
        records=payload['records']
    components=normalized_components(detections,detector_cfg['class_mapping'],records)
    result=associate_pins(candidates,components,cfg,size,wire)
    result.update(image_sha256=image_hash,association_config_sha256=sha(args.config),
                  detector_weights_sha256=detector_hash,unet_weights_sha256=unet_cfg['checkpoint_sha256'],
                  used_detector_cache=bool(args.detections),used_pin_cache=bool(args.pin_predictions),
                  OCR_component_records_used=bool(args.component_records),wall_seconds=time.perf_counter()-start)
    save(args.output/'detector_predictions.json',detector_payload)
    save(args.output/'unet_predictions.json',pin_payload)
    save(args.output/'pin_ownership.json',result)
    draw_ownership(rgb,result,args.output/'ownership_overlay.png')
    save(args.output/'summary.json',{k:v for k,v in result.items() if k not in ['components','pin_candidates']})
    print(json.dumps({k:v for k,v in result.items() if k not in ['components','pin_candidates']},ensure_ascii=False))


if __name__=='__main__':
    main()
