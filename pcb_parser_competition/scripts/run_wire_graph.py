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
from pcb_parser.wire_graph import build_wire_graph
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
    from pcb_parser.unet_model import WirePinUNet
    from pcb_parser.unet_inference import predict_configured
    torch.set_num_threads(4)
    from pcb_parser.unet_checkpoint import load_wire_pin_checkpoint
    weights=Path(inference['checkpoint']) if inference.get('checkpoint') else ROOT/'weights/unet_wire_pin_v1.pt'
    if not weights.is_file() and not inference.get('checkpoint'):
        weights=ROOT/'runs/training/unet_wire_pin_v1/best.pt'
    if sha(weights)!=inference['checkpoint_sha256']:raise ValueError('U-Net checkpoint changed')
    model,checkpoint,_unet_info=load_wire_pin_checkpoint(weights,device)
    return predict_configured(model,rgb,device,checkpoint['config'],inference)[0]


def process(image_path,pin_path,wire_path,output,cfg,metadata_path=None,device='cpu',junction_predictor=None,inference_path=None,box_library=None):
    start=time.perf_counter()
    with Image.open(image_path) as image:rgb=np.asarray(image.convert('RGB'))
    image_hash=sha(image_path);size=[rgb.shape[1],rgb.shape[0]]
    pin_result=read(pin_path)
    if pin_result.get('image_sha256')!=image_hash or pin_result.get('size')!=size or pin_result.get('coordinate_origin')!='top_left':
        raise ValueError('Pin text predictions do not match input image')
    inference_path=Path(inference_path) if inference_path else ROOT/'configs/unet_inference_v2.json';inference=read(inference_path)
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
    result,masks=build_wire_graph(wire,pin_result,cfg)
    if cfg.get('junction_model_available'):
        if junction_predictor is None:
            from pcb_parser.junction_classifier import JunctionPredictor
            junction_predictor=JunctionPredictor(ROOT/cfg['junction_checkpoint'],device,cfg['junction_checkpoint_sha256'])
        if junction_predictor.sha256!=cfg['junction_checkpoint_sha256']:raise ValueError('Unexpected junction model')
        decisions=junction_predictor.predict(rgb,result['junction_candidates'],cfg['junction_confidence_threshold'])
        result,masks=build_wire_graph(wire,pin_result,cfg,decisions)
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
    # R15 allows auto-generated numbers for pins whose number is missing.  Applying the
    # rule here (before export) also unblocks the net/hypergraph item: without numbers no
    # pin is exportable and therefore no net is exportable.
    if cfg.get('pin_completion_enabled'):
        from pcb_parser.pin_completion import complete_pins
        completion_audit = complete_pins(pin_result,
                                         two_pin_types=cfg.get('pin_completion_two_pin_types'),
                                         ic_types=cfg.get('pin_completion_ic_types'))
    else:
        completion_audit = None
    from pcb_parser.pin_numbering import assign_numbers
    inferred_numbers, numbering_audit = assign_numbers(pin_result['pin_candidates'], pin_result['components'],
        geometric_types=cfg.get('geometric_numbering_types'),
        dip_types=cfg.get('dip_numbering_types'))
    if completion_audit is not None:
        numbering_audit['pin_completion'] = completion_audit
    dip_numbered = numbering_audit.get('pins_dip_numbered', 0)
    for pin in pin_result['pin_candidates']:
        number = inferred_numbers.get(pin['candidate_id'])
        if number is None:
            continue
        # `assign_numbers` owns the precedence decision: it keeps an observed number
        # unless the component is a two-pin passive or an IC numbered by the package
        # convention, where the derived number is the better estimate.  Applying its
        # answer unconditionally keeps that decision in one place.
        if pin.get('number') is None:
            pin['number_source'] = 'spatial_reading_order_auto'
        elif str(pin['number']) != str(number):
            pin['number_source'] = 'package_or_geometry_override'
        pin['number'] = number
    # Component library (optional, off unless a config is passed): on datasheet pages the
    # schematic prints pin NAMES but not numbers, so the printed name is the only route to
    # a number.  Runs after `assign_numbers` so it can overrule both the observed number
    # and the package convention, and before the number->name backfill so a real library
    # name is never replaced by a bare number.
    if box_library is not None:
        from pcb_parser.box_ic_library import apply_library
        texts_path=pin_path.parent/'texts.json'
        tokens=read(texts_path).get('texts') or [] if texts_path.is_file() else []
        library,library_config=box_library
        numbering_audit['box_library']=apply_library(pin_result,tokens,library,library_config)
    # Pin names: 43.9% of the public reference pins use the pin number itself as the
    # name, and 0% are empty.  Supplying the number when no name text was bound keeps
    # the field non-null (the schema requires a string) and matches the reference
    # convention for those pins.  Anything the OCR actually read is never overwritten.
    name_filled = 0
    for pin in pin_result['pin_candidates']:
        if (pin.get('pinname') or '').strip():
            continue
        if pin.get('number') is not None:
            pin['pinname'] = str(pin['number'])
            pin['pinname_source'] = 'pin_number_fallback'
            name_filled += 1
    numbering_audit['pin_names_backfilled_from_number'] = name_filled
    if cfg.get('pin_tip_refine_enabled'):
        from pcb_parser.pin_tip_refine import refine_points
        gray = np.asarray(Image.fromarray(rgb).convert('L'))
        refine_audit = refine_points(pin_result, gray, enabled_types=cfg.get('pin_tip_refine_types'))
        numbering_audit['pin_tip_refine'] = refine_audit
    submission,coverage=partial_submission(pin_result,result,records['records'],
        auto_id_for_unnamed=bool(cfg.get('auto_id_for_unnamed_components',False)),
        auto_id_prefix=str(cfg.get('auto_id_prefix','DET')),
        gates=cfg.get('export_gates'))
    coverage['pin_numbering'] = numbering_audit
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
    parser.add_argument('--unet-inference-config',type=Path,default=None,help='U-Net inference config the wire cache was produced with')
    parser.add_argument('--box-library-config',type=Path,default=None,
                        help='Optional component-library config; omitted means the library stage is skipped entirely')
    args=parser.parse_args()
    box_library=None
    if args.box_library_config is not None:
        from pcb_parser.box_ic_library import library_from_config
        box_library=library_from_config(args.box_library_config)
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
    summary=process(args.image,pin_path,wire_path,args.output,read(args.config),metadata_path,args.device,inference_path=args.unet_inference_config,box_library=box_library)
    summary['used_frozen_pin_input']=args.pin_text is not None;summary['used_external_wire_cache']=args.wire_cache is not None
    summary['used_component_library']=box_library is not None
    save(args.output/'summary.json',summary)
    print(json.dumps({k:summary[k] for k in ['summary','exported_components','exported_pins','exported_nets','used_frozen_pin_input','used_external_wire_cache','official_NetF1']},ensure_ascii=False))


if __name__=='__main__':main()
