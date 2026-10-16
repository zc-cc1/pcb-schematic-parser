"""Experimental cross-scale U-Net tip proposals from PNG/model outputs only."""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
from PIL import Image,ImageDraw,ImageFont
import torch

from run_pin_text_v2 import ROOT,read,save,sha
from run_submission_fields import fingerprint
from run_wire_graph import process as wire_process
from run_network_labels import process as labels_process
from run_pin_output_rules import process as rules_process
from pcb_parser.component_tip_refinement import refine_component_tips
from pcb_parser.unet_model import WirePinUNet
from pcb_parser.unet_inference import predict_at_scale,decode_pins


def load_probability(path):
    with np.load(path,allow_pickle=False) as array:
        if 'probability' in array:return array['probability']
        if 'wire' in array and 'pin_tip' in array:return np.stack([array['wire'],array['pin_tip']])
    raise ValueError('Unsupported probability cache')


def validate_probabilities(image,pin_path,native_path,zoom_path,metadata_path):
    pins=read(pin_path);meta=read(metadata_path);cfg=read(ROOT/'configs/unet_inference_v2.json')
    with Image.open(image) as source:size=list(source.size)
    if (pins['image_sha256']!=sha(image) or pins['size']!=size or pins['coordinate_origin']!='top_left'
            or meta['image_sha256']!=sha(image) or meta['size']!=size or meta['coordinate_origin']!='top_left'
            or meta['weights_sha256']!=cfg['checkpoint_sha256'] or not meta.get('prediction_only')
            or meta['original_pin_text_sha256']!=sha(pin_path)
            or meta['native_probability_sha256']!=sha(native_path) or meta['zoom_probability_sha256']!=sha(zoom_path)
            or meta['zoom_scale']!=1.5):
        raise ValueError('Image/model/pin/probability provenance mismatch')
    native,zoom=load_probability(native_path),load_probability(zoom_path)
    base=decode_pins(native[1],cfg['pin_threshold'],cfg['pin_peak_kernel'],cfg['pin_refinement'],cfg['pin_nms_radius'],cfg['pin_candidate_limit'])
    if len(base)!=len(pins['pin_candidates']) or any(a['point']!=b['point'] or a['confidence']!=b['confidence'] for a,b in zip(base,pins['pin_candidates'])):
        raise ValueError('Native probabilities do not reproduce frozen original model tips')
    records=read(pin_path.parent/'component_records.json');ocr=read(pin_path.parent/'texts.json')
    if records['image_sha256']!=sha(image) or ocr['image_sha256']!=sha(image) or not ocr.get('prediction_only'):
        raise ValueError('Component/OCR cache differs from image')
    return pins,native,zoom,records,ocr


def render_added(rgb,pins,audit,output):
    font=ImageFont.truetype('C:/Windows/Fonts/consola.ttf',13)
    canvas=Image.fromarray(rgb);draw=ImageDraw.Draw(canvas)
    for old in pins['pin_candidates']:
        x,y=old['point'];draw.ellipse((x-2,y-2,x+2,y+2),outline='#41719e',width=1)
    for added in audit['added_candidates']:
        x,y=added['point'];draw.ellipse((x-5,y-5,x+5,y+5),outline='#b92222',width=2)
    canvas.thumbnail((1600,1600));card=Image.new('RGB',(max(900,canvas.width),canvas.height+45),'white');card.paste(canvas)
    draw=ImageDraw.Draw(card);draw.text((8,canvas.height+5),'Blue: preserved tips | Red: experimental native+zoom U-Net peaks; no numbers guessed',font=font,fill='#333')
    draw.text((8,canvas.height+24),'New model peaks require visual/held-set review; image topology is not certified.',font=font,fill='#333')
    card.save(output/'tip_refinement_overlay.png')


def process(image,pin_path,native_path,zoom_path,metadata_path,output,cfg,device='cpu'):
    start=time.perf_counter();pins,native,zoom,records,ocr=validate_probabilities(image,pin_path,native_path,zoom_path,metadata_path)
    result,audit=refine_component_tips(pins,native,zoom,cfg,read(ROOT/'configs/pin_ownership_v1.json'))
    output.mkdir(parents=True,exist_ok=True)
    result.update(component_tip_refinement_config_sha256=fingerprint(cfg),original_pin_text_sha256=sha(pin_path),
                  status='component_tip_refinement_experiment_not_default')
    save(output/'pin_text_result.json',result);save(output/'component_records.json',records);save(output/'texts.json',ocr)
    save(output/'refinement_audit.json',{'image_sha256':sha(image),'size':pins['size'],'config_sha256':fingerprint(cfg),**audit})
    wire_cache=output/'wire_probability.npz';np.savez_compressed(wire_cache,wire=native[0])
    wire_metadata=output/'wire_probability.metadata.json'
    save(wire_metadata,{'image_sha256':sha(image),'size':pins['size'],'coordinate_origin':'top_left',
          'weights_sha256':read(ROOT/'configs/unet_inference_v2.json')['checkpoint_sha256'],
          'inference_config_sha256':sha(ROOT/'configs/unet_inference_v2.json'),'probabilities_sha256':sha(wire_cache),'prediction_only':True})
    wire=wire_process(image,output/'pin_text_result.json',wire_cache,output/'physical',read(ROOT/'configs/wire_graph_v1.json'),wire_metadata,device)
    save(output/'physical/summary.json',wire)
    labels_process(image,output/'physical/graph_proposals.json',output/'pin_text_result.json',output/'texts.json',output/'network',read(ROOT/'configs/network_labels_v1.json'))
    fields=rules_process(image,output/'network',output/'pin_text_result.json',output/'texts.json',output/'output_rules',read(ROOT/'configs/pin_output_rules_v1.json'))
    with Image.open(image) as source:render_added(np.asarray(source.convert('RGB')),pins,audit,output)
    # Original points/fields are preserved. Additional pin-protection regions
    # can change the derived graph, while the learned wire map remains fixed.
    summary={'image_sha256':sha(image),'size':pins['size'],'summary':audit['summary'],'config_sha256':fingerprint(cfg),
             'native_probability_sha256':sha(native_path),'zoom_probability_sha256':sha(zoom_path),
             'metadata_sha256':sha(metadata_path),'original_pin_text_sha256':sha(pin_path),
             'power_guard_after':fields['power_guard'],'conservative_export':fields['conservative'],'candidate_export':fields['candidate'],
             'physical_summary':wire['summary'],'default_pipeline_adopted':False,'new_training':False,
             'learned_wire_probability_preserved':True,'body_guard_and_graph_may_change_with_added_tips':True,
             'input_contract':'PNG and image model outputs only; no source annotation or Golden reads',
             'official_PinF1':None,'official_NetF1':None,'complete_competition_JSON':False,'wall_seconds':time.perf_counter()-start}
    save(output/'summary.json',summary);return summary


def main():
    p=argparse.ArgumentParser();p.add_argument('--image',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cpu');p.add_argument('--pin-text',type=Path);p.add_argument('--native-probabilities',type=Path)
    p.add_argument('--zoom-probabilities',type=Path);p.add_argument('--metadata',type=Path)
    p.add_argument('--config',type=Path,default=ROOT/'configs/component_tip_refinement_v1.json');a=p.parse_args()
    provided=[a.pin_text,a.native_probabilities,a.zoom_probabilities,a.metadata]
    if any(provided) and not all(provided):p.error('Cached mode needs all four image-model inputs')
    if all(provided):pin,native,zoom,metadata=provided
    else:
        folder=a.output/'upstream'
        subprocess.run([sys.executable,str(ROOT/'scripts/run_pin_text_v2.py'),'--image',str(a.image),'--output',str(folder),'--device',a.device],check=True)
        pin=folder/'pin_text_result.json';native=folder/'ownership/probabilities.npz';zoom=a.output/'zoom15_probability.npz';metadata=a.output/'probability_inputs.json'
        weights=ROOT/'weights/unet_wire_pin_v1.pt'
        if not weights.is_file():weights=ROOT/'runs/training/unet_wire_pin_v1/best.pt'
        if sha(weights)!=read(ROOT/'configs/unet_inference_v2.json')['checkpoint_sha256']:raise ValueError('Unknown U-Net checkpoint')
        checkpoint=torch.load(weights,map_location='cpu',weights_only=True);model=WirePinUNet(checkpoint['base_channels'])
        model.load_state_dict(checkpoint['state_dict'],strict=True);model.to(a.device).eval();torch.set_num_threads(4)
        with Image.open(a.image) as source:rgb=np.asarray(source.convert('RGB'))
        probability=predict_at_scale(model,rgb,a.device,checkpoint['config']['tile_size'],checkpoint['config']['stride'],1.5)
        np.savez_compressed(zoom,probability=probability)
        save(metadata,{'image_sha256':sha(a.image),'size':[rgb.shape[1],rgb.shape[0]],'coordinate_origin':'top_left',
            'weights_sha256':sha(weights),'original_pin_text_sha256':sha(pin),'native_probability_sha256':sha(native),
            'zoom_probability_sha256':sha(zoom),'zoom_scale':1.5,'prediction_only':True,'source':'Current_PNG_model_inference'})
    summary=process(a.image,pin,native,zoom,metadata,a.output,read(a.config),a.device)
    summary['used_external_model_caches']=bool(all(provided));save(a.output/'summary.json',summary)
    print({'summary':summary['summary'],'candidate_export':summary['candidate_export'],'used_external_model_caches':bool(all(provided))},flush=True)


if __name__=='__main__':main()
