"""PNG -> frozen learned pipeline -> conservative fields and review queue."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from run_network_labels import ROOT, read, save, sha, validate_inputs, logical_partial_export
from pcb_parser.submission_fields import complete_fields, network_field_queue
from pcb_parser.label_roi_ocr import collect_label_readings
from pcb_parser.pin_text_ocr import create_engine


def fingerprint(cfg):
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()


def render_review(rgb, queue, output, count):
    font = ImageFont.truetype(str(Path(__file__).resolve().parents[1] / 'assets/fonts/DejaVuSansMono.ttf'), 13)
    rows = queue['networks'][:count]
    cards = []
    for rank, row in enumerate(rows, 1):
        points = [p['point'] for p in row['pin_candidates']]
        boxes = [p['component_bbox_top_left'] for p in row['pin_candidates'] if p['component_bbox_top_left']]
        xs = [p[0] for p in points] + [v for b in boxes for v in [b[0], b[2]]]
        ys = [p[1] for p in points] + [v for b in boxes for v in [b[1], b[3]]]
        crop = [max(0, int(min(xs))-30), max(0, int(min(ys))-30),
                min(rgb.shape[1], int(max(xs))+31), min(rgb.shape[0], int(max(ys))+31)]
        image = Image.fromarray(rgb).crop(crop); draw = ImageDraw.Draw(image)
        for i, pin in enumerate(row['pin_candidates'], 1):
            x, y = pin['point'][0]-crop[0], pin['point'][1]-crop[1]
            draw.ellipse((x-4, y-4, x+4, y+4), outline='#bd2424' if pin['missing_fields'] else '#218449', width=2)
            draw.text((x+5, y), str(i), font=font, fill='#a72424', stroke_width=1, stroke_fill='white')
        image.thumbnail((650, 370))
        card = Image.new('RGB', (680, 470), 'white'); card.paste(image, (15, 30))
        draw = ImageDraw.Draw(card)
        draw.text((8, 6), f"{rank}. {row['network_id']} | {row['pin_count']} pins", font=font, fill='#333')
        missing=row['missing_field_counts']
        text='Missing identity: {} | number: {} | pinname: {}'.format(
            missing.get('component_identity_unresolved',0),missing.get('pin_number_unresolved',0),missing.get('pin_name_unresolved',0))
        draw.text((8, 406), text, font=font, fill='#8f2323')
        draw.text((8, 426), 'Topology blockers: '+str(row['topology_blockers']), font=font, fill='#333')
        draw.text((8, 446), 'Red points need fields; listed topology is a model proposal.', font=font, fill='#555')
        name=f'network_fields_{rank:02d}.png'; card.save(output/name)
        row['review_card']=name; row['crop_top_left']=crop; cards.append(card)
    if cards:
        sheet=Image.new('RGB',(1360,470*((len(cards)+1)//2)),'white')
        for i,card in enumerate(cards):sheet.paste(card,((i%2)*680,(i//2)*470))
        sheet.save(output/'field_review_sheet.png')


def process(image_path, network_folder, pin_path, texts_path, output, cfg, engine=None, models=None):
    start=time.perf_counter()
    if cfg['local_ocr'].get('apply_spelling_changes') or cfg['single_tip_auto_number'] != '1' or cfg['blank_pinname_policy'] != 'keep_unresolved':
        raise ValueError('Unsupported field policy; spelling changes and blank pinname filling require a separately validated implementation')
    with Image.open(image_path) as source:rgb=np.asarray(source.convert('RGB'))
    size=[rgb.shape[1],rgb.shape[0]]; image_hash=sha(image_path)
    graph_path=network_folder/'logical_graph_proposals.json'
    graph, pins, ocr=read(graph_path),read(pin_path),read(texts_path)
    records=read(pin_path.parent/'component_records.json')
    validate_inputs(image_hash,size,graph,pins,ocr,records)
    if graph['pin_input_sha256'] != sha(pin_path) or graph['OCR_input_sha256'] != sha(texts_path):
        raise ValueError('Network graph pin/OCR bytes changed')
    if graph.get('label_config_sha256') != fingerprint(read(ROOT/'configs/network_labels_v1.json')):
        raise ValueError('Unknown network label configuration')
    if graph.get('unet_weights_sha256') != read(ROOT/'configs/unet_inference_v2.json')['checkpoint_sha256']:
        raise ValueError('Unknown U-Net checkpoint')
    valid={fingerprint(read(ROOT/'configs'/name)) for name in ['wire_graph_v1.json','wire_graph_v2.json']}
    if graph.get('wire_graph_config_sha256') not in valid:raise ValueError('Unknown wire graph configuration')
    updated, updated_graph, updated_records, identities=complete_fields(pins,graph,records['records'],cfg)
    baseline, before=logical_partial_export(pins,graph,records['records'])
    submission, coverage=logical_partial_export(updated,updated_graph,updated_records)
    coverage['field_completion']=identities['summary']
    coverage['exported_network_ids']=sorted(submission['nets'])
    coverage['blank_pin_names_guessed']=0
    output.mkdir(parents=True,exist_ok=True)
    updated.update(submission_fields_config_sha256=fingerprint(cfg),original_pin_text_sha256=sha(pin_path),
                   status='submission_fields_prediction_only_incomplete')
    save(output/'completed_pin_text_result.json',updated)
    save(output/'completed_component_records.json',{'image_sha256':image_hash,'records':updated_records})
    save(output/'identity_proposals.json',{'image_sha256':image_hash,'coordinate_origin':'top_left',**identities})
    save(output/'partial_submission.json',submission); save(output/'export_coverage.json',coverage)
    queue=network_field_queue(updated,updated_graph,updated_records,coverage)
    queue.update(image_sha256=image_hash,size=size,coordinate_origin='top_left')
    render_review(rgb,queue,output,cfg['review_network_count']);save(output/'network_field_review_queue.json',queue)
    local={'disabled':True,'roi_count':0,'suggestion_count':0,'spelling_changes_applied':0}
    if cfg['local_ocr']['enabled']:
        if engine is None:engine,models=create_engine({'onnx_threads':cfg['local_ocr']['onnx_threads']})
        local=collect_label_readings(rgb,graph.get('label_sites',[]),engine,cfg['local_ocr'])
    local.update(image_sha256=image_hash,coordinate_origin='top_left',OCR_models=models)
    save(output/'label_roi_readings.json',local)
    summary={'image_sha256':image_hash,'size':size,'summary':identities['summary'],
             'input_network_sha256':sha(graph_path),'input_pin_text_sha256':sha(pin_path),'input_OCR_sha256':sha(texts_path),
             'config_sha256':fingerprint(cfg),'before_export':{k:before[k] for k in ['exported_components','exported_pins','exported_nets','pin_reason_counts']},
             'after_export':{k:coverage[k] for k in ['exported_components','exported_pins','exported_nets','pin_reason_counts']},
             'multi_pin_networks':queue['multi_pin_networks'],'topology_clear_but_field_incomplete':queue['topology_clear_but_field_incomplete'],
             'local_OCR_rois':local['roi_count'],'local_spelling_suggestions':local['suggestion_count'],'spelling_changes_applied':0,
             'physical_edges_and_pin_points_preserved':True,'network_membership_changed':False,
             'input_contract':'PNG and image-model predictions only; no CAD/source annotations/Golden reads',
             'generated_number_is_recognition_accuracy':False,'reference_NetF1':None,'complete_target_JSON':False,
             'wall_seconds':time.perf_counter()-start}
    save(output/'summary.json',summary)
    return summary


def main():
    p=argparse.ArgumentParser();p.add_argument('--image',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--network-result',type=Path);p.add_argument('--pin-text',type=Path);p.add_argument('--texts',type=Path)
    p.add_argument('--device',default='cpu');p.add_argument('--skip-label-roi',action='store_true')
    p.add_argument('--config',type=Path,default=ROOT/'configs/submission_fields_v1.json');a=p.parse_args()
    provided=[a.network_result,a.pin_text,a.texts]
    if any(provided) and not all(provided):p.error('Cached mode requires network result, pin text and OCR together')
    if all(provided):folder,pins,texts=provided
    else:
        folder=a.output/'network'
        subprocess.run([sys.executable,str(ROOT/'scripts/run_network_labels.py'),'--image',str(a.image),
                        '--device',a.device,'--output',str(folder)],check=True)
        pins=folder/'physical/upstream/pin_text_result.json';texts=folder/'physical/upstream/texts.json'
    cfg=read(a.config)
    if a.skip_label_roi:cfg['local_ocr']['enabled']=False
    summary=process(a.image,folder,pins,texts,a.output,cfg)
    summary['used_external_model_caches']=bool(all(provided));save(a.output/'summary.json',summary)
    print(json.dumps({k:summary[k] for k in ['summary','after_export','used_external_model_caches']},ensure_ascii=True),flush=True)


if __name__=='__main__':main()
