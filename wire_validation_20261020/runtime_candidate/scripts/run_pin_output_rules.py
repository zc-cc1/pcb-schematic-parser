"""PNG or integrity-checked model cache -> conservative and candidate JSONs."""
import argparse
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from run_pin_text_v2 import ROOT, read, save, sha
from run_submission_fields import fingerprint
from run_network_labels import validate_inputs, logical_partial_export
from pcb_parser.submission_fields import complete_fields
from pcb_parser.pin_output_rules import complete_pin_output_names, power_network_guard, apply_power_export_gate, compress_candidate_edges
from pcb_parser.dataset_converter import validate_output


def validate_cache(image_path, network_folder, pin_path, texts_path):
    with Image.open(image_path) as source:
        rgb = np.asarray(source.convert('RGB'))
    graph_path = network_folder/'logical_graph_proposals.json'
    graph, pins, ocr = read(graph_path), read(pin_path), read(texts_path)
    records = read(pin_path.parent/'component_records.json')
    validate_inputs(sha(image_path), [rgb.shape[1], rgb.shape[0]], graph, pins, ocr, records)
    if graph['pin_input_sha256'] != sha(pin_path) or graph['OCR_input_sha256'] != sha(texts_path):
        raise ValueError('Network pin/OCR input bytes changed')
    if graph.get('label_config_sha256') != fingerprint(read(ROOT/'configs/network_labels_v1.json')):
        raise ValueError('Unknown network label configuration')
    if graph.get('unet_weights_sha256') != read(ROOT/'configs/unet_inference_v2.json')['checkpoint_sha256']:
        raise ValueError('Unknown U-Net checkpoint')
    if graph.get('wire_graph_config_sha256') not in {fingerprint(read(ROOT/'configs'/n)) for n in ['wire_graph_v1.json','wire_graph_v2.json']}:
        raise ValueError('Unknown wire graph configuration')
    return rgb, graph, pins, records['records'], graph_path


def render_network_cards(rgb, pins, graph, records, submission, output, limit):
    pmap = {p['candidate_id']: p for p in pins['pin_candidates']}
    cmap = {r['component_id']: r for r in records}
    edges = {e['edge_id']: e for e in graph['edges']}
    font = ImageFont.truetype(str(Path(__file__).resolve().parents[1] / 'assets/fonts/DejaVuSansMono.ttf'), 14)
    rows = []
    for net in graph['networks']:
        if net['network_id'] not in submission['nets'] or len(rows) >= limit:
            continue
        points = [p for e in net['edge_ids'] for p in edges[e]['polyline']]
        points += [pmap[c]['point'] for c in net['pin_candidate_ids']]
        # An ownerless candidate has no component box; a review card must never crash on it
        boxes = [cmap[pmap[c]['owner']]['bbox_top_left'] for c in net['pin_candidate_ids']
                 if pmap[c].get('owner') in cmap]
        xs = [p[0] for p in points]+[v for b in boxes for v in [b[0],b[2]]]
        ys = [p[1] for p in points]+[v for b in boxes for v in [b[1],b[3]]]
        crop = [max(0,int(min(xs))-24),max(0,int(min(ys))-24),
                min(rgb.shape[1],int(max(xs))+25),min(rgb.shape[0],int(max(ys))+25)]
        raw = Image.fromarray(rgb).crop(crop)
        overlay = raw.copy(); draw = ImageDraw.Draw(overlay)
        for eid in net['edge_ids']:
            line = [(x-crop[0],y-crop[1]) for x,y in edges[eid]['polyline']]
            if len(line)>1: draw.line(line,fill='#258b7d',width=2)
        members = []
        for i, cid in enumerate(net['pin_candidate_ids'],1):
            pin=pmap[cid]; x,y=pin['point'][0]-crop[0],pin['point'][1]-crop[1]
            draw.ellipse((x-5,y-5,x+5,y+5),outline='#bb2424',width=2)
            draw.text((x+7,y+3),str(i),fill='#b21b1b',font=font,stroke_width=1,stroke_fill='white')
            members.append({'candidate_id':cid,'reference':f"{pin['owner']}.{pin['number']}",
                            'pinname':pin['pinname'],'point_top_left':pin['point'],
                            'pinname_status':pin.get('pinname_status')})
        scale=min(2,650/raw.width,520/raw.height)
        dims=(round(raw.width*scale),round(raw.height*scale))
        raw=raw.resize(dims,Image.Resampling.NEAREST);overlay=overlay.resize(dims,Image.Resampling.NEAREST)
        card=Image.new('RGB',(max(940,dims[0]*2+36),dims[1]+125+20*len(members)),'white')
        draw=ImageDraw.Draw(card);draw.text((12,8),net['network_id']+' | candidate profile; not an official score',font=font,fill='#333')
        draw.text((12,30),'Original PNG',font=font,fill='#333');draw.text((dims[0]+24,30),'Model wire and unchanged U-Net tips',font=font,fill='#333')
        card.paste(raw,(12,54));card.paste(overlay,(dims[0]+24,54));draw=ImageDraw.Draw(card)
        for i,m in enumerate(members):
            draw.text((12,dims[1]+65+20*i),f"{i+1}: {m['reference']}  pinname={m['pinname']}  ({m['pinname_status']})",font=font,fill='#333')
        draw.text((12,card.height-28),'All model net members retained. Number-as-name remains a provisional passive convention.',font=font,fill='#8f4320')
        name=f'candidate_net_{len(rows)+1:02d}.png';card.save(output/name)
        rows.append({'network_id':net['network_id'],'hyperGraph':submission['nets'][net['network_id']]['hyperGraph'],
                     'members':members,'crop_top_left':crop,'review_card':name,
                     'all_model_members_included':True,'visually_reviewed':False,
                     'official_pinname_verified':False,'whole_image_complete':False})
    save(output/'network_review_manifest.json',{'networks':rows,'reference_annotations_used':False})


def process(image_path, network_folder, pin_path, texts_path, output, cfg):
    start=time.perf_counter()
    rgb, graph, original_pins, records, graph_path=validate_cache(image_path,network_folder,pin_path,texts_path)
    fields, graph, records, fields_audit=complete_fields(original_pins,graph,records,read(ROOT/'configs/submission_fields_v1.json'))
    before, before_coverage=logical_partial_export(fields,graph,records)
    strict, strict_graph, candidate, candidate_graph, audit=complete_pin_output_names(fields,graph,records,cfg)
    guard=power_network_guard(strict,strict_graph,records,cfg)
    strict_json, strict_coverage=logical_partial_export(strict,strict_graph,records)
    candidate_json, candidate_coverage=logical_partial_export(candidate,candidate_graph,records)
    pre_guard_candidate_nets=sorted(candidate_json['nets'])
    strict_json, strict_coverage=apply_power_export_gate(strict_json,strict_coverage,guard)
    candidate_json, candidate_coverage=apply_power_export_gate(candidate_json,candidate_coverage,guard)
    candidate_json, geometry=compress_candidate_edges(candidate_json,candidate_graph,fields['size'][1],cfg['candidate_edge_max_deviation_px'])
    for payload, coverage, name in [(strict_json,strict_coverage,'conservative'),(candidate_json,candidate_coverage,'candidate')]:
        errors,bounds=validate_output(payload,*fields['size'])
        if errors or bounds:raise ValueError('Output violates the local JSON contract')
        coverage.update(profile=name,complete_competition_JSON=False,official_evaluator_verified=False,
                        passive_pinname_rule_provisional=(name=='candidate'),recognition_accuracy_gain_claimed=False)
    output.mkdir(parents=True,exist_ok=True)
    provenance={'image_sha256':sha(image_path),'size':fields['size'],'coordinate_origin':'top_left',
                'input_pin_sha256':sha(pin_path),'input_network_sha256':sha(graph_path),'input_OCR_sha256':sha(texts_path),
                'output_rules_config_sha256':fingerprint(cfg),'prediction_only':True}
    for result in [strict,candidate]:result.update(**provenance)
    save(output/'conservative_pin_text_result.json',strict);save(output/'candidate_pin_text_result.json',candidate)
    save(output/'completed_component_records.json',{'image_sha256':sha(image_path),'records':records})
    save(output/'partial_submission.json',strict_json);save(output/'export_coverage.json',strict_coverage)
    save(output/'submission_candidate.json',candidate_json);save(output/'candidate_export_coverage.json',candidate_coverage)
    save(output/'pin_field_changes.json',{**provenance,**audit,'prior_fields':fields_audit['summary']})
    save(output/'power_review_queue.json',{**provenance,**guard})
    save(output/'candidate_edge_geometry.json',{**provenance,**geometry})
    render_network_cards(rgb,candidate,candidate_graph,records,candidate_json,output,cfg['review_network_count'])
    brief=lambda c:{k:c[k] for k in ['exported_components','exported_pins','exported_nets','pin_reason_counts']}
    summary={**provenance,'before':brief(before_coverage),'conservative':brief(strict_coverage),'candidate':brief(candidate_coverage),
             'name_completion':audit['summary'],'power_guard':guard['summary'],
             'candidate_edge_geometry':{k:geometry[k] for k in ['input_pixel_segments','output_segments','tolerance_original_px']},
             'pre_guard_candidate_network_ids':pre_guard_candidate_nets,
             'candidate_exported_network_ids':sorted(candidate_json['nets']),
             'power_guard_excluded_candidate_network_ids':candidate_coverage['power_guard_excluded_network_ids'],
             'all_tip_points_owner_indices_numbers_and_network_members_preserved':True,
             'input_contract':'PNG and image-model predictions only; no source annotations, reference JSON or Golden reads',
             'new_training':False,'official_PinF1':None,'official_NetF1':None,'complete_competition_JSON':False,
             'wall_seconds':time.perf_counter()-start}
    save(output/'summary.json',summary);return summary


def main():
    p=argparse.ArgumentParser();p.add_argument('--image',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cpu');p.add_argument('--network-result',type=Path)
    p.add_argument('--pin-text',type=Path);p.add_argument('--texts',type=Path)
    p.add_argument('--config',type=Path,default=ROOT/'configs/pin_output_rules_v1.json');a=p.parse_args()
    provided=[a.network_result,a.pin_text,a.texts]
    if any(provided) and not all(provided):p.error('Cached mode needs network result, pin text and OCR together')
    if all(provided):folder,pins,texts=provided
    else:
        upstream=a.output/'upstream'
        subprocess.run([sys.executable,str(ROOT/'scripts/run_pin_semantics.py'),'--image',str(a.image),
                        '--output',str(upstream),'--device',a.device],check=True)
        folder=upstream/'network';pins=upstream/'pin_text/pin_text_result.json';texts=upstream/'pin_text/texts.json'
    summary=process(a.image,folder,pins,texts,a.output,read(a.config))
    summary['used_external_model_caches']=bool(all(provided));save(a.output/'summary.json',summary)
    print({k:summary[k] for k in ['conservative','candidate','name_completion','used_external_model_caches']},flush=True)


if __name__=='__main__':main()
