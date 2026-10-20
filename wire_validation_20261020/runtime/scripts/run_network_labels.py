"""PNG/model outputs -> ordinary logical label proposals, without references."""
import argparse
from collections import Counter
import hashlib
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from PIL import Image,ImageDraw,ImageFont

from run_wire_graph import ROOT,read,save,sha
from pcb_parser.network_labels import build_logical_networks
from pcb_parser.wire_graph_artifacts import partial_submission
from pcb_parser.dataset_converter import validate_output


def validate_inputs(image_hash,size,graph,pins,ocr,records):
    for name,payload in [('graph',graph),('pins',pins),('OCR',ocr)]:
        if payload.get('image_sha256')!=image_hash or payload.get('size')!=size or payload.get('coordinate_origin')!='top_left':
            raise ValueError(name+' does not match PNG hash, size or coordinates')
    if not ocr.get('prediction_only'):raise ValueError('Expected image-model OCR predictions')
    if graph.get('pin_input_sha256') is None:raise ValueError('Graph input provenance missing')
    if records.get('image_sha256')!=image_hash:raise ValueError('Component records differ from image')
    expected={p['candidate_id']:p['point'] for p in pins['pin_candidates']}
    actual={p['candidate_id']:p['point'] for p in graph['pin_attachments']}
    if expected!=actual or len(actual)!=len(graph['pin_attachments']):raise ValueError('Graph pin coordinates or IDs differ')
    components={c['detection_index']:c for c in pins['components']}
    for r in records['records']:
        c=components.get(r['detection_index'])
        if not c or r['component_id']!=c['component_id'] or not np.allclose(r['bbox_top_left'],c['bbox_top_left'],atol=.05,rtol=0):
            raise ValueError('Component records differ from pin owners')


def render_labels(rgb,result,path):
    image=Image.fromarray(rgb);draw=ImageDraw.Draw(image);font=ImageFont.truetype(str(Path(__file__).resolve().parents[1] / 'assets/fonts/DejaVuSansMono.ttf'),12)
    applied={sid for d in result['label_merge_decisions'] if not d['withheld_reasons'] for sid in d['site_ids']}
    for site in result['label_sites']:
        if site['network_id'] is None:continue
        colour='#9133ad' if site['site_id'] in applied else '#737373' if site['status']=='power_semantics_pending' else '#d16c00'
        draw.rectangle(site['bbox'],outline=colour,width=2)
        if site['point']:
            x,y=site['point'];draw.ellipse((x-4,y-4,x+4,y+4),outline=colour,width=2)
            draw.line([tuple(site['point']),tuple(((site['bbox'][0]+site['bbox'][2])/2,(site['bbox'][1]+site['bbox'][3])/2))],fill=colour,width=1)
    image.thumbnail((1700,1700));canvas=Image.new('RGB',(max(900,image.width),image.height+44),'white');canvas.paste(image)
    legend=ImageDraw.Draw(canvas)
    legend.text((8,image.height+3),'Purple: ordinary same-name logical proposal | Orange: bound but withheld | Grey: power pending',font=font,fill='#444')
    legend.text((8,image.height+23),'Short text-to-anchor lines show association only; no remote physical wire is generated.',font=font,fill='#444')
    canvas.save(path)


def logical_partial_export(pins,logical,records):
    out,coverage=partial_submission(pins,logical,records)
    # Keep all logical members. Never remove unresolved label pins to make a
    # smaller, deceptively complete hypergraph pass the strict export gate.
    withheld=[]
    for net in logical['networks']:
        if not net.get('flywire') or net['network_id'] not in out['nets']:continue
        if any(not s.get('candidate_id') or s['candidate_id'] not in net['pin_candidate_ids'] for s in net['logical_label_sites']):
            del out['nets'][net['network_id']]
            withheld.append({'network_id':net['network_id'],'reasons':['logical_label_component_or_pin_unresolved']});continue
        points=sorted({tuple(s['point']) for s in net['logical_label_sites']})
        out['nets'][net['network_id']]['edges']={f'edge_{i+1}':[{'x':x,'y':pins['size'][1]-y},{'x':x,'y':pins['size'][1]-y}] for i,(x,y) in enumerate(points)}
    coverage['withheld_networks'].extend(withheld)
    coverage['network_reason_counts']=dict(Counter(reason for item in coverage['withheld_networks'] for reason in item['reasons']))
    coverage.update(exported_nets=len(out['nets']),logical_flywire_merges=logical['summary']['logical_label_merges'],
                    flywire_zero_length_geometry='provisional_from_PDF_example_no_official_Golden')
    errors,bounds=validate_output(out,*pins['size'])
    if errors or bounds:raise ValueError('Logical partial export violates JSON contract')
    return out,coverage


def process(image_path,graph_path,pin_path,texts_path,output,cfg):
    start=time.perf_counter()
    with Image.open(image_path) as image:rgb=np.asarray(image.convert('RGB'))
    image_hash=sha(image_path);size=[rgb.shape[1],rgb.shape[0]]
    graph,pins,ocr=read(graph_path),read(pin_path),read(texts_path)
    records_path=pin_path.parent/'component_records.json';records=read(records_path)
    validate_inputs(image_hash,size,graph,pins,ocr,records)
    inference=read(ROOT/'configs/unet_inference_v2.json')
    if graph.get('unet_weights_sha256')!=inference['checkpoint_sha256']:raise ValueError('Graph uses different U-Net weights')
    valid_graph_hashes={hashlib.sha256(__import__('json').dumps(read(ROOT/'configs'/name),sort_keys=True).encode()).hexdigest()
                        for name in ['wire_graph_v1.json','wire_graph_v2.json'] if (ROOT/'configs'/name).is_file()}
    if graph.get('wire_graph_config_sha256') not in valid_graph_hashes:raise ValueError('Unknown wire graph configuration')
    if graph['pin_input_sha256']!=sha(pin_path):raise ValueError('Graph pin input bytes changed')
    logical=build_logical_networks(pins,graph,ocr['texts'],records['records'],cfg)
    logical.update(physical_graph_input_sha256=sha(graph_path),OCR_input_sha256=sha(texts_path),
                   label_config_sha256=hashlib.sha256(__import__('json').dumps(cfg,sort_keys=True).encode()).hexdigest())
    output.mkdir(parents=True,exist_ok=True)
    save(output/'logical_graph_proposals.json',logical)
    save(output/'label_bindings.json',{'image_sha256':image_hash,'size':size,'coordinate_origin':'top_left','sites':logical['label_sites'],
         'merge_decisions':logical['label_merge_decisions'],'matching_rule':'exact_observed_spelling','prediction_only':True})
    submission,coverage=logical_partial_export(pins,logical,records['records'])
    save(output/'partial_submission.json',submission);save(output/'export_coverage.json',coverage)
    render_labels(rgb,logical,output/'network_labels_overlay.png')
    summary={'summary':logical['summary'],'image_sha256':image_hash,'input_graph_sha256':sha(graph_path),
             'OCR_input_sha256':sha(texts_path),'wall_seconds':time.perf_counter()-start,'label_config':cfg,
             'exported_components':coverage['exported_components'],'exported_pins':coverage['exported_pins'],'exported_nets':coverage['exported_nets'],
             'input_contract':'PNG/model predictions only; no CAD/reference labels/Golden reads','official_NetF1':None}
    save(output/'summary.json',summary);return summary


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--image',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--physical-graph',type=Path);parser.add_argument('--pin-text',type=Path);parser.add_argument('--texts',type=Path)
    parser.add_argument('--device',default='cpu');parser.add_argument('--learned-junction-pilot',action='store_true')
    parser.add_argument('--config',type=Path,default=ROOT/'configs/network_labels_v1.json');args=parser.parse_args()
    provided=[args.physical_graph,args.pin_text,args.texts]
    if any(provided) and not all(provided):parser.error('Cached mode needs graph, pins and OCR together')
    if all(provided):graph,pins,texts=provided
    else:
        upstream=args.output/'physical';graph_cfg=ROOT/'configs'/('wire_graph_v2.json' if args.learned_junction_pilot else 'wire_graph_v1.json')
        subprocess.run([sys.executable,str(ROOT/'scripts/run_wire_graph.py'),'--image',str(args.image),'--output',str(upstream),
                        '--device',args.device,'--config',str(graph_cfg)],check=True)
        graph=upstream/'graph_proposals.json';pins=upstream/'upstream/pin_text_result.json';texts=upstream/'upstream/texts.json'
    summary=process(args.image,graph,pins,texts,args.output,read(args.config))
    summary['used_external_model_caches']=all(provided);summary['learned_junction_pilot']=bool(read(graph).get('junction_classifier_used'))
    save(args.output/'summary.json',summary)
    print(summary['summary'],flush=True)


if __name__=='__main__':main()
