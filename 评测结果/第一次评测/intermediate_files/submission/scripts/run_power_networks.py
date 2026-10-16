"""PNG or hash-checked S14 model cache -> conservative power proposals."""
import argparse
from collections import Counter
from pathlib import Path
import subprocess
import sys
import time

from run_component_pin_library import ROOT, read, save, sha
from run_pin_output_rules import validate_cache, power_network_guard, apply_power_export_gate, compress_candidate_edges
from run_submission_fields import fingerprint
from run_network_labels import logical_partial_export
from pcb_parser.power_networks import propose_power_networks
from pcb_parser.submission_fields import complete_fields, network_field_queue, validate_fields
from pcb_parser.pin_output_rules import complete_pin_output_names
from pcb_parser.dataset_converter import validate_output


def export_fields(pins, graph, records, output, *, fields_completed=False):
    # Later stages may already have applied unique-label identity remapping.
    # Repeating completion against the original label-site IDs is invalid.
    if fields_completed:
        validate_fields(pins, graph, records)
        fields, updated = pins, graph
    else:
        fields, updated, records, _ = complete_fields(pins, graph, records, read(ROOT/'configs/submission_fields_v1.json'))
    strict, strict_graph, candidate, candidate_graph, _ = complete_pin_output_names(fields,updated,records,read(ROOT/'configs/pin_output_rules_v1.json'))
    guard = power_network_guard(strict,strict_graph,records,read(ROOT/'configs/pin_output_rules_v1.json'))
    results = {}
    for name,data,g in [('conservative',strict,strict_graph),('candidate',candidate,candidate_graph)]:
        payload, coverage = logical_partial_export(data,g,records)
        payload, coverage = apply_power_export_gate(payload,coverage,guard)
        provisional=[]
        for net in g['networks']:
            if net.get('power_logical_merge'):
                provisional.append(net['network_id'])
                payload['nets'].pop(net['network_id'],None)
                row=next((n for n in coverage['withheld_networks'] if n['network_id']==net['network_id']),None)
                if row is None:
                    row={'network_id':net['network_id'],'reasons':[]};coverage['withheld_networks'].append(row)
                if 'power_logical_membership_provisional' not in row['reasons']:row['reasons'].append('power_logical_membership_provisional')
        coverage.update(exported_nets=len(payload['nets']), exported_network_ids=sorted(payload['nets']),
                        provisional_power_networks_withheld=provisional,
                        network_reason_counts=dict(Counter(r for n in coverage['withheld_networks'] for r in n['reasons'])),
                        complete_competition_JSON=False, official_evaluator_verified=False)
        if name=='candidate':
            payload,_ = compress_candidate_edges(payload,g,data['size'][1],read(ROOT/'configs/pin_output_rules_v1.json')['candidate_edge_max_deviation_px'])
        errors,bounds=validate_output(payload,*data['size'])
        if errors or bounds:raise ValueError('Power export violates local JSON contract')
        save(output/('partial_submission.json' if name=='conservative' else 'submission_candidate.json'),payload)
        save(output/(name+'_export_coverage.json'),coverage)
        if name=='conservative':save(output/'network_field_queue.json',network_field_queue(data,g,records,coverage))
        results[name]={k:coverage[k] for k in ['exported_components','exported_pins','exported_nets']}
    return results


def process(image,network_folder,pin_path,texts_path,output,cfg):
    start=time.perf_counter()
    rgb, graph, pins, records, graph_path=validate_cache(image,network_folder,pin_path,texts_path)
    if pins.get('library_config_sha256')!=fingerprint(read(ROOT/'configs/component_pin_library_v1.json')):
        raise ValueError('Expected S14 pin-library trial input')
    ocr=read(texts_path)
    proposed,audit=propose_power_networks(pins,graph,records,ocr['texts'],cfg,read(ROOT/'configs/pin_output_rules_v1.json'))
    if graph['edges']!=proposed['edges'] or len(graph['pin_attachments'])!=len(proposed['pin_attachments']):
        raise ValueError('Power stage changed physical edge or pin count')
    stable=lambda p:{k:v for k,v in p.items() if k not in {'network_id','alternatives'}}
    if [stable(p) for p in graph['pin_attachments']]!=[stable(p) for p in proposed['pin_attachments']]:
        raise ValueError('Power stage changed learned pin fields, points or attachment projections')
    original_members=sorted(p for n in graph['networks'] for p in n['pin_candidate_ids'])
    new_members=sorted(p for n in proposed['networks'] for p in n['pin_candidate_ids'])
    if original_members!=new_members:raise ValueError('Power stage omitted or duplicated a network member')
    provenance={'image_sha256':sha(image),'size':pins['size'],'coordinate_origin':'top_left',
                'power_config_sha256':fingerprint(cfg),'input_graph_sha256':sha(graph_path),
                'input_pin_sha256':sha(pin_path),'input_OCR_sha256':sha(texts_path),'prediction_only':True}
    proposed.update(**provenance)
    save(output/'power_graph_proposals.json',proposed)
    save(output/'power_binding_audit.json',{**provenance,**audit})
    baseline_export=export_fields(pins,graph,records,output/'baseline_export')
    proposed_export=export_fields(pins,proposed,records,output/'proposal_export')
    summary={**provenance,'power':audit['summary'],'baseline_export':baseline_export,'proposal_export':proposed_export,
             'all_original_tips_fields_owners_and_physical_edges_preserved':True,'all_original_network_members_retained':True,
             'new_training':False,'default_pipeline_adopted':False,'official_PinF1':None,'official_NetF1':None,
             'power_components_canonical_identity_inferred':False,'power_pin_numbers_or_names_guessed':False,
             'input_contract':'PNG and YOLO/U-Net/RapidOCR/library predictions only; no dataset annotations or Golden reads',
             'wall_seconds':time.perf_counter()-start}
    save(output/'summary.json',summary)
    return summary


def main():
    p=argparse.ArgumentParser();p.add_argument('--image',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cpu');p.add_argument('--network-folder',type=Path);p.add_argument('--pin-text',type=Path)
    p.add_argument('--texts',type=Path);p.add_argument('--config',type=Path,default=ROOT/'configs/power_networks_v1.json');a=p.parse_args()
    provided=[a.network_folder,a.pin_text,a.texts]
    if any(provided) and not all(provided):p.error('Cached mode needs network folder, pin text and OCR together')
    if all(provided):network,pins,texts=provided
    else:
        upstream=a.output/'upstream'
        subprocess.run([sys.executable,str(ROOT/'scripts/run_component_pin_library.py'),'--image',str(a.image),
                        '--output',str(upstream),'--device',a.device],check=True)
        network=upstream/'network';pins=upstream/'pin_text/pin_text_result.json';texts=upstream/'pin_text/texts.json'
    summary=process(a.image,network,pins,texts,a.output,read(a.config))
    summary['used_external_model_caches']=bool(all(provided));save(a.output/'summary.json',summary)
    print({k:summary[k] for k in ['power','proposal_export','used_external_model_caches']},flush=True)


if __name__=='__main__':main()
