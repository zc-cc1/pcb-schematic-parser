"""PNG -> S17 -> selective missing-field OCR -> audited partial exports."""
import argparse
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from PIL import Image

from run_network_completion import ROOT, read, save, sha, export_fields
from run_submission_fields import fingerprint
from pcb_parser.targeted_pin_ocr import select_rois, collect_readings, adopt_missing_fields, existing_readings
from pcb_parser.submission_fields import network_field_queue, validate_fields
from pcb_parser.pin_text_ocr import create_engine
from pcb_parser.pin_output_rules import complete_pin_output_names


def process(image, stage_folder, output, cfg, engine=None, models=None):
    begin=time.perf_counter()
    with Image.open(image) as source:rgb=np.asarray(source.convert('RGB'))
    image_hash=sha(image);size=[rgb.shape[1],rgb.shape[0]]
    pin_path=stage_folder/'completed_pin_text_result.json';graph_path=stage_folder/'graph_proposals.json'
    records_path=stage_folder/'completed_component_records.json'
    pins,graph,records=read(pin_path),read(graph_path),read(records_path)
    for payload in [pins,graph]:
        if (payload.get('image_sha256')!=image_hash or payload.get('size')!=size
                or payload.get('coordinate_origin')!='top_left' or not payload.get('prediction_only')
                or payload.get('pin_role_config_sha256')!=fingerprint(read(ROOT/'configs/pin_number_validation_v1.json'))):
            raise ValueError('S17 input differs from PNG/configuration')
    if records.get('image_sha256')!=image_hash:raise ValueError('Records differ from PNG')
    records=records['records'];validate_fields(pins,graph,records)
    baseline=export_fields(pins,graph,records,output/'baseline_export',fields_completed=True)
    coverage=read(output/'baseline_export/conservative_export_coverage.json')
    queue=network_field_queue(pins,graph,records,coverage)
    text_cfg=read(ROOT/'configs/pin_text_v2.json');passive_cfg=read(ROOT/'configs/pin_output_rules_v1.json')
    rois,skipped=select_rois(pins,records,queue,cfg,text_cfg)
    if engine is None:engine,models=create_engine(cfg)
    texts,recovery=collect_readings(rgb,rois,engine,cfg)
    original_readings=existing_readings(pins)
    updated,new_graph,audit=adopt_missing_fields(pins,graph,records,original_readings+texts,{r['candidate_id'] for r in rois},cfg,text_cfg,passive_cfg)
    recovery['original_unsplit_candidate_readings']=len(original_readings)
    exports=export_fields(updated,new_graph,records,output/'proposal_export',fields_completed=True)
    strict,sg,candidate,cg,_=complete_pin_output_names(updated,new_graph,records,passive_cfg)
    queues={}
    for profile,data,g in [('conservative',strict,sg),('candidate',candidate,cg)]:
        cov=read(output/f'proposal_export/{profile}_export_coverage.json')
        queues[profile]=network_field_queue(data,g,records,cov)
        save(output/f'{profile}_completion_queue.json',queues[profile])
    provenance={'image_sha256':image_hash,'size':size,'coordinate_origin':'top_left',
        'input_pin_sha256':sha(pin_path),'input_graph_sha256':sha(graph_path),'input_records_sha256':sha(records_path),
        'targeted_OCR_config_sha256':fingerprint(cfg),'OCR_models':models,'prediction_only':True}
    save(output/'localized_texts.json',{**provenance,'texts':texts})
    save(output/'recovery_audit.json',{**provenance,**audit,**recovery,'skipped_by_budget':skipped})
    save(output/'completed_pin_text_result.json',{**strict,**provenance})
    save(output/'candidate_pin_text_result.json',{**candidate,**provenance})
    save(output/'completed_component_records.json',{'image_sha256':image_hash,'records':records})
    save(output/'graph_proposals.json',{**sg,**provenance})
    summary={**provenance,'recovery':audit['summary'],'roi_count':len(rois),'skipped_by_budget':skipped,
        'baseline_export':baseline,'proposal_export':exports,
        'completion':{profile:{k:queues[profile][k] for k in ['multi_pin_networks','field_only_incomplete_networks','non_field_export_blocker_counts']} for profile in queues},
        'all_original_tips_ownership_existing_fields_and_network_members_preserved':True,
        'default_pipeline_adopted':False,'official_PinF1':None,'official_NetF1':None,
        'input_contract':'PNG and S17 model predictions only; no source annotations/CAD/Golden reads',
        'wall_seconds':time.perf_counter()-begin}
    save(output/'summary.json',summary)
    return summary


def main():
    begin=time.perf_counter()
    p=argparse.ArgumentParser();p.add_argument('--image',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--stage-result',type=Path);p.add_argument('--device',default='cpu')
    p.add_argument('--config',type=Path,default=ROOT/'configs/targeted_pin_ocr_v1.json');a=p.parse_args()
    stage=a.stage_result
    if stage is None:
        stage=a.output/'stage17'
        subprocess.run([sys.executable,str(ROOT/'scripts/run_network_completion.py'),'--image',str(a.image),
                        '--output',str(stage),'--device',a.device],check=True)
    result=process(a.image,stage,a.output,read(a.config))
    result.update(used_external_model_caches=a.stage_result is not None,entrypoint_wall_seconds=time.perf_counter()-begin)
    save(a.output/'summary.json',result)
    print({k:result[k] for k in ['recovery','roi_count','baseline_export','proposal_export']},flush=True)


if __name__=='__main__':main()
