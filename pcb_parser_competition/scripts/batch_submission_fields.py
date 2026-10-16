"""Replay field stage on the frozen 26 development predictions."""
import argparse
from pathlib import Path
from run_submission_fields import ROOT,read,save,process
from pcb_parser.pin_text_ocr import create_engine


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'runs/s10_submission_fields_v1')
    p.add_argument('--case',action='append');a=p.parse_args();cfg=read(ROOT/'configs/submission_fields_v1.json')
    engine,models=create_engine({'onnx_threads':cfg['local_ocr']['onnx_threads']})
    rows=[]
    for row in read(ROOT/'runs/s9_network_labels_v1/batch_summary.json')['cases']:
        case=row['case_id']
        if a.case and case not in a.case:continue
        pin_folder=ROOT/f'runs/s6_pin_text_v1/predictions/{case}'
        result=process(ROOT/f'data/standardized/{case}/{case}.png',ROOT/f'runs/s9_network_labels_v1/predictions/{case}',
                       pin_folder/'pin_text_result.json',pin_folder/'texts.json',a.output/'predictions'/case,cfg,engine,models)
        rows.append({'case_id':case,'fields':result})
        save(a.output/'batch_summary.json',{'cases':rows,'config':cfg,'official_NetF1':None})
        print({'case':case,'identities':result['summary']['new_label_identities'],'auto_numbers':result['summary']['single_tip_label_auto_numbers'],
               'strict_pins':result['after_export']['exported_pins'],'strict_nets':result['after_export']['exported_nets']},flush=True)
    if a.case and set(a.case)!={r['case_id'] for r in rows}:raise ValueError('Unavailable case requested')


if __name__=='__main__':main()
