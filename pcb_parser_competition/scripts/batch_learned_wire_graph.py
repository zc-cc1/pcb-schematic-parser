"""Frozen image predictions plus learned junctions; no review/Golden reads."""
import argparse
from pathlib import Path

from run_wire_graph import ROOT,read,save,sha,process
from pcb_parser.junction_classifier import JunctionPredictor


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--device',default='cpu')
    parser.add_argument('--output',type=Path,default=ROOT/'runs/s8_junction_classifier_v1')
    parser.add_argument('--case',action='append');args=parser.parse_args()
    cfg=read(ROOT/'configs/wire_graph_v2.json')
    predictor=JunctionPredictor(ROOT/cfg['junction_checkpoint'],args.device,cfg['junction_checkpoint_sha256'])
    cases=[]
    for folder in sorted((ROOT/'runs/s6_pin_text_v1/predictions').iterdir()):
        case=folder.name
        if not folder.is_dir() or len(case)!=4 or not case.isdigit() or (args.case and case not in args.case):continue
        summary=process(ROOT/f'data/standardized/{case}/{case}.png',folder/'pin_text_result.json',
                        ROOT/f'runs/s7_wire_graph_v1/inputs/{case}/wire_probability.npz',
                        args.output/'predictions'/case,cfg,device=args.device,junction_predictor=predictor)
        cases.append({'case_id':case,'learned':summary})
        print({'case':case,'resolved':summary['summary']['learned_resolved_junctions'],
               'uncertain':summary['summary']['uncertain_junctions']},flush=True)
        save(args.output/'batch_summary.json',{'cases':cases,'config_sha256':sha(ROOT/'configs/wire_graph_v2.json'),
             'junction_checkpoint_sha256':predictor.sha256,'official_NetF1':None,'inputs_frozen':True})
    if args.case and set(args.case)!={r['case_id'] for r in cases}:raise ValueError('Requested case is not available')


if __name__=='__main__':main()
