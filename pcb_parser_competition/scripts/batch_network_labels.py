"""Prediction-only batch; freeze wire graph variant to isolate label changes."""
import argparse
from pathlib import Path

from run_network_labels import ROOT,read,save,sha,process


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--case',action='append')
    parser.add_argument('--output',type=Path,default=ROOT/'runs/s9_network_labels_v1');args=parser.parse_args()
    rows=[];cfg=read(ROOT/'configs/network_labels_v1.json')
    for folder in sorted((ROOT/'runs/s6_pin_text_v1/predictions').iterdir()):
        case=folder.name
        if not folder.is_dir() or len(case)!=4 or not case.isdigit() or args.case and case not in args.case:continue
        summary=process(ROOT/f'data/standardized/{case}/{case}.png',ROOT/f'runs/s7_wire_graph_v1/predictions/{case}/graph_proposals.json',
                        folder/'pin_text_result.json',folder/'texts.json',args.output/'predictions'/case,cfg)
        rows.append({'case_id':case,'logical':summary})
        save(args.output/'batch_summary.json',{'cases':rows,'base_graph':'wire_graph_v1','config_sha256':sha(ROOT/'configs/network_labels_v1.json'),'official_NetF1':None})
        print({'case':case,'label_status':summary['summary']['label_site_status_counts'],'merges':summary['summary']['logical_label_merges']},flush=True)
    if args.case and {r['case_id'] for r in rows}!=set(args.case):raise ValueError('Unavailable case requested')


if __name__=='__main__':main()
