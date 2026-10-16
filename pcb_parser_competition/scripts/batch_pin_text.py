"""Run PNG OCR against the frozen 26-page ownership predictions; no references."""
import argparse
import json
from pathlib import Path

from run_pin_text import ROOT,process,read,save,sha
from pcb_parser.pin_text_ocr import create_engine


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=ROOT/'runs/s6_pin_text_v1/predictions')
    parser.add_argument('--ownership-root',type=Path,default=ROOT/'runs/s5_pin_ownership_v1')
    parser.add_argument('--case',action='append',help='Optional subset of the frozen ownership case IDs')
    parser.add_argument('--enable-local',action='store_true',help='Explicit experimental local rereading')
    args=parser.parse_args()
    cfg=read(ROOT/'configs/pin_text_v1.json')
    if args.enable_local:
        cfg['local_ocr']['enabled']=True
    component_cfg=read(ROOT/'configs/pin_text_component_v2.json')
    engine,fingerprint=create_engine(cfg)
    folders=sorted(p for p in args.ownership_root.iterdir() if p.is_dir() and len(p.name)==4 and p.name.isdigit())
    if args.case:
        folders=[p for p in folders if p.name in args.case]
        if {p.name for p in folders}!=set(args.case):
            raise ValueError('Requested case has no ownership prediction')
    summaries=[]
    for folder in folders:
        summary=process(ROOT/f'data/standardized/{folder.name}/{folder.name}.png',folder/'pin_ownership.json',
                        args.output/folder.name,cfg,component_cfg,engine,fingerprint)
        summary['case_id']=folder.name
        summaries.append(summary)
        print(json.dumps({'case':folder.name,**summary['summary'],'OCR_seconds':summary['timing']['wall_seconds_to_binding']},ensure_ascii=False),flush=True)
        save(args.output/'batch_summary.json',{'cases':summaries,'all_candidates_preserved':True,'official_score':None,
             'pin_text_config_sha256':sha(ROOT/'configs/pin_text_v1.json'),'component_config_sha256':sha(ROOT/'configs/pin_text_component_v2.json')})


if __name__=='__main__':
    main()
