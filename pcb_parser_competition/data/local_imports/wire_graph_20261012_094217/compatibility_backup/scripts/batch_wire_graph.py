"""Build graphs on frozen model inputs; this prediction runner reads no labels."""
import argparse
import json
from pathlib import Path

import numpy as np

from run_wire_graph import ROOT,read,save,sha,process


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--case',action='append')
    parser.add_argument('--output',type=Path,default=ROOT/'runs/s7_wire_graph_v1')
    args=parser.parse_args()
    cfg=read(ROOT/'configs/wire_graph_v1.json')
    inference=read(ROOT/'configs/unet_inference_v2.json')
    assert inference['wire_mode']=='native'
    source=read(ROOT/'runs/unet_inference_v2/split.json')
    assert source['source_checkpoint_sha256']==inference['checkpoint_sha256']
    rows={r['case_id']:r for r in source['rows']}
    folders=sorted(p for p in (ROOT/'runs/s6_pin_text_v1/predictions').iterdir() if p.is_dir() and len(p.name)==4 and p.name.isdigit())
    if args.case:
        folders=[p for p in folders if p.name in args.case]
        if {p.name for p in folders}!=set(args.case):raise ValueError('Requested case is not a frozen model output')
    result=[]
    for folder in folders:
        case=folder.name;image=ROOT/f'data/standardized/{case}/{case}.png'
        assert sha(image)==rows[case]['image_sha256']
        cache=args.output/'inputs'/case/'wire_probability.npz'
        cache.parent.mkdir(parents=True,exist_ok=True)
        with np.load(ROOT/f'runs/unet_inference_v2/cache/{case}/scale_1.0.npz',allow_pickle=False) as original:wire=original['probability'][0]
        np.savez_compressed(cache,wire=wire)
        save(cache.with_suffix('.json'),{'image_sha256':sha(image),'size':[rows[case]['width'],rows[case]['height']],
             'coordinate_origin':'top_left','weights_sha256':inference['checkpoint_sha256'],'inference_config_sha256':sha(ROOT/'configs/unet_inference_v2.json'),
             'probabilities_sha256':sha(cache),'prediction_only':True,'source':'Frozen_native_U-Net_model_probabilities_from_existing_PNG_run'})
        adopted=process(image,folder/'pin_text_result.json',cache,args.output/'predictions'/case,cfg)
        observed=process(image,folder/'pin_text_result.json',cache,args.output/'observed_connected'/case,{**cfg,'junction_policy':'observed_connected'})
        row={'case_id':case,'default':adopted,'observed_connected':observed}
        result.append(row)
        print(json.dumps({'case':case,'default':adopted['summary'],'observed_components':observed['summary']['network_proposals']},ensure_ascii=False),flush=True)
        save(args.output/'batch_summary.json',{'cases':result,'config_sha256':sha(ROOT/'configs/wire_graph_v1.json'),
             'model_checkpoint_sha256':inference['checkpoint_sha256'],'inputs_frozen':True,'official_NetF1':None})


if __name__=='__main__':main()
