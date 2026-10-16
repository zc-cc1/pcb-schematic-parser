"""Build graphs on frozen model inputs; this prediction runner reads no labels."""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from run_wire_graph import ROOT,read,save,sha,process,validate_cache,predict_wire


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--case',action='append')
    parser.add_argument('--output',type=Path,default=ROOT/'runs/s7_wire_graph_v1')
    parser.add_argument('--device',default='cpu',help='Only used if no frozen wire prediction is available')
    args=parser.parse_args()
    cfg=read(ROOT/'configs/wire_graph_v1.json')
    inference=read(ROOT/'configs/unet_inference_v2.json')
    assert inference['wire_mode']=='native'
    folders=sorted(p for p in (ROOT/'runs/s6_pin_text_v1/predictions').iterdir() if p.is_dir() and len(p.name)==4 and p.name.isdigit())
    if args.case:
        folders=[p for p in folders if p.name in args.case]
        if {p.name for p in folders}!=set(args.case):raise ValueError('Requested case is not a frozen model output')
    result=[]
    for folder in folders:
        case=folder.name;image=ROOT/f'data/standardized/{case}/{case}.png'
        pin_result=read(folder/'pin_text_result.json')
        assert sha(image)==pin_result['image_sha256']
        cache=args.output/'inputs'/case/'wire_probability.npz'
        cache.parent.mkdir(parents=True,exist_ok=True)
        prepared=ROOT/f'runs/s7_wire_graph_v1/inputs/{case}/wire_probability.npz'
        existing=cache if cache.is_file() else prepared if prepared.is_file() else None
        if existing:
            metadata=read(existing.with_suffix('.json'))
            validate_cache(metadata,existing,sha(image),pin_result['size'],inference['checkpoint_sha256'],sha(ROOT/'configs/unet_inference_v2.json'))
            if existing.resolve()!=cache.resolve():
                shutil.copy2(existing,cache);shutil.copy2(existing.with_suffix('.json'),cache.with_suffix('.json'))
        else:
            original_path=ROOT/f'runs/unet_inference_v2/cache/{case}/scale_1.0.npz'
            if original_path.is_file():
                split=read(ROOT/'runs/unet_inference_v2/split.json')
                assert split['source_checkpoint_sha256']==inference['checkpoint_sha256']
                row=next(r for r in split['rows'] if r['case_id']==case)
                assert row['image_sha256']==sha(image)
                with np.load(original_path,allow_pickle=False) as original:wire=original['probability'][0]
                provenance='Frozen_native_U-Net_model_probabilities_from_existing_PNG_run'
            else:
                from PIL import Image
                with Image.open(image) as source:rgb=np.asarray(source.convert('RGB'))
                wire=predict_wire(rgb,args.device,inference)
                provenance='PNG_native_U-Net_inference_cache_missing'
            np.savez_compressed(cache,wire=wire)
            save(cache.with_suffix('.json'),{'image_sha256':sha(image),'size':pin_result['size'],
                 'coordinate_origin':'top_left','weights_sha256':inference['checkpoint_sha256'],'inference_config_sha256':sha(ROOT/'configs/unet_inference_v2.json'),
                 'probabilities_sha256':sha(cache),'prediction_only':True,'source':provenance})
        adopted=process(image,folder/'pin_text_result.json',cache,args.output/'predictions'/case,cfg)
        observed=process(image,folder/'pin_text_result.json',cache,args.output/'observed_connected'/case,{**cfg,'junction_policy':'observed_connected'})
        row={'case_id':case,'default':adopted,'observed_connected':observed}
        result.append(row)
        print(json.dumps({'case':case,'default':adopted['summary'],'observed_components':observed['summary']['network_proposals']},ensure_ascii=False),flush=True)
        save(args.output/'batch_summary.json',{'cases':result,'config_sha256':sha(ROOT/'configs/wire_graph_v1.json'),
             'model_checkpoint_sha256':inference['checkpoint_sha256'],'inputs_frozen':True,'official_NetF1':None})


if __name__=='__main__':main()
