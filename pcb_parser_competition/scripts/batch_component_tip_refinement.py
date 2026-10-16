"""Frozen development batch; recorded model cache provenance, no references."""
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from run_component_tip_refinement import ROOT,read,save,sha,process


def main():
    torch.set_num_threads(4);stage=ROOT/'runs/s13_component_tips';split=read(ROOT/'runs/unet_inference_v2/split.json')
    expected=read(ROOT/'configs/unet_inference_v2.json')['checkpoint_sha256']
    if split['source_checkpoint_sha256']!=expected or sha(ROOT/'weights/unet_wire_pin_v1.pt')!=expected:
        raise ValueError('Unknown cache checkpoint')
    rows=[]
    for item in split['rows']:
        case=item['case_id'];image=ROOT/f'data/standardized/{case}/{case}.png';pin=ROOT/f'runs/s11_pin_text_v2/predictions/{case}/pin_text_result.json'
        native=ROOT/f'runs/unet_inference_v2/cache/{case}/scale_1.0.npz';zoom=ROOT/f'runs/unet_inference_v2/cache/{case}/scale_1.5.npz'
        if sha(image)!=item['image_sha256']:raise ValueError('PNG differs from cache source')
        metadata=stage/f'inputs/{case}/probability_inputs.json'
        save(metadata,{'image_sha256':sha(image),'size':[item['width'],item['height']],'coordinate_origin':'top_left',
            'weights_sha256':expected,'original_pin_text_sha256':sha(pin),'native_probability_sha256':sha(native),
            'zoom_probability_sha256':sha(zoom),'zoom_scale':1.5,'prediction_only':True,'source':'Frozen_PNG_U-Net_cache_26_old_development_pages'})
        summary=process(image,pin,native,zoom,metadata,stage/f'predictions/{case}',read(ROOT/'configs/component_tip_refinement_v1.json'))
        graph=read(stage/f'predictions/{case}/physical/graph_proposals.json');old=read(ROOT/f'runs/s11_pin_text_v2/downstream/{case}/physical/graph_proposals.json')
        # Newly added tips change pin-protection regions in the existing body
        # guard. The learned wire probability stays fixed; resulting graph
        # differences must be measured, rather than hidden as an invariant.
        with np.load(native,allow_pickle=False) as a:current_wire=a['probability'][0]
        with np.load(ROOT/f'runs/s7_wire_graph_v1/inputs/{case}/wire_probability.npz',allow_pickle=False) as a:old_wire=a['wire']
        if not np.array_equal(current_wire,old_wire):raise ValueError('Frozen learned wire probability differs')
        original=read(pin)['pin_candidates']
        if [(p['candidate_id'],p['point']) for p in original]!=[(p['candidate_id'],p['point']) for p in graph['pin_attachments'][:len(original)]]:
            raise ValueError('Refinement altered an original point')
        with Image.open(stage/f'predictions/{case}/physical/guarded_wire_mask.png') as im:new_mask=np.asarray(im).copy()
        with Image.open(ROOT/f'runs/s11_pin_text_v2/downstream/{case}/physical/guarded_wire_mask.png') as im:old_mask=np.asarray(im).copy()
        summary['body_guard_mask_changed_pixels']=int(np.count_nonzero(new_mask!=old_mask))
        summary['graph_edges_changed']=graph['edges']!=old['edges']
        summary['learned_wire_probability_preserved']=True
        save(stage/f'predictions/{case}/summary.json',summary)
        rows.append({'case_id':case,**summary});save(stage/'predictions/batch_summary.json',{'completed':False,'cases':rows})
        print({'case':case,'added':summary['summary']['added_by_type'],'power_after':summary['power_guard_after']},flush=True)
    save(stage/'predictions/batch_summary.json',{'completed':True,'cases':rows,'old_development_data':True,'default_pipeline_adopted':False})


if __name__=='__main__':main()
