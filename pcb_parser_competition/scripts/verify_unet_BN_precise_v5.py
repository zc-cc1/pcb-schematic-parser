"""Precision-only extension; reuse the frozen calibration and original tolerance."""
import gc
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from unet_audit_common import ROOT,OUT,read,write,sha,counts,metrics,aggregate,assert_protected
from diagnose_unet_BN_order_v5 import training_schedule
from reestimate_unet_BN_v4 import fixed_weights_hash
from pcb_parser.unet_bn_statistics import fixed_batches
from pcb_parser.unet_bn_precise_statistics import reestimate_bn_buffers_precise
from pcb_parser.unet_model import WirePinUNet
from pcb_parser.unet_data import predict_tiled,annotation_targets
from pcb_parser.dataset_converter import load_source_annotation
from pcb_parser.unet_inference import decode_pins
from verify_unet_contrast import path_specs,path_records,pin_metrics


def main():
    output=ROOT/'runs/unet_BN_order_diagnostic_v5';protocol=read(output/'precise_extension_protocol.json')
    old=ROOT/'runs/unet_negative_sampling_pilot_v4';base=read(output/'evaluation.json')
    assert not (output/'precise_evaluation.json').exists()
    assert sha(ROOT/'src/pcb_parser/unet_bn_precise_statistics.py')==protocol['module_sha256']
    assert sha(ROOT/'src/pcb_parser/unet_bn_statistics.py')==protocol['original_module_sha256']
    frozen=read(old/'protected_inputs.json')|read(old/'additional_source_attestation.json')
    assert all(sha(p)==v for p,v in frozen.items());assert_protected()
    cfg=read(old/'configuration.json');rows=[json.loads(s) for s in (ROOT/'runs/unet_training_label_review_v4/training_manifest.jsonl').read_text(encoding='utf-8').splitlines()]
    train=[r for r in rows if r['split']=='train'];cal=[r for r in rows if r['split']=='calibration'];assert len(train)==93 and len(cal)==10
    regions=read(ROOT/'runs/unet_training_label_review_v4/confirmed_negative_regions.json')['regions']
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    script_sha=sha(__file__);start=time.perf_counter();checkpoints={};estimates=[]
    for arm in protocol['arms']:
        checkpoint=torch.load(old/arm/'last.pt',map_location='cpu',weights_only=True);state=checkpoint['state_dict']
        assert sha(old/arm/'last.pt')==base['protocol']['source_weights'][arm]
        dataset,_=training_schedule(train,regions,cfg,arm)
        for order in protocol['orders']:
            name=f'{arm}__precise_cumulative__{order}';folder=output/name;folder.mkdir(exist_ok=True)
            model=WirePinUNet(checkpoint['base_channels']).to('cuda:0');model.load_state_dict(state,strict=True);model.eval()
            loader=DataLoader(dataset,batch_sampler=fixed_batches(len(dataset),2,order=='reverse_batches'),num_workers=0)
            def batches():
                for i,batch in enumerate(loader):
                    if (i+1)%100==0:print(json.dumps(dict(phase='RGB_only_precise_BN',variant=name,batches=i+1,total=len(loader))),flush=True)
                    yield batch.to('cuda:0')
            before=time.perf_counter()
            with torch.autocast('cuda',enabled=True):count=reestimate_bn_buffers_precise(model,batches())
            after={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            assert fixed_weights_hash(after)==fixed_weights_hash(state)
            changed=[k for k in state if not torch.equal(state[k],after[k])]
            assert all(k.endswith(('running_mean','running_var','num_batches_tracked')) for k in changed)
            path=folder/'diagnostic.pt';torch.save(dict(checkpoint,state_dict=after,kind='precise_BN_order_diagnostic',gradient_updates=0),path)
            checkpoints[name]=dict(path=str(path),sha256=sha(path),arm=arm,order=order)
            estimates.append(dict(variant=name,**count,learned_weights_unchanged=True,changed_keys=changed,
                elapsed_seconds=time.perf_counter()-before,labels_or_calibration_used_for_estimation=False))
            print(json.dumps(dict(event='precise_estimated',variant=name,seconds=estimates[-1]['elapsed_seconds'])),flush=True)
            del model,after;gc.collect();torch.cuda.empty_cache()
        dataset.rgb.cache_clear()
    models={};records=[]
    for name,r in checkpoints.items():
        ckpt=torch.load(r['path'],map_location='cpu',weights_only=True);model=WirePinUNet(ckpt['base_channels']).to('cuda:0')
        model.load_state_dict(ckpt['state_dict'],strict=True);model.eval();models[name]=model
    raw={r['case_id']:r for r in map(json.loads,(ROOT/'data/manifest.jsonl').read_text(encoding='utf-8').splitlines())}
    for row in cal:
        case=row['case_id'];rgb=np.asarray(Image.open(row['image']).convert('RGB'));predictions={}
        for name,model in models.items():
            folder=output/'evaluation'/case/name;folder.mkdir(parents=True,exist_ok=True)
            with torch.inference_mode():p=predict_tiled(model,rgb,'cuda:0',512,384)
            np.savez_compressed(folder/'probabilities.npz',wire=p[0],pin=p[1]);predictions[name]=p
            write(folder/'prediction_meta.json',dict(checkpoint_sha256=checkpoints[name]['sha256'],image_sha256=sha(row['image']),image_only=True,tile_size=512,stride=384,resize=False))
        gt=np.asarray(Image.open(row['wire_mask']))>0;annotation,_=load_source_annotation(raw[case]['raw_annotation_path'])
        reference,*_=annotation_targets(annotation,row['width'],row['height'],3,2.);assert np.array_equal(np.asarray(reference)>0,gt)
        specs=path_specs(annotation,gt.shape);tips=read(Path(row['wire_mask']).parent/'pin_points.json')
        for name,p in predictions.items():
            mask=p[0]>=.5;paths=path_records(mask,specs);pins=decode_pins(p[1],threshold=.8,kernel=5,refinement='quadratic',nms_radius=0.,limit=500)
            r=dict(case_id=case,variant=name,wire=metrics(*counts(mask,gt)),pin=pin_metrics(pins,tips),connected_paths=sum(r['connected_within_reference_tube'] for r in paths),total_paths=len(paths))
            records.append(r);write(output/'evaluation'/case/name/'metrics.json',r)
            Image.fromarray(mask.astype(np.uint8)*255).save(output/'evaluation'/case/name/'wire_mask.png')
        print(json.dumps(dict(event='precise_cal_page_scored',case_id=case,pages=len({r['case_id'] for r in records}))),flush=True)
    summary={}
    for name in checkpoints:
        selected=[r for r in records if r['variant']==name]
        summary[name]=dict(wire=aggregate(selected),pin=aggregate(selected,'pin'),connected_paths=sum(r['connected_paths'] for r in selected),total_paths=sum(r['total_paths'] for r in selected))
    pairs=[];ids=sorted(r['case_id'] for r in cal)
    for arm in protocol['arms']:
        a=f'{arm}__precise_cumulative__forward';b=f'{arm}__precise_cumulative__reverse_batches'
        x=summary[a];y=summary[b];maxdiff=0.;changed=total=0;sumdiff=0.;fp32diff=0.;fp32changed=0
        sa=torch.load(checkpoints[a]['path'],map_location='cpu',weights_only=True)['state_dict'];sb=torch.load(checkpoints[b]['path'],map_location='cpu',weights_only=True)['state_dict']
        equal=all(torch.equal(sa[k],sb[k]) for k in sa)
        for case in ids:
            with np.load(output/'evaluation'/case/a/'probabilities.npz',allow_pickle=False) as source:p=source['wire']
            with np.load(output/'evaluation'/case/b/'probabilities.npz',allow_pickle=False) as source:q=source['wire']
            with np.load(output/'evaluation'/case/f'{arm}__cumulative__forward'/'probabilities.npz',allow_pickle=False) as source:r=source['wire']
            dv=np.abs(p-q);maxdiff=max(maxdiff,float(dv.max()));sumdiff+=float(dv.sum(dtype=np.float64));total+=p.size
            changed+=int(((p>=.5)!=(q>=.5)).sum());fp32changed+=int(((p>=.5)!=(r>=.5)).sum());fp32diff=max(fp32diff,float(np.abs(p-r).max()))
        pairs.append(dict(arm=arm,method='precise_cumulative',contrast='reverse_batches minus forward',delta_dice_pp=100*(y['wire']['dice']-x['wire']['dice']),
            delta_recall_pp=100*(y['wire']['recall']-x['wire']['recall']),delta_connected_paths=y['connected_paths']-x['connected_paths'],
            all_state_tensors_bitwise_equal=equal,changed_mask_pixels=changed,cal_pixels=total,changed_mask_fraction=changed/total,
            max_abs_probability_difference=maxdiff,mean_abs_probability_difference=sumdiff/total,
            stable_under_original_preregistered_tolerance=maxdiff<=1e-4 and changed/total<=1e-5 and abs(y['connected_paths']-x['connected_paths'])<=1,
            precision_change_only_vs_fp32_forward=dict(changed_mask_pixels=fp32changed,max_abs_probability_difference=fp32diff,
                delta_dice_pp=100*(x['wire']['dice']-base['summary'][f'{arm}__cumulative__forward']['wire']['dice']))))
    assert sha(__file__)==script_sha;assert all(sha(p)==v for p,v in frozen.items());assert_protected()
    result=dict(status='precise_BN_accumulation_verified',summary=summary,pairs=pairs,records=records,estimation=estimates,checkpoints=checkpoints,
        gradient_updates=0,original26_images_opened=0,production_adopted=False,protocol=protocol,
        elapsed_seconds=time.perf_counter()-start,script_sha256=script_sha,protected_original_files=assert_protected())
    write(output/'precise_evaluation.json',result);print(json.dumps(dict(summary=summary,pairs=pairs,seconds=result['elapsed_seconds'])))


if __name__=='__main__':main()
