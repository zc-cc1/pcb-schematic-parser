"""Fixed learned weights, identical batches: isolate BN estimation order."""
from collections import Counter
import gc
import json
from pathlib import Path
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from unet_audit_common import ROOT,OUT,read,write,sha,counts,metrics,aggregate,assert_protected
from reestimate_unet_BN_v4 import RGBOnlyCrops,fixed_weights_hash
from pcb_parser.unet_bn_statistics import fixed_batches,reestimate_bn_buffers
from pcb_parser.unet_negative_sampling import paired_schedule
from pcb_parser.unet_model import WirePinUNet
from pcb_parser.unet_data import predict_tiled,annotation_targets
from pcb_parser.dataset_converter import load_source_annotation
from pcb_parser.unet_inference import decode_pins
from verify_unet_contrast import path_specs,path_records,pin_metrics


def training_schedule(rows,regions,cfg,arm):
    dataset=RGBOnlyCrops(rows);base=list(dataset.tiles)
    indices={r['case_id']:i for i,r in enumerate(rows)};focus=[]
    for region in regions:
        if not region['hard_negative']:continue
        assert region['split']=='train' and region['case_id'] in indices
        i=indices[region['case_id']];row=rows[i];x0,y0,x1,y1=region['box']
        x=max(0,min(max(0,row['width']-512),(x0+x1)//2-256))
        y=max(0,min(max(0,row['height']-512),(y0+y1)//2-256))
        focus.append((i,x,y))
    a,b,changed=paired_schedule(base,focus,cfg['arms'][arm],cfg['seed'],16)
    dataset.tiles=b if cfg['arms'][arm] else a
    assert len(base)==len(dataset)==925 and len(focus)==9 and len(set(focus))==8
    return dataset,changed


def main():
    output=ROOT/'runs/unet_BN_order_diagnostic_v5';protocol=read(output/'protocol.json')
    assert not (output/'evaluation.json').exists(),'Completed diagnosis exists; do not silently rerun'
    old=ROOT/protocol['source_run'];cfg=read(old/'configuration.json')
    frozen=read(old/'protected_inputs.json')|read(old/'additional_source_attestation.json')
    assert all(sha(p)==v for p,v in frozen.items());assert_protected()
    assert sha(ROOT/'src/pcb_parser/unet_bn_statistics.py')==protocol['learner_sha256']
    assert all(sha(old/a/'last.pt')==h for a,h in protocol['source_weights'].items())
    write(output/'script_attestation.json',dict(script_sha256=sha(__file__),protocol_sha256=sha(output/'protocol.json'),new_training=False))
    rows=[json.loads(s) for s in (ROOT/'runs/unet_training_label_review_v4/training_manifest.jsonl').read_text(encoding='utf-8').splitlines()]
    train=[r for r in rows if r['split']=='train'];cal=[r for r in rows if r['split']=='calibration']
    assert len(train)==93 and len(cal)==10
    assert not set(r['case_id'] for r in rows)&set(read(OUT/'protocol.json')['original_validation_cases'])
    regions=read(ROOT/'runs/unet_training_label_review_v4/confirmed_negative_regions.json')['regions']
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    start=time.perf_counter();checkpoints={};estimates=[];schedules={}
    for arm in protocol['arms']:
        checkpoint=torch.load(old/arm/'last.pt',map_location='cpu',weights_only=True)
        state=checkpoint['state_dict'];dataset,changed=training_schedule(train,regions,cfg,arm)
        batch_list=fixed_batches(len(dataset),2)
        details=[]
        for index,slots in enumerate(batch_list):
            details.append(dict(batch=index+1,slots=slots,case_ids=[train[dataset.tiles[s][0]]['case_id'] for s in slots],
                crops=[list(dataset.tiles[s]) for s in slots],focus_slots=[s for s in slots if s in changed]))
        schedules[arm]=dict(batches=details,forward_last20_case_counts=dict(Counter(c for b in details[-20:] for c in b['case_ids'])),
            reverse_last20_case_counts=dict(Counter(c for b in details[:20] for c in b['case_ids'])),
            forward_last20_focus_slots=sum(len(b['focus_slots']) for b in details[-20:]),
            reverse_last20_focus_slots=sum(len(b['focus_slots']) for b in details[:20]))
        for method,momentum in protocol['momentum_methods'].items():
            for order in protocol['orders']:
                name=f'{arm}__{method}__{order}';folder=output/name;folder.mkdir(exist_ok=True)
                model=WirePinUNet(checkpoint['base_channels']).to('cuda:0');model.load_state_dict(state,strict=True);model.eval()
                batches=fixed_batches(len(dataset),2,reverse=order=='reverse_batches')
                assert sorted(map(tuple,batches))==sorted(map(tuple,batch_list))
                loader=DataLoader(dataset,batch_sampler=batches,num_workers=0)
                def images():
                    for i,batch in enumerate(loader):
                        if (i+1)%100==0:print(json.dumps(dict(phase='RGB_only_BN',variant=name,batches=i+1,total=len(loader))),flush=True)
                        yield batch.to('cuda:0')
                before=time.perf_counter()
                with torch.autocast('cuda',enabled=True):count=reestimate_bn_buffers(model,images(),momentum)
                after={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                assert fixed_weights_hash(after)==fixed_weights_hash(state)
                changed_keys=[k for k in state if not torch.equal(state[k],after[k])]
                assert all(k.endswith(('running_mean','running_var','num_batches_tracked')) for k in changed_keys)
                path=folder/'diagnostic.pt'
                torch.save(dict(checkpoint,state_dict=after,kind='BN_order_diagnostic',gradient_updates=0),path)
                checkpoints[name]=dict(path=str(path),sha256=sha(path),arm=arm,method=method,order=order)
                estimates.append(dict(variant=name,**count,changed_keys=changed_keys,learned_weights_unchanged=True,
                    learned_weights_sha256=fixed_weights_hash(after),elapsed_seconds=time.perf_counter()-before,
                    batch_membership_changed=False,cal_or_old26_images_used_for_estimation=False))
                print(json.dumps(dict(event='estimated',variant=name,seconds=estimates[-1]['elapsed_seconds'])),flush=True)
                del model,after;gc.collect();torch.cuda.empty_cache()
        dataset.rgb.cache_clear()
    write(output/'training_batch_schedules.json',schedules)
    write(output/'estimation.json',dict(checkpoints=checkpoints,records=estimates))
    raw={r['case_id']:r for r in map(json.loads,(ROOT/'data/manifest.jsonl').read_text(encoding='utf-8').splitlines())}
    records=[];models={}
    for name,r in checkpoints.items():
        ckpt=torch.load(r['path'],map_location='cpu',weights_only=True);model=WirePinUNet(ckpt['base_channels']).to('cuda:0')
        model.load_state_dict(ckpt['state_dict'],strict=True);model.eval();models[name]=model
    for row in cal:
        case=row['case_id'];rgb=np.asarray(Image.open(row['image']).convert('RGB'));predictions={}
        for name,model in models.items():
            folder=output/'evaluation'/case/name;folder.mkdir(parents=True,exist_ok=True)
            with torch.inference_mode():p=predict_tiled(model,rgb,'cuda:0',512,384)
            np.savez_compressed(folder/'probabilities.npz',wire=p[0],pin=p[1])
            write(folder/'prediction_meta.json',dict(checkpoint_sha256=checkpoints[name]['sha256'],image_sha256=sha(row['image']),image_only=True,resize=False,tile_size=512,stride=384))
            predictions[name]=p
        gt=np.asarray(Image.open(row['wire_mask']))>0
        annotation,_=load_source_annotation(raw[case]['raw_annotation_path']);reference,*_=annotation_targets(annotation,row['width'],row['height'],3,2.)
        assert np.array_equal(np.asarray(reference)>0,gt)
        specs=path_specs(annotation,gt.shape);tips=read(Path(row['wire_mask']).parent/'pin_points.json')
        for name,p in predictions.items():
            mask=p[0]>=.5;paths=path_records(mask,specs)
            pins=decode_pins(p[1],threshold=.8,kernel=5,refinement='quadratic',nms_radius=0.,limit=500)
            r=dict(case_id=case,variant=name,wire=metrics(*counts(mask,gt)),pin=pin_metrics(pins,tips),connected_paths=sum(v['connected_within_reference_tube'] for v in paths),total_paths=len(paths))
            records.append(r);write(output/'evaluation'/case/name/'metrics.json',r)
            Image.fromarray(mask.astype(np.uint8)*255).save(output/'evaluation'/case/name/'wire_mask.png')
        print(json.dumps(dict(event='cal_page_scored',case_id=case,pages=len({r['case_id'] for r in records}))),flush=True)
    summary={}
    for name in checkpoints:
        selected=[r for r in records if r['variant']==name]
        summary[name]=dict(wire=aggregate(selected),pin=aggregate(selected,'pin'),connected_paths=sum(r['connected_paths'] for r in selected),total_paths=sum(r['total_paths'] for r in selected))
    pairs=[];rng=np.random.default_rng(20261014);ids=sorted(r['case_id'] for r in cal);draw=rng.integers(0,10,(10000,10))
    sampled={}
    for name in summary:
        rr={r['case_id']:r for r in records if r['variant']==name}
        a=np.array([[rr[c]['wire'][k] for k in ('tp','fp','fn')] for c in ids]);t=a[draw].sum(axis=1)
        sampled[name]=2*t[:,0]/np.maximum(1,2*t[:,0]+t[:,1]+t[:,2])
    for arm in protocol['arms']:
        for method in protocol['momentum_methods']:
            a=f'{arm}__{method}__forward';b=f'{arm}__{method}__reverse_batches'
            drift=[];changed=gtpixels=totalpixels=0;maxdiff=0.;sumdiff=0.
            sa=torch.load(checkpoints[a]['path'],map_location='cpu',weights_only=True)['state_dict']
            sb=torch.load(checkpoints[b]['path'],map_location='cpu',weights_only=True)['state_dict']
            for k in sa:
                if k.endswith(('running_mean','running_var')):
                    av=sa[k].numpy();bv=sb[k].numpy();dv=av-bv
                    drift.append(dict(key=k,max_abs=float(np.abs(dv).max()),L2=float(np.linalg.norm(dv)),relative_L2=float(np.linalg.norm(dv)/max(1e-12,np.linalg.norm((av+bv)/2)))))
            for case in ids:
                with np.load(output/'evaluation'/case/a/'probabilities.npz',allow_pickle=False) as source:p=source['wire']
                with np.load(output/'evaluation'/case/b/'probabilities.npz',allow_pickle=False) as source:q=source['wire']
                diff=np.abs(p-q);maxdiff=max(maxdiff,float(diff.max()));sumdiff+=float(diff.sum(dtype=np.float64));totalpixels+=p.size
                changed+=int(((p>=.5)!=(q>=.5)).sum())
            x=summary[a];y=summary[b]
            pairs.append(dict(arm=arm,method=method,contrast='reverse_batches minus forward',delta_dice_pp=100*(y['wire']['dice']-x['wire']['dice']),
                delta_recall_pp=100*(y['wire']['recall']-x['wire']['recall']),delta_precision_pp=100*(y['wire']['precision']-x['wire']['precision']),
                delta_connected_paths=y['connected_paths']-x['connected_paths'],changed_mask_pixels=changed,cal_pixels=totalpixels,changed_mask_fraction=changed/totalpixels,
                max_abs_probability_difference=maxdiff,mean_abs_probability_difference=sumdiff/totalpixels,buffer_drift=drift,
                paired_page_bootstrap95_dice_pp=np.quantile(100*(sampled[b]-sampled[a]),[.025,.975]).tolist(),
                order_sensitive_observed=abs(100*(y['wire']['dice']-x['wire']['dice']))>=1 or abs(y['connected_paths']-x['connected_paths'])>=10,
                stable_under_preregistered_tolerance=maxdiff<=1e-4 and changed/totalpixels<=1e-5 and abs(y['connected_paths']-x['connected_paths'])<=1))
    assert all(sha(p)==v for p,v in frozen.items());assert_protected()
    assert read(output/'script_attestation.json')['script_sha256']==sha(__file__)
    result=dict(status='fixed_weights_BN_batch_order_diagnostic_complete',summary=summary,pairs=pairs,records=records,estimation=estimates,
        original26_images_opened=0,original26_metrics_computed=0,gradient_updates=0,production_adopted=False,
        parameters_selected_on_calibration=False,protocol=protocol,elapsed_seconds=time.perf_counter()-start,protected_original_files=assert_protected())
    write(output/'evaluation.json',result)
    print(json.dumps(dict(summary=summary,pairs=[{k:v for k,v in r.items() if k!='buffer_drift'} for r in pairs],seconds=result['elapsed_seconds'])))


if __name__=='__main__':main()
