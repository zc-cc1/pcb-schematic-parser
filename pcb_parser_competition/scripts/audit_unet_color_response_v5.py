"""Fixed image-only color probes; never select or deploy a transform."""
import json
import time
import numpy as np
from PIL import Image
import torch
from unet_audit_common import ROOT,read,write,sha,counts,metrics,aggregate,assert_protected
from pcb_parser.unet_model import WirePinUNet
from pcb_parser.unet_data import predict_tiled,annotation_targets
from pcb_parser.dataset_converter import load_source_annotation
from verify_unet_contrast import path_specs,path_records


def masks(rgb):
    color=(rgb.max(2).astype(np.int16)-rgb.min(2).astype(np.int16)>=32)&(rgb.min(2)<224)
    grey=(~color)&(rgb.min(2)<224)
    return dict(chromatic_ink=color,achromatic_ink=grey,light=~(color|grey))


def transform(rgb,policy):
    if policy=='native':return rgb
    if policy=='cycle_GBR':return rgb[:,:,[1,2,0]].copy()
    if policy=='cycle_BRG':return rgb[:,:,[2,0,1]].copy()
    if policy=='mean_gray':return np.repeat(np.rint(rgb.astype(np.uint16).sum(2)/3).astype(np.uint8)[:,:,None],3,axis=2)
    raise ValueError(policy)


def main():
    output=ROOT/'runs/unet_color_response_v5';protocol=read(output/'protocol.json')
    assert not (output/'evaluation.json').exists();assert sha(protocol['source_checkpoint'])==protocol['source_sha256']
    protected=read(ROOT/'runs/unet_negative_sampling_pilot_v4/protected_inputs.json')|read(ROOT/'runs/unet_negative_sampling_pilot_v4/additional_source_attestation.json')
    assert all(sha(p)==v for p,v in protected.items());assert_protected()
    scriptsha=sha(__file__);write(output/'script_attestation.json',dict(script_sha256=scriptsha,protocol_sha256=sha(output/'protocol.json')))
    rows=[json.loads(s) for s in (ROOT/'runs/unet_training_label_review_v4/training_manifest.jsonl').read_text(encoding='utf-8').splitlines()]
    train=[r for r in rows if r['split']=='train'];cal=[r for r in rows if r['split']=='calibration'];assert len(train)==93 and len(cal)==10
    style=[]
    for row in rows:
        rgb=np.asarray(Image.open(row['image']).convert('RGB'));gt=np.asarray(Image.open(row['wire_mask']))>0;roi=masks(rgb)
        style.append(dict(case_id=row['case_id'],split=row['split'],GT_pixels=int(gt.sum()),
            chromatic_GT_pixels=int((gt&roi['chromatic_ink']).sum()),achromatic_ink_GT_pixels=int((gt&roi['achromatic_ink']).sum()),
            light_GT_pixels=int((gt&roi['light']).sum())))
    write(output/'style_audit.json',dict(scope='image-pixel chroma proxy on existing labels; not certified source/style/line-width classes',records=style,
        summaries={split:dict(pages=len([r for r in style if r['split']==split]),
            **{k:sum(r[k] for r in style if r['split']==split) for k in ('GT_pixels','chromatic_GT_pixels','achromatic_ink_GT_pixels','light_GT_pixels')}) for split in ('train','calibration')}))
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    ckpt=torch.load(protocol['source_checkpoint'],map_location='cpu',weights_only=True);model=WirePinUNet(ckpt['base_channels']).to('cuda:0')
    model.load_state_dict(ckpt['state_dict'],strict=True);model.eval();start=time.perf_counter();records=[]
    raw={r['case_id']:r for r in map(json.loads,(ROOT/'data/manifest.jsonl').read_text(encoding='utf-8').splitlines())}
    for row in cal:
        case=row['case_id'];rgb=np.asarray(Image.open(row['image']).convert('RGB'));predictions={}
        for policy in protocol['policies']:
            value=transform(rgb,policy);assert value.shape==rgb.shape
            delta=float(np.abs(value.astype(np.float64).mean(2)-rgb.astype(np.float64).mean(2)).max())
            assert delta<(1/3+1e-10) if policy=='mean_gray' else delta==0
            folder=output/'evaluation'/case/policy;folder.mkdir(parents=True,exist_ok=True)
            if policy=='native':
                source=ROOT/'runs/unet_BN_order_diagnostic_v5/evaluation'/case/'negative_focus__precise_cumulative__forward'
                assert read(source/'prediction_meta.json')['checkpoint_sha256']==protocol['source_sha256']
                with np.load(source/'probabilities.npz',allow_pickle=False) as saved:p=saved['wire'].copy()
            else:
                with torch.inference_mode():p=predict_tiled(model,value,'cuda:0',512,384)[0]
            predictions[policy]=p;np.savez_compressed(folder/'probabilities.npz',wire=p)
            write(folder/'prediction_meta.json',dict(image_only=True,source_image_sha256=sha(row['image']),checkpoint_sha256=protocol['source_sha256'],transform=policy,
                max_arithmetic_mean_change=delta,geometry_changed=False,labels_used_for_transform=False))
        gt=np.asarray(Image.open(row['wire_mask']))>0;annotation,_=load_source_annotation(raw[case]['raw_annotation_path'])
        reference,*_=annotation_targets(annotation,row['width'],row['height'],3,2.);assert np.array_equal(np.asarray(reference)>0,gt)
        specs=path_specs(annotation,gt.shape);roi=masks(rgb)
        for policy,p in predictions.items():
            mask=p>=.5;paths=path_records(mask,specs)
            result=dict(case_id=case,policy=policy,wire=metrics(*counts(mask,gt)),connected_paths=sum(v['connected_within_reference_tube'] for v in paths),total_paths=len(paths),
                slices={k:metrics(*counts(mask,gt,r)) for k,r in roi.items()})
            assert sum(v['tp'] for v in result['slices'].values())==result['wire']['tp']
            records.append(result);write(output/'evaluation'/case/policy/'metrics.json',result)
            Image.fromarray(mask.astype(np.uint8)*255).save(output/'evaluation'/case/policy/'wire_mask.png')
        print(json.dumps(dict(case_id=case,completed=len({r['case_id'] for r in records}),total=10)),flush=True)
    summary={}
    for policy in protocol['policies']:
        selected=[r for r in records if r['policy']==policy]
        summary[policy]=dict(wire=aggregate(selected),connected_paths=sum(r['connected_paths'] for r in selected),total_paths=sum(r['total_paths'] for r in selected),
            slices={k:aggregate([dict(wire=r['slices'][k]) for r in selected]) for k in ('chromatic_ink','achromatic_ink','light')})
    native=summary['native'];diffs={}
    for policy,value in summary.items():
        diffs[policy]=dict(delta_dice_pp=100*(value['wire']['dice']-native['wire']['dice']),delta_recall_pp=100*(value['wire']['recall']-native['wire']['recall']),
            delta_connected_paths=value['connected_paths']-native['connected_paths'],chromatic_ink_recall_delta_pp=100*(value['slices']['chromatic_ink']['recall']-native['slices']['chromatic_ink']['recall']))
    assert sha(__file__)==scriptsha;assert all(sha(p)==v for p,v in protected.items());assert_protected()
    result=dict(status='fixed_color_response_probe_complete',summary=summary,comparisons=diffs,records=records,protocol=protocol,
        elapsed_seconds=time.perf_counter()-start,original26_images_opened=0,production_adopted=False,transform_selected=False,gradient_updates=0,protected_original_files=assert_protected())
    write(output/'evaluation.json',result);print(json.dumps(dict(summary=summary,comparisons=diffs,seconds=result['elapsed_seconds'])))


if __name__=='__main__':main()
