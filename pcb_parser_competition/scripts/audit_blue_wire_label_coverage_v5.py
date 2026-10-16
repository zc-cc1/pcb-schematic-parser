"""Locate long blue raster strokes missing from labels; never alter labels."""
from collections import Counter
import copy
import json
import numpy as np
from PIL import Image,ImageDraw
from scipy import ndimage
from unet_audit_common import ROOT,read,write,sha,assert_protected
from pcb_parser.dataset_converter import load_source_annotation
from pcb_parser.unet_data import annotation_targets,polylines
from pcb_parser.unet_retraining import CachedTrainingTiles
import torch


def main():
    out=ROOT/'runs/unet_BN_order_diagnostic_v5/blue_label_review';out.mkdir(exist_ok=True)
    rows=[json.loads(s) for s in (ROOT/'runs/unet_training_label_review_v4/training_manifest.jsonl').read_text(encoding='utf-8').splitlines()]
    raw={r['case_id']:r for r in map(json.loads,(ROOT/'data/manifest.jsonl').read_text(encoding='utf-8').splitlines())}
    candidates=[]
    for row in rows:
        rgb=np.asarray(Image.open(row['image']).convert('RGB')).astype(np.int16);gt=np.asarray(Image.open(row['wire_mask']))>0
        blue=(rgb[:,:,2]>=80)&(rgb[:,:,2]-rgb[:,:,0]>=40)&(rgb[:,:,2]-rgb[:,:,1]>=40)
        distance=ndimage.distance_transform_edt(~gt)
        for y,line in enumerate(blue):
            marks=np.diff(np.pad(line.astype(np.int8),(1,1)));starts=np.flatnonzero(marks==1);ends=np.flatnonzero(marks==-1)
            for x0,x1 in zip(starts,ends):
                if x1-x0<40:continue
                missing=float((distance[y,x0:x1]>2).mean())
                if missing<.9:continue
                candidates.append(dict(case_id=row['case_id'],split=row['split'],y=int(y),x_interval_half_open=[int(x0),int(x1)],
                    length=int(x1-x0),GT_pixels=int(gt[y,x0:x1].sum()),far_from_GT2_fraction=missing,semantic_class_reviewed=False))
    write(out/'candidate_scan.json',dict(scope='103 current train/cal pages; long blue horizontal raster strokes >=40px; >90pct farther than2px fromGT. Includes symbols/borders/text possibilities; NOT validated missing-wire counts or noise rate.',
        records=candidates,summary={split:dict(candidate_strokes=sum(r['split']==split for r in candidates),pages=len({r['case_id'] for r in candidates if r['split']==split}),
            total_candidate_pixels=sum(r['length'] for r in candidates if r['split']==split)) for split in ('train','calibration')}))
    case='0093';rgb=np.asarray(Image.open(ROOT/f'data/standardized/{case}/{case}.png').convert('RGB'))
    gt=np.asarray(Image.open(ROOT/f'data/segmentation_unet_v1/{case}/wire_mask.png'))>0
    annotation,_=load_source_annotation(raw[case]['raw_annotation_path']);all_annotation=copy.deepcopy(annotation)
    for e in all_annotation['edges'].values():e['state']='active';e['isFlywire']=False
    all_paths,*_=annotation_targets(all_annotation,rgb.shape[1],rgb.shape[0],3,2.)
    all_paths=np.asarray(all_paths)>0
    skeleton=Image.new('1',(rgb.shape[1],rgb.shape[0]));draw=ImageDraw.Draw(skeleton)
    for e in annotation['edges'].values():
        values=e.get('keyPoints',{}).get('skeleton',{}).get('wire',[])
        for p in values:
            if isinstance(p,list) and len(p)==2:draw.point(tuple(p),fill=1)
    skeleton=np.asarray(skeleton,dtype=bool);checks=[]
    model_sources={
        'v4_control':ROOT/'runs/unet_negative_sampling_pilot_v4/evaluation/0093/control/probabilities.npz',
        'v4_negative_focus':ROOT/'runs/unet_negative_sampling_pilot_v4/evaluation/0093/negative_focus/probabilities.npz',
        'focus_precise_native':ROOT/'runs/unet_color_response_v5/evaluation/0093/native/probabilities.npz',
        'focus_mean_gray':ROOT/'runs/unet_color_response_v5/evaluation/0093/mean_gray/probabilities.npz'}
    probs={}
    for name,path in model_sources.items():
        with np.load(path,allow_pickle=False) as saved:probs[name]=saved['wire'].copy()
    for y in (448,546):
        x0,x1=110,180;values=rgb[y,x0:x1]
        assert np.all(values==np.array([0,0,128])) and not gt[y,x0:x1].any()
        assert not all_paths[y,x0:x1].any() and not skeleton[y,x0:x1].any()
        checks.append(dict(y=y,x_interval_half_open=[x0,x1],pixels=x1-x0,RGB_unique=np.unique(values,axis=0).tolist(),
            current_GT_positives=int(gt[y,x0:x1].sum()),all_states_source_path_positives=int(all_paths[y,x0:x1].sum()),
            all_states_source_skeleton_points=int(skeleton[y,x0:x1].sum()),
            predictions={name:dict(detected_pixels=int((p[y,x0:x1]>=.5).sum()),
                counted_as_FP_by_current_GT=int((p[y,x0:x1]>=.5).sum()),median_probability=float(np.median(p[y,x0:x1]))) for name,p in probs.items()},
            local_visual_class='wire route between capacitor/XTAL junctions; inspected PNG; no second annotator'))
    counts=Counter((str(e.get('state')),str(e.get('category')),str(e.get('isFlywire'))) for e in annotation['edges'].values())
    local=[]
    for eid,e in annotation['edges'].items():
        paths=list(polylines(e.get('keyPointPaths',{}).get('wire',[])))
        if any(all(90<=x<340 and 440<=y<555 for x,y in p) for p in paths):local.append(dict(edge_id=eid,state=e.get('state'),paths=paths))
    result=dict(case_id=case,scope='two local agent visually reviewed70px wire segments; calibration frozen; no overall corrected Dice or annotation ceiling',
        source_annotation_path=raw[case]['raw_annotation_path'],source_sha256=sha(raw[case]['raw_annotation_path']),
        state_counts=[dict(state=s,category=c,isFlywire=f,count=n) for (s,c,f),n in counts.items()],checks=checks,local_source_paths=local,
        finding='140 visible dark-blue wire pixels have0 GT positives and0 source path/skeleton coverage even after including deleted edges; origin is missing source geometry, not rasterization or flywire/deleted filtering.',
        all_source_edges=len(annotation['edges']),source_edges_flywire=sum(bool(e.get('isFlywire')) for e in annotation['edges'].values()),
        labels_changed=False,calibration_removed=False,evaluation_metrics_recomputed_with_corrected_GT=False,independent_second_annotation=False,
        protected_original_files=assert_protected())
    write(out/'0093_review.json',result)
    # A train-side, same-page control: KEY1 is labeled; KEY4 is missing.
    case='0025';row=next(r for r in rows if r['case_id']==case)
    assert row['split']=='train'
    rgb=np.asarray(Image.open(row['image']).convert('RGB'));gt=np.asarray(Image.open(row['wire_mask']))>0
    annotation,_=load_source_annotation(raw[case]['raw_annotation_path']);all_annotation=copy.deepcopy(annotation)
    for e in all_annotation['edges'].values():e['state']='active';e['isFlywire']=False
    all_paths,*_=annotation_targets(all_annotation,row['width'],row['height'],3,2.);all_paths=np.asarray(all_paths)>0
    skeleton=Image.new('1',(row['width'],row['height']));draw=ImageDraw.Draw(skeleton)
    for e in annotation['edges'].values():
        for p in e.get('keyPoints',{}).get('skeleton',{}).get('wire',[]):
            if isinstance(p,list) and len(p)==2:draw.point(tuple(p),fill=1)
    skeleton=np.asarray(skeleton,dtype=bool);train_checks=[]
    image,target,valid=CachedTrainingTiles([row],512,384)[0]
    predictions={}
    for name in ('control','negative_focus'):
        with np.load(ROOT/'runs/unet_negative_sampling_pilot_v4/evaluation/0025'/name/'probabilities.npz',allow_pickle=False) as saved:predictions[name]=saved['wire'].copy()
    for y,label in ((151,'KEY1 labeled control'),(369,'KEY4 missing route')):
        x0,x1=50,140;v=rgb[y,x0:x1]
        assert np.all(v[:,2].astype(int)-v[:,0].astype(int)>=40)
        assert int(gt[y,x0:x1].sum())==(90 if y==151 else 0)
        assert int(all_paths[y,x0:x1].sum())==(90 if y==151 else 0)
        train_checks.append(dict(label=label,y=y,x_interval_half_open=[x0,x1],pixels=90,RGB_unique=np.unique(v,axis=0).tolist(),
            current_GT_positives=int(gt[y,x0:x1].sum()),all_states_source_path_positives=int(all_paths[y,x0:x1].sum()),
            source_skeleton_points=int(skeleton[y,x0:x1].sum()),actual_training_positive_targets=int(target[0,y,x0:x1].sum()),
            actual_training_valid_pixels=int(valid[y,x0:x1].sum()),
            predictions={name:dict(detected_pixels=int((p[y,x0:x1]>=.5).sum()),median_probability=float(np.median(p[y,x0:x1]))) for name,p in predictions.items()}))
    logits=torch.zeros(90,requires_grad=True)
    torch.nn.functional.binary_cross_entropy_with_logits(logits,torch.zeros_like(logits),pos_weight=torch.tensor(8.),reduction='sum').backward()
    assert torch.all(logits.grad==.5)
    write(out/'0025_review.json',dict(case_id=case,split='train',scope='two local visually checked90px samples on real KEY1/KEY4 wire routes; no independent annotation or whole-page certification',
        checks=train_checks,source_annotation_path=raw[case]['raw_annotation_path'],source_sha256=sha(raw[case]['raw_annotation_path']),
        background_BCE_gradient_at_zero_logit=float(logits.grad[0]),wire_pos_weight=8,
        finding='Visible KEY4 route has0 GT/source-path geometry and90 supervised-negative pixels. Same-page KEY1 route has90/90 positive labels. Unlabeled KEY4 is actively penalized as background in current93 training data.',
        labels_changed=False,calibration_used_for_training=False,independent_second_annotation=False,protected_original_files=assert_protected()))
    print(json.dumps(dict(candidate_scan_summary=read(out/'candidate_scan.json')['summary'],reviewed_cases=['0093','0025'],
        checks_0093=checks,checks_0025=train_checks),ensure_ascii=False))


if __name__=='__main__':main()
