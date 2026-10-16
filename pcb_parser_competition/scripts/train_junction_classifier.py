"""Train a small junction pilot; no changes to YOLO/U-Net weights or datasets."""
import argparse
from collections import Counter
import random
import sys
import time

from run_wire_graph import ROOT,read,save,sha
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader,ConcatDataset,TensorDataset
from pcb_parser.junction_classifier import JunctionCNN,JunctionPredictor,CLASSES,RADII
from pcb_parser.junction_training_data import ReviewedDataset,render_synthetic


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--config',type=__import__('pathlib').Path,default=ROOT/'configs/junction_training_v1.json')
    parser.add_argument('--output',type=__import__('pathlib').Path,default=ROOT/'runs/training/junction_classifier_v1')
    args=parser.parse_args();cfg=read(args.config);start=time.perf_counter()
    random.seed(cfg['seed']);np.random.seed(cfg['seed']);torch.manual_seed(cfg['seed']);torch.set_num_threads(4)
    reviews=ROOT/'data/junction_classifier_v1/reviewed_annotations.json';annotation=read(reviews);records=annotation['records']
    for row in records:
        if sha(ROOT/'data/topology_review_v1'/row['crop_file'])!=row['crop_sha256']:raise ValueError('Review crop changed')
        if sha(ROOT/f"data/standardized/{row['case_id']}/{row['case_id']}.png")!=row['source_image_sha256']:raise ValueError('Review source image changed')
    groups={s:{r['source_image_sha256'] for r in records if r['split']==s} for s in ['train','calibration','test']}
    if groups['train']&groups['calibration'] or groups['train']&groups['test'] or groups['test']&groups['calibration']:raise ValueError('Image group leakage')
    folder=ROOT/'data/topology_review_v1'
    train=ReviewedDataset(folder,[r for r in records if r['split']=='train'],True,cfg['real_train_repeats'])
    calibration=ReviewedDataset(folder,[r for r in records if r['split']=='calibration'])
    if not len(train) or not len(calibration) or {int(y) for x,y in train.items}!={0,1}:
        raise ValueError('Training needs both classes and a nonempty separate calibration group')
    synthetic=[render_synthetic(cfg['seed']+i) for i in range(cfg['synthetic_count'])]
    synth=TensorDataset(torch.stack([x for x,y in synthetic]),torch.tensor([y for x,y in synthetic]))
    loader=DataLoader(ConcatDataset([train,synth]),batch_size=cfg['batch_size'],shuffle=True,num_workers=0)
    validation=DataLoader(calibration,batch_size=64)
    device=torch.device(args.device);model=JunctionCNN().to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=.0001)
    args.output.mkdir(parents=True,exist_ok=True);best=float('inf');history=[];best_epoch=None
    for epoch in range(1,cfg['epochs']+1):
        model.train();losses=[]
        for x,y in loader:
            optimizer.zero_grad(set_to_none=True);logits=model(x.to(device));loss=torch.nn.functional.cross_entropy(logits,y.to(device))
            if not torch.isfinite(loss):raise ValueError('Non-finite classifier loss')
            loss.backward();optimizer.step();losses.append(float(loss.detach()))
        model.eval()
        with torch.inference_mode():
            val=torch.cat([model(x.to(device)).cpu() for x,y in validation]);expected=torch.tensor([int(calibration[i][1]) for i in range(len(calibration))])
            val_loss=float(torch.nn.functional.cross_entropy(val,expected));accuracy=float((val.argmax(1)==expected).float().mean())
        row={'epoch':epoch,'train_loss':sum(losses)/len(losses),'calibration_loss':val_loss,'calibration_accuracy':accuracy}
        history.append(row)
        if val_loss<best:
            best=val_loss;best_epoch=epoch
            torch.save({'kind':'junction_cnn_smoke_only' if cfg.get('smoke_only') else 'junction_cnn_development_trained',
                        'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},
                        'classes':CLASSES,'radii':RADII,'config':cfg,'epoch':epoch,'reviews_sha256':sha(reviews),
                        'split_groups':annotation['split_groups'],'upstream_test_independent':False},args.output/'best.pt')
        if epoch==1 or epoch%5==0:print(row,flush=True)
    if cfg.get('smoke_only'):
        result={'status':'training_execution_smoke_only','history':history,'best_epoch':best_epoch,
                'wall_seconds':time.perf_counter()-start,'checkpoint_sha256':sha(args.output/'best.pt'),
                'quality_scores_computed':False,'cannot_be_used_by_graph_inference':True}
        save(args.output/'evaluation.json',result);save(args.output/'training_config.json',cfg)
        print(result,flush=True);return
    predictor=JunctionPredictor(args.output/'best.pt',args.device)
    predictions=[]
    for row in records:
        if row['split'] not in {'calibration','test'}:continue
        with Image.open(ROOT/f"data/standardized/{row['case_id']}/{row['case_id']}.png") as source:rgb=np.asarray(source.convert('RGB'))
        candidate={'junction_id':row['review_id'],'point':row['center_top_left'],'degree':row['candidate']['degree']}
        decision=predictor.predict(rgb,[candidate],cfg['confidence_threshold'])[row['review_id']]
        predictions.append({'review_id':row['review_id'],'case_id':row['case_id'],'split':row['split'],'expected':row['label'],**decision})
    scores={}
    for split in ['calibration','test']:
        selected=[r for r in predictions if r['split']==split and r['expected'] in CLASSES]
        confusion=Counter((r['expected'],r['label']) for r in selected)
        accepted=[r for r in selected if r['accepted']]
        scores[split]={'count':len(selected),'raw_correct':sum(r['expected']==r['label'] for r in selected),
                       'accepted_count':len(accepted),'accepted_correct':sum(r['expected']==r['label'] for r in accepted),
                       'abstained':len(selected)-len(accepted),'confusion':{a+' -> '+b:n for (a,b),n in confusion.items()}}
    result={'history':history,'best_epoch':best_epoch,'parameters':sum(p.numel() for p in model.parameters()),
            'synthetic_count':cfg['synthetic_count'],'review_counts':annotation['split_counts'],'scores':scores,'predictions':predictions,
            'checkpoint_sha256':sha(args.output/'best.pt'),'reviews_sha256':sha(reviews),'wall_seconds':time.perf_counter()-start,
            'confidence_threshold_fixed_before_test':cfg['confidence_threshold'],'upstream_test_independent':False,
            'test_scope':'Source-group held out for this classifier only; 13 development crops, assistant-reviewed, not official Golden',
            'artifact_class_learned':False}
    save(args.output/'evaluation.json',result);save(args.output/'training_config.json',cfg)
    print({k:result[k] for k in ['best_epoch','parameters','scores','checkpoint_sha256','wall_seconds']},flush=True)


if __name__=='__main__':main()
