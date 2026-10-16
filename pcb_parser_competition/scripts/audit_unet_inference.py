"""Predeclared image-only inference ablations; no validation-based selection."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize
import torch

from unet_audit_common import ROOT, OUT, read, write, sha, image, truth, probability, counts, metrics, aggregate, val_cases, assert_protected
from pcb_parser.unet_data import predict_tiled, tile_starts
from pcb_parser.unet_model import WirePinUNet
from pcb_parser.dataset_converter import load_source_annotation
from verify_unet_contrast import path_specs, path_records

MODEL_PATHS = {
    "v1": ROOT/"weights/unet_wire_pin_v1.pt",
    "control": ROOT/"runs/unet_retrain_v2/control/best.pt",
    "contrast": ROOT/"runs/unet_retrain_v2/contrast/best.pt",
    "rotation": ROOT/"runs/unet_retrain_rotation_v3/rotation/best.pt",
    "v1_last": Path("C:/Users/Pc/Desktop/新建文件夹/pcb_parser_competition/runs/training/unet_wire_pin_v1/last.pt"),
}
CONFIG = dict(seed=20261014, threshold=.5, scales=[.75,1.,1.25], d4="flip horizontal then rotate 0/90/180/270; inverse exactly",
    resize_RGB="Pillow bilinear", resize_probability="Pillow F bilinear", resize_GT="nearest only in counterfactual fit512 input-grid evaluation",
    patches={"512":dict(stride=384),"1024":dict(stride=768)}, gaussian_sigma_fraction=.125, gaussian_weight_floor=1e-3,
    closing_kernel=[3,3], remove_component_area_le=[4,16],
    linear_filter="keep skeleton_length>=4 and (bbox_aspect>=3 or skeleton_length/sqrt(area)>=4)",
    bridge="mutual nearest skeleton endpoints; distance<=3px (<=2px missing); outgoing tangent angle<=30deg at both ends; width1; no GT",
    ensembles="equal mean probabilities; v1+control and v1+control+contrast+rotation",
    additional_controls_predeclared=["uniform1024 separates patch size from weighting", "original last60 vs best57", "rotation weights with frozen original v1 BN running buffers"],
    fitting_on_validation=False, adopting_result=False, training_started=False)


def resize_p(p, shape):
    return np.asarray(Image.fromarray(p.astype(np.float32)).resize((shape[1],shape[0]), Image.Resampling.BILINEAR)).copy()


def load_model(name, buffers_from_v1=False):
    checkpoint=torch.load(MODEL_PATHS[name],map_location="cpu",weights_only=True)
    state=checkpoint["state_dict"]
    if buffers_from_v1:
        original=torch.load(MODEL_PATHS["v1"],map_location="cpu",weights_only=True)["state_dict"]
        state={k:original[k] if k.endswith(("running_mean","running_var","num_batches_tracked")) else v for k,v in state.items()}
    model=WirePinUNet(checkpoint["base_channels"]).to("cuda:0")
    model.load_state_dict(state,strict=True);model.eval()
    return model


@torch.inference_mode()
def tiled(model,rgb,size=512,gaussian=False):
    if not gaussian:
        return predict_tiled(model,rgb,"cuda:0",size,3*size//4)[0]
    h,w=rgb.shape[:2];output=np.zeros((h,w),np.float32);total=np.zeros((h,w),np.float32)
    coordinate=np.arange(size)-(size-1)/2
    g=np.exp(-.5*(coordinate/(size*CONFIG["gaussian_sigma_fraction"]))**2)
    weight=np.maximum(np.outer(g,g),CONFIG["gaussian_weight_floor"]).astype(np.float32)
    for y in tile_starts(h,size,3*size//4):
        for x in tile_starts(w,size,3*size//4):
            hh,ww=min(size,h-y),min(size,w-x)
            crop=np.full((size,size,3),255,np.uint8);crop[:hh,:ww]=rgb[y:y+hh,x:x+ww]
            batch=torch.from_numpy(crop.transpose(2,0,1).copy()).float()[None].to("cuda:0")/255
            p=model(batch).sigmoid()[0,0,:hh,:ww].cpu().numpy()
            output[y:y+hh,x:x+ww]+=p*weight[:hh,:ww];total[y:y+hh,x:x+ww]+=weight[:hh,:ww]
    return output/total


def cached(case,name,fn):
    folder=OUT/"inference"/case;folder.mkdir(parents=True,exist_ok=True)
    path=folder/f"{name}.npz"
    if path.exists():
        with np.load(path,allow_pickle=False) as f:return f["wire"].copy()
    p=np.ascontiguousarray(fn(),dtype=np.float32)
    np.savez_compressed(path,wire=p)
    return p


def remove_small(mask,maximum):
    labels,_=ndimage.label(mask,np.ones((3,3)));sizes=np.bincount(labels.ravel())
    keep=sizes>maximum;keep[0]=False
    return keep[labels]


def linear_filter(mask):
    labels,n=ndimage.label(mask,np.ones((3,3)));out=np.zeros_like(mask)
    for i,box in enumerate(ndimage.find_objects(labels),1):
        if box is None:continue
        part=labels[box]==i;area=int(part.sum());length=int(skeletonize(part).sum())
        hh,ww=part.shape;aspect=max(hh,ww)/max(1,min(hh,ww))
        if length>=4 and (aspect>=3 or length/math.sqrt(area)>=4):out[box]|=part
    return out


def bridge(mask):
    sk=skeletonize(mask);degree=ndimage.convolve(sk.astype(np.uint8),np.ones((3,3)),mode="constant")-sk
    endpoints=np.argwhere(sk & (degree==1));out=Image.fromarray(mask.astype(np.uint8)*255)
    if len(endpoints)<2:return mask.copy()
    tree=cKDTree(endpoints);distance,index=tree.query(endpoints,k=2)
    def tangent(point):
        previous=None;current=tuple(point)
        for _ in range(4):
            y,x=current
            neighbors=[(yy,xx) for yy in range(max(0,y-1),min(sk.shape[0],y+2)) for xx in range(max(0,x-1),min(sk.shape[1],x+2))
                       if sk[yy,xx] and (yy,xx)!=current and (yy,xx)!=previous]
            if len(neighbors)!=1:break
            previous,current=current,neighbors[0]
        v=point-np.array(current);norm=np.linalg.norm(v)
        return v/norm if norm else v
    tangents=[tangent(p) for p in endpoints];draw=ImageDraw.Draw(out)
    for i in range(len(endpoints)):
        j=int(index[i,1]);d=distance[i,1]
        if j<=i or d>3 or int(index[j,1])!=i:continue
        direction=(endpoints[j]-endpoints[i])/d
        if np.dot(tangents[i],direction)>=math.cos(math.pi/6) and np.dot(tangents[j],-direction)>=math.cos(math.pi/6):
            y,x=endpoints[i];yy,xx=endpoints[j];draw.line([(int(x),int(y)),(int(xx),int(yy))],fill=255,width=1)
    return np.asarray(out)>0


def postprocess(mask):
    padded=np.pad(mask,1)
    return {"closing3":ndimage.binary_closing(padded,np.ones((3,3)))[1:-1,1:-1],
        "remove_area4":remove_small(mask,4),"remove_area16":remove_small(mask,16),
        "linear_components":linear_filter(mask),"endpoint_bridge2":bridge(mask)}


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--case",nargs="*");args=parser.parse_args()
    assert_protected();write(OUT/"inference_protocol.json",CONFIG|dict(checkpoints={k:dict(path=str(v),sha256=sha(v)) for k,v in MODEL_PATHS.items()}))
    torch.manual_seed(CONFIG["seed"]);np.random.seed(CONFIG["seed"]);torch.set_num_threads(4)
    if not torch.cuda.is_available():raise RuntimeError("CUDA needed; no training fallback")
    v1=load_model("v1");others={k:load_model(k) for k in ("control","contrast","rotation","v1_last")}
    others["rotation_v1_BN_buffers"]=load_model("rotation",True)
    raw={r["case_id"]:r for r in map(json.loads,(ROOT/"data/manifest.jsonl").read_text(encoding="utf-8").splitlines())}
    allrecords=[];start=time.perf_counter()
    for i,case in enumerate(args.case or val_cases()):
        begin=time.perf_counter();rgb=image(case);shape=rgb.shape[:2];variants={"native_uniform512":probability(case)};timings={}
        scale_means=[];d4_means=[]
        for scale in CONFIG["scales"]:
            t=time.perf_counter();scaled=np.asarray(Image.fromarray(rgb).resize((max(1,round(shape[1]*scale)),max(1,round(shape[0]*scale))),Image.Resampling.BILINEAR)) if scale!=1 else rgb
            single=variants["native_uniform512"] if scale==1 else cached(case,f"scale{scale}",lambda:tiled(v1,scaled))
            single_orig=single if scale==1 else resize_p(single,shape)
            if scale!=1:variants["scale075" if scale==.75 else "scale125"]=single_orig
            scale_means.append(single_orig)
            def d4_prediction():
                values=[]
                for flipped in (False,True):
                    for k in range(4):
                        transformed=np.rot90(np.fliplr(scaled) if flipped else scaled,k)
                        p=single if not flipped and k==0 else tiled(v1,np.ascontiguousarray(transformed))
                        p=np.rot90(p,-k)
                        if flipped:p=np.fliplr(p)
                        values.append(p)
                return np.mean(values,axis=0,dtype=np.float32)
            mean=cached(case,f"d4_scale{scale}",d4_prediction)
            d4_means.append(mean if scale==1 else resize_p(mean,shape))
            timings[f"d4_scale{scale}"]=time.perf_counter()-t
        variants["scale_mean"]=np.mean(scale_means,axis=0,dtype=np.float32)
        variants["d4_mean"]=d4_means[1];variants["d4_multiscale_mean"]=np.mean(d4_means,axis=0,dtype=np.float32)
        for name,size,gaussian in (("gaussian512",512,True),("uniform1024",1024,False),("gaussian1024",1024,True)):
            t=time.perf_counter();variants[name]=cached(case,name,lambda:tiled(v1,rgb,size,gaussian));timings[name]=time.perf_counter()-t
        factor=512/max(shape);fitshape=(max(1,round(shape[0]*factor)),max(1,round(shape[1]*factor)))
        fitrgb=np.asarray(Image.fromarray(rgb).resize((fitshape[1],fitshape[0]),Image.Resampling.BILINEAR))
        fit=cached(case,"single_fit512_input",lambda:tiled(v1,fitrgb));variants["single_fit512"]=resize_p(fit,shape)
        for name,model in others.items():
            t=time.perf_counter();variants[name]=cached(case,name,lambda:tiled(model,rgb));timings[name]=time.perf_counter()-t
        variants["ensemble_v1_control"]=(variants["native_uniform512"]+variants["control"])/2
        variants["ensemble_all4"]=np.mean([variants[n] for n in ("native_uniform512","control","contrast","rotation")],axis=0,dtype=np.float32)
        # All predictions above read only PNGs and frozen checkpoints. Labels enter only here.
        gt=truth(case);annotation,_=load_source_annotation(raw[case]["raw_annotation_path"]);specs=path_specs(annotation,shape)
        case_records=[]
        for name,p in variants.items():
            assert p.shape==gt.shape and np.isfinite(p).all()
            mask=p>=.5;paths=path_records(mask,specs)
            curve=[dict(threshold=t,wire=metrics(*counts(p>=t,gt))) for t in read(OUT/"protocol.json")["thresholds"]]
            row=dict(case_id=case,variant=name,wire=metrics(*counts(mask,gt)),threshold_curve=curve,
                connected_paths=sum(r["connected_within_reference_tube"] for r in paths),total_paths=len(paths),elapsed_own_seconds=timings.get(name),
                changed_prediction_pixels=int((mask!=(variants["native_uniform512"]>=.5)).sum()))
            if case=="0036":
                row["local_rois"]={label:dict(median=float(np.median(p[y,x0:x1])),coverage_pixels=int((p[y,x0:x1]>=.5).sum()),length=x1-x0)
                    for label,y,x0,x1 in (("X1_left",80,620,681),("DOUT",181,605,629))}
            if name=="single_fit512":
                gtfit=np.asarray(Image.fromarray(gt.astype(np.uint8)*255).resize((fitshape[1],fitshape[0]),Image.Resampling.NEAREST))>=128
                row["resized_input_grid_wire"]=metrics(*counts(fit>=.5,gtfit));row["fit_size_hw"]=list(fitshape)
            case_records.append(row)
        for name,mask in postprocess(variants["native_uniform512"]>=.5).items():
            paths=path_records(mask,specs)
            case_records.append(dict(case_id=case,variant=name,wire=metrics(*counts(mask,gt)),threshold_curve=None,
                connected_paths=sum(r["connected_within_reference_tube"] for r in paths),total_paths=len(paths),
                changed_prediction_pixels=int((mask!=(variants["native_uniform512"]>=.5)).sum())))
        write(OUT/"inference"/case/"metrics.json",dict(records=case_records,image_only=True,GT_used_only_after_predictions=True,timings=timings))
        allrecords+=case_records
        print(json.dumps(dict(case=case,completed=i+1,total=len(args.case or val_cases()),seconds=time.perf_counter()-begin,
            dice={r["variant"]:r["wire"]["dice"] for r in case_records})),flush=True)
    groups=defaultdict(list)
    for r in allrecords:groups[r["variant"]].append(r)
    summary=[];baseline=aggregate(groups["native_uniform512"])["dice"]
    for name,records in groups.items():
        micro=aggregate(records);curve=[]
        if records[0]["threshold_curve"] is not None:
            for index,t in enumerate(read(OUT/"protocol.json")["thresholds"]):
                rows=[r["threshold_curve"][index] for r in records]
                curve.append(dict(threshold=t,micro=aggregate(rows),macro_dice=float(np.mean([r["wire"]["dice"] for r in rows]))))
        row=dict(variant=name,pages=len(records),micro=micro,macro_dice=float(np.mean([r["wire"]["dice"] for r in records])),
            delta_dice_pp=100*(micro["dice"]-baseline),connected_paths=sum(r["connected_paths"] for r in records),total_paths=sum(r["total_paths"] for r in records),
            threshold_curve=curve,diagnostic_best=max(curve,key=lambda r:r["micro"]["dice"]) if curve else None,
            adopted=False,improved_pages=sum(r["wire"]["dice"]>next(b["wire"]["dice"] for b in groups["native_uniform512"] if b["case_id"]==r["case_id"]) for r in records))
        if name=="single_fit512":row["resized_input_grid_micro"]=metrics(*(sum(r["resized_input_grid_wire"][k] for r in records) for k in ("tp","fp","fn")))
        summary.append(row)
    result=dict(status="inference_ablation_complete_no_adoption",config=CONFIG,summary=summary,records=allrecords,
        protected_files=assert_protected(),elapsed_seconds=time.perf_counter()-start,training_started=False)
    write(OUT/("inference_audit.json" if not args.case else "inference_smoke.json"),result)
    print(json.dumps(dict(summary=[{k:v for k,v in r.items() if k not in ("threshold_curve","diagnostic_best")} for r in summary],seconds=result["elapsed_seconds"])),flush=True)


if __name__=="__main__":main()
