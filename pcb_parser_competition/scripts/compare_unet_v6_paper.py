"""Read-only image inference and exploratory evaluation of the completed v6 model."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from pcb_parser.dataset_converter import load_source_annotation
from pcb_parser.unet_data import predict_tiled,polylines
from pcb_parser.unet_model import WirePinUNet
from verify_unet_contrast import path_specs,path_records

RUN=ROOT/"runs/training/unet_wire_pin_v6_corrected_relations_20261015"
OUT=ROOT/"runs/unet_v6_paper_comparison_20261015"
MANIFEST=ROOT/"data/segmentation_unet_v6_corrected_relations_20261015/training_manifest.jsonl"
THRESHOLDS=[i/100 for i in range(10,91,5)]


def read(path):return json.loads(Path(path).read_text(encoding="utf-8-sig"))
def write(path,data):Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def metrics(tp,fp,fn):
    return dict(tp=int(tp),fp=int(fp),fn=int(fn),precision=float(tp/max(1,tp+fp)),recall=float(tp/max(1,tp+fn)),
                dice=float(2*tp/max(1,2*tp+fp+fn)),iou=float(tp/max(1,tp+fp+fn)))
def totals(records):return metrics(*(sum(r[k] for r in records) for k in ("tp","fp","fn")))


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter();torch.set_num_threads(4)
    weight=RUN/"best.pt";summary=read(RUN/"summary.json");history=read(RUN/"history.json")
    assert sha(weight)==summary["checkpoint_sha256"] and sha(MANIFEST)==summary["manifest_sha256"]
    protected={str(p):sha(p) for p in (weight,RUN/"last.pt",MANIFEST,ROOT/"weights/unet_wire_pin_v1.pt")}
    rows=[json.loads(s) for s in MANIFEST.read_text(encoding="utf-8").splitlines()]
    rows=[r for r in rows if r["split"]=="val"]
    split=read(ROOT/"runs/unet_inference_v2/split.json")
    unique=set(split["development"]+split["acceptance_holdout"])
    assert len(rows)==28 and len(unique)==26
    raw={r["case_id"]:r for r in map(json.loads,(ROOT/"data/manifest.jsonl").read_text(encoding="utf-8").splitlines())}
    ckpt=torch.load(weight,map_location="cpu",weights_only=True)
    assert ckpt["epoch"]==60
    model=WirePinUNet(ckpt["base_channels"]);model.load_state_dict(ckpt["state_dict"]);model.eval()
    for r in rows:
        for key in ("image","wire_mask","pin_heatmap"):protected[r[key]]=sha(r[key])
    write(OUT/"protocol.json",dict(checkpoint=str(weight),checkpoint_sha256=sha(weight),manifest_sha256=sha(MANIFEST),
          full_page_cases=[r["case_id"] for r in rows],comparison_unique_cases=sorted(unique),thresholds=THRESHOLDS,
          device="cpu",dtype="float32",image_input_only=True,overlap_probability_fusion="mean",
          labels_enter_scoring_only=True,threshold_sweep="exploratory_validation_not_independent_test",
          parameters_or_BN_updated=False,default_checkpoint_changed=False,protected_sha256=protected))
    records=[];sweeps=defaultdict(list)
    for index,row in enumerate(rows,1):
        case=row["case_id"];folder=OUT/case;folder.mkdir(exist_ok=True)
        cache=folder/"probabilities.npz";meta=folder/"prediction_meta.json"
        if cache.exists() and meta.exists():
            m=read(meta);assert m["checkpoint_sha256"]==sha(weight) and m["image_sha256"]==sha(row["image"])
            with np.load(cache,allow_pickle=False) as d:prob=d["wire"]
        else:
            rgb=np.asarray(Image.open(row["image"]).convert("RGB"))
            prediction=predict_tiled(model,rgb,"cpu",512,384)
            prob=prediction[0]
            np.savez_compressed(cache,wire=prob)
            write(meta,dict(checkpoint_sha256=sha(weight),image_sha256=sha(row["image"]),input="RGB_only",tile_size=512,stride=384,dtype="float32"))
        truth=np.asarray(Image.open(row["wire_mask"]))>0
        valid=np.ones_like(truth) if not row.get("valid_mask") else np.asarray(Image.open(row["valid_mask"]))>0
        sweep=[]
        for t in THRESHOLDS:
            mask=prob>=t
            m=metrics(int((mask&truth&valid).sum()),int((mask&~truth&valid).sum()),int((~mask&truth&valid).sum()))
            sweep.append(dict(threshold=t,**m));sweeps[t].append(dict(case_id=case,**m))
        mask=prob>=.5;fp=mask&~truth&valid;fn=truth&~mask&valid
        dist_truth=ndimage.distance_transform_edt(~truth)
        dist_pred=ndimage.distance_transform_edt(~mask)
        annotation,_=load_source_annotation(raw[case]["raw_annotation_path"])
        paths=path_records(mask,path_specs(annotation,truth.shape))
        center=Image.new("1",(row["width"],row["height"]));pen=ImageDraw.Draw(center)
        for edge in annotation["edges"].values():
            if edge.get("state")=="deleted" or edge.get("category","wire")!="wire" or str(edge.get("isFlywire")).lower()=="true":continue
            for line in polylines(edge.get("keyPointPaths",{}).get("wire",[])):pen.line(line,fill=1,width=1)
        center=np.asarray(center,dtype=bool)&valid
        values=next(v for v in sweep if v["threshold"]==.5)
        record=dict(case_id=case,width=row["width"],height=row["height"],unique_comparison=case in unique,
                    wire=values,fp_within_2px=int((fp&(dist_truth<=2)).sum()),fp_more_than_2px=int((fp&(dist_truth>2)).sum()),
                    fn_within_2px=int((fn&(dist_pred<=2)).sum()),fn_more_than_2px=int((fn&(dist_pred>2)).sum()),
                    centerline_pixels=int(center.sum()),centerline_covered_within_2px=int((center&(dist_pred<=2)).sum()),
                    connected_paths=sum(p["connected_within_reference_tube"] for p in paths),total_paths=len(paths))
        records.append(record);write(folder/"metrics.json",record);write(folder/"threshold_sweep.json",sweep)
        Image.fromarray(mask.astype(np.uint8)*255).save(folder/"wire_mask.png")
        print(json.dumps(dict(phase="v6_full_page_cpu",case_id=case,pages=index,total=28,iou=values["iou"],elapsed_seconds=round(time.perf_counter()-started,1))),flush=True)
    curves={}
    for group in ("all28","unique26"):
        selected=[r for r in records if group=="all28" or r["unique_comparison"]]
        curve=[]
        for t in THRESHOLDS:
            values=[r for r in sweeps[t] if group=="all28" or r["case_id"] in unique]
            curve.append(dict(threshold=t,**totals(values)))
        base=next(r for r in curve if r["threshold"]==.5)
        best=max(curve,key=lambda r:r["iou"])
        curves[group]=dict(pages=len(selected),at_05=base,exploratory_best=best,threshold_curve=curve,
            connected_paths=sum(r["connected_paths"] for r in selected),total_paths=sum(r["total_paths"] for r in selected),
            fp_within_2px=sum(r["fp_within_2px"] for r in selected),fp_more_than_2px=sum(r["fp_more_than_2px"] for r in selected),
            fn_within_2px=sum(r["fn_within_2px"] for r in selected),fn_more_than_2px=sum(r["fn_more_than_2px"] for r in selected),
            centerline_pixels=sum(r["centerline_pixels"] for r in selected),centerline_covered_within_2px=sum(r["centerline_covered_within_2px"] for r in selected),
            macro_iou=float(np.mean([r["wire"]["iou"] for r in selected])))
    old=read(ROOT/"runs/unet_contrast_verification_v1/evaluation.json")
    baseline=next(r for r in old["groups"]["all"] if r["gamma"]==1.)
    assert {r["case_id"] for r in old["cases"] if r["gamma"]==1.}==unique
    old_wire=metrics(*(baseline["wire_micro"][k] for k in ("tp","fp","fn")))
    assert old_wire["tp"]+old_wire["fn"]==curves["unique26"]["at_05"]["tp"]+curves["unique26"]["at_05"]["fn"]
    result=dict(status="completed_read_only_diagnostic",checkpoint_epoch=60,training_tile_metrics=history[-1]["val_wire_pixels"],
                train_tile_metrics=history[-1]["train_wire_pixels"],groups=curves,baseline_v1_full_page_unique26=old_wire,
                baseline_v1_connected_paths=baseline["connected_paths"],baseline_v1_total_paths=baseline["paths"],
                per_page=records,learned_parameters_unchanged=True,BN_buffers_unchanged=True,
                independent_test=False,threshold_adopted=False,official_score=None,
                elapsed_seconds=time.perf_counter()-started)
    assert all(sha(p)==digest for p,digest in protected.items())
    write(OUT/"evaluation.json",result)
    render(rows,records,history,result)
    print(json.dumps({k:v for k,v in result.items() if k not in ["per_page","groups"]}|dict(groups={k:{n:v for n,v in g.items() if n!="threshold_curve"} for k,g in curves.items()}),ensure_ascii=False),flush=True)


def render(rows,records,history,result):
    plt.rcParams.update({"font.sans-serif":["Microsoft YaHei","SimHei","DejaVu Sans"],"axes.unicode_minus":False})
    fig,axes=plt.subplots(1,3,figsize=(15,4.3),dpi=150,layout="constrained")
    epochs=[r["epoch"] for r in history]
    axes[0].plot(epochs,[r["train_wire_pixels"]["wire_pixel_iou"]*100 for r in history],label="训练切块")
    axes[0].plot(epochs,[r["val_wire_pixels"]["wire_pixel_iou"]*100 for r in history],label="验证切块")
    axes[0].set_title("60轮IoU轨迹");axes[0].set_xlabel("轮数");axes[0].set_ylabel("IoU %");axes[0].legend()
    axes[1].plot(epochs,[r["train_loss"] for r in history],label="训练")
    axes[1].plot(epochs,[r["val_loss"] for r in history],label="验证")
    axes[1].set_title("损失与验证波动");axes[1].set_xlabel("轮数");axes[1].legend()
    curve=result["groups"]["unique26"]["threshold_curve"]
    axes[2].plot([r["threshold"] for r in curve],[r["iou"]*100 for r in curve],marker="o")
    axes[2].axvline(.5,color="gray",linestyle="--");axes[2].set_title("26页整图阈值扫描（探索）");axes[2].set_xlabel("阈值");axes[2].set_ylabel("IoU %")
    for ax in axes:ax.grid(alpha=.2)
    fig.savefig(OUT/"learning_and_thresholds.png",facecolor="white");plt.close(fig)
    selected=sorted((r for r in records if r["unique_comparison"] and r["wire"]["tp"]+r["wire"]["fn"]>500),key=lambda r:r["wire"]["iou"])[:3]
    by_case={r["case_id"]:r for r in rows}
    fig,axes=plt.subplots(3,3,figsize=(13,12),dpi=140,layout="constrained")
    for i,r in enumerate(selected):
        row=by_case[r["case_id"]];rgb=np.asarray(Image.open(row["image"]).convert("RGB"))
        truth=np.asarray(Image.open(row["wire_mask"]))>0
        pred=np.asarray(Image.open(OUT/r["case_id"]/"wire_mask.png"))>0
        error=pred^truth
        density=ndimage.uniform_filter(error.astype(float),size=128)
        y,x=map(int,np.unravel_index(np.argmax(density),density.shape))
        x0=max(0,min(row["width"]-256,x-128));y0=max(0,min(row["height"]-256,y-128));x1=min(row["width"],x0+256);y1=min(row["height"],y0+256)
        panel=rgb.copy()
        for area,color in [(truth&pred,(0,170,70)),(pred&~truth,(240,40,40)),(truth&~pred,(0,100,240))]:
            panel[area]=(.2*panel[area]+.8*np.array(color)).astype(np.uint8)
        for ax,im,title in zip(axes[i],[rgb,truth,panel],["原图","JSON导线标签","预测：绿=正确 红=多出 蓝=漏检"]):
            ax.imshow(im[y0:y1,x0:x1],cmap="gray",interpolation="nearest");ax.axis("off");ax.set_title(title)
        axes[i,0].set_ylabel(r["case_id"])
        axes[i,1].set_title(f"{r['case_id']} · 整页IoU {r['wire']['iou']:.1%}\nJSON导线标签")
        r["review_crop"]=[x0,y0,x1,y1]
    fig.savefig(OUT/"error_examples.png",facecolor="white");plt.close(fig)
    write(OUT/"review_examples.json",selected)


if __name__=="__main__":main()
