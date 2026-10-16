"""Fixed16 sampling comparison and a separate, train-only BN buffer diagnostic."""
from collections import defaultdict
import copy
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage
import torch

from unet_audit_common import ROOT, OUT, read, write, sha, counts, metrics, aggregate, assert_protected
from pcb_parser.unet_data import predict_tiled, annotation_targets
from pcb_parser.unet_model import WirePinUNet
from pcb_parser.unet_inference import decode_pins
from pcb_parser.dataset_converter import load_source_annotation
from verify_unet_contrast import path_specs, path_records, pin_metrics


def main():
    run=ROOT/"runs/unet_negative_sampling_pilot_v4";review=ROOT/"runs/unet_training_label_review_v4"
    cfg=read(run/"configuration.json");completed=read(run/"summary.json")
    assert completed["status"]=="paired_fixed16_training_complete_external_evaluation_not_started"
    frozen=read(run/"protected_inputs.json")|read(run/"additional_source_attestation.json")
    assert all(sha(p)==v for p,v in frozen.items());assert_protected()
    models={};weights={};initial=cfg["initial_state_sha256"]
    torch.set_num_threads(4);start=time.perf_counter()
    names=list(cfg["arms"])+[name+"_BN_reestimated" for name in cfg["arms"]]
    bn_summary=read(run/"BN_diagnostic_summary.json")
    assert all(r["learned_weights_unchanged"] and r["gradient_updates"]==0 and not r["calibration_or_external26_images_read"] for r in bn_summary["records"])
    for name in names:
        path=run/name/"last.pt";checkpoint=torch.load(path,map_location="cpu",weights_only=True)
        assert checkpoint["epoch"]==16 and checkpoint["initial_state_sha256"]==initial
        model=WirePinUNet(checkpoint["base_channels"]).to("cuda:0");model.load_state_dict(checkpoint["state_dict"],strict=True);model.eval()
        models[name]=model;weights[name]=sha(path)
    rows=[json.loads(s) for s in (review/"training_manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    external=[json.loads(s) for s in (review/"frozen_validation_manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows)==103 and len(external)==26
    for r in external:r["evaluation_split"]="frozen_external26"
    for r in rows:r["evaluation_split"]="train93" if r["split"]=="train" else "internal_calibration10"
    raw={r["case_id"]:r for r in map(json.loads,(ROOT/"data/manifest.jsonl").read_text(encoding="utf-8").splitlines())}
    patches=read(ROOT/"configs/unet_training_geometry_candidates_v4.json")["corrections"]
    base_patches=read(ROOT/"configs/unet_retraining_v2_label_corrections.json")["corrections"]
    negatives=read(review/"confirmed_negative_regions.json")["regions"]
    records=[];thresholds=read(OUT/"protocol.json")["thresholds"]
    for index,row in enumerate(rows+external):
        case=row["case_id"];rgb=np.asarray(Image.open(row["image"]).convert("RGB"));predictions={}
        for name,model in models.items():
            folder=run/"evaluation"/case/name;folder.mkdir(parents=True,exist_ok=True)
            path=folder/"probabilities.npz";meta_path=folder/"prediction_meta.json"
            if path.exists():
                meta=read(meta_path);assert meta["checkpoint_sha256"]==weights[name] and meta["image_sha256"]==sha(row["image"])
                with np.load(path,allow_pickle=False) as v:p=np.stack([v["wire"],v["pin"]])
            else:
                with torch.inference_mode():p=predict_tiled(model,rgb,"cuda:0",512,384)
                np.savez_compressed(path,wire=p[0],pin=p[1]);write(meta_path,dict(image_only=True,
                    checkpoint_sha256=weights[name],image_sha256=sha(row["image"]),threshold=.5,tile_size=512,stride=384,resize=False))
            predictions[name]=p
        # Ground truth and annotation geometry enter only after all image-only predictions.
        gt=np.asarray(Image.open(row["wire_mask"]))>=128
        annotation,_=load_source_annotation(raw[case]["raw_annotation_path"])
        if case in base_patches:
            annotation=copy.deepcopy(annotation);correction=base_patches[case]
            for eid,paths in correction["replace_edges"].items():annotation["edges"][eid]["keyPointPaths"]={"wire":paths}
            for eid in correction["activate_edges"]:annotation["edges"][eid]["state"]="active"
            for eid,paths in correction["add_edges"].items():annotation["edges"][eid]=dict(state="active",category="wire",isFlywire=False,keyPointPaths={"wire":paths})
        if case in patches:
            correction=patches[case];annotation=copy.deepcopy(annotation)
            for eid,paths in correction["replace_edges"].items():annotation["edges"][eid]["keyPointPaths"]={"wire":paths}
            for eid,indices in correction["drop_path_indices"].items():
                paths=annotation["edges"][eid]["keyPointPaths"]["wire"]
                annotation["edges"][eid]["keyPointPaths"]["wire"]=[p for i,p in enumerate(paths) if i not in set(indices)]
        reference_wire,*_=annotation_targets(annotation,row["width"],row["height"],3,2.)
        assert np.array_equal(np.asarray(reference_wire)>0,gt),f"{case}: path reference and evaluated supervision differ"
        specs=path_specs(annotation,gt.shape)
        point_path=review/"candidates"/case/"pin_points_candidate.json" if case in patches else PathFromMask(row["wire_mask"])/"pin_points.json"
        tips=read(point_path)
        for name,p in predictions.items():
            mask=p[0]>=.5;paths=path_records(mask,specs)
            pins=decode_pins(p[1],threshold=.8,kernel=5,refinement="quadratic",nms_radius=0.,limit=500)
            dg=ndimage.distance_transform_edt(~gt)
            dmask=ndimage.distance_transform_edt(~mask) if mask.any() else np.full(mask.shape,np.inf)
            row_metrics=dict(case_id=case,arm=name,evaluation_split=row["evaluation_split"],wire=metrics(*counts(mask,gt)),
                pin=pin_metrics(pins,tips),connected_paths=sum(v["connected_within_reference_tube"] for v in paths),total_paths=len(paths),
                FP_far_GT2=int((mask & ~gt & (dg>2)).sum()),
                relaxed2=dict(pred_matched=int((mask & (dg<=2)).sum()),pred_pixels=int(mask.sum()),
                    gt_matched=int((gt & (dmask<=2)).sum()),gt_pixels=int(gt.sum())),
                negative_regions=[dict(id=v["id"],box=v["box"],kind=v["kind"],FP=int(mask[v["box"][1]:v["box"][3],v["box"][0]:v["box"][2]].sum())) for v in negatives if v["case_id"]==case],
                threshold_curve=[dict(threshold=t,wire=metrics(*counts(p[0]>=t,gt))) for t in thresholds] if row["evaluation_split"]!="train93" else None)
            records.append(row_metrics);write(run/"evaluation"/case/name/"metrics.json",row_metrics)
            Image.fromarray(mask.astype(np.uint8)*255).save(run/"evaluation"/case/name/"wire_mask.png")
        print(json.dumps(dict(case_id=case,completed=index+1,total=129,split=row["evaluation_split"])),flush=True)
    summaries=[]
    for split in ("train93","internal_calibration10","frozen_external26"):
        for name in models:
            selected=[r for r in records if r["arm"]==name and r["evaluation_split"]==split]
            rel={k:sum(r["relaxed2"][k] for r in selected) for k in selected[0]["relaxed2"]}
            precision=rel["pred_matched"]/rel["pred_pixels"] if rel["pred_pixels"] else 0
            recall=rel["gt_matched"]/rel["gt_pixels"] if rel["gt_pixels"] else 0
            curve=[]
            if selected[0]["threshold_curve"] is not None:
                for i,t in enumerate(thresholds):curve.append(dict(threshold=t,micro=aggregate([r["threshold_curve"][i] for r in selected])))
            summaries.append(dict(split=split,arm=name,pages=len(selected),wire=aggregate(selected),pin=aggregate(selected,"pin"),
                macro_dice=float(np.mean([r["wire"]["dice"] for r in selected])),connected_paths=sum(r["connected_paths"] for r in selected),total_paths=sum(r["total_paths"] for r in selected),
                FP_far_GT2=sum(r["FP_far_GT2"] for r in selected),relaxed2_precision=precision,relaxed2_recall=recall,
                relaxed2_f1=2*precision*recall/(precision+recall) if precision+recall else 0,
                negative_ROI_FP=sum(v["FP"] for r in selected for v in r["negative_regions"]),
                diagnostic_threshold_curve=curve,diagnostic_best=max(curve,key=lambda x:x["micro"]["dice"]) if curve else None))
    comparisons=[];bn_comparisons=[]
    rng=np.random.default_rng(20261014)
    for split in ("train93","internal_calibration10","frozen_external26"):
        ids=sorted(r["case_id"] for r in records if r["evaluation_split"]==split and r["arm"]=="control")
        pair={name:{r["case_id"]:r for r in records if r["arm"]==name and r["evaluation_split"]==split} for name in models}
        draw=rng.integers(0,len(ids),size=(10000,len(ids)))
        sampled={}
        for name in models:
            array=np.array([[pair[name][c]["wire"][k] for k in ("tp","fp","fn")] for c in ids]);totals=array[draw].sum(axis=1)
            sampled[name]=2*totals[:,0]/np.maximum(1,2*totals[:,0]+totals[:,1]+totals[:,2])
        contrasts=[("sampling_raw","control","negative_focus"),
            ("control_BN_only","control","control_BN_reestimated"),
            ("focus_BN_only","negative_focus","negative_focus_BN_reestimated"),
            ("sampling_with_common_BN_method","control_BN_reestimated","negative_focus_BN_reestimated")]
        for contrast,before,after in contrasts:
            control=next(r for r in summaries if r["split"]==split and r["arm"]==before)
            treated=next(r for r in summaries if r["split"]==split and r["arm"]==after)
            value=dict(split=split,contrast=contrast,before=before,after=after,
                delta_dice_pp=100*(treated["wire"]["dice"]-control["wire"]["dice"]),
                delta_recall_pp=100*(treated["wire"]["recall"]-control["wire"]["recall"]),delta_precision_pp=100*(treated["wire"]["precision"]-control["wire"]["precision"]),
                delta_connected_paths=treated["connected_paths"]-control["connected_paths"],negative_ROI_FP_delta=treated["negative_ROI_FP"]-control["negative_ROI_FP"],
                paired_page_bootstrap95_delta_pp=np.quantile(100*(sampled[after]-sampled[before]),[.025,.975]).tolist())
            (comparisons if contrast=="sampling_raw" else bn_comparisons).append(value)
    internal=next(r for r in comparisons if r["split"]=="internal_calibration10")
    train_delta=next(r for r in comparisons if r["split"]=="train93")
    success=internal["delta_dice_pp"]>=1 and internal["delta_recall_pp"]>=0 and internal["delta_connected_paths"]>=0 and train_delta["negative_ROI_FP_delta"]<0
    result=dict(status="fixed16_paired_pilot_evaluation_complete",summaries=summaries,comparisons=comparisons,bn_comparisons=bn_comparisons,records=records,
        preregistered_internal_success=success,decision="candidate_sampling_requires_longer_confirmatory_training" if success else "reject_current_sampling_recipe",
        external26_used_for_selection=False,checkpoint_selection="last16 fixed before training",production_adopted=False,
        shared_initial_state_sha256=initial,weights_sha256=weights,partial_agent_draft=True,independent_engineering_groups=False,
        independent_test=False,bootstrap_scope="paired page resampling; unknown engineering dependence not adjusted",elapsed_seconds=time.perf_counter()-start,
        calibration_selection="10 unreviewed singleton image groups; excludes the 10 worst training pages used in label review; not an unbiased random test",
        BN_diagnostic=bn_summary,
        protected_original_files=assert_protected())
    probes=[]
    for case,y,x0,x1,label in (("0036",80,620,681,"X1_fixed61"),("0036",181,605,629,"DOUT_fixed24")):
        for name in models:
            with np.load(run/"evaluation"/case/name/"probabilities.npz",allow_pickle=False) as source:
                values=source["wire"][y,x0:x1]
            probes.append(dict(case_id=case,arm=name,probe=label,y=y,x_interval_half_open=[x0,x1],
                pixels=len(values),detected_pixels=int((values>=.5).sum()),mean_probability=float(values.mean()),median_probability=float(np.median(values))))
    result["fixed_prior_audit_probes"]=probes
    write(run/"evaluation.json",result);render_and_report(run,cfg,result)
    assert all(sha(p)==v for p,v in frozen.items());assert_protected()
    print(json.dumps(dict(summaries=[{k:v for k,v in r.items() if k not in ("diagnostic_threshold_curve","diagnostic_best")} for r in summaries],comparisons=comparisons,success=success,decision=result["decision"])))


def PathFromMask(path):
    from pathlib import Path
    return Path(path).parent


def render_and_report(run,cfg,result):
    plt.rcParams.update({"font.sans-serif":["Microsoft YaHei","SimHei","DejaVu Sans"],"axes.unicode_minus":False})
    fig,axes=plt.subplots(1,2,figsize=(12,4),dpi=170,layout="constrained")
    for name in cfg["arms"]:
        history=read(run/name/"history.json")
        axes[0].plot([r["epoch"] for r in history],[r["train_loss"] for r in history],label=name)
        axes[1].plot([r["epoch"] for r in history],[100*r["internal_calibration_tile_metrics"]["wire_pixel_dice"] for r in history],label=name)
    axes[0].set(title="训练loss（采样分布不同）",xlabel="epoch",ylabel="loss")
    axes[1].set(title="同10页内部校准tile Dice@0.5",xlabel="epoch",ylabel="Dice（%）")
    for ax in axes:ax.legend();ax.grid(alpha=.2)
    fig.suptitle("相同随机初始化 · 固定16轮 · 只改变非导线focus采样25%")
    fig.savefig(run/"curves.png",facecolor="white");plt.close(fig)
    for case,box in (("0281",[0,0,310,336]),("0170",[306,136,683,293]),("0036",[536,31,788,119])):
        rgb=np.asarray(Image.open(ROOT/f"data/standardized/{case}/{case}.png").convert("RGB"));x0,y0,x1,y1=box
        fig,axes=plt.subplots(2,3,figsize=(13,7),dpi=170,layout="constrained")
        axes[0,0].imshow(rgb[y0:y1,x0:x1]);axes[0,0].set_title("原图")
        gt=np.asarray(Image.open(ROOT/f"data/segmentation_unet_v1/{case}/wire_mask.png"))>0
        shown_gt=rgb.copy();shown_gt[gt]=(.25*shown_gt[gt]+.75*np.array([20,180,95])).astype(np.uint8)
        axes[1,0].imshow(shown_gt[y0:y1,x0:x1]);axes[1,0].set_title("现有监督标签（整页未认证）")
        names=list(cfg["arms"])+[name+"_BN_reestimated" for name in cfg["arms"]]
        for ax,name in zip((axes[0,1],axes[0,2],axes[1,1],axes[1,2]),names):
            mask=np.asarray(Image.open(run/"evaluation"/case/name/"wire_mask.png"))>0
            over=rgb.copy()
            for region,color in ((mask & gt,[20,180,95]),(mask & ~gt,[25,140,235]),(~mask & gt,[230,55,60])):
                over[region]=(.25*over[region]+.75*np.array(color)).astype(np.uint8)
            record=next(r for r in result["records"] if r["arm"]==name and r["case_id"]==case)
            ax.imshow(over[y0:y1,x0:x1]);ax.set_title(f"{name}\n整页P={record['wire']['precision']:.1%} / R={record['wire']['recall']:.1%}")
        for ax in axes.flat:ax.axis("off")
        fig.suptitle(f"{case} · 阈值0.5 · 绿=TP / 蓝=FP / 红=FN（相对现有标签）")
        fig.savefig(run/f"comparison_{case}.png",facecolor="white");plt.close(fig)
    from report_unet_negative_sampling_v4 import write_report
    write_report(run,cfg,result)


if __name__=="__main__":main()
