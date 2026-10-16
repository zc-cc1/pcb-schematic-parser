"""Disjoint geometric error partitions and explicit semantic-region proxies."""
from collections import defaultdict
import json
import math
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from skimage.morphology import skeletonize

from unet_audit_common import ROOT, OUT, read, write, image, truth, probability, metrics, counts, aggregate, val_cases, assert_protected
from pcb_parser.dataset_converter import load_source_annotation
from verify_unet_contrast import path_specs


def rectangle_mask(shape, bbox, margin=0):
    mask = np.zeros(shape, bool)
    if isinstance(bbox, dict) and all(k in bbox for k in ("x1", "y1", "x2", "y2")):
        values = [bbox[k] for k in ("x1", "y1", "x2", "y2")]
    else:
        return mask
    if any(not isinstance(v, (float, int)) for v in values):
        return mask
    x0, y0, x1, y1 = map(lambda v: int(round(v)), values)
    x0, x1 = sorted((x0, x1)); y0, y1 = sorted((y0, y1)); h, w = shape
    mask[max(0,y0-margin):min(h,y1+margin+1), max(0,x0-margin):min(w,x1+margin+1)] = True
    return mask


def arms(skel, x, y, radius=4):
    h, w = skel.shape
    if not (radius <= x < w-radius and radius <= y < h-radius):
        return 0
    values = np.concatenate((skel[y-radius, x-radius:x+radius], skel[y-radius:y+radius, x+radius],
                             skel[y+radius, x+radius:x-radius:-1], skel[y+radius:y-radius:-1, x-radius]))
    return int((values & ~np.roll(values, 1)).sum())


def junction_regions(gt):
    skel = skeletonize(gt)
    degree = ndimage.convolve(skel.astype(np.uint8), np.ones((3,3), np.uint8), mode="constant")-skel
    candidate = (degree >= 3) & skel
    groups, n = ndimage.label(ndimage.binary_dilation(candidate, iterations=1), np.ones((3,3)))
    three, four = np.zeros_like(gt), np.zeros_like(gt)
    points = []
    for number,box in enumerate(ndimage.find_objects(groups),1):
        if box is None:
            continue
        ys, xs = np.where(candidate[box] & (groups[box] == number)); ys += box[0].start; xs += box[1].start
        if not len(xs):
            continue
        x, y = int(round(xs.mean())), int(round(ys.mean()))
        branch_count = arms(skel, x, y)
        if branch_count < 3:
            continue
        target = three if branch_count == 3 else four
        yy, xx = np.ogrid[:gt.shape[0], :gt.shape[1]]
        target |= (xx-x)**2+(yy-y)**2 <= 6**2
        points.append(dict(point=[x,y], arms=branch_count))
    return skel, three, four, points


def component_stats(pred, gt, distance_gt):
    labels, n = ndimage.label(pred, np.ones((3,3)))
    sizes = np.bincount(labels.ravel())
    attached = set(map(int, np.unique(labels[distance_gt <= 2])))
    isolated = np.array([i for i in range(1,n+1) if i not in attached], dtype=int)
    result = dict(all_prediction_components=n, isolated_no_GT_within2_components=len(isolated),
                  isolated_no_GT_within2_area=int(sizes[isolated].sum()))
    for limit in (4,16,64):
        selected = isolated[sizes[isolated] <= limit]
        result[f"isolated_area_le_{limit}_count"] = len(selected)
        result[f"isolated_area_le_{limit}_pixels"] = int(sizes[selected].sum())
    return result


def source_path_coverage(pred, specs):
    distance = ndimage.distance_transform_edt(~pred) if pred.any() else np.full(pred.shape, np.inf)
    records = []
    for spec in specs:
        x0,y0,x1,y1 = spec["box"]
        centerline = Image.new("1", (x1-x0,y1-y0))
        ImageDraw.Draw(centerline).line([(x-x0,y-y0) for x,y in spec["points"]], fill=1, width=1)
        line = np.asarray(centerline, dtype=bool)
        covered = distance[y0:y1,x0:x1][line] <= 2
        records.append(dict(edge_id=spec["edge_id"], points=[list(p) for p in spec["points"]], length=spec["length"],
            centerline_samples=len(covered), coverage=float(covered.mean()) if len(covered) else 0,
            completely_missed=bool(not covered.any())))
    return records


def render_case(case, folder, record):
    rgb = image(case); gt = truth(case); pred = probability(case) >= .5
    error = rgb.copy()
    for selected, color in ((gt & pred, (30,175,75)), (pred & ~gt, (238,40,40)), (gt & ~pred,(35,95,238))):
        error[selected] = (.15*error[selected]+.85*np.array(color)).astype(np.uint8)
    fig, axes = plt.subplots(2,2,figsize=(15,10),dpi=160,layout="constrained")
    for ax, values, title in zip(axes.ravel(), (rgb,gt,pred,error), ("原图", "GT · 3px草稿路径描边", "原模型预测 ≥0.5", "误差：红FP / 蓝FN / 绿TP")):
        ax.imshow(values, cmap="gray" if values.ndim == 2 else None, interpolation="nearest")
        ax.set_title(title,fontsize=15); ax.axis("off")
    w = record["wire"]
    fig.suptitle(f"{case} · {record['split']} · Dice={w['dice']:.2%} / P={w['precision']:.2%} / R={w['recall']:.2%}\nTP={w['tp']} / FP={w['fp']} / FN={w['fn']} · 草稿比较不等于真实错误",fontsize=16)
    fig.savefig(folder/"overview.png",facecolor="white");plt.close(fig)
    candidates = []
    for kind, selected in (("FP",pred & ~gt),("FN",gt & ~pred)):
        labels, n = ndimage.label(selected,np.ones((3,3)))
        sizes = np.bincount(labels.ravel());sizes[0]=0
        for number in np.argsort(sizes)[-2:][::-1]:
            if sizes[number] == 0:
                continue
            ys,xs=np.where(labels==number)
            x,y=int(round(xs.mean())),int(round(ys.mean()))
            width,height=min(400,max(120,int(xs.max()-xs.min()+45))),min(280,max(100,int(ys.max()-ys.min()+45)))
            x0=max(0,min(rgb.shape[1]-width,x-width//2));y0=max(0,min(rgb.shape[0]-height,y-height//2))
            box=[x0,y0,min(rgb.shape[1],x0+width),min(rgb.shape[0],y0+height)]
            candidates.append(dict(kind=kind,component_area=int(sizes[number]),box=box))
    fig,axes=plt.subplots(len(candidates),3,figsize=(13,max(2.8,2.6*len(candidates))),dpi=160,layout="constrained",squeeze=False)
    for row,region in enumerate(candidates):
        x0,y0,x1,y1=region["box"]
        for col,(background,title) in enumerate(((rgb,"原图"),(gt,"GT"),(error,"红FP/蓝FN"))):
            axes[row,col].imshow(background[y0:y1,x0:x1],cmap="gray" if background.ndim==2 else None,interpolation="nearest")
            axes[row,col].axis("off");axes[row,col].set_title(f"{region['kind']} 区域 {region['box']} · {title}",fontsize=12)
    fig.savefig(folder/"error_rois.png",facecolor="white");plt.close(fig)
    write(folder/"roi_coordinates.json",dict(case_id=case,regions=candidates,used_for_training=False))
    return candidates


def main():
    assert_protected();start=time.perf_counter()
    baseline=read(OUT/"baseline_reproduction.json")
    raw={r["case_id"]:r for r in map(json.loads,(ROOT/"data/manifest.jsonl").read_text(encoding="utf-8").splitlines())}
    records=[];buckets=defaultdict(list);struct=defaultdict(list)
    for index,c in enumerate(val_cases()):
        gt=truth(c);pred=probability(c)>=.5
        dg=ndimage.distance_transform_edt(~gt)
        dp=ndimage.distance_transform_edt(~pred) if pred.any() else np.full(pred.shape, np.inf)
        skel,three,four,nodes=junction_regions(gt)
        widths=np.maximum(1,np.rint(2*ndimage.distance_transform_edt(gt)-1).astype(np.int16))
        _,nearest=ndimage.distance_transform_edt(~skel,return_indices=True)
        nearest_width=widths[nearest[0],nearest[1]]
        tube=dg<=2
        for name,selection in (("GTwidth_1",nearest_width==1),("GTwidth_2_3",(nearest_width>=2)&(nearest_width<=3)),("GTwidth_ge4",nearest_width>=4)):
            region=selection & tube
            buckets[name].append(metrics(*counts(pred,gt,region))|dict(region_pixels=int(region.sum())))
        buckets["outside_GT2px"].append(metrics(*counts(pred,gt,~tube))|dict(region_pixels=int((~tube).sum())))
        annotation,_=load_source_annotation(raw[c]["raw_annotation_path"])
        body=np.zeros_like(gt);text=np.zeros_like(gt)
        for component in annotation.get("components",{}).values():
            if component.get("state") != "deleted":
                body|=rectangle_mask(gt.shape,component.get("realBBox") or component.get("location"),2)
        for t in annotation.get("texts",{}).values():
            if t.get("state") != "deleted":
                text|=rectangle_mask(gt.shape,t.get("location"),2)
        occupied=np.zeros_like(gt)
        for name,mask in (("text_bbox_overlap_proxy",text),("component_bbox_overlap_proxy",body),
                          ("four_plus_arm_geometry_proxy",four),("three_arm_geometry_proxy",three),
                          ("ordinary_reference_tube",tube)):
            region=mask & ~occupied;occupied|=mask
            struct[name].append(metrics(*counts(pred,gt,region))|dict(region_pixels=int(region.sum())))
        struct["remaining_background_unknown_semantics"].append(metrics(*counts(pred,gt,~occupied))|dict(region_pixels=int((~occupied).sum())))
        border=np.zeros_like(gt);border[:32]=True;border[-32:]=True;border[:,:32]=True;border[:,-32:]=True
        buckets["image_border32"].append(metrics(*counts(pred,gt,border)))
        buckets["image_interior"].append(metrics(*counts(pred,gt,~border)))
        density=ndimage.uniform_filter(gt.astype(np.float32),size=64,mode="constant")>=.03
        buckets["dense_GT_proxy3pct64"].append(metrics(*counts(pred,gt,density)))
        buckets["sparse_GT_proxy3pct64"].append(metrics(*counts(pred,gt,~density)))
        base=next(r for r in baseline["records"] if r["case_id"]==c)
        buckets[f"source_mode_{base['source_image_mode']}"].append(base["wire"])
        specs=path_specs(annotation,gt.shape)
        paths=source_path_coverage(pred,specs)
        components=component_stats(pred,gt,dg)
        fp_far=int((pred & ~gt & (dg>2)).sum());fp_near=int((pred & ~gt & (dg<=2)).sum())
        fn_far=int((gt & ~pred & (dp>2)).sum());fn_near=int((gt & ~pred & (dp<=2)).sum())
        record=dict(case_id=c,wire=base["wire"],source_mode=base["source_image_mode"],node_proxies=nodes,
            partitions=dict(FP_near_GT2=fp_near,FP_far_GT2=fp_far,FN_near_prediction2=fn_near,FN_far_prediction2=fn_far),
            components=components,source_path_coverage=paths)
        records.append(record)
        assert fp_near+fp_far==base["wire"]["fp"] and fn_near+fn_far==base["wire"]["fn"]
        print(json.dumps(dict(case_id=c,completed=index+1,total=26,FP_far=fp_far,FN_far=fn_far)),flush=True)
    def grouped(values):
        return {name:metrics(*(sum(r[k] for r in rows) for k in ("tp","fp","fn")))|dict(pages=len(rows)) for name,rows in values.items()}
    parts={k:sum(r["partitions"][k] for r in records) for k in records[0]["partitions"]}
    total_error=sum(parts.values())
    paths=[p for r in records for p in r["source_path_coverage"]]
    length=sum(p["length"] for p in paths)
    path_summary=dict(paths=len(paths),total_reference_length=length,
        completely_missed_paths=sum(p["completely_missed"] for p in paths),
        completely_missed_length=sum(p["length"] for p in paths if p["completely_missed"]),
        completely_missed_length_fraction=sum(p["length"] for p in paths if p["completely_missed"])/length,
        coverage_le10pct_length_fraction=sum(p["length"] for p in paths if p["coverage"]<=.1)/length,
        coverage_le50pct_length_fraction=sum(p["length"] for p in paths if p["coverage"]<=.5)/length,
        scope="unique draft physical source polylines >=10px; covered if model pixel within2px; not net correctness")
    worst_val=sorted([r for r in baseline["records"] if r["included_in_26"]],key=lambda r:r["wire"]["dice"])[:20]
    worst_train=sorted([r for r in baseline["records"] if r["split"]=="train"],key=lambda r:r["wire"]["dice"])[:10]
    review=[]
    plt.rcParams.update({"font.sans-serif":["Microsoft YaHei","SimHei","DejaVu Sans"],"axes.unicode_minus":False})
    for r in worst_val+worst_train:
        folder=OUT/"review30"/r["case_id"];folder.mkdir(parents=True,exist_ok=True)
        rois=render_case(r["case_id"],folder,r)
        review.append(dict(case_id=r["case_id"],split=r["split"],wire=r["wire"],overview=str(folder/"overview.png"),
            roi_picture=str(folder/"error_rois.png"),rois=rois,visual_review_status="pending_agent_visual_inspection"))
        print(json.dumps(dict(rendered=r["case_id"],review_index=len(review),review_total=30)),flush=True)
    components={k:sum(r["components"][k] for r in records) for k in records[0]["components"]}
    result=dict(status="error_statistics_complete_manual_review_pending",width_buckets=grouped(buckets),structure_proxies=grouped(struct),
        partitions=parts,partition_error_fractions={k:v/total_error for k,v in parts.items()},components=components,
        reference_path_length=path_summary,records=records,worst20_validation=[r["case_id"] for r in worst_val],
        manual_review_sampling="20 worst unique validation pages plus 10 worst train pages; not 30 validation pages because only26 exist",
        unavailable_semantic_labels=dict(junction_dot="no dedicated certified dot annotation",crossing_connection_status="cannot infer electrical connection from raster intersection alone",
            dashed_hidden="no independent style truth",bus="no certified bus annotations",frame_table="not explicitly annotated; manual ROIs required"),
        source_creator_groups=dict(known=0,unknown=26),GTwidth_scope="3px supervision widths, not actual PNG wire widths",
        structure_priority="text boxes > component boxes > 4+arm > 3arm > GT2px tube > remaining; proxy labels, not certified semantic classes",
        protected_files=assert_protected(),elapsed_seconds=time.perf_counter()-start)
    write(OUT/"error_decomposition.json",result);write(OUT/"review30/manifest.json",dict(samples=review,independent_human_annotations=False))
    print(json.dumps(dict(partitions=parts,error_fractions=result["partition_error_fractions"],components=components,
        source_path_length=path_summary,buckets=result["width_buckets"],structures=result["structure_proxies"],worst20=result["worst20_validation"])),flush=True)


if __name__=="__main__":main()
