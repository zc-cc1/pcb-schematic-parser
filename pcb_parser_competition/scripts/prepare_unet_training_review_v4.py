"""Training-only local label candidates, verified negative ROIs and split preview.

Produces review artifacts and a sampling dry run, never trains or changes a source.
"""
from collections import defaultdict
import copy
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree

from unet_audit_common import ROOT, OUT, read, write, sha, image, truth, probability, counts, metrics, assert_protected
from pcb_parser.dataset_converter import load_source_annotation
from pcb_parser.unet_data import annotation_targets, point, polylines, tile_starts

REVIEW = ROOT/"runs/unet_training_label_review_v4"
PATCH = ROOT/"configs/unet_training_geometry_candidates_v4.json"
NEGATIVES = {
    "0281": [("IC_top_border", [35,50,240,58]), ("IC_bottom_border", [35,318,240,326])],
    "0253": [("IC_top_border", [590,169,704,176]), ("IC_right_border", [712,195,720,260]),
             ("audio_source_box_top", [25,168,150,176]), ("audio_source_box_bottom", [25,524,150,532])],
    "0073": [("IC_left_border", [259,75,267,195]), ("IC_top_border", [285,15,425,23])],
    "0079": [("IC_top_border", [220,51,335,59]), ("IC_bottom_border", [295,423,335,431])],
    "0129": [("drawing_table_border", [400,786,1050,798]), ("table_logo_text", [1160,1010,1430,1090])],
}


def sparse_centerline(paths):
    points=[]
    for path in polylines(paths):
        for (x0,y0),(x1,y1) in zip(path,path[1:]):
            # Bresenham via the same label renderer, restricted to a segment bbox.
            x0,y0,x1,y1=map(round,(x0,y0,x1,y1));left,top=min(x0,x1),min(y0,y1)
            canvas=Image.new("1",(abs(x1-x0)+1,abs(y1-y0)+1))
            ImageDraw.Draw(canvas).line([(x0-left,y0-top),(x1-left,y1-top)],fill=1,width=1)
            yy,xx=np.where(np.asarray(canvas,dtype=bool));points.extend(zip(xx+left,yy+top))
    return np.unique(np.asarray(points,dtype=np.int32).reshape(-1,2),axis=0)


def source_consistency(train_ids, raw):
    records=[]
    for case in sorted(train_ids):
        annotation,_=load_source_annotation(raw[case]["raw_annotation_path"])
        for eid,edge in annotation.get("edges",{}).items():
            if edge.get("state")=="deleted" or edge.get("category","wire")!="wire" or str(edge.get("isFlywire")).lower()=="true":continue
            paths=edge.get("keyPointPaths",{});paths=paths.get("wire",[]) if isinstance(paths,dict) else paths
            center=sparse_centerline(paths)
            source=edge.get("keyPoints",{}).get("skeleton",{}).get("wire",[])
            source=np.unique(np.asarray([point(p) for p in source if point(p) is not None],dtype=float).reshape(-1,2),axis=0)
            row=dict(case_id=case,edge_id=eid,path_samples=len(center),source_skeleton_samples=len(source),
                path_gt2px_from_source=None,source_gt2px_from_path=None)
            if len(center) and len(source):
                d=cKDTree(source).query(center,k=1)[0];d2=cKDTree(center).query(source,k=1)[0]
                row.update(path_gt2px_from_source=int((d>2).sum()),source_gt2px_from_path=int((d2>2).sum()),
                    path_far_fraction=float((d>2).mean()),source_far_fraction=float((d2>2).mean()))
            records.append(row)
    compared=[r for r in records if r["path_gt2px_from_source"] is not None]
    summary=dict(scope="103 training pages only; two geometric fields in same annotation, NOT independent labels or image truth",
        active_physical_edges=len(records),compared_edges=len(compared),path_samples=sum(r["path_samples"] for r in compared),
        path_samples_gt2px_from_source=sum(r["path_gt2px_from_source"] for r in compared),
        edges_any_path_disagreement=sum(r["path_gt2px_from_source"]>0 for r in compared),
        edges_path_far_fraction_ge50pct=sum(r["path_far_fraction"]>=.5 for r in compared),
        automatic_label_repair_permitted=False)
    summary["path_far_fraction"]=summary["path_samples_gt2px_from_source"]/max(1,summary["path_samples"])
    write(REVIEW/"source_geometry_consistency.json",dict(summary=summary,records=records))
    return summary


def overlay(rgb,mask,color):
    out=rgb.copy();out[mask]=(.25*out[mask]+.75*np.array(color)).astype(np.uint8);return out


def render_candidate(case,before,after,config,old_tips,new_tips):
    rgb=image(case);x0,y0,x1,y1=config["roi"]
    # Keep the original background identical across the before/after panels.
    delta=rgb.copy();delta[before & ~after]=[235,85,40];delta[after & ~before]=[35,140,245]
    fig,axes=plt.subplots(2,2,figsize=(12,8),dpi=180,layout="constrained")
    pictures=(rgb,overlay(rgb,before,(235,70,50)),overlay(rgb,after,(30,150,85)),delta)
    titles=("原图（手工核对依据）","现有训练标签 · 红","局部候选标签 · 绿","差分：橙=删除 / 蓝=新增")
    for ax,picture,title in zip(axes.flat,pictures,titles):
        ax.imshow(picture[y0:y1,x0:x1],interpolation="nearest");ax.set_title(title);ax.axis("off")
    fig.suptitle(f"训练页 {case} · 仅局部几何候选 · 非独立重标/非全页认证")
    folder=REVIEW/"candidates"/case;folder.mkdir(parents=True,exist_ok=True)
    fig.savefig(folder/"comparison.png",facecolor="white");plt.close(fig)
    if config.get("pin_tip_overrides"):
        selected={(v["component_source_id"],v["pin_source_id"]) for v in config["pin_tip_overrides"]}
        fig,axes=plt.subplots(1,2,figsize=(12,5),dpi=180,layout="constrained")
        for ax,tips,title in zip(axes,(old_tips,new_tips),("原tip（蓝圈）","候选tip（蓝圈；仅位置修正）")):
            ax.imshow(rgb[y0:y1,x0:x1],interpolation="nearest")
            for p in tips:
                if (p["component_source_id"],p["pin_source_id"]) not in selected:continue
                x,y=p["point"];ax.scatter([x-x0],[y-y0],s=70,facecolors="none",edgecolors="#0575e6",linewidths=1.5)
                ax.text(x-x0+3,y-y0-3,f"{p['pin_source_id']} {x:g},{y:g}",fontsize=8,color="#0575e6")
            ax.set_title(title);ax.axis("off")
        fig.suptitle(f"{case} · 端点位置与wire候选对齐；不认证原图对象归属")
        fig.savefig(folder/"pin_review.png",facecolor="white");plt.close(fig)


def main():
    assert_protected();REVIEW.mkdir(parents=True,exist_ok=True)
    protocol=read(OUT/"protocol.json");train_ids=set(protocol["train_cases"]);val_ids=set(protocol["original_validation_cases"])
    patch=read(PATCH);base_rows=[json.loads(s) for s in (ROOT/patch["base_manifest"]).read_text(encoding="utf-8").splitlines()]
    base={r["case_id"]:r for r in base_rows};raw={r["case_id"]:r for r in map(json.loads,(ROOT/"data/manifest.jsonl").read_text(encoding="utf-8").splitlines())}
    assert set(patch["corrections"])<=train_ids and set(NEGATIVES)<=train_ids and not (train_ids & val_ids)
    frozen={str(ROOT/patch["base_manifest"]):sha(ROOT/patch["base_manifest"]),str(PATCH):sha(PATCH)}
    for c in train_ids:
        assert base[c]["split"]=="train"
        for key in ("image","wire_mask","pin_heatmap"):
            frozen[base[c][key]]=sha(base[c][key])
    plt.rcParams.update({"font.sans-serif":["Microsoft YaHei","SimHei","DejaVu Sans"],"axes.unicode_minus":False})
    consistency=source_consistency(train_ids,raw)
    candidates=[]
    for case,correction in patch["corrections"].items():
        row=base[case];annotation,_=load_source_annotation(raw[case]["raw_annotation_path"])
        frozen[raw[case]["raw_annotation_path"]]=sha(raw[case]["raw_annotation_path"])
        original_wire,original_heat,old_tips,_=annotation_targets(annotation,row["width"],row["height"],3,2.)
        before=np.asarray(Image.open(row["wire_mask"]))>0
        assert np.array_equal(np.asarray(original_wire)>0,before),"Unexpected correction already present in base"
        changed=copy.deepcopy(annotation)
        for eid,paths in correction["replace_edges"].items():
            edge=changed["edges"][eid];assert edge.get("state")!="deleted" and not edge.get("isFlywire")
            edge["keyPointPaths"]={"wire":paths}
        for eid,indices in correction["drop_path_indices"].items():
            paths=changed["edges"][eid]["keyPointPaths"]["wire"]
            assert len(set(indices))==len(indices) and all(0<=i<len(paths) for i in indices)
            changed["edges"][eid]["keyPointPaths"]["wire"]=[p for i,p in enumerate(paths) if i not in set(indices)]
        for p in correction.get("pin_tip_overrides",[]):
            cid,pid=p["component_source_id"],p["pin_source_id"]
            assert cid in changed["components"] and pid in changed["components"][cid]["pins"]
            changed["components"][cid]["pins"][pid]["tip"]=p["point"]
            if pid in changed.get("pins",{}):changed["pins"][pid]["tip"]=p["point"]
        wire,heat,new_tips,_=annotation_targets(changed,row["width"],row["height"],3,2.);after=np.asarray(wire)>0
        assert np.array_equal(np.asarray(original_heat),np.asarray(Image.open(row["pin_heatmap"])))
        x0,y0,x1,y1=correction["roi"];region=np.zeros_like(before);region[y0:y1,x0:x1]=True
        assert not ((before!=after)&~region).any(),f"{case}: candidate changes outside reviewed ROI"
        assert not ((np.asarray(original_heat)!=np.asarray(heat))&~region).any(),f"{case}: pin change outside reviewed ROI"
        folder=REVIEW/"candidates"/case;folder.mkdir(parents=True,exist_ok=True)
        wire.save(folder/"wire_mask_candidate.png")
        heat.save(folder/"pin_heatmap_candidate.png");write(folder/"pin_points_candidate.json",new_tips)
        render_candidate(case,before,after,correction,old_tips,new_tips)
        record=dict(case_id=case,source_split="train",ROI=correction["roi"],reason=correction["reason"],pending=correction["pending"],
            added_pixels=int((after & ~before).sum()),removed_pixels=int((before & ~after).sum()),
            base_wire_sha256=sha(row["wire_mask"]),candidate_wire_sha256=sha(folder/"wire_mask_candidate.png"),
            candidate_pin_points_changed=len(correction.get("pin_tip_overrides",[])),source_pin_heatmap_unchanged=True,
            candidate_pin_heatmap_sha256=sha(folder/"pin_heatmap_candidate.png"),
            independent_annotation=False,full_page_certified=False,automatic_joint_training_ready=False,
            local_pin_position_conflicts_repaired_in_candidate=True,graph_associations_certified=False,
            original_paths={eid:annotation["edges"][eid]["keyPointPaths"] for eid in set(correction["replace_edges"])|set(correction["drop_path_indices"])},
            candidate_paths={eid:changed["edges"][eid]["keyPointPaths"] for eid in set(correction["replace_edges"])|set(correction["drop_path_indices"])})
        write(folder/"review_record.json",record);candidates.append(record)
    negatives=[]
    for case,regions in NEGATIVES.items():
        gt=np.asarray(Image.open(base[case]["wire_mask"]))>0;rgb=image(case);p=probability(case)
        for kind,box in regions:
            x0,y0,x1,y1=box;assert 0<=x0<x1<=gt.shape[1] and 0<=y0<y1<=gt.shape[0]
            assert not gt[y0:y1,x0:x1].any(),f"{case} {kind}: GT wire inside proposed negative"
            number=len(negatives);fp=int((p[y0:y1,x0:x1]>=.5).sum())
            record=dict(id=f"negative_{number:02d}",case_id=case,split="train",kind=kind,box=box,area=(x1-x0)*(y1-y0),
                GT_positive_pixels=0,baseline_false_positive_pixels=fp,hard_negative=fp>0,
                verification="agent inspected original full page; rectangle restricted to body/frame/text, no wire shafts/intersections",
                box_only_is_negative=True,whole_512_crop_is_not_pure_background=True,source_image_sha256=sha(base[case]["image"]),
                source_wire_sha256=sha(base[case]["wire_mask"]))
            negatives.append(record)
        folder=REVIEW/"negatives"/case;folder.mkdir(parents=True,exist_ok=True)
        fig,axes=plt.subplots(len(regions),3,figsize=(12,max(3,len(regions)*2)),dpi=170,layout="constrained",squeeze=False)
        for i,(kind,box) in enumerate(regions):
            x0,y0,x1,y1=box;pad=20;left=max(0,x0-pad);top=max(0,y0-pad);right=min(gt.shape[1],x1+pad);bottom=min(gt.shape[0],y1+pad)
            error=overlay(rgb,p>=.5,(235,55,50))
            for j,(picture,title) in enumerate(((rgb,"原图"),(gt,"训练GT"),(error,"原模型预测 · 红"))):
                axes[i,j].imshow(picture[top:bottom,left:right],cmap="gray" if picture.ndim==2 else None,interpolation="nearest")
                from matplotlib.patches import Rectangle
                axes[i,j].add_patch(Rectangle((x0-left,y0-top),x1-x0,y1-y0,fill=False,edgecolor="#06a9db",linewidth=1.3))
                axes[i,j].set_title(f"{kind} · {title}",fontsize=10);axes[i,j].axis("off")
        fig.suptitle(f"{case} · 青框只标已核对非导线；周围真实导线保留")
        fig.savefig(folder/"review.png",facecolor="white");plt.close(fig)
    write(REVIEW/"confirmed_negative_regions.json",dict(regions=negatives,selection_uses_training_predictions_only=True,
        independent_human_certification=False,total_baseline_FP=sum(r["baseline_false_positive_pixels"] for r in negatives)))
    # A future training-internal split. No v1 checkpoint is used for initialization.
    reviewed_ids=set(read(OUT/"followup_protocol.json")["label_review_train_cases"])
    stats=read(OUT/"baseline_reproduction.json")["records"];groups=defaultdict(list)
    for r in stats:
        if r["split"]=="train":groups[r["decoded_pixel_sha256"]].append(r["case_id"])
    eligible=sorted(k for k,ids in groups.items() if len(ids)==1 and not(set(ids)&reviewed_ids))
    rng=np.random.default_rng(20261014);rng.shuffle(eligible)
    calibration=set(c for k in eligible[:10] for c in groups[k]);future_train=train_ids-calibration
    assert len(calibration)==10 and len(future_train)==93 and not(calibration & reviewed_ids)
    assert all(not(set(ids)&calibration) or set(ids)<=calibration for ids in groups.values())
    split=dict(status="preview_only_not_applied_to_original_manifest",seed=20261014,training_cases=sorted(future_train),
        calibration_cases=sorted(calibration),frozen_external_validation_cases=protocol["validation_unique_cases"],
        grouping="exact decoded image identity; engineering IDs unavailable",initialization="same random seed initialization from scratch for both arms; v1 has seen all103",
        fixed_epochs=16,checkpoint="last16 only; no external26 checkpoint/threshold/loss selection",draft_candidate_training_gate="resolve pin geometry and local uncertainty before applying candidates")
    write(REVIEW/"future_internal_split.json",split)
    derived=[]
    for c in sorted(train_ids):
        r=copy.deepcopy(base[c]);r["source_split"]="train";r["split"]="train" if c in future_train else "calibration"
        r["label_status"]="agent_local_geometry_draft_not_certified"
        if c in patch["corrections"]:
            folder=REVIEW/"candidates"/c
            r["wire_mask"]=str(folder/"wire_mask_candidate.png");r["pin_heatmap"]=str(folder/"pin_heatmap_candidate.png")
            r["wire_mask_sha256"]=sha(r["wire_mask"]);r["pin_heatmap_sha256"]=sha(r["pin_heatmap"])
        derived.append(r)
    (REVIEW/"training_manifest.jsonl").write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in derived),encoding="utf-8")
    (REVIEW/"frozen_validation_manifest.jsonl").write_text("".join(json.dumps(base[c],ensure_ascii=False)+"\n" for c in protocol["validation_unique_cases"]),encoding="utf-8")
    tiles=[dict(case_id=c,x=x,y=y,draw_kind="base") for c in sorted(future_train) for y in tile_starts(base[c]["height"],512,384) for x in tile_starts(base[c]["width"],512,384)]
    focus=[]
    for r in negatives:
        if not r["hard_negative"]:continue
        c=r["case_id"];assert c in future_train;x0,y0,x1,y1=r["box"]
        x=max(0,min(max(0,base[c]["width"]-512),(x0+x1)//2-256));y=max(0,min(max(0,base[c]["height"]-512),(y0+y1)//2-256))
        focus.append(dict(case_id=c,x=x,y=y,draw_kind="confirmed_negative_focus",negative_id=r["id"]))
    control=copy.deepcopy(tiles);rng.shuffle(control);treatment=copy.deepcopy(control)
    replace_indices=sorted(rng.choice(len(tiles),size=round(.25*len(tiles)),replace=False).tolist())
    for i in replace_indices:treatment[i]=copy.deepcopy(focus[int(rng.integers(len(focus)))])
    assert len(control)==len(treatment) and all(r["case_id"] in future_train for r in treatment)
    write(REVIEW/"sampling_dry_run.json",dict(status="schedule_only_no_model_training",epoch=1,seed=20261014,
        control=control,treatment=treatment,focus_pool=focus,changed_sampling_slots=len(replace_indices),
        slots_per_arm=len(tiles),focus_slot_fraction=len(replace_indices)/len(tiles),
        both_arms_same_images_labels_loss_norm_learning_rate_epochs=True,whole_crop_targets_are_not_forced_background=True))
    assert all(sha(p)==digest for p,digest in frozen.items());assert_protected()
    summary=dict(status="local_label_candidates_and_negative_sampling_preview_complete_no_training",candidates=candidates,
        existing_v2_corrected_training_cases=["0073","0088"],new_training_pages_with_candidates=len(candidates),
        negatives=len(negatives),hard_negatives=sum(r["hard_negative"] for r in negatives),baseline_FP_inside_negative_ROIs=sum(r["baseline_false_positive_pixels"] for r in negatives),
        source_geometry_consistency=consistency,training_preview_cases=len(future_train),internal_calibration_preview_cases=len(calibration),
        dry_run_slots_per_arm=len(tiles),dry_run_focus_slots=len(replace_indices),
        validation_labels_changed=False,source_pin_labels_changed=False,candidate_pin_points_changed=sum(r["candidate_pin_points_changed"] for r in candidates),default_model_changed=False,training_started=False,
        independent_second_annotations=0,full_page_certifications=0,protected_original_files=assert_protected(),protected_base_sources=len(frozen))
    write(REVIEW/"summary.json",summary);write(REVIEW/"protected_inputs.json",frozen)
    write_report(summary)
    print(json.dumps({k:v for k,v in summary.items() if k!="candidates"},ensure_ascii=False))


def write_report(s):
    absolute=lambda p:str(p).replace("\\","/")
    text=["# 训练标注核对与负样本准备 v4", "", "本轮只修改派生候选文件。原始标注、v2训练标签、验证集和默认模型均未改。没有启动训练，也没有生成独立第二人的标注。", "",
        f"新增4张训练页局部修正候选；另复核已有0073/0088修正。已核对{s['negatives']}处非导线区域，{s['hard_negatives']}处是原模型误检，合计{s['baseline_FP_inside_negative_ROIs']}个FP像素。这些是训练页中的明确器件框/图框/文字，不能据此预计验证Dice涨幅。", "",
        "| 训练页 | 新增像素 | 删除像素 | 待解决 | 对照 |", "|---|---:|---:|---|---|"]
    for r in s["candidates"]:
        text.append(f"| {r['case_id']} | {r['added_pixels']} | {r['removed_pixels']} | {r['pending']} | [原图/旧GT/候选/差分]({absolute(REVIEW/'candidates'/r['case_id']/'comparison.png')}) |")
    cs=s["source_geometry_consistency"]
    text += ["",f"源几何内部一致性：103张训练页，比较{cs['compared_edges']}条同时有路径和骨架的物理edge，路径采样点中{cs['path_far_fraction']:.2%}距源骨架>2px；{cs['edges_path_far_fraction_ge50pct']}条有至少一半路径点不受该edge骨架支持。这不是独立GT：0254/0088的路径和骨架可能一起偏离原图，合法的端点延伸也会导致内部不一致，不能自动用其中一份替换另一份。", "",
        f"局部候选保留3px监督描边，未根据预测补线；0254和0265中相关的{s['candidate_pin_points_changed']}个tip另按原图修正位置并生成独立对照图，未修改原pin标签。0129其他漏标接地路线、0215其他标注及0265对象归属仍需核对，未宣布全页正确。只有显式启用agent draft的受控pilot可读取此派生manifest；自动生产替换仍禁用。", "",
        f"已准备新的训练内部划分预览：{s['training_preview_cases']}训练页+{s['internal_calibration_preview_cases']}校准页。仅来自原103训练页；原26页继续作为冻结外部报告集。两个训练重复图组保持同侧，工程级来源未知。后续模型必须从相同随机初始化重新训练，不能用已看过全部103页的v1权重初始化后将校准页称为独立。", "",
        f"采样干跑：两臂各{s['dry_run_slots_per_arm']}个512裁块，处理臂替换{s['dry_run_focus_slots']}个位置（约25%），围绕已核对的误检区域采样。只改变抽样分布，未把整块强制标成背景；裁块里真实wire/pin仍使用原标签。", "",
        "当前标注约定：只记录原图存在的物理导线路径；器件框、图框、表格、文字、内部符号不是wire；有意未连接杆、箭头杆和pin/wire分界仍须按原始语义核对。junction dot不扩成整圆，crossing不自动判接点。使用固定3px名义描边，保留与实际PNG线宽的区别。不能从错误component bbox自动创建负类。", "",
        f"非导线ROI清单：[JSON]({absolute(REVIEW/'confirmed_negative_regions.json')})；采样清单：[干跑]({absolute(REVIEW/'sampling_dry_run.json')})；内部划分：[预览]({absolute(REVIEW/'future_internal_split.json')})；所有候选有新增/删除像素与原路径证据。", "",
        "复现：", "```powershell", "Set-Location -LiteralPath 'C:\\Users\\Pc\\Desktop\\vision\\pcb_parser_competition'", "& .\\.venv-gpu\\Scripts\\python.exe -X utf8 scripts\\prepare_unet_training_review_v4.py", "```", "",
        "相关pin位置修正只解决本次局部像素监督冲突；原graph对象归属未认证。配对pilot使用同一份派生标签、随机初始化和固定16轮，唯一处理变量为难负样本采样，不能与原60轮模型直接当成单变量比较。暂无双标注不能被代理记录冒充。"]
    (REVIEW/"report.md").write_text("\n".join(text)+"\n",encoding="utf-8")


if __name__=="__main__":main()
