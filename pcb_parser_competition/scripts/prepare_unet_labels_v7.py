"""Merge four locally reviewed train-only geometry corrections onto immutable v6."""
from collections import Counter
import copy
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from pcb_parser.dataset_converter import load_source_annotation
from pcb_parser.unet_data import annotation_targets,polylines
from repair_unet_0005_dp_path import read_rows,write_json,sha,usable
from repair_unet_relations_v6 import changes

BASE=ROOT/"data/segmentation_unet_v6_corrected_relations_20261015"
OUT=ROOT/"data/segmentation_unet_v7_train_review_20261015"
REPORT=ROOT/"runs/unet_v7_label_review_20261015"
PATCHES={
    "0025":dict(roi=[0,110,440,380],append={"E_6":[[[14,369],[340,369]],[[362,369],[427,369]]]},
                reason="补全KEY4水平导线两侧缺失段；接入已有E_6，保留R7支路与原pin归属。"),
    "0073":dict(roi=[450,90,705,305],replace={"E_11":[[[458,298],[619,298],[619,218]]],"E_18":[[[458,98],[699,98]]]},
                reason="扩展已有VOUT与CC2真实线路；复用E_11/E_18及现有端点，不激活重复E_8。"),
    "0088":dict(roi=[310,20,515,205],replace={
                    "E_1":[[[387,45],[491,45],[491,190]],[[433,113],[491,113]],[[475,190],[509,190]]],
                    "E_3":[[[390,113],[336,113],[336,160]],[[315,160],[360,160]]]},
                pins=[dict(component="C_3",pin="P_2",tip=[433,113],base=[432,113]),
                      dict(component="C_3",pin="P_9",tip=[390,113],base=[391,113])],
                component_bbox=dict(component="C_3",box=[391,97,432,131]),
                reason="两组源路径相对原图偏移11至12px；同步纠正C3两个tip、base和bbox。"),
    "0093":dict(roi=[105,440,185,555],add={"local_review_blue_upper_v7":[[[110,448],[179,448]]],
                  "local_review_blue_lower_v7":[[[110,546],[179,546]]]},
                reason="补充两段已核对的蓝色晶振支路自由线段，各70px；仅用于分割监督，不推断完整网络。"),
}


def reviewed_edge(edge,paths,reason):
    points=[];seen=set()
    for line in paths:
        for a,b in zip(line,line[1:]):
            assert a[0]==b[0] or a[1]==b[1]
            n=int(abs(b[0]-a[0])+abs(b[1]-a[1]))
            for k in range(n+1):
                t=k/n if n else 0
                p=(round(a[0]+t*(b[0]-a[0])),round(a[1]+t*(b[1]-a[1])))
                if p not in seen:points.append(list(p));seen.add(p)
    edge.update(keyPointPaths={"wire":paths},coordinates=points,originalCoordinatesCount=len(points),
        keyPoints={"skeleton":{"wire":points},"endPoints":{"wire":[p for line in paths for p in (line[0],line[-1])]},
                   "turnPoints":{"wire":[p for line in paths for p in line[1:-1]]},"divergePoints":{"wire":[]}},
        segmentation_review=dict(reason=reason,reviewer="agent",source="original_png_local_visual_review",full_page_certified=False,
                                 graph_associations_certified=False,stroke_width=3))


def main():
    if OUT.exists():raise ValueError("v7 dataset exists; do not overwrite a training dataset")
    rows=read_rows(BASE/"manifest.jsonl");old={r["case_id"]:copy.deepcopy(r) for r in rows}
    raw={r["case_id"]:r for r in read_rows(ROOT/"data/manifest.jsonl")}
    frozen=read_rows(BASE/"frozen_validation_manifest.jsonl")
    weights=ROOT/"runs/training/unet_wire_pin_v6_corrected_relations_20261015/best.pt"
    protected={str(p):sha(p) for p in (BASE/"manifest.jsonl",BASE/"training_manifest.jsonl",BASE/"frozen_validation_manifest.jsonl",weights,ROOT/"weights/unet_wire_pin_v1.pt")}
    for row in rows:
        for key in ("image","wire_mask","pin_heatmap","valid_mask","source_annotation_path"):
            if row.get(key):protected[row[key]]=sha(row[key])
    OUT.mkdir(parents=True);REPORT.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({"font.sans-serif":["Microsoft YaHei","SimHei","DejaVu Sans"],"axes.unicode_minus":False})
    records=[]
    for row in rows:
        case=row["case_id"]
        if case not in PATCHES:continue
        assert row["split"]=="train" and row["structurally_usable_draft"]
        patch=PATCHES[case];source_input=Path(row.get("source_annotation_path",raw[case]["raw_annotation_path"]))
        protected[str(source_input)]=sha(source_input)
        original,_=load_source_annotation(source_input);revised=copy.deepcopy(original)
        bw,bp,bt,bs=annotation_targets(original,row["width"],row["height"],3,2.)
        before=np.asarray(bw)>0
        assert np.array_equal(np.asarray(bw),np.asarray(Image.open(row["wire_mask"])))
        assert np.array_equal(np.asarray(bp),np.asarray(Image.open(row["pin_heatmap"])))
        rgb=np.asarray(Image.open(row["image"]).convert("RGB"))
        replaced=[];ink=[]
        for action in ("replace","append","add"):
            for eid,new_paths in patch.get(action,{}).items():
                if action=="add":
                    assert eid not in revised["edges"]
                    revised["edges"][eid]=dict(edgeId=eid,state="active",category="wire",isFlywire=False,connect={})
                edge=revised["edges"][eid]
                assert edge.get("state")!="deleted" and not edge.get("isFlywire")
                paths=copy.deepcopy(edge["keyPointPaths"]["wire"])+new_paths if action=="append" else new_paths
                reviewed_edge(edge,paths,patch["reason"]);replaced.append(eid)
                support=Image.new("1",(row["width"],row["height"]))
                for line in new_paths:ImageDraw.Draw(support).line([tuple(p) for p in line],fill=1,width=1)
                mask=np.asarray(support,dtype=bool)
                fraction=float((rgb[mask].min(axis=1)<224).mean())
                assert fraction==1.,(case,eid,"reviewed path includes blank pixels",fraction)
                ink.append(dict(edge_id=eid,action=action,new_centerline_pixels=int(mask.sum()),visible_ink_fraction=fraction))
        for pin in patch.get("pins",[]):
            cid,pid=pin["component"],pin["pin"]
            for target in (revised["pins"][pid],revised["components"][cid]["pins"][pid]):
                target.update(tip=pin["tip"],base=pin["base"])
            for edge in revised["edges"].values():
                if pid in edge.get("connect",{}):edge["connect"][pid]["location"]=pin["tip"]
        if patch.get("component_bbox"):
            change=patch["component_bbox"];box=dict(zip(("x1","y1","x2","y2"),change["box"]))
            revised["components"][change["component"]].update(realBBox=box,location=copy.deepcopy(box))
        wire,pin,tips,stats=annotation_targets(revised,row["width"],row["height"],3,2.)
        after=np.asarray(wire)>0;added=after&~before;removed=before&~after
        x0,y0,x1,y1=patch["roi"];allowed=np.zeros_like(before);allowed[y0:y1,x0:x1]=True
        assert not ((added|removed)&~allowed).any()
        assert not ((np.asarray(pin)!=np.asarray(bp))&~allowed).any()
        assert usable(stats) and stats["missing_physical_paths"]==0
        if case!="0088":assert np.array_equal(np.asarray(pin),np.asarray(bp)) and tips==bt
        folder=OUT/case;folder.mkdir();source=folder/"source_annotation.json"
        write_json(source,revised);write_json(folder/"source_changes.json",changes(original,revised))
        wire.save(folder/"wire_mask.png");pin.save(folder/"pin_heatmap.png");write_json(folder/"pin_points.json",tips)
        row.update(source_annotation_path=str(source.resolve()),source_annotation_sha256=sha(source),
            label_review_base_source=str(source_input),label_review_base_source_sha256=sha(source_input),
            wire_mask=str((folder/"wire_mask.png").resolve()),pin_heatmap=str((folder/"pin_heatmap.png").resolve()),
            source_changes_path=str((folder/"source_changes.json").resolve()),stats=stats,
            label_status="draft_v7_four_train_pages_locally_reviewed_not_full_page_certified",full_page_manually_approved=False,
            local_training_review=patch,structurally_usable_draft=usable(stats))
        for key in ("wire_mask","pin_heatmap"):row[key+"_sha256"]=sha(row[key])
        write_json(folder/"metadata.json",row)
        record=dict(case_id=case,split="train",added_wire_pixels=int(added.sum()),removed_wire_pixels=int(removed.sum()),
            pin_heatmap_changed_pixels=int((np.asarray(pin)!=np.asarray(bp)).sum()),pin_tips_changed=len(patch.get("pins",[])),
            roi=patch["roi"],reason=patch["reason"],geometry_evidence=ink,full_page_certified=False)
        records.append(record)
        fig,axes=plt.subplots(1,3,figsize=(13,4),dpi=145,layout="constrained")
        for ax,im,title in zip(axes,[rgb,before,after],["原图","v6标签","v7局部修订标签"]):
            ax.imshow(im[y0:y1,x0:x1],cmap="gray" if im.ndim==2 else None,interpolation="nearest");ax.set_title(title);ax.axis("off")
        fig.suptitle(f"训练页{case} · 新增{int(added.sum())} / 删除{int(removed.sum())}像素 · 完整页仍未认证")
        fig.savefig(REPORT/f"{case}_comparison.png",facecolor="white");plt.close(fig)
        print(json.dumps(record,ensure_ascii=False),flush=True)
    training=[r for r in rows if r["split"]=="train" and r["structurally_usable_draft"]]
    assert len(training)==171 and len(frozen)==28
    for name,values in [("manifest",rows),("training_manifest",training+frozen),("frozen_validation_manifest",frozen)]:
        (OUT/f"{name}.jsonl").write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in values),encoding="utf-8")
    assert len(records)==4 and all(row==old[row["case_id"]] for row in rows if row["case_id"] not in PATCHES)
    assert all(sha(p)==digest for p,digest in protected.items())
    summary=dict(status="four_train_only_local_reviews_applied_to_v7",reviewed_training_pages=records,
        train_cases=171,frozen_validation_cases=28,training_manifest_rows=199,full_page_human_certified=0,
        added_wire_pixels=sum(r["added_wire_pixels"] for r in records),removed_wire_pixels=sum(r["removed_wire_pixels"] for r in records),
        changed_pin_heatmap_pixels=sum(r["pin_heatmap_changed_pixels"] for r in records),changed_pin_tips=2,
        validation_changed=False,training_case_ids_changed=False,protected_inputs_sha256=protected,
        manifest_sha256=sha(OUT/"manifest.jsonl"),training_manifest_sha256=sha(OUT/"training_manifest.jsonl"),
        script_sha256=sha(Path(__file__)),graph_truth_certified=False)
    write_json(OUT/"summary.json",summary);write_json(REPORT/"summary.json",summary)


if __name__=="__main__":main()
