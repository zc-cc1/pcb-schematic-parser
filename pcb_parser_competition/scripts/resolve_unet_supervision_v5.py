"""Resolve the 13 remaining records for segmentation supervision, retaining graph uncertainty."""
from collections import Counter, defaultdict
import copy
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from pcb_parser.dataset_converter import load_source_annotation
from pcb_parser.unet_data import annotation_targets,SegmentationTiles,tile_starts,point
from pcb_parser.unet_retraining import CachedTrainingTiles
from repair_unet_0005_dp_path import sha,usable,write_json,read_rows

BASE=ROOT/"data/segmentation_unet_v4_path_recovery_20261015"
OUTPUT=ROOT/"data/segmentation_unet_v5_reviewed_supervision_20261015"
REPORT=ROOT/"runs/unet_supervision_review_v5"
RADIUS=7
SYMBOLS={
    ("0024","E_41"):"symbol_outline",
    ("0203","E_92"):"embedded_image",
    ("0205","E_86"):"embedded_image",
}


def main():
    base_rows=read_rows(BASE/"manifest.jsonl")
    old={r["case_id"]:r for r in base_rows}
    rows=copy.deepcopy(base_rows)
    by_case={r["case_id"]:r for r in rows}
    raw={r["case_id"]:r for r in read_rows(ROOT/"data/manifest.jsonl")}
    holds_path=ROOT/"runs/unet_path_recovery_v4/holds.json"
    holds=json.loads(holds_path.read_text(encoding="utf-8"))
    assert len(holds)==13
    groups=defaultdict(list)
    for hold in holds:groups[hold["case_id"]].append(hold)
    assert len(groups)==9
    protected={str(path):sha(path) for path in (BASE/"manifest.jsonl",BASE/"training_manifest.jsonl",BASE/"frozen_validation_manifest.jsonl",holds_path,ROOT/"configs/unet_wire_pin.json",ROOT/"configs/unet_label_reviews.json",ROOT/"weights/unet_wire_pin_v1.pt")}
    for r in base_rows:
        for key in ("wire_mask","pin_heatmap"):protected[r[key]]=sha(r[key])
        if r.get("source_annotation_path"):protected[r["source_annotation_path"]]=sha(r["source_annotation_path"])
    OUTPUT.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
    review_dir=REPORT/"review";review_dir.mkdir(exist_ok=True)
    decisions=[];page_records=[];all_pins=[]
    plt.rcParams.update({"font.sans-serif":["Microsoft YaHei","SimHei","DejaVu Sans"],"axes.unicode_minus":False})
    for case,case_holds in groups.items():
        previous=old[case];row=by_case[case]
        source_input=Path(previous.get("source_annotation_path",raw[case]["raw_annotation_path"]))
        protected[str(source_input)]=sha(source_input)
        annotation,_=load_source_annotation(source_input)
        revised=copy.deepcopy(annotation)
        before_wire,before_pin,before_tips,before_stats=annotation_targets(annotation,row["width"],row["height"],3,2)
        assert np.array_equal(np.asarray(before_wire),np.asarray(Image.open(previous["wire_mask"])))
        assert np.array_equal(np.asarray(before_pin),np.asarray(Image.open(previous["pin_heatmap"])))
        rgb=np.asarray(Image.open(row["image"]).convert("RGB"))
        valid=np.full((row["height"],row["width"]),255,np.uint8)
        ignored_points=set();uncertain_pins=set()
        for hold in case_holds:
            eid=hold["edge_id"];edge=revised["edges"][eid]
            assert edge["keyPointPaths"]=={"wire":[]} and edge.get("state")!="deleted"
            assert edge.get("category","wire")=="wire" and edge.get("isFlywire") is False
            classification=SYMBOLS.get((case,eid),"unresolved_association")
            decision="ignore" if classification=="unresolved_association" else "exclude"
            coords=[point(p) for p in edge["coordinates"]]
            assert all(p is not None for p in coords)
            xs,ys=zip(*coords)
            roi=[max(0,int(min(xs))-35),max(0,int(min(ys))-35),min(row["width"],int(max(xs))+36),min(row["height"],int(max(ys))+36)]
            reason=hold["review_reason_zh"]
            review=dict(decision=decision,classification=classification,reason=reason,
                        local_image_review=True,reviewer="agent",independent_annotation=False,
                        source_graph_association_verified=False,source_graph_relations_preserved=True,
                        roi=roi,review_artifact=str((review_dir/f"{case}_{eid}.png").resolve()))
            edge["wire_supervision"]=review
            edge_points=set(coords if decision=="ignore" else [])
            for pid,connection in edge.get("connect",{}).items():
                cid=connection.get("cpntId")
                local=annotation.get("components",{}).get(cid,{}).get("pins",{}).get(pid,{})
                pin={**annotation.get("pins",{}).get(pid,{}),**local}
                tip=point(pin.get("tip"))
                if tip is not None:
                    edge_points.add(tip);uncertain_pins.add((cid,pid));all_pins.append((case,cid,pid))
            for x,y in edge_points:
                assert 0<=x<row["width"] and 0<=y<row["height"]
                x,y=round(x),round(y)
                valid[max(0,y-RADIUS):min(row["height"],y+RADIUS+1),max(0,x-RADIUS):min(row["width"],x+RADIUS+1)]=0
            ignored_points.update(edge_points)
            record=dict(case_id=case,edge_id=eid,decision=decision,classification=classification,
                        reason=reason,roi=roi,uncertain_endpoints=[list(p) for p in sorted(edge_points)],
                        source_pointer=f"/edges/{eid}",review_image=review["review_artifact"],
                        pixel_supervision_resolved=True,graph_relation_review_pending=decision=="ignore",
                        source_graph_relations_preserved=True,full_page_certified=False)
            decisions.append(record)
        wire,pin,tips,stats=annotation_targets(revised,row["width"],row["height"],3,2)
        assert np.array_equal(np.asarray(wire),np.asarray(before_wire))
        assert np.array_equal(np.asarray(pin),np.asarray(before_pin)) and tips==before_tips
        assert stats["missing_physical_paths"]==0
        assert stats["excluded_missing_path_records"]==before_stats["missing_physical_paths"]==len(case_holds)
        restored=copy.deepcopy(revised)
        for hold in case_holds:del restored["edges"][hold["edge_id"]]["wire_supervision"]
        assert restored==annotation,"Only supervision review fields may change"
        folder=OUTPUT/case;folder.mkdir(exist_ok=True)
        source_output=folder/"source_annotation.json"
        write_json(source_output,revised)
        wire.save(folder/"wire_mask.png");pin.save(folder/"pin_heatmap.png")
        write_json(folder/"pin_points.json",tips)
        Image.fromarray(valid).save(folder/"valid_mask.png")
        row.update(source_annotation_path=str(source_output.resolve()),source_annotation_sha256=sha(source_output),
                   supervision_source_annotation_path=str(source_input),supervision_source_annotation_sha256=sha(source_input),
                   original_source_annotation_path=raw[case]["raw_annotation_path"],original_source_annotation_sha256=sha(raw[case]["raw_annotation_path"]),
                   wire_mask=str((folder/"wire_mask.png").resolve()),pin_heatmap=str((folder/"pin_heatmap.png").resolve()),
                   valid_mask=str((folder/"valid_mask.png").resolve()),stats=stats,
                   structurally_usable_draft=usable(stats),
                   label_status="draft_explicit_nonwire_exclusions_and_unknown_endpoint_ignore_not_full_page_certified",
                   supervision_ignored_pixels=int((valid==0).sum()),
                   supervised_wire_positive_pixels=int(((np.asarray(wire)>0)&(valid>0)).sum()),
                   uncertain_pin_targets=[dict(component_source_id=cid,pin_source_id=pid) for cid,pid in sorted(uncertain_pins)],
                   path_recovery_holds=[],supervision_decisions=[r for r in decisions if r["case_id"]==case])
        write_json(folder/"metadata.json",row)
        # Both existing training loaders must suppress the same uncertain pixels.
        for x,y in ignored_points:
            x,y=round(x),round(y)
            starts=[(xx,yy) for yy in tile_starts(row["height"],512,384) for xx in tile_starts(row["width"],512,384)]
            idx=next(i for i,(xx,yy) in enumerate(starts) if xx<=x<xx+512 and yy<=y<yy+512)
            ox,oy=starts[idx]
            direct=SegmentationTiles([row],512,384)[idx][2]
            cached=CachedTrainingTiles([row],512,384)[idx][2]
            assert np.array_equal(direct.numpy(),cached.numpy())
            assert direct[y-oy,x-ox]==0
        for record in row["supervision_decisions"]:
            eid=record["edge_id"];x0,y0,x1,y1=record["roi"];extent=[x0-.5,x1-.5,y1-.5,y0-.5]
            fig,axes=plt.subplots(1,3,figsize=(12,4.5),dpi=145,layout="constrained")
            for ax in axes[:2]:ax.imshow(rgb[y0:y1,x0:x1],extent=extent,interpolation="nearest")
            source_coords=np.asarray(annotation["edges"][eid]["coordinates"])
            axes[0].plot(source_coords[:,0],source_coords[:,1],color="#e47626",linestyle=":" if record["decision"]=="ignore" else "-",linewidth=1)
            axes[0].set_title("原图与被排除的源记录（橙）")
            for x,y in record["uncertain_endpoints"]:
                axes[1].add_patch(Rectangle((x-RADIUS-.5,y-RADIUS-.5),RADIUS*2+1,RADIUS*2+1,
                                           facecolor="#858585",alpha=.55,edgecolor="none"))
            axes[1].set_title("灰块：不确定端点不参与损失")
            display=np.asarray(wire,dtype=float)/255
            display[valid==0]=.45
            axes[2].imshow(display[y0:y1,x0:x1],extent=extent,cmap="gray",vmin=0,vmax=1,interpolation="nearest")
            axes[2].set_title("白=导线；灰=忽略；黑=背景")
            for ax in axes:
                ax.set_xlim(x0-.5,x1-.5);ax.set_ylim(y1-.5,y0-.5)
                ax.set_xlabel("原图 x");ax.set_ylabel("原图 y")
            fig.suptitle(f"{case}/{eid} · {'非导线轮廓排除' if record['decision']=='exclude' else '未决关联忽略，不制造物理路线'}")
            fig.savefig(record["review_image"],facecolor="white");plt.close(fig)
        page_records.append(dict(case_id=case,split=row["split"],decisions=len(case_holds),
                                 supervised_missing_paths=stats["missing_physical_paths"],
                                 raw_missing_records_retained=len(case_holds),
                                 ignored_pixels=row["supervision_ignored_pixels"],
                                 uncertain_pin_targets=len(uncertain_pins),
                                 structurally_usable_draft=row["structurally_usable_draft"],
                                 wire_and_pin_arrays_unchanged=True,both_training_loaders_verified=True))
        print(json.dumps(dict(case_id=case,decisions=len(case_holds),usable=row["structurally_usable_draft"],ignored_pixels=row["supervision_ignored_pixels"])),flush=True)
    assert len(decisions)==13 and Counter(r["decision"] for r in decisions)=={"exclude":3,"ignore":10}
    assert all(by_case[r["case_id"]]==r for r in base_rows if r["case_id"] not in groups)
    manifest=OUTPUT/"manifest.jsonl"
    manifest.write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in rows),encoding="utf-8")
    frozen=read_rows(BASE/"frozen_validation_manifest.jsonl")
    train=[r for r in rows if r["split"]=="train" and r["structurally_usable_draft"]]
    train_manifest=OUTPUT/"training_manifest.jsonl"
    train_manifest.write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in train+frozen),encoding="utf-8")
    (OUTPUT/"frozen_validation_manifest.jsonl").write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in frozen),encoding="utf-8")
    assert len(train)==171 and len(frozen)==28
    assert sum(r["stats"]["missing_physical_paths"] for r in rows)==0
    assert sum(r["stats"].get("ignored_source_associations",0) for r in rows)==10
    assert sum(r["stats"].get("excluded_nonwire_records",0) for r in rows)==3
    assert all(sha(p)==digest for p,digest in protected.items())
    graph_queue=[r for r in decisions if r["graph_relation_review_pending"]]
    summary=dict(status="13_remaining_records_resolved_for_segmentation_supervision_graph_review_retained",
                 prior_geometric_path_repairs=679,reviewed_records=13,nonwire_records_excluded=3,
                 unresolved_associations_ignored=10,source_graph_relations_pending=10,
                 raw_missing_path_records_retained=13,supervised_missing_paths=0,
                 new_wire_positive_pixels=0,new_pin_heatmap_pixels=0,
                 previous_train_candidates=166,train_candidates=len(train),new_train_cases=[r["case_id"] for r in train if not old[r["case_id"]]["structurally_usable_draft"]],
                 usable_by_split=dict(Counter(r["split"] for r in rows if r["structurally_usable_draft"])),
                 frozen_validation_cases=28,training_manifest_rows=len(train)+len(frozen),
                 reviewed_pages=9,ignored_pixels=sum(r["ignored_pixels"] for r in page_records),
                 uncertain_pin_targets=len(set(all_pins)),validity_radius=RADIUS,
                 pin_heatmaps_and_wire_masks_unchanged_from_v4=True,source_network_memberships_unchanged=True,
                 entire_pages_certified=False,training_started=False,default_weights_changed=False,
                 manifest=str(manifest.resolve()),training_manifest=str(train_manifest.resolve()),
                 manifest_sha256=sha(manifest),training_manifest_sha256=sha(train_manifest),
                 source_files_sha256=protected,script_sha256=sha(Path(__file__)),
                 modules_sha256={name:sha(ROOT/"src/pcb_parser"/name) for name in ("unet_data.py","unet_retraining.py","unet_supervision.py")})
    for path,value in ((OUTPUT/"summary.json",summary),(REPORT/"summary.json",summary),
                       (REPORT/"decisions.json",decisions),(OUTPUT/"supervision_decisions.json",decisions),
                       (REPORT/"pages.json",page_records),(OUTPUT/"graph_review_queue.json",graph_queue)):
        write_json(path,value)
    print(json.dumps({k:v for k,v in summary.items() if k not in {"source_files_sha256","modules_sha256"}},ensure_ascii=False))


if __name__=="__main__":
    main()
