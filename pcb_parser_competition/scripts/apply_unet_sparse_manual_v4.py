"""Apply two image-verified sparse-route repairs and three linked tip corrections."""
from collections import Counter
import copy
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pcb_parser.unet_data import annotation_targets, SegmentationTiles, tile_starts
from repair_unet_0005_dp_path import sha, usable, write_json, read_rows

OUTPUT = ROOT / "data/segmentation_unet_v4_path_recovery_20261015"
REPORT = ROOT / "runs/unet_path_recovery_v4"
PATCHES = [
    dict(case_id="0102",edge_id="E_222",path=[[360,643],[351,643],[351,649]],roi=[325,625,380,675],
         pins=[dict(component="C_42",pin="P_30",old_tip=[346.,649.],tip=[351,649],base=[351,650])],
         reason="EN has a visible L-shaped red lead toward the +5V symbol; source power-tip x346 is off the actual x351 lead."),
    dict(case_id="0180",edge_id="E_30",path=[[540,450],[540,458]],roi=[518,438,563,476],
         pins=[dict(component="C_1",pin="P_95",old_tip=[537.,450.],tip=[540,450],base=[550,450]),
               dict(component="C_28",pin="P_67",old_tip=[540.,467.],tip=[540,458],base=[540,459])],
         component_bbox=dict(component="C_28",bbox={"x1":529,"y1":459,"x2":548,"y2":470}),
         reason="The MCU GND lead connects vertically at x540 from y450 to the first ground bar at y459; old MCU tip is 3px left and old ground tip lies below the bar."),
]


def raster_path(path, size):
    image=Image.new("1",size)
    ImageDraw.Draw(image).line([tuple(p) for p in path],fill=1,width=1)
    ys,xs=np.where(np.asarray(image,dtype=bool))
    return image, [[int(x),int(y)] for x,y in zip(xs,ys)]


def main():
    summary=json.loads((OUTPUT/"summary.json").read_text(encoding="utf-8"))
    if summary.get("manual_sparse_repairs_applied"):
        assert all(sha(p)==v for p,v in summary["source_hashes"].items())
        assert sha(OUTPUT/"manifest.jsonl")==summary["manifest_sha256"]
        print(json.dumps(dict(status="manual_sparse_repairs_already_applied",recovered_total=summary["recovered_total"])))
        return
    assert summary["recovered_in_this_run"]==676 and summary["held_missing_edges"]==15
    assert all(sha(p)==v for p,v in summary["source_hashes"].items())
    rows=read_rows(OUTPUT/"manifest.jsonl")
    by_case={r["case_id"]:r for r in rows}
    repairs=json.loads((REPORT/"repairs.json").read_text(encoding="utf-8"))
    holds=json.loads((REPORT/"holds.json").read_text(encoding="utf-8"))
    pages=json.loads((REPORT/"pages.json").read_text(encoding="utf-8"))
    manual=[]
    staged=[]
    raw_rows={r["case_id"]:r for r in read_rows(ROOT/"data/manifest.jsonl")}
    plt.rcParams.update({"font.sans-serif":["Microsoft YaHei","SimHei","DejaVu Sans"],"axes.unicode_minus":False})
    for patch in PATCHES:
        case,eid=patch["case_id"],patch["edge_id"]
        row=by_case[case]
        source_input=Path(row.get("source_annotation_path",raw_rows[case]["raw_annotation_path"]))
        from pcb_parser.dataset_converter import load_source_annotation
        original,_=load_source_annotation(source_input)
        assert original["edges"][eid]["keyPointPaths"]=={"wire":[]}
        assert any(h["case_id"]==case and h["edge_id"]==eid for h in holds)
        rgb=np.asarray(Image.open(row["image"]).convert("RGB"))
        center,coords=raster_path(patch["path"],(row["width"],row["height"]))
        support=np.asarray(center,dtype=bool)
        assert np.all(rgb[support].min(axis=1)<224), "Manual route must lie on visible ink"
        before_wire,before_pin,before_tips,before_stats=annotation_targets(original,row["width"],row["height"],3,2)
        assert np.array_equal(np.asarray(before_wire),np.asarray(Image.open(row["wire_mask"])))
        assert np.array_equal(np.asarray(before_pin),np.asarray(Image.open(row["pin_heatmap"])))
        revised=copy.deepcopy(original)
        edge=revised["edges"][eid]
        edge["keyPointPaths"]={"wire":[patch["path"]]}
        edge["coordinates"]=coords
        edge["keyPoints"]={"skeleton":{"wire":coords},"endPoints":{"wire":[patch["path"][0],patch["path"][-1]]},
                           "divergePoints":{"wire":[]},"turnPoints":{"wire":patch["path"][1:-1]}}
        for change in patch["pins"]:
            cid,pid=change["component"],change["pin"]
            assert revised["pins"][pid]["tip"]==change["old_tip"]
            assert revised["components"][cid]["pins"][pid]["tip"]==change["old_tip"]
            for pin in (revised["pins"][pid],revised["components"][cid]["pins"][pid]):
                pin["tip"]=change["tip"]
                pin["base"]=change["base"]
            for other in revised["edges"].values():
                if pid in other.get("connect",{}):
                    other["connect"][pid]["location"]=change["tip"]
        if patch.get("component_bbox"):
            bbox=patch["component_bbox"]
            revised["components"][bbox["component"]]["realBBox"]=bbox["bbox"]
        wire,pin,tips,stats=annotation_targets(revised,row["width"],row["height"],3,2)
        before=np.asarray(before_wire)>0;after=np.asarray(wire)>0
        added=after&~before
        assert not (before&~after).any()
        assert stats["missing_physical_paths"]==before_stats["missing_physical_paths"]-1
        assert not added.any() or distance_transform_edt(~support)[added].max()<=2
        expected_changes={(p["component"],p["pin"]) for p in patch["pins"]}
        old_tip_map={(p["component_source_id"],p["pin_source_id"]):p["point"] for p in before_tips}
        new_tip_map={(p["component_source_id"],p["pin_source_id"]):p["point"] for p in tips}
        assert old_tip_map.keys()==new_tip_map.keys()
        assert {key for key in old_tip_map if old_tip_map[key]!=new_tip_map[key]}==expected_changes
        allowed_pin=np.zeros_like(before)
        for p in patch["pins"]:
            for x,y in (p["old_tip"],p["tip"]):
                x,y=map(int,(x,y));allowed_pin[max(0,y-6):y+7,max(0,x-6):x+7]=True
        pin_diff=np.asarray(before_pin)!=np.asarray(pin)
        assert not (pin_diff&~allowed_pin).any()
        folder=OUTPUT/case
        folder.mkdir(exist_ok=True)
        source=folder/"source_annotation.json"
        previous_added=np.asarray(Image.open(folder/"added_pixels.png"))>0 if (folder/"added_pixels.png").exists() else np.zeros_like(before)
        holds=[h for h in holds if (h["case_id"],h["edge_id"])!=(case,eid)]
        row.update(source_annotation_path=str(source.resolve()),
                   original_source_annotation_path=raw_rows[case]["raw_annotation_path"],
                   original_source_annotation_sha256=sha(raw_rows[case]["raw_annotation_path"]),
                   wire_mask=str((folder/"wire_mask.png").resolve()),pin_heatmap=str((folder/"pin_heatmap.png").resolve()),
                   stats=stats,structurally_usable_draft=usable(stats),
                   label_status="local_manual_sparse_route_and_pin_repair_draft_not_full_page_certified",
                   manual_sparse_repairs=[patch],path_recovery_holds=[h for h in holds if h["case_id"]==case])
        row.setdefault("path_recovery_edges",[]).append(eid)
        staged.append(dict(row=row,folder=folder,source=source,revised=revised,wire=wire,pin=pin,tips=tips,added=previous_added|added,first_point=patch["path"][0]))
        page=next(p for p in pages if p["case_id"]==case)
        page.update(recovered_edges=page["recovered_edges"]+1,held_edges=page["held_edges"]-1,
                    added_pixels=page["added_pixels"]+int(added.sum()),after_missing=stats["missing_physical_paths"],
                    pin_heatmap_and_points_unchanged=False,manual_pin_points_changed=len(patch["pins"]))
        record=dict(case_id=case,edge_id=eid,category="manual_visible_route_with_linked_tip_correction",
                    path=patch["path"],source_ink_fraction=1.,added_pixels=int(added.sum()),
                    pin_points_changed=len(patch["pins"]),pin_heatmap_changed_pixels=int(pin_diff.sum()),
                    pin_changes=patch["pins"],component_bbox_change=patch.get("component_bbox"),
                    reason=patch["reason"],full_page_certified=False,
                    old_source_coordinates=original["edges"][eid]["coordinates"])
        manual.append(record);repairs.append(record)
        x0,y0,x1,y1=patch["roi"];extent=[x0-.5,x1-.5,y1-.5,y0-.5]
        fig,axes=plt.subplots(1,3,figsize=(12,4),dpi=160,layout="constrained")
        axes[0].imshow(rgb[y0:y1,x0:x1],extent=extent,interpolation="nearest")
        axes[0].plot(*np.asarray(patch["path"]).T,color="#ed6b22",linewidth=1.3)
        axes[0].set_title("原图与复核路径（橙）")
        for ax,mask,title in ((axes[1],before,"连续恢复后 / 本条修订前"),(axes[2],after,"本条路线及关联端点修订后")):
            ax.imshow(mask[y0:y1,x0:x1],extent=extent,interpolation="nearest",cmap="gray",vmin=0,vmax=1)
            ax.set_title(title)
        for ax in axes:
            for p in patch["pins"]:
                ax.scatter(*p["old_tip"],s=65,facecolors="none",edgecolors="#ef6629",linewidths=1.5)
                ax.scatter(*p["tip"],s=65,facecolors="none",edgecolors="#2189d0",linewidths=1.5)
            ax.set_xlim(x0-.5,x1-.5);ax.set_ylim(y1-.5,y0-.5)
            ax.set_xlabel("原图 x");ax.set_ylabel("原图 y")
        fig.suptitle(f"{case}/{eid} · 橙圈=原tip，蓝圈=修订tip · 新增{int(added.sum())}导线像素")
        fig.savefig(REPORT/"review"/f"manual_{case}_{eid}.png",facecolor="white");plt.close(fig)
    assert len(staged)==2 and len(holds)==13
    # Validate every proposed correction before committing either page.
    for stage in staged:
        row,folder,source=stage["row"],stage["folder"],stage["source"]
        write_json(source,stage["revised"])
        stage["wire"].save(folder/"wire_mask.png");stage["pin"].save(folder/"pin_heatmap.png")
        write_json(folder/"pin_points.json",stage["tips"])
        Image.fromarray(stage["added"].astype(np.uint8)*255).save(folder/"added_pixels.png")
        row["source_annotation_sha256"]=sha(source)
        write_json(folder/"metadata.json",row)
        # Verify a loader patch containing this route, not only an in-memory target.
        x,y=stage["first_point"]
        starts=[(xx,yy) for yy in tile_starts(row["height"],512,384) for xx in tile_starts(row["width"],512,384)]
        idx=next(i for i,(xx,yy) in enumerate(starts) if xx<=x<xx+512 and yy<=y<yy+512)
        _,target,valid=SegmentationTiles([row],512,384)[idx];xx,yy=starts[idx]
        assert target[0,y-yy,x-xx]==1 and valid[y-yy,x-xx]==1
    assert len(holds)==13 and len(manual)==2
    manifest=OUTPUT/"manifest.jsonl"
    manifest.write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in rows),encoding="utf-8")
    frozen_val=read_rows(OUTPUT/"frozen_validation_manifest.jsonl")
    train=[r for r in rows if r["split"]=="train" and r["structurally_usable_draft"]]
    (OUTPUT/"training_manifest.jsonl").write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in train+frozen_val),encoding="utf-8")
    summary.update(status="path_recovery_and_two_manual_sparse_repairs_complete_with_exception_holds",
                   recovered_in_this_run=678,recovered_total=679,held_missing_edges=13,
                   remaining_missing_edges=sum(r["stats"]["missing_physical_paths"] for r in rows),
                   remaining_missing_pages=sum(r["stats"]["missing_physical_paths"]>0 for r in rows),
                   hold_categories=dict(Counter(r["category"] for r in holds)),
                   added_pixels=sum(p["added_pixels"] for p in pages),
                   continuous_recovery_pin_targets_unchanged=True,pin_targets_unchanged=False,
                   manually_corrected_pin_points=3,manually_corrected_component_boxes=1,
                   manual_sparse_repairs_applied=True,manual_sparse_repairs=manual,
                   no_new_centerline_points_or_gap_bridges=False,
                   no_unsupported_gap_bridges=True,manual_script_sha256=sha(Path(__file__)),
                   manifest_sha256=sha(manifest),training_candidates=len(train),modified_pages=sum(p["recovered_edges"]>0 for p in pages))
    assert summary["remaining_missing_edges"]==13
    assert all(sha(p)==v for p,v in summary["source_hashes"].items())
    for path,value in ((OUTPUT/"summary.json",summary),(REPORT/"summary.json",summary),
                       (REPORT/"repairs.json",repairs),(REPORT/"holds.json",holds),
                       (REPORT/"pages.json",pages),(REPORT/"manual_repairs.json",manual)):
        write_json(path,value)
    print(json.dumps(dict(recovered_total=679,held=13,manual_repairs=manual,original_files_unchanged=True),ensure_ascii=False))


if __name__=="__main__":
    main()
