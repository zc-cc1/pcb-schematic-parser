"""Independent artifact, graph, rollback and training-input checks for local v6 edits."""
from collections import Counter
import copy
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from pcb_parser.unet_data import annotation_targets, polylines, SegmentationTiles
from pcb_parser.unet_model import segmentation_loss
from pcb_parser.unet_retraining import CachedTrainingTiles
from pcb_parser.unet_supervision import load_valid_mask
from repair_unet_0005_dp_path import read_rows,sha,write_json

BASE = ROOT/"data/segmentation_unet_v5_reviewed_supervision_20261015"
OUT = ROOT/"data/segmentation_unet_v6_corrected_relations_20261015"
REPORT = ROOT/"runs/unet_relation_repair_v6"


def rollback(data, edits):
    value = copy.deepcopy(data)
    for edit in reversed(edits):
        parts = [p.replace("~1","/").replace("~0","~") for p in edit["path"].split("/")[1:]]
        target = value
        for key in parts[:-1]: target = target[key]
        key = parts[-1]
        if edit["op"] == "add":
            assert target[key] == edit["after"]
            del target[key]
        elif edit["op"] == "remove":
            assert key not in target
            target[key] = edit["before"]
        else:
            assert target[key] == edit["after"]
            target[key] = edit["before"]
    return value


def main():
    summary = json.loads((OUT/"summary.json").read_text(encoding="utf-8"))
    assert sha(OUT/"manifest.jsonl") == summary["manifest_sha256"]
    assert sha(OUT/"training_manifest.jsonl") == summary["training_manifest_sha256"]
    assert all(sha(p)==digest for p,digest in summary["protected_inputs_sha256"].items())
    # v5 recorded hashes also protect all older labels and external source inputs.
    old_summary = json.loads((BASE/"summary.json").read_text(encoding="utf-8"))
    assert all(sha(p)==digest for p,digest in old_summary["source_files_sha256"].items())
    rows = read_rows(OUT/"manifest.jsonl")
    prior = {r["case_id"]:r for r in read_rows(BASE/"manifest.jsonl")}
    pages = {p["case_id"]:p for p in json.loads((REPORT/"pages.json").read_text(encoding="utf-8"))}
    fixes = json.loads((OUT/"relation_repairs.json").read_text(encoding="utf-8"))
    operations = json.loads((REPORT/"operations.json").read_text(encoding="utf-8"))
    assert len(rows)==300 and len(pages)==6 and len(fixes)==10
    expected = {(r["case_id"],r["edge_id"]) for r in json.loads((BASE/"graph_review_queue.json").read_text(encoding="utf-8"))}
    assert {(r["case_id"],r["edge_id"]) for r in fixes} == expected
    assert json.loads((OUT/"graph_review_queue.json").read_text(encoding="utf-8")) == []
    changed_sources = regenerated = routes = restored = added = removed = pin_changed = 0
    sources = {}
    for index,row in enumerate(rows,1):
        case = row["case_id"]; old = prior[case]
        assert row["image"]==old["image"] and row["split"]==old["split"]
        for key in ("wire_mask","pin_heatmap"):
            with Image.open(row[key]) as im:
                assert im.size==(row["width"],row["height"])
                if key=="wire_mask": assert np.isin(np.asarray(im),[0,255]).all()
        if row.get("source_annotation_path"):
            path = Path(row["source_annotation_path"])
            assert sha(path)==row["source_annotation_sha256"]
            data = json.loads(path.read_text(encoding="utf-8"))
            w,p,t,stats = annotation_targets(data,row["width"],row["height"],3,2)
            assert stats==row["stats"]
            assert np.array_equal(np.asarray(w),np.asarray(Image.open(row["wire_mask"])))
            assert np.array_equal(np.asarray(p),np.asarray(Image.open(row["pin_heatmap"])))
            assert json.loads((path.parent/"metadata.json").read_text(encoding="utf-8"))==row
            assert json.loads((path.parent/"pin_points.json").read_text(encoding="utf-8"))==t
            regenerated += 1
        if case not in pages:
            assert row==old
        else:
            sources[case]=data; changed_sources += 1
            previous = json.loads(Path(old["source_annotation_path"]).read_text(encoding="utf-8"))
            edits = json.loads(Path(row["source_changes_path"]).read_text(encoding="utf-8"))
            assert rollback(data,edits)==previous
            assert all(e["path"].split("/")[1] in {"edges","components","pins"} for e in edits)
            assert row["supervision_ignored_pixels"]==0 and (load_valid_mask(row)==1).all()
            restored += int((load_valid_mask(old)==0).sum())
            before = np.asarray(Image.open(old["wire_mask"]))>0
            after = np.asarray(w)>0
            a = after&~before; r=before&~after
            assert np.array_equal(np.asarray(Image.open(path.parent/"added_pixels.png"))>0,a)
            assert np.array_equal(np.asarray(Image.open(path.parent/"removed_pixels.png"))>0,r)
            added += int(a.sum()); removed += int(r.sum())
            pin_changed += int((np.asarray(p)!=np.asarray(Image.open(old["pin_heatmap"]))).sum())
            assert stats["unowned_root_pins"]==old["stats"]["unowned_root_pins"]
            assert stats["missing_physical_paths"]==stats["invalid_pin_tips"]==stats["out_of_bounds_paths"]==0
            assert stats.get("ignored_source_associations",0)==0
            rgb = np.asarray(Image.open(row["image"]).convert("RGB"))
            for eid in pages[case]["retired_edges"]: assert data["edges"][eid]["state"]=="deleted"
            for route in pages[case]["visible_routes"]:
                edge = data["edges"][route["edge_id"]]
                assert edge["state"]=="active" and edge["isFlywire"] is False
                support = Image.new("1",(row["width"],row["height"]))
                for line in polylines(edge["keyPointPaths"]["wire"]): ImageDraw.Draw(support).line(line,fill=1,width=1)
                mask = np.asarray(support,dtype=bool)
                assert (rgb[mask].min(axis=1)<224).all()
                points = set(zip(*np.where(mask)))
                visited = {next(iter(points))}; pending=list(visited)
                while pending:
                    y,x=pending.pop()
                    for neighbor in ((y-1,x),(y+1,x),(y,x-1),(y,x+1)):
                        if neighbor in points and neighbor not in visited: visited.add(neighbor);pending.append(neighbor)
                assert visited==points, (case,route["edge_id"],"disconnected route")
                for pid,conn in edge["connect"].items():
                    local = data["components"][conn["cpntId"]]["pins"][pid]
                    assert data["pins"][pid]==local and local.get("state")!="deleted"
                    assert local["edgeId"]==route["edge_id"] and conn["location"]==local["tip"]
                    x,y=local["tip"];assert (y,x) in points
                routes += 1
        if index%50==0: print(json.dumps(dict(phase="verify_v6",pages=index,total=300)),flush=True)
    assert changed_sources==6 and routes==16
    assert (restored,added,removed,pin_changed)==(4425,1168,117,4229)
    assert sum(r["stats"]["missing_physical_paths"] for r in rows)==0
    assert sum(r["stats"].get("ignored_source_associations",0) for r in rows)==0
    assert sum(r["stats"].get("excluded_nonwire_records",0) for r in rows)==3
    assert sum(r.get("supervision_ignored_pixels",0) for r in rows)==675
    assert all(sources[c]["edges"][e]["state"]=="deleted" for c,e in expected)
    # Specific graph regressions: separate nets, explicit NC and independent decouplers.
    d=sources["0101"];assert d["pins"]["P_242"]["edgeId"]=="E_62" and d["pins"]["P_139"]["edgeId"]=="E_120"
    d=sources["0102"];assert d["pins"]["P_312"]["edgeId"]=="E_101" and d["pins"]["P_308"]["edgeId"]=="E_93"
    assert d["pins"]["P_121"]["edgeId"]!=d["pins"]["P_122"]["edgeId"] and d["pins"]["P_135"]["state"]=="deleted"
    assert sources["0124"]["pins"]["P_229"]["edgeId"] is None
    d=sources["0150"];assert [d["pins"][p]["edgeId"] for p in ("P_76","P_85","P_207")]==["E_22","E_3","E_38"]
    d=sources["0180"];assert all(d["pins"][p]["edgeId"] is None for p in ("P_93","P_94"))
    assert d["pins"]["P_76"]["edgeId"]!=d["pins"]["P_79"]["edgeId"] and d["pins"]["P_79"]["edgeId"]=="E_3"
    d=sources["0238"];assert d["pins"]["P_5"]["edgeId"]==d["pins"]["P_59"]["edgeId"]=="E_11"
    assert d["pins"]["P_56"]["edgeId"]==d["pins"]["P_55"]["edgeId"]=="E_28"
    frozen=read_rows(OUT/"frozen_validation_manifest.jsonl")
    assert frozen==read_rows(BASE/"frozen_validation_manifest.jsonl")
    training=read_rows(OUT/"training_manifest.jsonl")
    assert [r for r in training if r["split"]=="val"]==frozen
    assert len(training)==199 and Counter(r["split"] for r in training)=={"train":171,"val":28}
    # Both live loaders see the revised arrays and restored valid regions.
    row=next(r for r in rows if r["case_id"]=="0180")
    direct=SegmentationTiles([row],512,384);cached=CachedTrainingTiles([row],512,384)
    idx=next(i for i,(_,x,y) in enumerate(direct.tiles) if x<=540<x+512 and y<=410<y+512)
    a,b=direct[idx],cached[idx]
    assert all(torch.equal(x,y) for x,y in zip(a,b))
    _,ox,oy=direct.tiles[idx];assert a[2][410-oy,540-ox]==1
    logits=torch.zeros((1,2,512,512),requires_grad=True)
    loss,_=segmentation_loss(logits,a[1][None],a[2][None]);loss.backward()
    assert (logits.grad[0,:,410-oy,540-ox].abs()>0).all()
    cached.clear_cache()
    checks=dict(status="passed",manifest_pages=300,unchanged_rows=294,regenerated_source_labels=regenerated,
                corrected_graph_relations=10,connected_visible_routes=16,source_rollback_exact=True,
                route_centerlines_on_visible_ink=True,root_and_component_pins_synchronized=True,
                restored_supervised_pixels=restored,remaining_ignored_pixels=675,
                added_wire_positive_pixels=added,removed_false_wire_pixels=removed,
                pin_heatmap_changed_pixels=pin_changed,frozen_validation_rows_unchanged=28,
                train_rows=171,both_training_loaders_verified=True,restored_region_has_both_channel_gradients=True,
                prior_sources_labels_weights_hashes_unchanged=True,training_started=False,full_pages_certified=0)
    write_json(REPORT/"final_checks.json",checks)
    print(json.dumps(checks),flush=True)


if __name__=="__main__":main()
