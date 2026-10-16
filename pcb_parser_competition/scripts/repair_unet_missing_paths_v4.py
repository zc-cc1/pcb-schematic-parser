"""Apply verified source-trace recovery to a separate auditable draft revision."""
from collections import Counter, defaultdict
import copy
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pcb_parser.dataset_converter import load_source_annotation
from pcb_parser.unet_data import annotation_targets, SegmentationTiles, tile_starts
from pcb_parser.unet_path_recovery import recover_contiguous_path
from repair_unet_0005_dp_path import sha, usable, write_json, read_rows

OUTPUT = ROOT / "data/segmentation_unet_v4_path_recovery_20261015"
REPORT = ROOT / "runs/unet_path_recovery_v4"
BASE = ROOT / "data/segmentation_unet_v3_local_path_repairs/manifest.jsonl"
SEMANTIC_HOLDS = {
    ("0024", "E_41"): "Visual review shows the trace follows a Key switch body outline rather than a free wire; do not rasterize this source record as wire.",
    ("0203", "E_92"): "Visual review shows a long trace along the development-board image/pin legend, touching multiple different pin rows; electrical-wire semantics are not reliable.",
    ("0205", "E_86"): "Visual review shows a trace along the development-board image/pin legend, connecting distinct pin rows; exclude from automatic source-path recovery.",
}


def main():
    audit_path = ROOT / "runs/unet_missing_path_audit_20261015/audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    original_rows = read_rows(BASE)
    by_case = {row["case_id"]: row for row in original_rows}
    groups = defaultdict(list)
    for record in audit["records"]:
        groups[record["case_id"]].append(record)
    assert len(groups) == 121 and sum(map(len, groups.values())) == 691
    # Bind this mutation to the audited, unchanged source files and manifests.
    protected = dict(audit["summary"]["source_sha256"])
    assert all(sha(path) == digest for path, digest in protected.items())
    for path in (ROOT / "configs/unet_wire_pin.json", ROOT / "configs/unet_label_reviews.json", ROOT / "weights/unet_wire_pin_v1.pt"):
        if path.exists():
            protected[str(path)] = sha(path)
    for case in groups:
        for key in ("image", "wire_mask", "pin_heatmap"):
            protected[by_case[case][key]] = sha(by_case[case][key])
    OUTPUT.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    derived_rows = copy.deepcopy(original_rows)
    derived = {row["case_id"]: row for row in derived_rows}
    repairs = []
    holds = []
    page_records = []
    for index, (case, records) in enumerate(groups.items(), 1):
        row = by_case[case]
        source = Path(records[0]["source"])
        annotation, _ = load_source_annotation(source)
        rgb = np.asarray(Image.open(row["image"]).convert("RGB"))
        old_wire, old_pin, old_tips, old_stats = annotation_targets(annotation, row["width"], row["height"], 3, 2)
        before = np.asarray(old_wire) > 0
        assert np.array_equal(np.asarray(old_wire), np.asarray(Image.open(row["wire_mask"])))
        assert np.array_equal(np.asarray(old_pin), np.asarray(Image.open(row["pin_heatmap"])))
        revised = copy.deepcopy(annotation)
        source_support = np.zeros_like(before)
        changed_ids = []
        page_holds = []
        for record in records:
            eid = record["edge_id"]
            edge = revised["edges"][eid]
            reason = SEMANTIC_HOLDS.get((case, eid))
            if reason or record["category"] == "two_sparse_coordinates":
                hold = dict(case_id=case, edge_id=eid, category="semantic_conflict" if reason else "sparse_unreliable_route",
                            reason=reason or "Only two disconnected points are provided; visual review shows misplaced endpoints or no corresponding direct physical route. Do not invent a bridge.",
                            source_coordinates=edge["coordinates"], source=str(source),
                            review_image=str((REPORT / "review" / (f"sparse_{case}.png" if not reason else "continuous20.png")).resolve()))
                holds.append(hold); page_holds.append(hold)
                continue
            assert record["category"] in {"straight_ink_supported", "ordered_nonstraight"}
            path = recover_contiguous_path(edge["coordinates"], row["width"], row["height"], rgb)
            # Independent pixel-set check: recovery must not add any centerline point.
            raster = Image.new("1", (row["width"], row["height"]))
            ImageDraw.Draw(raster).line([tuple(p) for p in path], fill=1, width=1)
            ys, xs = np.where(np.asarray(raster, dtype=bool))
            assert set(zip(xs.tolist(), ys.tolist())) == {tuple(p) for p in edge["coordinates"]}, (case, eid)
            source_support[ys, xs] = True
            old_path = copy.deepcopy(edge["keyPointPaths"])
            edge["keyPointPaths"] = {"wire": [path]}
            changed_ids.append(eid)
            repairs.append(dict(case_id=case, edge_id=eid, original_paths=old_path, path=path,
                                category=record["category"], source_coordinate_count=len(edge["coordinates"]),
                                source_centerline_pixel_set_preserved=True, added_centerline_points=0,
                                supervision_status="source_geometry_recovered_draft_not_full_page_certified"))
        if changed_ids:
            wire, pin, tips, stats = annotation_targets(revised, row["width"], row["height"], 3, 2)
            after = np.asarray(wire) > 0
            added, removed = after & ~before, before & ~after
            assert not removed.any()
            assert np.array_equal(np.asarray(pin), np.asarray(old_pin)) and tips == old_tips
            assert stats["missing_physical_paths"] == old_stats["missing_physical_paths"] - len(changed_ids)
            # Width expansion may include neighboring white pixels, but cannot leave a 2px source tube.
            distance = distance_transform_edt(~source_support)
            assert not added.any() or float(distance[added].max()) <= 2.0
            for eid in changed_ids:
                revised["edges"][eid]["keyPointPaths"] = {"wire": revised["edges"][eid]["keyPointPaths"]["wire"]}
            restored = copy.deepcopy(revised)
            for eid in changed_ids:
                restored["edges"][eid]["keyPointPaths"] = copy.deepcopy(annotation["edges"][eid]["keyPointPaths"])
            assert restored == annotation, "Mutation outside selected path fields"
            folder = OUTPUT / case
            folder.mkdir(exist_ok=True)
            annotation_path = folder / "source_annotation.json"
            write_json(annotation_path, revised)
            wire.save(folder / "wire_mask.png")
            pin.save(folder / "pin_heatmap.png")
            Image.fromarray(added.astype(np.uint8) * 255).save(folder / "added_pixels.png")
            write_json(folder / "pin_points.json", tips)
            derived[case].update(
                wire_mask=str((folder / "wire_mask.png").resolve()),
                pin_heatmap=str((folder / "pin_heatmap.png").resolve()),
                source_annotation_path=str(annotation_path.resolve()),
                source_annotation_sha256=sha(annotation_path),
                original_source_annotation_path=str(source), original_source_annotation_sha256=sha(source),
                stats=stats, structurally_usable_draft=usable(stats) and not page_holds,
                label_status="source_coordinate_path_recovered_draft_not_full_page_certified",
                path_recovery_edges=changed_ids, path_recovery_holds=page_holds,
            )
            write_json(folder / "metadata.json", derived[case])
            if added.any():
                first_y, first_x = np.argwhere(added)[0]
                starts = [(x,y) for y in tile_starts(row["height"],512,384) for x in tile_starts(row["width"],512,384)]
                patch_index = next(i for i,(x,y) in enumerate(starts) if x <= first_x < x+512 and y <= first_y < y+512)
                images, targets, valid = SegmentationTiles([derived[case]],512,384)[patch_index]
                x, y = starts[patch_index]
                assert targets[0,first_y-y,first_x-x] == 1 and valid[first_y-y,first_x-x] == 1
            page_records.append(dict(case_id=case, split=row["split"], recovered_edges=len(changed_ids),
                                     held_edges=len(page_holds), added_pixels=int(added.sum()), removed_pixels=0,
                                     before_missing=old_stats["missing_physical_paths"], after_missing=stats["missing_physical_paths"],
                                     structurally_usable_draft=derived[case]["structurally_usable_draft"],
                                     pin_heatmap_and_points_unchanged=True, dataset_loader_verified=True))
        else:
            derived[case]["path_recovery_holds"] = page_holds
            page_records.append(dict(case_id=case, split=row["split"], recovered_edges=0, held_edges=len(page_holds),
                                     added_pixels=0, removed_pixels=0, before_missing=old_stats["missing_physical_paths"],
                                     after_missing=old_stats["missing_physical_paths"], structurally_usable_draft=False,
                                     pin_heatmap_and_points_unchanged=True, dataset_loader_verified=None))
        if index % 15 == 0:
            print(json.dumps(dict(phase="repair_and_verify", pages=index,total=len(groups),recovered=len(repairs),held=len(holds)),ensure_ascii=False),flush=True)
    assert len(repairs) == 676 and len(holds) == 15
    assert all(derived[row["case_id"]] == row for row in original_rows if row["case_id"] not in groups)
    manifest = OUTPUT / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(row,ensure_ascii=False)+"\n" for row in derived_rows),encoding="utf-8")
    # Keep original validation rows for comparable future model evaluation.
    v1 = read_rows(ROOT / "data/segmentation_unet_v1/manifest.jsonl")
    frozen_val = [row for row in v1 if row["split"] == "val" and row["structurally_usable_draft"]]
    training = [row for row in derived_rows if row["split"] == "train" and row["structurally_usable_draft"]]
    training_manifest = OUTPUT / "training_manifest.jsonl"
    training_manifest.write_text("".join(json.dumps(row,ensure_ascii=False)+"\n" for row in training+frozen_val),encoding="utf-8")
    (OUTPUT / "frozen_validation_manifest.jsonl").write_text("".join(json.dumps(row,ensure_ascii=False)+"\n" for row in frozen_val),encoding="utf-8")
    assert len(frozen_val) == 28
    assert all(sha(path) == digest for path, digest in protected.items())
    summary = dict(
        status="continuous_source_path_recovery_complete_with_explicit_exception_holds",
        recovered_in_this_run=len(repairs), previously_recovered_0005=1, recovered_total=677,
        held_missing_edges=len(holds), remaining_missing_pages=sum(row["stats"]["missing_physical_paths"]>0 for row in derived_rows),
        remaining_missing_edges=sum(row["stats"]["missing_physical_paths"] for row in derived_rows),
        hold_categories=dict(Counter(r["category"] for r in holds)),
        added_pixels=sum(row["added_pixels"] for row in page_records), removed_pixels=0,
        modified_pages=sum(row["recovered_edges"]>0 for row in page_records),
        structurally_usable_drafts=sum(row["structurally_usable_draft"] for row in derived_rows),
        usable_by_split=dict(Counter(row["split"] for row in derived_rows if row["structurally_usable_draft"])),
        previous_train_cases=104, training_candidates=len(training), frozen_validation_cases=28,
        dataset_manifest=str(manifest.resolve()), training_manifest=str(training_manifest.resolve()),
        original_files_unchanged=True, pin_targets_unchanged=True,
        no_new_centerline_points_or_gap_bridges=True, training_started=False,
        full_page_certified=False, official_score=None, default_training_manifest_changed=False,
        source_hashes=protected, recovery_module_sha256=sha(ROOT/"src/pcb_parser/unet_path_recovery.py"),
        repair_script_sha256=sha(Path(__file__)), manifest_sha256=sha(manifest),
    )
    write_json(OUTPUT / "summary.json", summary)
    write_json(REPORT / "repairs.json", repairs)
    write_json(REPORT / "holds.json", holds)
    write_json(REPORT / "pages.json", page_records)
    write_json(REPORT / "summary.json", summary)
    print(json.dumps({k:v for k,v in summary.items() if k!="source_hashes"},ensure_ascii=False))


if __name__ == "__main__":
    main()
