"""Build draft U-Net labels from source paths/tips, never from thresholded PNGs."""
import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from pcb_parser.dataset_converter import load_source_annotation
from pcb_parser.unet_data import annotation_targets


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--case-id",action="append")
    parser.add_argument("--output",type=Path,default=ROOT/"data/segmentation_unet_v1")
    args=parser.parse_args()
    cfg=json.loads((ROOT/"configs/unet_wire_pin.json").read_text(encoding="utf-8"))
    splits={r["case_id"]:r["split"] for r in map(json.loads,(ROOT/"data/detection_v1/manifest.jsonl").read_text(encoding="utf-8").splitlines())}
    inputs=list(map(json.loads,(ROOT/"data/manifest.jsonl").read_text(encoding="utf-8").splitlines()))
    if args.case_id:
        inputs=[row for row in inputs if row["case_id"] in args.case_id]
    args.output.mkdir(parents=True,exist_ok=True)
    output=[]
    failures=[]
    for row in inputs:
        case=row["case_id"]
        try:
            source,duplicates=load_source_annotation(row["raw_annotation_path"])
            image=ROOT/f"data/standardized/{case}/{case}.png"
            with Image.open(image) as im:
                width,height=im.size
            wire,heat,tips,stats=annotation_targets(source,width,height,cfg["label_line_width"],cfg["pin_sigma"])
            folder=args.output/case
            folder.mkdir(exist_ok=True)
            wire.save(folder/"wire_mask.png")
            heat.save(folder/"pin_heatmap.png")
            (folder/"pin_points.json").write_text(json.dumps(tips,ensure_ascii=False,indent=2),encoding="utf-8")
            usable=(stats["physical_edges"]>0 and stats["wire_positive_pixels"]>0 and stats["pin_tips"]>0
                    and not any(stats[k] for k in ["missing_physical_paths","out_of_bounds_paths","invalid_pin_tips","unowned_root_pins"]))
            record={"case_id":case,"image":str(image.resolve()),"width":width,"height":height,
                    "wire_mask":str((folder/"wire_mask.png").resolve()),"pin_heatmap":str((folder/"pin_heatmap.png").resolve()),
                    "split":splits.get(case,"excluded"),"source_group":row["source_group"],
                    "source_annotation_sha256":hashlib.sha256(Path(row["raw_annotation_path"]).read_bytes()).hexdigest(),
                    "image_sha256":hashlib.sha256(image.read_bytes()).hexdigest(),
                    "label_status":"draft_source_geometry_needs_pixel_and_tip_review",
                    "structurally_usable_draft":usable,"line_width":cfg["label_line_width"],
                    "pin_sigma":cfg["pin_sigma"],"stats":stats,"identical_source_duplicate_keys":len(duplicates)}
            (folder/"metadata.json").write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding="utf-8")
            output.append(record)
            if case in {"0001","0002","0131"}:
                # Diagnostic label display only, never a model input or a prediction.
                with Image.open(image) as im:
                    rgb=np.asarray(im.convert("RGB")).copy()
                selected=np.asarray(wire)>0
                rgb[selected]=(.35*rgb[selected]+.65*np.array([230,50,50])).astype(np.uint8)
                preview=Image.fromarray(rgb)
                pen=ImageDraw.Draw(preview)
                for tip in tips:
                    x,y=tip["point"]
                    pen.ellipse((x-4,y-4,x+4,y+4),outline=(0,130,240),width=2)
                preview.save(folder/"label_review.png")
        except Exception as error:
            failures.append({"case_id":case,"error":str(error)})
    name="manifest_selected.jsonl" if args.case_id else "manifest.jsonl"
    (args.output/name).write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in output),encoding="utf-8")
    summary={"status":"draft_labels_not_formal_Golden","cases_exported":len(output),"failures":failures,
             "structurally_usable_drafts":sum(r["structurally_usable_draft"] for r in output),
             "usable_by_split":dict(Counter(r["split"] for r in output if r["structurally_usable_draft"])),
             "source_engineering_groups_verified":False,"manually_approved_cases":0,
             "labels_are_generated_from_annotations_not_images":True,
             "unknown_width_policy":"configured 3px strokes are supervision drafts, not pixel-certified ground truth"}
    (args.output/("summary_selected.json" if args.case_id else "summary.json")).write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(summary,ensure_ascii=False))


if __name__=="__main__":
    main()
