"""Screen collected schematics for duplicates and prepare a human review queue."""
import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from prepare_detection_dataset import perceptual_hash

ROOT = Path(__file__).resolve().parents[1]


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def signature(path):
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        size = rgb.size
        pixel_hash = hashlib.sha256(str(size).encode()+rgb.tobytes()).hexdigest()
        return {"size": size, "pixel_sha256": pixel_hash, "phash": perceptual_hash(rgb), "aspect": size[0]/size[1]}


def normalize(value):
    return re.sub(r"[^a-z0-9]", "", value.lower())


def annotation_text(value):
    # Search actual text fields rather than every numeric annotation value.
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "content" and isinstance(item, str):
                yield item
            else:
                yield from annotation_text(item)
    elif isinstance(value, list):
        for item in value:
            yield from annotation_text(item)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT/"configs/external_data_review.json")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    folder = ROOT/cfg["collection"]
    external = rows(folder/"manifest.jsonl")
    original = rows(ROOT/cfg["reference_manifest"])
    reference = []
    for row in original:
        if file_hash(row["image_path"]) != row["image_sha256"] or file_hash(row["raw_annotation_path"]) != row["annotation_sha256"]:
            raise RuntimeError(f"Dataset300 source changed since intake: {row['case_id']}")
        raw = json.loads(Path(row["raw_annotation_path"]).read_text(encoding="utf-8-sig"))
        reference.append({"case_id": row["case_id"], **signature(row["image_path"]),
                          "available_text": normalize(" ".join(annotation_text(raw)))})
    external_signatures = {row["case_id"]: signature(row["image"]) for row in external}
    screen = []
    for row in external:
        sig = external_signatures[row["case_id"]]
        matches, nearest = [], []
        for other in reference:
            distance = (sig["phash"] ^ other["phash"]).bit_count()
            aspect_ok = abs(sig["aspect"]-other["aspect"])/max(sig["aspect"], other["aspect"]) <= cfg["aspect_relative_tolerance"]
            reasons = []
            if sig["pixel_sha256"] == other["pixel_sha256"]: reasons.append("exact_rgb_pixels")
            if distance <= cfg["phash_max_distance"] and aspect_ok: reasons.append("perceptual_near_duplicate_candidate")
            needles = cfg["source_text_needles"].get(row["source_id"], [])
            if any(normalize(needle) in other["available_text"] for needle in needles): reasons.append("source_name_text_match_candidate")
            if reasons: matches.append({"case_id": other["case_id"], "reasons": reasons, "phash_distance": distance})
            nearest.append({"case_id": other["case_id"], "phash_distance": distance, "aspect_compatible": aspect_ok})
        within = []
        for other in external:
            if other["case_id"] >= row["case_id"]: continue
            sig2 = external_signatures[other["case_id"]]
            distance = (sig["phash"] ^ sig2["phash"]).bit_count()
            aspect_ok = abs(sig["aspect"]-sig2["aspect"])/max(sig["aspect"],sig2["aspect"]) <= cfg["aspect_relative_tolerance"]
            if sig["pixel_sha256"] == sig2["pixel_sha256"] or (distance <= cfg["phash_max_distance"] and aspect_ok):
                within.append({"case_id": other["case_id"], "phash_distance": distance})
        reserved = "external_test_pending_annotation" if row["source_group"] in cfg["external_holdout_groups"] else "train_candidate_pending_annotation"
        screen.append({**row, "reserved_split": reserved, "training_eligible": False,
                       "pixel_sha256": sig["pixel_sha256"], "phash_hex": format(sig["phash"], "016x"),
                       "dataset300_duplicate_candidates": matches, "within_external_duplicate_candidates": within,
                       "nearest_reference_images": sorted(nearest,key=lambda x:x["phash_distance"])[:3],
                       "engineering_novelty": "manual_source_and_design_review_pending",
                       "deduplication_status": "review_candidate_matches" if matches or within else "no_matches_under_configured_screen"})
    save(folder/"duplicate_review.json", {"created_at_utc":datetime.now(timezone.utc).isoformat(),
         "reference_cases":len(original), "reference_manifest_sha256":file_hash(ROOT/cfg["reference_manifest"]),
         "external_pages":len(screen), "config":cfg, "cases":screen,
         "limitations":"Different renders, cropped schematics and modified projects can evade this screen. Official PCBnet overlap is unknown."})
    (folder/"review_manifest.jsonl").write_text("\n".join(json.dumps(r,ensure_ascii=False) for r in screen)+"\n",encoding="utf-8")
    native = json.loads((folder/"native_summary.json").read_text(encoding="utf-8"))
    target_queue = []
    for inventory in native:
        for obj in inventory["target_candidates"]:
            target_queue.append({"source_id":inventory["source_id"], "reference":obj["reference"],
                                 "value":obj["value"], "library_id":obj["library_id"], "hints":";".join(obj["hints"]),
                                 "verified_class":"", "pixel_bbox":"", "review_status":"needs_visual_review_and_coordinate_mapping"})
    with (folder/"rare_symbol_review.csv").open("w",encoding="utf-8-sig",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=["source_id","reference","value","library_id","hints","verified_class","pixel_bbox","review_status"])
        writer.writeheader();writer.writerows(target_queue)
    gallery=ROOT/"artifacts/external_data_review";gallery.mkdir(parents=True,exist_ok=True)
    font=ImageFont.truetype("C:/Windows/Fonts/msyh.ttc",17)
    sheet=Image.new("RGB",(1600,((len(screen)+1)//2)*600),"white")
    draw=ImageDraw.Draw(sheet)
    for idx,row in enumerate(screen):
        x,y=(idx%2)*800,(idx//2)*600
        with Image.open(row["image"]) as image:
            image=image.convert("RGB");image.thumbnail((780,540),Image.Resampling.LANCZOS)
            sheet.paste(image,(x+(800-image.width)//2,y+50+(540-image.height)//2))
        draw.text((x+10,y+7),row["source_id"],font=font,fill="black")
        draw.text((x+10,y+28),f"{row['width']}x{row['height']} | {row['reserved_split']}",font=font,fill="#444444")
        draw.rectangle((x,y,x+799,y+599),outline="#aaaaaa")
    sheet.save(gallery/"collected_pages.png")
    result={"external_pages":len(screen),"dataset300_reference_cases":len(original),
            "pages_with_duplicate_candidates":sum(bool(r["dataset300_duplicate_candidates"] or r["within_external_duplicate_candidates"]) for r in screen),
            "reserved_splits":dict(Counter(r["reserved_split"] for r in screen)),
            "native_rare_symbol_hints":len(target_queue),"training_eligible_pages":0,"official_bonus_score":None}
    training_audit=json.loads((ROOT/"data/detection_balanced_v2/audit.json").read_text(encoding="utf-8"))
    original_detection_manifest=ROOT/training_audit["config"]["source_manifest"] if "config" in training_audit else ROOT/"data/detection_v1/manifest.jsonl"
    actual_hash=file_hash(original_detection_manifest)
    if actual_hash != training_audit["source_manifest_sha256"]:
        raise RuntimeError("Frozen Dataset300 detection manifest changed")
    result["unchanged_original_detection_manifest_sha256"]=actual_hash
    result["original_validation_pages"]=sum(r["split"]=="val" for r in rows(original_detection_manifest))
    save(folder/"review_summary.json",result)
    print(json.dumps(result,ensure_ascii=False))


if __name__=="__main__": main()
