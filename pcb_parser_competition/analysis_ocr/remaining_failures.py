"""List every component the clean run still misses, and render a crop per case.

Two outputs:
  * a table of all remaining failures with GT vs predicted fields
  * image montages (never-emitted, bbox-only, value-only) so each one can be eyeballed
"""
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

PRED = Path(r"D:\pcb_final\out\final_results")
GT = Path(r"D:\pcb_final\gt")
OUT = Path(__import__("os").environ.get("EVIDENCE_DIR", str(Path(__file__).parent / "evidence")))
TOL = 20.0
POWER = {"gnd", "v"}


def norm_type(value):
    t = str(value or "").strip().lower()
    if t in {"ground", "gnd"}:
        return "gnd"
    if t in {"v", "vcc", "vdd"}:
        return "v"
    if t in {"box", "ic"}:
        return "box"
    return t


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default=str(PRED))
    ap.add_argument("--gt", default=str(GT))
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    pred_dir, gt_dir = Path(args.pred), Path(args.gt)
    never, field = [], []
    for folder in sorted((p for p in gt_dir.iterdir() if p.is_dir()), key=lambda p: (len(p.name), p.name)):
        case = folder.name
        gt = json.loads(next(folder.glob("*_target_named.json")).read_text(encoding="utf-8"))
        pred_path = pred_dir / f"{case}_Pred.json"
        if not pred_path.exists():
            continue
        pred = json.loads(pred_path.read_text(encoding="utf-8"))
        image = Image.open(next(folder.glob("*.png"))).convert("RGB")
        height = image.size[1]
        for key, gv in gt["components"].items():
            kind = norm_type(gv.get("type"))
            if kind in POWER:
                continue
            ov = pred["components"].get(key)
            if ov is None:
                never.append((case, key, gv, image, height))
                continue
            failed = []
            if kind != norm_type(ov.get("type")):
                failed.append("type")
            if (gv.get("value") or None) != (ov.get("value") or None):
                failed.append("value")
            if (gv.get("Name") or None) != (ov.get("Name") or None):
                failed.append("Name")
            if not all(abs(a - b) <= TOL for a, b in zip(gv["bbox"], ov["bbox"])):
                failed.append("bbox")
            if failed:
                field.append((case, key, gv, ov, image, height, failed))

    print(f"never emitted: {len(never)}")
    print(f"{'case':>4} {'key':>10} {'type':>14} {'GT Name':22s} {'GT value':12s}  GT bbox")
    for case, key, gv, _, _ in never:
        print(f"{case:>4} {key:>10} {str(gv.get('type')):>14} {str(gv.get('Name'))[:22]:22s} "
              f"{str(gv.get('value'))[:12]:12s}  {[round(v) for v in gv['bbox']]}")

    print(f"\nfield mismatches: {len(field)}")
    by_kind = Counter("+".join(row[6]) for row in field)
    print("  by failing field:", dict(by_kind))
    for case, key, gv, ov, _, _, failed in field:
        print(f"  case {case:>2} {key:>8} {str(gv.get('type')):>10} [{'+'.join(failed):12s}] "
              f"GT(Name={str(gv.get('Name'))[:15]!r} value={str(gv.get('value'))[:10]!r} "
              f"bbox={[round(v) for v in gv['bbox']]})  "
              f"OURS(Name={str(ov.get('Name'))[:15]!r} value={str(ov.get('value'))[:10]!r} "
              f"bbox={[round(v) for v in ov['bbox']]})")

    OUT.mkdir(parents=True, exist_ok=True)
    try:
        font = ImageFont.truetype("arial.ttf", 13)
    except OSError:
        font = ImageFont.load_default()

    def montage(rows, name, size=(150, 150), cols=6):
        rows = [(row[0], row[1], row[2], row[4], row[5]) if len(row) == 7 else row for row in rows]
        if not rows:
            return
        cell_w, cell_h = size[0] + 8, size[1] + 26
        grid_rows = (len(rows) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * cell_w, grid_rows * cell_h), (255, 255, 255))
        draw = ImageDraw.Draw(sheet)
        for index, row in enumerate(rows):
            case, key, gv = row[0], row[1], row[2]
            image, height = row[3], row[4]
            x1, y1, x2, y2 = gv["bbox"]
            top, bottom = height - y2, height - y1
            margin = 45
            crop = (max(0, x1 - margin), max(0, top - margin),
                    min(image.width, x2 + margin), min(image.height, bottom + margin))
            patch = image.crop(tuple(int(v) for v in crop))
            scale = min(size[0] / max(1, patch.width), size[1] / max(1, patch.height))
            patch = patch.resize((max(1, int(patch.width * scale)), max(1, int(patch.height * scale))),
                                 Image.LANCZOS)
            cx = (index % cols) * cell_w + 4
            cy = (index // cols) * cell_h + 4
            sheet.paste(patch, (cx, cy))
            draw.rectangle([cx, cy, cx + patch.width, cy + patch.height], outline=(0, 160, 160))
            draw.text((cx, cy + size[1] + 2), f"{case}/{key} {str(gv.get('type'))[:10]}",
                      fill=(0, 0, 0), font=font)
        target = OUT / name
        sheet.save(target)
        print(f"\nwrote {target}  ({len(rows)} crops, {sheet.width}x{sheet.height})")

    montage(never, f"remaining_never_emitted{args.tag}.png")
    montage([row for row in field if "bbox" in row[6]], f"remaining_bbox{args.tag}.png")
    montage([row for row in field if "value" in row[6]], f"remaining_value{args.tag}.png")
    montage([row for row in field if "Name" in row[6]], f"remaining_name{args.tag}.png")


if __name__ == "__main__":
    sys.exit(main())


