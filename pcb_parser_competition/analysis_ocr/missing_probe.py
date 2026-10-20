"""For each component the export still misses: was the text read, was a symbol detected?

Three questions per item, so the fix lands in the right stage:
  1. is the printed text in the OCR rows?  (OCR recall)
  2. is there a detected box at that place? (detector recall)
  3. if both exist, was a component record built and then dropped? (export/classification)
"""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

WORK = Path(r"D:\pcb_final2\out\work")
GT = Path(r"D:\pcb_final\gt")
PRED = Path(r"D:\pcb_final3\out\final_results")


def centre(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def work_for(case, digest):
    for folder in WORK.iterdir():
        for owner in folder.glob("**/ownership/pin_ownership.json"):
            if json.loads(owner.read_text(encoding="utf-8")).get("image_sha256") == digest:
                return folder
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--radius", type=float, default=45.0)
    args = ap.parse_args()
    for folder in sorted((p for p in GT.iterdir() if p.is_dir()), key=lambda p: (len(p.name), p.name)):
        case = folder.name
        gt = json.loads(next(folder.glob("*_target_named.json")).read_text(encoding="utf-8"))
        pred = json.loads((PRED / f"{case}_Pred.json").read_text(encoding="utf-8"))
        png = next(folder.glob("*.png"))
        work = work_for(case, hashlib.sha256(png.read_bytes()).hexdigest())
        if work is None:
            continue
        ownership = json.loads(next(work.glob("**/ownership/pin_ownership.json")).read_text(encoding="utf-8"))
        texts = json.loads(next(work.glob("**/upstream/upstream/texts.json")).read_text(encoding="utf-8"))["texts"]
        height = ownership["size"][1]

        for key, component in gt["components"].items():
            if str(component.get("type")) in ("ground", "gnd", "v"):
                continue
            if key in (pred["components"] or {}):
                continue
            x1, y1, x2, y2 = component["bbox"]
            gx, gy = (x1 + x2) / 2, (y1 + y2) / 2
            want_name = str(component.get("Name") or "")
            want_value = str(component.get("value") or "")
            # 1. text
            tokens = []
            for row in texts:
                tx, ty = centre(row["bbox"])
                distance = math.hypot(tx - gx, height - ty - gy)
                if distance <= 90:
                    tokens.append((round(distance), str(row["text"])))
            tokens.sort()
            text_hit = [t for t in tokens if t[1] and (t[1] in (want_name, want_value, key)
                                                       or want_name in t[1] or key == t[1])]
            # 2. detected box
            best = (1e9, None)
            for detected in ownership["components"]:
                dx, dy = centre(detected["bbox_top_left"])
                d = math.hypot(dx - gx, height - dy - gy)
                if d < best[0]:
                    best = (d, detected)
            print(f"\n=== case {case} / {key}  type={component.get('type')}  "
                  f"Name={want_name!r} value={want_value!r}")
            print(f"    GT centre=({gx:.0f},{gy:.0f})  size={x2-x1:.0f}x{y2-y1:.0f}")
            print(f"    text in OCR within 90px: {tokens[:6]}")
            print(f"    EXACT text hit: {text_hit if text_hit else 'NONE'}")
            if best[1] is not None:
                detected = best[1]
                bx1, by1, bx2, by2 = detected["bbox_top_left"]
                print(f"    nearest detected box: {best[0]:.1f}px away, class={detected['class_name']!r}, "
                      f"size={bx2-bx1:.0f}x{by2-by1:.0f}")
            else:
                print("    no detected box at all")


if __name__ == "__main__":
    sys.exit(main())

