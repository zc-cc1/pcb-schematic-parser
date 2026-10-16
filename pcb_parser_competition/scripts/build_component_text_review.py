"""Offline, visually transcribed quick-dev reference. NEVER used by inference."""
import hashlib
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]


def main():
    names = {"U1": "RP2350B", "U2": "RM2", "U3": "W25Q128JVPIM", "U4": "APS6404L-3SQR-ZR", "Q1": "RE1C001ZPTL"}
    values = {**{f"C{i}": "0.1uF" for i in range(1,21)},
              **{f"C{i}": "4.7uF" for i in [1,3,5,6,10]}, "C7": "15pF", "C9": "15pF",
              "R7": "200", "R8": "33", "R9": "1k", "R10": "1k", "R11": "100", "R12": "100",
              "R13": "470", "R14": "10k", "R15": "10k", "R16": "10k", "R17": "10k",
              "L1": "3.3uH", "L2": "470Ohm", "Y1": "12MHz"}
    aliases = {"C8": "UNRESOLVED_C_18", "Q1": "UNRESOLVED_C_27", "R12": "UNRESOLVED_C_35",
               "R14": "UNRESOLVED_C_36", "R11": "UNRESOLVED_C_37", "R13": "UNRESOLVED_C_40",
               "R9": "UNRESOLVED_C_43", "L1": "UNRESOLVED_C_44", "R10": "UNRESOLVED_C_82",
               "TP3": "UNRESOLVED_C_89", "TP6": "UNRESOLVED_C_91", "TP7": "UNRESOLVED_C_90"}
    references = {}
    for case in ["0001", "0002"]:
        path = ROOT / f"data/standardized/{case}/{case}.png"
        im = Image.open(path).convert("RGB")
        w, h = im.size
        converted = json.loads((path.with_suffix(".json")).read_text(encoding="utf-8"))["components"]
        refs = ([*values, *names, *[f"TP{i}" for i in range(1,8)]] if case == "0001"
                else ["C74", "C81", "C82", "C83", "C86", "C87", "IC2", "X2"])
        rows = []
        for ref in refs:
            source_id = aliases.get(ref, ref) if case == "0001" else ref
            component = converted[source_id]
            x1, y1, x2, y2 = component["bbox"]
            fields = {"designator": ref}
            if case == "0001":
                if ref in values:
                    fields["value"] = values[ref]
                if ref in names:
                    fields["Name"] = names[ref]
            else:
                fields.update({"IC2": {"Name": "BCM2835"}, "X2": {"value": "19M2"}, "C82": {"value": "1u0"}}
                              .get(ref, {"value": component["value"]}))
            rows.append({"designator": ref, "type": component["type"], "bbox_top_left": [x1,h-y2,x2,h-y1],
                         "fields": fields, "review_status": "visually_transcribed_from_original_PNG",
                         "geometry_source": source_id})
        references[case] = {"image_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "size": [w,h], "components": rows}
        # Context crops preserve the component plus surrounding text for auditing.
        for page in range((len(rows)+11)//12):
            subset = rows[page*12:(page+1)*12]
            sheet = Image.new("RGB", (1440, len(subset)//3*270 + (270 if len(subset)%3 else 0)), "white")
            draw = ImageDraw.Draw(sheet)
            font = ImageFont.truetype("C:/Windows/Fonts/consola.ttf", 15)
            for j, row in enumerate(subset):
                col, line = j%3, j//3
                b = row["bbox_top_left"]
                x1, y1, x2, y2 = max(0,int(b[0])-65), max(0,int(b[1])-70), min(w,int(b[2])+65), min(h,int(b[3])+55)
                crop = im.crop((x1,y1,x2,y2))
                cd = ImageDraw.Draw(crop)
                cd.rectangle([b[0]-x1,b[1]-y1,b[2]-x1,b[3]-y1], outline="orange", width=2)
                crop.thumbnail((468,220))
                if crop.width < 230 and crop.height < 110:
                    crop = crop.resize((crop.width*2,crop.height*2))
                sheet.paste(crop,(col*480+5,line*270+38))
                label = " ".join(f"{k}={v}" for k,v in row["fields"].items())
                draw.text((col*480+5,line*270+5),label,fill="black",font=font)
            out = ROOT/f"runs/s3_component_text_quick_dev/review/{case}_{page+1}.png"
            out.parent.mkdir(parents=True,exist_ok=True)
            sheet.save(out)
    out = ROOT/"data/reviews/component_text_quick_dev.json"
    out.parent.mkdir(parents=True,exist_ok=True)
    result = {"status": "manual_local_reference_not_official_Golden", "case_ids": ["0001","0002"],
              "review_method": "Visual reading of supplied PNG; source boxes reused after visual localization review; strict text spelling",
              "scope": "Physical part references, primary values and models; power/net/pin labels excluded",
              "split": "already_used_development_validation_not_independent_test",
              "field_count": sum(len(r["fields"]) for c in references.values() for r in c["components"]),
              "cases": references}
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    # High magnification for particularly ambiguous glyphs.
    im = Image.open(ROOT/"data/standardized/0001/0001.png")
    regions = [("L1",(275,227,362,290)),("L2",(918,65,1020,111)),("Q1",(1177,173,1278,212)),("U4",(1040,721,1165,767))]
    sheet = Image.new("RGB",(750,1000),"white")
    draw, y = ImageDraw.Draw(sheet), 0
    for title, box in regions:
        patch=im.crop(box).resize(((box[2]-box[0])*4,(box[3]-box[1])*4))
        draw.text((0,y),title,fill="black")
        sheet.paste(patch,(65,y))
        y += patch.height+10
    sheet.crop((0,0,750,y)).save(ROOT/"runs/s3_component_text_quick_dev/manual_review_details.png")
    print(json.dumps({"fields":result["field_count"],"components":sum(len(c["components"]) for c in references.values())}))


if __name__ == "__main__":
    main()
