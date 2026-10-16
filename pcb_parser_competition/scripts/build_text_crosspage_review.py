"""Offline visual transcription for four additional development regression pages."""
import hashlib
import json
from pathlib import Path

from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parents[1]


def main():
    specifications={
        "0007": {
            "C_1":("switch",{"designator":"S7"}),"C_2":("switch",{"designator":"S3"}),
            "C_3":("switch",{"designator":"S4"}),"C_4":("switch",{"designator":"S5"}),
            "C_5":("switch",{"designator":"S10"}),"C_6":("switch",{"designator":"S8"}),
            "C_7":("switch",{"designator":"S2"}),"C_8":("switch",{"designator":"S6"}),
            "C_9":("switch",{"designator":"S10"}),"C_11":("switch",{"designator":"S1"}),
            "C_16":("switch",{"designator":"S9"}),"C_17":("switch",{"designator":"S0"}),
            "C_10":("capacitor",{"value":"30p"}),"C_19":("capacitor",{"value":"30p"}),
            "C_38":("capacitor",{"value":"10uf"}),"C_15":("resistor",{"value":"10 K"}),
            "C_13":("crystal",{"value":"12MHZ"})},
        "0042":{
            "C_6":("resistor",{"designator":"R23","value":"100K"}),
            "C_5":("resistor",{"designator":"R24","value":"100K"}),
            "C_3":("resistor",{"designator":"R25","value":"100K"}),
            "C_4":("resistor",{"designator":"R26","value":"100K"}),
            "C_11":("resistor",{"designator":"R27","value":"100K"}),
            "C_9":("resistor",{"designator":"R28","value":"0"})},
        "0131":{
            "C_2":("resistor",{"designator":"R5","value":"270kΩ"}),
            "C_3":("resistor",{"designator":"R42","value":"4.7kΩ"}),
            "C_4":("ic",{"designator":"U5","Name":"MT3520B"}),
            "C_5":("resistor",{"designator":"R3","value":"10kΩ"}),
            "C_6":("capacitor",{"designator":"C7","value":"22pF"}),
            "C_7":("resistor",{"designator":"R4","value":"49.9kΩ"}),
            "C_8":("capacitor",{"designator":"C8","value":"22uF"}),
            "C_10":("capacitor",{"designator":"C3","value":"22uF"}),
            "C_11":("capacitor",{"designator":"C5","value":"22uF"}),
            "C_15":("inductor",{"designator":"L1","value":"2.2uH"}),
            "C_16":("vbus_in",{"designator":"TP3"}),
            "C_17":("led",{"designator":"LED1","Name":"19-217/R6C-AL1M2VY/3T"})},
        "0229":{
            "C_2":("ic",{"Name":"TMP118"}),"C_7":("capacitor",{"value":"10nF"})}
    }
    manifest={r["case_id"]:r for r in [json.loads(s) for s in (ROOT/"data/detection_v1/manifest.jsonl").read_text(encoding="utf-8").splitlines()]}
    result={"status":"visually_reviewed_local_development_reference_not_official_Golden",
            "split":"four_additional_development_regression_pages_not_independent_engineering_test",
            "review_notes":["0007 has two physical switches both printed S10; both enter coverage denominators although export IDs must be unique.",
                            "0007 IC has conflicting external labels STC89C51/52 and 89C52RC; model excluded before evaluation.",
                            "0131 L1 value is visually 2.2uH, correcting the derived reference's 22uH for this separate review only.",
                            "0229 contains no numerical part reference; evaluate model/value without inventing one."],"cases":{}}
    for case,items in specifications.items():
        image_path=ROOT/f"data/standardized/{case}/{case}.png"
        im=Image.open(image_path).convert("RGB")
        objects={o["component_id"]:o for o in manifest[case]["objects"]}
        rows=[]
        for cid,(kind,fields) in items.items():
            rows.append({"designator":fields.get("designator",f"UNNUMBERED_{cid}"),"review_id":cid,
                         "type":kind,"bbox_top_left":objects[cid]["bbox"],"fields":fields,
                         "review_status":"visually_transcribed_from_PNG"})
        result["cases"][case]={"image_sha256":hashlib.sha256(image_path.read_bytes()).hexdigest(),"size":list(im.size),"components":rows}
        # Magnified contextual review cards, independent of predictions.
        for page in range((len(rows)+8)//9):
            subset=rows[page*9:(page+1)*9]
            sheet=Image.new("RGB",(1500,((len(subset)+2)//3)*310),"white")
            draw=ImageDraw.Draw(sheet)
            font=ImageFont.truetype("C:/Windows/Fonts/consola.ttf",15)
            for j,row in enumerate(subset):
                b=row["bbox_top_left"]
                x1,y1=max(0,int(b[0])-45),max(0,int(b[1])-45)
                x2,y2=min(im.width,int(b[2])+85),min(im.height,int(b[3])+45)
                patch=im.crop((x1,y1,x2,y2))
                pd=ImageDraw.Draw(patch)
                pd.rectangle([b[0]-x1,b[1]-y1,b[2]-x1,b[3]-y1],outline="orange",width=1)
                patch=patch.resize((patch.width*2,patch.height*2))
                patch.thumbnail((490,265))
                col,line=j%3,j//3
                sheet.paste(patch,(col*500+5,line*310+40))
                draw.text((col*500+5,line*310+5)," ".join(f"{k}={v}" for k,v in row["fields"].items()),font=font,fill="black")
            out=ROOT/f"runs/s3_component_text_crosspage_v2/review/{case}_{page+1}.png"
            out.parent.mkdir(parents=True,exist_ok=True)
            sheet.save(out)
    result["field_count"]=sum(len(r["fields"]) for c in result["cases"].values() for r in c["components"])
    path=ROOT/"data/reviews/component_text_crosspage.json"
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"cases":len(result["cases"]),"fields":result["field_count"]}))


if __name__=="__main__":main()
