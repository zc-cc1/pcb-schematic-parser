"""Image-only local OCR experiments; no paired annotations are read."""
import json
import sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from rapidocr import RapidOCR

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from pcb_parser.component_text import tokens_from_texts


def main():
    rgb=np.asarray(Image.open(ROOT/"data/standardized/0001/0001.png").convert("RGB"))
    texts=json.loads((ROOT/"runs/s3_component_text_quick_dev/0001/texts.json").read_text(encoding="utf-8"))["texts"]
    tokens=tokens_from_texts(texts)
    candidates=[t for t in tokens if (t["role"]=="designator" and t["text"][:1].islower())
                or t["role"]=="inductance" or t["text"].endswith("0hm")]
    engine=RapidOCR()
    outputs=[]
    for token in candidates:
        b=token["bbox"]
        x1,y1,x2,y2=max(0,int(b[0])-1),max(0,int(b[1])-1),min(rgb.shape[1],int(b[2])+2),min(rgb.shape[0],int(b[3])+2)
        patch=rgb[y1:y2,x1:x2]
        variants={"rgb":patch,"gray":cv2.cvtColor(cv2.cvtColor(patch,cv2.COLOR_RGB2GRAY),cv2.COLOR_GRAY2RGB)}
        hsv=cv2.cvtColor(patch,cv2.COLOR_RGB2HSV)
        for hue in [0,30,60,90,120,150]:
            distance=np.minimum(abs(hsv[:,:,0].astype(int)-hue),180-abs(hsv[:,:,0].astype(int)-hue))
            keep=(distance<12)&(hsv[:,:,1]>60)&(hsv[:,:,2]<250)
            if keep.sum()>=12:
                colored=patch.copy()
                colored[~keep]=255
                variants[f"hue_{hue}"]=colored
        rows=[]
        for name,variant in variants.items():
            for interpolation in [cv2.INTER_NEAREST,cv2.INTER_CUBIC]:
                up=cv2.resize(variant,None,fx=4,fy=4,interpolation=interpolation)
                result=engine(np.ascontiguousarray(up[:,:,::-1]),use_det=False,use_cls=False,use_rec=True)
                rows.append({"variant":name,"interpolation":int(interpolation),"texts":list(result.txts),"scores":[float(s) for s in result.scores]})
        outputs.append({"original":token["text"],"crop":[x1,y1,x2,y2],"source":token["source_text"],"results":rows})
    path=ROOT/"runs/s3_component_text_v2_dev/local_ocr_probe.json"
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(outputs,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(outputs,ensure_ascii=False))


if __name__=="__main__":main()
