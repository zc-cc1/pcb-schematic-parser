"""Read difficult text again from its image crop; never consult reference labels."""
import math
import re
import time

import cv2
import numpy as np

from .component_text import bbox, iou, merge_texts, tokens_from_texts


def recovery_candidates(texts, cfg):
    tokens=tokens_from_texts(texts)
    selected=[]
    for t in tokens:
        reason=None
        if t["role"]=="designator" and (t["confidence"]<cfg.get("ref_confidence",.96) or t["text"][0].islower()):
            reason="low_confidence_or_case_alternative"
        elif t["role"] in {"inductance","capacitance","resistance","frequency"} and t["confidence"]<cfg.get("value_confidence",.97):
            reason="low_confidence_value"
        elif re.fullmatch(r"[O0](?:IC|SW|TP|FB|FET|LED|JP|[RCLDQUXYJPFS])\d+",t["text"],re.I):
            reason="possible_symbol_attached_to_reference"
        elif re.search(r"0hm$",t["text"]):
            reason="ambiguous_unit_spelling"
        if reason:
            if any(iou(t["bbox"],other["bbox"])>.8 for other in selected):
                continue
            selected.append({**t,"recovery_reason":reason})
    return selected[:cfg.get("max_rois",32)]


def color_variants(patch):
    output={"rgb":patch,"gray":cv2.cvtColor(cv2.cvtColor(patch,cv2.COLOR_RGB2GRAY),cv2.COLOR_GRAY2RGB)}
    hsv=cv2.cvtColor(patch,cv2.COLOR_RGB2HSV)
    chromatic=(hsv[:,:,1]>60)&(hsv[:,:,2]<250)
    candidates=[]
    for hue in [0,30,60,90,120,150]:
        difference=abs(hsv[:,:,0].astype(int)-hue)
        distance=np.minimum(difference,180-difference)
        keep=(distance<12)&chromatic
        if keep.sum()>=12:
            candidates.append((int(keep.sum()),hue,keep))
    for _,hue,keep in sorted(candidates,key=lambda x:-x[0])[:2]:
        isolated=patch.copy()
        isolated[~keep]=255
        output[f"hue_{hue}"]=isolated
    return output


def recover_texts(rgb, engine, texts, cfg):
    h,w=rgb.shape[:2]
    observations=[]
    for row in texts:
        for observed in row.get("observations",[]):
            b=observed.get("bbox",row.get("bbox",bbox(row["polygon"])))
            x1,y1,x2,y2=b
            observations.append({"text":observed["text"],"confidence":observed["confidence"],
                                 "polygon":[[x1,y1],[x2,y1],[x2,y2],[x1,y2]],"view":observed["view"]})
        if not row.get("observations"):
            observations.append(row)
    candidates=recovery_candidates(texts,cfg)
    records=[]
    started=time.perf_counter()
    for i,t in enumerate(candidates):
        b=t["bbox"]
        padding=cfg.get("padding",1)
        x1,y1=max(0,math.floor(b[0])-padding),max(0,math.floor(b[1])-padding)
        x2,y2=min(w,math.ceil(b[2])+padding+1),min(h,math.ceil(b[3])+padding+1)
        if x2<=x1 or y2<=y1:
            continue
        patch=rgb[y1:y2,x1:x2]
        variants=color_variants(patch)
        if t["recovery_reason"]=="possible_symbol_attached_to_reference":
            offset=max(1,int(patch.shape[1]/len(t["text"])))
            if offset<patch.shape[1]-2:
                variants["suffix_crop"]=patch[:,offset:]
        for variant, pixels in variants.items():
            for interpolation in [cv2.INTER_NEAREST,cv2.INTER_CUBIC]:
                enlarged=cv2.resize(pixels,None,fx=cfg.get("scale",4),fy=cfg.get("scale",4),interpolation=interpolation)
                result=engine(np.ascontiguousarray(enlarged[:,:,::-1]),use_det=False,use_cls=False,use_rec=True)
                for text,confidence in zip(result.txts,result.scores):
                    record={"source_token":t["text"],"source_bbox":b,"crop":[x1,y1,x2,y2],
                            "reason":t["recovery_reason"],"variant":variant,"interpolation":int(interpolation),
                            "text":str(text),"confidence":float(confidence)}
                    records.append(record)
                    if float(confidence)<cfg.get("min_confidence",.75) or not str(text).strip():
                        continue
                    bx1,by1,bx2,by2=b
                    observations.append({"text":str(text),"confidence":float(confidence),
                                         "polygon":[[bx1,by1],[bx2,by1],[bx2,by2],[bx1,by2]],
                                         "view":f"recovery_{i}_{variant}_{interpolation}"})
    return merge_texts(observations), {"roi_count":len(candidates),"observations":records,
                                      "wall_seconds":time.perf_counter()-started,
                                      "policy":"image rerecognition only; raw strings retained; no annotation or character rewrite"}
