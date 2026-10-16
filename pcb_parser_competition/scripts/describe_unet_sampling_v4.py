"""Describe actual paired draw distributions without reading model outcomes."""
from collections import Counter
import json
import numpy as np
from unet_audit_common import ROOT, read, write
from pcb_parser.unet_negative_sampling import NegativeFocusTrainingTiles


def main():
    review=ROOT/"runs/unet_training_label_review_v4"
    rows=[json.loads(s) for s in (review/"training_manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    train=[r for r in rows if r["split"]=="train"]
    regions=read(review/"confirmed_negative_regions.json")["regions"]
    dataset=NegativeFocusTrainingTiles(train,regions,.25)
    report=[];schedules={}
    for name,fraction in (("control",0.),("negative_focus",.25)):
        dataset.focus_fraction=fraction;dataset.set_epoch(1);schedules[name]=list(dataset.tiles)
        sizes=positive=empty=0
        for i,x,y in dataset.tiles:
            _,wire,_=dataset.source(i);crop=wire[y:y+512,x:x+512]
            sizes+=crop.size;positive+=int((crop>0).sum());empty+=not crop.any()
        report.append(dict(arm=name,valid_pixels=sizes,wire_positive_pixels=positive,positive_fraction=positive/sizes,
            empty_GT_tiles=empty,draws=len(dataset.tiles),case_counts=dict(Counter(train[i]["case_id"] for i,x,y in dataset.tiles))))
    result=dict(epoch=1,records=report,actual_changed_tiles=sum(a!=b for a,b in zip(schedules["control"],schedules["negative_focus"])),
        nominal_focus_draws=round(.25*len(dataset.tiles)),focus_entries=len(dataset.focus_tiles),unique_focus_tiles=len(set(dataset.focus_tiles)),
        focus_classes=dict(Counter(r["kind"] for r in dataset.focus_regions)),
        scope="925 draws per arm; overlap counted as training exposure; no labels forced background; not independent pixels",
        causal_limit="Crop resampling changes image/style/context and tile-origin phase distribution as part of the sampling intervention; gains cannot uniquely identify semantic negatives as the sole mechanism")
    write(review/"actual_sampling_distribution.json",result)
    print(json.dumps({k:v for k,v in result.items() if k!="records"}));print(json.dumps([{k:v for k,v in r.items() if k!="case_counts"} for r in report]))


if __name__=="__main__":main()
