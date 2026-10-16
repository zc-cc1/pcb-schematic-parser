"""Lossless whole-page 1px grid/context sensitivity; no parameter selection."""
import json
import time
import numpy as np
import torch
from unet_audit_common import ROOT, OUT, read, write, image, truth, probability, counts, metrics, aggregate, val_cases, assert_protected
from audit_unet_inference import load_model, tiled, cached
from pcb_parser.dataset_converter import load_source_annotation
from verify_unet_contrast import path_specs, path_records


def main():
    start=time.perf_counter();assert_protected()
    protocol=dict(predeclared_shifts_xy=[[0,1],[1,0],[1,1]],mean="equal probability mean of identity and3 shifts",
        transform="prepend white top/left rows/columns; preserve every original pixel; crop inverse output; no interpolation",
        caveat="changes pooling phase AND tile anchoring/padding context; cannot attribute all differences solely to first maxpool",
        validation_fitting=False,training_started=False,adopting_result=False)
    write(OUT/"phase_protocol.json",protocol)
    torch.set_num_threads(4);torch.manual_seed(20261014);model=load_model("v1")
    raw={r["case_id"]:r for r in map(json.loads,(ROOT/"data/manifest.jsonl").read_text(encoding="utf-8").splitlines())}
    allrecords=[]
    for case in val_cases():
        rgb=image(case);h,w=rgb.shape[:2];base=probability(case);variants={}
        for dx,dy in protocol["predeclared_shifts_xy"]:
            shifted=np.pad(rgb,((dy,0),(dx,0),(0,0)),constant_values=255)
            assert np.array_equal(shifted[dy:dy+h,dx:dx+w],rgb)
            name=f"shift_x{dx}_y{dy}"
            variants[name]=cached(case,name,lambda:tiled(model,shifted)[dy:dy+h,dx:dx+w])
        variants["phase_mean4"]=np.mean([base,*variants.values()],axis=0,dtype=np.float32)
        gt=truth(case);annotation,_=load_source_annotation(raw[case]["raw_annotation_path"]);specs=path_specs(annotation,gt.shape)
        records=[]
        for name,p in variants.items():
            paths=path_records(p>=.5,specs)
            r=dict(case_id=case,variant=name,wire=metrics(*counts(p>=.5,gt)),connected_paths=sum(v["connected_within_reference_tube"] for v in paths),
                total_paths=len(paths),changed_prediction_pixels=int(((p>=.5)!=(base>=.5)).sum()),
                threshold_curve=[dict(threshold=t,wire=metrics(*counts(p>=t,gt))) for t in read(OUT/"protocol.json")["thresholds"]])
            if case=="0036":r["local_rois"]={label:dict(median=float(np.median(p[y,x0:x1])),coverage_pixels=int((p[y,x0:x1]>=.5).sum()),length=x1-x0)
                for label,y,x0,x1 in (("X1_left",80,620,681),("DOUT",181,605,629))}
            records.append(r)
        write(OUT/"inference"/case/"phase_metrics.json",dict(records=records,protocol=protocol));allrecords+=records
    base_result=read(OUT/"inference_audit.json");summaries=[];baseline=next(r for r in base_result["summary"] if r["variant"]=="native_uniform512")
    for name in variants:
        rows=[r for r in allrecords if r["variant"]==name];micro=aggregate(rows);curves=[]
        for i,t in enumerate(read(OUT/"protocol.json")["thresholds"]):
            values=[r["threshold_curve"][i] for r in rows]
            curves.append(dict(threshold=t,micro=aggregate(values),macro_dice=float(np.mean([r["wire"]["dice"] for r in values]))))
        summaries.append(dict(variant=name,pages=26,micro=micro,macro_dice=float(np.mean([r["wire"]["dice"] for r in rows])),
            delta_dice_pp=100*(micro["dice"]-baseline["micro"]["dice"]),connected_paths=sum(r["connected_paths"] for r in rows),total_paths=sum(r["total_paths"] for r in rows),
            threshold_curve=curves,diagnostic_best=max(curves,key=lambda r:r["micro"]["dice"]),adopted=False,
            changed_prediction_pixels=sum(r["changed_prediction_pixels"] for r in rows),
            improved_pages=sum(r["wire"]["dice"]>next(b["wire"]["dice"] for b in base_result["records"] if b["variant"]=="native_uniform512" and b["case_id"]==r["case_id"]) for r in rows)))
    existing_names=set(variants)
    base_result["summary"]=[r for r in base_result["summary"] if r["variant"] not in existing_names]+summaries
    base_result["records"]=[r for r in base_result["records"] if r["variant"] not in existing_names]+allrecords
    base_result["phase_addendum_elapsed_seconds"]=time.perf_counter()-start;base_result["protected_files"]=assert_protected()
    write(OUT/"inference_audit.json",base_result)
    write(OUT/"phase_audit.json",dict(summary=summaries,records=allrecords,protocol=protocol,elapsed_seconds=time.perf_counter()-start))
    print(json.dumps([{k:v for k,v in r.items() if k not in ("threshold_curve","diagnostic_best")} for r in summaries]))


if __name__=="__main__":main()
