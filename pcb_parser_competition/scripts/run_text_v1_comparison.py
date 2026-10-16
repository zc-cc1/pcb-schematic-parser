"""Reproduce the archived v1 binder on the same frozen detections and base OCR."""
import argparse
import importlib.util
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--run",type=Path,required=True)
    p.add_argument("--detector-run",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--case-id",action="append",required=True)
    args=p.parse_args()
    archived=ROOT/"runs/s3_component_text_v1_archive"
    cfg=json.loads((archived/"config.json").read_text(encoding="utf-8"))
    spec=importlib.util.spec_from_file_location("archived_component_text_v1",archived/"component_text.py")
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for case in args.case_id:
        base=json.loads((args.run/case/"base_texts.json").read_text(encoding="utf-8"))
        det=json.loads((args.detector_run/case/"cpu/fused/detections.json").read_text(encoding="utf-8"))
        if base["image_sha256"]!=det["image_sha256"]:
            raise ValueError("Base OCR and frozen detections have different image hashes")
        output,records,debug=module.bind_components(det["detections"],base["texts"],*det["size"],cfg)
        out=args.output/case
        out.mkdir(parents=True,exist_ok=True)
        for name,value in [("component_records.json",{"image_sha256":base["image_sha256"],"records":records}),
                           ("components_partial.json",output),("binding_debug.json",debug)]:
            (out/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding="utf-8")
        print(json.dumps({"case_id":case,**debug["counts"]},ensure_ascii=False))


if __name__=="__main__":main()
