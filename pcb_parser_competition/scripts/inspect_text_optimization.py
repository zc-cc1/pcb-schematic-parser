"""Read-only diagnostics for frozen detections and OCR candidates."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    run=ROOT/"runs/s3_component_text_quick_dev/0001"
    debug=json.loads((run/"binding_debug.json").read_text(encoding="utf-8"))
    dets=json.loads((ROOT/"runs/s3_v3_detector_quick_dev/0001/cpu/fused/detections.json").read_text(encoding="utf-8"))["detections"]
    interesting=[i for i,d in enumerate(dets) if d["class_name"] in {"resistor","inductor","pin_port","net_input"}]
    print("DETECTIONS",json.dumps({i:dets[i] for i in interesting},ensure_ascii=False))
    print("REJECTED",json.dumps(debug["rejected_matches"],ensure_ascii=False))
    regions={"dense_resistors":[1100,150,1175,220],"coil":[280,232,350,286],
             "top_coil":[920,60,1020,136],"ports":[1215,106,1290,175],"ic_ref":[1048,74,1090,112]}
    texts=json.loads((run/"texts.json").read_text(encoding="utf-8"))["texts"]
    for name,(x1,y1,x2,y2) in regions.items():
        rows=[t for t in texts if x1<t["bbox"][0]<x2 and y1<t["bbox"][1]<y2]
        print(name,json.dumps(rows,ensure_ascii=False))
    print("TOKENS",json.dumps([t for t in debug["tokens"] if t["role"]=="designator" or t["role"]=="inductance"],ensure_ascii=False))


if __name__=="__main__":main()
