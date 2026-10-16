"""Check deployed U-Net code, dataset paths and environment without GPU training."""
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from importlib import metadata
from PIL import Image
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from pcb_parser.unet_model import WirePinUNet


def main():
    cfg=json.loads((ROOT/"configs/unet_wire_pin.json").read_text(encoding="utf-8"))
    provenance=ROOT/"src/pcb_parser/vendor/pytorch_unet"
    source=json.loads((provenance/"provenance.json").read_text(encoding="utf-8"))
    assert source["commit"]==cfg["source_commit"]
    for row in source["files"]:
        assert hashlib.sha256((provenance/row["local"]).read_bytes()).hexdigest()==row["sha256"]
    rows=list(map(json.loads,(ROOT/"data/segmentation_unet_v1/manifest.jsonl").read_text(encoding="utf-8").splitlines()))
    for row in rows:
        for key in ["image","wire_mask","pin_heatmap"]:
            path=Path(row[key]).resolve()
            assert path.is_relative_to(ROOT), f"External training data dependency: {path}"
            with Image.open(path) as im:
                assert im.size==(row["width"],row["height"]),f"Size mismatch: {path}"
    model=WirePinUNet(cfg["base_channels"])
    eligible=[r for r in rows if r["structurally_usable_draft"]]
    assert {"train","val"}<=set(r["split"] for r in eligible)
    result={"status":"passed","project_root":str(ROOT),"python":sys.executable,
            "versions":{name:metadata.version(name) for name in ["torch","numpy","Pillow"]},
            "source_commit":source["commit"],"parameters":sum(p.numel() for p in model.parameters()),
            "cases":len(rows),"structurally_usable_drafts":len(eligible),
            "usable_by_split":dict(Counter(r["split"] for r in eligible)),
            "dataset_paths_local_to_project":True,"unreviewed_labels_are_draft":True,
            "cuda_available":torch.cuda.is_available(),"formal_training_started":False}
    output=ROOT/"runs/unet_transfer_verification/runtime.json"
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False))


if __name__=="__main__":
    main()
