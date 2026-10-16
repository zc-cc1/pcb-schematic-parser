"""PNG-only learned wire/pin prediction; no threshold/colour image segmentation."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from pcb_parser.unet_data import predict_tiled
from pcb_parser.unet_model import WirePinUNet


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--image",type=Path,required=True)
    parser.add_argument("--weights",type=Path,required=True)
    parser.add_argument("--device",default="cpu")
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--allow-smoke",action="store_true")
    args=parser.parse_args()
    checkpoint=torch.load(args.weights,map_location="cpu",weights_only=True)
    debug_checkpoint=checkpoint["kind"] in {"cpu_smoke_only","partial_training_debug"}
    if debug_checkpoint and not args.allow_smoke:
        raise ValueError("Smoke checkpoint has no recognition-quality validation; use --allow-smoke only for pipeline debugging")
    torch.set_num_threads(4)
    model=WirePinUNet(checkpoint["base_channels"])
    model.load_state_dict(checkpoint["state_dict"])
    model.to(args.device)
    with Image.open(args.image) as im:
        rgb=np.asarray(im.convert("RGB"))
    cfg=checkpoint["config"]
    probability=predict_tiled(model,rgb,args.device,cfg["tile_size"],cfg["stride"])
    args.output.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output/"probabilities.npz",wire=probability[0],pin_tip=probability[1])
    Image.fromarray((probability[0]*255).astype(np.uint8)).save(args.output/"wire_probability.png")
    Image.fromarray((probability[1]*255).astype(np.uint8)).save(args.output/"pin_probability.png")
    Image.fromarray((probability[0]>=cfg["inference"]["wire_threshold"]).astype(np.uint8)*255).save(args.output/"wire_mask.png")
    # Peak extraction from predicted probability, not segmentation of raw pixels.
    heat=torch.from_numpy(probability[1])[None,None]
    peaks=(heat==torch.nn.functional.max_pool2d(heat,9,stride=1,padding=4))&(heat>=cfg["inference"]["pin_threshold"])
    ys,xs=torch.where(peaks[0,0])
    candidates=sorted([{"point":[int(x),int(y)],"confidence":float(heat[0,0,y,x]),"owner":None,"number":None,"pinname":None}
                       for y,x in zip(ys,xs)],key=lambda p:-p["confidence"])[:500]
    (args.output/"pin_candidates.json").write_text(json.dumps(candidates,indent=2),encoding="utf-8")
    summary={"status":"pipeline_debug_only" if debug_checkpoint else "unet_mask_and_tip_prediction_ownership_topology_pending",
             "checkpoint_kind":checkpoint["kind"],"input_contract":"PNG + U-Net checkpoint; no annotation read",
             "image_sha256":hashlib.sha256(args.image.read_bytes()).hexdigest(),
             "weights_sha256":hashlib.sha256(args.weights.read_bytes()).hexdigest(),
             "size":[rgb.shape[1],rgb.shape[0]],"candidate_pins":len(candidates),
             "traditional_image_segmentation_used":False,"pins_identity_and_nets_pending":True,"official_score":None}
    (args.output/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(summary,ensure_ascii=False))


if __name__=="__main__":
    main()
