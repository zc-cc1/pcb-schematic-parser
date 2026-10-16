"""PNG and frozen U-Net weights -> two-point wire candidates, no label reads."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image, ImageDraw
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from pcb_parser.unet_data import predict_tiled
from pcb_parser.unet_inference import decode_pins
from pcb_parser.unet_model import WirePinUNet
from pcb_parser.wire_segments import decode_wire_segments, export_candidate_edges, component_bridge_guard
from pcb_parser.wire_segment_nodes import refine_segment_nodes


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path, data): Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT/"configs/wire_segments_v1.json")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--components", type=Path, help="Optional image-hash-verified component model predictions")
    parser.add_argument("--node-config", type=Path, help="Optional predicted-tip/node refinement; acceptance is recorded in its config")
    args = parser.parse_args()
    tick = time.perf_counter(); torch.set_num_threads(4)
    cfg = json.loads(args.config.read_text(encoding="utf-8-sig"))
    checkpoint = torch.load(args.weights, map_location="cpu", weights_only=True)
    if checkpoint.get("kind") in {"cpu_smoke_only", "partial_training_debug"}:
        raise ValueError("Debug checkpoint cannot be used as a trained geometry predictor")
    model = WirePinUNet(checkpoint["base_channels"])
    model.load_state_dict(checkpoint["state_dict"], strict=True); model.to(args.device).eval()
    with Image.open(args.image) as image:
        rgb = np.asarray(image.convert("RGB"))
    model_cfg = checkpoint["config"]
    probability = predict_tiled(model, rgb, args.device, model_cfg["tile_size"], model_cfg["stride"])
    pins = decode_pins(probability[1], threshold=.8, kernel=5, refinement="quadratic", nms_radius=0, limit=500)
    guard = None
    if args.components:
        components = json.loads(args.components.read_text(encoding="utf-8-sig"))
        if components.get("coordinate_origin") != "top_left" or components.get("size") != list(rgb.shape[1::-1]) or components.get("image_sha256") != sha(args.image):
            raise ValueError("Component prediction provenance differs from the PNG")
        guard = component_bridge_guard(rgb.shape[:2], components["components"])
    effective = dict(cfg)
    if guard is None:
        # Without symbol-body evidence keep fitted segments, but do not bridge a possible component.
        effective["maximum_gap_px"] = 0
    result = decode_wire_segments(probability[0], effective, pin_points=[p["point"] for p in pins], bridge_blocked_mask=guard)
    if args.node_config:
        node_cfg = json.loads(args.node_config.read_text(encoding="utf-8-sig"))
        result = refine_segment_nodes(result, pins, node_cfg, bridge_guard=guard)
    provenance = dict(image_sha256=sha(args.image), weights_sha256=sha(args.weights), config_sha256=sha(args.config),
                      inputs="PNG + trained checkpoint + fixed decoder config + optional component model predictions", prediction_only=True,
                      annotations_read=False, coordinate_origin="top_left",
                      component_predictions_sha256=sha(args.components) if args.components else None,
                      node_config_sha256=sha(args.node_config) if args.node_config else None,
                      gap_merging_disabled_without_component_predictions=guard is None)
    result.update(provenance)
    args.output.mkdir(parents=True, exist_ok=True)
    write(args.output/"segments_top_left.json", result)
    write(args.output/"candidate_edges_bottom_left.json", export_candidate_edges(result))
    write(args.output/"pin_candidates.json", pins)
    np.savez_compressed(args.output/"probabilities.npz", wire=probability[0], pin_tip=probability[1])
    Image.fromarray((probability[0]>=cfg["wire_threshold"]).astype(np.uint8)*255).save(args.output/"wire_mask.png")
    overlay = Image.fromarray(rgb); draw = ImageDraw.Draw(overlay)
    for segment in result["segments"]:
        draw.line([tuple(p) for p in segment["points"]], fill=(0, 170, 95), width=2)
        for x, y in segment["points"]:
            draw.ellipse((x-1.5, y-1.5, x+1.5, y+1.5), fill=(215, 70, 30))
    overlay.save(args.output/"segments_overlay.png")
    summary = provenance | dict(status=result["status"], geometry=result["summary"],
        seconds=time.perf_counter()-tick, original_size=result["size"], official_NetLineF1=None,
        complete_competition_JSON=False, default_weights_changed=False)
    write(args.output/"summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
