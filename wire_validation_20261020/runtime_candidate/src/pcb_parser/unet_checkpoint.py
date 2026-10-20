"""Single entry point for loading a U-Net checkpoint, whatever its architecture.

The U-Net went through two architectures (shared decoder = `WirePinUNet`,
split decoder = `SplitWirePinUNet`) and several supervision generations.  Every
consumer used to hard-code `WirePinUNet` plus one weight path, which is exactly
how a frozen pipeline silently keeps running old weights.  This module makes the
checkpoint self-describing:

    load_wire_pin_checkpoint(path, device)  ->  (model, checkpoint, info)

`info` records the architecture, the inference-relevant fields and the SHA256 so
downstream stages can hash-bind their caches to the actual model they used.
"""
import hashlib
from pathlib import Path

import torch

from .unet_model import WirePinUNet
from .unet_split_model import SplitWirePinUNet

ARCHITECTURES = {"shared": WirePinUNet, "split": SplitWirePinUNet}


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_wire_pin_checkpoint(path, device="cpu", allow_debug=False):
    path = Path(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    kind = checkpoint.get("kind", "unknown")
    if not allow_debug and kind in {"cpu_smoke_only", "partial_training_debug"}:
        raise ValueError(f"debug checkpoint cannot be used for inference: {kind}")
    variant = checkpoint.get("model_variant", "shared")
    if variant not in ARCHITECTURES:
        raise ValueError(f"unknown model_variant {variant!r}")
    model = ARCHITECTURES[variant](checkpoint["base_channels"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device).eval()
    info = {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "epoch": checkpoint.get("epoch"),
        "kind": kind,
        "model_variant": variant,
        "pin_target": checkpoint.get("pin_target", "pin_heatmap"),
        "parameters": sum(v.numel() for v in model.parameters()),
        "init": checkpoint.get("init"),
    }
    return model, checkpoint, info
