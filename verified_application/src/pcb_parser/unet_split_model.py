"""Two-decoder variant: shared U-Net encoder, separate wire and pin decoders.

Why: in `WirePinUNet` both tasks share the whole decoder, so one 2-channel head
has to represent two spatially different targets (thin polylines vs. blobs at
component terminals).  The pin head's precision plateau (0.75) survives a decoder
sweep and a loss reweighting, which points at shared capacity rather than decoding.

Interface is deliberately identical to `WirePinUNet` (same `forward` signature and
same `base_channels` argument) so `train_unet_v2.py` and the evaluators can load
either class; `load_wire_pin_weights` warms a split model from a shared-decoder
checkpoint by copying the encoder and the 2-channel output head.
"""
import torch
from torch import nn

from .vendor.pytorch_unet.unet_parts import DoubleConv, Down, OutConv, Up


class SplitWirePinUNet(nn.Module):
    """Shared encoder; wire and pin each get their own 2-stage decoder."""

    def __init__(self, base_channels=32):
        super().__init__()
        b = base_channels
        # encoder (identical layout to WirePinUNet so weights transfer 1:1)
        self.inc = DoubleConv(3, b)
        self.down1, self.down2 = Down(b, b * 2), Down(b * 2, b * 4)
        self.down3, self.down4 = Down(b * 4, b * 8), Down(b * 8, b * 8)
        # wire decoder
        self.wire_up1, self.wire_up2 = Up(b * 16, b * 4), Up(b * 8, b * 2)
        self.wire_up3, self.wire_up4 = Up(b * 4, b), Up(b * 2, b)
        self.wire_out = OutConv(b, 1)
        # pin decoder
        self.pin_up1, self.pin_up2 = Up(b * 16, b * 4), Up(b * 8, b * 2)
        self.pin_up3, self.pin_up4 = Up(b * 4, b), Up(b * 2, b)
        self.pin_out = OutConv(b, 1)

    def forward(self, image):
        a = self.inc(image)
        b = self.down1(a)
        c = self.down2(b)
        d = self.down3(c)
        e = self.down4(d)
        wire = self.wire_up4(self.wire_up3(self.wire_up2(self.wire_up1(e, d), c), b), a)
        pin = self.pin_up4(self.pin_up3(self.pin_up2(self.pin_up1(e, d), c), b), a)
        return torch.cat([self.wire_out(wire), self.pin_out(pin)], dim=1)


SPLIT_PREFIX = ("wire_up", "pin_up", "wire_out", "pin_out")


def load_shared_decoder_weights(model, state_dict):
    """Warm a split model from a shared-decoder checkpoint.

    The 2-channel OutConv becomes two 1-channel OutConvs; channel 0 of the old
    head goes to the wire branch and channel 1 to the pin branch, so neither
    task starts from random output filters.
    """
    shared = dict(state_dict)
    renamed = {}
    for key, value in shared.items():
        if key == "out.conv.weight":            # (2, b, 1, 1)
            renamed["wire_out.conv.weight"] = value[0:1].clone()
            renamed["pin_out.conv.weight"] = value[1:2].clone()
        elif key == "out.conv.bias":            # (2,)
            renamed["wire_out.conv.bias"] = value[0:1].clone()
            renamed["pin_out.conv.bias"] = value[1:2].clone()
        elif key.startswith("up"):
            renamed[f"wire_{key}"] = value.clone()
            renamed[f"pin_{key}"] = value.clone()
        else:
            renamed[key] = value.clone()
    missing, unexpected = model.load_state_dict(renamed, strict=False)
    return {"missing": list(missing), "unexpected": list(unexpected),
            "copied_encoder": sum(1 for k in renamed if k.startswith(("inc", "down"))),
            "copied_decoders": sum(1 for k in renamed if k.startswith(SPLIT_PREFIX))}
