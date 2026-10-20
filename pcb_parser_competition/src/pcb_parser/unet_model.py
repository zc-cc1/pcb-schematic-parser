"""PCB adapter around the pinned upstream U-Net building blocks (GPL-3.0)."""
import torch
from torch import nn
from .vendor.pytorch_unet.unet_parts import DoubleConv, Down, Up, OutConv


class WirePinUNet(nn.Module):
    """Shared U-Net with independent wire-mask and pin-tip heatmap channels."""
    def __init__(self, base_channels=32):
        super().__init__()
        b = base_channels
        self.inc = DoubleConv(3, b)
        self.down1, self.down2 = Down(b, b*2), Down(b*2, b*4)
        self.down3, self.down4 = Down(b*4, b*8), Down(b*8, b*8)
        self.up1, self.up2 = Up(b*16, b*4), Up(b*8, b*2)
        self.up3, self.up4 = Up(b*4, b), Up(b*2, b)
        self.out = OutConv(b, 2)

    def forward(self, image):
        a = self.inc(image)
        b = self.down1(a)
        c = self.down2(b)
        d = self.down3(c)
        e = self.down4(d)
        x = self.up4(self.up3(self.up2(self.up1(e, d), c), b), a)
        return self.out(x)


def segmentation_loss(logits, targets, valid, wire_positive_weight=8., pin_positive_weight=20., pin_loss_weight=.5):
    """Mask padding/unknown pixels; wire BCE+Dice, pin Gaussian weighted BCE.

    `pin_loss_weight` exists because the two tasks no longer share one schedule:
    v1/v2 hard-coded 0.5x pin BCE, which leaves the pin head under-trained once
    the wire task starts converging.  The default keeps old behaviour bit-exact.
    """
    if not bool(valid.any()):
        raise ValueError("No supervised pixels in batch")
    bce = nn.functional.binary_cross_entropy_with_logits
    losses = []
    for channel, positive_weight in enumerate([wire_positive_weight, pin_positive_weight]):
        pixel = bce(logits[:,channel], targets[:,channel], reduction="none",
                    pos_weight=logits.new_tensor(positive_weight))
        losses.append((pixel*valid).sum()/valid.sum().clamp_min(1))
    probability = logits[:,0].sigmoid()*valid
    truth = targets[:,0]*valid
    dice = 1-(2*(probability*truth).sum()+1)/(probability.sum()+truth.sum()+1)
    return losses[0]+dice+pin_loss_weight*losses[1], {"wire_bce": losses[0].detach(),
                                      "wire_dice_loss": dice.detach(), "pin_bce": losses[1].detach()}
