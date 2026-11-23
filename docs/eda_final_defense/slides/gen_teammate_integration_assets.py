# -*- coding: utf-8 -*-
"""Assets for review/eda-final-defense-teammate-integration.
All imagery derives from real pipeline artifacts already in the workspace:
  fig1b/fig3b/fig3c/fig3d/fig2c/fig5 (assets_build), smoke/input page PNG, teammate port crop.
"""
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

WS = r'E:\AI_Workplace\精英挑战赛\_audit_workspace'
A = os.path.join(WS, 'assets_build')
D = os.path.join(WS, 'deck_workspace')
OUT = os.path.join(WS, 'deck_workspace')
os.makedirs(OUT, exist_ok=True)

FB = r'C:\Windows\Fonts\arialbd.ttf'


def up2(img, f=2):
    return img.resize((img.width * f, img.height * f), Image.LANCZOS)


# ---------- 1. P6 main image: fig3d U3 crop + three numbered step markers ----------
src = Image.open(os.path.join(A, 'fig3d_graph_networks_real.png')).convert('RGB')
BOX = (270, 50, 890, 545)
SX, SY = BOX[0], BOX[1]
SCALE = 2
MARKERS = [('1', (521, 323)), ('2', (402, 337)), ('3', (459, 346))]
crop = src.crop(BOX)
up = crop.resize((crop.width * SCALE, crop.height * SCALE), Image.LANCZOS)
d = ImageDraw.Draw(up)
F = ImageFont.truetype(FB, 34)
R = 22
for label, p in MARKERS:
    x, y = (p[0] - SX) * SCALE, (p[1] - SY) * SCALE
    d.ellipse([x - R - 4, y - R - 4, x + R + 4, y + R + 4], fill=(255, 255, 255))
    d.ellipse([x - R, y - R, x + R, y + R], fill=(18, 60, 109))
    bb = d.textbbox((0, 0), label, font=F)
    d.text((x - (bb[2] - bb[0]) / 2, y - (bb[3] - bb[1]) / 2 - 2), label, font=F, fill=(255, 255, 255))
up.save(os.path.join(OUT, 'ti_p6_main.png'))
print('ti_p6_main', up.size)

# ---------- 2. P5 minis ----------
page = Image.open(os.path.join(WS, 'smoke', 'input', 'rp2350_page11.png')).convert('RGB')
# input mini (X1/USB/U3 region)
mini = page.crop((250, 180, 700, 420))
up2(mini).save(os.path.join(OUT, 'ti_p5_input.png'))
print('ti_p5_input', up2(mini).size)

# YOLO evidence: real detection overlay crop (boxes + designators)
fig1b = Image.open(os.path.join(A, 'fig1b_detection_clean.png')).convert('RGB')
yolo = fig1b.crop((255, 28, 665, 215))
up2(yolo).save(os.path.join(OUT, 'ti_p5_yolo.png'))
print('ti_p5_yolo', up2(yolo).size)

# OCR evidence: real evidence-gated OCR figure crop (SW1 token with red box)
fig5 = Image.open(os.path.join(A, 'fig5_evidence_gated_ocr.png')).convert('RGB')
ocr = fig5.crop((165, 86, 400, 142))
up2(ocr).save(os.path.join(OUT, 'ti_p5_ocr.png'))
print('ti_p5_ocr', up2(ocr).size)

# U-Net evidence: wire mask + pin heatmap, same real region as input mini
f2c = Image.open(os.path.join(A, 'fig2c_unet_real_artifacts.png')).convert('RGB')
wire = f2c.crop((250, 910 + 180, 700, 910 + 420))
up2(wire).save(os.path.join(OUT, 'ti_p5_wire.png'))
pin = f2c.crop((250, 1790 + 180, 700, 1790 + 420))
up2(pin).save(os.path.join(OUT, 'ti_p5_pin.png'))
print('ti_p5_wire', up2(wire).size, 'ti_p5_pin', up2(pin).size)

# ---------- 3. P8 port crop: real directional port (public example 1, teammate crop) ----------
port = Image.open(os.path.join(WS, 'teammate_extract', 'imgs', 's06_b6b3bc36.png')).convert('RGB')
port3 = port.resize((port.width * 3, port.height * 3), Image.LANCZOS)
dp = ImageDraw.Draw(port3)
W3, H3 = port3.size
# thin navy ticks: under the purple terminal dot (left) and under the flag tip (right)
dp.line([(46, 150), (46, H3 - 2)], fill=(18, 60, 109), width=4)
dp.line([(438, 150), (438, H3 - 2)], fill=(18, 60, 109), width=4)
port3.save(os.path.join(OUT, 'ti_p8_port.png'))
print('ti_p8_port', port3.size)

# ---------- 4. P11 original + composite overlay ----------
orig = Image.open(os.path.join(WS, 'smoke', 'input', 'rp2350_page11.png')).convert('RGB')
orig.save(os.path.join(OUT, 'ti_p11_orig.png'))
print('ti_p11_orig', orig.size)

base = np.asarray(Image.open(os.path.join(A, 'fig3b_network_overlay.png')).convert('RGB')).astype(np.int16)
ov = np.asarray(Image.open(os.path.join(A, 'fig1b_detection_clean.png')).convert('RGB')).astype(np.int16)
gm = (ov[:, :, 1] > 130) & (ov[:, :, 1] > ov[:, :, 0] + 60) & (ov[:, :, 1] > ov[:, :, 2] + 60)
comp = base.copy()
comp[gm] = ov[gm]
Image.fromarray(comp.astype(np.uint8)).save(os.path.join(OUT, 'ti_p11_overlay.png'))
print('ti_p11_overlay green px:', int(gm.sum()))

# ---------- 5. Backup A skeleton strip (39 junction nodes) ----------
sk = Image.open(os.path.join(D, 'p6_zoom_skeleton.png')).convert('RGB')
up2(sk).save(os.path.join(OUT, 'ti_ba_skeleton.png'))
print('ti_ba_skeleton', up2(sk).size)
print('ALL ASSETS DONE')
