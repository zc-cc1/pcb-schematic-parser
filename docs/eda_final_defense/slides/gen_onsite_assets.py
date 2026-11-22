"""Onsite-final asset generation: fig2c panel crops, topology zooms, U3 bbox comparison,
recovery-flow regeneration, page-13 crop trim."""
import io
import json
import os
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import matplotlib
matplotlib.use('Agg')
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fm.fontManager.addfont(r'C:\Windows\Fonts\arial.ttf')
plt.rcParams['font.family'] = ['Arial', 'DejaVu Sans']

WS = r'E:\AI_Workplace\精英挑战赛\_audit_workspace'
A = os.path.join(WS, 'assets_build')
D = os.path.join(WS, 'deck_workspace')
SUB = os.path.join(WS, 'v102_composed')
REPO = os.path.join(WS, 'vision-fetch')
FONT = os.path.join(SUB, 'assets', 'fonts', 'DejaVuSansMono.ttf')
F14 = ImageFont.truetype(FONT, 14)
F16 = ImageFont.truetype(FONT, 16)
F18 = ImageFont.truetype(FONT, 18)

# ---------- 1. fig2c panel crops ----------
src = Image.open(os.path.join(A, 'fig2c_unet_real_artifacts.png'))
# layout: input at y30..875, wire at 905..1750, pin at 1800..2645 (panel height 845, label bands)
src.crop((0, 0, 1190, 872)).save(os.path.join(D, 'p5_input.png'))        # input + its label
src.crop((0, 880, 1190, 1768)).save(os.path.join(D, 'p5_wire.png'))      # wire panel + label
src.crop((0, 1775, 1190, 2635)).save(os.path.join(D, 'p5_pin.png'))      # pin panel (label inside panel text)
print('fig2c crops saved')

# ---------- 2. topology zooms ----------
fig3c = Image.open(os.path.join(A, 'fig3c_topology_real.png'))
fig3d = Image.open(os.path.join(A, 'fig3d_graph_networks_real.png'))
ZB = (430, 130, 780, 430)  # U3 area with junction nodes / colored edges
fig3c.crop(ZB).save(os.path.join(D, 'p6_zoom_skeleton.png'))
fig3d.crop(ZB).save(os.path.join(D, 'p6_zoom_networks.png'))
print('topology zooms saved')

# ---------- 3. U3 bbox comparison ----------
rgb = np.asarray(Image.open(os.path.join(WS, 'smoke', 'input', 'rp2350_page11.png')).convert('RGB'))
H, W = rgb.shape[:2]
# official bbox bottom-left [540,280,690,730] -> top-left y = H - y
ox1, oy1, ox2, oy2 = 540, H - 730, 690, H - 280
mx1, my1, mx2, my2 = 539, H - 731, 689, H - 281
pad = 30
x0, y0 = max(0, ox1 - pad), max(0, oy1 - pad)
x1c, y1c = min(W, ox2 + pad), min(H, oy2 + pad)
crop = Image.fromarray(rgb[y0:y1c, x0:x1c].copy())
dd = ImageDraw.Draw(crop)
dd.rectangle([ox1 - x0, oy1 - y0, ox2 - x0, oy2 - y0], outline=(30, 90, 220), width=3)   # official (blue)
dd.rectangle([mx1 - x0, my1 - y0, mx2 - x0, my2 - y0], outline=(240, 130, 20), width=2)  # ours (orange)
crop.save(os.path.join(D, 'p8_u3_bbox_compare.png'))
print('U3 bbox compare saved', crop.size)

# ---------- 4. recovery flow regenerate ----------
fig, ax = plt.subplots(figsize=(7.6, 3.3), dpi=200)
ax.set_xlim(0, 108); ax.set_ylim(0, 40); ax.axis('off')
NAVY2 = '#123C6D'; ORANGE = '#F59E0B'; INK = '#17263B'; MUTED = '#5A6B80'; PANEL = '#EEF4FA'
def mbox(ax, x, y, w, h, title, sub='', fc='white', ec=NAVY2, tc=INK, fs=9.4):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.4', facecolor=fc, edgecolor=ec, lw=1.4))
    ax.text(x + w / 2, y + h / 2 + (2.4 if sub else 0), title, ha='center', va='center', fontsize=fs, weight='bold', color=tc)
    if sub:
        ax.text(x + w / 2, y + h / 2 - 2.8, sub, ha='center', va='center', fontsize=7.8, color=MUTED)
def marr(ax, x1, y1, x2, y2, c=NAVY2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=13, color=c, lw=1.6))
flow = [('Stage error', 'try/except', '#FDF3E3', ORANGE),
        ('Freeze last valid', 'remember() checkpoint', PANEL, NAVY2),
        ('Validate & write', 'sanitize + validate', NAVY2, NAVY2),
        ('Whole-case fail', 'recover_current_run', '#FDF3E3', ORANGE),
        ('Log cause', 'decision_log', PANEL, NAVY2)]
xs = [1]; widths = [18, 20, 19, 21, 18]
for i in range(4):
    xs.append(xs[-1] + widths[i] + 1.4)
for i, (t, s2, fc, ec) in enumerate(flow):
    w = widths[i]
    mbox(ax, xs[i], 10, w, 17, t, s2, fc=fc, ec=ec, tc='white' if fc == NAVY2 else INK, fs=9.4)
    if i < 4:
        marr(ax, xs[i] + w, 18.5, xs[i + 1] - 0.3, 18.5)
ax.text(1, 35, 'Checkpointed failure recovery - no crash, work-dir rescue only', fontsize=10.5, weight='bold', color=INK)
fig.savefig(rf'{D}\d_recovery_flow_en.png', bbox_inches='tight', pad_inches=0.15, facecolor='white')
plt.close(fig)
print('recovery flow regenerated')

# ---------- 5. page-13 crop trim (content bbox) ----------
gap = Image.open(os.path.join(REPO, 'gap_crop_case13.png')).convert('RGB')
arr = np.asarray(grep_rgb := gap).astype(int)
nonwhite = (arr.sum(axis=2) < 720)
ys, xs_ = np.nonzero(nonwhite)
y0b, y1b, x0b, x1b = ys.min(), ys.max(), xs_.min(), xs_.max()
m = 12
crop = gap.crop((max(0, x0b - m), max(0, y0b - m), min(gap.width, x1b + m), min(gap.height, y1b + m)))
crop.save(os.path.join(D, 'p7_gap_trim.png'))
print('page-13 crop trimmed ->', crop.size, 'from content bbox', (x0b, y0b, x1b, y1b))
