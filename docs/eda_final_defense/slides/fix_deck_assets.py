"""Fix round: tighter fig4, fig2c wire-panel crop, fixed json structure, fixed recovery flow."""
import io
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

fm.fontManager.addfont(r'C:\Windows\Fonts\msyh.ttc')
CN = fm.FontProperties(fname=r'C:\Windows\Fonts\msyh.ttc').get_name()
plt.rcParams['font.family'] = [CN, 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

WS = r'E:\AI_Workplace\精英挑战赛\_audit_workspace'
A = os.path.join(WS, 'assets_build')
D = os.path.join(WS, 'deck_workspace')
NAVY = '#0B1F3A'; NAVY2 = '#123C6D'; CYAN = '#0FB5C9'; CYAN_D = '#0E7C8C'; ORANGE = '#F59E0B'
INK = '#17263B'; MUTED = '#5A6B80'; PANEL = '#EEF4FA'
FONT = fm.FontProperties(fname=r'C:\Windows\Fonts\msyh.ttc')
MFONT = fm.FontProperties(fname=r'C:\Windows\Fonts\msyh.ttc')

# ---- 1. fig4 tighter: big crop left, compact numbers right ----
gap = Image.open(os.path.join(A, '..', 'vision-fetch', 'gap_crop_case13.png')).convert('RGB')
gap = Image.open(rf'{WS}\vision-fetch\gap_crop_case13.png').convert('RGB')
panel = Image.new('RGB', (1720, 1000), 'white')
d = ImageDraw.Draw(panel)
F22 = ImageFont.truetype(rf'{WS}\v102_composed\assets\fonts\DejaVuSansMono.ttf', 22)
F16 = ImageFont.truetype(rf'{WS}\v102_composed\assets\fonts\DejaVuSansMono.ttf', 16)
F18 = ImageFont.truetype(rf'{WS}\v102_composed\assets\fonts\DejaVuSansMono.ttf', 18)
d.text((14, 8), 'Page 13 - junction reconciliation: recover wire segments', font=F22, fill=(20, 20, 20))
gap_r = gap.resize((760, 760))
panel.paste(gap_r, (10, 60))
d.rectangle([10, 60, 770, 820], outline=(200, 200, 200), width=1)
d.text((16, 828), 'crop: gap_crop_case13.png (development record, repo vision)', font=F16, fill=(120, 120, 120))
X = 800
d.text((X, 70), 'Before (pre-fix record)', font=F18, fill=(180, 30, 30))
d.text((X + 18, 104), 'wire  32/28/17   F1 58.72%', font=F16, fill=(60, 60, 60))
d.text((X + 18, 132), 'connectivity 67.14%   score 82.871', font=F16, fill=(60, 60, 60))
d.text((X, 190), 'After (junction reconciliation)', font=F18, fill=(20, 110, 30))
d.text((X + 18, 224), 'wire  46/13/3   F1 85.19%', font=F16, fill=(60, 60, 60))
d.text((X + 18, 252), 'connectivity 87.53%   score 92.656', font=F16, fill=(60, 60, 60))
d.text((X, 310), 'Mechanism (network_junction_reconcile)', font=F18, fill=(20, 20, 20))
mech = [
    '1. two segments overlap >=60px on one line',
    '   (|angle|<3deg, perp offset <=2px), or',
    '2. both approach same measured red dot',
    '   (<=30px) with >=95% ink support along gap.',
    'Merge members/edges only; comps/pins intact;',
    'max 8 merges/page; other 12 pages identical.',
]
for i, t in enumerate(mech):
    d.text((X + 18, 344 + i * 30), t, font=F16, fill=(70, 70, 70))
d.text((X, 560), 'Fix is local, not global (cache replay).', font=F18, fill=(110, 110, 110))
d.text((14, 960), 'Data provenance: before = page13_status_20261020/status.json (pin_row_library deployed pred);', font=F16, fill=(130, 130, 130))
d.text((14, 982), 'after = docs/network_junction_reconciliation.md. Full-page re-render needs original page13 PNG (not shipped).', font=F16, fill=(130, 130, 130))
panel.save(rf'{A}\fig4_page13_repair_case.png')
print('fig4 regenerated', panel.size)

# ---- 2. fig2c crop: input + wire head only ----
src = Image.open(rf'{A}\fig2c_unet_real_artifacts.png')
crop = src.crop((0, 0, 1190, 845 + 70))
crop.save(rf'{D}\d_unet_wire_panel.png')
print('wire panel crop saved', crop.size)

# ---- 3. json structure fix (ASCII values, badge fit) ----
fig, ax = plt.subplots(figsize=(6.9, 4.5), dpi=200)
ax.set_xlim(0, 100); ax.set_ylim(0, 66); ax.axis('off')
ax.text(1, 62, '输出 JSON(官方示例页真实片段)', fontsize=11.5, weight='bold', color=INK, fontproperties=FONT)
mono = fm.FontProperties(fname=r'C:\Windows\Fonts\consola.ttf')
rows = [
    ('components', '"U3": {"type":"box","Name":"RP2350A",', CYAN_D),
    ('', '"bbox":[539,281,689,731]}', CYAN_D),
    ('pins', '"U3": {"pin_51": {"pinname":"USB_DM",', NAVY2),
    ('', '"point":{"x":513,"y":561}}}', NAVY2),
    ('nets', '"LOG_NET": {"hyperGraph":"(C1.2,U3.U3)",', ORANGE),
    ('', '"edges":{...3 edges}}', ORANGE),
]
y = 54
for key, val, col in rows:
    if key:
        ax.add_patch(FancyBboxPatch((1, y - 7.5), 14.5, 10, boxstyle='round,pad=0.35', facecolor=col, edgecolor=col))
        ax.text(8.2, y - 2.5, key, ha='center', va='center', color='white', fontsize=9.0, weight='bold', fontproperties=FONT)
        ax.text(17.5, y - 2.5, val, va='center', fontsize=8.6, color=INK, fontproperties=mono)
    else:
        ax.text(17.5, y - 2.5, val, va='center', fontsize=8.6, color=INK, fontproperties=mono)
    y -= 10.5
ax.add_patch(FancyBboxPatch((1, 2), 96, 10.5, boxstyle='round,pad=0.4', facecolor=PANEL, edgecolor='#B9C6D6', lw=1))
ax.text(49, 7.2, '坐标:左下原点 (H-y) · 重复位号自动编号 _1/_2 · 缺失编号自动 pin_N · 飞线 edges 为零长度段',
        ha='center', va='center', fontsize=8.6, color=INK, fontproperties=FONT)
fig.savefig(rf'{D}\d_json_structure.png', bbox_inches='tight', facecolor='white')
plt.close(fig)
print('json structure fixed')

# ---- 4. recovery flow fix (short labels, wider boxes) ----
fig, ax = plt.subplots(figsize=(7.4, 3.2), dpi=200)
ax.set_xlim(0, 100); ax.set_ylim(0, 40); ax.axis('off')
def mbox(x, y, w, h, title, sub='', fc='white', ec=NAVY2, tc=INK, fs=9.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.4', facecolor=fc, edgecolor=ec, lw=1.4))
    ax.text(x + w / 2, y + h / 2 + (2.4 if sub else 0), title, ha='center', va='center', fontsize=fs, weight='bold', color=tc, fontproperties=FONT)
    if sub:
        ax.text(x + w / 2, y + h / 2 - 2.8, sub, ha='center', va='center', fontsize=7.6, color=MUTED, fontproperties=FONT)
def marr(x1, y1, x2, y2, c=NAVY2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=13, color=c, lw=1.6))
flow = [
    ('阶段异常', 'try/except', '#FDF3E3', ORANGE),
    ('冻结最近有效', 'remember() 校验点', PANEL, NAVY2),
    ('校验后写入', 'sanitize+validate', NAVY2, NAVY2),
    ('整例失败', 'recover_current_run', '#FDF3E3', ORANGE),
    ('记录原因', 'decision_log', PANEL, NAVY2),
]
xs = [1]
widths = [17, 20, 19, 21, 17]
for i in range(4):
    xs.append(xs[-1] + widths[i] + 1.2)
for i, (t, s, fc, ec) in enumerate(flow):
    w = widths[i]
    tc = 'white' if fc == NAVY2 else INK
    mbox(xs[i], 10, w, 17, t, s, fc=fc, ec=ec, tc=tc, fs=9.4)
    if i < 4:
        marr(xs[i] + w, 18.5, xs[i + 1] - 0.3, 18.5)
ax.text(1, 35, '失败不崩溃:逐阶段校验点 + 仅读本图工作目录的抢救恢复', fontsize=10.5, weight='bold', color=INK, fontproperties=FONT)
fig.savefig(rf'{D}\d_recovery_flow.png', bbox_inches='tight', facecolor='white')
plt.close(fig)
print('recovery flow fixed')
print('FIX ROUND DONE')
