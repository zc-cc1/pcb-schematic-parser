"""English diagram assets for the official-template deck."""
import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import matplotlib
matplotlib.use('Agg')
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

for c in [r'C:\Windows\Fonts\arial.ttf', r'C:\Windows\Fonts\msyh.ttc']:
    try:
        fm.fontManager.addfont(c)
    except Exception:
        pass
plt.rcParams['font.family'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

OUT = r'E:\AI_Workplace\精英挑战赛\_audit_workspace\deck_workspace'
NAVY2 = '#123C6D'; CYAN = '#0FB5C9'; CYAN_D = '#0E7C8C'; ORANGE = '#F59E0B'
INK = '#17263B'; MUTED = '#5A6B80'; PANEL = '#EEF4FA'

def mbox(ax, x, y, w, h, title, sub='', fc='white', ec=NAVY2, tc=INK, lw=1.4, fs=10.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.5', facecolor=fc, edgecolor=ec, lw=lw))
    ax.text(x + w / 2, y + h / 2 + (2.6 if sub else 0), title, ha='center', va='center',
            fontsize=fs, weight='bold', color=tc)
    if sub:
        ax.text(x + w / 2, y + h / 2 - 3.0, sub, ha='center', va='center', fontsize=8.2, color=MUTED)

def marr(ax, x1, y1, x2, y2, c=NAVY2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=13, color=c, lw=1.6))

# ============ 1. architecture (English) ============
fig, ax = plt.subplots(figsize=(15.5, 10.2), dpi=150)
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis('off')
stages = [
    (2.0, 72, 21, 24, '1  Input', ['Single schematic PNG', '(30-50 cases, 1/3 public)', 'subfolders supported'], '#ECEFF1', '#78909C'),
    (26.0, 72, 23, 24, '2  Perception', ['YOLO whole + SAHI tiled', 'WBF fusion (35 classes)', 'PP-OCRv6 offline OCR', 'run_baseline.py'], '#E3F2FD', '#1976D2'),
    (52.0, 72, 22, 24, '3  Dual-head U-Net', ['wire prob map + pin heatmap', 'tiles 512/384 native', 'peak decode + subpixel', 'unet_inference.py'], '#E8F5E9', '#388E3C'),
    (77.0, 72, 21, 24, '4  Pin Ownership', ['heatmap peaks -> candidates', 'cost-optimal assignment', 'margin / ambiguity gates', 'pin_ownership.py'], '#FFF3E0', '#F57C00'),
    (2.0, 42, 21, 24, '5  Wire Graph', ['mask -> skeleton -> segments', 'conservative junctions', 'red-dot detect / gap log', 'wire_graph.py'], '#EDE7F6', '#7E57C2'),
    (26.0, 42, 23, 24, '6  Electrical Networks', ['connected components = nets', 'same-name labels -> flywires', 'VCC/GND power merge', 'network_labels.py'], '#FCE4EC', '#C2185B'),
    (52.0, 42, 22, 24, '7  Field Completion S14-S24', ['1000-part pin library', '6-view targeted OCR', 'strip OCR / row reread', 'run_observed_net_fields'], '#E0F7FA', '#00838F'),
    (77.0, 42, 21, 24, '8  Evidence-Gated Repair x15', ['transactional rule engine', '15 structural patterns', '0.98 x 2-view OCR gate', 'agent.finalize_*'], '#FFFDE7', '#F9A825'),
    (2.0, 12, 30, 24, '9  Official Format Calibration', ['type aliases r/c/l/d/box/led', 'Name==key convention', 'template bbox + page scale', 'IC frame / LED / tip snap', 'adapter.py + export_geometry'], '#E8EAF6', '#3F51B5'),
    (36.0, 12, 26, 24, '10  Export & Resilience', ['validate_target checks', 'sanitize (remove only invalid)', 'fallback + audit logs', 'hash-locked end to end', 'export_resilience.py'], '#ECEFF1', '#546E7A'),
    (66.0, 12, 30, 24, '11  Output JSON', ['decision_log.json', 'final_results/<case>_Pred.json', 'components / pins / nets', 'coords: bottom-left, H-y'], '#E8F5E9', '#2E7D32'),
]
for s in stages:
    x, y, w, h, title, lines, fc, ec = s
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.4', facecolor=fc, edgecolor=ec, linewidth=1.4))
    ax.text(x + w / 2, y + h - 3.2, title, ha='center', va='top', fontsize=11, weight='bold')
    for i, t in enumerate(lines):
        ax.text(x + 1.2, y + h - 7.5 - i * 4.0, t, ha='left', va='top', fontsize=8.6)
def elbow(ax, x1, y1, x2, y2):
    midy = y1 - (y1 - y2) * 0.55
    ax.plot([x1, x1], [y1, midy], color='#455A64', lw=1.6)
    ax.plot([x1, x2], [midy, midy], color='#455A64', lw=1.6)
    ax.add_patch(FancyArrowPatch((x2, midy), (x2, y2), arrowstyle='-|>', mutation_scale=16, color='#455A64', lw=1.6))
for a, b, c, d2 in [(23, 84, 25.8, 84), (49, 84, 51.8, 84), (74, 84, 76.8, 84)]:
    marr(ax, a, b, c, d2)
elbow(ax, 87.5, 71.4, 12.5, 66.6)
for a, b, c, d2 in [(23, 54, 25.8, 54), (49, 54, 51.8, 54), (74, 54, 76.8, 54)]:
    marr(ax, a, b, c, d2)
elbow(ax, 87.5, 41.4, 17, 36.6)
for a, b, c, d2 in [(32, 11.6, 35.8, 11.6), (62, 11.6, 65.8, 11.6)]:
    marr(ax, a, b, c, d2)

fig.savefig(rf'{OUT}\arch_pipeline_en.png', bbox_inches='tight', facecolor='white')
plt.close(fig)
print('arch_pipeline_en saved')

# ============ 2. YOLO fusion (English) ============
fig, ax = plt.subplots(figsize=(7.4, 4.4), dpi=200)
ax.set_xlim(0, 100); ax.set_ylim(0, 62); ax.axis('off')
mbox(ax, 2, 40, 20, 14, 'Whole-image pass', 'imgsz 1536 · conf 0.2', fc=PANEL, ec=NAVY2)
mbox(ax, 2, 8, 20, 14, 'SAHI tiled pass', '1024 px · overlap 0.25', fc=PANEL, ec=CYAN_D)
mbox(ax, 30, 40, 22, 14, 'YOLO nano detector', '35 classes · self-trained', fc='white')
mbox(ax, 30, 24, 22, 11, 'Per-class confidence', '27 thresholds (ic .60 / gnd .41)', fc='white', fs=9.2)
mbox(ax, 30, 8, 22, 12, 'WBF fusion', 'IoU 0.55 · score-weighted', fc='#E8F6F8', ec=CYAN_D)
mbox(ax, 62, 24, 20, 14, 'Fused detections', '59-case micro F1 0.883', fc=NAVY2, tc='white', ec=NAVY2)
mbox(ax, 85, 24, 13, 14, 'OCR binding', 'designator -> part', fc=PANEL, ec=MUTED, fs=9.6)
marr(ax, 22, 47, 29.6, 47)
marr(ax, 22, 15, 29.6, 13.4)
ax.plot([52, 56], [13.6, 13.6], color=NAVY2, lw=1.6)
marr(ax, 52, 13.6, 61.6, 26.5)
marr(ax, 52, 47, 61.6, 36)
marr(ax, 82.2, 31, 84.6, 31)
ax.text(2, 58.6, 'Dual-path detection + class-aware WBF (score-weighted union, not NMS winner-takes-all)',
        fontsize=10, color=INK, weight='bold')
fig.savefig(rf'{OUT}\d_yolo_fusion_en.png', bbox_inches='tight', facecolor='white')
plt.close(fig)
print('yolo fusion en saved')

# ============ 3. JSON structure (English) ============
fig, ax = plt.subplots(figsize=(6.9, 4.5), dpi=200)
ax.set_xlim(0, 100); ax.set_ylim(0, 64); ax.axis('off')
ax.text(1, 61.5, 'Output JSON (real fragments, official example page)', fontsize=11, weight='bold', color=INK)
groups = [
    ('components', ['"U3": {"type":"box","Name":"RP2350A",', '"bbox":[539,281,689,731]}'], CYAN_D),
    ('pins', ['"U3": {"pin_51": {"pinname":"USB_DM",', '"point":{"x":513,"y":561}}}'], NAVY2),
    ('nets', ['"LOG_NET": {"hyperGraph":"(C1.2,U3.U3)",', '"edges":{...3 edges}}'], ORANGE),
]
y = 56
for key, vals, col in groups:
    ax.add_patch(FancyBboxPatch((1, y - 7.5), 16, 10, boxstyle='round,pad=0.35', facecolor=col, edgecolor=col))
    ax.text(9, y - 2.5, key, ha='center', va='center', color='white', fontsize=8.6, weight='bold')
    ax.text(19.5, y - 2.5, vals[0], va='center', fontsize=8.0, color=INK, family='Consolas')
    ax.text(19.5, y - 10.5, vals[1], va='center', fontsize=8.0, color=INK, family='Consolas')
    y -= 15.5
ax.add_patch(FancyBboxPatch((1, 1), 96, 8, boxstyle='round,pad=0.4', facecolor=PANEL, edgecolor='#B9C6D6', lw=1))
ax.text(49, 5, 'Bottom-left origin (H-y) · auto suffix _1/_2 · auto pin_N · flywires: zero-length edges',
        ha='center', va='center', fontsize=8.0, color=INK)
fig.savefig(rf'{OUT}\d_json_structure_en.png', bbox_inches='tight', facecolor='white')
plt.close(fig)
print('json structure en saved')

# ============ 4. recovery flow (English, B3) ============
fig, ax = plt.subplots(figsize=(7.4, 3.2), dpi=200)
ax.set_xlim(0, 100); ax.set_ylim(0, 40); ax.axis('off')
flow = [('Stage error', 'try/except', '#FDF3E3', ORANGE),
        ('Freeze last valid', 'remember() checkpoint', PANEL, NAVY2),
        ('Validate & write', 'sanitize + validate', NAVY2, NAVY2),
        ('Whole-case fail', 'recover_current_run', '#FDF3E3', ORANGE),
        ('Log cause', 'decision_log', PANEL, NAVY2)]
xs = [1]; widths = [17, 20, 19, 21, 17]
for i in range(4):
    xs.append(xs[-1] + widths[i] + 1.2)
for i, (t, s, fc, ec) in enumerate(flow):
    w = widths[i]
    mbox(ax, xs[i], 10, w, 17, t, s, fc=fc, ec=ec, tc='white' if fc == NAVY2 else INK, fs=9.2)
    if i < 4:
        marr(ax, xs[i] + w, 18.5, xs[i + 1] - 0.3, 18.5)
ax.text(1, 35, 'No crash: per-stage checkpoints + work-dir-only rescue recovery', fontsize=10.5, weight='bold', color=INK)
fig.savefig(rf'{OUT}\d_recovery_flow_en.png', bbox_inches='tight', facecolor='white')
plt.close(fig)
print('recovery flow en saved')

# ============ 5. module map (English, B1) ============
fig, ax = plt.subplots(figsize=(15.5, 9.6), dpi=150)
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis('off')
groups = [
    (2, 58, 30, 36, 'Visual Model Layer', ['YOLO nano detector (35 cls, self-trained)', 'WirePinUNet dual-head (milesial parts)',
     'PP-OCRv6 OCR (called, offline ONNX)', 'JunctionCNN (self-trained, OFF by default)'], '#E3F2FD', '#1976D2'),
    (2, 14, 30, 36, 'OCR Evidence System', ['Multi-view: 2/3/4x x 0/90 deg rotation', 'Gate: >=2 views @ confidence 0.98',
     'Reject truncated / conflicting glyphs', 'Hard budget 12-160 calls per page'], '#E8F5E9', '#388E3C'),
    (35, 58, 30, 36, 'Geometry & Topology', ['ic_geometry observed four-edge frame', 'LED / capacitor dual-scale tips',
     'export_geometry canonical boxes', 'skeleton -> segments -> conservative graph'], '#FFF3E0', '#F57C00'),
    (35, 14, 30, 36, 'Transactional Repair Engine', ['repair_engine atomic commits', 'capability whitelist + mutual veto',
     '15 structural patterns + 12 sub-stages', 'members only grow / NC pins barred'], '#FCE4EC', '#C2185B'),
    (68, 58, 30, 36, 'Field & Pairing Repair', ['14 pin-repair modules (field/geometry', '/pairing/cycle/NC/vertical/library/...)',
     'ic_name_selection part-number priority', 'component_kind_repair type fixes'], '#EDE7F6', '#7E57C2'),
    (68, 14, 30, 36, 'Output Adapter', ['adapter.to_target type aliases', 'Name==key convention (9186 comps)',
     'template bbox + page scale (159/163)', 'validate/sanitize/fallback (626 tests)'], '#E0F7FA', '#00838F'),
]
for x, y, w, h, title, lines, fc, ec in groups:
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.4', facecolor=fc, edgecolor=ec, linewidth=1.5))
    ax.text(x + w / 2, y + h - 2.5, title, ha='center', va='top', fontsize=11, weight='bold')
    for i, t in enumerate(lines):
        ax.text(x + 1.2, y + h - 7 - i * 4.2, t, ha='left', va='top', fontsize=8.2)
def arr2(x1, y1, x2, y2, lbl=''):
    marr(ax, x1, y1, x2, y2)
    if lbl:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 1.2, lbl, ha='center', fontsize=7.2, color='#37474F')
arr2(32, 76, 34.8, 76, 'maps / boxes')
arr2(32, 40, 34.8, 40, 'gated reads')
arr2(65, 76, 67.8, 76, 'geometry')
arr2(65, 40, 67.8, 40, 'repaired')
arr2(50, 57.6, 50, 50.4, 'candidates')
arr2(83, 57.6, 83, 50.4, 'prediction')
ax.text(50, 6, 'Models propose -> evidence certifies -> rule engine repairs conservatively -> adapter aligns to official convention -> auditable export',
        ha='center', fontsize=10, color='#263238')
fig.savefig(rf'{OUT}\arch_modules_en.png', bbox_inches='tight', facecolor='white')
plt.close(fig)
print('modules en saved')
print('ALL EN DIAGRAMS DONE')
