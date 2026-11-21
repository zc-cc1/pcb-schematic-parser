"""Supplementary diagrams for the defense deck (matplotlib, deck palette)."""
import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import matplotlib
matplotlib.use('Agg')
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

for cand in [r'C:\Windows\Fonts\msyh.ttc', r'C:\Windows\Fonts\msyh.ttf']:
    try:
        fm.fontManager.addfont(cand)
        CN = fm.FontProperties(fname=cand).get_name()
        break
    except Exception:
        CN = None
plt.rcParams['font.family'] = [CN or 'DejaVu Sans', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

OUT = r'E:\AI_Workplace\精英挑战赛\_audit_workspace\deck_workspace'
NAVY = '#0B1F3A'; NAVY2 = '#123C6D'; CYAN = '#0FB5C9'; CYAN_D = '#0E7C8C'
ORANGE = '#F59E0B'; INK = '#17263B'; MUTED = '#5A6B80'; PANEL = '#EEF4FA'
FONT = fm.FontProperties(fname=r'C:\Windows\Fonts\msyh.ttc')

# ================= 1. cover art: dark navy PCB traces =================
def pcb_panel(path, w_in=5.6, h_in=7.5, seed=5, n_traces=26):
    fig, ax = plt.subplots(figsize=(w_in, h_in), dpi=200)
    ax.set_xlim(0, 100); ax.set_ylim(0, 141); ax.axis('off')
    ax.add_patch(plt.Rectangle((0, 0), 100, 141, color=NAVY, zorder=0))
    ax.add_patch(plt.Rectangle((0, 0), 100, 141, color='#0A1B31', zorder=1, alpha=0.55))
    rng = np.random.default_rng(seed)
    # horizontal buses on the right edge
    for i in range(7):
        y = 14 + i * 17
        ax.plot([64, 97], [y, y], color=CYAN, lw=1.6, alpha=0.5, zorder=2)
        ax.add_patch(Circle((97, y), 1.5, color=CYAN, alpha=0.55, zorder=3))
    # L-shaped traces with pads
    for _ in range(n_traces):
        x0 = rng.uniform(2, 60); y0 = rng.uniform(8, 132)
        dx = rng.uniform(14, 40) * rng.choice([-1, 1])
        dy = rng.uniform(10, 34) * rng.choice([-1, 1])
        x1 = min(max(x0 + dx, 2), 88); y1 = y0 + dy
        x2 = min(max(x1 + (rng.uniform(8, 26) * rng.choice([-1, 1])), 2), 92); y2 = y1
        col = CYAN if rng.random() < 0.72 else '#3E7FB8'
        ax.plot([x0, x1], [y0, y1], color=col, lw=1.4, alpha=0.5, zorder=2)
        ax.plot([x1, x2], [y1, y2], color=col, lw=1.4, alpha=0.5, zorder=2)
        ax.add_patch(Circle((x2, y2), 1.15, color=col, alpha=0.6, zorder=3))
        if rng.random() < 0.3:
            ax.add_patch(Circle((x0, y0), 1.0, color=col, alpha=0.45, zorder=3))
    # chip silhouette
    cx, cy = 74, 96
    ax.add_patch(FancyBboxPatch((cx-9, cy-7), 18, 14, boxstyle='round,pad=0.6',
                                facecolor='#123A5F', edgecolor=CYAN, lw=1.5, alpha=0.9, zorder=4))
    for i in range(6):
        ax.plot([cx-9, cx-12], [cy-5+i*2, cy-5+i*2], color=CYAN, lw=1.2, alpha=0.6, zorder=4)
        ax.plot([cx+9, cx+12], [cy-5+i*2, cy-5+i*2], color=CYAN, lw=1.2, alpha=0.6, zorder=4)
    ax.text(cx, cy, 'PCB', ha='center', va='center', color='#9BD8E2',
            fontsize=9, fontproperties=FONT, weight='bold', zorder=5)
    # detection box + pins motif (visual detection motif)
    ax.add_patch(plt.Rectangle((30, 30), 16, 9, fill=False, edgecolor=ORANGE, lw=1.8, alpha=0.9, zorder=4))
    ax.text((38), (27.5), 'box', color=ORANGE, fontsize=7, ha='center', fontproperties=FONT, zorder=4)
    for px in np.arange(31.5, 46, 2.4):
        ax.add_patch(Circle((px, 25.5), 0.55, color=CYAN, alpha=0.8, zorder=4))
    fig.savefig(path, bbox_inches='tight', pad_inches=0, facecolor=NAVY)
    plt.close(fig)

pcb_panel(rf'{OUT}\d_cover_pcb.png')
print('cover art saved')

# ================= 2. score weights bar =================
fig, ax = plt.subplots(figsize=(5.4, 3.1), dpi=200)
labels = ['元件 ComponentF1', '引脚 PinF1', '网络超图 NetHypergraphF1', '线段 NetLineF1']
vals = [0.30, 0.25, 0.35, 0.10]
colors = [NAVY2, NAVY2, ORANGE, CYAN_D]
bars = ax.barh(range(4), vals, color=colors, height=0.62)
ax.set_yticks(range(4)); ax.set_yticklabels(labels, fontproperties=FONT, fontsize=11)
ax.invert_yaxis()
for i, v in enumerate(vals):
    ax.text(v + 0.008, i, f'{v:.2f}', va='center', fontsize=12, weight='bold',
            color=(ORANGE if i == 2 else INK))
ax.set_xlim(0, 0.42); ax.set_xticks([])
for s in ['top', 'right', 'bottom', 'left']:
    ax.spines[s].set_visible(False)
ax.set_title('单案例评分权重  Score_Accuracy_SingleCase', fontproperties=FONT,
             fontsize=12.5, color=INK, pad=10, weight='bold', loc='left')
fig.tight_layout()
fig.savefig(rf'{OUT}\d_score_weights.png', bbox_inches='tight', facecolor='white')
plt.close(fig)
print('score weights saved')

# ================= 3. YOLO dual-path fusion =================
fig, ax = plt.subplots(figsize=(7.4, 4.4), dpi=200)
ax.set_xlim(0, 100); ax.set_ylim(0, 62); ax.axis('off')

def mbox(x, y, w, h, title, sub='', fc='white', ec=NAVY2, tc=INK, lw=1.4, fs=10.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.5', facecolor=fc, edgecolor=ec, lw=lw))
    ax.text(x + w/2, y + h/2 + (2.6 if sub else 0), title, ha='center', va='center',
            fontsize=fs, weight='bold', color=tc, fontproperties=FONT)
    if sub:
        ax.text(x + w/2, y + h/2 - 3.0, sub, ha='center', va='center', fontsize=8.4,
                color=MUTED, fontproperties=FONT)

def marr(x1, y1, x2, y2, c=NAVY2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=13, color=c, lw=1.6))

mbox(2, 40, 20, 14, '整图推理', 'imgsz 1536 · conf 0.2', fc=PANEL, ec=NAVY2)
mbox(2, 8, 20, 14, 'SAHI 切片推理', '1024² · overlap 0.25', fc=PANEL, ec=CYAN_D)
mbox(30, 40, 22, 14, 'YOLO nano 检测器', '35 类 · 自训 imgsz1536', fc='white')
mbox(30, 24, 22, 11, '逐类置信阈值', '27 组(如 ic 0.60 / gnd 0.41)', fc='white', fs=9.2)
mbox(30, 8, 22, 12, 'WBF 融合', 'IoU 0.55 · 置信加权坐标', fc='#E8F6F8', ec=CYAN_D)
mbox(62, 24, 20, 14, 'fused 检测结果', '59 例微平均 F1 0.883', fc=NAVY2, tc='white', ec=NAVY2)
mbox(85, 24, 13, 14, 'OCR 绑定', '位号→元件', fc=PANEL, ec=MUTED, fs=9.6)
marr(22, 47, 29.6, 47)
marr(22, 15, 29.6, 13.4)
ax.plot([52, 56], [13.6, 13.6], color=NAVY2, lw=1.6)
marr(52, 13.6, 61.6, 26.5)
marr(52, 47, 61.6, 36)
marr(82.2, 31, 84.6, 31)
ax.text(14, 58.6, '双路检测 + 类感知 WBF 融合(不下 NMS 二选一,置信加权取并)', fontsize=10.5,
        color=INK, fontproperties=FONT, weight='bold')
fig.savefig(rf'{OUT}\d_yolo_fusion.png', bbox_inches='tight', facecolor='white')
plt.close(fig)
print('yolo fusion saved')

# ================= 4. evidence gate + transaction flow =================
fig, ax = plt.subplots(figsize=(7.6, 4.6), dpi=200)
ax.set_xlim(0, 100); ax.set_ylim(0, 64); ax.axis('off')

steps = [
    ('多视角 OCR', '2/3/4× × 0/90°\n预算 12~160 次/页', 2, 40, PANEL, CYAN_D),
    ('证据门控', '≥2 视角一致\n且置信 ≥ 0.98', 27, 40, '#E8F6F8', CYAN_D),
    ('冲突检测', '截断字形/拼写冲突\n与既有字段互斥', 52, 40, PANEL, NAVY2),
    ('原子事务', '成员只增不减\nNC pin 禁入网', 77, 40, PANEL, NAVY2),
]
for t, s, x, y, fc, ec in steps:
    mbox(x, y, 21, 15, t, s, fc=fc, ec=ec, fs=10.2)
    if x < 70:
        marr(x + 21, y + 7.5, x + 24.8, y + 7.5, ec)
mbox(2, 12, 45, 15, '接受 → 审计日志 + 校验点冻结', 'logs/<case>_*.json · remember() 最近有效预测', fc=NAVY2, tc='white', fs=10.2)
mbox(52, 12, 46, 15, '拒绝 → 保留原值,不猜测', '失败回滚 recover_current_run(仅读本图)', fc='#FDF3E3', ec=ORANGE, fs=10.2)
marr(87.5, 39.6, 75, 27.4, ORANGE)
marr(37, 39.6, 25, 27.4, NAVY2)
ax.text(2, 60.5, '证据门控 × 事务式修复:每一处改写都要"过门",失败自动回退', fontsize=11.5,
        color=INK, fontproperties=FONT, weight='bold')
fig.savefig(rf'{OUT}\d_evidence_flow.png', bbox_inches='tight', facecolor='white')
plt.close(fig)
print('evidence flow saved')

# ================= 5. JSON structure mini =================
fig, ax = plt.subplots(figsize=(6.9, 4.5), dpi=200)
ax.set_xlim(0, 100); ax.set_ylim(0, 66); ax.axis('off')
ax.text(1, 62, '输出 JSON(官方示例页真实片段)', fontsize=11.5, weight='bold', color=INK, fontproperties=FONT)
mono = fm.FontProperties(fname=r'C:\Windows\Fonts\consola.ttf')
rows = [
    ('components', '"U3": {"type":"box","Name":"RP2350A","bbox":[539,281,689,731]}', CYAN_D),
    ('pins', '"U3": {"pin_51": {"pinname":"USB_DM","point":{"x":513,"y":561}}}', NAVY2),
    ('nets', '"LOG_NET": {"hyperGraph":"(C1.2,U3.U3)","edges":{...3 段}}', ORANGE),
]
y = 50
for key, val, col in rows:
    ax.add_patch(FancyBboxPatch((1, y-7.5), 12, 10, boxstyle='round,pad=0.35', facecolor=col, edgecolor=col))
    ax.text(7, y-2.5, key, ha='center', va='center', color='white', fontsize=9.6, weight='bold', fontproperties=FONT)
    ax.text(15.5, y-2.5, val, va='center', fontsize=8.6, color=INK, fontproperties=mono)
    y -= 13.5
ax.add_patch(FancyBboxPatch((1, 2), 96, 11.5, boxstyle='round,pad=0.4', facecolor=PANEL, edgecolor=MUTED, lw=1))
ax.text(49, 7.8, '坐标:左下原点 (H-y) · 重复位号自动编号 _1/_2 · pinNumber 缺失自动 pin_N · 飞线 edges 为零长度段',
        ha='center', va='center', fontsize=8.8, color=INK, fontproperties=FONT)
fig.savefig(rf'{OUT}\d_json_structure.png', bbox_inches='tight', facecolor='white')
plt.close(fig)
print('json structure saved')

# ================= 6. failure recovery flow (engineering page) =================
fig, ax = plt.subplots(figsize=(7.2, 3.4), dpi=200)
ax.set_xlim(0, 100); ax.set_ylim(0, 46); ax.axis('off')
flow = ['阶段 N 异常', 'remember() 冻结\n最近有效预测', 'sanitize 后写入\nfinal_results', '整例失败→\nrecover_current_run', 'decision_log\n记录原因']
xs = [1, 21, 41, 61, 82]
for i, (t, x) in enumerate(zip(flow, xs)):
    w = 18 if i < 4 else 17
    fc = '#FDF3E3' if i in (0, 3) else (NAVY2 if i == 2 else PANEL)
    tc = 'white' if i == 2 else INK
    mbox(x, 8, w, 18, t.split('\n')[0], t.split('\n')[1] if '\n' in t else '', fc=fc, tc=tc,
         ec=(ORANGE if i in (0, 3) else NAVY2), fs=9.2)
    if i < 4:
        marr(x + w, 17, xs[i+1] - 0.4, 17)
ax.text(1, 40, '失败不崩溃:逐阶段校验点 + 仅读本图工作目录的抢救恢复', fontsize=10.5,
        weight='bold', color=INK, fontproperties=FONT)
fig.savefig(rf'{OUT}\d_recovery_flow.png', bbox_inches='tight', facecolor='white')
plt.close(fig)
print('recovery flow saved')

# ================= 7. three-tier evidence (experiment page) =================
fig, ax = plt.subplots(figsize=(6.8, 3.6), dpi=200)
ax.set_xlim(0, 100); ax.set_ylim(0, 48); ax.axis('off')
tiers = [
    ('官方平台评测', '74 分', '30~50 案例 · 公开:黑盒=1:2\n难例 60% · 排名 47 / 晋级 61 强', NAVY2, 'white'),
    ('本地公开集回放', '98.93', '13 页公开数据 · 官方口径复算\n开发评估分,勿与官方分比较', PANEL, INK),
    ('本机端到端冒烟', '89/165/63', '官方示例页 1190×845 · CPU 2000s\nU3 bbox 与官方标注差 1~2px', '#E8F6F8', INK),
]
for i, (t, num, s, fc, tc) in enumerate(tiers):
    x = 1 + i * 33.4
    ax.add_patch(FancyBboxPatch((x, 6), 31, 34, boxstyle='round,pad=0.5', facecolor=fc,
                                edgecolor=(NAVY2 if i else ORANGE), lw=1.6))
    ax.text(x + 15.5, 33.5, t, ha='center', fontsize=10.2, weight='bold', color=tc, fontproperties=FONT)
    ax.text(x + 15.5, 24.5, num, ha='center', fontsize=17, weight='bold',
            color=(ORANGE if i == 0 else (NAVY2 if i == 1 else CYAN_D)))
    ax.text(x + 15.5, 13.5, s, ha='center', va='center', fontsize=7.8, color=(MUTED if i else '#D8E4F0'), fontproperties=FONT)
ax.text(1, 45, '三层证据,口径分明', fontsize=11.5, weight='bold', color=INK, fontproperties=FONT)
fig.savefig(rf'{OUT}\d_three_tiers.png', bbox_inches='tight', facecolor='white')
plt.close(fig)
print('three tiers saved')
print('ALL DIAGRAMS DONE')
