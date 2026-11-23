# -*- coding: utf-8 -*-
"""Sync build_cn_deck.py to the teammate-integration layout (translated).
Mirrors the six page rewrites + typography pass applied to build_official_deck.py."""
import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
P = r'E:\AI_Workplace\精英挑战赛\_audit_workspace\vision-fetch\docs\eda_final_defense\slides\build_cn_deck.py'
s = open(P, encoding='utf-8').read()

def block_replace(s, start_marker, end_marker, new_block):
    lines = s.split('\n')
    si = next(i for i, l in enumerate(lines) if start_marker in l)
    ei = next(i for i, l in enumerate(lines) if end_marker in l and i > si)
    lines[si:ei] = new_block.split('\n')
    return '\n'.join(lines)

def rep1(s, old, new):
    assert s.count(old) == 1, f'not unique: {old[:50]!r} count={s.count(old)}'
    return s.replace(old, new)

# ---- 0. title 26 / subtitle 16 ----
s = rep1(s, "r.font.size = Pt(24); r.font.bold = True; r.font.name = FONT",
         "r.font.size = Pt(26); r.font.bold = True; r.font.name = FONT")
s = rep1(s, "set_text(sh, subtitle, size=13, color=MUTED, font=FONT)",
         "set_text(sh, subtitle, size=16, color=MUTED, font=FONT)")

# ---- 1. Part I ----
s = block_replace(s, '# ---------- Part I ----------', '# ---------- Part II:四层架构 ----------',
"""# ---------- Part I ----------
set_title_subtitle(part1, None, '问题理解与解决策略')
cx, cy, cw, ch = content_shapes(part1)
add_text(part1, cx, cy + 0.02, cw, 0.56, '难点不在找到符号。', size=22, bold=True, color=NAVY)
tbx = part1.shapes[-1].text_frame
r = tbx.paragraphs[0].add_run(); r.text = '而在恢复电气连接。'
r.font.size = Pt(22); r.font.bold = True; r.font.color.rgb = ORANGE; r.font.name = FONT
add_text(part1, cx, cy + 0.62, cw, 0.32,
         [('感知', {'size': 15, 'bold': True, 'color': CYAN})], size=15)
tbx = part1.shapes[-1].text_frame
r = tbx.paragraphs[0].add_run(); r.text = '  识别元件、文字、引脚与导线   '; r.font.size = Pt(14); r.font.color.rgb = MUTED; r.font.name = FONT
r = tbx.paragraphs[0].add_run(); r.text = '连接'; r.font.size = Pt(15); r.font.bold = True; r.font.color.rgb = NAVY; r.font.name = FONT
r = tbx.paragraphs[0].add_run(); r.text = '  建立几何 + 电气拓扑   '; r.font.size = Pt(14); r.font.color.rgb = MUTED; r.font.name = FONT
r = tbx.paragraphs[0].add_run(); r.text = '验证'; r.font.size = Pt(15); r.font.bold = True; r.font.color.rgb = ORANGE; r.font.name = FONT
r = tbx.paragraphs[0].add_run(); r.text = '  用证据修复 + 校准输出'; r.font.size = Pt(14); r.font.color.rgb = MUTED; r.font.name = FONT

# 共享 ID 示意条:PNG -> 元件/引脚/网络 -> Target JSON
sy = cy + 1.00
add_card(part1, cx, sy, cw, 0.84, fill=WHITE)
place_pic(part1, rf'{D}\\ti_p5_input.png', cx + 0.15, sy + 0.09, 1.26, 0.66)
ar = part1.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(cx + 1.52), Inches(sy + 0.31), Inches(0.30), Inches(0.22))
ar.fill.solid(); ar.fill.fore_color.rgb = CYAN; ar.line.fill.background()
add_text(part1, cx + 1.94, sy + 0.13, 3.75, 0.32, '元件 · 引脚 · 网络', size=16, bold=True, color=NAVY)
add_text(part1, cx + 1.94, sy + 0.47, 3.75, 0.28, '由同一个共享 Component ID 关联', size=14, color=MUTED)
ar2 = part1.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(cx + 5.80), Inches(sy + 0.31), Inches(0.30), Inches(0.22))
ar2.fill.solid(); ar2.fill.fore_color.rgb = CYAN; ar2.line.fill.background()
add_text(part1, cx + 6.22, sy + 0.06, 5.15, 0.28, 'Target JSON', size=14, bold=True, color=NAVY)
add_text(part1, cx + 6.22, sy + 0.36, 5.15, 0.26,
         '"U3": {"bbox": [539,281,689,731]}', size=13.5, color=INK, font=MONO)
add_text(part1, cx + 6.22, sy + 0.62, 5.15, 0.26,
         '"LOG_NET": ["C1.2", "U3.U3"]', size=13.5, color=INK, font=MONO)

ch_items = [
    ('密集元器件检测', '小型贴片器件易漏检、类别混淆', '双路 YOLO(整图+SAHI 切片)+ WBF 融合', '逐类置信阈值,检测 F1 0.883'),
    ('引脚定位与归属', '多引脚 IC 易错位、漏检、绑定错误', '双头 U-Net 热图 + 代价最优指派', '峰解码 + 亚像素修正'),
    ('文字与型号关联', '标注拥挤、倾斜', '多视角 OCR 证据门控', '≥2 视角一致 @ 置信 0.98,预算制'),
    ('拓扑与连通性', 'T 型十字、断线、飞线', '保守建图 + 事务式修复', '先防连错,再补漏连'),
]
row_h = 0.74
for i, (ct, cd, st, sd) in enumerate(ch_items):
    y = cy + 1.98 + i * (row_h + 0.10)
    add_card(part1, cx, y, 5.28, row_h, fill=WHITE)
    add_text(part1, cx + 0.14, y + 0.05, 5.0, 0.32, ct, size=16, bold=True, color=NAVY)
    add_text(part1, cx + 0.14, y + 0.40, 5.0, 0.30, cd, size=14, color=MUTED)
    ar = part1.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(cx + 5.44), Inches(y + row_h / 2 - 0.13), Inches(0.46), Inches(0.26))
    ar.fill.solid(); ar.fill.fore_color.rgb = CYAN; ar.line.fill.background()
    add_card(part1, cx + 6.1, y, 5.4, row_h, fill=PANEL)
    add_text(part1, cx + 6.24, y + 0.05, 5.15, 0.32, st, size=16, bold=True, color=INK)
    add_text(part1, cx + 6.24, y + 0.40, 5.15, 0.30, sd, size=14, color=MUTED)

""")

# ---- 2. Part II four-layer: font bumps ----
s = block_replace(s, '# ---------- Part II:四层架构 ----------', '# ---------- 正文页克隆函数 ----------',
"""# ---------- Part II:四层架构 ----------
set_title_subtitle(part2, None, '端到端系统架构')
cx, cy, cw, ch = content_shapes(part2)
layers = [
    ('1   视觉感知', 'YOLO 双路 + WBF 融合 · 双头 U-Net(导线+引脚)· PP-OCRv6 离线 OCR', PANEL, NAVY),
    ('2   几何处理', '引脚归属指派 · 骨架→线段 · IC/LED 实框 · 尖端吸附', WHITE, NAVY),
    ('3   连接推理', '导线图与结点策略 · 物理网络 · 飞线/电源合并 · 字段补全 S14-S24', RGBColor(0xE8, 0xF6, 0xF8), CYAN),
    ('4   证据门控修正', '事务式修复引擎(15 阶段)· 官方格式校准 · JSON 导出校验', RGBColor(0xFF, 0xF7, 0xE9), ORANGE),
]
for i, (t, d, fc, ec) in enumerate(layers):
    y = cy + 0.02 + i * 1.25
    add_card(part2, cx, y, cw, 1.12, fill=fc, line_c=ec)
    add_text(part2, cx + 0.22, y + 0.16, 3.75, 0.8, t, size=20, bold=True, color=NAVY)
    add_text(part2, cx + 4.05, y + 0.2, 7.3, 0.76, d, size=16, color=INK)
    if i < 3:
        da = part2.shapes.add_shape(MSO_SHAPE.DOWN_ARROW, Inches(cx + 1.55), Inches(y + 1.15), Inches(0.26), Inches(0.13))
        da.fill.solid(); da.fill.fore_color.rgb = CYAN; da.line.fill.background()
add_text(part2, cx, cy + 5.12, cw, 0.34,
         '完整 11 阶段见备份页 A · 零 LLM Token · 逐阶段哈希锁定与审计', size=14, color=MUTED)

""")

# ---- 3. P5 视觉感知 ----
s = block_replace(s, '# ---- P5 视觉感知 ----', '# ---- P6 拓扑重建 ----',
'''# ---- P5 视觉感知(队友式:PNG -> 模块 -> 证据) ----
pP3 = new_body('视觉感知', '三个默认启用的模型产生候选;结论由证据门控给出')
p = pP3
cx, cy, cw, ch = content_shapes(p)

add_card(p, cx, cy + 0.08, 2.35, 2.30, fill=WHITE)
add_text(p, cx + 0.12, cy + 0.16, 2.1, 0.32, '原理图 PNG', size=16, bold=True, color=NAVY)
place_pic(p, rf'{D}\\ti_p5_input.png', cx + 0.12, cy + 0.56, 2.11, 1.13)
add_text(p, cx + 0.12, cy + 1.78, 2.1, 0.26, '官方示例页', size=14, color=MUTED)
add_text(p, cx, cy + 2.60, 2.35, 0.30, '模型来源', size=15, bold=True, color=INK)
origins = [
    ('YOLO nano', '自研训练 · F1 0.883', CYAN),
    ('WirePinUNet', '自研训练(milesial)', NAVY),
    ('PP-OCRv6', '调用开源预训练(离线)', NAVY),
    ('JunctionCNN', '默认关闭(可选)', ORANGE),
]
for i, (t1, d1, dc) in enumerate(origins):
    y = cy + 2.98 + i * 0.60
    dot = p.shapes.add_shape(MSO_SHAPE.OVAL, Inches(cx + 0.02), Inches(y + 0.05), Inches(0.13), Inches(0.13))
    dot.fill.solid(); dot.fill.fore_color.rgb = dc; dot.line.fill.background()
    add_text(p, cx + 0.24, y, 2.1, 0.28, t1, size=15, bold=True, color=INK)
    add_text(p, cx + 0.24, y + 0.30, 2.1, 0.26, d1, size=13.5, color=MUTED)

tr = p.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(cx + 2.51), Inches(cy + 0.48), Inches(0.025), Inches(3.42))
tr.fill.solid(); tr.fill.fore_color.rgb = LINE; tr.line.fill.background()
stub = p.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(cx + 2.37), Inches(cy + 1.18), Inches(0.15), Inches(0.025))
stub.fill.solid(); stub.fill.fore_color.rgb = LINE; stub.line.fill.background()

add_text(p, cx + 2.80, cy + 0.06, 3.30, 0.32, '视觉模块', size=18, bold=True, color=NAVY)
modules = [
    ('YOLO nano 检测器', '整图 + SAHI 切片;WBF 按逐类阈值融合'),
    ('PP-OCRv6(RapidOCR)', '在元件、引脚与文本堆栈附近做多视角 OCR'),
    ('双头 U-Net', '共享图像特征 → 导线掩膜 + 引脚尖端热图'),
]
mx = cx + 2.80
for i, (t, d) in enumerate(modules):
    y = cy + 0.50 + i * 1.24
    add_card(p, mx, y, 3.30, 1.10, fill=PANEL)
    add_text(p, mx + 0.14, y + 0.08, 3.0, 0.32, t, size=17, bold=True, color=NAVY)
    add_text(p, mx + 0.14, y + 0.44, 3.02, 0.58, d, size=14, color=INK)
    arw = p.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(mx - 0.28), Inches(y + 0.45), Inches(0.24), Inches(0.20))
    arw.fill.solid(); arw.fill.fore_color.rgb = CYAN; arw.line.fill.background()

ex = cx + 6.50
add_text(p, ex, cy + 0.06, 5.05, 0.32, '传给下游的证据', size=18, bold=True, color=NAVY)
ev_rows = [
    ('ti_p5_yolo.png', '元件', '35 类 + 边界框'),
    ('ti_p5_ocr.png', 'OCR 令牌', '文字 · 位置 · 置信度(≥2 视角 @0.98)'),
]
for i, (img, t, d) in enumerate(ev_rows):
    y = cy + 0.50 + i * 1.24
    place_pic(p, rf'{D}\\' + img, ex, y + 0.02, 1.90, 0.96)
    add_text(p, ex + 2.10, y + 0.06, 2.95, 0.30, t, size=16, bold=True, color=INK)
    add_text(p, ex + 2.10, y + 0.40, 2.95, 0.56, d, size=14, color=MUTED)
    arw = p.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(ex - 0.36), Inches(y + 0.40), Inches(0.32), Inches(0.20))
    arw.fill.solid(); arw.fill.fore_color.rgb = CYAN; arw.line.fill.background()
y3 = cy + 0.50 + 2 * 1.24
place_pic(p, rf'{D}\\ti_p5_wire.png', ex, y3 + 0.10, 0.92, 0.49)
place_pic(p, rf'{D}\\ti_p5_pin.png', ex + 0.98, y3 + 0.10, 0.92, 0.49)
add_text(p, ex, y3 + 0.64, 0.92, 0.24, '导线掩膜', size=14, color=MUTED, align=PP_ALIGN.CENTER)
add_text(p, ex + 0.98, y3 + 0.64, 0.92, 0.24, '引脚热图', size=14, color=MUTED, align=PP_ALIGN.CENTER)
add_text(p, ex + 2.10, y3 + 0.06, 2.95, 0.30, '导线与引脚', size=16, bold=True, color=INK)
add_text(p, ex + 2.10, y3 + 0.40, 2.95, 0.56, '工作导线掩膜 + 引脚尖端热图(双头)', size=14, color=MUTED)
arw = p.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(ex - 0.36), Inches(y3 + 0.40), Inches(0.32), Inches(0.20))
arw.fill.solid(); arw.fill.fore_color.rgb = CYAN; arw.line.fill.background()

add_text(p, cx + 2.80, cy + 4.22, 6.60, 0.32,
         '解码出 431 个引脚候选:', size=16, bold=True, color=INK)
add_text(p, cx + 2.80, cy + 4.58, 6.60, 0.34,
         [('287', {'size': 17, 'bold': True, 'color': NAVY})], size=17)
tbx = p.shapes[-1].text_frame
r = tbx.paragraphs[0].add_run(); r.text = ' 已归属  ·  '; r.font.size = Pt(14); r.font.color.rgb = MUTED; r.font.name = FONT
r = tbx.paragraphs[0].add_run(); r.text = '115'; r.font.size = Pt(17); r.font.bold = True; r.font.color.rgb = NAVY; r.font.name = FONT
r = tbx.paragraphs[0].add_run(); r.text = ' 歧义  ·  '; r.font.size = Pt(14); r.font.color.rgb = MUTED; r.font.name = FONT
r = tbx.paragraphs[0].add_run(); r.text = '29'; r.font.size = Pt(17); r.font.bold = True; r.font.color.rgb = NAVY; r.font.name = FONT
r = tbx.paragraphs[0].add_run(); r.text = ' 无主'; r.font.size = Pt(14); r.font.color.rgb = MUTED; r.font.name = FONT
add_card(p, cx + 9.65, cy + 4.22, 2.85, 0.92, fill=WHITE)
add_text(p, cx + 9.75, cy + 4.32, 2.65, 0.44, '检测 F1 0.883', size=19, bold=True, color=ORANGE, align=PP_ALIGN.CENTER)
add_text(p, cx + 9.75, cy + 4.80, 2.65, 0.28, '59 例 · fused 配置', size=14, color=MUTED, align=PP_ALIGN.CENTER)

''')

# ---- 4. P6 拓扑重建 ----
s = block_replace(s, '# ---- P6 拓扑重建 ----', '# ---- P7 不靠猜测的修复 ----',
'''# ---- P6 拓扑重建(队友式:大图 + 三步) ----
pP4 = new_body('拓扑重建', '导线概率图 → 骨架 → 线段 → 网络')
p = pP4
cx, cy, cw, ch = content_shapes(p)
place_pic(p, rf'{D}\\ti_p6_main.png', cx, cy + 0.02, 6.16, 4.91)
add_text(p, cx, cy + 4.98, 6.16, 0.26,
         '局部预测叠加图——标记 1/2/3 对应右侧三步',
         size=14, color=MUTED, align=PP_ALIGN.CENTER)
sx = cx + 6.45
steps = [
    ('1 · 引脚归属', '把每个解码端点匹配到元件边界与邻近引脚方向——431 个候选中 287 个确定归属。'),
    ('2 · 导线追踪', '在工作掩膜中追踪端点、拐点与结点之间的线段——T 型十字一律保守拆分,断线只凭硬证据恢复。'),
    ('3 · 网络归属', '把共享支撑路径的端点归入同一网络并保留走线几何——同名飞线标签逻辑合并,电源网(VCC/GND)过四道门。'),
]
for i, (t, d) in enumerate(steps):
    y = cy + 0.04 + i * 1.16
    add_text(p, sx, y, 5.10, 0.34, t, size=18, bold=True, color=NAVY)
    add_text(p, sx, y + 0.38, 5.10, 0.74, d, size=14.5, color=INK)
add_card(p, sx, cy + 3.58, 5.10, 1.41, fill=PANEL)
add_text(p, sx + 0.16, cy + 3.68, 4.8, 0.60,
         [('238 个物理网络候选 / 362 条边 → 63 个逻辑网络', {'size': 17, 'bold': True, 'color': NAVY})], size=17)
add_text(p, sx + 0.16, cy + 4.34, 4.8, 0.56,
         '候选是建图阶段中间态;合并后得到最终输出。',
         size=14, color=INK)

''')

# ---- 5. P7: width + font bumps ----
s = block_replace(s, '# ---- P7 不靠猜测的修复 ----', '# ---- P8 输出校准 ----',
'''# ---- P7 不靠猜测的修复 ----
pP5 = new_body('不靠猜测的修复', '页 13 结点归并——只在硬证据下合并')
p = pP5
cx, cy, cw, ch = content_shapes(p)
place_pic(p, rf'{D}\\p7_gap_trim.png', cx, cy + 0.1, 3.6, 3.6)
add_text(p, cx, cy + 3.78, 3.65, 0.30, '页 13 真实裁切(开发记录)', size=14, color=MUTED, align=PP_ALIGN.CENTER)
add_text(p, cx + 3.95, cy + 0.05, 7.70, 0.5, '页 13 导线修复', size=20, bold=True, color=NAVY)
mets = [
    ('TP / FP / FN', '32/28/17', '46/13/3'),
    ('导线 F1', '58.72%', '85.19%'),
    ('连通率', '67.14%', '87.53%'),
    ('单页得分', '82.871', '92.656'),
]
for i, (m, b, a) in enumerate(mets):
    y = cy + 0.72 + i * 0.5
    add_text(p, cx + 3.95, y, 2.6, 0.36, m, size=15, bold=True, color=INK)
    add_text(p, cx + 6.6, y, 1.9, 0.36, b, size=16, color=MUTED)
    add_text(p, cx + 8.35, y, 0.6, 0.36, '→', size=15, bold=True, color=INK)
    add_text(p, cx + 8.95, y, 2.2, 0.36, a, size=16, bold=True, color=ORANGE)
mech = [
    '合并证据:≥60px 同线重合(±3°、垂距≤2px)或红点 + 95% 墨迹',
    '只合并成员与边——元件与引脚不动;每页 ≤8 组',
]
for i, t in enumerate(mech):
    add_text(p, cx + 3.95, cy + 2.85 + i * 0.40, 7.70, 0.36, t, size=14, color=INK)
add_text(p, cx + 3.95, cy + 3.72, 7.70, 0.52,
         '其余 12 页逐字节不变(冻结缓存回放)——单页消融,非官方整体指标。',
         size=14, italic=True, color=MUTED)
add_text(p, cx + 3.95, cy + 4.30, 7.70, 0.34,
         [('导线 F1 提升来自结点归并机制——与 OCR 门控无关。', {'size': 14.5, 'bold': True, 'color': NAVY})], size=14.5)
add_text(p, cx + 3.95, cy + 4.72, 7.70, 0.34,
         [('证据门控 OCR(≥2 视角 @0.98)是另一项独立能力——见备份页 E。', {'size': 14, 'color': MUTED})], size=14)

''')

# ---- 6. P8 输出校准 ----
s = block_replace(s, '# ---- P8 输出校准 ----', '# ---------- Part III ----------',
'''# ---- P8 输出校准(端口方向规则 + 压缩校准卡) ----
pP6 = new_body('面向官方评测的输出校准',
               '对官方标注约定做统计校准——不是逐图查表')
p = pP6
cx, cy, cw, ch = content_shapes(p)

add_text(p, cx, cy + 0.04, 5.25, 0.34, '对官方约定的统计校准', size=17, bold=True, color=NAVY)
cards = [
    ('类型与字段约定', 'resistor → r …(145 个匹配元件验证)· Name==key 来自 9186 组件统计——box 保留型号', CYAN),
    ('bbox 校准(两套口径)', 'canonical(40 页公开子集):20px 达标率 0.905 → 1.000 · 按页缩放(13 页):150/163 → 159/163', NAVY),
    ('引脚尖端吸附', '18px 吸附(实测 16-22px 平台区)+ 1000 型号引脚库', ORANGE),
]
for i, (t, d, c) in enumerate(cards):
    y = cy + 0.44 + i * 1.16
    add_card(p, cx, y, 5.25, 1.04, fill=WHITE)
    add_text(p, cx + 0.16, y + 0.07, 4.9, 0.32, t, size=16, bold=True, color=INK)
    add_text(p, cx + 0.16, y + 0.42, 4.95, 0.56, d, size=14, color=MUTED)
    dot = p.shapes.add_shape(MSO_SHAPE.OVAL, Inches(cx + 4.97), Inches(y + 0.10), Inches(0.14), Inches(0.14))
    dot.fill.solid(); dot.fill.fore_color.rgb = c; dot.line.fill.background()
uy = cy + 3.94
add_card(p, cx, uy, 5.25, 1.00, fill=PANEL)
place_pic(p, rf'{D}\\p8_u3_bbox_compare.png', cx + 0.12, uy + 0.07, 0.50, 0.86)
add_text(p, cx + 0.78, uy + 0.08, 1.55, 0.46, '1~2 px', size=28, bold=True, color=ORANGE)
add_text(p, cx + 2.40, uy + 0.10, 2.75, 0.80,
         'U3(RP2350A)边界框与官方示例标注之差(官方容差 20px)', size=14, color=INK)
add_text(p, cx, uy + 1.10, 5.25, 0.28,
         '两套 bbox 口径相互独立——见备份页 D。', size=14, color=MUTED)

rx = cx + 5.60
add_text(p, rx, cy + 0.04, 5.95, 0.34, '基于图像证据的导出规则——方向端口', size=17, bold=True, color=NAVY)
place_pic(p, rf'{D}\\ti_p8_port.png', rx, cy + 0.46, 4.42, 1.35)
add_text(p, rx - 0.36, cy + 1.86, 1.40, 0.28, '有导线端子', size=14, color=INK, align=PP_ALIGN.CENTER)
add_text(p, rx + 2.33, cy + 1.86, 1.40, 0.28, '箭头尖端', size=14, color=INK, align=PP_ALIGN.CENTER)
add_text(p, rx, cy + 2.20, 5.90, 0.58,
         '端子在尖端对侧 → 输出端口;在尖端一侧 → 输入。说明文字保留在 Name;value → null。',
         size=16, color=INK)
add_text(p, rx, cy + 2.82, 5.90, 0.30,
         '仅在旗形轮廓与单一端子证据一致时生效(14 项单测)。',
         size=14, color=MUTED)
add_card(p, rx, cy + 3.20, 5.95, 2.10, fill=PANEL)
add_text(p, rx + 0.20, cy + 3.30, 5.5, 0.32, 'Target JSON 中的端口字段', size=16, bold=True, color=NAVY)
add_text(p, rx + 0.20, cy + 3.66, 5.5, 0.32, '"type": "net_output"', size=17, color=INK, font=MONO)
add_text(p, rx + 0.20, cy + 4.02, 5.5, 0.32, '"Name": "VOUT"', size=17, color=INK, font=MONO)
add_text(p, rx + 0.20, cy + 4.38, 5.5, 0.32, '"value": null', size=17, color=INK, font=MONO)
add_text(p, rx + 0.20, cy + 4.74, 5.55, 0.50,
         '旧口径 net_bidirection → 观测几何。旧标注:PCB_EXPORT_SEMANTICS=legacy。',
         size=14, color=MUTED)

''')

# ---- 7. Part III: font bumps ----
s = block_replace(s, '# ---------- Part III ----------', '# ---- P10 定量结果 ----',
'''# ---------- Part III ----------
set_title_subtitle(part3, None, '官方评测成绩与定量证据')
cx, cy, cw, ch = content_shapes(part3)
chip(part3, cx + 0.1, cy + 0.15, 3.6, 1.9, '74', '官方评测成绩', ORANGE, 54, 14, fill=WHITE)
chip(part3, cx + 3.9, cy + 0.15, 3.6, 1.9, '47', '全国排名', NAVY, 54, 14, fill=WHITE)
chip(part3, cx + 7.7, cy + 0.15, 3.6, 1.9, '前 61 强', '晋级决赛', NAVY, 44, 14, fill=WHITE)
add_card(part3, cx, cy + 2.3, 11.55, 1.42, fill=PANEL)
add_text(part3, cx + 0.2, cy + 2.4, 11.2, 0.4, '版本迭代', size=16, bold=True, color=INK)
tbx = part3.shapes[-1].text_frame
r = tbx.paragraphs[0].add_run(); r.text = '   28.92'; r.font.size = Pt(16); r.font.bold = True; r.font.color.rgb = NAVY; r.font.name = FONT
r = tbx.paragraphs[0].add_run(); r.text = '(旧版 8422)   →   '; r.font.size = Pt(14); r.font.color.rgb = MUTED; r.font.name = FONT
r = tbx.paragraphs[0].add_run(); r.text = '74'; r.font.size = Pt(16); r.font.bold = True; r.font.color.rgb = ORANGE; r.font.name = FONT
r = tbx.paragraphs[0].add_run(); r.text = '(最终提交,+45)'; r.font.size = Pt(14); r.font.color.rgb = MUTED; r.font.name = FONT
add_text(part3, cx + 0.2, cy + 2.82, 11.2, 0.8,
         '提升来源:无位号电源/地出口 · LED 文字与框 · IC 型号优先级与实框 · 工程编号 · 导出语义。',
         size=14, color=INK)
add_text(part3, cx, cy + 4.4, cw, 0.36,
         '本地开发回放分(13 页公开集):移至备份页 D——仅开发用途,与官方结果不可直接比较。',
         size=14, color=MUTED)

''')

# ---- 8. P10: font bumps + positions ----
s = block_replace(s, '# ---- P10 定量结果 ----', '# ---- P11 端到端 ----',
'''# ---- P10 定量结果 ----
pP8 = new_body('定量结果与案例', '四项实验指标——开发集实验与单页消融')
p = pP8
cx, cy, cw, ch = content_shapes(p)
mets = [
    ('元器件', 'bbox 20px 达标率', '0.905', '1.000', 'canonical 约定 · 40 页公开子集'),
    ('引脚', 'PinF1', '0.031', '0.549', '引脚库映射 · 13 页 released'),
    ('连通率', '网络连通', '0.188', '0.612', '结点门控放行 · 13 页 released'),
    ('导线', '导线 F1(页13案例)', '58.72%', '85.19%', '结点归并 · 单页消融'),
]
for i, (dim, met, b, a, scope) in enumerate(mets):
    col, row = i % 2, i // 2
    x = cx + col * 5.95
    y = cy + 0.45 + row * 2.15
    add_card(p, x, y, 5.6, 1.95, fill=WHITE)
    add_text(p, x + 0.2, y + 0.1, 5.2, 0.36, dim, size=16, bold=True, color=NAVY)
    add_text(p, x + 0.2, y + 0.55, 2.1, 0.6, b, size=24, bold=True, color=MUTED)
    add_text(p, x + 2.15, y + 0.62, 0.7, 0.5, '→', size=22, bold=True, color=INK)
    add_text(p, x + 2.95, y + 0.55, 2.6, 0.6, a, size=24, bold=True, color=ORANGE)
    add_text(p, x + 0.2, y + 1.25, 5.2, 0.30, met, size=14, color=INK)
    add_text(p, x + 0.2, y + 1.57, 5.2, 0.30, scope, size=13.5, color=MUTED)
add_text(p, cx, cy + 4.64, cw, 0.32,
         '开发集实验与单页消融——非官方分项成绩。完整表见备份页 D。',
         size=14, bold=True, color=INK)
add_text(p, cx, cy + 5.00, cw, 0.28,
         '消融协议:冻结模型/OCR 缓存,只开一个模块;其余 12 页逐字节不变。',
         size=13.5, color=MUTED)

''')

# ---- 9. P11 端到端 ----
s = block_replace(s, '# ---- P11 端到端 ----', '# ---------- Part IV ----------',
'''# ---- P11 端到端(队友式:原图 | 叠加 + 图例 + 记录) ----
pP9 = new_body('端到端输出与工程验证',
               '官方示例页,完整管线——从 PNG 到校验通过的 JSON')
p = pP9
cx, cy, cw, ch = content_shapes(p)
place_pic(p, rf'{D}\\ti_p11_orig.png', cx, cy + 0.04, 4.25, 3.02)
add_text(p, cx, cy + 3.14, 4.25, 0.30, '原始原理图(官方示例页)', size=14, color=MUTED, align=PP_ALIGN.CENTER)
place_pic(p, rf'{D}\\ti_p11_overlay.png', cx + 5.60, cy + 0.04, 4.25, 3.02)
add_text(p, cx + 5.60, cy + 3.14, 4.25, 0.30, '预测叠加图(同页)', size=14, color=MUTED, align=PP_ALIGN.CENTER)
recs = [('89', '元器件'), ('165', '引脚'), ('63', '逻辑网络')]
for i, (n, lb) in enumerate(recs):
    y = cy + 0.10 + i * 1.02
    add_text(p, cx + 10.10, y, 2.35, 0.48, n, size=28, bold=True, color=NAVY)
    add_text(p, cx + 10.10, y + 0.50, 2.35, 0.28, lb, size=14, color=MUTED)
ly = cy + 3.62
lg = p.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(cx), Inches(ly + 0.05), Inches(0.24), Inches(0.16))
lg.fill.solid(); lg.fill.fore_color.rgb = RGBColor(0x00, 0x98, 0x51); lg.line.fill.background()
add_text(p, cx + 0.34, ly, 1.85, 0.28, '元件边界框', size=14, color=INK)
dot = p.shapes.add_shape(MSO_SHAPE.OVAL, Inches(cx + 2.30), Inches(ly + 0.04), Inches(0.17), Inches(0.17))
dot.fill.solid(); dot.fill.fore_color.rgb = RGBColor(0xC0, 0x39, 0x2B); dot.line.fill.background()
add_text(p, cx + 2.57, ly, 1.60, 0.28, '引脚端点', size=14, color=INK)
ln = p.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(cx + 4.28), Inches(ly + 0.10), Inches(0.30), Inches(0.055))
ln.fill.solid(); ln.fill.fore_color.rgb = ORANGE; ln.line.fill.background()
add_text(p, cx + 4.68, ly, 2.45, 0.28, '导线路径(按网着色)', size=14, color=INK)
add_text(p, cx + 7.35, ly, 5.15, 0.28, '经校验——绝不推断文本、坐标或连接', size=14, color=MUTED)
for i, (key, l1, c) in enumerate([
        ('components', '"U3": box RP2350A · bbox [539,281,689,731]', RGBColor(0x0E, 0x7C, 0x8C)),
        ('pins', '"U3".pin_51: USB_DM @ (513, 561)', NAVY),
        ('nets', '"LOG_NET": (C1.2, U3.U3) · 3 条边', ORANGE)]):
    y = cy + 4.06 + i * 0.44
    kc = p.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(cx), Inches(y), Inches(1.42), Inches(0.38))
    kc.fill.solid(); kc.fill.fore_color.rgb = c; kc.line.fill.background()
    try:
        kc.shadow.inherit = False
    except Exception:
        pass
    ktf = kc.text_frame; ktf.word_wrap = True
    kpa = ktf.paragraphs[0]; kpa.alignment = PP_ALIGN.CENTER
    kr = kpa.add_run(); kr.text = key; kr.font.size = Pt(12.5); kr.font.bold = True; kr.font.color.rgb = WHITE; kr.font.name = FONT
    add_text(p, cx + 1.60, y + 0.05, 6.20, 0.30, l1, size=14, color=INK, font=MONO)
add_text(p, cx + 8.10, cy + 4.14, 4.40, 0.70,
         '输出 JSON(真实片段,官方示例页)——ID 把每条记录拉回原图。',
         size=14, color=INK)

''')

# ---- 10. Part IV ----
s = block_replace(s, '# ---------- Part IV ----------', '# ---------- 备份页 ----------',
'''# ---------- Part IV(双栏) ----------
set_title_subtitle(part4, None, '结论与未来方向')
cx, cy, cw, ch = content_shapes(part4)
add_text(part4, cx, cy + 0.04, 5.60, 0.36, '系统交付什么', size=20, bold=True, color=NAVY)
add_text(part4, cx + 6.10, cy + 0.04, 5.45, 0.36, '未来方向', size=20, bold=True, color=NAVY)
div = part4.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(cx + 5.82), Inches(cy + 0.10), Inches(0.014), Inches(4.62))
div.fill.solid(); div.fill.fore_color.rgb = LINE; div.line.fill.background()
contribs = [
    ('引脚到网络的重建', '元件边界、引脚位置与导线路径共同决定哪些引脚同网——只在硬证据上合并。'),
    ('可追溯的 Target 导出', 'Component ID 与 pin ID 把每条属性、每个网络成员、每段走线都拉回原图。'),
    ('面向评分的输出校准', 'bbox 约定 · Name==key · 引脚库映射,全部对齐官方标注统计。'),
]
for i, (t, d) in enumerate(contribs):
    y = cy + 0.56 + i * 1.16
    dot = part4.shapes.add_shape(MSO_SHAPE.OVAL, Inches(cx), Inches(y + 0.02), Inches(0.30), Inches(0.30))
    dot.fill.solid(); dot.fill.fore_color.rgb = ORANGE; dot.line.fill.background()
    add_text(part4, cx, y + 0.03, 0.30, 0.30, str(i + 1), size=13, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    add_text(part4, cx + 0.44, y, 5.15, 0.32, t, size=17, bold=True, color=INK)
    add_text(part4, cx + 0.44, y + 0.36, 5.15, 0.70, d, size=14.5, color=MUTED)
add_text(part4, cx, cy + 4.10, 5.55, 0.60,
         'JunctionCNN:已完成自训但默认管线关闭——可选增强,不是推理主干。',
         size=14, italic=True, color=MUTED)
futures = [
    ('独立评估', '留出全新来源的图纸,测跨源泛化(陌生符号与绘制风格)。'),
    ('密集引脚与交叉走线', '审计漏检引脚与伪结点,定位局部图像证据的改进点。'),
    ('结点模型与数据增长', '验证后把已训练结点 CNN 转正;扩充难例与跨域数据。'),
]
for i, (t, d) in enumerate(futures):
    y = cy + 0.56 + i * 1.16
    dot = part4.shapes.add_shape(MSO_SHAPE.OVAL, Inches(cx + 6.10), Inches(y + 0.02), Inches(0.30), Inches(0.30))
    dot.fill.solid(); dot.fill.fore_color.rgb = CYAN; dot.line.fill.background()
    add_text(part4, cx + 6.10, y + 0.03, 0.30, 0.30, str(i + 1), size=13, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    add_text(part4, cx + 6.54, y, 5.00, 0.32, t, size=17, bold=True, color=INK)
    add_text(part4, cx + 6.54, y + 0.36, 5.00, 0.70, d, size=14.5, color=MUTED)
add_text(part4, cx + 6.10, cy + 4.10, 5.45, 0.60,
         '所报指标均来自公开开发图;官方 30~50 例集保持黑盒。',
         size=14, italic=True, color=MUTED)
add_card(part4, cx, cy + 4.80, cw, 0.50, fill=PANEL)
add_text(part4, cx + 0.2, cy + 4.90, cw - 0.4, 0.32,
         '工程交付:Docker pcb-parser:v102 · SHA256 清单 · 逐阶段审计日志 · 626+ 单元测试 · 零 LLM Token',
         size=14, bold=True, color=NAVY, align=PP_ALIGN.CENTER)

''')

# ---- 11. B1 ----
s = block_replace(s, '# B1', '# B2',
'''# B1
b1 = new_body('备份 A · 详细架构(11 阶段)', '主页压缩为四层(Part II)——零 LLM Token · 哈希锁定 · 全程审计')
p = b1
cx, cy, cw, ch = content_shapes(p)
place_pic(p, rf'{D}\\arch_pipeline_en.png', cx - 0.10, cy - 0.05, 7.55, 5.34)
place_pic(p, rf'{D}\\ti_ba_skeleton.png', cx + 7.75, cy - 0.05, 3.75, 3.21)
add_text(p, cx + 7.75, cy + 3.24, 3.75, 0.28, '骨架上的结点(U3 局部)', size=14, color=MUTED, align=PP_ALIGN.CENTER)
add_text(p, cx + 7.75, cy + 3.62, 3.75, 1.60,
         '工作导线掩膜 → 骨架:7210 px + 39 个结点(品红)。全局按网阶段见主页 P6;合并策略见 P7。(图内文字为英文)',
         size=14, color=INK)

''')

# ---- 12. B2 ----
s = block_replace(s, '# B2', '# B3',
'''# B2
b2 = new_body('备份 B · 模型训练与数据', '全部推理权重随镜像携带并 sha256 锁定;训练在提交包外完成')
p = b2
cx, cy, cw, ch = content_shapes(p)
rows = [
    ('模型', '架构', '训练数据', '关键参数', '来源'),
    ('YOLO nano 检测器', 'Ultralytics YOLO nano · 35 类', '238 训练页 + 180 局部图 · 59 验证页', 'imgsz1536 · 120 epochs · batch4', '自训(HR-DONK 热启动)'),
    ('WirePinUNet 双头', 'base32 共享 decoder(导线+引脚)', '源注释栅格化 + 人工复核掩码', '瓦片 512/384 · 40 epochs · pos_weight 8/20', '自训(milesial,GPL-3)'),
    ('JunctionCNN', '6 通道双视角 CNN', '1024 合成渲染 + 人工复核裁片×12', '30 epochs · 阈值 0.95 · 4 旋转一致', '自训(默认管线关闭)'),
    ('PP-OCRv6 OCR', 'det/rec small + cls(ONNX)', '开源预训练', '页级 2048px;引脚 2/3/4× × 0/90°', '直接调用(rapidocr 3.9.2)'),
]
tbl_shape = p.shapes.add_table(len(rows), 5, Inches(cx), Inches(cy + 0.05), Inches(11.55), Inches(3.30))
tbl = tbl_shape.table
for c, wd in zip(tbl.columns, [1.95, 2.30, 2.75, 2.60, 1.95]):
    c.width = Inches(wd)
for ri, row in enumerate(rows):
    for ci, val in enumerate(row):
        cell = tbl.cell(ri, ci); cell.text = val
        r0 = cell.text_frame.paragraphs[0].runs[0]
        r0.font.size = Pt(14); r0.font.bold = (ri == 0) or (ci == 0)
        r0.font.name = FONT
        r0.font.color.rgb = WHITE if ri == 0 else (NAVY if ci == 0 else INK)
        cell.fill.solid(); cell.fill.fore_color.rgb = NAVY if ri == 0 else (WHITE if ri % 2 else RGBColor(0xF6, 0xF9, 0xFC))
add_text(p, cx, cy + 3.62, cw, 0.34,
         '检测器按原始页划分:局部图父页只来自训练集;0001/0002 固定在验证(2462 个验证框)。',
         size=14, color=INK)
add_text(p, cx, cy + 4.02, cw, 0.34,
         '数据资产:1000 型号 EasyEDA 符号库 + 337 型号切片 + STM32F103C8T6 LQFP48 引脚库,全部内置。',
         size=14, color=INK)
add_text(p, cx, cy + 4.42, cw, 0.60,
         '溯源:baseline_provenance.json(HR-DONK,MIT)· milesial U-Net 积木 GPL-3(逐文件 sha256)· 训练数据无评测标注外泄。',
         size=14, color=MUTED)

''')

# ---- 13. B3 ----
s = block_replace(s, '# B3', '# B4',
'''# B3
b3 = new_body('备份 C · 运行与可靠性', '评测端 docker run;开发可用已装依赖的 Python 直跑')
p = b3
cx, cy, cw, ch = content_shapes(p)
add_card(p, cx, cy, 5.9, 2.75, fill=RGBColor(0x0D, 0x22, 0x37), line_c=RGBColor(0x1E, 0x4D, 0x8F))
add_text(p, cx + 0.25, cy + 0.2, 5.5, 2.4,
         [('docker run --rm --gpus all ' + chr(92), {'color': RGBColor(0x8F, 0xD8, 0xE2), 'font': MONO}),
          ('  -v <输入>:/input:ro -v <输出>:/output:rw ' + chr(92), {'color': RGBColor(0x8F, 0xD8, 0xE2), 'font': MONO}),
          ('  pcb-parser:v102 -input /input -output /output', {'color': RGBColor(0x8F, 0xD8, 0xE2), 'font': MONO}),
          ('', {}),
          ('# CPU 回退', {'color': MUTED, 'font': MONO}),
          ('python agent.py -input <dir> -output <dir> --device cpu', {'color': RGBColor(0x8F, 0xD8, 0xE2), 'font': MONO}),
          ('# 约 30 个 PCB_* 环境开关,默认值即调参结果', {'color': MUTED, 'font': MONO})],
         size=13)
place_pic(p, rf'{D}\\d_recovery_flow_en.png', cx + 6.1, cy + 0.1, 5.6, 2.6)
add_text(p, cx, cy + 2.95, cw, 0.28,
         '(右上流程图为英文;中文对照:阶段异常 → 冻结最近有效预测 → 校验后写入 → 整例失败抢救 → 记录原因)', size=13, color=MUTED)
chain = [('run.sh', '镜像入口'), ('agent.py', 'run_case 逐图'), ('S 阶段链', 'S14→S24 子进程'),
         ('finalize ×15', '证据门控修复'), ('adapter', 'target 校准'), ('final_results', 'Pred.json')]
for i, (t, d) in enumerate(chain):
    x = cx + i * 1.945
    cv = p.shapes.add_shape(MSO_SHAPE.CHEVRON, Inches(x), Inches(cy + 3.3), Inches(1.86), Inches(0.72))
    cv.fill.solid(); cv.fill.fore_color.rgb = PANEL; cv.line.color.rgb = LINE; cv.line.width = Pt(0.75)
    tf = cv.text_frame; tf.word_wrap = True
    tf.margin_left = Inches(0.14); tf.margin_right = Inches(0.03)
    pa = tf.paragraphs[0]; pa.alignment = PP_ALIGN.CENTER
    r = pa.add_run(); r.text = t; r.font.size = Pt(11); r.font.bold = True; r.font.color.rgb = INK; r.font.name = FONT
    pb = tf.add_paragraph(); pb.alignment = PP_ALIGN.CENTER
    r = pb.add_run(); r.text = d; r.font.size = Pt(10); r.font.color.rgb = MUTED; r.font.name = FONT
add_text(p, cx, cy + 4.15, cw, 0.30,
         '冒烟回放产物已归档:decision_log.json + final_results/rp2350_page11_Pred.json + logs/(20+ 审计 JSON)。',
         size=13.5, color=INK)
add_text(p, cx, cy + 4.47, cw, 0.30,
         '入口语义:镜像 ENTRYPOINT=/submission/run.sh(OCI config 已验证);agent_system 仍为官方占位。',
         size=13, color=MUTED)
add_text(p, cx, cy + 4.80, cw, 0.56,
         '工程可靠性:Docker 交付 · SHA256 完整性(249 文件清单、3 权重指纹、12 项预检)· 每图 20+ 审计日志 · 输出校验(validate_target + sanitize)· 626+ 单元测试 + 构建期门禁。',
         size=13.5, color=INK)
add_text(p, cx, cy + 5.40, cw, 0.28,
         '运行口径:CPU 单例 2000.5 秒(官方示例页);官方 GPU 机时未实测。',
         size=13, color=MUTED)

''')

# ---- 14. B4 ----
s = block_replace(s, '# B4', '# B5',
'''# B4
b4 = new_body('备份 D · 指标与消融', '每个数字可溯源;完整 17 行表见 defense_supplement.md §3')
p = b4
cx, cy, cw, ch = content_shapes(p)
add_text(p, cx, cy + 0.0, cw, 0.30,
         '口径警示:canonical(40 页)与页缩放(13 页)是独立实验;"+0.31"是总分增量。',
         size=13.5, color=MUTED)
add_text(p, cx, cy + 0.32, cw, 0.32,
         '本地开发回放:98.93(13 页公开集,v101-score)——仅开发用途,与官方 74 分不可比。',
         size=14, bold=True, color=INK)
rows = [
    ('实验', '数据范围', '指标', '结果'),
    ('检测配置选型', '59 例微平均', '检测 F1', '0.883(fused 胜出)'),
    ('引脚库映射', '13 页 released', 'PinF1', '0.031 → 0.549(+47)'),
    ('未决结点网放行', '13 页 released', '连通率', '0.188 → 0.612 @0.83'),
    ('断线网放行', '13 页 released', '连通 / 总分', '0.569→0.576 / +0.31 分'),
    ('结点归并', '页 13', 'wire TP/FP/FN · F1 · 总分', '32/28/17→46/13/3 · 58.72→85.19 · 82.871→92.656'),
    ('端子拓扑', '页 5', '连通 · 总分', '84.96%→100% · 93.954→99.474'),
    ('bbox canonical', '40 页公开子集', 'component F1 · 达标率', '.8641→.8999 · 0.905→1.000'),
    ('bbox 页缩放', '13 页 released', '20px 内模板元件', '150/163 → 159/163'),
    ('本机冒烟复现', '官方示例页', '输出规模 · bbox 对照', '89/165/63 · U3 差 1~2px'),
]
tbl_shape = p.shapes.add_table(len(rows), 4, Inches(cx), Inches(cy + 0.74), Inches(11.55), Inches(4.46))
tbl = tbl_shape.table
for c, wd in zip(tbl.columns, [2.65, 2.05, 2.55, 4.30]):
    c.width = Inches(wd)
for ri, row in enumerate(rows):
    for ci, val in enumerate(row):
        cell = tbl.cell(ri, ci); cell.text = val
        cell.margin_left = Inches(0.06); cell.margin_right = Inches(0.06)
        para = cell.text_frame.paragraphs[0]
        para.alignment = PP_ALIGN.RIGHT if ci == 3 else PP_ALIGN.LEFT
        r0 = para.runs[0]; r0.font.size = Pt(14)
        r0.font.bold = (ri == 0) or (ci == 3); r0.font.name = FONT
        r0.font.color.rgb = WHITE if ri == 0 else (NAVY if ci == 3 else INK)
        cell.fill.solid(); cell.fill.fore_color.rgb = NAVY if ri == 0 else (WHITE if ri % 2 else RGBColor(0xF6, 0xF9, 0xFC))

''')

# ---- 15. B5 ----
s = block_replace(s, '# B5', '# ---------- 删除要求页',
'''# B5
b5 = new_body('备份 E · 技术问答', '完整中文手册:EDA决赛答辩_QA.md(20 问,含依据)')
p = b5
cx, cy, cw, ch = content_shapes(p)
qas = [
    ('为什么不用大模型端到端?', '95% 权重在像素级结构精度;LLM 坐标不可靠;Token 越少分越高——我们为 0'),
    ('官方 74 与本地 98.93 的差距?', '口径不同:官方含 2/3 黑盒(60% 难例);本地是 13 页公开集回放'),
    ('T 型十字怎么处理?', '保守拆分 + 红点检测;学习型分类器保留为可选(默认关)'),
    ('断线 / 漏检怎么办?', 'gap 只记录不乱接;凭 60px 共线重合或红点+95% 墨迹恢复(+0.31 分)'),
    ('引脚号太小读不清?', '条带 OCR + 密行重读 + 6 视角定向 OCR + 1000 型号库(PinF1 0.031→0.549)'),
    ('型号是怎么恢复的?', 'SW1/SW2:错误回退 "SW1" → 真实型号 "PTS8153JK250SMTRLFS"(置信 0.998/0.992)'),
]
for i, (q, a) in enumerate(qas):
    col, row = i % 2, i // 2
    x = cx + col * 5.93
    y = cy + row * 1.62
    add_card(p, x, y, 5.62, 1.46, fill=WHITE)
    dot = p.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x + 0.14), Inches(y + 0.16), Inches(0.3), Inches(0.3))
    dot.fill.solid(); dot.fill.fore_color.rgb = NAVY; dot.line.fill.background()
    add_text(p, x + 0.14, y + 0.17, 0.3, 0.3, 'Q', size=13, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    add_text(p, x + 0.56, y + 0.10, 4.95, 0.32, q, size=14, bold=True, color=INK)
    add_text(p, x + 0.56, y + 0.46, 4.95, 0.94, a, size=14, color=MUTED)

''')

open(P, 'w', encoding='utf-8').write(s)
print('cn patch applied OK')
