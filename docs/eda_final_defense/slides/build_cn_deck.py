"""中文对照版答辩 PPT(仅用于队内阅读核对,非答辩提交版)。
与英文版 build_official_deck.py 同版式、同图片,文字全部为中文。
"""
import copy
import io
import os
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from PIL import Image

TPL = r'E:\AI_Workplace\精英挑战赛\EDA_Final_Defense\official_template.pptx'
OUT = r'E:\AI_Workplace\精英挑战赛\EDA_Final_Defense\EDA精英挑战赛_决赛答辩_中文对照版.pptx'
A = r'E:\AI_Workplace\精英挑战赛\_audit_workspace\assets_build'
D = r'E:\AI_Workplace\精英挑战赛\_audit_workspace\deck_workspace'

NAVY = RGBColor(0x12, 0x3C, 0x6D); CYAN = RGBColor(0x0E, 0x7C, 0x8C); ORANGE = RGBColor(0xF5, 0x9E, 0x0B)
INK = RGBColor(0x17, 0x26, 0x3B); MUTED = RGBColor(0x5A, 0x6B, 0x80); WHITE = RGBColor(0xFF, 0xFF, 0xFF)
PANEL = RGBColor(0xEE, 0xF4, 0xFA); LINE = RGBColor(0xC9, 0xD6, 0xE6)
FONT = '微软雅黑'; MONO = 'Consolas'

prs = Presentation(TPL)


def set_text(shape, text, size=None, bold=None, color=None, font=FONT, align=None):
    tf = shape.text_frame
    tf.word_wrap = True
    for p_ in list(tf.paragraphs[1:]):
        p_._p.getparent().remove(p_._p)
    para = tf.paragraphs[0]
    for r in list(para.runs):
        r._r.getparent().remove(r._r)
    r = para.add_run(); r.text = text
    if size: r.font.size = Pt(size)
    if bold is not None: r.font.bold = bold
    if color: r.font.color.rgb = color
    r.font.name = font
    r.font._rPr.set('{http://schemas.openxmlformats.org/drawingml/2006/main}ea', None) if False else None
    from pptx.oxml.ns import qn
    rPr = r._r.get_or_add_rPr()
    ea = rPr.find(qn('a:ea'))
    if ea is None:
        ea = rPr.makeelement(qn('a:ea'), {}); rPr.append(ea)
    ea.set('typeface', font)
    if align: para.alignment = align
    return shape


def add_text(slide, x, y, w, h, runs, size=18, bold=False, color=INK, align=PP_ALIGN.LEFT,
             anchor=MSO_ANCHOR.TOP, font=FONT, italic=False):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    lines = runs if isinstance(runs, list) else [(runs, {})]
    first = True
    from pptx.oxml.ns import qn
    for t, ov in lines:
        para = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        para.alignment = ov.get('align', align)
        r = para.add_run(); r.text = t
        r.font.size = Pt(ov.get('size', size))
        r.font.bold = ov.get('bold', bold)
        r.font.name = ov.get('font', font)
        r.font.color.rgb = ov.get('color', color)
        if ov.get('italic', italic): r.font.italic = True
        rPr = r._r.get_or_add_rPr()
        ea = rPr.find(qn('a:ea'))
        if ea is None:
            ea = rPr.makeelement(qn('a:ea'), {}); rPr.append(ea)
        ea.set('typeface', font)
    return tb


def place_pic(slide, path, bx, by, bw, bh):
    im = Image.open(path); pw, ph = im.size
    sc = min(bw / pw, bh / ph)
    w, h = pw * sc, ph * sc
    return slide.shapes.add_picture(path, Inches(bx + (bw - w) / 2), Inches(by + (bh - h) / 2), Inches(w), Inches(h))


def add_card(slide, x, y, w, h, fill=PANEL, line_c=LINE):
    sh = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    sh.fill.solid(); sh.fill.fore_color.rgb = fill
    sh.line.color.rgb = line_c; sh.line.width = Pt(1)
    try: sh.shadow.inherit = False
    except Exception: pass
    return sh


def chip(slide, x, y, w, h, big, small, big_c=ORANGE, big_size=24, small_size=12, fill=WHITE):
    add_card(slide, x, y, w, h, fill=fill)
    add_text(slide, x + 0.06, y + 0.06, w - 0.12, h * 0.52, big, size=big_size, bold=True,
             color=big_c, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    add_text(slide, x + 0.06, y + h * 0.55, w - 0.12, h * 0.4, small, size=small_size,
             color=MUTED, align=PP_ALIGN.CENTER)


TEMPLATE_SHAPE_NAMES = {'文本框 9', '内容占位符 5', '矩形: 圆角 17'}

def clone_body(prs_, src):
    layout = src.slide_layout
    dest = prs_.slides.add_slide(layout)
    for ph in list(dest.placeholders):
        ph._element.getparent().remove(ph._element)
    for sh in src.shapes:
        if sh.name not in TEMPLATE_SHAPE_NAMES:
            continue
        el = copy.deepcopy(sh._element)
        for cd in el.findall('.//' + '{http://schemas.openxmlformats.org/presentationml/2006/main}custDataLst'):
            cd.getparent().remove(cd)
        dest.shapes._spTree.append(el)
    return dest


def set_title_subtitle(page, title=None, subtitle=None):
    for sh in list(page.shapes):
        if sh.name == '文本框 9':
            sh.width = Inches(9.9)
            if title is not None:
                tf = sh.text_frame
                para = tf.paragraphs[0]
                for r in list(para.runs):
                    r._r.getparent().remove(r._r)
                r = para.add_run(); r.text = title
                r.font.size = Pt(24); r.font.bold = True; r.font.name = FONT
                from pptx.oxml.ns import qn
                rPr = r._r.get_or_add_rPr()
                ea = rPr.find(qn('a:ea'))
                if ea is None:
                    ea = rPr.makeelement(qn('a:ea'), {}); rPr.append(ea)
                ea.set('typeface', FONT)
        elif sh.name == '内容占位符 5' and subtitle is not None:
            sh.width = Inches(9.6)
            set_text(sh, subtitle, size=13, color=MUTED, font=FONT)
    return page


def content_shapes(page):
    return 0.95, 1.42, 11.55, 5.24


slides = list(prs.slides)
cover, toc, req, part1, part2, part3, part4, thanks = slides

# ---------- 封面 + 目录 ----------
for sh in cover.shapes:
    if sh.name == '标题 7':
        set_text(sh, 'AI 驱动的 PCB 原理图自动解析系统', size=32, bold=True, color=WHITE)
    elif sh.name == '副标题 8':
        set_text(sh, '2026 年 11 月 28 日', size=20, color=WHITE)
add_text(cover, 1.37, 3.92, 10.7, 0.4, '从像素到经验证的电气连接', size=18, color=RGBColor(0x8F, 0xD8, 0xE2))
for sh in list(toc.shapes):
    if sh.name == '标题' and sh.has_text_frame and sh.text_frame.text.strip() == '目录':
        sh._element.getparent().remove(sh._element)

# ---------- Part I ----------
set_title_subtitle(part1, None, '问题理解与解决策略')
cx, cy, cw, ch = content_shapes(part1)
add_text(part1, cx, cy + 0.02, cw, 0.6, '难点不在找到符号。', size=22, bold=True, color=NAVY)
tbx = part1.shapes[-1].text_frame
r = tbx.paragraphs[0].add_run(); r.text = '而在恢复电气连接。'
r.font.size = Pt(22); r.font.bold = True; r.font.color.rgb = ORANGE; r.font.name = FONT
add_text(part1, cx, cy + 0.64, cw, 0.36,
         [('感知', {'size': 15, 'bold': True, 'color': CYAN})], size=15)
tbx = part1.shapes[-1].text_frame
r = tbx.paragraphs[0].add_run(); r.text = '  识别元件、文字、引脚与导线      '; r.font.size = Pt(13); r.font.color.rgb = MUTED; r.font.name = FONT
r = tbx.paragraphs[0].add_run(); r.text = '连接'; r.font.size = Pt(15); r.font.bold = True; r.font.color.rgb = NAVY; r.font.name = FONT
r = tbx.paragraphs[0].add_run(); r.text = '  建立几何关系与电气拓扑      '; r.font.size = Pt(13); r.font.color.rgb = MUTED; r.font.name = FONT
r = tbx.paragraphs[0].add_run(); r.text = '验证'; r.font.size = Pt(15); r.font.bold = True; r.font.color.rgb = ORANGE; r.font.name = FONT
r = tbx.paragraphs[0].add_run(); r.text = '  用证据修复并校准输出'; r.font.size = Pt(13); r.font.color.rgb = MUTED; r.font.name = FONT
add_text(part1, cx, cy + 1.08, cw, 0.36,
         '输入:一张原理图 PNG   →   输出:结构化 JSON(components / pins / nets)', size=15, color=INK)
ch_items = [
    ('密集元器件检测', '小型贴片器件易漏检、类别混淆', '双路 YOLO(整图+SAHI 切片)+ WBF 融合', '逐类置信阈值,检测 F1 0.883'),
    ('引脚定位与归属', '多引脚 IC 易错位、漏检、绑定错误', '双头 U-Net 热图 + 代价最优指派', '峰解码 + 亚像素修正'),
    ('文字与型号关联', '标注拥挤、倾斜', '多视角 OCR 证据门控', '≥2 视角一致 @ 置信 0.98,预算制'),
    ('拓扑与连通性', 'T 型十字、断线、飞线', '保守建图 + 事务式修复', '先防连错,再补漏连'),
]
row_h = 0.98
for i, (ct, cd, st, sd) in enumerate(ch_items):
    y = cy + 1.12 + i * (row_h + 0.09)
    add_card(part1, cx, y, 5.28, row_h, fill=WHITE)
    add_text(part1, cx + 0.14, y + 0.08, 5.0, 0.36, ct, size=16, bold=True, color=NAVY)
    add_text(part1, cx + 0.14, y + 0.48, 5.0, 0.42, cd, size=13.5, color=MUTED)
    ar = part1.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(cx + 5.44), Inches(y + row_h / 2 - 0.14), Inches(0.5), Inches(0.28))
    ar.fill.solid(); ar.fill.fore_color.rgb = CYAN; ar.line.fill.background()
    add_card(part1, cx + 6.1, y, 5.4, row_h, fill=PANEL)
    add_text(part1, cx + 6.24, y + 0.08, 5.15, 0.36, st, size=16, bold=True, color=INK)
    add_text(part1, cx + 6.24, y + 0.48, 5.15, 0.42, sd, size=13.5, color=MUTED)

# ---------- Part II:四层架构 ----------
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
    add_text(part2, cx + 4.05, y + 0.2, 7.3, 0.76, d, size=15, color=INK)
    if i < 3:
        da = part2.shapes.add_shape(MSO_SHAPE.DOWN_ARROW, Inches(cx + 1.55), Inches(y + 1.15), Inches(0.26), Inches(0.13))
        da.fill.solid(); da.fill.fore_color.rgb = CYAN; da.line.fill.background()
add_text(part2, cx, cy + 5.12, cw, 0.32,
         '完整 11 阶段见备份页 A · 零 LLM Token · 逐阶段哈希锁定与审计', size=12, color=MUTED)

# ---------- 正文页克隆函数 ----------
def new_body(title, subtitle):
    pg = clone_body(prs, part2)
    set_title_subtitle(pg, title, subtitle)
    return pg

# ---- P5 视觉感知 ----
pP3 = new_body('视觉感知', '三个默认启用的模型产生候选;结论由证据门控给出')
p = pP3
cx, cy, cw, ch = content_shapes(p)
panels = [
    (rf'{D}\p5_input.png', '输入原理图 PNG(官方示例页)'),
    (rf'{D}\p5_wire.png', '导线概率图(阈值 0.35)'),
    (rf'{D}\p5_pin.png', '引脚热图 → 431 个候选'),
]
for i, (img, cap) in enumerate(panels):
    x = cx + i * 3.94
    place_pic(p, img, x, cy, 3.78, 2.68)
    add_text(p, x, cy + 2.74, 3.78, 0.3, cap, size=12, color=MUTED, align=PP_ALIGN.CENTER)
add_text(p, cx, cy + 3.14, cw, 0.38,
         [('解码出 431 个引脚候选:  ', {'size': 16, 'bold': True, 'color': INK})], size=16)
tbx = p.shapes[-1].text_frame
r = tbx.paragraphs[0].add_run(); r.text = '287'; r.font.size = Pt(16); r.font.bold = True; r.font.color.rgb = NAVY; r.font.name = FONT
r = tbx.paragraphs[0].add_run(); r.text = ' 已归属 · '; r.font.size = Pt(14); r.font.color.rgb = MUTED; r.font.name = FONT
r = tbx.paragraphs[0].add_run(); r.text = '115'; r.font.size = Pt(16); r.font.bold = True; r.font.color.rgb = NAVY; r.font.name = FONT
r = tbx.paragraphs[0].add_run(); r.text = ' 歧义 · '; r.font.size = Pt(14); r.font.color.rgb = MUTED; r.font.name = FONT
r = tbx.paragraphs[0].add_run(); r.text = '29'; r.font.size = Pt(16); r.font.bold = True; r.font.color.rgb = NAVY; r.font.name = FONT
r = tbx.paragraphs[0].add_run(); r.text = ' 无主'; r.font.size = Pt(14); r.font.color.rgb = MUTED; r.font.name = FONT
flow = [('整图 YOLO', '大器件精度'), ('SAHI 切片', '小器件召回'), ('WBF 融合', '置信加权合并')]
for i, (t, d) in enumerate(flow):
    x = cx + i * 3.2
    add_card(p, x, cy + 3.62, 3.0, 0.92, fill=PANEL)
    add_text(p, x + 0.12, cy + 3.68, 2.8, 0.34, t, size=15, bold=True, color=NAVY)
    add_text(p, x + 0.12, cy + 4.04, 2.8, 0.32, d, size=11.5, color=MUTED)
    if i < 2:
        ar = p.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(x + 3.02), Inches(cy + 3.94), Inches(0.34), Inches(0.26))
        ar.fill.solid(); ar.fill.fore_color.rgb = CYAN; ar.line.fill.background()
chip(p, cx + 9.85, cy + 3.62, 1.7, 0.92, 'F1 0.883', '检测 · 59 例', ORANGE, 19, 10.5)
origins = [
    ('YOLO nano', '自研训练 · F1 0.883(59 例)', 'WirePinUNet', '自研训练(milesial U-Net 积木)'),
    ('PP-OCRv6', '直接调用开源预训练(离线)', 'JunctionCNN', '自研训练,默认关闭(可选)'),
]
for i, (t1, d1, t2, d2) in enumerate(origins):
    y = cy + 4.68 + i * 0.38
    dot = p.shapes.add_shape(MSO_SHAPE.OVAL, Inches(cx + 0.05), Inches(y + 0.06), Inches(0.12), Inches(0.12))
    dot.fill.solid(); dot.fill.fore_color.rgb = NAVY; dot.line.fill.background()
    add_text(p, cx + 0.26, y, 1.85, 0.3, t1, size=12.5, bold=True, color=INK)
    add_text(p, cx + 2.15, y + 0.02, 3.55, 0.3, d1, size=11, color=MUTED)
    dot2 = p.shapes.add_shape(MSO_SHAPE.OVAL, Inches(cx + 5.95), Inches(y + 0.06), Inches(0.12), Inches(0.12))
    dot2.fill.solid(); dot2.fill.fore_color.rgb = ORANGE; dot2.line.fill.background()
    add_text(p, cx + 6.16, y, 1.95, 0.3, t2, size=12.5, bold=True, color=INK)
    add_text(p, cx + 8.15, y + 0.02, 4.35, 0.3, d2, size=11, color=MUTED)

# ---- P6 拓扑重建 ----
pP4 = new_body('拓扑重建', '导线概率图 → 骨架 → 线段 → 网络')
p = pP4
cx, cy, cw, ch = content_shapes(p)
add_text(p, cx, cy, cw, 0.4, '先避免连错,再恢复漏连', size=17, bold=True, color=NAVY)
place_pic(p, rf'{A}\fig3c_topology_real.png', cx, cy + 0.5, 4.01, 2.85)
add_text(p, cx, cy + 3.4, 4.05, 0.3, '全局上下文:导线骨架 + 39 个结点', size=11.5, color=MUTED, align=PP_ALIGN.CENTER)
place_pic(p, rf'{D}\p6_zoom_skeleton.png', cx + 4.25, cy + 0.5, 3.32, 2.85)
add_text(p, cx + 4.25, cy + 3.4, 3.35, 0.3, '局部放大:骨架上的结点', size=11.5, color=MUTED, align=PP_ALIGN.CENTER)
place_pic(p, rf'{D}\p6_zoom_networks.png', cx + 7.85, cy + 0.5, 3.32, 2.85)
add_text(p, cx + 7.85, cy + 3.4, 3.35, 0.3, '局部放大:按网着色的连通', size=11.5, color=MUTED, align=PP_ALIGN.CENTER)
cases = [
    ('T 型 / 十字结点', '保守拆分,显式不确定'),
    ('断线', '仅凭硬证据恢复'),
    ('网络标签(飞线)', '同名合并;零长度边'),
    ('电源网(VCC/GND)', '合并前过四道门'),
]
for i, (t, s2) in enumerate(cases):
    x = cx + i * 2.94
    add_card(p, x, cy + 3.85, 2.78, 0.95, fill=PANEL)
    add_text(p, x + 0.12, cy + 3.9, 2.55, 0.3, t, size=12.5, bold=True, color=NAVY)
    add_text(p, x + 0.12, cy + 4.22, 2.55, 0.54, s2, size=11, color=INK)
add_text(p, cx, cy + 4.95, cw, 0.32,
         '238 个物理网络候选 / 362 条边 → 63 个逻辑网络——候选是建图阶段中间态;合并后得到最终输出。',
         size=12, color=INK)

# ---- P7 不靠猜测的修复 ----
pP5 = new_body('不靠猜测的修复', '页 13 结点归并——只在硬证据下合并')
p = pP5
cx, cy, cw, ch = content_shapes(p)
place_pic(p, rf'{D}\p7_gap_trim.png', cx, cy + 0.1, 3.6, 3.6)
add_text(p, cx, cy + 3.78, 3.65, 0.3, '页 13 真实裁切(开发记录)', size=11.5, color=MUTED, align=PP_ALIGN.CENTER)
add_text(p, cx + 3.95, cy + 0.05, 8.55, 0.5, '页 13 导线修复', size=20, bold=True, color=NAVY)
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
    '合并证据:≥60px 同线重合(±3°、垂距≤2px),或红点 + ≥95% 墨迹支持',
    '只合并成员与边——元件与引脚不动;每页 ≤8 组',
]
for i, t in enumerate(mech):
    add_text(p, cx + 3.95, cy + 2.85 + i * 0.4, 8.55, 0.36, t, size=12.5, color=INK)
add_text(p, cx + 3.95, cy + 3.72, 8.55, 0.3,
         '其余 12 页逐字节不变(冻结缓存回放)——单页开发集消融,非官方整体指标。',
         size=12, italic=True, color=MUTED)
add_text(p, cx + 3.95, cy + 4.15, 8.55, 0.34,
         [('导线 F1 提升来自结点归并机制——与 OCR 置信门控无关。', {'size': 13, 'bold': True, 'color': NAVY})], size=13)
add_text(p, cx + 3.95, cy + 4.6, 8.55, 0.34,
         [('证据门控 OCR(≥2 视角 @0.98)是另一项独立能力——见备份页 E。', {'size': 12.5, 'color': MUTED})], size=12.5)

# ---- P8 输出校准 ----
pP6 = new_body('面向官方评测的输出校准',
               '对官方标注约定做统计校准——不是逐图查表')
p = pP6
cx, cy, cw, ch = content_shapes(p)
cards = [
    ('类型与字段约定', 'resistor → r …(145 个匹配元件验证)· Name==key 约定来自 9186 组件统计(box 保留型号)', CYAN),
    ('bbox 校准(两套口径)', 'canonical(40 页公开子集):20px 达标率 0.905 → 1.000 · 按页缩放(13 页 released):150/163 → 159/163', NAVY),
    ('引脚尖端吸附', '18px 吸附(实测 16-22px 平台区)+ 1000 型号引脚库', ORANGE),
]
for i, (t, d, c) in enumerate(cards):
    y = cy + i * 1.3
    add_card(p, cx, y, 6.6, 1.15, fill=WHITE)
    add_text(p, cx + 0.18, y + 0.08, 6.2, 0.36, t, size=15, bold=True, color=INK)
    add_text(p, cx + 0.18, y + 0.5, 5.9, 0.6, d, size=12.5, color=MUTED)
    dot = p.shapes.add_shape(MSO_SHAPE.OVAL, Inches(cx + 6.24), Inches(y + 0.5), Inches(0.16), Inches(0.16))
    dot.fill.solid(); dot.fill.fore_color.rgb = c; dot.line.fill.background()
add_text(p, cx, cy + 3.98, 6.7, 0.6,
         '两套 bbox 口径相互独立——不可相乘。\n完整指标表见备份页 D。',
         size=12.5, color=MUTED)
place_pic(p, rf'{D}\p8_u3_bbox_compare.png', cx + 7.0, cy, 1.5, 3.5)
add_text(p, cx + 8.75, cy + 0.35, 3.75, 0.62, '1~2 px', size=32, bold=True, color=ORANGE)
add_text(p, cx + 8.75, cy + 1.05, 3.8, 0.66,
         'U3(RP2350A)边界框与官方示例标注之差(官方容差 20px)', size=12.5, color=INK)
add_text(p, cx + 8.75, cy + 1.78, 3.8, 0.6,
         '我方 [539,281,689,731]\n官方 [540,280,690,730]', size=12, color=MUTED, font=MONO)
add_text(p, cx + 7.0, cy + 3.95, 5.55, 0.3, '输出 JSON(真实片段)', size=12.5, bold=True, color=INK)
json_rows = [
    ('components', '"U3": box RP2350A · bbox [539,281,689,731]', RGBColor(0x0E, 0x7C, 0x8C)),
    ('pins', '"U3".pin_51: USB_DM @ (513, 561)', NAVY),
    ('nets', '"LOG_NET": (C1.2, U3.U3) · 3 条边', ORANGE),
]
for i, (key, txt, c) in enumerate(json_rows):
    y = cy + 4.32 + i * 0.55
    kc = p.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(cx + 7.0), Inches(y), Inches(1.5), Inches(0.4))
    kc.fill.solid(); kc.fill.fore_color.rgb = c; kc.line.fill.background()
    try: kc.shadow.inherit = False
    except Exception: pass
    ktf = kc.text_frame; ktf.word_wrap = True
    kpa = ktf.paragraphs[0]; kpa.alignment = PP_ALIGN.CENTER
    kr = kpa.add_run(); kr.text = key; kr.font.size = Pt(11.5); kr.font.bold = True; kr.font.color.rgb = WHITE; kr.font.name = FONT
    add_text(p, cx + 8.65, y + 0.03, 3.95, 0.3, txt, size=11.5, color=INK, font=MONO)

# ---------- Part III ----------
set_title_subtitle(part3, None, '官方评测成绩与定量证据')
cx, cy, cw, ch = content_shapes(part3)
chip(part3, cx + 0.1, cy + 0.15, 3.6, 1.9, '74', '官方评测成绩', ORANGE, 54, 14, fill=WHITE)
chip(part3, cx + 3.9, cy + 0.15, 3.6, 1.9, '47', '全国排名', NAVY, 54, 14, fill=WHITE)
chip(part3, cx + 7.7, cy + 0.15, 3.6, 1.9, '前 61 强', '晋级决赛', NAVY, 44, 14, fill=WHITE)
add_card(part3, cx, cy + 2.3, 11.55, 1.42, fill=PANEL)
add_text(part3, cx + 0.2, cy + 2.4, 11.2, 0.4, '版本迭代', size=16, bold=True, color=INK)
tbx = part3.shapes[-1].text_frame
r = tbx.paragraphs[0].add_run(); r.text = '   28.92'; r.font.size = Pt(16); r.font.bold = True; r.font.color.rgb = NAVY; r.font.name = FONT
r = tbx.paragraphs[0].add_run(); r.text = '(旧版 8422)   →   '; r.font.size = Pt(13); r.font.color.rgb = MUTED; r.font.name = FONT
r = tbx.paragraphs[0].add_run(); r.text = '74'; r.font.size = Pt(16); r.font.bold = True; r.font.color.rgb = ORANGE; r.font.name = FONT
r = tbx.paragraphs[0].add_run(); r.text = '(最终提交,+45)'; r.font.size = Pt(13); r.font.color.rgb = MUTED; r.font.name = FONT
add_text(part3, cx + 0.2, cy + 2.82, 11.2, 0.8,
         '提升来源:无位号电源/地出口 · LED 文字与框 · IC 型号优先级与实框 · 工程编号 · 导出语义。',
         size=12.5, color=INK)
add_text(part3, cx, cy + 3.95, cw, 0.36,
         '本地开发回放分(13 页公开集):移至备份页 D——仅开发用途,与官方结果不可直接比较。',
         size=12, color=MUTED)

# ---- P10 定量结果 ----
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
    add_text(p, x + 0.2, y + 1.25, 5.2, 0.3, met, size=12.5, color=INK)
    add_text(p, x + 0.2, y + 1.55, 5.2, 0.3, scope, size=11.5, color=MUTED)
add_text(p, cx, cy + 4.7, cw, 0.34,
         '开发集实验与单页消融——非官方分项成绩。完整表见备份页 D。',
         size=13, bold=True, color=INK)
add_text(p, cx, cy + 5.08, cw, 0.3,
         '消融协议:冻结模型/OCR 缓存,只开一个模块;其余 12 页逐字节不变。',
         size=11.5, color=MUTED)

# ---- P11 端到端 ----
pP9 = new_body('端到端输出与工程验证',
               '官方示例页,完整管线——从 PNG 到校验通过的 JSON')
p = pP9
cx, cy, cw, ch = content_shapes(p)
place_pic(p, rf'{A}\fig1b_detection_clean.png', cx, cy, 4.09, 2.9)
add_text(p, cx, cy + 2.94, 4.15, 0.28, '检测与识别标注(32 个绘图类元件)', size=11, color=MUTED, align=PP_ALIGN.CENTER)
place_pic(p, rf'{A}\fig3b_network_overlay.png', cx + 4.35, cy, 4.09, 2.9)
add_text(p, cx + 4.35, cy + 2.94, 4.15, 0.28, '最终网络:63 个逻辑网,按网着色', size=11, color=MUTED, align=PP_ALIGN.CENTER)
stats = [('89', '元器件'), ('165', '引脚'), ('63', '逻辑网络')]
for i, (n, lb) in enumerate(stats):
    y = cy + 0.06 + i * 0.94
    add_text(p, cx + 9.15, y, 2.2, 0.52, n, size=30, bold=True, color=NAVY)
    add_text(p, cx + 9.15, y + 0.54, 2.2, 0.28, lb, size=11.5, color=MUTED)
add_text(p, cx + 9.15, cy + 2.92, 2.3, 0.28, 'U3 bbox 差', size=11.5, color=INK)
add_text(p, cx + 9.15, cy + 3.18, 2.3, 0.44, '1~2 px', size=20, bold=True, color=ORANGE)
add_text(p, cx, cy + 3.28, cw, 0.32, '输出 JSON(真实片段,官方示例页)', size=12.5, bold=True, color=INK)
json_rows = [
    ('components', '"U3": box RP2350A · bbox [539,281,689,731]', RGBColor(0x0E, 0x7C, 0x8C)),
    ('pins', '"U3".pin_51: USB_DM @ (513, 561)', NAVY),
    ('nets', '"LOG_NET": (C1.2, U3.U3) · 3 条边', ORANGE),
]
for i, (key, l1, c) in enumerate(json_rows):
    y = cy + 3.62 + i * 0.52
    kc = p.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(cx), Inches(y), Inches(1.42), Inches(0.44))
    kc.fill.solid(); kc.fill.fore_color.rgb = c; kc.line.fill.background()
    try: kc.shadow.inherit = False
    except Exception: pass
    ktf = kc.text_frame; ktf.word_wrap = True
    kpa = ktf.paragraphs[0]; kpa.alignment = PP_ALIGN.CENTER
    kr = kpa.add_run(); kr.text = key; kr.font.size = Pt(11.5); kr.font.bold = True; kr.font.color.rgb = WHITE; kr.font.name = FONT
    add_text(p, cx + 1.6, y + 0.04, 4.35, 0.3, l1, size=12, color=INK, font=MONO)
add_text(p, cx + 5.95, cy + 3.7, 6.55, 0.4, '经校验的结构化输出——最终交付物', size=17, bold=True, color=NAVY)
add_text(p, cx + 5.95, cy + 4.16, 6.55, 0.36,
         'validate_target + sanitize:只删除无效对象,绝不推断文本、坐标或连接。',
         size=13, color=INK)

# ---------- Part IV ----------
set_title_subtitle(part4, None, '结论与未来方向')
cx, cy, cw, ch = content_shapes(part4)
add_text(part4, cx, cy + 0.02, cw, 0.42, '三项技术贡献', size=19, bold=True, color=NAVY)
contribs = [
    ('多模型视觉感知', '自训 YOLO nano / 双头 WirePinUNet / 结点 CNN——全部 sha256 可溯'),
    ('证据门控拓扑修复', '0.98×2 视角门控 + 事务式引擎 + 回滚'),
    ('面向评分的输出校准', 'bbox 约定 / Name==key / 引脚库映射'),
]
for i, (t, d) in enumerate(contribs):
    y = cy + 0.52 + i * 0.56
    dot = part4.shapes.add_shape(MSO_SHAPE.OVAL, Inches(cx), Inches(y + 0.03), Inches(0.28), Inches(0.28))
    dot.fill.solid(); dot.fill.fore_color.rgb = ORANGE; dot.line.fill.background()
    add_text(part4, cx, y + 0.04, 0.28, 0.28, str(i + 1), size=11, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    add_text(part4, cx + 0.4, y - 0.02, 11.1, 0.3, t, size=14.5, bold=True, color=INK)
    add_text(part4, cx + 0.4, y + 0.27, 11.1, 0.3, d, size=11, color=MUTED)
add_text(part4, cx, cy + 2.22, cw, 0.3,
         'JunctionCNN:已完成自训但默认管线关闭——可选增强,不是默认推理主干。',
         size=12.5, italic=True, color=MUTED)
add_text(part4, cx, cy + 2.6, cw, 0.42, '工程价值', size=19, bold=True, color=NAVY)
ev = [
    ('官方基础镜像上的 Docker 交付', 'SHA256 完整性:249 文件清单 + 权重指纹'),
    ('每图 20+ 份审计日志', 'validate/sanitize + 回滚 · 626+ 单元测试 · 零 LLM Token'),
]
for i, (a, b) in enumerate(ev):
    col, row = i % 2, i // 2
    x = cx + col * 5.95
    y = cy + 3.06 + row * 0.42
    dot = part4.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x), Inches(y + 0.06), Inches(0.12), Inches(0.12))
    dot.fill.solid(); dot.fill.fore_color.rgb = NAVY; dot.line.fill.background()
    add_text(part4, x + 0.24, y, 5.6, 0.32, a, size=12.5, color=INK)
    add_text(part4, x + 0.24, y + 0.3, 5.6, 0.32, b, size=11, color=MUTED)
add_text(part4, cx, cy + 4.02, cw, 0.42, '未来方向', size=19, bold=True, color=NAVY)
outs = ['扩充难例数据与域适配', '提升结点分类(将已训练 CNN 转正)',
        '增强跨工程图纸泛化']
for i, t in enumerate(outs):
    y = cy + 4.48 + i * 0.34
    dot = part4.shapes.add_shape(MSO_SHAPE.OVAL, Inches(cx + 0.06), Inches(y + 0.07), Inches(0.12), Inches(0.12))
    dot.fill.solid(); dot.fill.fore_color.rgb = CYAN; dot.line.fill.background()
    add_text(part4, cx + 0.3, y, 11.2, 0.3, t, size=12.5, color=INK)

# ---------- 备份页 ----------
# B1
b1 = new_body('备份 A · 详细架构(11 阶段)', '主页压缩为四层(Part II)——零 LLM Token · 哈希锁定 · 全程审计')
p = b1
cx, cy, cw, ch = content_shapes(p)
place_pic(p, rf'{D}\arch_pipeline_en.png', cx - 0.1, cy - 0.05, cw + 0.2, ch + 0.1)
add_text(p, cx, cy + ch - 0.08, cw, 0.28, '(图内文字为英文;中文对照见主页四层架构与本页各阶段名称)', size=10.5, color=MUTED)

# B2
b2 = new_body('备份 B · 模型训练与数据', '全部推理权重随镜像携带并 sha256 锁定;训练在提交包外完成')
p = b2
cx, cy, cw, ch = content_shapes(p)
rows = [
    ('模型', '架构', '训练数据', '关键参数', '来源'),
    ('YOLO nano 检测器', 'Ultralytics YOLO nano · 35 类', 'detection_balanced_v2(本地)', 'imgsz1536 · 120 epochs · batch4', '自训(v2 热启动;HR-DONK 仅初始基线)'),
    ('WirePinUNet 双头', 'base32 共享 decoder(导线+引脚)', '源注释栅格化 + 人工复核掩码', '瓦片 512/384 · 40 epochs · pos_weight 8/20 · epoch32 入选', '自训(milesial 积木,GPL-3 溯源)'),
    ('JunctionCNN', '6 通道双视角 CNN', '1024 合成渲染 + 人工复核裁片×12', '30 epochs · 阈值 0.95 · 4 旋转一致', '自训(默认管线关闭,可选)'),
    ('PP-OCRv6 OCR', 'det/rec small + cls(ONNX)', '开源预训练', '页级 2048px;引脚 2/3/4× × 0/90°', '直接调用(rapidocr 3.9.2)'),
]
tbl_shape = p.shapes.add_table(len(rows), 5, Inches(cx), Inches(cy + 0.05), Inches(11.55), Inches(3.4))
tbl = tbl_shape.table
for c, wd in zip(tbl.columns, [2.0, 2.5, 2.6, 2.5, 1.95]):
    c.width = Inches(wd)
for ri, row in enumerate(rows):
    for ci, val in enumerate(row):
        cell = tbl.cell(ri, ci); cell.text = val
        r0 = cell.text_frame.paragraphs[0].runs[0]
        r0.font.size = Pt(11 if ri else 12); r0.font.bold = (ri == 0) or (ci == 0)
        r0.font.name = FONT
        r0.font.color.rgb = WHITE if ri == 0 else (NAVY if ci == 0 else INK)
        cell.fill.solid(); cell.fill.fore_color.rgb = NAVY if ri == 0 else (WHITE if ri % 2 else RGBColor(0xF6, 0xF9, 0xFC))
add_text(p, cx, cy + 3.7, cw, 0.4,
         '数据资产:1000 型号 EasyEDA 符号库 + 337 型号切片 + STM32F103C8T6 LQFP48 引脚库(ST DS5319 数据表)——全部随镜像内置,零联网。',
         size=11.5, color=INK)
add_text(p, cx, cy + 4.14, cw, 0.4,
         '溯源:weights/baseline_provenance.json(HR-DONK 初始基线,MIT)· vendor/pytorch_unet provenance(milesial,GPL-3,逐文件 sha256)· 训练数据无评测标注外泄。',
         size=11, color=MUTED)

# B3
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
         size=11.5)
place_pic(p, rf'{D}\d_recovery_flow_en.png', cx + 6.1, cy + 0.1, 5.6, 2.6)
add_text(p, cx, cy + 2.95, cw, 0.28,
         '(右上流程图为英文;中文对照:阶段异常 → 冻结最近有效预测 → 校验后写入 → 整例失败抢救 → 记录原因)', size=10.5, color=MUTED)
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
add_text(p, cx, cy + 4.15, cw, 0.3,
         '冒烟回放产物已归档:decision_log.json + final_results/rp2350_page11_Pred.json + logs/(20+ 审计 JSON)。',
         size=11.5, color=INK)
add_text(p, cx, cy + 4.5, cw, 0.3,
         '入口语义:镜像 ENTRYPOINT=/submission/run.sh(OCI config 已验证);agent_system 仍为官方占位——请与组委会确认调用方式。',
         size=11, color=MUTED)
add_text(p, cx, cy + 4.88, cw, 0.44,
         '工程可靠性:Docker 交付 · SHA256 完整性(249 文件清单、3 权重指纹、12 项预检)· 每图 20+ 审计日志 · 输出校验(validate_target + sanitize)· 626+ 单元测试 + 构建期门禁。',
         size=11.5, color=INK)
add_text(p, cx, cy + 5.36, cw, 0.28,
         '运行口径:CPU 单例 2000.5 秒(官方示例页);官方 GPU 机时未实测——不宣称满足 2 小时。',
         size=12, color=MUTED)

# B4
b4 = new_body('备份 D · 指标与消融', '每个数字可溯源;完整 17 行表见 defense_supplement.md §3')
p = b4
cx, cy, cw, ch = content_shapes(p)
add_text(p, cx, cy + 0.0, cw, 0.3,
         '口径警示:canonical(40 页)与页缩放(13 页)是独立实验;"+0.31"是总分增量(连通 +0.7pp)。',
         size=12, color=MUTED)
add_text(p, cx, cy + 0.32, cw, 0.3,
         '本地开发回放:98.93(13 页公开集,v101-score)——仅开发用途,与官方 74 分不可比。',
         size=12, color=INK)
rows = [
    ('实验', '数据范围', '指标', '结果'),
    ('检测配置选型', '59 例公开微平均', '检测 F1', '0.883(fused 胜出)'),
    ('引脚库映射', '13 页 released', 'PinF1', '0.031 → 0.549(+47)'),
    ('未决结点网放行', '13 页 released', '连通率', '0.188 → 0.612 @0.83'),
    ('断线网放行', '13 页 released', '连通 / 总分', '0.569→0.576 / +0.31 分'),
    ('结点归并', '页 13', 'wire TP/FP/FN · F1 · 总分', '32/28/17→46/13/3 · 58.72→85.19 · 82.871→92.656'),
    ('端子拓扑', '页 5', '连通 · 总分', '84.96%→100% · 93.954→99.474'),
    ('bbox canonical', '40 页公开子集', 'component F1 · 达标率', '.8641→.8999 · 0.905→1.000'),
    ('bbox 页缩放', '13 页 released', '20px 内模板元件', '150/163 → 159/163'),
    ('本机冒烟复现', '官方示例页', '输出规模 · bbox 对照', '89/165/63 · U3 差 1~2px'),
]
tbl_shape = p.shapes.add_table(len(rows), 4, Inches(cx), Inches(cy + 0.72), Inches(11.55), Inches(4.5))
tbl = tbl_shape.table
for c, wd in zip(tbl.columns, [2.6, 2.4, 2.75, 3.8]):
    c.width = Inches(wd)
for ri, row in enumerate(rows):
    for ci, val in enumerate(row):
        cell = tbl.cell(ri, ci); cell.text = val
        para = cell.text_frame.paragraphs[0]
        para.alignment = PP_ALIGN.RIGHT if ci == 3 else PP_ALIGN.LEFT
        r0 = para.runs[0]; r0.font.size = Pt(14 if ri == 0 else 13.5)
        r0.font.bold = (ri == 0) or (ci == 3); r0.font.name = FONT
        r0.font.color.rgb = WHITE if ri == 0 else (NAVY if ci == 3 else INK)
        cell.fill.solid(); cell.fill.fore_color.rgb = NAVY if ri == 0 else (WHITE if ri % 2 else RGBColor(0xF6, 0xF9, 0xFC))

# B5
b5 = new_body('备份 E · 技术问答', '完整中文手册:EDA决赛答辩_QA.md(18 问,含依据)')
p = b5
cx, cy, cw, ch = content_shapes(p)
qas = [
    ('为什么不用大模型端到端?', '95% 权重在像素级结构精度;LLM 坐标不可靠;Token 越少分越高——我们为 0'),
    ('官方 74 与本地 98.93 的差距?', '口径不同:官方含 2/3 黑盒(60% 难例);本地是 13 页公开集回放'),
    ('T 型十字怎么处理?', '保守拆分 + 红点检测;学习型分类器保留为可选(默认关)'),
    ('断线 / 漏检怎么办?', 'gap 只记录不乱接;凭 60px 共线重合或红点+95% 墨迹恢复(+0.31 分)'),
    ('引脚号太小读不清?', '条带 OCR(引线外端 30-50px)+ 密行重读 + 6 视角定向 OCR + 1000 型号库(PinF1 0.031→0.549)'),
    ('型号是怎么恢复的?', 'SW1/SW2:错误回退 "SW1" → 真实型号 "PTS8153JK250SMTRLFS",邻近文本堆栈、旋转视角、置信 0.998/0.992'),
]
for i, (q, a) in enumerate(qas):
    col, row = i % 2, i // 2
    x = cx + col * 5.93
    y = cy + row * 1.62
    add_card(p, x, y, 5.62, 1.46, fill=WHITE)
    dot = p.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x + 0.14), Inches(y + 0.16), Inches(0.3), Inches(0.3))
    dot.fill.solid(); dot.fill.fore_color.rgb = NAVY; dot.line.fill.background()
    add_text(p, x + 0.14, y + 0.17, 0.3, 0.3, 'Q', size=12, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    add_text(p, x + 0.56, y + 0.1, 4.95, 0.4, q, size=13, bold=True, color=INK)
    add_text(p, x + 0.56, y + 0.52, 4.95, 0.85, a, size=11, color=MUTED)

# ---------- 删除要求页 + 重排 ----------
from pptx.oxml.ns import qn as _qn
sldIdLst = prs.slides._sldIdLst
req_id_el = None
for el in list(sldIdLst):
    rId = el.get(_qn('r:id'))
    if prs.part.rels[rId].target_part is req.part:
        req_id_el = el
        prs.part.drop_rel(rId)
        break
sldIdLst.remove(req_id_el)

desired = [cover, toc, part1, part2, pP3, pP4, pP5, pP6, part3, pP8, pP9, part4,
           b1, b2, b3, b4, b5, thanks]
el_by_part = {}
for el in list(sldIdLst):
    rId = el.get(_qn('r:id'))
    el_by_part[prs.part.rels[rId].target_part] = el
for el in list(sldIdLst):
    sldIdLst.remove(el)
for s_ in desired:
    sldIdLst.append(el_by_part[s_.part])
print('slides total:', len(prs.slides._sldIdLst))

prs.save(OUT)
print('WROTE', OUT)
