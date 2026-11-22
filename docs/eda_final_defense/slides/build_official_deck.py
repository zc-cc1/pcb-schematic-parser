"""Build the official-template defense deck (English) by cloning the organizer template."""
import copy
import io
import os
import sys

CKPT = os.environ.get('CKPT') == '1'
_CKN = [0]
_CKDIR = r'E:\AI_Workplace\精英挑战赛\EDA_Final_Defense\out'
def ck():
    if CKPT:
        _CKN[0] += 1
        prs.save(rf'{_CKDIR}\ckpt_{_CKN[0]:02d}.pptx')
        print('ckpt saved', _CKN[0])

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from PIL import Image

TPL = r'E:\AI_Workplace\精英挑战赛\EDA_Final_Defense\official_template.pptx'
OUT = r'E:\AI_Workplace\精英挑战赛\EDA_Final_Defense\EDA精英挑战赛_决赛答辩_官方模板版.pptx'
A = r'E:\AI_Workplace\精英挑战赛\_audit_workspace\assets_build'
D = r'E:\AI_Workplace\精英挑战赛\_audit_workspace\deck_workspace'

NAVY = RGBColor(0x12, 0x3C, 0x6D); CYAN = RGBColor(0x0E, 0x7C, 0x8C); ORANGE = RGBColor(0xF5, 0x9E, 0x0B)
INK = RGBColor(0x17, 0x26, 0x3B); MUTED = RGBColor(0x5A, 0x6B, 0x80); WHITE = RGBColor(0xFF, 0xFF, 0xFF)
PANEL = RGBColor(0xEE, 0xF4, 0xFA); LINE = RGBColor(0xC9, 0xD6, 0xE6)
FONT = 'Arial'

prs = Presentation(TPL)
print('template slides:', len(prs.slides))


def set_text(shape, text, size=None, bold=None, color=None, font=FONT, align=None):
    tf = shape.text_frame
    tf.word_wrap = True
    # rewrite first paragraph, drop the rest
    for p in list(tf.paragraphs[1:]):
        p._p.getparent().remove(p._p)
    p = tf.paragraphs[0]
    for r in list(p.runs):
        r._r.getparent().remove(r._r)
    r = p.add_run(); r.text = text
    if size: r.font.size = Pt(size)
    if bold is not None: r.font.bold = bold
    if color: r.font.color.rgb = color
    r.font.name = font
    if align: p.alignment = align
    return shape


def add_text(slide, x, y, w, h, runs, size=18, bold=False, color=INK, align=PP_ALIGN.LEFT,
             anchor=MSO_ANCHOR.TOP, font=FONT, spacing=1.0, italic=False):
    """runs: str or list of (text, dict-overrides)."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    lines = runs if isinstance(runs, list) else [(runs, {})]
    first = True
    for t, ov in lines:
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.alignment = ov.get('align', align)
        if ov.get('space_before'): p.space_before = Pt(ov['space_before'])
        if ov.get('space_after') is not None: p.space_after = Pt(ov.get('space_after', 4))
        r = p.add_run(); r.text = t
        r.font.size = Pt(ov.get('size', size))
        r.font.bold = ov.get('bold', bold)
        r.font.name = ov.get('font', font)
        r.font.color.rgb = ov.get('color', color)
        if ov.get('italic', italic): r.font.italic = True
    return tb


def add_pic(slide, path, bx, by, bw, bh):
    im = Image.open(path)
    pw, ph = im.size
    r = min(bw / (pw / 914400 * 914400 / 914400), bh / (ph / 914400)) if False else min(bw / pw, bh / ph) * 0  # placeholder
    return None


def place_pic(slide, path, bx, by, bw, bh):
    im = Image.open(path); pw, ph = im.size
    sc = min(bw / pw, bh / ph)
    w, h = pw * sc, ph * sc
    return slide.shapes.add_picture(path, Inches(bx + (bw - w) / 2), Inches(by + (bh - h) / 2), Inches(w), Inches(h))


def add_card(slide, x, y, w, h, fill=PANEL, line_c=LINE, radius=None):
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
def clone_body(prs, src):
    layout = src.slide_layout
    dest = prs.slides.add_slide(layout)
    for ph in list(dest.placeholders):
        ph._element.getparent().remove(ph._element)
    for sh in src.shapes:
        if sh.name not in TEMPLATE_SHAPE_NAMES:
            continue
        el = copy.deepcopy(sh._element)
        # strip template-internal tag references (custDataLst -> r:id) that would
        # dangle in the cloned slide and trigger PowerPoint/WPS repair
        for cd in el.findall('.//' + '{http://schemas.openxmlformats.org/presentationml/2006/main}custDataLst'):
            cd.getparent().remove(cd)
        dest.shapes._spTree.append(el)
    return dest


def set_title_subtitle(page, title=None, subtitle=None):
    for sh in list(page.shapes):
        if sh.name == '文本框 9':
            sh.width = Inches(8.9)
            if title is not None:
                tf = sh.text_frame
                p = tf.paragraphs[0]
                for r in list(p.runs):
                    r._r.getparent().remove(r._r)
                r = p.add_run(); r.text = title
                r.font.size = Pt(22); r.font.bold = True; r.font.name = FONT
        elif sh.name == '内容占位符 5' and subtitle is not None:
            sh.width = Inches(9.6)
            set_text(sh, subtitle, size=13, color=MUTED, font=FONT)
    return page


def content_shapes(page):
    """Return the big rounded content-rect bounds of a cloned body page."""
    return 0.95, 1.42, 11.55, 5.24  # usable area inside rect (x, y, w, h)


# ---------- slide bookkeeping (deletion happens LAST to avoid partname collision) ----------
slides = list(prs.slides)
cover, toc, req, part1, part2, part3, part4, thanks = slides

# ---------- cover ----------
for sh in cover.shapes:
    if sh.name == '标题 7':
        set_text(sh, 'EDA2606K5', size=None)  # keep template size; verify against registration record
    elif sh.name == '副标题 8':
        set_text(sh, 'November 28, 2026', size=None)
add_text(cover, 1.37, 3.42, 10.7, 0.66, 'AI-Driven PCB Schematic Parsing System', size=30, bold=True, color=WHITE)
add_text(cover, 1.37, 4.18, 10.7, 0.4, 'From Pixels to Verified Electrical Connectivity', size=16, color=RGBColor(0x8F, 0xD8, 0xE2))


# ---------- Part I page: problem & strategy ----------
set_title_subtitle(part1, None, 'Problem understanding & solution strategy')
cx, cy, cw, ch = content_shapes(part1)
add_text(part1, cx, cy + 0.02, cw, 0.62,
         'The Hard Part Is Not Finding Symbols. ' , size=22, bold=True, color=NAVY)
tbx = part1.shapes[-1].text_frame
r = tbx.paragraphs[0].add_run(); r.text = "It's Recovering Connectivity."
r.font.size = Pt(22); r.font.bold = True; r.font.color.rgb = ORANGE; r.font.name = FONT
add_text(part1, cx, cy + 0.66, cw, 0.36,
         [('Perceive', {'size': 15, 'bold': True, 'color': CYAN}),
         ], size=15)
tbx = part1.shapes[-1].text_frame
r = tbx.paragraphs[0].add_run(); r.text = '  identify parts, text, pins and wires      '; r.font.size = Pt(13); r.font.color.rgb = MUTED; r.font.name = FONT
r = tbx.paragraphs[0].add_run(); r.text = 'Connect'; r.font.size = Pt(15); r.font.bold = True; r.font.color.rgb = NAVY; r.font.name = FONT
r = tbx.paragraphs[0].add_run(); r.text = '  build geometry and electrical topology      '; r.font.size = Pt(13); r.font.color.rgb = MUTED; r.font.name = FONT
r = tbx.paragraphs[0].add_run(); r.text = 'Verify'; r.font.size = Pt(15); r.font.bold = True; r.font.color.rgb = ORANGE; r.font.name = FONT
r = tbx.paragraphs[0].add_run(); r.text = '  repair and calibrate with evidence'; r.font.size = Pt(13); r.font.color.rgb = MUTED; r.font.name = FONT
add_text(part1, cx, cy + 1.08, cw, 0.36,
         'Input: one PCB schematic PNG   →   Output: structured JSON (components / pins / nets)', size=15, color=INK)
ch_items = [
    ('Dense component detection', 'small SMD parts, missed / confused classes', 'Dual-path YOLO (whole + SAHI) + WBF fusion', 'per-class thresholds, F1 0.883'),
    ('Pin localization & ownership', 'multi-pin ICs, offsets, wrong binding', 'Dual-head U-Net heatmaps + cost assignment', 'peak decode with sub-pixel refinement'),
    ('Text - part association', 'crowded / tilted annotations', 'Multi-view OCR under evidence gates', '>=2 views @ 0.98 confidence, budgets'),
    ('Topology & connectivity', 'T/cross nodes, broken wires, flywires', 'Conservative graph + transactional repair', 'avoid false links before recovery'),
]
row_h = 0.84
for i, (ct, cd, st, sd) in enumerate(ch_items):
    y = cy + 1.46 + i * (row_h + 0.06)
    add_card(part1, cx, y, 5.28, row_h, fill=WHITE)
    add_text(part1, cx + 0.14, y + 0.07, 5.0, 0.34, ct, size=15, bold=True, color=NAVY)
    add_text(part1, cx + 0.14, y + 0.44, 5.0, 0.4, cd, size=12.5, color=MUTED)
    sld_arw = part1.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(cx + 5.44), Inches(y + row_h / 2 - 0.14), Inches(0.5), Inches(0.28))
    sld_arw.fill.solid(); sld_arw.fill.fore_color.rgb = CYAN; sld_arw.line.fill.background()
    add_card(part1, cx + 6.1, y, 5.4, row_h, fill=PANEL)
    add_text(part1, cx + 6.24, y + 0.07, 5.15, 0.34, st, size=15, bold=True, color=INK)
    add_text(part1, cx + 6.24, y + 0.44, 5.15, 0.4, sd, size=12.5, color=MUTED)
add_text(part1, cx, cy + 5.06, cw, 0.3,
         'Detection + OCR alone are not enough: pixel-level connectivity requires geometry, graph reasoning and evidence-gated repair.',
         size=12.5, italic=True, color=MUTED)


# ---------- Part II page: four-layer architecture ----------
set_title_subtitle(part2, None, 'End-to-end system architecture')
cx, cy, cw, ch = content_shapes(part2)
layers = [
    ('1   Visual Perception', 'YOLO dual-path + WBF fusion  ·  dual-head U-Net (wire + pin)  ·  PP-OCRv6 offline OCR', PANEL, NAVY),
    ('2   Geometric Processing', 'pin ownership assignment  ·  skeleton → segments  ·  IC / LED frames  ·  tip alignment', WHITE, NAVY),
    ('3   Connectivity Reasoning', 'wire graph & junction policy  ·  physical networks  ·  flywire / power merges  ·  field completion S14-S24', RGBColor(0xE8, 0xF6, 0xF8), CYAN),
    ('4   Evidence-Gated Correction', 'transactional repair engine (15 stages)  ·  official format calibration  ·  validated JSON export', RGBColor(0xFF, 0xF7, 0xE9), ORANGE),
]
for i, (t, d, fc, ec) in enumerate(layers):
    y = cy + 0.02 + i * 1.18
    add_card(part2, cx, y, cw, 1.0, fill=fc, line_c=ec)
    add_text(part2, cx + 0.22, y + 0.12, 3.75, 0.76, t, size=18, bold=True, color=NAVY)
    add_text(part2, cx + 4.05, y + 0.16, 7.3, 0.72, d, size=14, color=INK)
    if i < 3:
        da = part2.shapes.add_shape(MSO_SHAPE.DOWN_ARROW, Inches(cx + 1.55), Inches(y + 1.03), Inches(0.26), Inches(0.13))
        da.fill.solid(); da.fill.fore_color.rgb = CYAN; da.line.fill.background()
add_text(part2, cx, cy + 4.82, cw, 0.34,
         'Full 11-stage detail: Backup A  ·  zero LLM tokens  ·  per-stage hash locks & audit logs  ·  fallback to last valid prediction',
         size=11.5, color=MUTED)


# ---------- body pages (cloned from part2) ----------
def new_body(title, subtitle):
    pg = clone_body(prs, part2)
    set_title_subtitle(pg, title, subtitle)
    return pg

# ---- P3 Visual Perception ----
pP3 = new_body('Visual Perception', 'Three models, one job each - candidates only; conclusions come from evidence gates')
p = pP3
cx, cy, cw, ch = content_shapes(p)
place_pic(p, rf'{D}\d_yolo_fusion_en.png', cx - 0.05, cy + 0.05, 6.35, 3.1)
add_text(p, cx, cy + 3.24, 6.45, 0.34,
         'Whole-image pass keeps precision, SAHI tiles recover small parts; WBF merges both with score-weighted boxes instead of an NMS winner.',
         size=11.5, color=INK)
origins = [
    ('YOLO nano detector', 'self-trained (35 classes) · detection F1 0.883 (59-case micro-avg)', NAVY),
    ('WirePinUNet dual-head', 'self-trained (milesial U-Net parts as the base structure)', NAVY),
    ('PP-OCRv6 OCR', 'called open pre-trained models, offline ONNX', MUTED),
    ('JunctionCNN', 'self-trained but OFF in the default pipeline - optional enhancement', ORANGE),
]
for i, (t, d, c) in enumerate(origins):
    y = cy + 3.68 + i * 0.36
    dot = p.shapes.add_shape(MSO_SHAPE.OVAL, Inches(cx + 0.05), Inches(y + 0.06), Inches(0.13), Inches(0.13))
    dot.fill.solid(); dot.fill.fore_color.rgb = c; dot.line.fill.background()
    add_text(p, cx + 0.28, y, 2.6, 0.3, t, size=12.5, bold=True, color=INK)
    add_text(p, cx + 3.0, y + 0.02, 3.45, 0.3, d, size=11.5, color=MUTED)
place_pic(p, rf'{A}\fig2c_unet_real_artifacts.png', cx + 6.7, cy, 2.4, 4.95)
add_text(p, cx + 6.55, cy + 5.0, 2.7, 0.28,
         'Real pipeline output: input / wire map / pin heatmap', size=11, color=MUTED, align=PP_ALIGN.CENTER)


# ---- P4 Topology Reconstruction ----
pP4 = new_body('Topology Reconstruction', 'Wire probability map -> skeleton -> segments -> networks')
p = pP4
cx, cy, cw, ch = content_shapes(p)
add_text(p, cx, cy, cw, 0.4, 'Avoid False Connections Before Recovering Missing Links', size=17, bold=True, color=NAVY)
place_pic(p, rf'{A}\fig3c_topology_real.png', cx, cy + 0.5, 5.55, 3.1)
add_text(p, cx, cy + 3.66, 5.55, 0.3, 'Skeleton 7,210 px + 39 junction nodes (magenta)', size=11, color=MUTED, align=PP_ALIGN.CENTER)
place_pic(p, rf'{A}\fig3d_graph_networks_real.png', cx + 5.9, cy + 0.5, 5.55, 3.1)
add_text(p, cx + 5.9, cy + 3.66, 5.6, 0.3, 'Physical-network candidates: 238 nets / 362 edges (pre-merge)', size=11, color=MUTED, align=PP_ALIGN.CENTER)
cases = [
    ('T / cross nodes', 'conservative split, explicit uncertainty'),
    ('Broken wires', 'recovered only by hard evidence: 60px collinear overlap or red dot + 95% ink'),
    ('Net labels (flywires)', 'same-name labels merge logically; zero-length edges'),
    ('Power nets (VCC/GND)', 'four gates before merging; no physical long lines'),
]
for i, (t, s) in enumerate(cases):
    x = cx + i * 2.92
    add_card(p, x, cy + 4.06, 2.78, 1.0, fill=PANEL)
    add_text(p, x + 0.12, cy + 4.12, 2.55, 0.3, t, size=12.5, bold=True, color=NAVY)
    add_text(p, x + 0.12, cy + 4.44, 2.55, 0.58, s, size=11, color=INK)
add_text(p, cx, cy + 5.18, cw, 0.32,
         '238 Physical-Network Candidates / 362 Edges → 63 Logical Nets - candidates are the graph-stage intermediate; merges give the final output.',
         size=11.5, color=INK)


# ---- P5 Evidence-Gated Repair ----
pP5 = new_body('Repair Without Guessing', 'Every rewrite carries evidence; failures roll back to the last valid prediction')
p = pP5
cx, cy, cw, ch = content_shapes(p)
place_pic(p, rf'{A}\fig4_page13_repair_case.png', cx - 0.05, cy, 7.5, 3.95)
add_text(p, cx, cy + 4.02, 7.5, 0.36,
         [('Page-13 repair: 32/28/17 → 46/13/3 · F1 58.72% → ', {'size': 14, 'bold': True, 'color': INK})], size=14)
tbx = p.shapes[-1].text_frame
r = tbx.paragraphs[0].add_run(); r.text = '85.19%'; r.font.size = Pt(14); r.font.bold = True; r.font.color.rgb = ORANGE; r.font.name = FONT
r = tbx.paragraphs[0].add_run(); r.text = ' · score 82.871→92.656'; r.font.size = Pt(14); r.font.bold = True; r.font.color.rgb = INK; r.font.name = FONT
add_text(p, cx, cy + 4.44, 7.5, 0.3,
         'Single-page development-set case (frozen cache replay) - not an official overall metric.',
         size=11.5, italic=True, color=MUTED)
steps = [
    ('Detect', 'multi-view OCR · hard budgets', PANEL, NAVY),
    ('Verify', '>=2 views @ 0.98 · geometric rules', RGBColor(0xE8, 0xF6, 0xF8), CYAN),
    ('Commit | Reject', 'atomic transaction · conflicts vetoed', PANEL, NAVY),
    ('Rollback', 'remember() + per-image rescue', RGBColor(0xFF, 0xF7, 0xE9), ORANGE),
]
for i, (t, d, fc, ec) in enumerate(steps):
    x = cx + i * 2.92
    cv = p.shapes.add_shape(MSO_SHAPE.CHEVRON, Inches(x), Inches(cy + 4.85), Inches(2.8), Inches(0.92))
    cv.fill.solid(); cv.fill.fore_color.rgb = fc; cv.line.color.rgb = ec; cv.line.width = Pt(1)
    try:
        cv.shadow.inherit = False
    except Exception:
        pass
    tf = cv.text_frame; tf.word_wrap = True
    tf.margin_left = Inches(0.22); tf.margin_right = Inches(0.05); tf.margin_top = Inches(0.06)
    pa = tf.paragraphs[0]; pa.alignment = PP_ALIGN.CENTER
    r = pa.add_run(); r.text = t; r.font.size = Pt(16); r.font.bold = True; r.font.color.rgb = INK; r.font.name = FONT
    pb = tf.add_paragraph(); pb.alignment = PP_ALIGN.CENTER
    r = pb.add_run(); r.text = d; r.font.size = Pt(11); r.font.color.rgb = MUTED; r.font.name = FONT
# right column: auxiliary case
add_card(p, cx + 7.85, cy, 3.7, 4.1, fill=WHITE)
place_pic(p, rf'{A}\fig5_evidence_gated_ocr.png', cx + 7.95, cy + 0.1, 3.5, 2.6)
add_text(p, cx + 7.95, cy + 2.78, 3.5, 0.32, 'Auxiliary case - part-number recovery', size=13, bold=True, color=INK)
add_text(p, cx + 7.95, cy + 3.12, 3.5, 0.34,
         [('"SW1" → "PTS8153JK250SMTRLFS"', {'size': 12.5, 'bold': True, 'color': NAVY, 'font': 'Consolas'})], size=12.5)
add_text(p, cx + 7.95, cy + 3.48, 3.5, 0.3, 'rotated view · conf 0.998 / 0.992', size=11, color=MUTED)
add_text(p, cx + 7.95, cy + 3.8, 3.5, 0.3, '110+ repair modules share this one discipline.', size=11, color=MUTED)


# ---- P6 Output Calibration ----
pP6 = new_body('Output Calibration for Official Evaluation',
               'Statistical calibration to the official annotation conventions - not per-image lookups')
p = pP6
cx, cy, cw, ch = content_shapes(p)
cards = [
    ('Type & field conventions', 'resistor → r ... (verified on 145 matched parts) · Name==key convention from 9,186-component statistics (box keeps its part number)', CYAN),
    ('Bounding-box calibration (two scopes)', 'canonical (40-page public subset): within-20px 0.905 → 1.000  ·  page-adaptive (13 released): 150/163 → 159/163', NAVY),
    ('Pin-tip alignment', '18 px tip snap (measured 16-22 px plateau) + 1000-part pin library', ORANGE),
]
for i, (t, d, c) in enumerate(cards):
    y = cy + i * 1.34
    add_card(p, cx, y, 6.6, 1.2, fill=WHITE)
    add_text(p, cx + 0.18, y + 0.08, 6.2, 0.36, t, size=15, bold=True, color=INK)
    add_text(p, cx + 0.18, y + 0.5, 5.65, 0.62, d, size=12.5, color=MUTED)
    dot = p.shapes.add_shape(MSO_SHAPE.OVAL, Inches(cx + 6.1), Inches(y + 0.52), Inches(0.16), Inches(0.16))
    dot.fill.solid(); dot.fill.fore_color.rgb = c; dot.line.fill.background()
place_pic(p, rf'{D}\d_json_structure_en.png', cx + 6.95, cy, 5.2, 3.37)
add_card(p, cx + 6.95, cy + 3.55, 5.2, 1.72, fill=RGBColor(0xFF, 0xF7, 0xE9), line_c=ORANGE)
add_text(p, cx + 6.95, cy + 3.66, 5.2, 0.74, '1~2 px', size=32, bold=True, color=ORANGE, align=PP_ALIGN.CENTER)
add_text(p, cx + 7.05, cy + 4.42, 5.0, 0.78,
         'U3 (RP2350A) bbox vs official example annotation\nours [539,281,689,731] vs official [540,280,690,730]\n(official 20 px tolerance, official example page)', size=11.5, color=INK, align=PP_ALIGN.CENTER)
add_text(p, cx, cy + 5.02, cw, 0.32,
         'Both bbox scopes are independent experiments - never multiplied. Full metric table: Backup D.',
         size=12, color=MUTED)


# ---------- Part III page: official results ----------
set_title_subtitle(part3, None, 'Official evaluation results & quantitative evidence')
cx, cy, cw, ch = content_shapes(part3)
chip(part3, cx + 0.1, cy + 0.15, 3.6, 1.9, '74', 'official final score', ORANGE, 54, 14, fill=WHITE)
chip(part3, cx + 3.9, cy + 0.15, 3.6, 1.9, '47', 'national rank', NAVY, 54, 14, fill=WHITE)
chip(part3, cx + 7.7, cy + 0.15, 3.6, 1.9, 'Top 61', 'finalist qualification', NAVY, 44, 14, fill=WHITE)
add_card(part3, cx, cy + 2.3, 11.55, 1.42, fill=PANEL)
add_text(part3, cx + 0.2, cy + 2.4, 11.2, 0.4, 'Version iteration', size=16, bold=True, color=INK)
tbx = part3.shapes[-1].text_frame
r = tbx.paragraphs[0].add_run(); r.text = '   28.92'; r.font.size = Pt(16); r.font.bold = True; r.font.color.rgb = NAVY; r.font.name = FONT
r = tbx.paragraphs[0].add_run(); r.text = ' (old build 8422)   →   '; r.font.size = Pt(13); r.font.color.rgb = MUTED; r.font.name = FONT
r = tbx.paragraphs[0].add_run(); r.text = '74'; r.font.size = Pt(16); r.font.bold = True; r.font.color.rgb = ORANGE; r.font.name = FONT
r = tbx.paragraphs[0].add_run(); r.text = ' (final submission, +45)'; r.font.size = Pt(13); r.font.color.rgb = MUTED; r.font.name = FONT
add_text(part3, cx + 0.2, cy + 2.82, 11.2, 0.8,
         'Fixes behind the jump: unnumbered power/ground exports · LED text & boxes · IC name priority & observed frames · engineering numbering · export semantics.',
         size=12.5, color=INK)
add_text(part3, cx, cy + 3.95, cw, 0.4,
         'Official score / rank / finalist screenshots: pending insertion before the defense (recorded as missing; no mock screenshots used).',
         size=12, italic=True, color=MUTED)
add_text(part3, cx, cy + 4.4, cw, 0.36,
         'Local development replay score (13-page public set): moved to Backup D - development only, not comparable with official results.',
         size=12, color=MUTED)


# ---- P8 Quantitative Results and Case Studies ----
pP8 = new_body('Quantitative Results and Case Studies',
               'Four scoring dimensions, one card each - development-set experiments and single-case ablations')
p = pP8
cx, cy, cw, ch = content_shapes(p)
mets = [
    ('Component', 'bbox within-20px rate', '0.905', '1.000', 'canonical convention · 40-page public subset'),
    ('Pin', 'PinF1', '0.031', '0.549', 'pin-library mapping · 13 released pages'),
    ('Connectivity', 'network connectivity', '0.188', '0.612', 'junction-gate release · 13 released pages'),
    ('Wire', 'wire F1 (page-13 case)', '58.72%', '85.19%', 'junction reconciliation · single-page ablation'),
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
         'Development-set experiments and single-case ablations - not official per-dimension scores. Full table: Backup D.',
         size=13, bold=True, color=INK)
add_text(p, cx, cy + 5.08, cw, 0.3,
         'Ablation protocol: one module toggled on a frozen model/OCR cache; the other 12 pages stay byte-identical.',
         size=11.5, color=MUTED)


# ---- P9 End-to-End Output and Engineering Validation ----
pP9 = new_body('End-to-End Output and Engineering Validation',
               'Official example page, full pipeline - from PNG to validated JSON')
p = pP9
cx, cy, cw, ch = content_shapes(p)
place_pic(p, rf'{A}\fig1b_detection_clean.png', cx, cy, 4.09, 2.9)
add_text(p, cx, cy + 2.94, 4.15, 0.28, 'Detection & recognition overlay (32 drawn-symbol parts)', size=11, color=MUTED, align=PP_ALIGN.CENTER)
place_pic(p, rf'{A}\fig3b_network_overlay.png', cx + 4.35, cy, 4.09, 2.9)
add_text(p, cx + 4.35, cy + 2.94, 4.15, 0.28, 'Final networks: 63 logical nets, edges color-coded', size=11, color=MUTED, align=PP_ALIGN.CENTER)
stats = [('89', 'components'), ('165', 'pins'), ('63', 'logical nets')]
for i, (n, lb) in enumerate(stats):
    y = cy + 0.06 + i * 0.94
    add_text(p, cx + 9.15, y, 2.2, 0.52, n, size=30, bold=True, color=NAVY)
    add_text(p, cx + 9.15, y + 0.54, 2.2, 0.28, lb, size=11.5, color=MUTED)
add_text(p, cx + 9.15, cy + 2.92, 2.3, 0.28, 'U3 bbox diff', size=11.5, color=INK)
add_text(p, cx + 9.15, cy + 3.18, 2.3, 0.44, '1~2 px', size=20, bold=True, color=ORANGE)
add_text(p, cx, cy + 3.28, cw, 0.32, 'Output JSON (real fragments, official example page)', size=12.5, bold=True, color=INK)
json_rows = [
    ('components', '"U3": {"type":"box","Name":"RP2350A",', '"bbox":[539,281,689,731]}', RGBColor(0x0E, 0x7C, 0x8C)),
    ('pins', '"U3": {"pin_51": {"pinname":"USB_DM",', '"point":{"x":513,"y":561}}}', NAVY),
    ('nets', '"LOG_NET": {"hyperGraph":"(C1.2,U3.U3)",', '"edges":{...3 edges}}', ORANGE),
]
for i, (key, l1, l2, c) in enumerate(json_rows):
    y = cy + 3.62 + i * 0.62
    kc = p.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(cx), Inches(y), Inches(1.42), Inches(0.44))
    kc.fill.solid(); kc.fill.fore_color.rgb = c; kc.line.fill.background()
    try:
        kc.shadow.inherit = False
    except Exception:
        pass
    ktf = kc.text_frame; ktf.word_wrap = True
    kpa = ktf.paragraphs[0]; kpa.alignment = PP_ALIGN.CENTER
    kr = kpa.add_run(); kr.text = key; kr.font.size = Pt(11.5); kr.font.bold = True; kr.font.color.rgb = WHITE; kr.font.name = FONT
    add_text(p, cx + 1.6, y - 0.02, 4.1, 0.3, l1, size=11.5, color=INK, font='Consolas')
    add_text(p, cx + 1.6, y + 0.26, 4.1, 0.3, l2, size=11.5, color=INK, font='Consolas')
add_text(p, cx + 5.95, cy + 3.7, 6.55, 0.4, 'Verified structured output - the deliverable', size=17, bold=True, color=NAVY)
add_text(p, cx + 5.95, cy + 4.16, 6.55, 0.36,
         'validate_target + sanitize: remove only invalid objects, never infer text, coordinates or joins.',
         size=13, color=INK)
add_text(p, cx + 5.95, cy + 4.54, 6.55, 0.32,
         'Engineering details (Docker · SHA256 · audit · 626+ tests · runtime): Backup C.',
         size=12.5, color=MUTED)

# ---------- Part IV page: conclusions ----------
set_title_subtitle(part4, None, 'Conclusions and future work')
cx, cy, cw, ch = content_shapes(part4)
add_text(part4, cx, cy + 0.02, cw, 0.44, 'Three main contributions', size=19, bold=True, color=NAVY)
contribs = [('Multi-model visual perception', 'self-trained YOLO nano / dual-head WirePinUNet / junction CNN - all sha256-traceable'),
            ('Evidence-gated topology repair', '0.98 x 2-view OCR gates + transactional engine + rollback'),
            ('Score-aware output calibration', 'bbox conventions / Name==key / pin-library mapping')]
for i, (t, s) in enumerate(contribs):
    y = cy + 0.53 + i * 0.55
    dot = part4.shapes.add_shape(MSO_SHAPE.OVAL, Inches(cx), Inches(y + 0.03), Inches(0.28), Inches(0.28))
    dot.fill.solid(); dot.fill.fore_color.rgb = ORANGE; dot.line.fill.background()
    add_text(part4, cx, y + 0.04, 0.28, 0.28, str(i + 1), size=11, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    add_text(part4, cx + 0.4, y - 0.02, 11.1, 0.3, t, size=14.5, bold=True, color=INK)
    add_text(part4, cx + 0.4, y + 0.27, 11.1, 0.3, s, size=11, color=MUTED)
add_text(part4, cx, cy + 2.24, cw, 0.3,
         'JunctionCNN: self-trained but OFF in the default pipeline - an optional enhancement, not the default inference backbone.',
         size=12.5, italic=True, color=MUTED)
add_text(part4, cx, cy + 2.63, cw, 0.4, 'Current limitations', size=19, bold=True, color=NAVY)
lims = ['Generalization to unseen schematic styles (2/3 of evaluation cases are black-box, 60% hard)',
        'Complex junction recognition (learned classifier still optional; default conservative)',
        'Dense pin identification on low-resolution pages']
for i, t in enumerate(lims):
    y = cy + 3.08 + i * 0.32
    dot = part4.shapes.add_shape(MSO_SHAPE.OVAL, Inches(cx + 0.06), Inches(y + 0.07), Inches(0.12), Inches(0.12))
    dot.fill.solid(); dot.fill.fore_color.rgb = CYAN; dot.line.fill.background()
    add_text(part4, cx + 0.3, y, 11.2, 0.3, t, size=12.5, color=INK)
add_text(part4, cx, cy + 4.08, cw, 0.4, 'Future work', size=19, bold=True, color=NAVY)
outs = ['Expand hard-case datasets and domain adaptation', 'Improve junction classification (promote the trained CNN)',
        'Enhance cross-domain generalization for engineering drawings']
for i, t in enumerate(outs):
    y = cy + 4.52 + i * 0.3
    dot = part4.shapes.add_shape(MSO_SHAPE.OVAL, Inches(cx + 0.06), Inches(y + 0.07), Inches(0.12), Inches(0.12))
    dot.fill.solid(); dot.fill.fore_color.rgb = NAVY; dot.line.fill.background()
    add_text(part4, cx + 0.3, y, 11.2, 0.3, t, size=12.5, color=INK)
# ---------- Backups B1-B5 ----------
# B1 detailed architecture
b1 = new_body('Backup A · Detailed System Architecture (11 stages)',
             'Compressed to four layers on the main deck (Part II) - zero LLM tokens · hash-locked · audited')
p = b1
p = b1
cx, cy, cw, ch = content_shapes(p)
place_pic(p, rf'{D}\arch_pipeline_en.png', cx, cy, cw, ch - 0.1)


# B2 training
ck()
b2 = new_body('Backup B · Model Training and Dataset',
             'All inference weights ship inside the image and are sha256-locked; training happens outside the package')
p = b2
cx, cy, cw, ch = content_shapes(p)
rows = [
    ('Model', 'Architecture', 'Training data', 'Key parameters', 'Origin'),
    ('YOLO nano detector', 'Ultralytics YOLO nano, 35 classes', 'detection_balanced_v2 (local)', 'imgsz 1536 · 120 epochs · batch 4', 'self-trained (v2 warm start; HR-DONK initial baseline only)'),
    ('WirePinUNet dual-head', 'base-32 shared decoder (wire+pin)', 'rasterized source annotations + reviewed masks', 'tiles 512/384 · 40 epochs · pos_weight 8/20 · epoch 32', 'self-trained (milesial parts, GPL-3 provenance)'),
    ('JunctionCNN', '6-channel dual-view CNN', '1,024 synthetic renders + reviewed crops x12', '30 epochs · 0.95 threshold · 4-rotation agreement', 'self-trained (OFF in default pipeline)'),
    ('PP-OCRv6 OCR', 'det/rec small + cls (ONNX)', 'open pre-trained', 'page 2048 px; pin 2/3/4x x 0/90°', 'called (rapidocr 3.9.2)'),
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
         'Data assets: 1000-part EasyEDA symbol library + 337-part slice + STM32F103C8T6 LQFP48 pin library (ST DS5319 datasheet) - all bundled, zero network access.',
         size=11.5, color=INK)
add_text(p, cx, cy + 4.14, cw, 0.4,
         'Provenance: weights/baseline_provenance.json (HR-DONK initial baseline, MIT) · vendor/pytorch_unet provenance (milesial, GPL-3, per-file sha256) · no evaluation annotation leakage in training data.',
         size=11, color=MUTED)

# B3 runtime
ck()
b3 = new_body('Backup C · Runtime Workflow and JSON Output',
             'docker run on the evaluation side; direct Python run for development')
p = b3
cx, cy, cw, ch = content_shapes(p)
add_card(p, cx, cy, 5.9, 2.75, fill=RGBColor(0x0D, 0x22, 0x37), line_c=RGBColor(0x1E, 0x4D, 0x8F))
add_text(p, cx + 0.25, cy + 0.2, 5.5, 2.4,
         [('docker run --rm --gpus all \\', {'color': RGBColor(0x8F, 0xD8, 0xE2), 'font': 'Consolas'}),
          ('  -v <in>:/input:ro -v <out>:/output:rw \\', {'color': RGBColor(0x8F, 0xD8, 0xE2), 'font': 'Consolas'}),
          ('  pcb-parser:v102 -input /input -output /output', {'color': RGBColor(0x8F, 0xD8, 0xE2), 'font': 'Consolas'}),
          ('', {}),
          ('# CPU fallback', {'color': MUTED, 'font': 'Consolas'}),
          ('python agent.py -input <dir> -output <dir> --device cpu', {'color': RGBColor(0x8F, 0xD8, 0xE2), 'font': 'Consolas'}),
          ('# ~30 PCB_* env switches; defaults are tuned values', {'color': MUTED, 'font': 'Consolas'})],
         size=11.5)
place_pic(p, rf'{D}\d_recovery_flow_en.png', cx + 6.2, cy + 0.1, 5.35, 2.5)
chain = [('run.sh', 'image entry'), ('agent.py', 'run_case per image'), ('S-stage chain', 'S14→S24 subprocess'),
         ('finalize x15', 'evidence-gated repair'), ('adapter', 'target calibration'), ('final_results', 'Pred.json')]
for i, (t, s) in enumerate(chain):
    x = cx + i * 1.945
    cv = p.shapes.add_shape(MSO_SHAPE.CHEVRON, Inches(x), Inches(cy + 3.15), Inches(1.86), Inches(0.72))
    cv.fill.solid(); cv.fill.fore_color.rgb = PANEL; cv.line.color.rgb = LINE; cv.line.width = Pt(0.75)
    tf = cv.text_frame; tf.word_wrap = True
    tf.margin_left = Inches(0.14); tf.margin_right = Inches(0.03)
    pa = tf.paragraphs[0]; pa.alignment = PP_ALIGN.CENTER
    r = pa.add_run(); r.text = t; r.font.size = Pt(9.5); r.font.bold = True; r.font.color.rgb = INK; r.font.name = FONT
    pb = tf.add_paragraph(); pb.alignment = PP_ALIGN.CENTER
    r = pb.add_run(); r.text = s; r.font.size = Pt(7.2); r.font.color.rgb = MUTED; r.font.name = FONT
add_text(p, cx, cy + 3.95, cw, 0.32,
         'Smoke-replay artifacts archived: decision_log.json + final_results/rp2350_page11_Pred.json + logs/ (20+ audit JSONs).',
         size=11.5, color=INK)
add_text(p, cx, cy + 4.2, cw, 0.28,
         'Entry semantics: ENTRYPOINT=/submission/run.sh (verified in OCI config); agent_system stays the official placeholder.',
         size=11, color=MUTED)
add_text(p, cx, cy + 4.52, cw, 0.44,
         'Engineering reliability: Docker delivery · SHA256 integrity (249-file manifest, 3 weight fingerprints, 12-asset preflight) · per-stage audit logs (20+ JSONs/image) · output validation (validate_target + sanitize) · 626+ unit tests with a build-time gate.',
         size=11.5, color=INK)
add_text(p, cx, cy + 5.06, cw, 0.28,
         'Runtime scope: CPU datum 2000.5 s/case (official example page); official GPU timing not measured - no 2-hour compliance claim.',
         size=12, color=MUTED)

# B4 metrics
ck()
b4 = new_body('Backup D · Experimental Metrics and Ablation (excerpt)',
             'Every number traceable; full 17-row table in defense_supplement.md §3')
p = b4
cx, cy, cw, ch = content_shapes(p)
rows = [
    ('Experiment', 'Scope', 'Metric', 'Result'),
    ('Detection config selection', '59-case public micro-avg', 'detection F1', '0.883 (fused wins)'),
    ('Pin library mapping', '13 released pages', 'PinF1', '0.031 → 0.549 (+47)'),
    ('Unresolved-junction release', '13 released pages', 'connectivity', '0.188 → 0.612 @0.83'),
    ('Gapped-wire release', '13 released pages', 'connectivity / score', '0.569→0.576 / +0.31 pts'),
    ('Junction reconciliation', 'page 13', 'wire TP/FP/FN · F1 · score', '32/28/17→46/13/3 · 58.72→85.19 · 82.871→92.656'),
    ('Terminal topology', 'page 5', 'connectivity · score', '84.96%→100% · 93.954→99.474'),
    ('bbox canonical', '40-page public subset', 'component F1 · within-20px', '.8641→.8999 · 0.905→1.000'),
    ('bbox page scaling', '13 released pages', 'templated parts within 20px', '150/163 → 159/163'),
    ('Local smoke reproduction', 'official example page', 'output scale · bbox check', '89/165/63 · U3 diff 1~2 px'),
]
tbl_shape = p.shapes.add_table(len(rows), 4, Inches(cx), Inches(cy + 0.05), Inches(11.55), Inches(4.4))
tbl = tbl_shape.table
for c, wd in zip(tbl.columns, [2.6, 2.4, 2.75, 3.8]):
    c.width = Inches(wd)
for ri, row in enumerate(rows):
    for ci, val in enumerate(row):
        cell = tbl.cell(ri, ci); cell.text = val
        para = cell.text_frame.paragraphs[0]
        para.alignment = PP_ALIGN.RIGHT if ci == 3 else PP_ALIGN.LEFT
        r0 = para.runs[0]; r0.font.size = Pt(11 if ri else 12)
        r0.font.bold = (ri == 0) or (ci == 3); r0.font.name = FONT
        r0.font.color.rgb = WHITE if ri == 0 else (NAVY if ci == 3 else INK)
        cell.fill.solid(); cell.fill.fore_color.rgb = NAVY if ri == 0 else (WHITE if ri % 2 else RGBColor(0xF6, 0xF9, 0xFC))
add_text(p, cx, cy + 4.62, cw, 0.32,
         'Caliber caution: canonical (40 pages) and page scaling (13 pages) are independent experiments; "+0.31" is the total-score delta (connectivity +0.7pp).',
         size=11.5, color=MUTED)
add_text(p, cx, cy + 4.98, cw, 0.32,
         'Local development replay: 98.93 (13-page public set, v101-score) - development only, not comparable with the official 74.',
         size=12, color=INK)

# B5 Q&A
ck()
b5 = new_body('Backup E · Technical Q&A', 'Full Chinese handbook: EDA决赛答辩_QA.md (10 questions with evidence-backed answers)')
p = b5
cx, cy, cw, ch = content_shapes(p)
qas = [('Why not an end-to-end LLM?', '95% of weight is pixel-level structural accuracy; LLM coordinates unreliable; tokens scored - we use 0'),
       ('Gap between official 74 and local 98.93?', 'Different scopes: official includes 2/3 black-box (60% hard); local is a 13-page public replay'),
       ('How are T/cross junctions handled?', 'Conservative split + red-dot detection; learned classifier kept optional (default off)'),
       ('Broken / missed wires?', 'Gaps logged, never guessed; recovered by 60px collinear overlap or red-dot + 95% ink (+0.31 pts)'),
       ('Tiny pin numbers unreadable?', 'Strip OCR (30-50px beyond lead) + row reread + 6-view targeted OCR + 1000-part library (PinF1 0.031→0.549)'),
       ('Within the 2-hour budget?', 'GPU 3-worker design; wall_seconds recorded per case; official machine timing not yet measured')]
for i, (q, a) in enumerate(qas):
    col, row = i % 2, i // 2
    x = cx + col * 5.93; y = cy + row * 1.62
    add_card(p, x, y, 5.62, 1.46, fill=WHITE)
    dot = p.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x + 0.14), Inches(y + 0.16), Inches(0.3), Inches(0.3))
    dot.fill.solid(); dot.fill.fore_color.rgb = NAVY; dot.line.fill.background()
    add_text(p, x + 0.14, y + 0.17, 0.3, 0.3, 'Q', size=12, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    add_text(p, x + 0.56, y + 0.1, 4.95, 0.4, q, size=13, bold=True, color=INK)
    add_text(p, x + 0.56, y + 0.52, 4.95, 0.85, a, size=11, color=MUTED)

# ---------- delete requirements slide + reorder to final narrative order ----------
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
print('WROTE', OUT, 'slides:', len(prs.slides._sldIdLst))
