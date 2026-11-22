"""Story-polish patch: rewrite page blocks in build_official_deck.py (Perceive-Connect-Verify narrative)."""
import io
import re
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
BS = chr(92)
p = r'E:\AI_Workplace\精英挑战赛\_audit_workspace\scripts\build_official_deck.py'
s = open(p, encoding='utf-8').read()

def block_replace(s, start_marker, end_marker, new_block, include_start=False):
    """Replace lines from the line containing start_marker up to (not including) the line containing end_marker."""
    lines = s.split('\n')
    si = next(i for i, l in enumerate(lines) if start_marker in l)
    ei = next(i for i, l in enumerate(lines) if end_marker in l and i > si)
    if not include_start:
        si += 0
    lines[si:ei] = new_block.split('\n')
    return '\n'.join(lines)

# ============ COVER: add name + tagline ============
s = block_replace(s, '# ---------- cover ----------', '# ---------- Part I page', """# ---------- cover ----------
for sh in cover.shapes:
    if sh.name == '标题 7':
        set_text(sh, 'EDA2606K5', size=None)  # keep template size; verify against registration record
    elif sh.name == '副标题 8':
        set_text(sh, 'November 28, 2026', size=None)
add_text(cover, 1.37, 3.42, 10.7, 0.66, 'AI-Driven PCB Schematic Parsing System', size=30, bold=True, color=WHITE)
add_text(cover, 1.37, 4.18, 10.7, 0.4, 'From Pixels to Verified Electrical Connectivity', size=16, color=RGBColor(0x8F, 0xD8, 0xE2))

""")

# ============ P3 (Part I page): new banner + thread strip ============
s = block_replace(s, '# ---------- Part I page', '# ---------- Part II page', """# ---------- Part I page: problem & strategy ----------
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
row_h = 0.88
for i, (ct, cd, st, sd) in enumerate(ch_items):
    y = cy + 1.52 + i * (row_h + 0.08)
    add_card(part1, cx, y, 5.28, row_h, fill=WHITE)
    add_text(part1, cx + 0.14, y + 0.07, 5.0, 0.34, ct, size=15, bold=True, color=NAVY)
    add_text(part1, cx + 0.14, y + 0.44, 5.0, 0.4, cd, size=12.5, color=MUTED)
    sld_arw = part1.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(cx + 5.44), Inches(y + row_h / 2 - 0.14), Inches(0.5), Inches(0.28))
    sld_arw.fill.solid(); sld_arw.fill.fore_color.rgb = CYAN; sld_arw.line.fill.background()
    add_card(part1, cx + 6.1, y, 5.4, row_h, fill=PANEL)
    add_text(part1, cx + 6.24, y + 0.07, 5.15, 0.34, st, size=15, bold=True, color=INK)
    add_text(part1, cx + 6.24, y + 0.44, 5.15, 0.4, sd, size=12.5, color=MUTED)
add_text(part1, cx, cy + 1.52 + 4 * (row_h + 0.08) - 0.02, cw, 0.3,
         'Detection + OCR alone are not enough: pixel-level connectivity requires geometry, graph reasoning and evidence-gated repair.',
         size=12.5, italic=True, color=MUTED)

""")

# ============ P4 (Part II page): native 4-layer architecture ============
s = block_replace(s, '# ---------- Part II page', '# ---------- body pages (cloned from part2)', """# ---------- Part II page: four-layer architecture ----------
set_title_subtitle(part2, None, 'End-to-end system architecture')
cx, cy, cw, ch = content_shapes(part2)
layers = [
    ('1   Visual Perception', 'YOLO dual-path + WBF fusion  ·  dual-head U-Net (wire + pin)  ·  PP-OCRv6 offline OCR', PANEL, NAVY),
    ('2   Geometric Processing', 'pin ownership assignment  ·  skeleton → segments  ·  IC / LED frames  ·  tip alignment', WHITE, NAVY),
    ('3   Connectivity Reasoning', 'wire graph & junction policy  ·  physical networks  ·  flywire / power merges  ·  field completion S14-S24', '#E8F6F8', CYAN),
    ('4   Evidence-Gated Correction', 'transactional repair engine (15 stages)  ·  official format calibration  ·  validated JSON export', '#FFF7E9', ORANGE),
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

""")

# ============ P5 (Visual Perception) ============
s = block_replace(s, "# ---- P3 Visual Perception ----", "# ---- P4 Topology Reconstruction ----", """# ---- P3 Visual Perception ----
pP3 = new_body('Visual Perception', 'Three models, one job each - candidates only; conclusions come from evidence gates')
p = pP3
cx, cy, cw, ch = content_shapes(p)
place_pic(p, rf'{D}\\d_yolo_fusion_en.png', cx - 0.05, cy + 0.05, 6.35, 3.1)
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
place_pic(p, rf'{A}\\fig2c_unet_real_artifacts.png', cx + 6.7, cy, 2.4, 4.95)
add_text(p, cx + 6.55, cy + 5.0, 2.7, 0.28,
         'Real pipeline output: input / wire map / pin heatmap', size=11, color=MUTED, align=PP_ALIGN.CENTER)

""")

# ============ P6 (Topology) ============
s = block_replace(s, "# ---- P4 Topology Reconstruction ----", "# ---- P5 Evidence-Gated Repair ----", """# ---- P4 Topology Reconstruction ----
pP4 = new_body('Topology Reconstruction', 'Wire probability map -> skeleton -> segments -> networks')
p = pP4
cx, cy, cw, ch = content_shapes(p)
add_text(p, cx, cy, cw, 0.4, 'Avoid False Connections Before Recovering Missing Links', size=17, bold=True, color=NAVY)
place_pic(p, rf'{A}\\fig3c_topology_real.png', cx, cy + 0.5, 5.55, 3.1)
add_text(p, cx, cy + 3.66, 5.55, 0.3, 'Skeleton 7,210 px + 39 junction nodes (magenta)', size=11, color=MUTED, align=PP_ALIGN.CENTER)
place_pic(p, rf'{A}\\fig3d_graph_networks_real.png', cx + 5.9, cy + 0.5, 5.55, 3.1)
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
         '238 Physical-Network Candidates / 362 Edges → 63 Logical Nets - the candidates are the graph-stage intermediate; flywire and power merges give the final output.',
         size=12, color=INK)

""")

# ============ P7 (Repair Without Guessing) ============
s = block_replace(s, "# ---- P5 Evidence-Gated Repair ----", "# ---- P6 Output Calibration ----", """# ---- P5 Evidence-Gated Repair ----
pP5 = new_body('Repair Without Guessing', 'Every rewrite carries evidence; failures roll back to the last valid prediction')
p = pP5
cx, cy, cw, ch = content_shapes(p)
place_pic(p, rf'{A}\\fig4_page13_repair_case.png', cx - 0.05, cy, 7.5, 3.95)
add_text(p, cx, cy + 4.02, 7.5, 0.4,
         [('Page-13 wire repair:  32/28/17 → 46/13/3    F1 58.72% → ', {'size': 15.5, 'bold': True, 'color': INK})], size=15.5)
tbx = p.shapes[-1].text_frame
r = tbx.paragraphs[0].add_run(); r.text = '85.19%'; r.font.size = Pt(15.5); r.font.bold = True; r.font.color.rgb = ORANGE; r.font.name = FONT
r = tbx.paragraphs[0].add_run(); r.text = '    score 82.871 → 92.656'; r.font.size = Pt(15.5); r.font.bold = True; r.font.color.rgb = INK; r.font.name = FONT
add_text(p, cx, cy + 4.44, 7.5, 0.3,
         'Single-page development-set case (frozen cache replay) - not an official overall metric.',
         size=11.5, italic=True, color=MUTED)
steps = [
    ('Detect', 'multi-view OCR · hard budgets', PANEL, NAVY),
    ('Verify', '>=2 views @ 0.98 · geometric rules', '#E8F6F8', CYAN),
    ('Commit | Reject', 'atomic transaction · conflicts vetoed', PANEL, NAVY),
    ('Rollback', 'remember() + per-image rescue', '#FFF7E9', ORANGE),
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
place_pic(p, rf'{A}\\fig5_evidence_gated_ocr.png', cx + 7.95, cy + 0.1, 3.5, 2.6)
add_text(p, cx + 7.95, cy + 2.78, 3.5, 0.32, 'Auxiliary case - part-number recovery', size=13, bold=True, color=INK)
add_text(p, cx + 7.95, cy + 3.12, 3.5, 0.34,
         [('"SW1" → "PTS8153JK250SMTRLFS"', {'size': 12.5, 'bold': True, 'color': NAVY, 'font': 'Consolas'})], size=12.5)
add_text(p, cx + 7.95, cy + 3.48, 3.5, 0.3, 'rotated view · conf 0.998 / 0.992', size=11, color=MUTED)
add_text(p, cx + 7.95, cy + 3.8, 3.5, 0.3, '110+ repair modules share this one discipline.', size=11, color=MUTED)

""")

# ============ P8 (Output Calibration) ============
s = block_replace(s, "# ---- P6 Output Calibration ----", "# ---------- Part III page", """# ---- P6 Output Calibration ----
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
    add_text(p, cx + 0.18, y + 0.5, 6.25, 0.62, d, size=12.5, color=MUTED)
    dot = p.shapes.add_shape(MSO_SHAPE.OVAL, Inches(cx + 6.1), Inches(y + 0.52), Inches(0.16), Inches(0.16))
    dot.fill.solid(); dot.fill.fore_color.rgb = c; dot.line.fill.background()
place_pic(p, rf'{D}\\d_json_structure_en.png', cx + 6.95, cy, 5.2, 3.37)
add_card(p, cx + 6.95, cy + 3.55, 5.2, 1.72, fill=RGBColor(0xFF, 0xF7, 0xE9), line_c=ORANGE)
add_text(p, cx + 6.95, cy + 3.66, 5.2, 0.74, '1~2 px', size=32, bold=True, color=ORANGE, align=PP_ALIGN.CENTER)
add_text(p, cx + 7.05, cy + 4.42, 5.0, 0.78,
         'U3 (RP2350A) bbox vs official example annotation\\nours [539,281,689,731] vs official [540,280,690,730]\\n(official 20 px tolerance, official example page)', size=11.5, color=INK, align=PP_ALIGN.CENTER)
add_text(p, cx, cy + 5.02, cw, 0.32,
         'Both bbox scopes are independent experiments - never multiplied. Full metric table: Backup D.',
         size=12, color=MUTED)

""")

# ============ P9 (Part III) ============
s = block_replace(s, '# ---------- Part III page', '# ---- P8 Quantitative Results and Case Studies ----', """# ---------- Part III page: official results ----------
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

""")

# ============ P10 (Quantitative) ============
s = block_replace(s, '# ---- P8 Quantitative Results and Case Studies ----', '# ---- P9 End-to-End Output and Engineering Validation ----', """# ---- P8 Quantitative Results and Case Studies ----
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

""")

# ============ P11 (End-to-End) ============
s = block_replace(s, '# ---- P9 End-to-End Output and Engineering Validation ----', '# ---------- Part IV page', """# ---- P9 End-to-End Output and Engineering Validation ----
pP9 = new_body('End-to-End Output and Engineering Validation',
               'Official example page, full pipeline - from PNG to validated JSON')
p = pP9
cx, cy, cw, ch = content_shapes(p)
place_pic(p, rf'{A}\\fig1b_detection_clean.png', cx, cy, 4.5, 3.19)
add_text(p, cx, cy + 3.26, 4.55, 0.28, 'Detection & recognition overlay (32 drawn-symbol parts)', size=11, color=MUTED, align=PP_ALIGN.CENTER)
place_pic(p, rf'{A}\\fig3b_network_overlay.png', cx + 4.7, cy, 4.5, 3.19)
add_text(p, cx + 4.7, cy + 3.26, 4.55, 0.28, 'Final networks: 63 logical nets, edges color-coded', size=11, color=MUTED, align=PP_ALIGN.CENTER)
stats = [('89', 'components'), ('165', 'pins'), ('63', 'logical nets')]
for i, (n, lb) in enumerate(stats):
    y = cy + 0.1 + i * 1.02
    add_text(p, cx + 9.5, y, 2.0, 0.56, n, size=32, bold=True, color=NAVY)
    add_text(p, cx + 9.5, y + 0.58, 2.0, 0.28, lb, size=11.5, color=MUTED)
add_text(p, cx + 9.5, cy + 3.15, 2.2, 0.3, 'U3 bbox diff', size=11.5, color=INK)
add_text(p, cx + 9.5, cy + 3.42, 2.2, 0.44, '1~2 px', size=21, bold=True, color=ORANGE)
place_pic(p, rf'{D}\\d_json_structure_en.png', cx, cy + 3.62, 4.63, 2.98)
add_text(p, cx + 4.95, cy + 3.72, 7.55, 0.42, 'Verified structured output - the deliverable', size=17, bold=True, color=NAVY)
add_text(p, cx + 4.95, cy + 4.18, 7.55, 0.36,
         'validate_target + sanitize: remove only invalid objects, never infer text, coordinates or joins.',
         size=13, color=INK)
add_text(p, cx + 4.95, cy + 4.6, 7.55, 0.36,
         'Engineering details (Docker · SHA256 · audit logs · 626+ tests · runtime scope): Backup C.',
         size=13, color=MUTED)

""")

# ============ P12 (Part IV): add JunctionCNN note ============
anchor = "add_text(part4, cx, cy + 2.43, cw, 0.42, 'Current limitations', size=19, bold=True, color=NAVY)"
assert anchor in s
s = s.replace(anchor, """add_text(part4, cx, cy + 2.4, cw, 0.32,
         'JunctionCNN: self-trained but OFF in the default pipeline - an optional enhancement, not the default inference backbone.',
         size=12.5, italic=True, color=MUTED)
""" + anchor, 1)

# ============ B1 ============
s = block_replace(s, '# B1 module map', '# B2 training', """# B1 detailed architecture
p = new_body('Backup A · Detailed System Architecture (11 stages)',
             'Compressed to four layers on the main deck (Part II) - zero LLM tokens · hash-locked · audited')
p = b1
cx, cy, cw, ch = content_shapes(p)
place_pic(p, rf'{D}\\arch_pipeline_en.png', cx, cy, cw, ch - 0.1)

""")

# ============ B3: engineering list + runtime scope ============
old_b3_tail = """add_text(p, cx, cy + 4.1, cw, 0.35,
         'Smoke-replay artifacts archived: decision_log.json + final_results/rp2350_page11_Pred.json + logs/ (20+ audit JSONs).',
         size=11, color=INK)
add_text(p, cx, cy + 4.5, cw, 0.35,
         'Entry semantics: image ENTRYPOINT=/submission/run.sh (verified in OCI config); /submission/agent_system stays the official placeholder - confirm invocation with organizers.',
         size=10.5, color=MUTED)"""
new_b3_tail = """add_text(p, cx, cy + 3.95, cw, 0.32,
         'Smoke-replay artifacts archived: decision_log.json + final_results/rp2350_page11_Pred.json + logs/ (20+ audit JSONs).',
         size=11.5, color=INK)
add_text(p, cx, cy + 4.3, cw, 0.32,
         'Entry semantics: image ENTRYPOINT=/submission/run.sh (verified in OCI config); /submission/agent_system stays the official placeholder - confirm invocation with organizers.',
         size=11.5, color=MUTED)
add_text(p, cx, cy + 4.68, cw, 0.44,
         'Engineering reliability: Docker delivery · SHA256 integrity (249-file manifest, 3 weight fingerprints, 12-asset preflight) · per-stage audit logs (20+ JSONs/image) · output validation (validate_target + sanitize) · 626+ unit tests with a build-time gate.',
         size=11.5, color=INK)
add_text(p, cx, cy + 5.14, cw, 0.3,
         'Runtime scope: CPU datum 2000.5 s/case (official example page); official GPU timing not measured - no 2-hour compliance claim.',
         size=12, color=MUTED)"""
assert old_b3_tail in s, 'B3 tail not found'
s = s.replace(old_b3_tail, new_b3_tail)

# ============ B4: 98.93 line ============
old_b4 = """add_text(p, cx, cy + 4.7, cw, 0.35,
         'Caliber caution: canonical (40 pages) and page scaling (13 pages) are independent experiments; "+0.31" is the total-score delta (connectivity +0.7pp).',
         size=11, color=MUTED)"""
new_b4 = """add_text(p, cx, cy + 4.62, cw, 0.32,
         'Caliber caution: canonical (40 pages) and page scaling (13 pages) are independent experiments; "+0.31" is the total-score delta (connectivity +0.7pp).',
         size=11.5, color=MUTED)
add_text(p, cx, cy + 4.98, cw, 0.32,
         'Local development replay: 98.93 (13-page public set, v101-score) - development only, not comparable with the official 74.',
         size=12, color=INK)"""
assert old_b4 in s, 'B4 note not found'
s = s.replace(old_b4, new_b4)

# ============ B5: answers 10.5 -> 11 ============
s = s.replace("add_text(p, x + 0.56, y + 0.52, 4.95, 0.85, a, size=10.5, color=MUTED)",
              "add_text(p, x + 0.56, y + 0.52, 4.95, 0.85, a, size=11, color=MUTED)")

# ============ B2: provenance 10.5 -> 11 ============
s = s.replace("size=10.5, color=MUTED)\n\n# B3 runtime", "size=11, color=MUTED)\n\n# B3 runtime")

open(p, 'w', encoding='utf-8').write(s)
print('story-polish patch applied OK')
