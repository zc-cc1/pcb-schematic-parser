# -*- coding: utf-8 -*-
"""Teammate-integration patch: rewrite build_official_deck.py page blocks so the deck
absorbs the teammate deck's clearer layouts (their P3/P4/P5/P6/P8/P10) while keeping
every verified number and the official-template shell.

Verified-data rules baked in here:
- v100/v102 97.46->99.02 table is NOT added (no original experiment record found; Backup D keeps 98.93 v101-score).
- 238 train pages + 180 crops + 59 val pages ARE added to Backup B (confirmed in detector_training_report.md
  and detector_balanced_training_report.md).
- Port-direction rule presented as implemented capability (export_semantics.py, 14 unit tests pass locally).
"""
import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
P = r'E:\AI_Workplace\精英挑战赛\_audit_workspace\vision-fetch\docs\eda_final_defense\slides\build_official_deck.py'
s = open(P, encoding='utf-8').read()

def block_replace(s, start_marker, end_marker, new_block):
    lines = s.split('\n')
    si = next(i for i, l in enumerate(lines) if start_marker in l)
    ei = next(i for i, l in enumerate(lines) if end_marker in l and i > si)
    lines[si:ei] = new_block.split('\n')
    return '\n'.join(lines)

def rep1(s, old, new):
    assert s.count(old) == 1, f'marker not unique: {old[:60]!r} count={s.count(old)}'
    return s.replace(old, new)

# ============ 0. helper: title 26pt / subtitle 16pt ============
s = rep1(s, "r.font.size = Pt(22); r.font.bold = True; r.font.name = FONT",
         "r.font.size = Pt(26); r.font.bold = True; r.font.name = FONT")
s = rep1(s, "set_text(sh, subtitle, size=13, color=MUTED, font=FONT)",
         "set_text(sh, subtitle, size=16, color=MUTED, font=FONT)")

# ============ 1. P3 (deck) Part I: banner + 4 rows + PNG->JSON strip ============
s = block_replace(s, '# ---------- Part I page: problem & strategy ----------',
                  '# ---------- Part II page: four-layer architecture ----------',
"""# ---------- Part I page: problem & strategy ----------
set_title_subtitle(part1, None, 'Problem understanding & solution strategy')
cx, cy, cw, ch = content_shapes(part1)
add_text(part1, cx, cy + 0.02, cw, 0.56, 'The Hard Part Is Not Finding Symbols. ', size=22, bold=True, color=NAVY)
tbx = part1.shapes[-1].text_frame
r = tbx.paragraphs[0].add_run(); r.text = "It's Recovering Connectivity."
r.font.size = Pt(22); r.font.bold = True; r.font.color.rgb = ORANGE; r.font.name = FONT
add_text(part1, cx, cy + 0.62, cw, 0.32,
         [('Perceive', {'size': 15, 'bold': True, 'color': CYAN})], size=15)
tbx = part1.shapes[-1].text_frame
r = tbx.paragraphs[0].add_run(); r.text = '  identify parts, text, pins and wires      '; r.font.size = Pt(14); r.font.color.rgb = MUTED; r.font.name = FONT
r = tbx.paragraphs[0].add_run(); r.text = 'Connect'; r.font.size = Pt(15); r.font.bold = True; r.font.color.rgb = NAVY; r.font.name = FONT
r = tbx.paragraphs[0].add_run(); r.text = '  build geometry and electrical topology      '; r.font.size = Pt(14); r.font.color.rgb = MUTED; r.font.name = FONT
r = tbx.paragraphs[0].add_run(); r.text = 'Verify'; r.font.size = Pt(15); r.font.bold = True; r.font.color.rgb = ORANGE; r.font.name = FONT
r = tbx.paragraphs[0].add_run(); r.text = '  repair and calibrate with evidence'; r.font.size = Pt(14); r.font.color.rgb = MUTED; r.font.name = FONT

# shared-ID strip: PNG -> Components/Pins/Nets -> Target JSON (teammate P3 idea, our real data)
sy = cy + 1.00
add_card(part1, cx, sy, cw, 0.84, fill=WHITE)
place_pic(part1, rf'{D}\\ti_p5_input.png', cx + 0.15, sy + 0.09, 1.26, 0.66)
ar = part1.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(cx + 1.52), Inches(sy + 0.31), Inches(0.30), Inches(0.22))
ar.fill.solid(); ar.fill.fore_color.rgb = CYAN; ar.line.fill.background()
add_text(part1, cx + 1.94, sy + 0.13, 3.75, 0.32, 'Components · Pins · Nets', size=16, bold=True, color=NAVY)
add_text(part1, cx + 1.94, sy + 0.47, 3.75, 0.28, 'linked by one shared component ID', size=14, color=MUTED)
ar2 = part1.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(cx + 5.80), Inches(sy + 0.31), Inches(0.30), Inches(0.22))
ar2.fill.solid(); ar2.fill.fore_color.rgb = CYAN; ar2.line.fill.background()
add_text(part1, cx + 6.22, sy + 0.06, 5.15, 0.28, 'Target JSON', size=14, bold=True, color=NAVY)
add_text(part1, cx + 6.22, sy + 0.36, 5.15, 0.26,
         '"components": {"U3": {"bbox": [539,281,689,731]}, ...}', size=13.5, color=INK, font='Consolas')
add_text(part1, cx + 6.22, sy + 0.62, 5.15, 0.26,
         '"nets": {"LOG_NET": ["C1.2", "U3.U3"]}', size=13.5, color=INK, font='Consolas')

ch_items = [
    ('Dense component detection', 'small SMD parts, missed / confused classes', 'Dual-path YOLO (whole + SAHI) + WBF fusion', 'per-class thresholds, detection F1 0.883'),
    ('Pin localization & ownership', 'multi-pin ICs, offsets, wrong binding', 'Dual-head U-Net heatmaps + cost assignment', 'peak decode with sub-pixel refinement'),
    ('Text - part association', 'crowded / tilted annotations', 'Multi-view OCR under evidence gates', '>=2 views @ 0.98 confidence, budgets'),
    ('Topology & connectivity', 'T/cross nodes, broken wires, flywires', 'Conservative graph + transactional repair', 'avoid false links before recovery'),
]
row_h = 0.74
for i, (ct, cd, st, sd) in enumerate(ch_items):
    y = cy + 1.98 + i * (row_h + 0.10)
    add_card(part1, cx, y, 5.28, row_h, fill=WHITE)
    add_text(part1, cx + 0.14, y + 0.05, 5.0, 0.32, ct, size=16, bold=True, color=NAVY)
    add_text(part1, cx + 0.14, y + 0.40, 5.0, 0.30, cd, size=14, color=MUTED)
    sld_arw = part1.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(cx + 5.44), Inches(y + row_h / 2 - 0.13), Inches(0.46), Inches(0.26))
    sld_arw.fill.solid(); sld_arw.fill.fore_color.rgb = CYAN; sld_arw.line.fill.background()
    add_card(part1, cx + 6.1, y, 5.4, row_h, fill=PANEL)
    add_text(part1, cx + 6.24, y + 0.05, 5.15, 0.32, st, size=16, bold=True, color=INK)
    add_text(part1, cx + 6.24, y + 0.40, 5.15, 0.30, sd, size=14, color=MUTED)

""")

# ============ 2. P4 (deck) four-layer: font bumps ============
s = block_replace(s, '# ---------- Part II page: four-layer architecture ----------',
                  '# ---- P3 Visual Perception ----',
"""# ---------- Part II page: four-layer architecture ----------
set_title_subtitle(part2, None, 'End-to-end system architecture')
cx, cy, cw, ch = content_shapes(part2)
layers = [
    ('1   Visual Perception', 'YOLO dual-path + WBF fusion  ·  dual-head U-Net (wire + pin)  ·  PP-OCRv6 offline OCR', PANEL, NAVY),
    ('2   Geometric Processing', 'pin ownership assignment  ·  skeleton → segments  ·  IC / LED frames  ·  tip alignment', WHITE, NAVY),
    ('3   Connectivity Reasoning', 'wire graph & junction policy  ·  physical networks  ·  flywire / power merges  ·  field completion S14-S24', '#E8F6F8', CYAN),
    ('4   Evidence-Gated Correction', 'transactional repair engine (15 stages)  ·  official format calibration  ·  validated JSON export', '#FFF7E9', ORANGE),
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
         'Full 11-stage detail: Backup A  ·  zero LLM tokens  ·  hash-locked & audited',
         size=14, color=MUTED)

""")

# ============ 3. P5 (deck) Visual Perception: teammate P4 structure ============
s = block_replace(s, '# ---- P3 Visual Perception ----', '# ---- P4 Topology Reconstruction ----',
'''# ---- P3 Visual Perception (teammate-style: PNG -> modules -> evidence) ----
pP3 = new_body('Visual Perception', 'Three default-on models produce candidates; conclusions come from evidence gates')
p = pP3
cx, cy, cw, ch = content_shapes(p)

# left: input card + model-origin legend
add_card(p, cx, cy + 0.08, 2.35, 2.30, fill=WHITE)
add_text(p, cx + 0.12, cy + 0.16, 2.1, 0.32, 'Schematic PNG', size=16, bold=True, color=NAVY)
place_pic(p, rf'{D}\\ti_p5_input.png', cx + 0.12, cy + 0.56, 2.11, 1.13)
add_text(p, cx + 0.12, cy + 1.78, 2.1, 0.26, 'official example page', size=14, color=MUTED)
add_text(p, cx, cy + 2.66, 2.35, 0.30, 'Model origins', size=15, bold=True, color=INK)
origins = [
    ('YOLO nano', 'self-trained · F1 0.883', CYAN),
    ('WirePinUNet', 'self-trained (milesial)', NAVY),
    ('PP-OCRv6', 'open pre-trained (offline)', NAVY),
    ('JunctionCNN', 'OFF by default (optional)', ORANGE),
]
for i, (t1, d1, dc) in enumerate(origins):
    y = cy + 3.04 + i * 0.64
    dot = p.shapes.add_shape(MSO_SHAPE.OVAL, Inches(cx + 0.02), Inches(y + 0.05), Inches(0.13), Inches(0.13))
    dot.fill.solid(); dot.fill.fore_color.rgb = dc; dot.line.fill.background()
    add_text(p, cx + 0.24, y, 2.1, 0.28, t1, size=15, bold=True, color=INK)
    add_text(p, cx + 0.24, y + 0.30, 2.1, 0.26, d1, size=13.5, color=MUTED)

# trunk from input card to the three module cards
tr = p.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(cx + 2.51), Inches(cy + 0.48), Inches(0.025), Inches(3.42))
tr.fill.solid(); tr.fill.fore_color.rgb = LINE; tr.line.fill.background()
stub = p.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(cx + 2.37), Inches(cy + 1.18), Inches(0.15), Inches(0.025))
stub.fill.solid(); stub.fill.fore_color.rgb = LINE; stub.line.fill.background()

# middle: visual modules
add_text(p, cx + 2.80, cy + 0.06, 3.30, 0.32, 'Visual modules', size=18, bold=True, color=NAVY)
modules = [
    ('YOLO nano detector', 'whole image + SAHI tiles; WBF fusion with per-class thresholds'),
    ('PP-OCRv6 (RapidOCR)', 'multi-view OCR tokens beside parts, pins and text stacks'),
    ('Dual-output U-Net', 'shared image features -> wire mask + pin-tip heatmap'),
]
mx = cx + 2.80
for i, (t, d) in enumerate(modules):
    y = cy + 0.50 + i * 1.24
    add_card(p, mx, y, 3.30, 1.10, fill=PANEL)
    add_text(p, mx + 0.14, y + 0.08, 3.0, 0.32, t, size=17, bold=True, color=NAVY)
    add_text(p, mx + 0.14, y + 0.44, 3.02, 0.58, d, size=14, color=INK)
    arw = p.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(mx - 0.28), Inches(y + 0.45), Inches(0.24), Inches(0.20))
    arw.fill.solid(); arw.fill.fore_color.rgb = CYAN; arw.line.fill.background()

# right: evidence passed downstream (real local crops)
ex = cx + 6.50
add_text(p, ex, cy + 0.06, 5.05, 0.32, 'Evidence passed downstream', size=18, bold=True, color=NAVY)
ev_rows = [
    ('ti_p5_yolo.png', 'Components', '35 classes with bounding boxes'),
    ('ti_p5_ocr.png', 'OCR tokens', 'text · position · confidence (>=2 views @ 0.98)'),
]
for i, (img, t, d) in enumerate(ev_rows):
    y = cy + 0.50 + i * 1.24
    place_pic(p, rf'{D}\\' + img, ex, y + 0.02, 1.90, 0.96)
    add_text(p, ex + 2.10, y + 0.06, 2.95, 0.30, t, size=16, bold=True, color=INK)
    add_text(p, ex + 2.10, y + 0.40, 2.95, 0.56, d, size=14, color=MUTED)
    arw = p.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(ex - 0.36), Inches(y + 0.40), Inches(0.32), Inches(0.20))
    arw.fill.solid(); arw.fill.fore_color.rgb = CYAN; arw.line.fill.background()
# dual-head evidence row: wire mask + pin heatmap minis
y3 = cy + 0.50 + 2 * 1.24
place_pic(p, rf'{D}\\ti_p5_wire.png', ex, y3 + 0.10, 0.92, 0.49)
place_pic(p, rf'{D}\\ti_p5_pin.png', ex + 0.98, y3 + 0.10, 0.92, 0.49)
add_text(p, ex, y3 + 0.64, 0.92, 0.24, 'wire mask', size=14, color=MUTED, align=PP_ALIGN.CENTER)
add_text(p, ex + 0.98, y3 + 0.64, 0.92, 0.24, 'pin heatmap', size=14, color=MUTED, align=PP_ALIGN.CENTER)
add_text(p, ex + 2.10, y3 + 0.06, 2.95, 0.30, 'Wires and pins', size=16, bold=True, color=INK)
add_text(p, ex + 2.10, y3 + 0.40, 2.95, 0.56, 'operating wire mask + pin-tip heatmap (dual head)', size=14, color=MUTED)
arw = p.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(ex - 0.36), Inches(y3 + 0.40), Inches(0.32), Inches(0.20))
arw.fill.solid(); arw.fill.fore_color.rgb = CYAN; arw.line.fill.background()

# bottom stats band
add_text(p, cx + 2.80, cy + 4.28, 6.55, 0.36,
         [('431 pin candidates decoded', {'size': 18, 'bold': True, 'color': INK})], size=18)
tbx = p.shapes[-1].text_frame
r = tbx.paragraphs[0].add_run(); r.text = '    287'; r.font.size = Pt(17); r.font.bold = True; r.font.color.rgb = NAVY; r.font.name = FONT
r = tbx.paragraphs[0].add_run(); r.text = ' assigned · '; r.font.size = Pt(14); r.font.color.rgb = MUTED; r.font.name = FONT
r = tbx.paragraphs[0].add_run(); r.text = '115'; r.font.size = Pt(17); r.font.bold = True; r.font.color.rgb = NAVY; r.font.name = FONT
r = tbx.paragraphs[0].add_run(); r.text = ' ambiguous · '; r.font.size = Pt(14); r.font.color.rgb = MUTED; r.font.name = FONT
r = tbx.paragraphs[0].add_run(); r.text = '29'; r.font.size = Pt(17); r.font.bold = True; r.font.color.rgb = NAVY; r.font.name = FONT
r = tbx.paragraphs[0].add_run(); r.text = ' unowned'; r.font.size = Pt(14); r.font.color.rgb = MUTED; r.font.name = FONT
add_text(p, cx + 2.80, cy + 4.66, 6.55, 0.30,
         'pin ownership assignment follows in Geometric Processing (part 2)', size=13.5, color=MUTED)
add_card(p, cx + 9.65, cy + 4.22, 2.85, 0.92, fill=WHITE)
add_text(p, cx + 9.75, cy + 4.32, 2.65, 0.44, 'Detection F1 0.883', size=19, bold=True, color=ORANGE, align=PP_ALIGN.CENTER)
add_text(p, cx + 9.75, cy + 4.80, 2.65, 0.28, '59 cases · fused config', size=14, color=MUTED, align=PP_ALIGN.CENTER)

''')

# ============ 4. P6 (deck) Topology: big real overlay + three steps ============
s = block_replace(s, '# ---- P4 Topology Reconstruction ----', '# ---- P5 Evidence-Gated Repair ----',
'''# ---- P4 Topology Reconstruction (teammate-style: big overlay + three steps) ----
pP4 = new_body('Topology Reconstruction', 'Wire probability map -> skeleton -> segments -> networks')
p = pP4
cx, cy, cw, ch = content_shapes(p)
place_pic(p, rf'{D}\\ti_p6_main.png', cx, cy + 0.04, 6.20, 4.95)
add_text(p, cx, cy + 5.02, 6.20, 0.24,
         'Local prediction overlay, official example page - markers 1/2/3 = steps at right',
         size=14, color=MUTED, align=PP_ALIGN.CENTER)
sx = cx + 6.45
steps = [
    ('1 · Pin ownership', 'Match each decoded endpoint to the component edge and its nearby lead direction - 287 of 431 candidates assigned.'),
    ('2 · Wire tracing', 'Trace segments between endpoints, turns and junctions in the operating wire mask - T/cross nodes split conservatively, gaps recovered only by hard evidence.'),
    ('3 · Net membership', 'Group terminals that share a supported path and keep the route geometry - same-name flywire labels and four-gate power merges included.'),
]
for i, (t, d) in enumerate(steps):
    y = cy + 0.04 + i * 1.16
    add_text(p, sx, y, 5.10, 0.34, t, size=18, bold=True, color=NAVY)
    add_text(p, sx, y + 0.38, 5.10, 0.74, d, size=14.5, color=INK)
add_card(p, sx, cy + 3.58, 5.10, 1.41, fill=PANEL)
add_text(p, sx + 0.16, cy + 3.68, 4.8, 0.60,
         [('238 Physical-Network Candidates / 362 Edges → 63 Logical Nets', {'size': 17, 'bold': True, 'color': NAVY})], size=17)
add_text(p, sx + 0.16, cy + 4.34, 4.8, 0.56,
         'Candidates are the graph-stage intermediate; merges give the final output.',
         size=14, color=INK)

''')

# ============ 5. P7 (deck) Repair: font bumps only ============
s = block_replace(s, '# ---- P5 Evidence-Gated Repair ----', '# ---- P6 Output Calibration ----',
'''# ---- P5 Evidence-Gated Repair ----
pP5 = new_body('Repair Without Guessing', 'Page-13 junction reconciliation - merge only on hard evidence')
p = pP5
cx, cy, cw, ch = content_shapes(p)
place_pic(p, rf'{D}\\p7_gap_trim.png', cx, cy + 0.1, 3.6, 3.6)
add_text(p, cx, cy + 3.78, 3.65, 0.30, 'Real crop from page 13 (development record)', size=14, color=MUTED, align=PP_ALIGN.CENTER)
add_text(p, cx + 3.95, cy + 0.05, 8.55, 0.5, 'Page-13 wire repair', size=20, bold=True, color=NAVY)
mets = [
    ('TP / FP / FN', '32/28/17', '46/13/3'),
    ('Wire F1', '58.72%', '85.19%'),
    ('Connectivity', '67.14%', '87.53%'),
    ('Score (single page)', '82.871', '92.656'),
]
for i, (m, b, a) in enumerate(mets):
    y = cy + 0.72 + i * 0.5
    add_text(p, cx + 3.95, y, 2.6, 0.36, m, size=15, bold=True, color=INK)
    add_text(p, cx + 6.6, y, 1.9, 0.36, b, size=16, color=MUTED)
    add_text(p, cx + 8.35, y, 0.6, 0.36, '→', size=15, bold=True, color=INK)
    add_text(p, cx + 8.95, y, 2.2, 0.36, a, size=16, bold=True, color=ORANGE)
mech = [
    'Merge evidence: >=60 px collinear overlap (+-3 deg, <=2 px), or red dot + >=95% ink support',
    'Members and edges only - components and pins untouched; <=8 merges per page',
]
for i, t in enumerate(mech):
    add_text(p, cx + 3.95, cy + 2.85 + i * 0.40, 8.55, 0.36, t, size=14, color=INK)
add_text(p, cx + 3.95, cy + 3.72, 8.55, 0.52,
         'Other 12 pages byte-identical (frozen cache replay) - single-page development-set ablation, not an official overall metric.',
         size=14, italic=True, color=MUTED)
add_text(p, cx + 3.95, cy + 4.30, 8.55, 0.34,
         [('The wire-F1 gain comes from junction reconciliation - not from the OCR confidence gate.', {'size': 14.5, 'bold': True, 'color': NAVY})], size=14.5)
add_text(p, cx + 3.95, cy + 4.72, 8.55, 0.34,
         [('Evidence-gated OCR (>=2 views @ 0.98) is a separate capability - see Backup E.', {'size': 14, 'color': MUTED})], size=14)

''')

# ============ 6. P8 (deck) Output Calibration: port rule + compact calibration ============
s = block_replace(s, '# ---- P6 Output Calibration ----', '# ---------- Part III page: official results ----------',
'''# ---- P6 Output Calibration (teammate P6 image-based export rules + our calibration) ----
pP6 = new_body('Output Calibration for Official Evaluation',
               'Statistical calibration to the official annotation conventions - not per-image lookups')
p = pP6
cx, cy, cw, ch = content_shapes(p)

# left: calibration cards (compact)
add_text(p, cx, cy + 0.04, 5.25, 0.34, 'Calibration to official conventions', size=17, bold=True, color=NAVY)
cards = [
    ('Type & field conventions', 'resistor -> r ... (145 matched parts) · Name==key from 9,186-part statistics - the box keeps its part number', CYAN),
    ('Bounding-box calibration (two scopes)', 'canonical (40-page public subset): within-20px 0.905 -> 1.000 · page-adaptive (13 released): 150/163 -> 159/163', NAVY),
    ('Pin-tip alignment', '18 px tip snap (measured 16-22 px plateau) + 1000-part pin library', ORANGE),
]
for i, (t, d, c) in enumerate(cards):
    y = cy + 0.44 + i * 1.16
    add_card(p, cx, y, 5.25, 1.04, fill=WHITE)
    add_text(p, cx + 0.16, y + 0.07, 4.9, 0.32, t, size=16, bold=True, color=INK)
    add_text(p, cx + 0.16, y + 0.42, 4.95, 0.56, d, size=14, color=MUTED)
    dot = p.shapes.add_shape(MSO_SHAPE.OVAL, Inches(cx + 4.97), Inches(y + 0.10), Inches(0.14), Inches(0.14))
    dot.fill.solid(); dot.fill.fore_color.rgb = c; dot.line.fill.background()
# U3 bbox strip (kept from our deck; not repeated on P11)
uy = cy + 3.94
add_card(p, cx, uy, 5.25, 1.00, fill=PANEL)
place_pic(p, rf'{D}\\p8_u3_bbox_compare.png', cx + 0.12, uy + 0.07, 0.50, 0.86)
add_text(p, cx + 0.78, uy + 0.08, 1.55, 0.46, '1~2 px', size=28, bold=True, color=ORANGE)
add_text(p, cx + 2.40, uy + 0.10, 2.75, 0.80,
         'U3 (RP2350A) bbox vs official example annotation (official 20 px tolerance)', size=14, color=INK)
add_text(p, cx, uy + 1.10, 5.25, 0.28,
         'Both bbox scopes are independent - never multiplied (Backup D).', size=14, color=MUTED)

# right: directional port rule + its JSON fields
rx = cx + 5.60
add_text(p, rx, cy + 0.04, 5.95, 0.34, 'Image-based export rules - directional port', size=17, bold=True, color=NAVY)
place_pic(p, rf'{D}\\ti_p8_port.png', rx, cy + 0.46, 4.42, 1.35)
add_text(p, rx + 0.28, cy + 1.86, 1.55, 0.28, 'Wired terminal', size=14, color=INK, align=PP_ALIGN.CENTER)
add_text(p, rx + 3.30, cy + 1.86, 1.20, 0.28, 'Arrow tip', size=14, color=INK, align=PP_ALIGN.CENTER)
add_text(p, rx, cy + 2.20, 5.90, 0.58,
         'The terminal sits opposite the tip -> output port; on the tip side -> input. Caption stays in Name; value -> null.',
         size=16, color=INK)
add_text(p, rx, cy + 2.82, 5.90, 0.30,
         'Fires only when the 5-vertex convex flag and a single terminal agree - else fields kept (14 unit tests).',
         size=14, color=MUTED)
add_card(p, rx, cy + 3.24, 5.95, 2.30, fill=PANEL)
add_text(p, rx + 0.20, cy + 3.34, 5.5, 0.32, 'Port fields in Target JSON', size=16, bold=True, color=NAVY)
add_text(p, rx + 0.20, cy + 3.70, 5.5, 0.34, '"type": "net_output"    (legacy: net_bidirection)', size=17, color=INK, font='Consolas')
add_text(p, rx + 0.20, cy + 4.10, 5.5, 0.34, '"Name": "VOUT"', size=17, color=INK, font='Consolas')
add_text(p, rx + 0.20, cy + 4.50, 5.5, 0.34, '"value": null', size=17, color=INK, font='Consolas')
add_text(p, rx + 0.20, cy + 4.92, 5.55, 0.56,
         'Written only on proven geometry - export_semantics.py, default profile observed; legacy via PCB_EXPORT_SEMANTICS=legacy.',
         size=14, color=MUTED)

''')

# ============ 7. deck P9 (Part III): font bumps ============
s = block_replace(s, '# ---------- Part III page: official results ----------',
                  '# ---- P8 Quantitative Results and Case Studies ----',
'''# ---------- Part III page: official results ----------
set_title_subtitle(part3, None, 'Official evaluation results & quantitative evidence')
cx, cy, cw, ch = content_shapes(part3)
chip(part3, cx + 0.1, cy + 0.15, 3.6, 1.9, '74', 'official final score', ORANGE, 54, 14, fill=WHITE)
chip(part3, cx + 3.9, cy + 0.15, 3.6, 1.9, '47', 'national rank', NAVY, 54, 14, fill=WHITE)
chip(part3, cx + 7.7, cy + 0.15, 3.6, 1.9, 'Top 61', 'finalist qualification', NAVY, 44, 14, fill=WHITE)
add_card(part3, cx, cy + 2.3, 11.55, 1.42, fill=PANEL)
add_text(part3, cx + 0.2, cy + 2.4, 11.2, 0.4, 'Version iteration', size=16, bold=True, color=INK)
tbx = part3.shapes[-1].text_frame
r = tbx.paragraphs[0].add_run(); r.text = '   28.92'; r.font.size = Pt(16); r.font.bold = True; r.font.color.rgb = NAVY; r.font.name = FONT
r = tbx.paragraphs[0].add_run(); r.text = ' (old build 8422)   →   '; r.font.size = Pt(14); r.font.color.rgb = MUTED; r.font.name = FONT
r = tbx.paragraphs[0].add_run(); r.text = '74'; r.font.size = Pt(16); r.font.bold = True; r.font.color.rgb = ORANGE; r.font.name = FONT
r = tbx.paragraphs[0].add_run(); r.text = ' (final submission, +45)'; r.font.size = Pt(14); r.font.color.rgb = MUTED; r.font.name = FONT
add_text(part3, cx + 0.2, cy + 2.82, 11.2, 0.8,
         'Fixes behind the jump: unnumbered power/ground exports · LED text & boxes · IC name priority & observed frames · engineering numbering · export semantics.',
         size=14, color=INK)
add_text(part3, cx, cy + 4.4, cw, 0.36,
         'Local development replay score (13-page public set): moved to Backup D - development only, not comparable with official results.',
         size=14, color=MUTED)

''')

# ============ 8. deck P10 (Quantitative): font bumps ============
s = block_replace(s, '# ---- P8 Quantitative Results and Case Studies ----',
                  '# ---- P9 End-to-End Output and Engineering Validation ----',
'''# ---- P8 Quantitative Results and Case Studies ----
pP8 = new_body('Quantitative Results and Case Studies',
               'Four experimental indicators - development-set experiments and single-case ablations')
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
    add_text(p, x + 0.2, y + 1.25, 5.2, 0.30, met, size=14, color=INK)
    add_text(p, x + 0.2, y + 1.57, 5.2, 0.30, scope, size=13.5, color=MUTED)
add_text(p, cx, cy + 4.70, cw, 0.34,
         'Development-set experiments and single-case ablations - not official per-dimension scores. Full table: Backup D.',
         size=14, bold=True, color=INK)
add_text(p, cx, cy + 5.10, cw, 0.30,
         'Ablation protocol: one module toggled on a frozen model/OCR cache; the other 12 pages stay byte-identical.',
         size=13.5, color=MUTED)

''')

# ============ 9. deck P11 (End-to-End): teammate P8 layout ============
s = block_replace(s, '# ---- P9 End-to-End Output and Engineering Validation ----',
                  '# ---------- Part IV page: conclusions ----------',
'''# ---- P9 End-to-End Output (teammate P8: original | overlay + legend + counts) ----
pP9 = new_body('End-to-End Output and Engineering Validation',
               'Official example page, full pipeline - from PNG to validated JSON')
p = pP9
cx, cy, cw, ch = content_shapes(p)
place_pic(p, rf'{D}\\ti_p11_orig.png', cx, cy + 0.04, 4.25, 3.02)
add_text(p, cx, cy + 3.14, 4.25, 0.30, 'Original schematic (official example page)', size=14, color=MUTED, align=PP_ALIGN.CENTER)
place_pic(p, rf'{D}\\ti_p11_overlay.png', cx + 5.60, cy + 0.04, 4.25, 3.02)
add_text(p, cx + 5.60, cy + 3.14, 4.25, 0.30, 'Prediction overlay (same page)', size=14, color=MUTED, align=PP_ALIGN.CENTER)
# stacked output records (right)
recs = [('89', 'components'), ('165', 'pins'), ('63', 'logical nets')]
for i, (n, lb) in enumerate(recs):
    y = cy + 0.10 + i * 1.02
    add_text(p, cx + 10.10, y, 2.35, 0.48, n, size=28, bold=True, color=NAVY)
    add_text(p, cx + 10.10, y + 0.50, 2.35, 0.28, lb, size=14, color=MUTED)
# legend row
ly = cy + 3.62
lg = p.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(cx), Inches(ly + 0.05), Inches(0.24), Inches(0.16))
lg.fill.solid(); lg.fill.fore_color.rgb = RGBColor(0x00, 0x98, 0x51); lg.line.fill.background()
add_text(p, cx + 0.34, ly, 1.85, 0.28, 'Component boxes', size=14, color=INK)
dot = p.shapes.add_shape(MSO_SHAPE.OVAL, Inches(cx + 2.30), Inches(ly + 0.04), Inches(0.17), Inches(0.17))
dot.fill.solid(); dot.fill.fore_color.rgb = RGBColor(0xC0, 0x39, 0x2B); dot.line.fill.background()
add_text(p, cx + 2.57, ly, 1.60, 0.28, 'Pin endpoints', size=14, color=INK)
ln = p.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(cx + 4.28), Inches(ly + 0.10), Inches(0.30), Inches(0.055))
ln.fill.solid(); ln.fill.fore_color.rgb = ORANGE; ln.line.fill.background()
add_text(p, cx + 4.68, ly, 2.45, 0.28, 'Wire paths (per-net color)', size=14, color=INK)
add_text(p, cx + 7.35, ly, 5.15, 0.28, 'validated - never infers text, coordinates or joins', size=14, color=MUTED)
# real JSON fragments
for i, (key, l1, c) in enumerate([
        ('components', '"U3": box RP2350A · bbox [539,281,689,731]', RGBColor(0x0E, 0x7C, 0x8C)),
        ('pins', '"U3".pin_51: USB_DM @ (513, 561)', NAVY),
        ('nets', '"LOG_NET": (C1.2, U3.U3) · 3 edges', ORANGE)]):
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
    add_text(p, cx + 1.60, y + 0.05, 6.20, 0.30, l1, size=14, color=INK, font='Consolas')
add_text(p, cx + 8.10, cy + 4.14, 4.40, 0.70,
         'Output JSON (real fragments, official example page) - IDs tie every record back to the image.',
         size=14, color=INK)

''')

# ============ 10. deck P12 (Part IV): two columns + engineering band ============
s = block_replace(s, '# ---------- Part IV page: conclusions ----------',
                  '# ---------- Backups B1-B5 ----------',
'''# ---------- Part IV page: conclusions (teammate P10 two columns) ----------
set_title_subtitle(part4, None, 'Conclusions and future work')
cx, cy, cw, ch = content_shapes(part4)
add_text(part4, cx, cy + 0.04, 5.60, 0.36, 'What the system delivers', size=20, bold=True, color=NAVY)
add_text(part4, cx + 6.10, cy + 0.04, 5.45, 0.36, 'Future development', size=20, bold=True, color=NAVY)
div = part4.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(cx + 5.82), Inches(cy + 0.10), Inches(0.014), Inches(4.62))
div.fill.solid(); div.fill.fore_color.rgb = LINE; div.line.fill.background()
contribs = [
    ('Pin-to-net reconstruction', 'Symbol boundaries, pin locations and wire paths decide which pins share a net - merged only on hard evidence.'),
    ('Traceable Target export', 'Component and pin IDs tie every attribute, net member and wire segment back to the source image.'),
    ('Score-aware output calibration', 'bbox conventions · Name==key · pin-library mapping, fitted to official annotation statistics.'),
]
for i, (t, d) in enumerate(contribs):
    y = cy + 0.56 + i * 1.16
    dot = part4.shapes.add_shape(MSO_SHAPE.OVAL, Inches(cx), Inches(y + 0.02), Inches(0.30), Inches(0.30))
    dot.fill.solid(); dot.fill.fore_color.rgb = ORANGE; dot.line.fill.background()
    add_text(part4, cx, y + 0.03, 0.30, 0.30, str(i + 1), size=13, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    add_text(part4, cx + 0.44, y, 5.15, 0.32, t, size=17, bold=True, color=INK)
    add_text(part4, cx + 0.44, y + 0.36, 5.15, 0.70, d, size=14.5, color=MUTED)
add_text(part4, cx, cy + 4.10, 5.55, 0.60,
         'JunctionCNN: self-trained but OFF in the default pipeline - an optional enhancement, not the inference backbone.',
         size=14, italic=True, color=MUTED)
futures = [
    ('Independent evaluation', 'Hold out schematics from unseen sources; measure transfer across unfamiliar symbols and drawing styles.'),
    ('Dense pins & crossing wires', 'Audit missed pins and false junctions to target where local image evidence must improve.'),
    ('Junction model & data growth', 'Promote the trained junction CNN after validation; expand hard-case and cross-domain datasets.'),
]
for i, (t, d) in enumerate(futures):
    y = cy + 0.56 + i * 1.16
    dot = part4.shapes.add_shape(MSO_SHAPE.OVAL, Inches(cx + 6.10), Inches(y + 0.02), Inches(0.30), Inches(0.30))
    dot.fill.solid(); dot.fill.fore_color.rgb = CYAN; dot.line.fill.background()
    add_text(part4, cx + 6.10, y + 0.03, 0.30, 0.30, str(i + 1), size=13, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    add_text(part4, cx + 6.54, y, 5.00, 0.32, t, size=17, bold=True, color=INK)
    add_text(part4, cx + 6.54, y + 0.36, 5.00, 0.70, d, size=14.5, color=MUTED)
add_text(part4, cx + 6.10, cy + 4.10, 5.45, 0.60,
         'Reported aggregates apply to public development images; official 30~50-case set stays black-box.',
         size=14, italic=True, color=MUTED)
add_card(part4, cx, cy + 4.80, cw, 0.50, fill=PANEL)
add_text(part4, cx + 0.2, cy + 4.90, cw - 0.4, 0.32,
         'Engineering delivery: Docker pcb-parser:v102 · SHA256 manifest (249 files) · audit logs per stage · 626+ unit tests · zero LLM tokens',
         size=14.5, bold=True, color=NAVY, align=PP_ALIGN.CENTER)

''')

# ============ 11. Backup A: + skeleton strip ============
s = block_replace(s, '# B1 detailed architecture', '# B2 training',
'''# B1 detailed architecture (+ topology detail: 39 junction nodes / skeleton)
b1 = new_body('Backup A · Detailed Architecture',
             'The full 11-stage pipeline - compressed to four layers on the main deck (Part II)')
p = b1
p = b1
cx, cy, cw, ch = content_shapes(p)
place_pic(p, rf'{D}\\arch_pipeline_en.png', cx - 0.10, cy - 0.05, 7.55, 5.34)
place_pic(p, rf'{D}\\ti_ba_skeleton.png', cx + 7.75, cy - 0.05, 3.75, 3.21)
add_text(p, cx + 7.75, cy + 3.24, 3.75, 0.28, 'Junction nodes on the skeleton (U3 zoom)', size=14, color=MUTED, align=PP_ALIGN.CENTER)
add_text(p, cx + 7.75, cy + 3.62, 3.75, 1.60,
         'Operating wire mask -> skeleton: 7210 px + 39 junction nodes (magenta). Global per-net stage: main deck P6; merge policy: P7.',
         size=14, color=INK)

''')

# ============ 12. Backup B: table >=14pt + confirmed detector-data line ============
s = block_replace(s, '# B2 training', '# B3 runtime',
'''# B2 training
ck()
b2 = new_body('Backup B · Model Training and Dataset',
             'All inference weights ship inside the image and are sha256-locked; training happens outside the package')
p = b2
cx, cy, cw, ch = content_shapes(p)
rows = [
    ('Model', 'Architecture', 'Training data', 'Key parameters', 'Origin'),
    ('YOLO nano detector', 'Ultralytics YOLO nano · 35 classes', 'detection_balanced_v2: 238 train pages + 180 crops · 59 val pages', 'imgsz 1536 · 120 ep · batch 4', 'self-trained (HR-DONK warm start, MIT)'),
    ('WirePinUNet dual-head', 'base-32 shared decoder (wire+pin)', 'rasterized source annotations + reviewed masks', 'tiles 512/384 · 40 ep · pos_weight 8/20', 'self-trained (milesial parts, GPL-3)'),
    ('JunctionCNN', '6-channel dual-view CNN', '1,024 synthetic renders + reviewed crops ×12', '30 ep · 0.95 thr · 4-rotation agreement', 'self-trained (OFF by default)'),
    ('PP-OCRv6 OCR', 'det/rec small + cls (ONNX)', 'open pre-trained', 'page 2048 px · pin 2/3/4× × 0/90°', 'called (rapidocr 3.9.2)'),
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
add_text(p, cx, cy + 3.48, cw, 0.34,
         'Detector split is original-page level: crop parent pages come from the train split only; 0001/0002 stay fixed in validation (2462 val boxes).',
         size=14, color=INK)
add_text(p, cx, cy + 3.88, cw, 0.34,
         'Data assets: 1000-part EasyEDA symbol library + 337-part slice + STM32F103C8T6 LQFP48 pin library - bundled, zero network access.',
         size=14, color=INK)
add_text(p, cx, cy + 4.28, cw, 0.60,
         'Provenance: baseline_provenance.json (HR-DONK initial baseline, MIT) · milesial U-Net parts GPL-3 (per-file sha256) · no evaluation annotation leakage in training data.',
         size=14, color=MUTED)

''')

# ============ 13. Backup C: font bumps ============
s = block_replace(s, '# B3 runtime', '# B4 metrics',
'''# B3 runtime
ck()
b3 = new_body('Backup C · Runtime & Reliability',
             'docker run on the evaluation side; direct Python run for development')
p = b3
cx, cy, cw, ch = content_shapes(p)
add_card(p, cx, cy, 5.9, 2.75, fill=RGBColor(0x0D, 0x22, 0x37), line_c=RGBColor(0x1E, 0x4D, 0x8F))
add_text(p, cx + 0.25, cy + 0.2, 5.5, 2.4,
         [('docker run --rm --gpus all \\\\', {'color': RGBColor(0x8F, 0xD8, 0xE2), 'font': 'Consolas'}),
          ('  -v <in>:/input:ro -v <out>:/output:rw \\\\', {'color': RGBColor(0x8F, 0xD8, 0xE2), 'font': 'Consolas'}),
          ('  pcb-parser:v102 -input /input -output /output', {'color': RGBColor(0x8F, 0xD8, 0xE2), 'font': 'Consolas'}),
          ('', {}),
          ('# CPU fallback', {'color': MUTED, 'font': 'Consolas'}),
          ('python agent.py -input <dir> -output <dir> --device cpu', {'color': RGBColor(0x8F, 0xD8, 0xE2), 'font': 'Consolas'}),
          ('# ~30 PCB_* env switches; defaults are tuned values', {'color': MUTED, 'font': 'Consolas'})],
         size=13)
place_pic(p, rf'{D}\\d_recovery_flow_en.png', cx + 6.1, cy + 0.1, 5.6, 2.6)
chain = [('run.sh', 'image entry'), ('agent.py', 'run_case per image'), ('S-stage chain', 'S14→S24 subprocess'),
         ('finalize x15', 'evidence-gated repair'), ('adapter', 'target calibration'), ('final_results', 'Pred.json')]
for i, (t, s2) in enumerate(chain):
    x = cx + i * 1.945
    cv = p.shapes.add_shape(MSO_SHAPE.CHEVRON, Inches(x), Inches(cy + 3.15), Inches(1.86), Inches(0.72))
    cv.fill.solid(); cv.fill.fore_color.rgb = PANEL; cv.line.color.rgb = LINE; cv.line.width = Pt(0.75)
    tf = cv.text_frame; tf.word_wrap = True
    tf.margin_left = Inches(0.14); tf.margin_right = Inches(0.03)
    pa = tf.paragraphs[0]; pa.alignment = PP_ALIGN.CENTER
    r = pa.add_run(); r.text = t; r.font.size = Pt(11); r.font.bold = True; r.font.color.rgb = INK; r.font.name = FONT
    pb = tf.add_paragraph(); pb.alignment = PP_ALIGN.CENTER
    r = pb.add_run(); r.text = s2; r.font.size = Pt(9); r.font.color.rgb = MUTED; r.font.name = FONT
add_text(p, cx, cy + 3.98, cw, 0.32,
         'Smoke-replay artifacts archived: decision_log.json + final_results/rp2350_page11_Pred.json + logs/ (20+ audit JSONs).',
         size=13.5, color=INK)
add_text(p, cx, cy + 4.32, cw, 0.30,
         'Entry semantics: ENTRYPOINT=/submission/run.sh (verified in OCI config); agent_system stays the official placeholder.',
         size=13.5, color=MUTED)
add_text(p, cx, cy + 4.68, cw, 0.60,
         'Engineering reliability: Docker delivery · SHA256 integrity (249-file manifest, 3 weight fingerprints, 12-asset preflight) · per-stage audit logs (20+ JSONs/image) · output validation (validate_target + sanitize) · 626+ unit tests with a build-time gate.',
         size=13.5, color=INK)
add_text(p, cx, cy + 5.36, cw, 0.32,
         'Runtime scope: CPU datum 2000.5 s/case (official example page); official GPU timing not measured - no 2-hour compliance claim.',
         size=13.5, color=MUTED)

''')

# ============ 14. Backup D: table >=14pt, keep 98.93 record ============
s = block_replace(s, '# B4 metrics', '# B5 Q&A',
'''# B4 metrics
ck()
b4 = new_body('Backup D · Metrics & Ablation',
             'Every number traceable; full 17-row table in defense_supplement.md §3')
p = b4
cx, cy, cw, ch = content_shapes(p)
add_text(p, cx, cy + 0.0, cw, 0.30,
         'Caliber caution: canonical (40 pages) and page scaling (13 pages) are independent experiments; "+0.31" is the total-score delta (connectivity +0.7pp).',
         size=13.5, color=MUTED)
add_text(p, cx, cy + 0.32, cw, 0.32,
         'Local development replay: 98.93 (13-page public set, v101-score) - development only, not comparable with the official 74.',
         size=14, bold=True, color=INK)
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
tbl_shape = p.shapes.add_table(len(rows), 4, Inches(cx), Inches(cy + 0.74), Inches(11.55), Inches(4.46))
tbl = tbl_shape.table
for c, wd in zip(tbl.columns, [2.55, 2.20, 2.70, 4.10]):
    c.width = Inches(wd)
for ri, row in enumerate(rows):
    for ci, val in enumerate(row):
        cell = tbl.cell(ri, ci); cell.text = val
        para = cell.text_frame.paragraphs[0]
        para.alignment = PP_ALIGN.RIGHT if ci == 3 else PP_ALIGN.LEFT
        r0 = para.runs[0]; r0.font.size = Pt(14)
        r0.font.bold = (ri == 0) or (ci == 3); r0.font.name = FONT
        r0.font.color.rgb = WHITE if ri == 0 else (NAVY if ci == 3 else INK)
        cell.fill.solid(); cell.fill.fore_color.rgb = NAVY if ri == 0 else (WHITE if ri % 2 else RGBColor(0xF6, 0xF9, 0xFC))

''')

# ============ 15. Backup E: 14pt cards + 18-question pointer ============
s = block_replace(s, '# B5 Q&A', '# ---------- delete requirements slide',
'''# B5 Q&A
ck()
b5 = new_body('Backup E · Technical Q&A', 'Full Chinese handbook: EDA决赛答辩_QA.md (18 questions with evidence-backed answers)')
p = b5
cx, cy, cw, ch = content_shapes(p)
qas = [('Why not an end-to-end LLM?', '95% of the score is pixel-level structural accuracy; LLM coordinates are unreliable and tokens are billed - we use 0.'),
       ('Gap between official 74 and local 98.93?', 'Different scopes: official = 30~50 cases, 2/3 black-box (60% hard); local = 13-page public replay.'),
       ('How are T/cross junctions handled?', 'Conservative split + red-dot detection; learned classifier kept optional (OFF in default pipeline).'),
       ('Broken / missed wires?', 'Gaps logged, never guessed; recovered by >=60 px collinear overlap or red-dot + >=95% ink (+0.31 pts total).'),
       ('Tiny pin numbers unreadable?', 'Strip OCR (30-50 px beyond lead) + row reread + 6-view targeted OCR + 1000-part library (PinF1 0.031->0.549).'),
       ('How is a part number recovered?', 'SW1/SW2: fallback "SW1" -> real part "PTS8153JK250SMTRLFS" via nearby text stack, rotated views (conf 0.998/0.992).')]
for i, (q, a) in enumerate(qas):
    col, row = i % 2, i // 2
    x = cx + col * 5.93; y = cy + row * 1.62
    add_card(p, x, y, 5.62, 1.46, fill=WHITE)
    dot = p.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x + 0.14), Inches(y + 0.16), Inches(0.30), Inches(0.30))
    dot.fill.solid(); dot.fill.fore_color.rgb = NAVY; dot.line.fill.background()
    add_text(p, x + 0.14, y + 0.17, 0.30, 0.30, 'Q', size=13, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    add_text(p, x + 0.56, y + 0.10, 4.95, 0.32, q, size=14, bold=True, color=INK)
    add_text(p, x + 0.56, y + 0.46, 4.95, 0.94, a, size=14, color=MUTED)

''')

open(P, 'w', encoding='utf-8').write(s)
print('patch applied OK')
