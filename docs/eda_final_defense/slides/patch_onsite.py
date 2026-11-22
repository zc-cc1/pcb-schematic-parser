"""Onsite-final patch: rewrite page blocks in build_official_deck.py for projection reading."""
import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
p = r'E:\AI_Workplace\精英挑战赛\_audit_workspace\scripts\build_official_deck.py'
s = open(p, encoding='utf-8').read()

def block_replace(s, start_marker, end_marker, new_block):
    lines = s.split('\n')
    si = next(i for i, l in enumerate(lines) if start_marker in l)
    ei = next(i for i, l in enumerate(lines) if end_marker in l and i > si)
    lines[si:ei] = new_block.split('\n')
    return '\n'.join(lines)

# ============ COVER + TOC ============
s = block_replace(s, '# ---------- cover ----------', '# ---------- Part I page: problem & strategy ----------', """# ---------- cover + TOC ----------
for sh in cover.shapes:
    if sh.name == '标题 7':
        # entry ID EDA2606K5 removed: could not be verified against registration
        # records (folder name alone is not evidence) - project name takes its place
        set_text(sh, 'AI-Driven PCB Schematic Parsing System', size=36, bold=True, color=WHITE)
    elif sh.name == '副标题 8':
        set_text(sh, 'November 28, 2026', size=20, color=WHITE)
add_text(cover, 1.37, 3.92, 10.7, 0.4, 'From Pixels to Verified Electrical Connectivity',
         size=18, color=RGBColor(0x8F, 0xD8, 0xE2))
# remove the duplicate small '目录' text that overlaps the top-right logos
for sh in list(toc.shapes):
    if sh.name == '标题' and sh.has_text_frame and sh.text_frame.text.strip() == '目录':
        sh._element.getparent().remove(sh._element)

""")

# ============ P3 ============
s = block_replace(s, '# ---------- Part I page: problem & strategy ----------',
                  '# ---------- Part II page: four-layer architecture ----------', """# ---------- Part I page: problem & strategy ----------
set_title_subtitle(part1, None, 'Problem understanding & solution strategy')
cx, cy, cw, ch = content_shapes(part1)
add_text(part1, cx, cy + 0.02, cw, 0.6, 'The Hard Part Is Not Finding Symbols. ', size=22, bold=True, color=NAVY)
tbx = part1.shapes[-1].text_frame
r = tbx.paragraphs[0].add_run(); r.text = "It's Recovering Connectivity."
r.font.size = Pt(22); r.font.bold = True; r.font.color.rgb = ORANGE; r.font.name = FONT
add_text(part1, cx, cy + 0.64, cw, 0.36,
         [('Perceive', {'size': 15, 'bold': True, 'color': CYAN})], size=15)
tbx = part1.shapes[-1].text_frame
r = tbx.paragraphs[0].add_run(); r.text = '  identify parts, text, pins and wires      '; r.font.size = Pt(13); r.font.color.rgb = MUTED; r.font.name = FONT
r = tbx.paragraphs[0].add_run(); r.text = 'Connect'; r.font.size = Pt(15); r.font.bold = True; r.font.color.rgb = NAVY; r.font.name = FONT
r = tbx.paragraphs[0].add_run(); r.text = '  build geometry and electrical topology      '; r.font.size = Pt(13); r.font.color.rgb = MUTED; r.font.name = FONT
r = tbx.paragraphs[0].add_run(); r.text = 'Verify'; r.font.size = Pt(15); r.font.bold = True; r.font.color.rgb = ORANGE; r.font.name = FONT
r = tbx.paragraphs[0].add_run(); r.text = '  repair and calibrate with evidence'; r.font.size = Pt(13); r.font.color.rgb = MUTED; r.font.name = FONT
ch_items = [
    ('Dense component detection', 'small SMD parts, missed / confused classes', 'Dual-path YOLO (whole + SAHI) + WBF fusion', 'per-class thresholds, detection F1 0.883'),
    ('Pin localization & ownership', 'multi-pin ICs, offsets, wrong binding', 'Dual-head U-Net heatmaps + cost assignment', 'peak decode with sub-pixel refinement'),
    ('Text - part association', 'crowded / tilted annotations', 'Multi-view OCR under evidence gates', '>=2 views @ 0.98 confidence, budgets'),
    ('Topology & connectivity', 'T/cross nodes, broken wires, flywires', 'Conservative graph + transactional repair', 'avoid false links before recovery'),
]
row_h = 0.98
for i, (ct, cd, st, sd) in enumerate(ch_items):
    y = cy + 1.12 + i * (row_h + 0.09)
    add_card(part1, cx, y, 5.28, row_h, fill=WHITE)
    add_text(part1, cx + 0.14, y + 0.08, 5.0, 0.36, ct, size=16, bold=True, color=NAVY)
    add_text(part1, cx + 0.14, y + 0.48, 5.0, 0.42, cd, size=13.5, color=MUTED)
    sld_arw = part1.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(cx + 5.44), Inches(y + row_h / 2 - 0.14), Inches(0.5), Inches(0.28))
    sld_arw.fill.solid(); sld_arw.fill.fore_color.rgb = CYAN; sld_arw.line.fill.background()
    add_card(part1, cx + 6.1, y, 5.4, row_h, fill=PANEL)
    add_text(part1, cx + 6.24, y + 0.08, 5.15, 0.36, st, size=16, bold=True, color=INK)
    add_text(part1, cx + 6.24, y + 0.48, 5.15, 0.42, sd, size=13.5, color=MUTED)

""")

# ============ P4 ============
s = block_replace(s, '# ---------- Part II page: four-layer architecture ----------',
                  '# ---- P3 Visual Perception ----', """# ---------- Part II page: four-layer architecture ----------
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
    add_text(part2, cx + 4.05, y + 0.2, 7.3, 0.76, d, size=15, color=INK)
    if i < 3:
        da = part2.shapes.add_shape(MSO_SHAPE.DOWN_ARROW, Inches(cx + 1.55), Inches(y + 1.15), Inches(0.26), Inches(0.13))
        da.fill.solid(); da.fill.fore_color.rgb = CYAN; da.line.fill.background()
add_text(part2, cx, cy + 5.12, cw, 0.32,
         'Full 11-stage detail: Backup A  ·  zero LLM tokens  ·  hash-locked & audited',
         size=12, color=MUTED)

""")

# ============ P5 ============
s = block_replace(s, '# ---- P3 Visual Perception ----', '# ---- P4 Topology Reconstruction ----', """# ---- P3 Visual Perception ----
pP3 = new_body('Visual Perception', 'Three default-on models produce candidates; conclusions come from evidence gates')
p = pP3
cx, cy, cw, ch = content_shapes(p)
panels = [
    (rf'{D}\\p5_input.png', 'Input schematic PNG (official example page)'),
    (rf'{D}\\p5_wire.png', 'Wire probability map (threshold 0.35)'),
    (rf'{D}\\p5_pin.png', 'Pin heatmap → 431 candidates'),
]
for i, (img, cap) in enumerate(panels):
    x = cx + i * 3.94
    place_pic(p, img, x, cy, 3.78, 2.68)
    add_text(p, x, cy + 2.74, 3.78, 0.3, cap, size=12, color=MUTED, align=PP_ALIGN.CENTER)
add_text(p, cx, cy + 3.14, cw, 0.38,
         [('431 pin candidates decoded:  ', {'size': 16, 'bold': True, 'color': INK})], size=16)
tbx = p.shapes[-1].text_frame
r = tbx.paragraphs[0].add_run(); r.text = '287'; r.font.size = Pt(16); r.font.bold = True; r.font.color.rgb = NAVY; r.font.name = FONT
r = tbx.paragraphs[0].add_run(); r.text = ' assigned · '; r.font.size = Pt(14); r.font.color.rgb = MUTED; r.font.name = FONT
r = tbx.paragraphs[0].add_run(); r.text = '115'; r.font.size = Pt(16); r.font.bold = True; r.font.color.rgb = NAVY; r.font.name = FONT
r = tbx.paragraphs[0].add_run(); r.text = ' ambiguous · '; r.font.size = Pt(14); r.font.color.rgb = MUTED; r.font.name = FONT
r = tbx.paragraphs[0].add_run(); r.text = '29'; r.font.size = Pt(16); r.font.bold = True; r.font.color.rgb = NAVY; r.font.name = FONT
r = tbx.paragraphs[0].add_run(); r.text = ' unowned'; r.font.size = Pt(14); r.font.color.rgb = MUTED; r.font.name = FONT
flow = [('Whole-image YOLO', 'precision on large parts'), ('SAHI tiles', 'recall on small parts'), ('WBF fusion', 'score-weighted union')]
for i, (t, d) in enumerate(flow):
    x = cx + i * 3.2
    add_card(p, x, cy + 3.62, 3.0, 0.92, fill=PANEL)
    add_text(p, x + 0.12, cy + 3.68, 2.8, 0.34, t, size=15, bold=True, color=NAVY)
    add_text(p, x + 0.12, cy + 4.04, 2.8, 0.32, d, size=11.5, color=MUTED)
    if i < 2:
        sld_arw = p.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(x + 3.02), Inches(cy + 3.94), Inches(0.34), Inches(0.26))
        sld_arw.fill.solid(); sld_arw.fill.fore_color.rgb = CYAN; sld_arw.line.fill.background()
chip(p, cx + 9.85, cy + 3.62, 1.7, 0.92, 'F1 0.883', 'detection, 59 cases', ORANGE, 19, 10.5)
origins = [
    ('YOLO nano — self-trained (35 classes)', 'WirePinUNet — self-trained (milesial parts)'),
    ('PP-OCRv6 — called open pre-trained (offline)', 'JunctionCNN — self-trained, OFF by default (optional)'),
]
for i, (a, b) in enumerate(origins):
    col, row = i % 2, i // 2
    x = cx + col * 5.9
    y = cy + 4.68 + row * 0.38
    dot = p.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x), Inches(y + 0.06), Inches(0.13), Inches(0.13))
    dot.fill.solid(); dot.fill.fore_color.rgb = ORANGE if row else NAVY; dot.line.fill.background()
    add_text(p, x + 0.24, y, 5.6, 0.3, a, size=12, color=INK)
    add_text(p, x + 0.24, y + 0.0, 0.0, 0.0, '', size=11)
    add_text(p, cx + 6.14, y, 0.0, 0.0, '', size=11)
    add_text(p, x + 0.24, y, 5.6, 0.3, a, size=12, color=INK)

""")

# ============ P6 ============
s = block_replace(s, '# ---- P4 Topology Reconstruction ----', '# ---- P5 Evidence-Gated Repair ----', """# ---- P4 Topology Reconstruction ----
pP4 = new_body('Topology Reconstruction', 'Wire probability map -> skeleton -> segments -> networks')
p = pP4
cx, cy, cw, ch = content_shapes(p)
add_text(p, cx, cy, cw, 0.4, 'Avoid False Connections Before Recovering Missing Links', size=17, bold=True, color=NAVY)
place_pic(p, rf'{A}\\fig3c_topology_real.png', cx, cy + 0.5, 4.01, 2.85)
add_text(p, cx, cy + 3.4, 4.05, 0.3, 'Global context: wire skeleton + 39 junction nodes', size=11.5, color=MUTED, align=PP_ALIGN.CENTER)
place_pic(p, rf'{D}\\p6_zoom_skeleton.png', cx + 4.25, cy + 0.5, 3.32, 2.85)
add_text(p, cx + 4.25, cy + 3.4, 3.35, 0.3, 'Zoom: junction nodes on the skeleton', size=11.5, color=MUTED, align=PP_ALIGN.CENTER)
place_pic(p, rf'{D}\\p6_zoom_networks.png', cx + 7.85, cy + 0.5, 3.32, 2.85)
add_text(p, cx + 7.85, cy + 3.4, 3.35, 0.3, 'Zoom: per-net colored connectivity', size=11.5, color=MUTED, align=PP_ALIGN.CENTER)
cases = [
    ('T / cross nodes', 'conservative split, explicit uncertainty'),
    ('Broken wires', 'recovered only by hard evidence'),
    ('Net labels (flywires)', 'same-name labels merge; zero-length edges'),
    ('Power nets (VCC/GND)', 'four gates before merging'),
]
for i, (t, s2) in enumerate(cases):
    x = cx + i * 2.94
    add_card(p, x, cy + 3.85, 2.78, 0.95, fill=PANEL)
    add_text(p, x + 0.12, cy + 3.9, 2.55, 0.3, t, size=12.5, bold=True, color=NAVY)
    add_text(p, x + 0.12, cy + 4.22, 2.55, 0.54, s2, size=11, color=INK)
add_text(p, cx, cy + 4.95, cw, 0.32,
         '238 Physical-Network Candidates / 362 Edges → 63 Logical Nets - candidates are the graph-stage intermediate; merges give the final output.',
         size=12, color=INK)

""")

# ============ P7 ============
s = block_replace(s, '# ---- P5 Evidence-Gated Repair ----', '# ---- P6 Output Calibration ----', """# ---- P5 Evidence-Gated Repair ----
pP5 = new_body('Repair Without Guessing', 'Page-13 junction reconciliation - merge only on hard evidence')
p = pP5
cx, cy, cw, ch = content_shapes(p)
place_pic(p, rf'{D}\\p7_gap_trim.png', cx, cy + 0.1, 3.6, 3.6)
add_text(p, cx, cy + 3.78, 3.65, 0.3, 'Real crop from page 13 (development record)', size=11.5, color=MUTED, align=PP_ALIGN.CENTER)
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
    'Merge evidence: ≥60 px collinear overlap (±3°, ≤2 px), or red dot + ≥95% ink support',
    'Members and edges only - components and pins untouched; ≤8 merges per page',
]
for i, t in enumerate(mech):
    add_text(p, cx + 3.95, cy + 2.85 + i * 0.4, 8.55, 0.36, t, size=12.5, color=INK)
add_text(p, cx + 3.95, cy + 3.72, 8.55, 0.3,
         'Other 12 pages byte-identical (frozen cache replay) - single-page development-set ablation, not an official overall metric.',
         size=12, italic=True, color=MUTED)
add_text(p, cx + 3.95, cy + 4.15, 8.55, 0.34,
         [('The wire-F1 gain comes from junction reconciliation - not from the OCR confidence gate.', {'size': 13, 'bold': True, 'color': NAVY})], size=13)
add_text(p, cx + 3.95, cy + 4.6, 8.55, 0.34,
         [('Evidence-gated OCR (≥2 views @ 0.98) is a separate capability - see Backup E.', {'size': 12.5, 'color': MUTED})], size=12.5)

""")

# ============ P8 ============
s = block_replace(s, '# ---- P6 Output Calibration ----', '# ---------- Part III page: official results ----------', """# ---- P6 Output Calibration ----
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
    y = cy + i * 1.3
    add_card(p, cx, y, 6.6, 1.15, fill=WHITE)
    add_text(p, cx + 0.18, y + 0.08, 6.2, 0.36, t, size=15, bold=True, color=INK)
    add_text(p, cx + 0.18, y + 0.5, 5.9, 0.6, d, size=12.5, color=MUTED)
    dot = p.shapes.add_shape(MSO_SHAPE.OVAL, Inches(cx + 6.24), Inches(y + 0.5), Inches(0.16), Inches(0.16))
    dot.fill.solid(); dot.fill.fore_color.rgb = c; dot.line.fill.background()
add_text(p, cx, cy + 3.98, 6.7, 0.6,
         'Both bbox scopes are independent experiments - never multiplied.\\nFull metric table: Backup D.',
         size=12.5, color=MUTED)
place_pic(p, rf'{D}\\p8_u3_bbox_compare.png', cx + 7.0, cy, 1.5, 3.5)
add_text(p, cx + 7.0, cy + 3.56, 1.6, 0.3, 'U3 region zoom', size=11.5, color=MUTED, align=PP_ALIGN.CENTER)
add_text(p, cx + 8.75, cy + 0.35, 3.75, 0.62, '1~2 px', size=32, bold=True, color=ORANGE)
add_text(p, cx + 8.75, cy + 1.05, 3.8, 0.66,
         'U3 (RP2350A) bbox vs official example annotation (official 20 px tolerance)', size=12.5, color=INK)
add_text(p, cx + 8.75, cy + 1.78, 3.8, 0.6,
         'ours [539,281,689,731]\\nofficial [540,280,690,730]', size=12, color=MUTED, font='Consolas')
add_text(p, cx + 7.0, cy + 3.95, 5.55, 0.3, 'Output JSON (real fragments)', size=12.5, bold=True, color=INK)
json_rows = [
    ('components', '"U3": box RP2350A · bbox [539,281,689,731]', RGBColor(0x0E, 0x7C, 0x8C)),
    ('pins', '"U3".pin_51: USB_DM @ (513, 561)', NAVY),
    ('nets', '"LOG_NET": (C1.2, U3.U3) · 3 edges', ORANGE),
]
for i, (key, txt, c) in enumerate(json_rows):
    y = cy + 4.32 + i * 0.55
    kc = p.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(cx + 7.0), Inches(y), Inches(1.5), Inches(0.4))
    kc.fill.solid(); kc.fill.fore_color.rgb = c; kc.line.fill.background()
    try:
        kc.shadow.inherit = False
    except Exception:
        pass
    ktf = kc.text_frame; ktf.word_wrap = True
    kpa = ktf.paragraphs[0]; kpa.alignment = PP_ALIGN.CENTER
    kr = kpa.add_run(); kr.text = key; kr.font.size = Pt(11.5); kr.font.bold = True; kr.font.color.rgb = WHITE; kr.font.name = FONT
    add_text(p, cx + 8.65, y + 0.04, 3.9, 0.32, txt, size=12, color=INK, font='Consolas')

""")

# ============ P9: drop pending-screenshots line ============
old_p9_line = """add_text(part3, cx, cy + 3.95, cw, 0.4,
         'Official score / rank / finalist screenshots: pending insertion before the defense (recorded as missing; no mock screenshots used).',
         size=12, italic=True, color=MUTED)
"""
assert old_p9_line in s
s = s.replace(old_p9_line, "", 1)

# ============ P10: subtitle wording ============
s = s.replace("'Four scoring dimensions, one card each - development-set experiments and single-case ablations'",
              "'Four experimental indicators - development-set experiments and single-case ablations'", 1)

# ============ P11: engineering line removed (Backup C has it) ============
old_p11_line = """add_text(p, cx + 5.95, cy + 4.54, 6.55, 0.32,
         'Engineering details (Docker · SHA256 · audit · 626+ tests · runtime): Backup C.',
         size=12.5, color=MUTED)"""
assert old_p11_line in s
s = s.replace(old_p11_line, "", 1)

# ============ P12: no limitations; engineering value ============
s = block_replace(s, '# ---------- Part IV page: conclusions ----------', '# ---------- Backups B1-B5 ----------', """# ---------- Part IV page: conclusions ----------
set_title_subtitle(part4, None, 'Conclusions and future work')
cx, cy, cw, ch = content_shapes(part4)
add_text(part4, cx, cy + 0.02, cw, 0.42, 'Three Technical Contributions', size=19, bold=True, color=NAVY)
contribs = [
    ('Multi-model visual perception', 'self-trained YOLO nano / dual-head WirePinUNet / junction CNN - all sha256-traceable'),
    ('Evidence-gated topology repair', '0.98 x 2-view OCR gates + transactional engine + rollback'),
    ('Score-aware output calibration', 'bbox conventions / Name==key / pin-library mapping'),
]
for i, (t, d) in enumerate(contribs):
    y = cy + 0.52 + i * 0.56
    dot = part4.shapes.add_shape(MSO_SHAPE.OVAL, Inches(cx), Inches(y + 0.03), Inches(0.28), Inches(0.28))
    dot.fill.solid(); dot.fill.fore_color.rgb = ORANGE; dot.line.fill.background()
    add_text(part4, cx, y + 0.04, 0.28, 0.28, str(i + 1), size=11, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    add_text(part4, cx + 0.4, y - 0.02, 11.1, 0.3, t, size=14.5, bold=True, color=INK)
    add_text(part4, cx + 0.4, y + 0.27, 11.1, 0.3, d, size=11, color=MUTED)
add_text(part4, cx, cy + 2.22, cw, 0.3,
         'JunctionCNN: self-trained but OFF in the default pipeline - an optional enhancement, not the default inference backbone.',
         size=12.5, italic=True, color=MUTED)
add_text(part4, cx, cy + 2.6, cw, 0.42, 'Engineering Value', size=19, bold=True, color=NAVY)
ev = [
    ('Docker delivery on the official base image', 'SHA256 integrity: 249-file manifest + weight fingerprints'),
    ('Per-stage audit logs (20+ JSONs per image)', 'validate/sanitize + rollback · 626+ unit tests · zero LLM tokens'),
]
for i, (a, b) in enumerate(ev):
    col, row = i % 2, i // 2
    x = cx + col * 5.95
    y = cy + 3.06 + row * 0.42
    dot = part4.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x), Inches(y + 0.06), Inches(0.12), Inches(0.12))
    dot.fill.solid(); dot.fill.fore_color.rgb = NAVY; dot.line.fill.background()
    add_text(part4, x + 0.24, y, 5.6, 0.32, a, size=12.5, color=INK)
    add_text(part4, x + 0.24, y + 0.3, 5.6, 0.0, '', size=11)
    add_text(part4, x + 0.24, y + 0.3, 5.6, 0.32, b, size=11, color=MUTED)
add_text(part4, cx, cy + 4.02, cw, 0.42, 'Future Development', size=19, bold=True, color=NAVY)
outs = ['Expand hard-case datasets and domain adaptation', 'Improve junction classification (promote the trained CNN)',
        'Enhance cross-domain generalization for engineering drawings']
for i, t in enumerate(outs):
    y = cy + 4.48 + i * 0.34
    dot = part4.shapes.add_shape(MSO_SHAPE.OVAL, Inches(cx + 0.06), Inches(y + 0.07), Inches(0.12), Inches(0.12))
    dot.fill.solid(); dot.fill.fore_color.rgb = CYAN; dot.line.fill.background()
    add_text(part4, cx + 0.3, y, 11.2, 0.3, t, size=12.5, color=INK)

""")

# ============ B1: shorter title + bigger image ============
s = s.replace("b1 = new_body('Backup A · Detailed System Architecture (11 stages)',\n             'Compressed to four layers on the main deck (Part II) - zero LLM tokens · hash-locked · audited')",
              "b1 = new_body('Backup A · Detailed Architecture',\n             'The full 11-stage pipeline - compressed to four layers on the main deck (Part II)')", 1)
s = s.replace("place_pic(p, rf'{D}\\arch_pipeline_en.png', cx, cy, cw, ch - 0.1)",
              "place_pic(p, rf'{D}\\arch_pipeline_en.png', cx - 0.1, cy - 0.05, cw + 0.2, ch + 0.1)", 1)

# ============ B3: shorter title + new recovery image ============
s = s.replace("b3 = new_body('Backup C · Runtime Workflow and JSON Output',\n             'docker run on the evaluation side; direct Python run for development')",
              "b3 = new_body('Backup C · Runtime & Reliability',\n             'docker run on the evaluation side; direct Python run for development')", 1)
s = s.replace("place_pic(p, rf'{D}\\d_recovery_flow_en.png', cx + 6.2, cy + 0.1, 5.35, 2.5)",
              "place_pic(p, rf'{D}\\d_recovery_flow_en.png', cx + 6.1, cy + 0.1, 5.6, 2.6)", 1)

# ============ B4: shorter title + notes on top + 14pt table ============
s = s.replace("b4 = new_body('Backup D · Experimental Metrics and Ablation (excerpt)',\n             'Every number traceable; full 17-row table in defense_supplement.md §3')",
              "b4 = new_body('Backup D · Metrics & Ablation',\n             'Every number traceable; full 17-row table in defense_supplement.md §3')", 1)
old_b4_head = """cx, cy, cw, ch = content_shapes(p)
rows = [
    ('Experiment', 'Scope', 'Metric', 'Result'),"""
new_b4_head = """cx, cy, cw, ch = content_shapes(p)
add_text(p, cx, cy + 0.0, cw, 0.3,
         'Caliber caution: canonical (40 pages) and page scaling (13 pages) are independent experiments; "+0.31" is the total-score delta (connectivity +0.7pp).',
         size=12, color=MUTED)
add_text(p, cx, cy + 0.32, cw, 0.3,
         'Local development replay: 98.93 (13-page public set, v101-score) - development only, not comparable with the official 74.',
         size=12, color=INK)
rows = [
    ('Experiment', 'Scope', 'Metric', 'Result'),"""
assert old_b4_head in s
s = s.replace(old_b4_head, new_b4_head, 1)
old_b4_tbl = """tbl_shape = p.shapes.add_table(len(rows), 4, Inches(cx), Inches(cy + 0.05), Inches(11.55), Inches(4.4))
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
         size=12, color=INK)"""
new_b4_tbl = """tbl_shape = p.shapes.add_table(len(rows), 4, Inches(cx), Inches(cy + 0.72), Inches(11.55), Inches(4.5))
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
        cell.fill.solid(); cell.fill.fore_color.rgb = NAVY if ri == 0 else (WHITE if ri % 2 else RGBColor(0xF6, 0xF9, 0xFC))"""
assert old_b4_tbl in s
s = s.replace(old_b4_tbl, new_b4_tbl, 1)

# ============ B5: drop 2-hour card, add aux OCR case ============
old_q = "    ('Within the 2-hour budget?', 'GPU 3-worker design; wall_seconds recorded per case; official machine timing not yet measured')"
new_q = "    ('How is a part number recovered?', 'SW1/SW2: wrong fallback \"SW1\" → real part \"PTS8153JK250SMTRLFS\" via nearby text stack, rotated views, conf 0.998/0.992')"
assert old_q in s
s = s.replace(old_q, new_q, 1)

open(p, 'w', encoding='utf-8').write(s)
print('onsite-final patch applied OK')
