"""Story-polish round 2 fixes."""
import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
p = r'E:\AI_Workplace\精英挑战赛\_audit_workspace\scripts\build_official_deck.py'
s = open(p, encoding='utf-8').read()
ok = []

def rep(old, new, tag):
    global s
    assert old in s, 'NOT FOUND: ' + tag
    s = s.replace(old, new, 1)
    ok.append(tag)

# 1. P3 rows compress + note inside rect
rep("    y = cy + 1.52 + i * (row_h + 0.08)", "    y = cy + 1.46 + i * (row_h + 0.06)", 'P3 rows')
rep("row_h = 0.88\nfor i, (ct, cd, st, sd) in enumerate(ch_items):", "row_h = 0.84\nfor i, (ct, cd, st, sd) in enumerate(ch_items):", 'P3 rowh')
rep("""add_text(part1, cx, cy + 1.52 + 4 * (row_h + 0.08) - 0.02, cw, 0.3,""",
    """add_text(part1, cx, cy + 5.06, cw, 0.3,""", 'P3 note y')

# 2. P6 note single line
rep("""         '238 Physical-Network Candidates / 362 Edges → 63 Logical Nets - the candidates are the graph-stage intermediate; flywire and power merges give the final output.',
         size=12, color=INK)""",
    """         '238 Physical-Network Candidates / 362 Edges → 63 Logical Nets - candidates are the graph-stage intermediate; merges give the final output.',
         size=11.5, color=INK)""", 'P6 note')

# 3. P7 numbers row single line 14pt
rep("""add_text(p, cx, cy + 4.02, 7.5, 0.4,
         [('Page-13 wire repair:  32/28/17 → 46/13/3    F1 58.72% → ', {'size': 15.5, 'bold': True, 'color': INK})], size=15.5)""",
    """add_text(p, cx, cy + 4.02, 7.5, 0.36,
         [('Page-13 repair: 32/28/17 → 46/13/3 · F1 58.72% → ', {'size': 14, 'bold': True, 'color': INK})], size=14)""", 'P7 row')
rep("r = tbx.paragraphs[0].add_run(); r.text = '85.19%'; r.font.size = Pt(15.5); r.font.bold = True; r.font.color.rgb = ORANGE; r.font.name = FONT\nr = tbx.paragraphs[0].add_run(); r.text = '    score 82.871 → 92.656'; r.font.size = Pt(15.5); r.font.bold = True; r.font.color.rgb = INK; r.font.name = FONT",
    "r = tbx.paragraphs[0].add_run(); r.text = '85.19%'; r.font.size = Pt(14); r.font.bold = True; r.font.color.rgb = ORANGE; r.font.name = FONT\nr = tbx.paragraphs[0].add_run(); r.text = ' · score 82.871→92.656'; r.font.size = Pt(14); r.font.bold = True; r.font.color.rgb = INK; r.font.name = FONT", 'P7 runs')

# 4. P8 desc width
rep("    add_text(p, cx + 0.18, y + 0.5, 6.25, 0.62, d, size=12.5, color=MUTED)",
    "    add_text(p, cx + 0.18, y + 0.5, 5.65, 0.62, d, size=12.5, color=MUTED)", 'P8 desc')

# 5. P9 subtitle
rep("""# ---------- Part III page: official results ----------
cx, cy, cw, ch = content_shapes(part3)""",
    """# ---------- Part III page: official results ----------
set_title_subtitle(part3, None, 'Official evaluation results & quantitative evidence')
cx, cy, cw, ch = content_shapes(part3)""", 'P9 subtitle')

# 6. P11 relayout (native JSON block)
old_p11 = s[s.find('p = pP9\n'):s.find('# ---------- Part IV page')]
new_p11 = """p = pP9
cx, cy, cw, ch = content_shapes(p)
place_pic(p, rf'{A}\\fig1b_detection_clean.png', cx, cy, 4.09, 2.9)
add_text(p, cx, cy + 2.94, 4.15, 0.28, 'Detection & recognition overlay (32 drawn-symbol parts)', size=11, color=MUTED, align=PP_ALIGN.CENTER)
place_pic(p, rf'{A}\\fig3b_network_overlay.png', cx + 4.35, cy, 4.09, 2.9)
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
    y = cy + 3.66 + i * 0.68
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
add_text(p, cx + 5.95, cy + 4.58, 6.55, 0.36,
         'Engineering details (Docker · SHA256 · audit logs · 626+ tests · runtime scope): Backup C.',
         size=13, color=MUTED)

"""
s = s.replace(old_p11, new_p11, 1)
ok.append('P11 relayout')

# 7. P12 subtitle + spacing
rep("""# ---------- Part IV page: conclusions ----------
cx, cy, cw, ch = content_shapes(part4)""",
    """# ---------- Part IV page: conclusions ----------
set_title_subtitle(part4, None, 'Conclusions and future work')
cx, cy, cw, ch = content_shapes(part4)""", 'P12 subtitle')
rep("add_text(part4, cx, cy + 2.43, cw, 0.42, 'Current limitations', size=19, bold=True, color=NAVY)",
    "add_text(part4, cx, cy + 2.78, cw, 0.42, 'Current limitations', size=19, bold=True, color=NAVY)", 'P12 lim header')
rep("    y = cy + 2.88 + i * 0.34", "    y = cy + 3.24 + i * 0.36", 'P12 lim rows')
rep("add_text(part4, cx, cy + 3.98, cw, 0.42, 'Future work', size=19, bold=True, color=NAVY)",
    "add_text(part4, cx, cy + 4.42, cw, 0.42, 'Future work', size=19, bold=True, color=NAVY)", 'P12 fut header')
rep("    y = cy + 4.43 + i * 0.34", "    y = cy + 4.88 + i * 0.36", 'P12 fut rows')

open(p, 'w', encoding='utf-8').write(s)
print('round2 fixes applied:', ok)
