# -*- coding: utf-8 -*-
"""Third-round micro-fixes: P3 narrative width, P5 stats block, P8 two lines, P11 asset."""
import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
P = r'E:\AI_Workplace\精英挑战赛\_audit_workspace\vision-fetch\docs\eda_final_defense\slides\build_official_deck.py'
s = open(P, encoding='utf-8').read()

FIXES = []
def rep(old, new, tag):
    FIXES.append((old, new, tag))

# ---- P3: shorter narrative runs, 3-space gaps ----
rep("""r = tbx.paragraphs[0].add_run(); r.text = '  identify parts, text, pins, wires    '; r.font.size = Pt(14); r.font.color.rgb = MUTED; r.font.name = FONT
r = tbx.paragraphs[0].add_run(); r.text = 'Connect'; r.font.size = Pt(15); r.font.bold = True; r.font.color.rgb = NAVY; r.font.name = FONT
r = tbx.paragraphs[0].add_run(); r.text = '  build geometry and electrical topology    '; r.font.size = Pt(14); r.font.color.rgb = MUTED; r.font.name = FONT
r = tbx.paragraphs[0].add_run(); r.text = 'Verify'; r.font.size = Pt(15); r.font.bold = True; r.font.color.rgb = ORANGE; r.font.name = FONT
r = tbx.paragraphs[0].add_run(); r.text = '  repair and calibrate with evidence'; r.font.size = Pt(14); r.font.color.rgb = MUTED; r.font.name = FONT""",
"""r = tbx.paragraphs[0].add_run(); r.text = '  identify parts, text, pins, wires   '; r.font.size = Pt(14); r.font.color.rgb = MUTED; r.font.name = FONT
r = tbx.paragraphs[0].add_run(); r.text = 'Connect'; r.font.size = Pt(15); r.font.bold = True; r.font.color.rgb = NAVY; r.font.name = FONT
r = tbx.paragraphs[0].add_run(); r.text = '  build geometry + electrical topology   '; r.font.size = Pt(14); r.font.color.rgb = MUTED; r.font.name = FONT
r = tbx.paragraphs[0].add_run(); r.text = 'Verify'; r.font.size = Pt(15); r.font.bold = True; r.font.color.rgb = ORANGE; r.font.name = FONT
r = tbx.paragraphs[0].add_run(); r.text = '  repair + calibrate with evidence'; r.font.size = Pt(14); r.font.color.rgb = MUTED; r.font.name = FONT""",
    'p3-narrative')

# ---- P5: stats as two clean lines ----
rep("""add_text(p, cx + 2.80, cy + 4.24, 6.60, 0.34,
         [('431', {'size': 17, 'bold': True, 'color': INK})], size=17)
tbx = p.shapes[-1].text_frame
r = tbx.paragraphs[0].add_run(); r.text = ' pin candidates decoded:   '; r.font.size = Pt(14); r.font.color.rgb = MUTED; r.font.name = FONT
r = tbx.paragraphs[0].add_run(); r.text = '287'; r.font.size = Pt(16); r.font.bold = True; r.font.color.rgb = NAVY; r.font.name = FONT
r = tbx.paragraphs[0].add_run(); r.text = ' assigned · '; r.font.size = Pt(14); r.font.color.rgb = MUTED; r.font.name = FONT
r = tbx.paragraphs[0].add_run(); r.text = '115'; r.font.size = Pt(16); r.font.bold = True; r.font.color.rgb = NAVY; r.font.name = FONT
r = tbx.paragraphs[0].add_run(); r.text = ' ambiguous · '; r.font.size = Pt(14); r.font.color.rgb = MUTED; r.font.name = FONT
r = tbx.paragraphs[0].add_run(); r.text = '29'; r.font.size = Pt(16); r.font.bold = True; r.font.color.rgb = NAVY; r.font.name = FONT
r = tbx.paragraphs[0].add_run(); r.text = ' unowned'; r.font.size = Pt(14); r.font.color.rgb = MUTED; r.font.name = FONT
add_text(p, cx + 2.80, cy + 4.62, 6.60, 0.28,
         'pin ownership assignment follows in Geometric Processing', size=13.5, color=MUTED)""",
"""add_text(p, cx + 2.80, cy + 4.22, 6.60, 0.32,
         '431 pin candidates decoded:', size=16, bold=True, color=INK)
add_text(p, cx + 2.80, cy + 4.58, 6.60, 0.34,
         [('287', {'size': 17, 'bold': True, 'color': NAVY})], size=17)
tbx = p.shapes[-1].text_frame
r = tbx.paragraphs[0].add_run(); r.text = ' assigned  ·  '; r.font.size = Pt(14); r.font.color.rgb = MUTED; r.font.name = FONT
r = tbx.paragraphs[0].add_run(); r.text = '115'; r.font.size = Pt(17); r.font.bold = True; r.font.color.rgb = NAVY; r.font.name = FONT
r = tbx.paragraphs[0].add_run(); r.text = ' ambiguous  ·  '; r.font.size = Pt(14); r.font.color.rgb = MUTED; r.font.name = FONT
r = tbx.paragraphs[0].add_run(); r.text = '29'; r.font.size = Pt(17); r.font.bold = True; r.font.color.rgb = NAVY; r.font.name = FONT
r = tbx.paragraphs[0].add_run(); r.text = ' unowned'; r.font.size = Pt(14); r.font.color.rgb = MUTED; r.font.name = FONT""",
    'p5-stats2')

# ---- P8: guard + left note one line ----
rep("'Fires only when the 5-vertex flag and a single terminal agree (14 unit tests).',",
    "'Fires only when the 5-vertex flag and one terminal agree (14 unit tests).',", 'p8-guard2')
rep("'Both bbox scopes independent - never multiplied (Backup D).', size=14, color=MUTED)",
    "'Both bbox scopes independent - see Backup D.', size=14, color=MUTED)", 'p8-note2')

ok = True
for old, new, tag in FIXES:
    n = s.count(old)
    if n != 1:
        print(f'ASSERT FAIL [{tag}] count={n}')
        ok = False
if not ok:
    print('NO FILE WRITTEN')
    sys.exit(1)
for old, new, tag in FIXES:
    s = s.replace(old, new)
open(P, 'w', encoding='utf-8').write(s)
print('patch3 applied OK')
