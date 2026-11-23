# -*- coding: utf-8 -*-
"""Second-round fixes after first visual pass: overflow trims + label alignment.
Idempotent-guarded single replaces; aborts without writing if any assert fails."""
import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
P = r'E:\AI_Workplace\精英挑战赛\_audit_workspace\vision-fetch\docs\eda_final_defense\slides\build_official_deck.py'
s = open(P, encoding='utf-8').read()

FIXES = []

def rep(old, new, tag):
    FIXES.append((old, new, tag))

# ---- P3: narrative line spacing + JSON mono trim ----
rep("r.text = '  identify parts, text, pins and wires      '", "r.text = '  identify parts, text, pins, wires    '", 'p3-narr1')
rep("r.text = '  build geometry and electrical topology      '", "r.text = '  build geometry and electrical topology    '", 'p3-narr2')
rep('''add_text(part1, cx + 6.22, sy + 0.36, 5.15, 0.26,
         '"components": {"U3": {"bbox": [539,281,689,731]}, ...}', size=13.5, color=INK, font='Consolas')
add_text(part1, cx + 6.22, sy + 0.62, 5.15, 0.26,
         '"nets": {"LOG_NET": ["C1.2", "U3.U3"]}', size=13.5, color=INK, font='Consolas')''',
     '''add_text(part1, cx + 6.22, sy + 0.36, 5.15, 0.26,
         '"U3": {"bbox": [539,281,689,731]}', size=13.5, color=INK, font='Consolas')
add_text(part1, cx + 6.22, sy + 0.62, 5.15, 0.26,
         '"LOG_NET": ["C1.2", "U3.U3"]', size=13.5, color=INK, font='Consolas')''', 'p3-json')

# ---- P5: legend spacing + single-line stats ----
rep("add_text(p, cx, cy + 2.66, 2.35, 0.30, 'Model origins', size=15, bold=True, color=INK)",
    "add_text(p, cx, cy + 2.60, 2.35, 0.30, 'Model origins', size=15, bold=True, color=INK)", 'p5-origins')
rep("    y = cy + 3.04 + i * 0.64\n    dot = p.shapes.add_shape(MSO_SHAPE.OVAL, Inches(cx + 0.02), Inches(y + 0.05), Inches(0.13), Inches(0.13))",
    "    y = cy + 2.98 + i * 0.60\n    dot = p.shapes.add_shape(MSO_SHAPE.OVAL, Inches(cx + 0.02), Inches(y + 0.05), Inches(0.13), Inches(0.13))", 'p5-orig-row')
rep("('WirePinUNet', 'self-trained (milesial)', NAVY)", "('WirePinUNet', 'self-trained (milesial)', NAVY)", 'noop1')
rep("""add_text(p, cx + 2.80, cy + 4.28, 6.55, 0.36,
         [('431 pin candidates decoded', {'size': 18, 'bold': True, 'color': INK})], size=18)
tbx = p.shapes[-1].text_frame
r = tbx.paragraphs[0].add_run(); r.text = '    287'; r.font.size = Pt(17); r.font.bold = True; r.font.color.rgb = NAVY; r.font.name = FONT
r = tbx.paragraphs[0].add_run(); r.text = ' assigned · '; r.font.size = Pt(14); r.font.color.rgb = MUTED; r.font.name = FONT
r = tbx.paragraphs[0].add_run(); r.text = '115'; r.font.size = Pt(17); r.font.bold = True; r.font.color.rgb = NAVY; r.font.name = FONT
r = tbx.paragraphs[0].add_run(); r.text = ' ambiguous · '; r.font.size = Pt(14); r.font.color.rgb = MUTED; r.font.name = FONT
r = tbx.paragraphs[0].add_run(); r.text = '29'; r.font.size = Pt(17); r.font.bold = True; r.font.color.rgb = NAVY; r.font.name = FONT
r = tbx.paragraphs[0].add_run(); r.text = ' unowned'; r.font.size = Pt(14); r.font.color.rgb = MUTED; r.font.name = FONT
add_text(p, cx + 2.80, cy + 4.66, 6.55, 0.30,
         'pin ownership assignment follows in Geometric Processing (part 2)', size=13.5, color=MUTED)""",
     """add_text(p, cx + 2.80, cy + 4.24, 6.60, 0.34,
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
         'pin ownership assignment follows in Geometric Processing', size=13.5, color=MUTED)""", 'p5-stats')

# ---- P6: caption one line + image shift ----
rep("place_pic(p, rf'{D}\\ti_p6_main.png', cx, cy + 0.04, 6.20, 4.95)",
    "place_pic(p, rf'{D}\\ti_p6_main.png', cx, cy + 0.02, 6.16, 4.91)", 'p6-img')
rep("""add_text(p, cx, cy + 5.02, 6.20, 0.24,
         'Local prediction overlay, official example page - markers 1/2/3 = steps at right',
         size=14, color=MUTED, align=PP_ALIGN.CENTER)""",
    """add_text(p, cx, cy + 4.98, 6.16, 0.26,
         'Local prediction overlay - markers 1/2/3 = steps at right',
         size=14, color=MUTED, align=PP_ALIGN.CENTER)""", 'p6-cap')

# ---- P7: width + caption ----
rep("add_text(p, cx, cy + 3.78, 3.65, 0.30, 'Real crop from page 13 (development record)', size=14, color=MUTED, align=PP_ALIGN.CENTER)",
    "add_text(p, cx, cy + 3.78, 3.65, 0.30, 'Real crop from page 13 (dev record)', size=14, color=MUTED, align=PP_ALIGN.CENTER)", 'p7-cap')
for yold in ['cy + 2.85 + i * 0.40, 8.55, 0.36', 'cy + 3.72, 8.55, 0.52', 'cy + 4.30, 8.55, 0.34', 'cy + 4.72, 8.55, 0.34']:
    rep(yold, yold.replace('8.55', '7.70'), f'p7-w({yold[:12]})')

# ---- P8: labels, guard, JSON card geometry, left note ----
rep("add_text(p, rx + 0.28, cy + 1.86, 1.55, 0.28, 'Wired terminal', size=14, color=INK, align=PP_ALIGN.CENTER)",
    "add_text(p, rx - 0.36, cy + 1.86, 1.40, 0.28, 'Wired terminal', size=14, color=INK, align=PP_ALIGN.CENTER)", 'p8-lab1')
rep("add_text(p, rx + 3.30, cy + 1.86, 1.20, 0.28, 'Arrow tip', size=14, color=INK, align=PP_ALIGN.CENTER)",
    "add_text(p, rx + 2.33, cy + 1.86, 1.40, 0.28, 'Arrow tip', size=14, color=INK, align=PP_ALIGN.CENTER)", 'p8-lab2')
rep("""add_text(p, rx, cy + 2.82, 5.90, 0.30,
         'Fires only when the 5-vertex convex flag and a single terminal agree - else fields kept (14 unit tests).',
         size=14, color=MUTED)""",
    """add_text(p, rx, cy + 2.82, 5.90, 0.30,
         'Fires only when the 5-vertex flag and a single terminal agree (14 unit tests).',
         size=14, color=MUTED)""", 'p8-guard')
rep("""add_card(p, rx, cy + 3.24, 5.95, 2.30, fill=PANEL)
add_text(p, rx + 0.20, cy + 3.34, 5.5, 0.32, 'Port fields in Target JSON', size=16, bold=True, color=NAVY)
add_text(p, rx + 0.20, cy + 3.70, 5.5, 0.34, '"type": "net_output"    (legacy: net_bidirection)', size=17, color=INK, font='Consolas')
add_text(p, rx + 0.20, cy + 4.10, 5.5, 0.34, '"Name": "VOUT"', size=17, color=INK, font='Consolas')
add_text(p, rx + 0.20, cy + 4.50, 5.5, 0.34, '"value": null', size=17, color=INK, font='Consolas')
add_text(p, rx + 0.20, cy + 4.92, 5.55, 0.56,
         'Written only on proven geometry - export_semantics.py, default profile observed; legacy via PCB_EXPORT_SEMANTICS=legacy.',
         size=14, color=MUTED)""",
    """add_card(p, rx, cy + 3.20, 5.95, 2.10, fill=PANEL)
add_text(p, rx + 0.20, cy + 3.30, 5.5, 0.32, 'Port fields in Target JSON', size=16, bold=True, color=NAVY)
add_text(p, rx + 0.20, cy + 3.66, 5.5, 0.32, '"type": "net_output"', size=17, color=INK, font='Consolas')
add_text(p, rx + 0.20, cy + 4.02, 5.5, 0.32, '"Name": "VOUT"', size=17, color=INK, font='Consolas')
add_text(p, rx + 0.20, cy + 4.38, 5.5, 0.32, '"value": null', size=17, color=INK, font='Consolas')
add_text(p, rx + 0.20, cy + 4.74, 5.55, 0.50,
         'Legacy net_bidirection -> observed geometry. Old labels: PCB_EXPORT_SEMANTICS=legacy.',
         size=14, color=MUTED)""", 'p8-json')
rep("""add_text(p, cx, uy + 1.10, 5.25, 0.28,
         'Both bbox scopes are independent - never multiplied (Backup D).', size=14, color=MUTED)""",
    """add_text(p, cx, uy + 1.10, 5.25, 0.28,
         'Both bbox scopes independent - never multiplied (Backup D).', size=14, color=MUTED)""", 'p8-note')

# ---- P10: bottom line positions ----
rep("""add_text(p, cx, cy + 4.70, cw, 0.34,
         'Development-set experiments and single-case ablations - not official per-dimension scores. Full table: Backup D.',
         size=14, bold=True, color=INK)
add_text(p, cx, cy + 5.10, cw, 0.30,
         'Ablation protocol: one module toggled on a frozen model/OCR cache; the other 12 pages stay byte-identical.',
         size=13.5, color=MUTED)""",
    """add_text(p, cx, cy + 4.64, cw, 0.32,
         'Development-set experiments and single-case ablations - not official per-dimension scores. Full table: Backup D.',
         size=14, bold=True, color=INK)
add_text(p, cx, cy + 5.00, cw, 0.28,
         'Ablation protocol: one module toggled on a frozen model/OCR cache; the other 12 pages stay byte-identical.',
         size=13.5, color=MUTED)""", 'p10-bottom')

# ---- P12: engineering band single line ----
rep("""add_text(part4, cx + 0.2, cy + 4.90, cw - 0.4, 0.32,
         'Engineering delivery: Docker pcb-parser:v102 · SHA256 manifest (249 files) · audit logs per stage · 626+ unit tests · zero LLM tokens',
         size=14.5, bold=True, color=NAVY, align=PP_ALIGN.CENTER)""",
    """add_text(part4, cx + 0.2, cy + 4.90, cw - 0.4, 0.32,
         'Engineering delivery: Docker pcb-parser:v102 · SHA256 manifest · per-stage audit logs · 626+ unit tests · zero LLM tokens',
         size=14, bold=True, color=NAVY, align=PP_ALIGN.CENTER)""", 'p12-band')

# ---- B2: shorter cells + footnote positions ----
rep("('YOLO nano detector', 'Ultralytics YOLO nano · 35 classes', 'detection_balanced_v2: 238 train pages + 180 crops · 59 val pages', 'imgsz 1536 · 120 ep · batch 4', 'self-trained (HR-DONK warm start, MIT)'),",
    "('YOLO nano detector', 'Ultralytics YOLO nano · 35 classes', '238 train pages + 180 crops · 59 val pages', 'imgsz 1536 · 120 ep · batch 4', 'self-trained (HR-DONK warm start)'),", 'b2-r1')
rep("('WirePinUNet dual-head', 'base-32 shared decoder (wire+pin)', 'rasterized source annotations + reviewed masks', 'tiles 512/384 · 40 ep · pos_weight 8/20', 'self-trained (milesial parts, GPL-3)'),",
    "('WirePinUNet dual-head', 'base-32 shared decoder (wire+pin)', 'rasterized source annotations + reviewed masks', 'tiles 512/384 · 40 ep · pos_weight 8/20', 'self-trained (milesial, GPL-3)'),", 'b2-r2')
rep("""add_text(p, cx, cy + 3.48, cw, 0.34,
         'Detector split is original-page level: crop parent pages come from the train split only; 0001/0002 stay fixed in validation (2462 val boxes).',
         size=14, color=INK)
add_text(p, cx, cy + 3.88, cw, 0.34,
         'Data assets: 1000-part EasyEDA symbol library + 337-part slice + STM32F103C8T6 LQFP48 pin library - bundled, zero network access.',
         size=14, color=INK)
add_text(p, cx, cy + 4.28, cw, 0.60,
         'Provenance: baseline_provenance.json (HR-DONK initial baseline, MIT) · milesial U-Net parts GPL-3 (per-file sha256) · no evaluation annotation leakage in training data.',
         size=14, color=MUTED)""",
    """add_text(p, cx, cy + 3.62, cw, 0.34,
         'Detector split is original-page level: crop parents from the train split only; 0001/0002 fixed in validation (2462 val boxes).',
         size=14, color=INK)
add_text(p, cx, cy + 4.02, cw, 0.34,
         'Data assets: 1000-part EasyEDA symbol library + 337-part slice + STM32F103C8T6 LQFP48 pin library - bundled, zero network access.',
         size=14, color=INK)
add_text(p, cx, cy + 4.42, cw, 0.60,
         'Provenance: baseline_provenance.json (HR-DONK, MIT) · milesial U-Net parts GPL-3 (per-file sha256) · no evaluation-annotation leakage in training.',
         size=14, color=MUTED)""", 'b2-notes')

# ---- B3: footnote positions ----
rep("add_text(p, cx, cy + 3.98, cw, 0.32,\n         'Smoke-replay artifacts archived: decision_log.json + final_results/rp2350_page11_Pred.json + logs/ (20+ audit JSONs).',\n         size=13.5, color=INK)",
    "add_text(p, cx, cy + 3.98, cw, 0.32,\n         'Smoke-replay artifacts archived: decision_log.json + final_results/rp2350_page11_Pred.json + logs/ (20+ audit JSONs).',\n         size=13.5, color=INK)", 'noop2')
rep("add_text(p, cx, cy + 4.32, cw, 0.30,", "add_text(p, cx, cy + 4.34, cw, 0.30,", 'b3-l2')
rep("add_text(p, cx, cy + 4.68, cw, 0.60,", "add_text(p, cx, cy + 4.68, cw, 0.56,", 'b3-l3')
rep("add_text(p, cx, cy + 5.36, cw, 0.32,", "add_text(p, cx, cy + 5.28, cw, 0.32,", 'b3-l4')

# ---- B4: note trim, col widths, margins ----
rep("""add_text(p, cx, cy + 0.0, cw, 0.30,
         'Caliber caution: canonical (40 pages) and page scaling (13 pages) are independent experiments; "+0.31" is the total-score delta (connectivity +0.7pp).',
         size=13.5, color=MUTED)""",
    """add_text(p, cx, cy + 0.0, cw, 0.30,
         'Caliber caution: canonical (40 pages) and page scaling (13 pages) are independent; "+0.31" is the total-score delta.',
         size=13.5, color=MUTED)""", 'b4-note')
rep("for c, wd in zip(tbl.columns, [2.55, 2.20, 2.70, 4.10]):",
    "for c, wd in zip(tbl.columns, [2.65, 2.05, 2.55, 4.30]):", 'b4-cols')
rep("""for ri, row in enumerate(rows):
    for ci, val in enumerate(row):
        cell = tbl.cell(ri, ci); cell.text = val
        para = cell.text_frame.paragraphs[0]
        para.alignment = PP_ALIGN.RIGHT if ci == 3 else PP_ALIGN.LEFT
        r0 = para.runs[0]; r0.font.size = Pt(14)""",
    """for ri, row in enumerate(rows):
    for ci, val in enumerate(row):
        cell = tbl.cell(ri, ci); cell.text = val
        cell.margin_left = Inches(0.06); cell.margin_right = Inches(0.06)
        para = cell.text_frame.paragraphs[0]
        para.alignment = PP_ALIGN.RIGHT if ci == 3 else PP_ALIGN.LEFT
        r0 = para.runs[0]; r0.font.size = Pt(14)""", 'b4-margins')

ok = True
for old, new, tag in FIXES:
    n = s.count(old)
    if n != 1 and tag != 'noop1' and tag != 'noop2':
        print(f'ASSERT FAIL [{tag}] count={n} for: {old[:70]!r}')
        ok = False
if not ok:
    print('NO FILE WRITTEN')
    sys.exit(1)
for old, new, tag in FIXES:
    if tag in ('noop1', 'noop2'):
        continue
    s = s.replace(old, new)
open(P, 'w', encoding='utf-8').write(s)
print('patch2 applied OK')
