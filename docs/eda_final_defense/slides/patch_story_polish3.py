"""Story-polish round 3 micro-fixes."""
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

# P11: json rows tighter + right col
rep("    y = cy + 3.66 + i * 0.68", "    y = cy + 3.62 + i * 0.62", 'P11 rows')
rep("add_text(p, cx + 5.95, cy + 4.58, 6.55, 0.36,\n         'Engineering details (Docker · SHA256 · audit logs · 626+ tests · runtime scope): Backup C.',\n         size=13, color=MUTED)",
    "add_text(p, cx + 5.95, cy + 4.54, 6.55, 0.32,\n         'Engineering details (Docker · SHA256 · audit · 626+ tests · runtime): Backup C.',\n         size=12.5, color=MUTED)", 'P11 bkp line')

# P12 absolute layout
rep("    y = cy + 0.58 + i * 0.6", "    y = cy + 0.53 + i * 0.55", 'P12 contrib rows')
rep("add_text(part4, cx, cy + 2.4, cw, 0.32,", "add_text(part4, cx, cy + 2.24, cw, 0.3,", 'P12 note y')
rep("add_text(part4, cx, cy + 2.78, cw, 0.42, 'Current limitations', size=19, bold=True, color=NAVY)",
    "add_text(part4, cx, cy + 2.63, cw, 0.4, 'Current limitations', size=19, bold=True, color=NAVY)", 'P12 lim hdr')
rep("    y = cy + 3.24 + i * 0.36", "    y = cy + 3.08 + i * 0.32", 'P12 lim rows')
rep("add_text(part4, cx, cy + 4.42, cw, 0.42, 'Future work', size=19, bold=True, color=NAVY)",
    "add_text(part4, cx, cy + 4.08, cw, 0.4, 'Future work', size=19, bold=True, color=NAVY)", 'P12 fut hdr')
rep("    y = cy + 4.88 + i * 0.36", "    y = cy + 4.52 + i * 0.3", 'P12 fut rows')

# B3 compress bottom
rep("""add_text(p, cx, cy + 4.3, cw, 0.32,
         'Entry semantics: image ENTRYPOINT=/submission/run.sh (verified in OCI config); /submission/agent_system stays the official placeholder - confirm invocation with organizers.',
         size=11.5, color=MUTED)
add_text(p, cx, cy + 4.68, cw, 0.44,""",
    """add_text(p, cx, cy + 4.2, cw, 0.28,
         'Entry semantics: ENTRYPOINT=/submission/run.sh (verified in OCI config); agent_system stays the official placeholder.',
         size=11, color=MUTED)
add_text(p, cx, cy + 4.52, cw, 0.44,""", 'B3 entry')
rep("add_text(p, cx, cy + 5.14, cw, 0.3,\n         'Runtime scope: CPU datum 2000.5 s/case (official example page); official GPU timing not measured - no 2-hour compliance claim.',\n         size=12, color=MUTED)",
    "add_text(p, cx, cy + 5.06, cw, 0.28,\n         'Runtime scope: CPU datum 2000.5 s/case (official example page); official GPU timing not measured - no 2-hour compliance claim.',\n         size=12, color=MUTED)", 'B3 runtime y')

open(p, 'w', encoding='utf-8').write(s)
print('round3 applied:', ok)
