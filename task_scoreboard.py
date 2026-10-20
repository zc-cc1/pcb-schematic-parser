"""Score decomposition chart for the PCB schematic-parsing task (Pillow only)."""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

FONT = r"C:\Windows\Fonts\msyh.ttc"


def f(size: int):
    try:
        return ImageFont.truetype(FONT, size)
    except OSError:
        return ImageFont.load_default()


W, H = 1600, 980
img = Image.new("RGB", (W, H), "white")
dr = ImageDraw.Draw(img)

rounds = [
    ("第3次\n20.45", 0.0499, 0.6929, 0.0429, 0.0130),
    ("第4次\n28.92", 0.5419, 0.4376, 0.0458, 0.0124),
    ("第5次\n34.65", 0.7482, 0.4376, 0.0330, 0.0107),
    ("第6次\n0.00", 0.0, 0.0, 0.0, 0.0),
]
weights = (0.30, 0.25, 0.35, 0.10)
names = ("元件 C (30%)", "引脚 P (25%)", "连通 H (35%)", "导线 L (10%)")
cols = ((70, 120, 200), (240, 150, 60), (200, 70, 70), (120, 180, 120))

# ---------- left panel: stacked contributions ----------
L, R, T, B = 90, 760, 120, 780
dr.text((L, 40), "官方得分拆解：每一分从哪来", fill=(0, 0, 0), font=f(30))
dr.text((L, 82), "分数 = Σ 权重 × 指标，单位：分（满分 100）", fill=(110, 110, 110), font=f(20))

maxv = 100.0
bw = 110
gap = 40
for gv in range(0, 101, 20):
    y = B - gv / maxv * (B - T)
    dr.line([(L, y), (R, y)], fill=(238, 238, 238), width=1)
    dr.text((L - 12, y), str(gv), fill=(120, 120, 120), font=f(18), anchor="rm")
dr.line([(L, T), (L, B)], fill=(190, 190, 190))
dr.line([(L, B), (R, B)], fill=(190, 190, 190))

for i, (label, c, p, h, l) in enumerate(rounds):
    x = L + 60 + i * (bw + gap)
    vals = [c * weights[0] * 100, p * weights[1] * 100, h * weights[2] * 100, l * weights[3] * 100]
    y = B
    for v, col in zip(vals, cols):
        if v <= 0:
            continue
        top = y - v / maxv * (B - T)
        dr.rectangle([x, top, x + bw, y], fill=col)
        if v >= 2.0:
            dr.text((x + bw / 2, (top + y) / 2), f"{v:.1f}", fill="white", font=f(19), anchor="mm")
        y = top
    total = sum(vals)
    dr.text((x + bw / 2, y - 14), f"{total:.1f}", fill=(30, 30, 30), font=f(24), anchor="mb")
    for j, line in enumerate(label.split("\n")):
        dr.text((x + bw / 2, B + 16 + j * 26), line, fill=(50, 50, 50), font=f(19), anchor="ma")

# legend
lx = L
for nm, col in zip(names, cols):
    dr.rectangle([lx, B + 84, lx + 22, B + 104], fill=col)
    dr.text((lx + 30, B + 94), nm, fill=(50, 50, 50), font=f(19), anchor="lm")
    lx += 34 + int(dr.textlength(nm, font=f(19))) + 24

# ---------- right panel: bucket utilisation ----------
L2, R2 = 900, 1540
dr.text((L2, 40), "四个分项：权重上限 vs 第 5 次实际拿到", fill=(0, 0, 0), font=f(30))
dr.text((L2, 82), "45% 的权重（连通+导线）只换回 1.3 分", fill=(180, 60, 60), font=f(20))
buckets = [("元件 C", 0.7482, 0.30, cols[0]), ("引脚 P", 0.4376, 0.25, cols[1]),
           ("连通 H", 0.032967, 0.35, cols[2]), ("导线 L", 0.010661, 0.10, cols[3])]
row_h = 96
top0 = T + 40
maxbar = R2 - L2 - 150
for i, (nm, val, wgt, col) in enumerate(buckets):
    y = top0 + i * row_h
    cap = wgt * 100
    w_full = maxbar * cap / 35.0
    w_got = maxbar * (val * cap) / 35.0
    dr.text((L2, y - 26), f"{nm}   上限 {cap:.0f} 分", fill=(50, 50, 50), font=f(21))
    dr.rectangle([L2, y, L2 + w_full, y + 30], fill=(236, 236, 236))
    if w_got > 0.5:
        dr.rectangle([L2, y, L2 + w_got, y + 30], fill=col)
    dr.text((L2 + max(w_full, w_got) + 14, y + 15), f"{val * cap:.1f} 分（指标 {val * 100:.0f}%）",
            fill=(40, 40, 40), font=f(19), anchor="lm")

dr.text((L2, top0 + 4 * row_h + 26),
        "第 5 次合计 34.65 分：元件 22.4 + 引脚 10.9 + 连通 1.2 + 导线 0.1",
        fill=(60, 60, 60), font=f(20))
dr.text((L2, top0 + 4 * row_h + 58),
        "13 个 case 里 nets 总数为 2 → 45% 的权重几乎空转",
        fill=(180, 60, 60), font=f(20))

out = Path(__file__).with_name("task_scoreboard.png")
img.save(out)
print("wrote", out)
