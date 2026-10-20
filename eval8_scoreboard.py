"""8th eval scoreboard: where the +21.40 came from, and what is left."""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

FP = r"C:\Windows\Fonts\msyh.ttc"


def f(s):
    try:
        return ImageFont.truetype(FP, s)
    except OSError:
        return ImageFont.load_default()


W, H = 1600, 700
img = Image.new("RGB", (W, H), "white")
d = ImageDraw.Draw(img)

COL = {"C": (70, 120, 200), "P": (240, 150, 60), "H": (215, 70, 60), "L": (120, 180, 120)}

d.text((70, 30), "第 8 次评测 54.92：+21.40 分从哪来、还剩多少", fill=(0, 0, 0), font=f(31))
d.text((70, 74), "模型一个没换；nets 2→82、pins 139→437，全部来自把出口闸门打开", fill=(110, 110, 110), font=f(20))

# ---------- left: delta ----------
T, B = 180, 520
d.text((70, 124), "① 与第 7 次相比的分项变化（分）", fill=(30, 30, 30), font=f(24))
zero = 500
d.line([(zero, T), (zero, B)], fill=(180, 180, 180), width=2)
d.text((zero, B + 8), "0", fill=(120, 120, 120), font=f(17), anchor="ma")
scale = 10.5
deltas = [("元件 C", -0.70, "C"), ("引脚 P", 3.41, "P"), ("连通 H", 15.65, "H"), ("导线 L", 3.04, "L")]
for i, (name, v, key) in enumerate(deltas):
    y = T + 30 + i * 78
    x2 = zero + v * scale
    d.rectangle([min(zero, x2), y, max(zero, x2), y + 40], fill=COL[key])
    label = f"{v:+.2f}"
    if v > 0:
        d.text((x2 + 10, y + 20), label, fill=(40, 40, 40), font=f(21), anchor="lm")
    else:
        d.text((x2 - 10, y + 20), label, fill=(40, 40, 40), font=f(21), anchor="rm")
    d.text((70, y + 20), name, fill=(50, 50, 50), font=f(21), anchor="lm")
d.text((70, B + 40), "合计 +21.40 → 33.52 → 54.92", fill=(160, 65, 30), font=f(21))

# ---------- right: utilisation ----------
R0 = 900
d.text((R0, 124), "② 四个桶的现状（分）", fill=(30, 30, 30), font=f(24))
bars = [("元件 C", 20.62, 30, "C", "缺口 9.4 ｜ 12/13 两例几乎全灭"),
        ("引脚 P", 14.35, 25, "P", "缺口 10.7 ｜ IC 引脚只导出 104/191"),
        ("连通 H", 16.80, 35, "H", "缺口 18.2 ｜ 精确率 .78、召回 .44"),
        ("导线 L", 3.15, 10, "L", "缺口 6.9 ｜ 精确率 .56、召回 .51")]
mx = 35.0
bar_w = 400
for i, (name, got, cap, key, note) in enumerate(bars):
    y = T + 20 + i * 88
    d.text((R0, y - 28), f"{name}   已得 {got:.1f} / 满分 {cap}", fill=(50, 50, 50), font=f(20))
    d.rectangle([R0, y, R0 + bar_w * cap / mx, y + 30], fill=(238, 238, 238))
    d.rectangle([R0, y, R0 + bar_w * got / mx, y + 30], fill=COL[key])
    d.text((R0 + bar_w * cap / mx + 12, y + 15), f"{int(got / cap * 100)}%", fill=(60, 60, 60), font=f(20), anchor="lm")
    d.text((R0, y + 38), note, fill=(130, 130, 130), font=f(17))

# ---------- bottom note ----------
d.rectangle([70, 600, 1530, 664], fill=(250, 244, 238))
d.text((86, 615), "下一个 20 分不在模型里：713 个引脚编号未观测 · 54 个网络卡在 junction 语义 · 169 个网络退化成单点碎片",
       fill=(160, 65, 30), font=f(21))

out = Path(__file__).with_name("eval8_scoreboard.png")
img.save(out)
print("wrote", out)
