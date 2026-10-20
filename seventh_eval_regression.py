"""Explain the 7th-eval regression: the canonical box table transposes W/H."""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

F = r"C:\Windows\Fonts\msyh.ttc"


def f(s):
    try:
        return ImageFont.truetype(F, s)
    except OSError:
        return ImageFont.load_default()


BLUE, RED, GREY = (70, 120, 200), (215, 70, 60), (140, 140, 140)
img = Image.new("RGB", (1600, 620), "white")
d = ImageDraw.Draw(img)

d.text((70, 34), "第 7 次掉分原因：canonical 尺寸表把横向符号的框转置了", fill=(0, 0, 0), font=f(32))
d.text((70, 78), "中心不变、宽高互换 → 角点各移 6px（电阻）/ 2px（电容）", fill=(110, 110, 110), font=f(21))

S = 5  # screen px per model px


def sym(x, y, w, h, color, tag):
    W, H = w * S, h * S
    d.rectangle([x, y, x + W, y + H], outline=color, width=4, fill=(color[0], color[1], color[2], 20) if False else None)
    for cx, cy in ((x, y), (x + W, y), (x, y + H), (x + W, y + H)):
        d.ellipse([cx - 5, cy - 5, cx + 5, cy + 5], fill=color)
    d.text((x + W / 2, y - 26), tag, fill=color, font=f(21), anchor="mb")


TOP = 260
# ---- resistor ----
d.text((110, 170), "电阻：表里只存 8×20 一种朝向", fill=(30, 30, 30), font=f(25))
sym(140, TOP, 20, 8, BLUE, "第5次 20×8")
sym(330, TOP, 8, 20, RED, "第7次 8×20")
d.line([(255, TOP + 20), (315, TOP + 20)], fill=GREY, width=3)
d.polygon([(315, TOP + 20), (303, TOP + 13), (303, TOP + 27)], fill=GREY)
d.text((140, TOP + 112), "四个角两轴各移 6px", fill=RED, font=f(20))

# ---- capacitor ----
d.text((530, 170), "电容：表里只存 16×20", fill=(30, 30, 30), font=f(25))
sym(560, TOP, 20, 16, BLUE, "第5次  20×16")
sym(740, TOP, 16, 20, RED, "第7次  16×20")
d.line([(675, TOP + 40), (725, TOP + 40)], fill=GREY, width=3)
d.polygon([(725, TOP + 40), (713, TOP + 33), (713, TOP + 47)], fill=GREY)
d.text((560, TOP + 112), "四个角两轴各移 2px", fill=RED, font=f(20))

# ---- right: the numbers ----
x0 = 960
d.text((x0, 150), "只有 case 12 / 13 掉，为什么？", fill=(30, 30, 30), font=f(25))
rows = [("case 12  C2", "19.50 → 21.50"), ("case 12  C3", "18.03 → 20.03"),
        ("case 13  C1", "20.36 → 22.36"), ("case 13  C5", "20.15 → 21.55"),
        ("case 13  C6", "20.52 → 22.52")]
yy = 196
for name, txt in rows:
    d.text((x0, yy), name, fill=(60, 60, 60), font=f(21))
    d.text((x0 + 165, yy), txt, fill=RED, font=f(21))
    yy += 32
d.text((x0, yy + 10), "判定线：每个坐标 ≤ 20px", fill=(90, 90, 90), font=f(20))
d.text((x0, yy + 40), "这两个案例原本就差 13–14px，加 6px 刚好越线", fill=(90, 90, 90), font=f(20))
d.text((x0, yy + 70), "其余 11 例：框动了但没越线 → 分数逐位不变", fill=(90, 90, 90), font=f(20))
d.text((x0, yy + 100), "0 例变好", fill=RED, font=f(21))

d.rectangle([70, 520, 1530, 592], fill=(250, 244, 238))
d.text((86, 536), "净效果：component_f1 .7482 → .7108（−3.7 点）→ 总分 34.65 → 33.52（−1.12）", fill=(160, 65, 30), font=f(24))

out = Path(__file__).with_name("seventh_eval_regression.png")
img.save(out)
print("wrote", out)

