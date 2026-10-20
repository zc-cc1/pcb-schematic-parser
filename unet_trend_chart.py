"""Noise band + paired delta stats, then a Pillow chart of the three A3 curves."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

RUNS = Path(r"D:/pcb_unet_v2/runs")
RUNS_OF_INTEREST = ["A3_pin_l1", "A3_pin_split", "A3_split_w2"]


def load(run: str) -> pd.DataFrame:
    rows = [json.loads(l) for l in (RUNS / run / "epochs.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    return pd.DataFrame(rows).sort_values("epoch").reset_index(drop=True)


frames = {r: load(r) for r in RUNS_OF_INTEREST}

print("noise band (std of epoch-to-epoch metric) over epochs 5..20")
for r, d in frames.items():
    w = d["wire_pixel_f1"].iloc[4:20]
    p = d["pin_point_f1"].iloc[4:20]
    c = d["composite"].iloc[4:20]
    print(f"  {r:<14} wire std={w.std(ddof=1):.4f} range={w.max()-w.min():.4f} | "
          f"pin std={p.std(ddof=1):.4f} range={p.max()-p.min():.4f} | comp std={c.std(ddof=1):.4f}")

a = frames["A3_split_w2"]
b = frames["A3_pin_split"].head(len(a))
dw = (a["wire_pixel_f1"] - b["wire_pixel_f1"])
dp = (a["pin_point_f1"] - b["pin_point_f1"])
dc = (a["composite"] - b["composite"])
print("\npaired per-epoch delta  A3_split_w2 minus A3_pin_split (same epochs 1..%d)" % len(a))
print(f"  wire delta mean={dw.mean():+.4f} std={dw.std(ddof=1):.4f}  wins={int((dw>0).sum())}/{len(dw)}")
print(f"  pin  delta mean={dp.mean():+.4f} std={dp.std(ddof=1):.4f}  wins={int((dp>0).sum())}/{len(dp)}")
print(f"  comp delta mean={dc.mean():+.4f} std={dc.std(ddof=1):.4f}  wins={int((dc>0).sum())}/{len(dc)}")
# how many epochs the peak beats the ep3 wire value in split_w2
d = a
print(f"\nA3_split_w2: epochs beating ep3 wire 0.8398 = {int((d['wire_pixel_f1'] > 0.8398).sum())}")
print(f"A3_split_w2: epochs beating ep14 comp 0.7983 = {int((d['composite'] > 0.7983).sum())}")

# ---------------- chart ----------------
def font(size: int):
    for p in (r"C:\Windows\Fonts\segoeui.ttf", r"C:\Windows\Fonts\arial.ttf"):
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            continue
    return ImageFont.load_default()

W, H = 1500, 1000
pad_l, pad_r, pad_t, pad_b = 90, 40, 70, 70
img = Image.new("RGB", (W, H), "white")
dr = ImageDraw.Draw(img)
colors = {"A3_pin_l1": (150, 150, 150), "A3_pin_split": (60, 120, 220), "A3_split_w2": (220, 70, 60)}
labels = {"A3_pin_l1": "A3_pin_l1 (w1.0, shared decoder)",
          "A3_pin_split": "A3_pin_split (w1.0, split decoder)",
          "A3_split_w2": "A3_split_w2 (w2.0, split decoder)  <- running"}

panels = [("wire_pixel_f1", "Wire pixel F1 (internal val)"), ("pin_point_f1", "Pin point F1 (internal val)")]
panel_h = (H - pad_t - pad_b - 40) // 2
max_ep = 40
for pi, (key, title) in enumerate(panels):
    top = pad_t + pi * (panel_h + 40)
    bot = top + panel_h
    vals = [v for d in frames.values() for v in d[key]]
    lo, hi = min(vals) - 0.01, max(vals) + 0.01
    def y(v):
        return bot - (v - lo) / (hi - lo) * (bot - top)
    def x(e):
        return pad_l + (e - 1) / (max_ep - 1) * (W - pad_l - pad_r)
    # grid
    for gv in [lo + (hi - lo) * i / 5 for i in range(6)]:
        dr.line([(pad_l, y(gv)), (W - pad_r, y(gv))], fill=(235, 235, 235), width=1)
        dr.text((pad_l - 12, y(gv) - 8), f"{gv:.3f}", fill=(90, 90, 90), font=font(17), anchor="ra")
    for e in range(1, max_ep + 1, 5):
        dr.line([(x(e), top), (x(e), bot)], fill=(245, 245, 245), width=1)
        dr.text((x(e), bot + 10), str(e), fill=(90, 90, 90), font=font(17), anchor="ma")
    dr.rectangle([pad_l, top, W - pad_r, bot], outline=(200, 200, 200))
    dr.text((pad_l + 10, top + 8), title, fill=(20, 20, 20), font=font(22))
    for r, d in frames.items():
        pts = [(x(int(e)), y(v)) for e, v in zip(d["epoch"], d[key])]
        dr.line(pts, fill=colors[r], width=3)
        dr.ellipse([pts[-1][0] - 4, pts[-1][1] - 4, pts[-1][0] + 4, pts[-1][1] + 4], fill=colors[r])
    # mark best point of split_w2
    d = frames["A3_split_w2"]
    i = d[key].idxmax()
    bx, by = x(int(d["epoch"][i])), y(d[key][i])
    dr.ellipse([bx - 8, by - 8, bx + 8, by + 8], outline=(220, 70, 60), width=3)
    dr.text((bx + 12, by - 26), f"best {d[key][i]:.4f} @ep{int(d['epoch'][i])}", fill=(220, 70, 60), font=font(18))

dr.text((pad_l, 24), "UNet v2 pin-head experiment: metric per epoch (eval on internal val split)", fill=(0, 0, 0), font=font(26))
lx = pad_l
for r in frames:
    dr.line([(lx, H - 30), (lx + 34, H - 30)], fill=colors[r], width=4)
    dr.text((lx + 42, H - 40), labels[r], fill=(40, 40, 40), font=font(19))
    lx += 42 + int(dr.textlength(labels[r], font=font(19))) + 40

out = Path(__file__).with_name("unet_trend_chart.png")
img.save(out)
print("\nwrote", out)
