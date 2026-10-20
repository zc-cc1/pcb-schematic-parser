"""Inspect whether the current A3_split_w2 run is trending up vs earlier A3/A2 runs.

Read-only over D:/pcb_unet_v2/runs/*/epochs.jsonl.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

RUNS = Path(r"D:/pcb_unet_v2/runs")
RUNS_OF_INTEREST = ["A2_full60", "A3_pin_l1", "A3_pin_disk", "A3_pin_split", "A3_split_w2"]


def load(run: str) -> pd.DataFrame:
    rows = []
    path = RUNS / run / "epochs.jsonl"
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    df = pd.DataFrame(rows).sort_values("epoch").reset_index(drop=True)
    df["run"] = run
    return df


def smooth(s: pd.Series, w: int = 5) -> pd.Series:
    return s.rolling(w, min_periods=1).mean()


def slope(s: pd.Series) -> float:
    """Least-squares slope per epoch over the given series."""
    n = len(s)
    if n < 2:
        return float("nan")
    xs = list(range(n))
    mx, my = sum(xs) / n, float(s.mean())
    num = sum((x - mx) * (y - my) for x, y in zip(xs, s))
    den = sum((x - mx) ** 2 for x in xs)
    return num / den


frames = {r: load(r) for r in RUNS_OF_INTEREST if (RUNS / r / "epochs.jsonl").exists()}

print("=" * 96)
print("per-run summary (internal val, 59-page protocol)")
print("=" * 96)
print(f"{'run':<14}{'ep':>4}{'bestC@ep':>10}{'bestC':>8}{'bestWire@ep':>12}{'bestWire':>10}{'bestPin@ep':>11}{'bestPin':>9}{'lastWire':>10}{'lastPin':>9}")
for run, df in frames.items():
    bc = df.loc[df["composite"].idxmax()]
    bw = df.loc[df["wire_pixel_f1"].idxmax()]
    bp = df.loc[df["pin_point_f1"].idxmax()]
    print(f"{run:<14}{len(df):>4}{int(bc['epoch']):>10}{bc['composite']:>8.4f}"
          f"{int(bw['epoch']):>12}{bw['wire_pixel_f1']:>10.4f}"
          f"{int(bp['epoch']):>11}{bp['pin_point_f1']:>9.4f}"
          f"{df['wire_pixel_f1'].iloc[-1]:>10.4f}{df['pin_point_f1'].iloc[-1]:>9.4f}")

print()
print("=" * 96)
print("head-to-head at equal epoch budget (epochs 1..min_len) -- wire F1 / pin F1")
print("=" * 96)
head = {r: df for r, df in frames.items() if r in ("A3_pin_l1", "A3_pin_split", "A3_split_w2")}
n = min(len(df) for df in head.values())
print(f"common budget = {n} epochs")
print(f"{'run':<14}{'meanWire':>10}{'maxWire':>9}{'meanPin':>9}{'maxPin':>8}{'meanComp':>10}{'maxComp':>9}{'wireSlope':>11}{'pinSlope':>10}")
for run, df in head.items():
    d = df.head(n)
    print(f"{run:<14}{d['wire_pixel_f1'].mean():>10.4f}{d['wire_pixel_f1'].max():>9.4f}"
          f"{d['pin_point_f1'].mean():>9.4f}{d['pin_point_f1'].max():>8.4f}"
          f"{d['composite'].mean():>10.4f}{d['composite'].max():>9.4f}"
          f"{slope(d['wire_pixel_f1']):>+11.5f}{slope(d['pin_point_f1']):>+10.5f}")

print()
print("=" * 96)
print("A3_split_w2 epoch-by-epoch with 5-epoch rolling means")
print("=" * 96)
df = frames["A3_split_w2"].copy()
df["wire5"] = smooth(df["wire_pixel_f1"])
df["pin5"] = smooth(df["pin_point_f1"])
df["comp5"] = smooth(df["composite"])
print(f"{'ep':>3}{'wire':>8}{'wire5':>8}{'pin':>8}{'pin5':>8}{'comp':>8}{'comp5':>8}{'trainLoss':>11}{'valLoss':>9}")
for _, r in df.iterrows():
    print(f"{int(r['epoch']):>3}{r['wire_pixel_f1']:>8.4f}{r['wire5']:>8.4f}{r['pin_point_f1']:>8.4f}"
          f"{r['pin5']:>8.4f}{r['composite']:>8.4f}{r['comp5']:>8.4f}"
          f"{r['train_loss']:>11.4f}{r['val_loss']:>9.4f}")

print()
print("rolling-window means over last 5 epochs (trend signal)")
print(f"{'run':<14}{'wire_last5':>11}{'wire_prev5':>11}{'delta':>9}{'pin_last5':>10}{'pin_prev5':>10}{'delta':>9}")
for run, d in head.items():
    if len(d) < 10:
        continue
    w = d["wire_pixel_f1"]
    p = d["pin_point_f1"]
    print(f"{run:<14}{w.tail(5).mean():>11.4f}{w.iloc[-10:-5].mean():>11.4f}{w.tail(5).mean()-w.iloc[-10:-5].mean():>+9.4f}"
          f"{p.tail(5).mean():>10.4f}{p.iloc[-10:-5].mean():>10.4f}{p.tail(5).mean()-p.iloc[-10:-5].mean():>+9.4f}")

# save the tidy table for later plotting
out = pd.concat(frames.values(), ignore_index=True)
out.to_csv(Path(__file__).with_name("unet_trend_epochs.csv"), index=False)
print("\nwrote", Path(__file__).with_name("unet_trend_epochs.csv"))
