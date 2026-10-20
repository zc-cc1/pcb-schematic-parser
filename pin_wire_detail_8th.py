import json
from pathlib import Path

E8 = Path(r"C:/Users/Pc/Desktop/新建文件夹/评测结果/第8次测评-54.92")
d = next((E8 / "scoring_result/report/submission").glob("*"))
rows = {}
for p in sorted(d.glob("*.json"), key=lambda x: int(x.stem) if x.stem.isdigit() else 0):
    if not p.stem.isdigit():
        continue
    j = json.loads(p.read_text(encoding="utf-8"))
    pin = j["pin"]["f1"]
    diag = j["pin"]["diagnostics"]
    con = j["connectivity"]["f1"]
    cdiag = j["connectivity"]["diagnostics"]
    rows[p.stem] = (pin, diag, con, cdiag)

print(f"{'case':>4} | {'pin tp/fp/fn':>14} {'P':>5}{'R':>6} | {'tip_mean':>8} | {'nets gt/pred':>13} {'H P':>6}{'H R':>6}")
print("-" * 84)
for c in sorted(rows, key=int):
    pin, diag, con, cdiag = rows[c]
    tip = diag.get("tip_distance_px", {}).get("mean")
    print(f"{c:>4} | {pin['tp']:>4}/{pin['fp']:>3}/{pin['fn']:>4}   {pin['precision']:>5.2f}{pin['recall']:>6.2f} | "
          f"{(tip if tip is not None else -1):>8.2f} | "
          f"{cdiag['gt_logical_net_count']:>5}/{cdiag['pred_logical_net_count']:<5} "
          f"{con['precision']:>5.2f}{con['recall']:>6.2f}")

print()
print("wire (segment) detail")
for c in sorted(rows, key=int):
    pass
d2 = {}
for p in sorted(d.glob("*.json"), key=lambda x: int(x.stem) if x.stem.isdigit() else 0):
    if not p.stem.isdigit():
        continue
    j = json.loads(p.read_text(encoding="utf-8"))
    w = j["wire"]["f1"]
    wd = j["wire"]["diagnostics"]
    d2[p.stem] = (w, wd)
for c in sorted(d2, key=int):
    w, wd = d2[c]
    print(f"{c:>4} | seg tp={w['tp']:>3} fp={w['fp']:>3} fn={w['fn']:>3} P={w['precision']:.2f} R={w['recall']:.2f} | "
          f"gt_seg={wd['gt_segment_count']:>3} pred_seg={wd['pred_segment_count']:>3}")
