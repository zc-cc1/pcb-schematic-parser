import json
from pathlib import Path

B = Path(r"C:/Users/Pc/Desktop/新建文件夹/评测结果")


def load(root, case):
    p = root / "intermediate_files/final_results" / f"{case}_Pred.json"
    return json.loads(p.read_text(encoding="utf-8"))


for case in ("13", "12", "1"):
    a = load(B / "第七次评测-33.525", case)["components"]
    b = load(B / "第8次测评-54.92", case)["components"]
    moved = [(k, a[k]["bbox"], b[k]["bbox"]) for k in a if k in b and a[k]["bbox"] != b[k]["bbox"]]
    print(f"case {case}: {len(a)} vs {len(b)} comps, {len(moved)} moved")
    for k, x, y in moved[:5]:
        dw = round((y[2] - y[0]) - (x[2] - x[0]), 2)
        dh = round((y[3] - y[1]) - (x[3] - x[1]), 2)
        print(f"   {k:<10} {[round(v,1) for v in x]} -> {[round(v,1) for v in y]}  dW={dw:+.1f} dH={dh:+.1f}")
    print()
