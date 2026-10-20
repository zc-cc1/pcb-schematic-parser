import json
from pathlib import Path

B = Path(r"C:/Users/Pc/Desktop/新建文件夹/评测结果")


def load(root, case):
    p = root / "intermediate_files/final_results" / f"{case}_Pred.json"
    return json.loads(p.read_text(encoding="utf-8"))


rows = []
for case in [str(i) for i in range(1, 14)]:
    a = load(B / "第七次评测-33.525", case)["components"]
    b = load(B / "第8次测评-54.92", case)["components"]
    for cid, comp in a.items():
        other = b.get(cid)
        if not other:
            continue
        for k in ("Name", "value", "type"):
            if comp.get(k) != other.get(k):
                rows.append((case, cid, comp.get("type"), k, comp.get(k), other.get(k)))
        if comp.get("bbox") != other.get("bbox"):
            rows.append((case, cid, comp.get("type"), "bbox", "", "moved"))

print(f"{'case':>4} {'id':<12} {'type':<12} {'field':<6} {'7th':<22} {'8th':<22}")
print("-" * 84)
for r in rows:
    if r[3] == "bbox":
        continue
    print(f"{r[0]:>4} {r[1]:<12} {str(r[2]):<12} {r[3]:<6} {str(r[4]):<22} {str(r[5]):<22}")

print()
from collections import Counter
c = Counter((r[3],) for r in rows)
print("field change counts:", dict(c))
print("bbox moved per case:", dict(Counter(r[0] for r in rows if r[3] == "bbox")))

print()
print("Name changes by type:")
for (typ,), n in Counter((r[2],) for r in rows if r[3] == "Name").items():
    print(f"  {typ}: {n}")
print()
print("value -> None cases:")
for r in rows:
    if r[3] == "value" and r[5] is None:
        print(f"  case {r[0]} {r[1]} ({r[2]}): {r[4]!r} -> None")
