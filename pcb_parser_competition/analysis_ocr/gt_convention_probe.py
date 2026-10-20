"""GT conventions on the 200 public pages: is Name just the designator? is value often null?"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(r"C:\Users\Pc\Desktop\新建文件夹\赛题六公开数据集\赛题六公开数据集\200_train_cases")

TYPES = defaultdict(lambda: {"n": 0, "name_eq_key": 0, "name_empty": 0,
                             "value_null": 0, "value_empty": 0, "pins": 0})


def main():
    cases = sorted(p for p in ROOT.iterdir() if p.is_dir())
    key_name_pairs = Counter()
    for case in cases:
        f = next(case.glob("*_target_named.json"), None)
        if f is None:
            continue
        g = json.loads(f.read_text(encoding="utf-8"))
        for key, comp in (g.get("components") or {}).items():
            t = str(comp.get("type") or "")
            s = TYPES[t]
            s["n"] += 1
            name = comp.get("Name")
            if name == key:
                s["name_eq_key"] += 1
            elif not name:
                s["name_empty"] += 1
            else:
                key_name_pairs[(t, name)] += 1
            v = comp.get("value")
            if v is None:
                s["value_null"] += 1
            elif v == "":
                s["value_empty"] += 1
    print("type                      n   Name==key  Name empty   value null  value empty")
    for t, s in sorted(TYPES.items(), key=lambda kv: -kv[1]["n"]):
        print("%-22s %5d   %5.1f%%    %5.1f%%      %5.1f%%     %5.1f%%" % (
            t, s["n"],
            100.0 * s["name_eq_key"] / s["n"]
            if s["n"] else 0,
            100.0 * s["name_empty"] / s["n"],
            100.0 * s["value_null"] / s["n"],
            100.0 * s["value_empty"] / s["n"]))
    print("\nName != key examples (non-empty) by type:")
    for (t, name), c in sorted(key_name_pairs.items(), key=lambda kv: -kv[1])[:25]:
        print(f"  {t:20s} {name!r} x{c}")
    print("\nnum cases:", len(cases))


if __name__ == "__main__":
    sys.exit(main())
