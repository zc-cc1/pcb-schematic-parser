# -*- coding: utf-8 -*-
"""生成派生工件：型号→库索引、type 候选表、GT 清单。

只读官方原件；输出到 _derive/。
"""
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(r"C:\Users\Pc\Desktop\新建文件夹")
OFF = ROOT / "官方资料"
OUT = OFF / "_derive"
DS = ROOT / "赛题六公开数据集" / "赛题六公开数据集"

FILES = {
    "BoxICLibrary": OFF / "BoxICLibrary.json",
    "BoxICLibrary_200cases": OFF / "BoxICLibrary_200cases.json",
    "cpntLibrary": OFF / "cpntLibrary",
}
libs = {k: json.loads(v.read_text(encoding="utf-8")) for k, v in FILES.items()}


def norm(s):
    return re.sub(r"[\s\-_]+", "", str(s or "")).lower()


# ---------------------------------------------------------------- 1. 型号索引
idx = {}
for name, d in libs.items():
    for key, rec in d.items():
        n = norm(key)
        if not n:
            continue
        e = idx.setdefault(n, {"keys": [], "sources": [], "pins": []})
        if key not in e["keys"]:
            e["keys"].append(key)
        e["sources"].append(name)
        e["pins"].extend(rec.get("pins") or [])

for n, e in idx.items():
    pairs = []
    seen = set()
    for pin in e["pins"]:
        num = str(pin.get("number", ""))
        nm = str(pin.get("name", ""))
        if (num, nm) in seen:
            continue
        seen.add((num, nm))
        pairs.append([num, nm])
    e["pins"] = pairs
    e["sources"] = sorted(set(e["sources"]))

big = sum(1 for e in idx.values() if len(e["pins"]) > 64)
(OUT / "box_model_index.json").write_text(
    json.dumps(idx, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

# ---------------------------------------------------------------- 2. type 候选
def types_of(cases):
    c = Counter()
    for d in cases.values():
        for comp in d.get("components", {}).values():
            c[comp.get("type")] += 1
    return c


def load_cases(folder, pattern):
    out = {}
    for f in sorted(folder.iterdir()):
        if f.is_dir():
            j = next(f.glob(pattern), None)
            if j:
                out[f.name] = json.loads(j.read_text(encoding="utf-8"))
    return out


open_gt = load_cases(DS / "open_GT" / "open_GT", "*.json")
gt10 = load_cases(DS / "10_GTcase", "*_target.json")
train = load_cases(DS / "200_train_cases", "*_target_named.json")
ext1 = {"x": json.loads((ROOT / "case_001_扩展示例.json").read_text(encoding="utf-8"))}

sets = {
    "open_GT": types_of(open_gt),
    "10_GTcase": types_of(gt10),
    "200_train": types_of(train),
    "case_001_ext": types_of(ext1),
}
union = Counter()
for c in sets.values():
    union.update(c)
authoritative = Counter()
for k in ("open_GT", "10_GTcase"):
    authoritative.update(sets[k])

(OUT / "type_candidates.json").write_text(json.dumps({
    "note": "官方 type_list（Q&A 称 34 类）未收到；以下为各资料来源实测 type 与并集，供对齐/校验使用。",
    "per_source": {k: dict(v) for k, v in sets.items()},
    "union_all": dict(union),
    "union_gt_only": dict(authoritative),
    "gt_only_sorted": sorted(authoritative),
    "in_train_not_in_gt": sorted(set(sets["200_train"]) - set(authoritative)),
    "in_ext_not_in_gt": sorted(set(sets["case_001_ext"]) - set(authoritative)),
}, ensure_ascii=False, indent=2), encoding="utf-8")

# ---------------------------------------------------------------- 3. GT 清单
inv = {}
for tag, cases in (("10_GTcase", gt10), ("open_GT", open_gt)):
    rows = []
    for case, d in cases.items():
        comps = d.get("components", {})
        box = [k for k, c in comps.items() if c.get("type") in ("box", "ic")]
        name_eq_key = sum(1 for k, c in comps.items() if norm(c.get("Name")) == norm(k))
        rows.append({
            "case": case,
            "components": len(comps),
            "pins": sum(len(v) for v in d.get("pins", {}).values()),
            "nets": len(d.get("nets", {})),
            "box_components": len(box),
            "name_equals_key": name_eq_key,
            "net_key_style": "net_N" if any(re.fullmatch(r"net_\d+", k) for k in d.get("nets", {})) else "label",
        })
    inv[tag] = rows

(OUT / "gt_inventory.json").write_text(
    json.dumps(inv, ensure_ascii=False, indent=2), encoding="utf-8")

print("index entries:", len(idx), "size:",
      (OUT / "box_model_index.json").stat().st_size, "bytes; oversized:", big)
print("done")
