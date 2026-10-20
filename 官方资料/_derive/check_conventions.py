# -*- coding: utf-8 -*-
"""官方约定的回归检查：坐标系 / bbox 语义 / type 白名单 / 库覆盖。

用途：每次改完管道跑一遍，任何一条 FAIL 都说明格式口径回退了。
    python _derive/check_conventions.py

数据源（只读）：
    赛题六公开数据集/…/open_GT   —— 新口径 golden（Name=型号）
    赛题六公开数据集/…/10_GTcase —— 旧 golden（Name=位号）
"""
import json
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(r"C:\Users\Pc\Desktop\新建文件夹")
OFF = ROOT / "官方资料"
DS = ROOT / "赛题六公开数据集" / "赛题六公开数据集"
OPEN_GT = DS / "open_GT" / "open_GT"
GT10 = DS / "10_GTcase"
TRAIN = DS / "200_train_cases"

results = []


def check(name, ok, detail):
    results.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")


def load_cases(base, pattern):
    out = {}
    for f in sorted(base.iterdir()):
        if f.is_dir():
            j = next(f.glob(pattern), None)
            if j:
                out[f.name] = json.loads(j.read_text(encoding="utf-8"))
    return out


def norm_model(s):
    return re.sub(r"[\s\-_]+", "", str(s or "")).lower()


# ---------------------------------------------------------------- 1. 坐标系
def axis_check():
    import numpy as np
    from PIL import Image
    folder = next(GT10.glob("0025-*"))
    gt = json.loads(next(folder.glob("*_target.json")).read_text(encoding="utf-8"))
    with Image.open(next(folder.glob("*.png"))) as im:
        a = np.asarray(im.convert("L"))
        h = im.height
    hit = {}
    for flip in (False, True):
        n = t = 0
        for pins in gt["pins"].values():
            for v in pins.values():
                x, y = int(v["point"]["x"]), int(v["point"]["y"])
                if flip:
                    y = h - y
                if 0 <= y < a.shape[0] and 0 <= x < a.shape[1]:
                    t += 1
                    n += bool(a[max(0, y - 2):y + 3, max(0, x - 2):x + 3].min() < 128)
        hit[flip] = (n, t)
    flat, flipped = hit[False], hit[True]
    check("坐标系 = 左下原点 y 向上（GT 需翻转才落在墨迹上）",
          flipped[0] / max(flipped[1], 1) > 0.8 and flat[0] < flipped[0],
          f"直接取 {flat[0]}/{flat[1]}，翻转后 {flipped[0]}/{flipped[1]}")


# ---------------------------------------------------------------- 2. bbox 语义
def bbox_check():
    tot = outside = onedge = 0
    gaps = []
    for base, pattern in ((OPEN_GT, "*.json"), (GT10, "*_target.json")):
        for d in load_cases(base, pattern).values():
            for ck, c in d["components"].items():
                if c.get("type") not in ("box", "ic"):
                    continue
                x0, y0, x1, y1 = c["bbox"]
                ds = []
                for v in d.get("pins", {}).get(ck, {}).values():
                    x, y = v["point"]["x"], v["point"]["y"]
                    tot += 1
                    dx, dy = max(x0 - x, x - x1, 0), max(y0 - y, y - y1, 0)
                    dist = max(dx, dy) if (dx or dy) else min(abs(x - x0), abs(x - x1), abs(y - y0), abs(y - y1))
                    if dist > 3:
                        outside += 1
                    else:
                        onedge += 1
                    ds.append(dist)
                if ds:
                    gaps.append(statistics.median(ds))
    check("components.bbox 是本体框（引脚 tip 在框外）",
          outside / max(tot, 1) > 0.9,
          f"框外 {outside}/{tot}，框边 {onedge}；引线距中位 {statistics.median(gaps):.0f}px，"
          f"最大 {max(gaps):.0f}px，超 20px 的元件 {sum(1 for g in gaps if g > 20)}/{len(gaps)}")


# ---------------------------------------------------------------- 3. type 白名单
def type_check():
    cand = json.loads((OFF / "_derive" / "type_candidates.json").read_text(encoding="utf-8"))
    allow = set(cand["union_gt_only"])
    bad = {}
    for tag, base, pattern in (("open_GT", OPEN_GT, "*.json"), ("10GT", GT10, "*_target.json")):
        seen = set()
        for d in load_cases(base, pattern).values():
            seen |= {c.get("type") for c in d["components"].values()}
        extra = seen - allow
        if extra:
            bad[tag] = sorted(extra)
    check("golden 的 type 都落在官方已确认的 21 类内", not bad, bad or f"白名单 {len(allow)} 类")
    train_types = set()
    for d in load_cases(TRAIN, "*_target_named.json").values():
        train_types |= {c.get("type") for c in d["components"].values()}
    print(f"       [INFO] 训练集另有 {len(train_types - allow)} 类不在 golden 白名单内："
          f"{sorted(train_types - allow)}")


# ---------------------------------------------------------------- 4. 库覆盖
def lib_check():
    idx = json.loads((OFF / "_derive" / "box_model_index.json").read_text(encoding="utf-8"))
    hit = num_ok = name_ok = 0
    for d in load_cases(OPEN_GT, "*.json").values():
        for ck, c in d["components"].items():
            if c.get("type") not in ("box", "ic") or norm_model(c.get("Name")) == norm_model(ck):
                continue
            e = idx.get(norm_model(c.get("Name")))
            if not e:
                continue
            hit += 1
            lib = {n: nm for n, nm in e["pins"]}
            gt = d["pins"].get(ck, {})
            if all(re.sub(r"^pin_", "", k) in lib for k in gt):
                num_ok += 1
            if all(lib.get(re.sub(r"^pin_", "", k)) == (v.get("pinname") or "") for k, v in gt.items()):
                name_ok += 1
    check("库能复现 open_GT 的 box 引脚编号（应 100%）",
          hit and num_ok == hit, f"命中 {hit} 个元件，编号全对 {num_ok}，名称全对 {name_ok}")
    print("       [INFO] 名称未全对是库的聚合别名所致，需用丝印 OCR 在候选中消歧")


# ---------------------------------------------------------------- 5. 假命中陷阱
def name_trap_check():
    idx = json.loads((OFF / "_derive" / "box_model_index.json").read_text(encoding="utf-8"))
    gt10_all_designator = True
    for d in load_cases(GT10, "*_target.json").values():
        for k, c in d["components"].items():
            if norm_model(c.get("Name")) != norm_model(k):
                gt10_all_designator = False
    false_hit = norm_model("U1") in idx
    check("10GT 的 Name 全是位号，直接查库会假命中",
          gt10_all_designator and false_hit,
          f"10GT Name==key 全部成立={gt10_all_designator}；库里存在名为 U1 的元件={false_hit}"
          "（所以查询前必须要求 Name != key）")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    axis_check()
    bbox_check()
    type_check()
    lib_check()
    name_trap_check()
    bad = [r for r in results if not r[1]]
    print(f"\n总结：{len(results) - len(bad)}/{len(results)} 通过")
    sys.exit(1 if bad else 0)
