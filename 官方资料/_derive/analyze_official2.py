# -*- coding: utf-8 -*-
"""官方资料 第二轮分析：open_GT / 扩展示例 / 三库对真实型号的覆盖与引脚对齐。

只读；结果写入 _derive/official_analysis2.md
"""
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(r"C:\Users\Pc\Desktop\新建文件夹")
OFF = ROOT / "官方资料"
OUT = OFF / "_derive"
DS = ROOT / "赛题六公开数据集" / "赛题六公开数据集"
OPENG = DS / "open_GT" / "open_GT"

libs = {
    "BoxICLibrary.json": json.loads((OFF / "BoxICLibrary.json").read_text(encoding="utf-8")),
    "BoxICLibrary_200cases.json": json.loads((OFF / "BoxICLibrary_200cases.json").read_text(encoding="utf-8")),
    "cpntLibrary": json.loads((OFF / "cpntLibrary").read_text(encoding="utf-8")),
}

L = []
def p(s=""):
    L.append(str(s))


def norm(s):
    return re.sub(r"\s+", "", str(s or "")).lower()


# 归一化索引（容错：大小写/空格/斜杠差异）
norm_idx = {name: {} for name in libs}
for name, d in libs.items():
    for k, v in d.items():
        norm_idx[name].setdefault(norm(k), (k, v))

p("# 官方资料分析（二）：真实型号覆盖与引脚对齐")
p()

# ------------------------------------------------------------ open_GT 概览
pg = sorted([f for f in OPENG.iterdir() if f.is_dir()], key=lambda f: int(f.name))
p("## 1. `open_GT`（13 例）—— 目前唯一 Name=芯片型号 的标注集")
p()
p("| case | 元件 | 引脚 | 网络 | 图片尺寸 |")
p("|---|---:|---:|---:|---|")
openg = {}
for f in pg:
    j = next(f.glob("*.json"))
    d = json.loads(j.read_text(encoding="utf-8"))
    openg[f.name] = d
    from PIL import Image
    with Image.open(next(f.glob("*.png"))) as im:
        size = f"{im.width}×{im.height}"
    npin = sum(len(v) for v in d["pins"].values())
    p(f"| {f.name} | {len(d['components'])} | {npin} | {len(d['nets'])} | {size} |")
p()
p(f"合计：{sum(len(d['components']) for d in openg.values())} 元件 / "
  f"{sum(len(d['pins']) for d in openg.values())} 引脚 / "
  f"{sum(len(d['nets']) for d in openg.values())} 网络")
p()

# ------------------------------------------------------------ type 约定对比
def types_of(dict_cases):
    c = Counter()
    for d in dict_cases.values():
        for comp in d.get("components", {}).values():
            c[comp.get("type")] += 1
    return c

gt10 = {}
for folder in sorted((DS / "10_GTcase").iterdir()):
    if folder.is_dir():
        gt10[folder.name.split("-")[0]] = json.loads(next(folder.glob("*_target.json")).read_text(encoding="utf-8"))
train = {}
for folder in sorted((DS / "200_train_cases").iterdir()):
    if folder.is_dir():
        train[folder.name] = json.loads(next(folder.glob("*_target_named.json")).read_text(encoding="utf-8"))
ext1 = json.loads((ROOT / "case_001_扩展示例.json").read_text(encoding="utf-8"))

sets = {
    "open_GT(13)": types_of(openg),
    "10_GTcase(10)": types_of(gt10),
    "200_train(200)": types_of(train),
    "case_001扩展示例": types_of({"x": ext1}),
}
p("## 2. 四份资料里 `type` 的写法不一致（这是 ComponentF1 的隐性扣分点）")
p()
p("| 资料 | 类型数 | 取值示例 |")
p("|---|---:|---|")
for k, c in sets.items():
    p(f"| {k} | {len(c)} | " + ", ".join(f"`{x}`" for x, _ in c.most_common(12)) + " |")
p()
inter = set(sets["open_GT(13)"]) & set(sets["10_GTcase(10)"]) & set(sets["200_train(200)"])
p(f"- 三份数据共有的 type 只有 {len(inter)} 个：" + ", ".join(f"`{x}`" for x in sorted(inter)))
p(f"- open_GT 与 10GT 的并集 {len(set(sets['open_GT(13)'])|set(sets['10_GTcase(10)']))} 类，")
p(f"  再与 200 训练集并集 {len(set(sets['open_GT(13)'])|set(sets['10_GTcase(10)'])|set(sets['200_train(200)']))} 类。")
p("  官方 Q&A 说 `type_list` 是 34 类且评测不会超出——**该文件仍未出现在本文件夹**，")
p("  但上面这个并集给出了候选范围；`ic`/`resistor`/`vbus_in` 这类长名只出现在 `case_001扩展示例`，")
p("  与 10GT 的短枚举（`box`/`r`）互斥，混用会直接判错 type。")
p()

# ------------------------------------------------------------ 覆盖：真实型号 vs 三库
p("## 3. 真实型号（Name 字段）在三库中的可查率")
p()
for label, cases, keyfield in [("open_GT(13)", openg, "Name"), ("10_GTcase(10)", gt10, "Name")]:
    rows = []
    for case, d in cases.items():
        for k, comp in d["components"].items():
            if comp.get("type") not in ("box", "ic"):
                continue
            nm = comp.get(keyfield)
            hit = [n for n in libs if norm(nm) in norm_idx[n]]
            rows.append((case, k, nm, hit, comp.get("value")))
    p(f"### {label} 的 box/ic 元件（{len(rows)} 个）")
    p()
    p("| case | key | Name | 命中库 | value |")
    p("|---|---|---|---|---|")
    for case, k, nm, hit, val in rows:
        p(f"| {case} | {k} | `{nm}` | {' / '.join(hit) if hit else '**无**'} | {val} |")
    p()
    p(f"可查率：{sum(1 for r in rows if r[3])}/{len(rows)}")
    p()

# ------------------------------------------------------------ 引脚对齐
p("## 4. 库里的引脚与 GT 对齐情况（Name 能查到的 box 元件）")
p()
p("| case | key | Name | GT引脚 | 库引脚 | 编号完全一致 | 共同编号 | 名称完全一致 | 库中 `x_y` 形式编号 |")
p("|---|---|---|---:|---:|---|---|---|---|")
suffix = re.compile(r"^\d+_\d+$")
agg = Counter()
for cases, tag in [(openg, "open_GT"), (gt10, "10GT")]:
    for case, d in cases.items():
        for k, comp in d["components"].items():
            if comp.get("type") not in ("box", "ic"):
                continue
            nm = comp.get("Name")
            src = None
            for n in libs:
                if norm(nm) in norm_idx[n]:
                    src = norm_idx[n][norm(nm)][1]
                    break
            if not src:
                continue
            gpins = d.get("pins", {}).get(k, {})
            gt_num = {re.sub(r"^pin_", "", x) for x in gpins}
            gt_name = {(v.get("pinname") or "") for v in gpins.values()}
            lib_num = {str(x.get("number", "")) for x in src.get("pins", [])}
            lib_name = {str(x.get("name", "")) for x in src.get("pins", [])}
            weird = sum(1 for x in lib_num if suffix.match(x))
            same_n = gt_num == lib_num
            same_v = {x for x in gt_name if x} == {x for x in lib_name if x}
            agg["compared"] += 1
            agg["num_exact"] += same_n
            agg["name_exact"] += same_v
            p(f"| {tag}/{case} | {k} | `{nm}` | {len(gt_num)} | {len(lib_num)} | "
              f"{'Y' if same_n else 'N'} | {len(gt_num & lib_num)} | "
              f"{'Y' if same_v else 'N'} | {weird} |")
p()
p(f"编号集合**完全一致** {agg['num_exact']}/{agg['compared']}；"
  f"名称集合完全一致 {agg['name_exact']}/{agg['compared']}。")
p()
p("> 结论：库的 `pins` 是“多个同源符号实例”的聚合，条目数通常远多于物理引脚，")
p("> 且重复编号带 `_1/_2` 后缀。**不能直接把库引脚当输出**；只能按编号/名称做交集匹配后回填。")
p()

# ------------------------------------------------------------ 能回填什么
p("## 5. 可回填性小结（对当前 PinF1/NetF1 的口径）")
p()
p("- 10GT 与 open_GT 的 `pinname` 覆盖率都是 100%，378 / 全量引脚都带名字 → pinname 是可比字段。")
p("- 官方 Q&A 第 5/29 条：电源/地类 pin number 是**随机生成**，不参与评分；第 4 条：VCC/GND 直接按坐标匹配，")
p("  同名 GND/VCC 在打分时**合并为同一 net**。")
p("- 官方 Q&A 第 3 条（评测问题）：非 box 元件的 pinname/pinnumber **不参与评分**。")
p("  → 库的价值集中在 box/ic 的引脚名与编号上，其余器件只需 bbox+type。")
p()

(OUT / "official_analysis2.md").write_text("\n".join(L), encoding="utf-8")
print("written", len(L), "lines")
