# -*- coding: utf-8 -*-
"""分析官方资料：三个元器件库 + 10GT/200训练集标注的一致性。

只读，不修改任何原件；结果写入 _derive/ 下的 JSON / Markdown。
"""
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(r"C:\Users\Pc\Desktop\新建文件夹")
OFF = ROOT / "官方资料"
OUT = OFF / "_derive"
OUT.mkdir(exist_ok=True)

DS = ROOT / "赛题六公开数据集" / "赛题六公开数据集"
GT_DIR = DS / "10_GTcase"
TRAIN_DIR = DS / "200_train_cases"

L = []
def p(s=""):
    L.append(str(s))

def load(p_):
    return json.loads(Path(p_).read_text(encoding="utf-8"))


# ---------------------------------------------------------------- 1. 文件清单
p("# 官方资料自动分析")
p()
p("## 1. 文件清单与重复")
p()
files = [f for f in OFF.iterdir() if f.is_file() and not f.name.startswith("_qa_extract")]
def sha(f):
    h = hashlib.sha256()
    with open(f, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

ds_cpntl = DS / "cpntLibrary"
same = "相同" if sha(OFF / "cpntLibrary") == sha(ds_cpntl) else "不同"
p("| 文件 | 大小 | SHA256 前12 | 与公开数据集对比 |")
p("|---|---:|---|---|")
for f in sorted(files, key=lambda x: x.name):
    p(f"| {f.name} | {f.stat().st_size:,} | {sha(f)[:12]} | {same if f.name == 'cpntLibrary' else '-'} |")
p()

# ---------------------------------------------------------------- 2. 三个库
libs = {
    "BoxICLibrary.json": load(OFF / "BoxICLibrary.json"),
    "BoxICLibrary_200cases.json": load(OFF / "BoxICLibrary_200cases.json"),
    "cpntLibrary": load(OFF / "cpntLibrary"),
}
p("## 2. 三个元器件库的关系")
p()
p("| 库 | 条目数 | 每条字段 |")
p("|---|---:|---|")
for name, d in libs.items():
    fields = Counter(tuple(sorted(v.keys())) for v in d.values())
    fs = "; ".join(f"{'/'.join(k)} x{c}" for k, c in fields.most_common())
    p(f"| {name} | {len(d)} | {fs} |")
p()

sets = {k: set(v.keys()) for k, v in libs.items()}
bic, b200, cp = sets["BoxICLibrary.json"], sets["BoxICLibrary_200cases.json"], sets["cpntLibrary"]
p("| 组合 | 条目数 |")
p("|---|---:|")
p(f"| BoxICLibrary_200cases ∩ cpntLibrary | {len(b200 & cp)} |")
p(f"| BoxICLibrary_200cases − cpntLibrary | {len(b200 - cp)} |")
p(f"| cpntLibrary − BoxICLibrary_200cases | {len(cp - b200)} |")
p(f"| BoxICLibrary ∩ BoxICLibrary_200cases | {len(bic & b200)} |")
p(f"| BoxICLibrary − BoxICLibrary_200cases | {len(bic - b200)} |")
p()
p("`BoxICLibrary_200cases − cpntLibrary` 示例：" + ", ".join(sorted(b200 - cp)[:20]))
p()
p("`cpntLibrary − BoxICLibrary_200cases` 示例：" + ", ".join(sorted(cp - b200)[:20]))
p()

# 引脚号形态
suffix = re.compile(r"^\d+_\d+$")
def numstat(d):
    n_total = n_suffix = 0
    for v in d.values():
        for pin in v.get("pins", []):
            n_total += 1
            if suffix.match(str(pin.get("number", ""))):
                n_suffix += 1
    return n_total, n_suffix
p("| 库 | 引脚条目 | 形如 `1_1` 的编号 | 占比 |")
p("|---|---:|---:|---:|")
for name, d in libs.items():
    t, s = numstat(d)
    p(f"| {name} | {t:,} | {s:,} | {s/max(t,1):.1%} |")
p()

# ---------------------------------------------------------------- 3. 标注 type 分布
gt_cases = {}
for folder in sorted(GT_DIR.iterdir()):
    if not folder.is_dir():
        continue
    j = next(folder.glob("*_target.json"))
    gt_cases[folder.name.split("-")[0]] = load(j)
train_cases = {}
for folder in sorted(TRAIN_DIR.iterdir()):
    if not folder.is_dir():
        continue
    j = next(folder.glob("*_target_named.json"))
    train_cases[folder.name] = load(j)

def types_of(cases):
    c = Counter()
    for d in cases.values():
        for comp in d.get("components", {}).values():
            c[comp.get("type")] += 1
    return c

gt_types, tr_types = types_of(gt_cases), types_of(train_cases)
p("## 3. `type` 取值：10GT 与 200 训练集")
p()
p(f"- 10GT：{len(gt_types)} 类 —— " + ", ".join(f"`{k}`×{v}" for k, v in gt_types.most_common()))
p(f"- 200 训练集：{len(tr_types)} 类")
p(f"- 训练集有、10GT 没有：{sorted(set(tr_types)-set(gt_types))}")
p(f"- 10GT 有、训练集没有：{sorted(set(gt_types)-set(tr_types))}")
p()
p("> 官方 Q&A 第3/4/14/17条：以官方 `type_list` 为准（34 类），评测不会超出该表；")
p("> 该文件**不在本文件夹中**，需要另外索取。10GT 是当前唯一能直接观测到的官方 `type` 样本。")
p()

# ---------------------------------------------------------------- 4. GT 元件与库的覆盖
p("## 4. 10GT 元件能否在库中查到（按 Name 精确匹配）")
p()
p("| case | 元件 | type | Name | value | 在 cpntLibrary | 在 BoxICLibrary_200cases | 在 BoxICLibrary |")
p("|---|---|---|---|---|---|---|---|")
gt_comp = []
for case, d in gt_cases.items():
    for key, comp in d["components"].items():
        nm = comp.get("Name") or ""
        row = [
            case, key, comp.get("type"), nm, comp.get("value"),
            "Y" if nm in cp else "",
            "Y" if nm in b200 else "",
            "Y" if nm in bic else "",
        ]
        gt_comp.append(row)
        if comp.get("type") in ("ic", "box") or (nm in cp or nm in b200 or nm in bic):
            p("| " + " | ".join("" if x is None else str(x) for x in row) + " |")
p()
p(f"10GT 元件总数 {len(gt_comp)}；Name 命中 cpntLibrary 的 "
  f"{sum(1 for r in gt_comp if r[5])}，命中 BoxICLibrary_200cases 的 "
  f"{sum(1 for r in gt_comp if r[6])}，命中 BoxICLibrary 的 {sum(1 for r in gt_comp if r[7])}。")
p()

# 关键：Q&A 提到的两个缺失 IC
for probe in ["STMPS2141STR", "MP2236CGJ-Z"]:
    p(f"- `{probe}`：cpntLibrary={'有' if probe in cp else '无'}，"
      f"BoxICLibrary_200cases={'有' if probe in b200 else '无'}，"
      f"BoxICLibrary={'有' if probe in bic else '无'}")
p()

# ---------------------------------------------------------------- 5. IC 引脚号/名与 GT 对照
p("## 5. box 类元件的引脚：库里查到的编号/名称 vs GT 实际")
p()
p("| case | 元件 | GT type | Name | GT引脚数 | 库引脚数 | 编号集合是否一致 | 名称集合是否一致 | 差异示例 |")
p("|---|---|---|---|---:|---:|---|---|---|")
num_ok = num_tot = 0
for case, d in gt_cases.items():
    for key, comp in d["components"].items():
        nm = comp.get("Name") or ""
        pins = d.get("pins", {}).get(key, {})
        if not pins:
            continue
        if not (nm in bic or nm in b200 or nm in cp):
            continue
        src = (libs["BoxICLibrary.json"].get(nm)
               or libs["BoxICLibrary_200cases.json"].get(nm)
               or libs["cpntLibrary"].get(nm))
        gt_nums = {re.sub(r"^pin_", "", k) for k in pins}
        gt_names = {(v.get("pinname") or "") for v in pins.values()}
        lib_nums = {str(x.get("number", "")) for x in src.get("pins", [])}
        lib_names = {str(x.get("name", "")) for x in src.get("pins", [])}
        n_ok = bool(gt_nums & lib_nums)
        nm_ok = bool({x for x in gt_names if x} & lib_names)
        num_tot += 1
        num_ok += 1 if gt_nums == lib_nums else 0
        example = ""
        if gt_nums != lib_nums:
            example = f"GT缺 {sorted(lib_nums-gt_nums)[:4]} / 库缺 {sorted(gt_nums-lib_nums)[:4]}"
        p(f"| {case} | {key} | {comp.get('type')} | {nm} | {len(gt_nums)} | {len(lib_nums)} | "
          f"{'Y' if gt_nums == lib_nums else 'N'} | {'Y' if nm_ok else 'N'} | {example} |")
p()
p(f"编号集合完全相同：{num_ok}/{num_tot}。库的 `number` 是同名符号多实例的**聚合**，")
p("出现重复编号时会加 `_1/_2` 后缀（见第2节统计），直接当 `pin_<number>` 输出会造出 GT 里不存在的键。")
p()

# ---------------------------------------------------------------- 6. 引脚编号能否从库恢复的关键结论
p("## 6. 200 训练集里 GT 引脚名的可观测性（供 OCR/库回填评估）")
p()
tot_pins = named = 0
for d in train_cases.values():
    for pins in d.get("pins", {}).values():
        for v in pins.values():
            tot_pins += 1
            if v.get("pinname"):
                named += 1
p(f"- 200 训练集引脚总数 {tot_pins:,}，其中 `pinname` 非空 {named:,}（{named/max(tot_pins,1):.1%}）")
gt_tot = gt_named = 0
for d in gt_cases.values():
    for pins in d.get("pins", {}).values():
        for v in pins.values():
            gt_tot += 1
            if v.get("pinname"):
                gt_named += 1
p(f"- 10GT 引脚总数 {gt_tot:,}，其中 `pinname` 非空 {gt_named:,}（{gt_named/max(gt_tot,1):.1%}）")
p()

# ---------------------------------------------------------------- 7. 输出 JSON
summary = {
    "libs": {k: len(v) for k, v in libs.items()},
    "lib_overlap": {
        "b200_and_cpntLibrary": len(b200 & cp),
        "b200_minus_cpntLibrary": sorted(b200 - cp),
        "cpntLibrary_minus_b200": sorted(cp - b200),
        "boxic_and_b200": len(bic & b200),
        "boxic_minus_b200": sorted(bic - b200),
    },
    "gt_types": dict(gt_types),
    "train_types": dict(tr_types),
    "gt_components": gt_comp,
    "pin_number_coverage": {"exact_match_cases": num_ok, "compared": num_tot},
    "pinname_filled": {"train": [named, tot_pins], "gt": [gt_named, gt_tot]},
}
(OUT / "official_analysis.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
(OUT / "official_analysis.md").write_text("\n".join(L), encoding="utf-8")
print("written", OUT / "official_analysis.md", len(L), "lines")
