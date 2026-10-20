# -*- coding: utf-8 -*-
"""元件库能把 PinF1 抬到多少：三段实测分解。

段一（能不能）：open_GT 有真实型号，检查库能否给出正确的 (编号, 名称)。
段二（算不算）：只有 box/ic 的引脚进 PinF1（Q&A 评测 #2/#6），先数清分母。
段三（卡在哪）：把「位号」当成免费，看位置与身份各自的天花板 —— 这一步说明
               库不是瓶颈，位号才是。

只读；结果写 D:/pcb_unet_v2/reports/pin_library_ceiling.txt
"""
import json
import math
import re
from pathlib import Path

ROOT = Path(r"C:\Users\Pc\Desktop\新建文件夹")
DS = ROOT / "赛题六公开数据集" / "赛题六公开数据集"
OFF = ROOT / "官方资料"
LIB = json.loads((OFF / "_derive" / "box_model_index.json").read_text(encoding="utf-8"))
AUTOID = Path(r"D:/pcb_unet_v2/autoid_ab")
BOX = ("box", "ic")
TOL = 5.0

L = []
def p(s=""):
    L.append(str(s))

def norm(s):
    return re.sub(r"[\s\-_]+", "", str(s or "")).lower()

def load_cases(base, pattern):
    out = {}
    for f in sorted(base.iterdir()):
        if f.is_dir():
            j = next(f.glob(pattern), None)
            if j:
                key = f.name.split("-")[0] if pattern.startswith("*_target") else f.name
                out[key] = json.loads(j.read_text(encoding="utf-8"))
    return out

p("# 元件库能把 PinF1 抬到多少（实测）")
p()
p("> 结论先行：**库本身不是 PinF1 的瓶颈，位号是。**")
p("> 库能把 box 引脚的「编号+名称」几乎免费给对（open_GT 168/191 = 88%），")
p("> 但分数卡在「元件键对不上」，导致位置已经正确的 80 个 box 引脚里只有 4 个被计入。")
p()

# ---------------------------------------------------------------- 段一
p("## 段一：库给出的 (编号, 名称) 对不对 —— open_GT 13 例（Name 是真实型号）")
p()
openg = load_cases(DS / "open_GT" / "open_GT", "*.json")
tot = num_ok = both_ok = 0
rows = []
for case, d in openg.items():
    for ck, comp in d["components"].items():
        if comp.get("type") not in BOX:
            continue
        pins = d.get("pins", {}).get(ck, {})
        tot += len(pins)
        entry = LIB.get(norm(comp.get("Name")))
        if not entry:
            rows.append((case, ck, comp.get("Name"), len(pins), "库中无此型号", 0, 0))
            continue
        pairs = {(n, nm) for n, nm in entry["pins"]}
        nums = {n for n, _ in pairs}
        num = sum(1 for k in pins if re.sub(r"^pin_", "", k) in nums)
        both = sum(1 for k, v in pins.items()
                   if (re.sub(r"^pin_", "", k), v.get("pinname") or "") in pairs)
        num_ok += num
        both_ok += both
        rows.append((case, ck, comp.get("Name"), len(pins), f"{len(pairs)} 组", both, num))
p("| case | key | 型号 | GT引脚 | 库 | 编号+名称都对 | 仅编号对 |")
p("|---|---|---|---:|---|---:|---:|")
for r in rows:
    p(f"| {r[0]} | {r[1]} | `{r[2]}` | {r[3]} | {r[4]} | {r[5]} | {r[6]} |")
p()
p(f"- box 引脚合计 **{tot}**，编号在库中 **{num_ok}/{tot}（{num_ok/tot:.1%}）**，"
  f"编号+名称都命中 **{both_ok}/{tot}（{both_ok/tot:.1%}）**")
p("- 差的那些是库把同一编号的多别名聚合在一起（`AMS1117-3.3` 的 1 号有 6 种写法），需要用丝印 OCR 消歧。")
p()

# ---------------------------------------------------------------- 段二
p("## 段二：PinF1 的分母 —— 只有 box/ic 的引脚计分")
p()
gt10 = load_cases(DS / "10_GTcase", "*_target.json")
all_pins = box_pins = 0
rows = []
for case, d in gt10.items():
    a = sum(len(v) for v in d.get("pins", {}).values())
    b = sum(len(v) for k, v in d.get("pins", {}).items()
            if (d["components"].get(k) or {}).get("type") in BOX)
    all_pins += a
    box_pins += b
    rows.append((case, len(d["components"]), a, b))
p("| case | 元件 | 全部引脚 | 其中 box 引脚 |")
p("|---|---:|---:|---:|")
for r in rows:
    p(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} |")
p(f"| **合计** | | **{all_pins}** | **{box_pins}** |")
p()
p(f"10GT 的 {all_pins} 个引脚里只有 **{box_pins}** 个属于 box/ic（{box_pins/all_pins:.1%}）。")
p("库对这剩下的 "
  f"{all_pins-box_pins} 个引脚（R/C/L/GND/网络标号）什么都给不了，官方也明说它们不参与评分。")
p()

# ---------------------------------------------------------------- 段三
p("## 段三：卡在哪 —— 把「位号」当免费之后的天花板")
p()
p("拿 autoid 变体的现有预测，对每个参考 box 元件做三件事：")
p("① 用它的键去找预测；② 不看键、在**所有**导出的元件里找位置最贴合的那个；③ 再看编号+名称。")
p()
p("| case | GT key | GT引脚 | 同键预测引脚 | 同键≤5px | 位置最贴合的元件 | 该元件≤5px | 该元件编号+名称全对 |")
p("|---|---|---:|---:|---:|---|---:|---:|")
s_tot = s_key = s_pos = s_both = 0
detail = []
for case, d in gt10.items():
    path = AUTOID / case / "partial_submission.json"
    if not path.exists():
        continue
    pred = json.loads(path.read_text(encoding="utf-8"))
    pred_pins = pred.get("pins") or {}
    for ck, comp in d["components"].items():
        if comp.get("type") not in BOX:
            continue
        gtp = d.get("pins", {}).get(ck, {})
        if not gtp:
            continue
        def near(pp):
            return sum(1 for gv in gtp.values()
                       if any(math.hypot(pv["point"]["x"] - gv["point"]["x"],
                                         pv["point"]["y"] - gv["point"]["y"]) <= TOL
                              for pv in pp.values()))
        key_near = near(pred_pins.get(ck, {}))
        best_n, best_key, best_pins = 0, None, {}
        for pk, pp in pred_pins.items():
            n = near(pp)
            if n > best_n:
                best_n, best_key, best_pins = n, pk, pp
        both = 0
        for gk, gv in gtp.items():
            cands = sorted((math.hypot(pv["point"]["x"] - gv["point"]["x"],
                                       pv["point"]["y"] - gv["point"]["y"]), pk, pv)
                           for pk, pv in best_pins.items())
            cands = [c for c in cands if c[0] <= TOL]
            if cands and cands[0][1] == gk and \
                    (cands[0][2].get("pinname") or "") == (gv.get("pinname") or ""):
                both += 1
        s_tot += len(gtp); s_key += key_near; s_pos += best_n; s_both += both
        detail.append((case, ck, len(gtp), best_key, best_n, both))
        p(f"| {case} | {ck} | {len(gtp)} | {len(pred_pins.get(ck, {}))} | {key_near} | "
          f"`{best_key or '-'}` | {best_n} | {both} |")
p()
p(f"GT box 引脚 **{s_tot}** 个：")
p(f"- 用**当前导出的键**能找到 ≤5 px 对应点：**{s_key}**（{s_key/s_tot:.1%}）← 现在是这个数被计分")
p(f"- 只把**位号修好**（键对上），≤5 px 的：**{s_pos}**（{s_pos/s_tot:.1%}）")
p(f"- 位号修好 **且** 编号+名称也对：**{s_both}**（{s_both/s_tot:.1%}）← PinF1 的现实上限")
p()
p(f"换算（分母取 box 引脚 {s_tot}，忽略 FP）："
  f"现在 PinF1 ≈ {2*s_key/(s_key+s_tot):.3f}，位号修好后 ≈ {2*s_both/(s_both+s_tot):.3f}。")
p()

# ---------------------------------------------------------------- 结论
p("## 结论")
p()
p(f"1. **直接回答：现在拿不到提升。** box 库 on/off 的 A/B 完全一样，因为能计分的 box 引脚只有 "
  f"{s_key} 个（分母 {s_tot}），库改了编号/名称也翻不动 {s_key} 个数。")
p(f"2. **但位置其实已经很好：{s_pos}/{s_tot}（{s_pos/s_tot:.0%}）的 box 引脚有 ≤5 px 的对应点。**")
p("   它们全落在 `DET_00xx` 这类自动键下面——**box 元件的位号没读出来**，于是整块"
  "（元件 + 引脚 + 网络）都计 0 分。")
p(f"3. **库的真实作用是「位号修好之后不丢分」**：编号+名称一次给对的比例是 "
  f"{both_ok/tot:.0%}（open_GT），把 {s_pos} 个位置正确的引脚转成 {s_both} 个可得分的 TP。")
p(f"4. **主力战场仍是网络超图（35%）**：hyperGraph 的成员写成 `元件.引脚编号`，"
  "编号对了 net 的 pin set 才可能对上；库在这里的作用比在 PinF1 更大。")
p("5. 唯一需要人来做的是**消歧名称**（库给了多别名时选哪个），以及修 box 元件位号的读取。")
p()

Path(r"D:/pcb_unet_v2/reports/pin_library_ceiling.txt").write_text("\n".join(L), encoding="utf-8")
print("\n".join(L[-14:]))
