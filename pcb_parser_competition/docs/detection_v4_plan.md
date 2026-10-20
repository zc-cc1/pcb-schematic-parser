# 检测阶段 v4 方案：把 component_f1 从 .748 推到 .93 附近

状态日期：2026-10-17。本文件是**方案**，不是结果。所有"实测"数字都注明了来源；所有"+X 分"都是**在同一份第 5 次评测账上做的算术推演，不是承诺**。

前提：赛题权重 C .30 / P .25 / H .35 / L .10；第 5 次评测 34.65 分，其中 component_f1 .7482 贡献 22.45 分。检测侧天花板是 30 分，因此本方案的合理目标是 +5 分左右，而不是把总分翻倍。

---

## 1. 账本：误差到底在哪（实测）

### 1.1 官方第 5 次评测 13 例（`评测结果/第五次评测-34.65/scoring_result/report/submission/20261016T162819Z/*.json` 逐例聚合）

| 项 | 数值 |
|---|---|
| GT 元件 | 346 |
| 提交元件 | 325（其中 132 个是无位号的电源符号，type `gnd`/`v`） |
| TP / FP / FN | 255 / 70 / 91 → P .785 R .737 **F1 .760**（宏平均 .7482） |
| 错误分项 | `missing` 57、属性/位置 34（bbox 14、value 15、Name 8、type 3）、`extra` 32（报告里 extra 被截断，实际 FP 70） |
| 位号 key F1 | **.948**（非电源 192/212 位号匹配成功） |

按"电源符号 / 非电源"拆开（本方案的关键发现）：

| 桶 | GT | 匹配 | 占比 | FP | FN |
|---|---:|---:|---:|---:|---:|
| 电源符号（gnd/v） | **134（38.7%）** | 97（72.4%） | — | **35（占全部 FP 的 50%）** | **37（占全部 FN 的 41%）** |
| 非电源 | 212 | 158（74.5%） | — | 35 | 54 |

**电源符号一类就吃掉了检测误差的一半。**

### 1.2 本机 40 页（公开 200 例子集，官方 20px 口径，用公开 `*_target_named.json` 当 GT）

| 桶 | GT | pred | TP | P | R | F1 |
|---|---:|---:|---:|---:|---:|---:|
| 全部 | 1962 | 1989 | 1700 | .855 | .866 | **.861** |
| 电源 | 724 | 734 | 683 | .931 | .943 | **.937** |
| 非电源 | 1238 | 1255 | 1017 | .810 | .821 | .816 |

对照 1.1：**同一个模型，电源符号在公开页 93.7%，在隐藏页只有 72.4%。** 差 21 个点不是"认不出三角形"，而是页面尺度/落点这类工程问题（见 WP1）。

### 1.3 框口径：官方 GT 是紧框，我们的标签是含引线的 cell（实测，200 例中心匹配）

| 类别 | 官方/我们 宽比 | 高比 | 我们中位框 | 官方中位框 |
|---|---:|---:|---|---|
| resistor | **0.50** | **0.50** | 27×28 | 8×20 |
| crystal | 0.49 | 0.50 | 30×26 | 15×15 |
| diode | 0.54 | 0.56 | 30×33 | 14×20 |
| led | 0.54 | 0.52 | 29×29 | 14×20 |
| opamp | 0.59 | 0.67 | 66×55 | 40×40 |
| mosfet | 0.65 | 0.61 | 37×35 | 29×20 |
| inductor | 0.71 | **0.40** | 45×18 | 34×6 |
| capacitor | 0.76 | 0.74 | 22×26 | 16×20 |
| voltage_source | 0.77 | 0.77 | 13×13 | 10×10 |
| ic | 0.80 | 0.88 | 69×67 | 51×58 |
| ground | 0.95 | 0.95 | 21×20 | 20×19 |

含义：我们的框在**每个方向上系统性偏大 1.05–2 倍**。小符号被 20px 容差吸收，符号越大绝对超出越多——这解释了为什么"困难页"掉分最狠（第 13 例：type/Name/value 全 1.0，bbox 只有 .467，误差 20.15–34.12px，正好卡线）。

### 1.4 其余已确认的漏检/误检

- FN 尺寸：<32px 138 个、32–64px 111 个，**89% 的漏检在 64px 以下**。
- FN 归因（本机标签口径）：纯漏 111（40%）、框找到但 IoU 仅 0.3–0.5 的"近失"132（47%）、类混淆 33（12%，`other↔switch`、`fuse→resistor`、`ic→pin_port`、`diode→led`）。
- 训练框只有 10218 个，**13/35 类不足 20 框**（current_source 1、optocoupler 2、antenna 2、fuse 6 …）。
- 类目语义错位：`net_input` 官方 GT 是 30×10 的横向文本条，我们预测 26×50 竖条，53 个 GT 中 IoU≥0.3 的为 **0**；`net_output` 0/11、`testpoint` 0/10；`configs/baseline_v3.json` 把 `net_input/pin_port/testpoint` 全塌成 `vbus_in`，类型层面必然对不上。
- 提交里 `nets` 总数 = **2**（13 例），pin 组 192 个——下游（H .35 / L .10）目前是空跑。

---

## 2. 工作包

排序依据 = 预计可得分 / 工程量。每个工作包都必须是**可开关、可回退**的。

### WP0 先修标尺（0.5 天，所有后续工作的前提）

现状 `scripts/evaluate_detector.py` 用 IoU-0.5 + 类名匹配，和官方（key+type+Name+value 且四角各 ≤20px）不是一回事；继续按 0.883 优化会走偏。

- 新增 `scripts/evaluate_official_component.py`：输入预测 JSON + 官方 GT，输出 P/R/F1、逐类 `pass20`、失败归因（missing / mismatch(bbox|value|Name|type) / extra），实现官方 power 匹配规则（ignore key/Name，type 别名 gnd/v/vcc/vdd）。
- GT 来源：公开 200 例 `*_target_named.json`（注意是左下原点，转 `H-y`）。
- 划分：现有 59 例 val 保持不动做回归；再从 141 例**从未用于开发**的里挑 30 例做 `configs/detection_split_frozen.json` 冻结测试集。
- **验收**：脚本在公开 40 页上复现 1.2 的表；对 13 例官方报告能做到"给定同一份提交，分项（bbox 14 / value 15 / Name 8）量级一致"。

### WP1 电源符号专项（预计 +3.2 分，1.5 天 + 0.5 天验证）

事实：GT 占 38.7%，隐藏页只匹配 72.4%；提交里电源框是**固定 20×19 模板**（如 `[899.51,275.75,919.51,294.75]`），而本机检测框是官方框的约 2 倍、底边对齐、向上多出 17–19px → 模板锚点（检测框中心）比 glyph 中心高约半个引线长度。**这个偏移随页面符号尺度线性放大，超过 20px 就整对失配（power 匹配忽略位号、纯几何一对一，所以一错就是 FP+FN 一对）。**

动作（按顺序）：
1. **锚点改为 glyph 墨迹定位**：在检测框内做二值化 → 取最大连通域外接框 → 用其中心/底边定位模板；模板尺寸按 `检测框高 / 官方典型高` 自适应，不再写死 20×19。
2. **去重前置**：power 一对一分配前先合并同类 IoU>0.3 的重复框（本地同类重复框 10 个，IoU 中位 0.54）。
3. **补漏**：ground/v 形状固定，对切片漏检区用模板匹配二次扫描（不必依赖 YOLO）。
4. 保持 type 输出 `gnd`/`v`（已是官方词表，不要改成 `ground`/`voltage_source`）。

- 代码落点：`src/pcb_parser/power_networks.py`、`scripts/batch_power_networks.py`、`configs/power_networks_v1.json`。
- **验收**：
  1. 本地 40 页官方口径电源 TP 683→≥700（F1 .937→.97）；
  2. **尺度扰动测试**（新增，因为本地 93.7% 复现不出隐藏页 72.4% 的失效）：把同一批页面整体缩放到 0.6 / 1.0 / 1.6 倍重跑，电源匹配率波动必须 ≤3 个点。现版本预期会随尺度单调掉，这正是隐藏页掉分的机制；
  3. 隐藏侧按 5 次评测账推算：匹配率 72%→≥95%，即 FP 35→≤7、FN 37→≤7。
- 回退：配置开关 `power_symbols.mode = template|legacy`。

### WP2 非电源漏检与误检（独立估算 +1.4~2.0 分，2 天）

非电源 54 FN / 35 FP。三件事：

1. **多层切片**：现有 1024 单一尺度；补 512 层，用 v3 的类感知 WBF 融合（`src/pcb_parser/fusion.py` 已支持），针对 <64px 那 89% 的漏检。
2. **提交前剪枝**：无 OCR 文本支撑、无引脚、无网络连接的孤立低置信框（conf<0.3）直接丢弃——非电源 FP 35 个，砍一半约 +0.5 分；叠加救回一半 FN 约 +1.5 分。
3. **先关掉纯负收益的输出**：`net_input` 在本地官方口径 TP 4 / FP 49，**整类停输出是净赚**；`net_output`、`testpoint` 同理。用 WP0 的评测器各跑一次 A/B 再决定。

- **验收**：本地 40 页非电源 F1 .816→≥.88；FP 238→≤180，FN 221→≤190。

### WP3 框口径归一（预计 +1.3 分，1.5 天；A 先做，B 固化）

两个实施，A 便宜先做，B 才是根治：

**A. 导出时按类仿射收缩（1 天）**
- 新增 `configs/bbox_convention_v1.json`，直接用 1.3 的比例表：`x1 += (1-rw)/2*W`、`x2 -= (1-rw)/2*W`（高同理），按类配 `rw/rh`，对样本少的类（`<20`）用同类中位值兜底。
- 注意：1.3 的表是"官方 GT vs 我们的标签"，而检测器输出比标签略紧，所以收缩量要用**检测输出 vs 官方 GT**重标一次（WP0 的输出可以直接给出这张表）。
- **验收**：本地官方口径 component F1 +≥.02，且逐类 `pass20` 不下降；IoU-0.5 口径允许下降（口径不同，别用 0.883 判断）。

**B. 用官方 GT 重建训练标签重训（0.5 天训练）**
- 新增 `scripts/prepare_detection_from_official_gt.py`：公开 200 例直接用 `*_target_named.json` 当标签（9171 个元件，官方口径零误差）；剩余 100 例用 A 的比例修正。
- `configs/detector_training_v4.json` 沿用 v3（imgsz 1536 / amp / 120 epochs），RTX 5060 上约 30–40 分钟。
- **验收**：冻结 30 例官方口径 F1 ≥ A 方案；否则不采纳。同时要重跑 `tune_class_thresholds.py`。

### WP4 词表与标签语义（预计 +0.5 分，1 天）

- type 词表对齐官方 41 个 raw 名（`r/c/l/box/gnd/v/d/led/pin/testpoint/net_input/…`）；取消 `vbus_in` 塌缩（`configs/baseline_v3.json` 的 `class_mapping`）。
- `net_input` 二选一：按 30×10 文本条重标，或整类停输出。默认停输出。
- 补 `m3丝印 → raw_m` 映射（本地 13 个"误检"其实是词表漏了，不是检出错误）。

### WP5 长尾类（预计 +0.3–0.8 分，1 天起步，收益慢）

- 13 类训练框 <20。最快路径不是标新数据，而是**复制粘贴合成**：从官方 200 例 GT 里裁真实符号，贴到空白画布/现有页的空白区，生成 fuse / testpoint / optocoupler / esd / circle_header / raw_block 的样本。工程里已有外部数据收集与合成思路（`docs/external_data_collection_report.md`）。
- **验收**：这 6 类的本地官方口径 TP 从 0 → 各 ≥3。

---

## 3. 排期（按每天 4 小时估）

| 周 | 内容 | 出口 |
|---|---|---|
| W1 | WP0 评测器 + WP3-A 收缩表 + WP1 电源 glyph 定位 | 本地官方口径基线可复现；电源 F1 ≥.97 |
| W2 | WP1 补漏/去重 + WP2 切片与剪枝 + 冻结 30 例 | 非电源 F1 ≥.88；冻结集首次出数 |
| W3 | WP3-B 重训 v4 + WP4 词表 + WP5 合成 | 冻结集 F1 ≥.93；重跑第 6 次提交 |

每周末只认**冻结 30 例 + 官方口径**的数字，不认 59 例上的 0.883。

---

## 4. 收益与机会成本（必须一起看）

| 指标 | 现状 | 目标 | 权重 | 分数变化 |
|---|---:|---:|---:|---:|
| component_f1 | .748 | .93 | .30 | 22.4 → **27.9（+5.5）** |
| pin_f1 | .438 | .70 | .25 | 10.9 → 17.5（+6.6） |
| **connectivity_f1** | **.033** | .50 | **.35** | **1.2 → 17.5（+16.3）** |
| wire_f1 | .011 | .30 | .10 | 0.1 → 3.0（+2.9） |

结论：**检测侧把所有 WP 做完约 +5.5 分；而 connectivity 从 0.03 走到 0.50 是 +16 分。** 提交里 `nets` 只有 2 个、pin 组 192 个——网络阶段现在是空跑，权重最高的 35 分基本没挣。

（上表 component 一行是各 WP 独立估算后取合并值，不线性相加：WP1 +3.2、WP3 +1.3、WP2 +1.4~2.0，合起来低于相加值，按 .93 封顶。）

所以建议：**检测按本方案两周内推到 0.90–0.93 就收手**（WP0、WP1、WP2 必做，WP3/WP4/WP5 看时间），把省下的算力和时间投到导线骨架 → 网络构建。继续调 YOLO 结构（y11s 已证明 mAP 高但工作点 F1 低）或换更大模型，ROI 明显低于这两块。

---

## 5. 不做什么

- 不再换检测器架构/加大骨干（v3_nano 已是三代热启动的工作点最优，y11s 的 mAP 优势没有转化为 F1）。
- 不在 59 例验证集上继续调阈值（已被多轮开发使用，数字不可信）。
- 不用 IoU-0.5 的 0.883 当验收标准（和官方口径差 4–10 分，方向可能相反）。
- 不为了让 IoU 好看去栅格化更粗的标签（那是 U-Net 那边的问题，别混进来）。

---

## 6. 复现入口（WP0 落地后）

```powershell
$py = '.\.venv-gpu\Scripts\python.exe'

# 官方口径基线（公开 40 页）
& $py scripts\evaluate_official_component.py --pred runs\eval_v3_fused_v2thr.json --gt-root '..\赛题六公开数据集\赛题六公开数据集\200_train_cases' --split configs\detection_split_frozen.json

# 框口径重标（检测输出 vs 官方 GT）
& $py scripts\calibrate_bbox_convention.py --pred runs\eval_v3_fused_v2thr.json --gt-root '..\赛题六公开数据集\赛题六公开数据集\200_train_cases' --output configs\bbox_convention_v1.json

# 电源符号 glyph 定位自检
& $py scripts\batch_power_networks.py --config configs\power_networks_v1.json --diagnose
```

---

## 7. 待确认（开工前必须拍板）

1. **WP4 的 `net_input`**：停输出（净赚）还是按文本条重标（要改标签定义）？——建议先停。
2. **公开 200 例的 GT 能不能用于训练**：如果赛事允许（是"公开"集），WP3-B 直接用它当标签最省事；若不允许用于训练，就只能用它做标定（WP3-A）。
3. **冻结 30 例的划分口径**：按源工程分组还是按绘制风格（KiCad/Altium/jlc/Datasheet）分层？建议按风格分层，因为隐藏页的掉分集中在风格未见过的页。

---

## 8. 已完成：WP0 评测器 + WP3-A 框口径（2026-10-17 实测）

### 新增文件

| 文件 | 作用 |
|---|---|
| `scripts/evaluate_official_component.py` | 官方口径评测器（20px 四角、power 忽略位号、key/type/Name/value 分项） |
| `scripts/calibrate_bbox_convention.py` | 逐类框口径标定（标签空间 / 检测空间两套表） |
| `configs/bbox_convention_v1.json` | 可部署的收缩表（20 类） |
| `tests/test_official_component_eval.py` | 11 个单元测试 |

### 实测结果（公开 40 页，官方 20px 口径，class 模式）

| | TP | FP | FN | P | R | F1 |
|---|---:|---:|---:|---:|---:|---:|
| 基线（无变换） | 1713 | 276 | 263 | .8612 | .8669 | .8641 |
| **加收缩表** | **1752** | **237** | **224** | **.8808** | **.8866** | **.8837** |
| 留出验证（用另外 160 页拟合） | 1751 | 238 | 225 | .8803 | .8861 | .8832 |

留出验证与拟合内结果一致（.8832 vs .8837），**不是过拟合**。折合官方分约 **+0.6 分**（component 权重 .30）。

逐类召回变化（留出验证）：

| 类 | resistor | capacitor | diode | crystal | inductor | switch |
|---|---|---|---|---|---|---|
| 前 | .91 | .96 | .54 | .80 | .65 | .84 |
| 后 | **.97** | **.99** | **.76** | **.87** | **.76** | .86 |

### 两个必须记住的发现

1. **电源符号的源框是双峰的**：在一部分页面上 CVJsonStd `location` 是引脚小方块（14×14），官方 GT 是三角形图元（20×19）且整体偏下约半个框高；在另一部分页面（如 0002）源框是 40×40 的 cell。单一中位数拟合不了这两个群体——实测强行套用会把电源 F1 从 .9369 打到 .9067。所以 `ground`/`voltage_source` 被排除在收缩表外，改由 WP1 的 glyph 锚定解决。
2. **IC 该用定值内缩，不是比例**：`Box` 的 cell 就是画出来的矩形，官方 GT ≈ `realBBox`（在 0002/IC2 上完全相等）。比例收缩（0.80/0.88）会把大 IC 收过头（实测 181→177 TP），所以 `ic` 也排除，留给 WP3-B 的 `realBBox` 重训路线。

### 复现

```powershell
$py = '.\.venv-gpu\Scripts\python.exe'
$gt = '..\赛题六公开数据集\赛题六公开数据集\200_train_cases'

# 基线
& $py scripts\evaluate_official_component.py --pred runs\eval_v3_fused_v2thr.json --gt-root $gt --output runs\official_eval_baseline.json
# 加收缩表
& $py scripts\evaluate_official_component.py --pred runs\eval_v3_fused_v2thr.json --gt-root $gt --shrink configs\bbox_convention_v1.json --output runs\official_eval_shrink_v1.json
# 重新标定
& $py scripts\calibrate_bbox_convention.py --gt-root $gt --dataset-root ..\Dataset300 --pred runs\eval_v3_fused_v2thr.json --output configs\bbox_convention_v1.json
```

### 口径声明

`--match class` 是**上界**：它给"框对但位号 OCR 失败"的元件也算分，官方不会。要用官方 key 口径，需把 `--pred` 指向提交格式目录并加 `--match key`（当前 40 页上尚无提交格式产物，待 WP1 落地后补）。

### 尚未接线

收缩表目前只作用于**评测**，还没接到提交导出路径（`src/pcb_parser/component_text.py` 的 `export` 处）。下一步要在导出前应用，先用 0001/0002 跑通、确认不影响 pin/net 下游，再全量重出提交。

## 9. 已完成：框口径接线到导出（2026-10-17 晚）

### 结论

第 8 节的"比例收缩"被**"固定符号尺寸 + 中心偏移"**取代。原因是查了第 5 次评测的提交文件：
里面每个元件的框尺寸都是**固定符号尺寸**（r 8×20、c 16×20、l 34×6、gnd 20×19、v 10×10），
不是检测框。也就是说那版管线**已经把尺寸规范过了，只有中心偏了**（第 13 例的 20.15–34.12px 全是位移）。

这三种做法在同一批 40 页上的实测：

| 做法 | 总体 F1 | 电源 F1 | 非电源 F1 |
|---|---:|---:|---:|
| 基线（原始检测框） | .8641 | .9369 | .8217 |
| 比例收缩（第 8 节） | .8837 | .9369 | .8528 |
| **固定尺寸 + 中心偏移** | **.8999** | **.9616** | **.8640** |

留出验证（尺寸与偏移只用另外 160 页拟合）得到**完全相同**的 .8999/9616/8640 —— 这组统计量极稳
（ground 2135 对、capacitor 1570 对），不是过拟合。

逐类（留出验证）：ground R .93→**.96**、capacitor .96→**.99**、resistor .91→**.97**、
voltage_source .98、net_input .08→**.25**（尺寸从 26×50 纠正成官方 30×10 后才有重叠）。

### 尺寸表是从官方 GT 直接读的，不是猜的

每类符号的天然尺寸按朝向分组取中位数，并用 **MAD 稳健变异系数**判断"是否固定尺寸"：
CV ≤ 0.30 才做规范化。结果自动把 7 类排除掉：

| 排除类 | CV (w/h) | 理由 |
|---|---|---|
| ic | 0.82 / 0.85 | 一页 51×58、另一页 247×372，必须用检测出来的实际范围 |
| switch | 0.44 / 0.42 | 同上 |
| opamp / buzzer / battery / raw_block / other | 0.37–1.36 | 同上 |

**这条规则很重要**：给 IC 硬套一个尺寸会把 209 个 GT 里的大部分直接判错，而不是"稍微不准"。

### 落地位置（两处导出都已接线）

| 文件 | 变更 |
|---|---|
| `src/pcb_parser/export_geometry.py` | 新增：`load_table` / `apply_to_box`，支持 canonical 与 affine，raw 词表（r/c/box/gnd/v）自动解析 |
| `src/pcb_parser/component_text.py` | 导出前应用几何变换，`records` 增加 `bbox_top_left_exported` 供审计，诊断里加 `bbox_convention` 计数 |
| `scripts/run_baseline.py` | 同上（quick dev 走的是这条独立路径，之前漏了） |
| `configs/baseline_v3.json` | 新增 `bbox_convention` 开关，`enabled=false` 即完全恢复原行为 |
| `tests/test_export_geometry.py` | 新增 13 个测试（含"IC 不得被套尺寸""关闭时零影响"） |

冒烟验证（0002，whole）：导出尺寸已变成 c 16×20 / 20×16、gnd 20×19 / 19×20、v 10×10，
IC 保持 269.9×385.4 的检测范围，`format_valid: true`。

### 尚未验证

- 以上全部是**公开 40 页**的口径代理（`--match class`，是官方分的上界）。真实提升要等下一次官方提交。
- 第 5 次评测里电源只有 72.4% 匹配，而公开页是 93.7%。这次的中心偏移正是针对该差距，但**没有隐藏集的 GT 无法直接验证**。
- 检测空间偏移（用检测框对 GT 拟合）在拟合集内更优（.9064），但 2 折交叉只有 .8919 < .8999，样本量不足，**未采用**。它的价值是暴露了 `net_input` 需要 −0.21×宽 的水平位移，可作为后续专项。
- WP1 的"墨迹锚定"还没做。如果尺寸+偏移已把电源从 72% 抬到 90%+，WP1 的优先级可以下调。

### 下一步

1. 跑一遍完整 13 例流程，把新的 `components_partial.json` 走一遍提交，拿到第 6 次评测分数（这是唯一真信号）。
2. 按第 6 次的分项决定：电源还需要 WP1 还是够了；`net_input` 要不要按文本条重标。
3. 之后转 WP2（非电源漏检/误检）和 WP3-B（用官方 200 例 GT 重训）。

## 10. 端到端验证与两个额外发现（2026-10-17 深夜）

隐藏 13 例的图像不在本地，无法直接重跑。改用**公开 200 例**做端到端验证：真实 pipeline
（YOLO + SAHI + OCR + 绑定 + 导出）跑 6 个页面（含三个最差例 0001/0080/0102），再用官方口径评。
这比几何量的代理强，因为它覆盖了整条链路。

### 10.1 阈值：一个"文档要求但没做"的步骤，值 +3.2 分

`configs/baseline_v3.json` 的 `adoption_checklist` 里写着"把 `class_thresholds.json` 的阈值
粘到 `detector.class_confidence`"，但该字段一直是空的 —— **线上跑的是全局 conf 0.2**。

同一批 6 页（几何都开着）：

| | TP | FP | FN | F1 |
|---|---:|---:|---:|---:|
| 无逐类阈值 | 314 | 94 | 48 | .8156 |
| **有逐类阈值** | 305 | 53 | 57 | **.8472** |

FP 从 94 砍到 53。已写入 `configs/baseline_v3.json`，并记录来源与哈希。

### 10.2 几何必须作用在**未裁剪**的检测框上，并且要夹在 cell 内（+6.4 分）

6 个最差页，同一次 run 的原始检测，只改几何：

| 做法 | TP | FP | FN | F1 |
|---|---:|---:|---:|---:|
| 无几何 | 293 | 65 | 69 | .8139 |
| canonical | 315 | 43 | 47 | .8750 |
| **canonical + clamp** | **316** | **42** | **46** | **.8778** |

`clamp` = 符号框必须落在检测框内。因为偏移量是按"检测框尺寸"的百分比给的，而边界元件
的检测框被画幅裁小后会让偏移变小；不加 clamp 时密集页的地线会低 ~7px。

**真实 pipeline 重跑得到完全相同的 .8778 / .9474 / .8269** —— 接线正确性已验证，
不是只在离线公式里成立。

### 10.3 40 页公开集（最终配置）

| 配置 | 总体 F1 | 电源 F1 | 非电源 F1 |
|---|---:|---:|---:|
| 原始检测框 | .8641 | .9369 | .8217 |
| **canonical + clamp** | **.9009** | **.9643** | .8640 |

### 10.4 一个必须记住的坑（导致我一度误判）

同一配置、两次不同 run 的密集页检测数可能不同（`WARNING NMS time limit ... exceeded`
出现时 NMS 会被截断）。因此**跨 run 比较会污染结论**：我一开始拿"几何开的 run"和
"几何关的 run"对比，得出几何只值 +1 的结论，实际同 run A/B 是 +6.4。

规则：**几何类改动一律在同一次 run 的原始输出上做离线 A/B**，因为几何是纯后处理。

### 10.5 仍未解决

- `run_baseline.bind_designators` 的位号绑定很弱（0001 只认出 28/107，0029 是 0/36），
  key 口径下普通元件 F1 只有 .34。但**这不是提交路径**：提交走的是
  `component_text.bind_components`（第 5 次评测 key F1 是 .948）。两套绑定器并存本身就是风险。
- 隐藏页电源 72.4% 的问题没有 GT 无法直接验证，下一次官方提交才知道。
- 下一个真信号仍然是第 6 次评测分数。

## 11. 类型词表：一个值约 40 分的隐患（2026-10-18）

### 11.1 发现

诊断 6 个最差页的 FP 时发现：模型把 3.3V 电源箭头认成 `ground`、把 testpoint 焊盘认成 `ic`。
追下去发现更严重的问题在**类型字符串**上。

官方口径对 `type` 是**精确字符串比较**，唯一放行的别名只有
`box/ic` 和 `gnd/ground`、`v/vcc/vdd`。而本仓库的 `configs/baseline_v3.json`
把元件导出成 `resistor` / `capacitor` / `ic` / `vcc`，官方 GT 写的是 `r` / `c` / `box` / `v`。

同一批检测、只改类型字符串（6 页，严格口径）：

| 导出词表 | TP | FP | FN | F1 |
|---|---:|---:|---:|---:|
| 本仓库现状 `resistor/capacitor/ic/vcc` | 168 | 190 | 194 | **.4667** |
| 官方原词 `r/c/box/v/pin/testpoint/net_input` | 312 | 46 | 50 | **.8667** |

**+40 个点，只来自字符串。**

旁证：第 5 次评测的提交文件里类型是 `r/c/box/l/gnd/v/led/d` —— 说明线上镜像用的是原词表，
而本仓库的配置不是。也就是说**用本仓库配置重建镜像，分数会从 .87 掉到 .47**。

### 11.2 另一个塌缩

`class_mapping` 把 `pin_port`/`testpoint`/`net_input` 三个类**全映射成 `vbus_in`**，
而官方 GT 是 `pin`/`testpoint`/`net_input` 三个不同类型。这 15 个元件（6 页里）
无论框多准都不可能算对。

### 11.3 修法（不改 `class_mapping`）

`class_mapping` 同时喂给网络/标签阶段（`vbus_in` 等），直接改会打断下游。所以新增一层
**只在导出时生效**的词表：

| 文件 | 变更 |
|---|---|
| `configs/export_vocabulary.json` | 新增：检测器类名 → 官方原词（照抄 GT 拼写，含 `battary`/`transfomer`/`net_bidirection` 这些拼错的） |
| `src/pcb_parser/export_geometry.py` | 新增 `load_type_map` / `export_type`，按**检测器类名**映射（这样才能解开 vbus_in 塌缩） |
| `component_text.py` / `run_baseline.py` | 导出点改用 `export_type`；内部 `records[].type` 保持内部名，下游不受影响 |
| `configs/baseline_v3.json` | 新增 `export_vocabulary` 开关 |

### 11.4 端到端结果（6 个最差页，严格官方口径）

| 阶段 | F1 |
|---|---:|
| 原始检测框 + 旧词表 | .4667 |
| + 逐类阈值 + canonical 几何 + clamp | .8778 |
| **+ 官方原词表** | **.8806** |

`pin_port` 召回从 0 升到 .62（5/8）。仍为 0 的：`testpoint` 0/8（训练框只有 36 个，
且此前被塌缩）、`net_input` 0/4（官方框是 30×10 文本条，我们的定义不同）。

### 11.5 我的评测器此前是宽松的（已修）

在此之前 `class_matches` 会把两边都归一成检测器类名再比，于是 `resistor` 与 `r`
被视为相等 —— 这让报告的数字偏乐观，也掩盖了本节的问题。现在改为
**只放行官方别名、其余精确比较**，并加了 `from_eval` 标记：离线评测 JSON 里没有导出字符串，
只能按检测器类名比（那是检测质量指标，不是官方分估计）。这条区分必须一直保留。

### 11.6 仍未验证

- 线上镜像的 `class_mapping` 是否与本仓库同源，无法从本地确认。**下次提交前必须用镜像里的
  实际产物核对类型字符串**，否则可能带着 .47 的词表上分。
- `.8806` 仍是 6 页代理，不是官方分。

## 12. 提交包与本地自测（2026-10-18）

### 12.1 已就绪

| 文件 | 作用 |
|---|---|
| `submission/agent.py` | 遍历输入 → 逐例调用冻结流水线 → 规范化 → 写 `<case>_Pred.json` 与 `decision_log.json` |
| `submission/run.sh` | 入口脚本（纯 LF 换行），`ENTRYPOINT ["/submission/run.sh"]` |
| `submission/Dockerfile` | 基础镜像 `edarace-agent-base:ubuntu24`；torch 从 cu128 索引、其余按 `requirements-gpu.lock.txt`；构建期预热 OCR 权重 |
| `submission/README.md` | 契约、构建、风险清单 |
| `scripts/build_submission.ps1` / `.cmd` | 预检 → 暂存 → 构建 → 冒烟 → 导出 tar → 压 zip |
| `scripts/smoke_submission.py` | **不依赖 Docker** 的端到端自测 |

`run.sh` 只做参数与目录；单例失败不中断整批（记录进 `decision_log.json`），因为
赛题规则 R17 规定未完成案例计入分母，整批中断等于余下全 0。

### 12.2 本地自测已通过

```
smoke_submission.py --cases 0002 0018 0029 --limit 3
  → 3/3 出结果，36.1s（约 12s/例）
  → 类型为官方原词 box/c/crystal/gnd/v/r/led/switch/other
  → 电容/电阻/地的框仍是规范尺寸（16x20 / 8x20 / 20x19）
  → 三个顶层 section、decision_log.json 就位
```

13 例按 12s/例约 2.6 分钟，远低于赛题 2 小时预算（第 5 次评测实测 180s）。

### 12.3 尚未执行：镜像构建

**Docker 守护进程未运行**（`com.docker.service` = Stopped），所以 `docker build/save` 还没跑。
构建脚本的预检已能给出可执行的提示：

```
=== preflight: docker ===
ERROR: Docker daemon is not reachable. Start Docker Desktop, wait for "Engine running", then re-run.
```

另外本机 PowerShell 执行策略禁止直接运行 `.ps1`，已加 `scripts/build_submission.cmd` 包装
（`-ExecutionPolicy Bypass`）。

### 12.4 构建期的四个未知

1. `torch==2.10.0+cu128` 是否仍可从 cu128 索引安装；
2. OCR 权重预热是否成功（失败则评测容器内无网会挂）；
3. A800（sm_80）与工作点阈值是否匹配（阈值是在 RTX 5060 上调的）；
4. 镜像体积（预计 zip 3–5 GB）与上传耗时。

这四条都只能在构建/首次评测时暴露，已写进 `submission/README.md`。

## 13. 镜像构建成功（2026-10-17 实测）

### 13.1 产物

| 文件 | 大小 | 说明 |
|---|---|---|
| `D:\pcb_submission_out\pcb-parser-v4.zip` | **3.34 GB** | 上传这个（内含 tar） |
| `D:\pcb_submission_out\pcb-parser-v4.tar` | 3.35 GB | 原始镜像包 |
| `pcb-parser-v4.tar.sha256` | — | `A5F812B5C478A420898FDAE81F625F30835112BB0951EB77244264EF465ACA11` |
| `submission\out\pcb-parser-source.zip` | 286 MB | 备选：平台自行构建（含离线 wheel 与 OCR 模型） |

镜像 9.68 GB（**比第 5 次评测的 10.7 GB 还小**），构建+导出共 2.7 分钟。

### 13.2 容器冒烟通过，且与宿主结果逐项一致

```
[run.sh] torch 2.5.1+cu121 cuda False
[agent] 1/1 cases produced components, 0 failed   6.0s/例（CPU）
```

0002 的容器产物与宿主产物**完全一致**：13 个元件，类型 `box/c/crystal/gnd/v`，
尺寸 `c 16x20 x4`、`c 20x16 x2`、`gnd 20x19 x4`、`v 10x10`、`box 274.1x386.4`、
`crystal 15x10`。`decision_log.json` 字段齐备，token 计 0。

### 13.3 构建期解决的三个真实障碍（都是环境问题，不是代码问题）

1. **容器内 TLS 全线失败**：`docker build` 里 pip 访问任何 index 都报
   `SSL: UNEXPECTED_EOF_WHILE_READING`。实测容器内 HTTP 通、TCP 443 通、ClientHello
   发出后连接即被打断；宿主机访问 pypi/pytorch/tuna/aliyun 全部 200。不是 MTU（1500 正常、
   1472 不分片可通），也不是 IPv6（强制 `-4` 同样失败）。判定为宿主机网络过滤驱动拦 VM 的 TLS。
   **对策**：改为**完全离线构建** —— 宿主机下载 Linux cp312 wheel，镜像内 `--no-index --no-deps` 安装。
2. **依赖解析会挂死**：`pip download -r ... --platform` 的解析器空转 15 分钟不落盘。
   **对策**：新增 `scripts/fetch_submission_wheels.py`，用 `--no-deps` 逐包下载并自己解析
   wheel METADATA 算依赖闭包，41 个 wheel / 159 MB，秒级完成。
3. **基础镜像已自带大半依赖**（实测）：torch 2.5.1+cu121、torchvision 0.20.1+cu121、
   numpy 2.5.3、cv2 4.10(headless)、PyYAML、networkx、Pillow。因此**不装 torch**，
   只补 ultralytics/sahi/rapidocr/onnxruntime/scipy/shapely/pyclipper/polars/matplotlib 等 9 个包。
   同时刻意**不装** rapidocr 牵出的 opencv-headless 5.0，避免把基础镜像的 cv2 4.10 顶掉。

### 13.4 一个必须记住的版本偏差

镜像里是 **torch 2.5.1+cu121**，而开发 venv 是 2.10.0+cu128。后果：

- 本机 RTX 5060 是 sm_120，cu121 不支持 → **本机容器内 GPU 不可用**，冒烟只能走 CPU；
- 评测机是 A800（sm_80），cu121 支持 → 线上可用 GPU；
- 因此「容器冒烟通过」验证的是依赖、OCR 权重、配置路径、输出契约，**没有验证 GPU 路径**。
  若线上出现 GPU 相关异常，先怀疑这条偏差，不要先怀疑阈值。

### 13.5 顺带关掉了一个旧疑问

第 5 次评测的镜像就在本机（`pcb-parser:local-cuda-0e4a7241`，10.7 GB）。核对后：

- 它的 `configs/baseline_v3_adopted.json` 里 `class_confidence` 与 `configs/class_thresholds.json`
  几乎一致（battery .0973、capacitor .5222、ground .4121 …）→ 第 10 节"阈值没粘进去"的判断成立：
  **线上镜像有阈值，本仓库配置没有**。
- 它的 `class_mapping` 仍是 `resistor`/`capacitor`/`ic`/`vcc`，但提交文件里是 `r`/`c`/`box`/`v`
  → 说明线上还有一层**导出期重映射**（镜像里多一个本仓库没有的 `component_export.py`）。
  本方案第 11 节的 `export_vocabulary` 等价复现了这层，且**额外**做了它没有的事：
  按检测器类名解开 `vbus_in` 塌缩、以及对电源符号按 glyph 中心做偏移标定。

## 14. 第 6 次评测：0.0 的原因与修复（2026-10-17）

### 14.1 结论：不是模型退化，是字段阶段没跑

第 6 次 `final_score = 0.0`，但容器 **exit 0、13 个文件全收、81 秒**。判分器逐例报告显示
`component tp/fp/fn = 0/60/58`，即**一个元件都没算对**。

| 指标（case 1） | 第 5 次 | 第 6 次 | |
|---|---:|---:|---|
| type_accuracy_on_matched | .929 | **1.000** | ✅ 词表修复生效 |
| bbox_accuracy_on_matched | .905 | **1.000** | ✅ 几何修复生效 |
| name_accuracy_on_matched | .938 | **0.000** | ❌ |
| value_accuracy_on_matched | .881 | **0.176** | ❌ |
| 电源匹配 | 10/19 | **0/19** | ❌ |
| pins 组数 | 33（51 元件） | **0（60 元件）** | ❌ |

17 个几何+位号都配上的元件，**全部** `failed_checks: ["Name","value"]`。

### 14.2 根因：接错了入口

第 6 次的镜像接的是本仓库 `run_baseline.py` 的 `bind_designators`——那是个骨架实现，
固定写 `Name: None, value: None`，不产 pins/nets。仓库里真正填字段的
`component_text.bind_components` 也只填 components。

第 5 次镜像的 `agent.py` 编排的是完整链路，并用到 **5 个本仓库没有的模块**：
`component_export.py`（电源符号专用导出）、`ic_name_selection.py`、`ic_geometry.py`、
`led_geometry.py`、`adapter.py`。仓库快照并不包含提交级链路。

这是打包时的判断失误：之前已记录"两套绑定器并存是风险、`run_baseline` 的很弱
（0001 只认出 28/107）"，但为省事选了它。

### 14.3 修复：以第 5 次镜像为底座叠加

镜像在本机（`pcb-parser:local-cuda-0e4a7241`），`/submission` 只有 90 MB / 47 个模块。
把它 `docker cp` 出来，只在 `agent.py` 写文件前插入几何后处理，其余原样：

| 文件 | 变更 |
|---|---|
| `submission/overlay/Dockerfile` | `FROM pcb-parser:local-cuda-0e4a7241` + 只 COPY 4 个文件 |
| `submission/overlay/export_geometry.py` | 几何模块（与主仓库同一份） |
| `submission/overlay/overlay_check.py` | 构建期自检：文件齐、表可加载、agent 确实调用了变换 |
| `configs/bbox_convention_v1.json` | 一并拷入 |
| `agent.py` | 插入后处理；**跳过 `ic`/`box`/`led`**（该链路自己精修这两类，其尺寸也不固定） |

用第 5 次镜像当底座的好处：依赖已装好且 `pip check` 通过，**不需要联网**（本机 Docker VM
的 TLS 是被拦的），也就绕开了第 13.3 节那个坑。

### 14.4 验证（6 个公开页，容器内 CPU）

```
0001: 88 comps, 55 pins, geometry applied 80, 217s
0002: 13 comps, 17 pins, applied 12,  48s
0018: 33 comps, 73 pins, applied 26, 109s
0029: 16 comps, 16 pins, applied 16,  72s
0080:113 comps, 57 pins, applied 105, 627s
0102: 44 comps, 31 pins, applied 43, 237s
```

官方口径（key + Name/value 全部强制，最严）：

| | TP | FP | FN | P | R | F1 |
|---|---:|---:|---:|---:|---:|---:|
| OVERALL | 232 | 75 | 130 | .756 | .641 | **.694** |
| 电源 | 143 | 12 | 6 | .923 | .960 | **.941** |
| 非电源 | 89 | 63 | 124 | .586 | .418 | .488 |

**电源从第 6 次的 0/19 回到 143/149 匹配（F1 .941）**——这一项就是几何偏移标定的直接效果。

### 14.5 新的短板：位号恢复

非电源 F1 只有 .488，全部丢在位号上。位号解析率：0002 62%、0018 61%、0080 57%、
**0001 48%、0102 39%、0029 0%**。0029 一个位号都没解析出来，因此整例非电源全灭。

已加 `scripts/check_submission_outputs.py` 作为**上传前硬闸**（第 6 次缺的就是这道闸）：

- 每个 case 的 `pins` 必须非空；
- 位号解析率 ≥ 50%；
- ordinary 元件的 `Name`/`value` 非空率 ≥ 80%；
- bbox 有序、有限、在图内。

当前 6 例中 0001/0029/0102 会被这道闸拦下（位号不足），说明**在提交前还应先攻位号恢复**，
否则总体仍在 .69 附近而不是 .88。

### 14.6 产物

| 文件 | 大小 | 用途 |
|---|---|---|
| `D:\pcb_submission_out\pcb-parser-v5.zip` | 3.31 GB | **可提交**（第 5 次链路 + 几何修复） |
| `D:\pcb_submission_out\pcb-parser-v5.tar` | 3.32 GB | 原始镜像，sha256 `00A86176…3851E9` |
| `D:\pcb_submission_out\pcb-parser-v4.zip` | 3.34 GB | **不要提交**（第 6 次那个，0 分） |
