# OCR 阶段现状与首轮修正（v6 基线）

状态日期：2026-10-18。基线代码是 `D:\pcb5_submission`（从第 5 次评测镜像
`pcb-parser:local-cuda-0e4a7241` 的 `/submission` cp 出来的那份），不是本仓库的
`src/` 快照。仓库快照缺少镜像独有的 15 个模块，且 `component_text.py` 与镜像不同源
（仓库多 `export_geometry` 接线、少 LED 位号片段修复）。

本文件只记录**实测**数字；没有隐藏集 GT 的推断都已标注。

## 1. 现在的 OCR 链路

```
PNG → YOLO(v3_nano)+SAHI → 检测框
    → RapidOCR(PP-OCRv6 det_small + rec_small + ch_ppocr cls, onnxruntime **CPU**)
       整图 1 视图 + 768 切片、overlap 0.2、放大 2×、旋转 0/90、min_confidence .55
       Det.limit_side_len 2048
    → merge_texts（IoU .45 合并，保留多视图观测与原始拼写）
    → tokens_from_texts（切分融合串、判 role）
    → 匈牙利分配（designator / value / Name，每字段私有 dummy 列、ambiguity margin 3）
    → component_identity_recovery / engineering 出口 / adapter.to_target（类型、Name、模板框）
```

配置：`configs/pin_text_component_v2.json`（`component_text.ocr`）、
`configs/baseline_v3_adopted.json`（`ocr`、`component_text.binding`）。
`component_text.recovery` 段存在（scale 4、max_rois 32），但 `text_recovery.recover_texts`
**全工程无人调用**，是死代码。

耗时（v5 六页、容器 CPU）：0001 217s、0002 48s、0018 109s、0029 72s、0080 627s、0102 237s；
官方预算 13 例 1200s（第 5 次线上实测 180s）。任何加重 OCR 的改动都必须先测单例墙钟。

## 2. 六个公开页的误差预算（实测）

按官方口径（key+type+Name+value 全等，四角各 ≤20px；电源按几何一对一、忽略 key/Name），
用冻结的 v5 检测框与 OCR 结果离线重放 binder：

| 项 | 数值 |
|---|---:|
| GT 元件（6 页） | 362（普通 213、电源 149） |
| 预测元件 | 307（普通 152，其中 `UNRESOLVED_DET_*` 164 含电源） |
| 普通元件 key 命中 | 124 / 152（81%） |
| 命中后字段全对 | 99 |
| 普通 FP | 26 个自造 key（如 `R12100` 类残片）+ 其余为错配 |
| 普通 FN | 89（含 GT 重编号 ~30、图上无位号 ~19、字段错 25） |

字段级失败（命中之后）：value 25、type 8、Name 2、bbox 6（有重叠）。

## 3. 三条把数字直接算出来的规则

### 3.1 GT 的 `Name` 就等于元件 key，`box` 例外

200 页 9186 个元件逐类统计 `Name == key` 的比例：

| 类型 | 比例 | 类型 | 比例 |
|---|---:|---|---:|
| c / r / l / d / led / pin / switch | .999–1.000 | testpoint | .977 |
| gnd / v | 1.000 / .999 | crystal_2pin | .976 |
| **box（IC/连接器）** | **.242** | mosfet / bjt | .985 / .980 |

所以除 `box` 外，Name 不需要 OCR，把已经绑上的位号回填即可。
v5 只在 `to_target` 里给 `r/c/l` 回填，crystal/led/switch/testpoint/vbus_* 全部留空。

### 3.2 无源器件的 value 从不写成纯数字

3200 个 r/c/l 的 GT value：**带单位字母 2152、简写（27p / 100n）87、null 981、纯数字 0**。
而 binder 允许 `numeric` 参与 value 分配，于是把引脚号、邻居刻度当成值写出去。

### 3.3 位号字符串在 OCR 输出里的"天花板"

6 页 203 个 GT 普通位号：**145 个（71.4%）在 OCR 输出里出现过**，只有 125 个（61.6%）
被绑到元件上。对剩下 57 个做三分诊断：

| 分类 | 数量 | 含义 |
|---|---:|---|
| `LABEL_ON_SYMBOL` | 11 | 标签就在符号上，是我们丢了 → 可修（实际由 `component_identity_recovery` 补回大半） |
| `LABEL_ELSEWHERE` | 9 | 该位号字符串在页面别处（图上印的不是 GT 那个号） |
| `NO_TOKEN` | 57 | OCR 从未产出该字符串 |

`NO_TOKEN` 里最大的一块不是识别问题：**0029 整页没有印刷位号**（TI 数据手册图只有
`1.8V / 10kΩ / 1uF` 和引脚号），GT 的 R1–R5、C1–C10 与图上位置不呈任何顺序，只能承认上限；
0102 的 TP1–TP8 同理（图上无 TP 标注，OCR 在那几个焊盘附近读到的是 RP2040 的 GPIO 名）。

## 4. 一个必须先知道的口径陷阱

公开 200 页的位号**不总是印刷体**：

| 页 | 图上印的 | GT 写的 | 证据 |
|---|---|---|---|
| 0002 | `X2`（19M2） | `X1` | [`evidence/0002_X1_printed_X2_but_gt_X1.png`](../analysis_ocr/evidence/0002_X1_printed_X2_but_gt_X1.png) |
| 0102 | `U19` | `U1` | [`evidence/0102_U1_printed_U19_but_gt_U1.png`](../analysis_ocr/evidence/0102_U1_printed_U19_but_gt_U1.png) |
| 0001 | `R11 / R12 / R9` | `R3 / R1 / R5` | [`evidence/0001_R1_printed_R11_but_gt_R1.png`](../analysis_ocr/evidence/0001_R1_printed_R11_but_gt_R1.png) |
| 0080 | `C140 / C130 / R135` | `C1 / C2 / R1` | 同类规律，见第 3.3 节诊断 |
| 0029 | **完全没有位号** | `R1–R5 / C1–C10` | [`evidence/0029_page_has_no_printed_designators.png`](../analysis_ocr/evidence/0029_page_has_no_printed_designators.png) |

`Dataset300/*_CVJsonStd.json` 里 `searchNameByOCR` 全为 `null`，说明这批 `_target_named.json`
的位号是**生成**的，不是 CAD 导出。第 5 次隐藏 13 例的 key F1 是 .948，说明隐藏集与印刷体基本一致。

结论：**本地 key 口径分数是下界，不能当 OCR 唯一 KPI**。评估 OCR 时同时看
① `resolved_designators / detections`（与 GT 无关）、② visible-and-bound 比例、
③ 官方 key 口径只做回归。

## 5. 本轮改动（三处，均可开关回退）

| 文件 | 改动 | 开关 | 影响 |
|---|---|---|---|
| `src/pcb_parser/component_text.py` | value 候选剔除 `numeric` 角色、剔除 `1u0` 这类"单位字母后接数字"的串 | `binding.passive_value_allow_numeric`（默认 false） | **+12 TP** |
| `src/pcb_parser/component_text.py` | 新增"剩余标签二次分配"：未被使用的位号组可以配给仍无位号的检测（放宽前缀白名单），只动未使用组与未定检测 | `binding.leftover_reference_pass`（默认 **false**） | 解析数 143→153，**TP 不变**；会抢在 `component_identity_recovery` 之前，故默认关 |
| `adapter.py` | `to_target` 对"非 box / 非电源"类型回填 `Name = key` | `DERIVED_NAME_TYPES` | 修正 crystal/led/switch/testpoint 的空 Name；实测 `X2: Name null → X2` |

`adapter.to_target` 的规则本体（部署副本）：

```python
if typ in DERIVED_NAME_TYPES:
    component['Name'] = derived_name(cid, result['components'])
```

`DERIVED_NAME_TYPES` 是"除 box 与电源外的全部类型"；`derived_name` 复用 `passive_name`
（保留 `C6_1 / C6_2` 这类实例后缀回落到基名 `C6` 的行为）。

两处改动的位置：
- `D:\pcb5_submission\src\pcb_parser\component_text.py`、`D:\pcb5_submission\adapter.py`（部署基线，已改）
- `pcb_parser_competition/src/pcb_parser/component_text.py`（本仓库快照，已用同样的两条规则合并，
  并保留仓库独有的 `export_geometry` 接线；`NAME_FROM_EVIDENCE_TYPES` 在文件顶部）

仓库里没有 `adapter.py`——它是镜像独有的模块。**下一次用仓库重建镜像前，必须先把部署副本
（`D:\pcb5_submission`，47 个模块）整体纳入仓库，否则会再次丢掉 `adapter.py`、
`component_export.py`、`text_recovery.py` 等 15 个模块（第 6 次 0 分即此类失误）。

### 离线 A/B（同一批冻结检测+OCR，官方口径）

| 配置 | TP | FP | FN | component F1 |
|---|---:|---:|---:|---:|
| v5 行为（两条新规则关闭、Name 只回填 r/c/l） | 229 | 78 | 133 | .6846 |
| **本轮默认** | **241** | **66** | **121** | **.7205** |

component 权重 .30，折合全场 **+1.1 分**。这仍是 6 页代理，不是官方分。

### 端到端（六页真实管线重跑，本地 RTX 5060）

用改后代码跑 `agent.py`（`-input D:\pcb_v5_batch\input`），再用
`scripts/evaluate_official_component.py --match key --require-fields` 打分，
与同一条命令重打的 v5 产物对比：

| 产物 | TP | FP | FN | P | R | component F1 |
|---|---:|---:|---:|---:|---:|---:|
| v5（容器 CPU，历史产物） | 232 | 75 | 130 | .7557 | .6409 | .6936 |
| **v6（本轮改动，GPU 重跑）** | **245** | **61** | **117** | **.8007** | **.6768** | **.7335** |

**+4.0 个 component 点 = 全场 +1.2 分**，与离线预测的 +3.6 一致。

不加 `--require-fields`（不比较 Name/value）时两版是 255/52/107 (.7623) vs 255/51/107 (.7635)，
说明**检测框与位号几乎没变，全部增益来自字段**。逐元件比对（`analysis_ocr/decompose_v6.py`）：

- **13 个元件由错变对，0 个由对变错**：0001 `R7/R8`、0002 `C82`、0080 `R116/R118/R128/R129/R133/R145/R146/R16/R193`、0102 `R3`（`47` → `27Ω`）。
- 0002/0001/0018/0029/0102 的元件数与 pin 数与 v5 完全相同；0080 少 1 个元件、少 5 个 pin，
  是 GPU 与容器 CPU 的 NMS/浮点差异（同一配置两次运行的已知噪声，见 `detection_v4_plan.md` 10.4）。
- `final_results` 结构、`bbox_convention_applied`、`to_target` 契约均校验通过。

组件数与 pin 数对照（v5 → v6）：0001 88/55 → 88/55、0002 13/17 → 13/17、0018 33/73 → 33/73、
0029 16/0 → 16/0、0080 113/57 → 112/52、0102 44/9 → 44/9。

## 6. 第二轮：类型词表已修，引脚的天花板在几何不在文字

### 6.1 导出期类型词表（已改，可测）

`adapter.refined_type` 在导出前把内部 kind 映射成标注拼写，并拆开两处塌缩：
`vbus_in` 按标签条形状拆（宽条 w/h≥2 → `net_input`，否则 → `pin`；`testpoint`
与 `pin` 的 GT 框都是 20×20，几何不可分，取多数类），`crystal` 按已挂引脚数拆
（2/3/4 → `crystal_2pin/3pin/4pin`，不足两个则回退 2pin，占公开 GT 的 75%），
另外 `battery→battary`、`vbus_out→net_output`、`vbus_bidirectional→net_bidirection`。
标签类同时套用标注的规范框（net 标签 30×10、端口 20×20、晶振 2pin/3pin 10×15、4pin 40×40）。
晶振框必须跟着类型一起规范，否则类型猜对、框仍是检测 cell（41×44 vs GT 15×10）照样判错。

| 数据集 | 改前 TP/FP/FN | 改后 | component F1 |
|---|---|---|---:|
| 公开 6 页（隐藏集代理） | 243 / 63 / 119 | **246 / 60 / 116** | .7275 → **.7365** |
| 官方 10 个 Golden Case | 78 / 26 / 106 | 79 / 25 / 105 | .5417 → .5486 |

端到端抽查 0036：`DOUT` 由 `vbus_in` 变成 `net_input` 并套上 30×10 框，
元件数/pin 数/网数不变（18/22/3），结构校验通过。

试过但**回退**的做法：用检测框尺寸判 crystal 拼写——检测 cell 是标注符号框的 2–3 倍，
规则把两个 2pin 晶振判成 `crystal_4pin`，TP 反而从 79 掉回 78。

### 6.2 引脚：为什么"多产引脚"这条便宜路走不通

用 10 个 Golden Case（干净的官方 GT：位号、1..N 引号脚、net 全有）做了三组测量：

| 测量 | 结果 | 含义 |
|---|---|---|
| 官方第 5 次评测（隐藏集） | pin precision **1.00**、recall .125–.50、pin 号准确 1.00 | 我们**产得太少**，不是识别错 |
| Golden 10 页：GT 引脚 98，我方 tip 落在 5px 内 | **16（16%）** | 瓶颈是 tip 几何 |
| Golden 10 页：同一批 tip 的 pin 号/名正确 | 号 3 / 名 6 | 文字只在少数落到位置上的引脚才起作用 |

再把 tip 偏差做网格搜索（先平移再一对一重配，不偷看）：

| 数据集 | 不平移 | 最优全局平移 | 结论 |
|---|---|---|---|
| Golden 10 | 16/98 = 16.3% | (+0.5,−1.5) → 40/98 = **40.8%** | 单个常数偏移救不了 |
| 逐页最优 | — | 0042 任意平移都 10/20；0100 需要 (2.5,−4.0) 才 26/30；0099 需要 (5,−6) | **偏移是逐页/逐符号的**，不是统一约定 |

对比上一轮元件框的成功经验（固定尺寸＋中心偏移，公开 40 页拟合、留出验证不变，+6.4 分），
引脚 tip 没有这种稳定的全局偏差，所以**不能照搬成一次标定**。这条属于 U-Net 分割精度
（或按页面尺度归一化后重训），不是 OCR 改动能解决的。

### 6.3 由此修正的优先级

1. ~~引脚字段覆盖率（松闸＋重识别）~~：**降级**。pin 文字本身已经对（隐藏集 1.00），
   松闸只会把 precision 从 1.00 拉下来；真正的缺口在 tip 几何，属检测/U-Net。
2. **网/导线导出闸门**：六页内部算出 260 条多引脚网，210 条"只缺一个引脚字段"、
   65 条被几何/结点语义挡住，实际导出 0 条（Golden 页能出 1–3 条）。连通性 35% 权重
   目前只有 ~1 分，这是剩下最大的一块；但要先有一个能离线复现的 net 打分器（官方
   `exact_net_f1` / soft F1 口径），否则改闸门只能凭猜。
3. **模型名字符级重识别**（`TPS3422EGDRY` 丢尾字母、`NRF24101` 的 1/I）：
   `text_recovery` 已存在但无人调用，接线成本低，量级 +0.5~1 分。
4. 密集电阻排对齐（0080 的 `R122–R125` 整排错位一格）。

## 7. 复现

```powershell
$py = 'C:\Users\Pc\Desktop\新建文件夹\pcb_parser_competition\.venv-gpu\Scripts\python.exe'

# 离线 A/B（秒级，不需要 GPU/OCR）
& $py analysis_ocr\offline_eval.py            # 本轮默认
& $py analysis_ocr\offline_eval.py --legacy-name `
       --set leftover_reference_pass=false --set passive_value_allow_numeric=true
& $py analysis_ocr\ab_types.py                # 导出期类型词表的 A/B（golden10 + big6）

# 诊断
& $py analysis_ocr\unbound_probe2.py          # 未绑定位号三级分类
& $py analysis_ocr\ocr_ceiling_probe.py       # OCR 里的位号天花板
& $py analysis_ocr\error_budget.py            # 字段级误差预算
& $py analysis_ocr\gt_convention_probe.py     # Name/value 标注约定
& $py analysis_ocr\pin_text_budget.py         # 引脚口径预算
& $py analysis_ocr\network_guard_budget.py    # 网络闸门账单
& $py analysis_ocr\label_type_geometry.py     # 标签类框尺寸可分性
& $py analysis_ocr\tip_calibrate.py           # tip 全局偏移网格搜索
& $py analysis_ocr\tip_overlay.py 0042        # tip 对照图（红=GT，蓝/橙=我们）

# 引脚离线重放（需要 --keep-work 的中间产物）
& $py analysis_ocr\pin_replay.py --case 0042 --work D:\pcb_v6_check\golden_out `
       --image D:\pcb_v6_check\golden_input\0042\0042.png

# 端到端（真实管线）
& $py D:\pcb5_submission\agent.py -input D:\pcb_v5_batch\input -output D:\pcb_v6_check\out_v6_full
```

脚本说明见 [`analysis_ocr/README.md`](../analysis_ocr/README.md)。这些脚本只读缓存与 GT，
不写任何预测产物。

## 8. 待确认

- 第 7 次评测尚未发生：`D:\pcb_submission_out\pcb-parser-v5.zip`（第 5 次链路 + 几何）还没提交，
  本轮 OCR 改动还没有进任何镜像。是否先用 v5 提交拿一次真信号，再打含本轮改动的 v6？
- `leftover_reference_pass` 是否要在隐藏页上试开（需要一次提交对比，不能只看 6 页）。
