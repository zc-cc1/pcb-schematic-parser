# EDA 决赛答辩补充材料:比赛成绩、关键实验数据与事实核查

- 依据:本地提交镜像 `pcb-parser:v102`(第一事实来源)、GitHub 开发仓库 `zc-cc1/vision`(辅助)、本机冒烟复现(2026-11-21)
- 配套:`source_audit.md`(完整审查)、`defense_slide_plan.md`(10 页 PPT 规划)、`assets/`(可视化素材)
- 本文档所有数字均可追溯到"出处"列;标 ⚠️ 者为用户提供、本地无证明文件,建议答辩前补充截图

---

## 一、比赛成绩与晋级(答辩定稿口径)

### 1.1 成绩事实

| 项 | 内容 | 证据状态 |
|---|---|---|
| **官方评测成绩** | **74 分**(FinalScore,最终提交版本) | ⚠️ 用户提供;建议补平台成绩页截图 |
| **本队排名** | **第 47 名** | ⚠️ 用户提供;建议补排名/晋级名单截图 |
| **晋级线** | 全国前 61 支队伍晋级决赛,**本队晋级** | ⚠️ 用户提供;建议补官方晋级公告 |
| 提交版本 | `pcb-parser:v102`(提交目录内 tar,镜像 config 入口 `/submission/run.sh` 已验证) | 本地 tar + OCI config |
| 官方提交 ID | 本地材料未记录 → 需人工从评测平台核对并截图 | 缺失 |

### 1.2 74 分、28.92 分、98.93 分三者关系(必须讲清)

| 分数 | 口径 | 数据范围 | 版本 | 含义 |
|---|---|---|---|---|
| **28.92223** | 官方平台分 | 30~50 案例平台评测 | **旧版镜像 8422**(在电源出口/LED/IC Name/IC 实框等修复之前) | 历史 |
| **98.93** | 本地离线回放分 | 13 页公开集(官方评分口径复算) | v101-score(与 v102 同源) | 开发用,不可与官方分直接比较 |
| **74** | 官方平台分 | 30~50 案例(**公开:黑盒 = 1:2,中等 40%/难 60%**) | 最终提交 v102 | **最终成绩** |

- 74 − 28.92 ≈ +45 分的提升来源(全部有代码与文档证据):无位号电源/地出口、LED 文字与框、IC Name 优先级与实框定位、工程编号(R/C/L 1/2 号)、观测字段链(S23/S24)、结点归并、导出语义(observed)、导出鲁棒性(回退+清洗)——见 `source_audit.md` §5/§7。
- 98.93 → 74 的差距来源:①官方集含 **2/3 黑盒案例且 60% 为难例**,本地 98.93 只在 13 页公开集回放;②黑盒页分辨率/绘制风格/元件密度分布未知,泛化性是主要扣分来源;③本地回放无法复现平台评测环境差异。**这是答辩时"为什么本地分这么高、官方 74"的标准解释,坦诚不回避。**
- Token 贡献:全程 0 次 LLM 调用(decision_log token_usage=0),Token 赋分制取**最优区间**,该 5% 权重按满分贡献;94 分主权重全部来自结构化识别精度。

### 1.3 成绩证明材料(答辩前人工补齐)

1. 平台成绩页截图(74 分 + 四项分指标若可见);
2. 排名/晋级名单截图(第 47 名/前 61 晋级);
3. 最终提交的提交记录截图(提交 ID、时间、镜像 tag);
4. 若有官方成绩单 PDF/邮件一并归档。
建议归档到 `assets/official/`(不要 PPT 里全放,备查页即可)。

---

## 二、两处数据口径修正(对上一轮审查报告的更正)

### 2.1 「断线网 +0.31」的正确含义

- 原文(`src/pcb_parser/wire_graph_artifacts.py:116-117`,最终层):**"Ship nets that came from a gapped wire too: measured +0.31 points on the 13 pages (connectivity 0.569 -> 0.576)"**
- 正确口径:**连通率(connectivity)0.569 → 0.576(即 +0.7 个百分点);"+0.31" 指总分的增量(0.31 分,百分制)**。
- 上一轮报告将该行写成"连通 +0.31(0.569→0.576)",把总分增量和连通率混写,予以更正。

### 2.2 「bbox 0.905→1.000」与「150/163→159/163」是两套独立口径

| 口径 | 数据范围 | 机制 | 数字 | 出处 |
|---|---|---|---|---|
| **A:canonical 框约定** | **40 页公开子集** | 用校准表把检测框(CAD cell,大 1.05~2×)替换为符号自然尺寸 + 校准中心 | component F1 **.8641→.8999**;power F1 .9369→.9616;**bbox 达标率(20px 规则).905→1.000** | `export_geometry.py:1-30`(头注释);`agent.py:988-994`(最终层) |
| **B:按页缩放** | **13 页 released 案例** | 模板尺寸(8×20 等)按页内中位数比例缩放,解决"同符号在 23×58 与 40×16 两种画法" | 20px 容差内模板元件 **150/163 → 159/163** | `adapter.py:60-64` |

两者都度量"20px 容差达标率",但**数据集不同(40 页公开 vs 13 页 released)、机制不同(查表校准 vs 页缩放)**,且两机制同时生效(canonical 先、页缩放对 r/c/l/d/led)。PPT 中建议分开标注,勿写成一条链。

---

## 三、统一实验数据表(PPT 直接可用)

| # | 实验 | 数据范围 | 指标 | 结果 | 出处(可回查) |
|---|---|---|---|---|---|
| 1 | 检测配置选型(fused 模式胜出) | 59 例公开微平均 | 检测 F1 | **0.8830** | `configs/baseline_v3_adopted.json:101` |
| 2 | 元件引脚库映射(1000 型号) | 13 页 released | PinF1 | **0.031 → 0.549**(+47 引脚号) | `scripts/run_engineering_networks.py:26-27` |
| 3 | 未决结点网放行(门控) | 13 页 released | 连通率 | **0.188 → 0.612** @0.83 精度 | `wire_graph_artifacts.py:16-18` |
| 4 | 部分成员导出 | 13 页 released | 连通率 | **0.259 → ~0.61** | `wire_graph_artifacts.py:21-23` |
| 5 | 断线网放行 | 13 页 released | 连通率 / 总分 | 0.569→0.576 / **+0.31 分** | `wire_graph_artifacts.py:116-117` |
| 6 | 结点归并(修复案例) | 页 13 | wire TP/FP/FN、F1、总分、连通 | 32/28/17→**46/13/3**;**58.72%→85.19%**;**82.871→92.656**;67.14%→87.53% | `docs/network_junction_reconciliation.md`;修复前快照 `page13_status_20261020/status.json`(score_100=65.58) |
| 7 | 端子拓扑(修复案例) | 页 5 | 连通、总分、wire F1 | **84.96%→100%;93.954→99.474;92.17%→94.74%** | `docs/terminal_topology.md:17-18` |
| 8 | 端子身份恢复 | 页 3 | 连通、总分 | 91.63%→100%;95.2462→99.2130 | `docs/terminal_identity_recovery.md:21` |
| 9 | 地符号拆分 | 页 11 | 元件 F1、总分 | 96.55%→100%;98.965→100 | `docs/page11_ground_identity_recovery.md:28` |
| 10 | bbox canonical 约定 | 40 页公开子集 | component F1 / bbox 达标率 | **.8641→.8999;.905→1.000** | `export_geometry.py:1-30`;`agent.py:992` |
| 11 | bbox 按页缩放 | 13 页 released | 20px 内模板元件 | **150/163 → 159/163** | `adapter.py:60-64` |
| 12 | Name==key 约定校准 | 200 页公开训练页 / 9186 组件 | 覆盖率 | r 1.000 / c .999 / l 1.000 / d 1.000 / led 1.000 / pin 1.000 / testpoint .977 / box 仅 .242(故 box 保留型号) | `adapter.py:28-40` |
| 13 | 本机冒烟复现(2026-11-21) | 官方示例页(1190×845,与赛题 PDF 同源) | 输出规模/精度/耗时 | **89 components / 165 pins / 63 nets**;U3 bbox=[539,281,689,731] vs 官方示例标注 [540,280,690,730]**差 1~2px**;CPU 2000.5s;v102 export_semantics 生效 6 处(SW1/SW2→PTS8153JK250SMTRLFS,置信 0.998/0.992) | `_audit_workspace/smoke/output_v102/`(decision_log、final_results、logs) |
| 14 | 官方成绩(最终) | 30~50 平台案例 | FinalScore | **74 分 / 第 47 名 / 晋级决赛** ⚠️ | 用户提供,建议补截图 |
| 15 | 官方成绩(历史) | 旧版 8422 | FinalScore | 28.92223 | `README.md:50` |
| 16 | 本地开发分 | 13 页公开集回放 | development_score | 98.93(v101-score) | L18 `package_manifest.json → export_resilience_update` |
| 17 | 测试体系 | 42 文件 | unittest 函数/断言 | 626+ / 978+(L16+L18 合并) | `tests/` 实测计数 |

---

## 四、事实核查结论(逐项)

### 4.1 Junction CNN 是否进入默认运行管线?

**否(默认关闭)**。证据:
- 默认链配置 `configs/wire_graph_v1.json:20`:`junction_model_available=false`,四向交叉采用**保守策略 split_four_way**(显式不确定,不猜连通);
- 学习型决策仅存在于 v2 pilot:`configs/wire_graph_v2.json`(junction_model_available=true,checkpoint sha256 锁定 901b0e69…,置信阈值 0.95,4 旋转一致才接受,且只对度=4 结点);
- 默认管线用的是**规则型**结点处理:保守拆分 + 红点检测(`detect_junction_dots`,纯颜色/形态学)+ 结点归并阶段(几何/墨迹证据)。
**答辩口径**:学习型分类器已训练并保留为可选增强,默认链选择"判错代价高于漏连"的保守语义——这是面向 soft per-net F1 评分的有意取舍。

### 4.2 200 页校准数据与最终评测数据的关系

- **200 页公开训练页(9186 组件)**:用于统计"标注约定"(Name==key 覆盖率、模板 bbox 尺寸),属开发期合法校准;运行时不读任何参考。
- **13 页 released**:用于修复模块的 before/after 离线回放(冻结模型/OCR 缓存),全部文档有记录。
- **40 页公开子集**:canonical 框约定校准(`scripts/calibrate_bbox_convention.py` 产出表)。
- **最终评测 30~50 案例**:公开:黑盒=1:2(中等 40%/难 60%)。**校准数据 ⊄ 评测黑盒**,泛化性因此是官方分与本地分的差距主因之一。
- 依据合同:`run_observed_net_fields.py:101 input_contract='PNG and frozen image-model/OCR predictions only; no annotation/CAD/Golden reads'`;各阶段 `official_*=null` 显式占位。

### 4.3 各实验的评价方法

| 类型 | 方法 | 代表 |
|---|---|---|
| 官方平台分 | 评测端按赛题公式(ComponentF1/PinF1/NetHypergraphF1/NetLineF1 加权) | 74 分 |
| 本地回放分 | 冻结模型/OCR 缓存 + 公开标注复算官方口径 | 98.93;docs 各页 before/after |
| 单模块消融 | 同一缓存下开关某模块,只允许目标页变化(其余 12 页逐字段相同) | 结点归并、端子拓扑 |
| 冒烟复现 | 真实端到端运行 + 与官方示例标注人工对照 | 本报告 #13 |
| 单元/回归测试 | 合成夹具(哈希/预算/幂等/不可变契约) | 626+ 测试 |

---

## 五、仍需人工确认的事实与材料(清单)

| # | 事项 | 建议 |
|---|---|---|
| 1 | 官方 74 分/第 47 名/晋级 61 强的截图与提交 ID | 从评测平台导出,归档 assets/official/ |
| 2 | 74 分对应的确切提交版本号(应为 v102 tar;若平台另有提交快照需核对) | 平台提交记录 |
| 3 | 决赛是否允许现场演示、是否提供 GPU | 与组委会确认;CPU 单例约 33 分钟(1190×845),A800 预期分钟级 |
| 4 | 官方成绩单是否给出四项分指标分解(若可见,可精确定位失分项) | 平台成绩页 |
| 5 | 技术报告 Word(赛题硬性要求) | 由本材料集生成,重点取 §三数据表+assets 图 |
| 6 | 数据贡献加分(FinalScore_Data,10 分)是否已提交数据 | 查提交记录 |

---

## 六、可视化素材与来源索引(对应 assets/)

| 文件 | 内容 | 来源 | 建议用途 |
|---|---|---|---|
| `assets/fig1_detection_overlay.png` / `fig1b_detection_clean.png` | 官方示例页检测+识别标注(89 元件全量/32 主要元件精简版) | 本机冒烟 v102 真实输出 | P3/P4/P6 |
| `assets/fig2_unet_segmentation.png` / `fig2b_unet_operating.png` | U-Net 双头:wire 概率图 + pin 热图(三联/工作阈值版) | 本机真实前向(weights/unet_wire_pin_v1.pt) | P4 |
| `assets/fig3_topology_skeleton.png` / `fig3b_network_overlay.png` | 骨架+39 结点 / 63 网络彩色编码 | 冒烟产物渲染 | P5 |
| `assets/fig4_page13_repair_case.png` | 页 13 修复案例:真实裁剪(gap_crop_case13.png,开发仓库)+ 修复前后数字与机制 | 开发仓库裁剪 + docs 数字 | P6 |
| `assets/fig5_evidence_gated_ocr.png` | SW1/SW2 证据门控案例:OCR bbox 裁剪 + 置信 0.998/0.992 | 冒烟 export_semantics 审计 | P6 |
| `assets/fig6_final_output.png` | 最终输出全景(检测标注 + 网络叠加,89/165/63) | 冒烟 v102 final JSON | P1/P8/P9 |
| `assets/arch_pipeline.png` / `arch_modules.png` | 端到端架构 / 模块关系 | 本报告绘制 | P3 |
| `assets/dev_repo/*.png`(12 张) | 开发期真实记录:U-Net 训练曲线、导线叠加示例、GT 标注、评分板、页 13 H1 局部、页 1 USB1 问题点等 | `zc-cc1/vision` 仓库 | 备查页/附录 |
| `assets/dev_repo/README.md` | 每张 dev 图的来源与说明 | 本审查编写 | — |

每张自产图的生成脚本保存在 `_audit_workspace/scripts/`(gen_asset_*.py),可复现。
