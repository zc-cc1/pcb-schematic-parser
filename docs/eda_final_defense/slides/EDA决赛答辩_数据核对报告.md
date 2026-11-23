# EDA 决赛答辩 · 实验数据核对报告(队友 PPT 融合版)

> 日期:2026-11-23 · 分支:`review/eda-final-defense-teammate-integration`(基于 onsite-final @65e8bf5)
> 核对对象:队友 PPT `eda2606k5_决赛答辩.pptx`(11 页)引用的全部实验数字
> 核对范围:本机全部可用原始记录——v102 OCI 镜像解包(`_audit_workspace/v102_composed`)、L16/L18 提取层、开发仓库快照(github-vision / vision-fetch 含 git 历史)、smoke 回放产物、赛题六 PDF
> 原则:**只有本地原始记录可溯源的数字才写入 PPT;不可溯源的保留原记录并记录差异**

---

## 一、队友 P7:v100/v102 对照表(97.46 → 99.02,+1.56 分)

### 结论:**不确认,不写入 PPT;备份 D 保留原始记录 98.93(v101-score)**

队友表格内容(引用其"Problem 6 Design Report §2.1–2.3 + November 6, 2026 export comparison"):

| Metric | v100 | v102 |
|---|---|---|
| Components (F1) | 94.80% | 100.00% |
| Pins (F1) | 98.43% | 98.43% |
| Connectivity (F1) | 98.64% | 98.64% |
| Wires (F1) | 98.86% | 98.86% |
| Composite score (/100) | 97.46 | 99.02 |

任务书要求的五项核实,逐条结果:

| # | 核实点 | 结果 | 证据 |
|---|---|---|---|
| 1 | 是否同一组 13 张公开开发图 | **无法确认** | `gt1106`(注解集名)与 `97.46/99.02/99.2631/94.80/98.43` 在本地全部 md/json/txt/py 及 git 历史(`git log -S`)中 **0 命中** |
| 2 | 是否复用相同上游预测 | **无法确认** | 同上;"frozen upstream"仅见于队友 PPT 备注的自述 |
| 3 | 是否同一评分器 | **无法确认** | 队友备注给出的权重(0.30/0.25/0.35/0.10)本地无对应评分器配置记录 |
| 4 | 是否仅改变导出语义 | **机制可证,分数不可核** | v102 镜像内 `EXPORT_SEMANTICS.md` 明示 legacy→observed 切换、`package_manifest.json` 记录 `final_export_semantics=observed_v1`——"仅导出语义差异"的机制存在;但该次评分实验的原始记录文件(`export_semantics_optimization_20261106` 报告)**不在包内**(此前审查已记录:source_audit.md:214) |
| 5 | v102 是否为最终代码对应版本 | **是** | L18 overlay = v102 镜像 = 最终提交物;但队友表中 "v100" 版本名不在我们版本链中(本地链为 v101-score → v102) |

内部自洽性(佐证但不构成原始记录):按队友备注权重复算,0.30×94.80+0.25×98.43+0.35×98.64+0.10×98.86 = 97.4575 ≈ 97.46;v102 = 99.0175 ≈ 99.02。表格**数学上自洽**,但四个前提条件均无本地原始记录支持。

**处理**:
- PPT 不引用 97.46/99.02/+1.56;
- 备份 D 保留 `Local development replay: 98.93 (13-page public set, v101-score)` 原文;
- 新增 QA **Q20**:若评委/队友引用该表,答复口径见 QA(实现与单测可展示,幅度以设计报告为准)。

---

## 二、队友 P4:训练数据口径(238 训练页 / 180 训练局部图 / 59 验证页)

### 结论:**确认,已写入备份 B**

| 事实 | 原始记录 |
|---|---|
| 300 例源数据导出检测标签,剔除 3 张完全重复图后 297 张:**训练 238 / 验证 59**;验证框 2462;0001/0002 固定在验证集 | `pcb_parser_competition/docs/detector_training_report.md` |
| 第二轮:**保留 238 张原始训练页,新增 180 张训练局部图**(合计 418 训练样本);验证沿用第一轮 59 页 | `pcb_parser_competition/docs/detector_balanced_training_report.md` |
| **局部图父页只来自训练集**,父图组与源图像哈希均不跨训练/验证(与队友备注 "the detector split is made at original-page level, so training crops inherit their source split" 一致) | 同上("新增局部图的父图只来自训练集…源组交集为 0") |

**处理**:备份 B YOLO 行 Training data 更新为 "238 train pages + 180 crops · 59 val pages";新增脚注一行说明 original-page 级划分与 2462 验证框。P5 主页不引入训练口径(保持推理叙事)。

---

## 三、端口方向规则(VOUT / net_output)

### 结论:**已在最终代码实现且测试通过,可写为已实现能力**

| 核实点 | 结果 |
|---|---|
| 最终代码实现 | `v102_composed/src/pcb_parser/export_semantics.py` `_port_direction()`(L25-78):五顶点凸多边形红旗、面积 180~8000 px²、长宽比 1.4~5.8、与元件框重叠 ≥0.65、**必须有且仅有一个引脚端子**;尖端=单顶点侧;端子在尖端侧 → `net_input`,对侧 → `net_output`;命中后 `value→null`、说明文字保留在 Name;审计规则名 `flag_tip_relative_to_terminal` |
| 默认行为 | `normalize_export_semantics(..., profile='observed')` 默认启用;legacy 可用 `PCB_EXPORT_SEMANTICS=legacy` 整体切回(不逐图选择) |
| 单元测试 | `tests/test_export_semantics.py` **14 项本地全过**(2026-11-23 实测,含"端子在尖端侧→input/对侧→output"的 4 旋转 × 2 镜像方向子测试、双端子不强制、无几何保留字段) |
| 文档 | 镜像内 `EXPORT_SEMANTICS.md`(随包提交) |
| 官方示例页核验 | smoke 回放 `rp2350_full_export_semantics.json`:changes=0/skipped=0;最终 Pred.json 含 17 个 `net_bidirection`(全部为 GPIO*/QSPI*/USB_D± 等排针连接器,该页**不存在**带旗形箭头的方向端口)→ 规则"证据不足不动"的行为与该页一致,不构成反例 |

**素材来源说明(P8 端口裁切)**:本机真实图(官方示例页 PNG、赛题六 PDF 全部插图)按同判据扫描 **0 命中**——本地没有可裁的端口符号样本。P8 采用的端口裁切图来自队友 PPT(公共示例 1 的真实局部,含其预测叠加标注:橙色元件框、紫色端子点),页面标注 "public example 1" 如实标明来源。队友备注中的 "nine port changes / eight Name fallbacks removed / one switch recovered" 数字同样属于设计报告口径,**未写入 PPT**(同第一节原因)。

---

## 四、队友其他数字的处理

| 队友数字 | 处理 | 依据 |
|---|---|---|
| 公共示例 1:58 components / 133 pins / 40 nets(P8) | **不采用**——仅为该案例局部统计,与我们的官方示例页(89/165/63)不同页不同口径;任务书明确不得替换 | 任务要求;本地无示例 1 预测记录 |
| 公共示例 12:21/37/16(P9) | 不采用(未进入我们的页面) | 同上 |
| 队友 P10 未来方向(独立评估/密集引脚与交叉走线) | 吸收为 P12 右栏第 1、2 条,与我们的"结点 CNN 转正/难例数据"合并;无新数字 | 任务书 P12 要求 |
| 队友 P3/P4/P5 版式(共享 ID 链/模块→证据/大图三步) | 吸收,配图全部换为我们自己的真实管线产物 | 任务书 P3/P5/P6 |

---

## 五、遗留事项与风险提示

1. **官方成绩截图/提交记录**仍待人工提供后插入 P9(严禁伪造);参赛 ID 待核实后可加回封面。
2. **Word 技术报告**仍为赛题硬性要求,待人工完成。
3. 封面与 THANKS 页背景由组委会官方模板自带(含国防科技大学等合作方 Logo,onsite-final 即如此);如需移除需修改模板版式层,请用户确认后再动。
4. **中文对照版**(`EDA精英挑战赛_决赛答辩_中文对照版.pptx`)本轮未同步新版式,仍为 onsite-final 内容;如需同步请另行安排(build_cn_deck.py 需按本轮六页重建)。
5. PPT 素材合规性:全部图像来自真实管线产物(fig1b/fig3b/fig3d/fig2c/fig5、官方示例页 PNG、队友真实端口裁切);唯一合成是 P11 右图(两份真实渲染叠加:fig3b 网络层 + fig1b 绿色元件框层,几何同源、无虚构内容),页面图例如实标注。
6. 备份 A 右侧新增"39 结点骨架细节"(来自原 P6 的 fig3c 局部),满足"39 个结点及更细的骨架图放入备份材料"。

---

## 六、本轮改动清单(PPT)

- P3:新增 "PNG → Components·Pins·Nets(共享 Component ID)→ Target JSON" 示意条;四组问题/对策结构保留;正文 13.5→14pt
- P5:改为 "Schematic PNG → Visual modules → Evidence passed downstream" 三栏流;证据列配真实局部图(YOLO 框 / SW1 OCR 令牌 / wire mask+pin heatmap);保留 F1 0.883、431/287/115/29、模型来源四条(JunctionCNN OFF);去掉三张整页缩略并排
- P6:改为单张大图(真实局部预测叠加,~50% 版面)+ 右侧三步(Pin ownership / Wire tracing / Net membership,含 T/cross 保守拆分、断线硬证据、飞线四门电源合并);238/362→63 保留;三张小图移出(骨架细节入备份 A)
- P7:数字不动;文字 12→14pt 级提升
- P8:新增 "Image-based export rules" 端口方向规则面板(真实裁切 + 规则 + JSON 字段);三张校准卡压缩;U3 1~2px 保留;U3 JSON 片段移至 P11(去重)
- P11:改为左原图/右叠加对照 + 图例(元件框/引脚端点/按网着色导线)+ 89/165/63 大数字 + 真实 JSON 片段;删除 U3 bbox 重复项
- P12:左右两栏(What the system delivers / Future development)+ 底部一条工程能力说明;JunctionCNN OFF 事实保留
- 全稿:内容页标题 22→26pt,副标题 13→16pt,备份表 B/D 单元格 ≥14pt,图注 ≥14pt,关键数字保持大号
