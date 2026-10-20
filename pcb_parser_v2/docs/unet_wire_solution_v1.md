# U-Net 导线分割与网络识别改进方案 v1

日期：2026-10-17 ｜ 适用工程：`pcb_parser_v2`（junction 链接原项目的 data/weights/runs）
前提：本方案不改 YOLO 检测训练配方（imgsz 1536/fp16/120ep 冻结约束仍然有效）；U-Net 与拓扑层可动。

---

## 0. 现状诊断（依据，供执行者复核）

1. **U-Net v1**（`runs/training/unet_wire_pin_v1`）：val Dice 0.794 / IoU 0.659。
   - 像素 IoU 已被标签线宽封顶（标签 3px 固定描边 vs 图内实渲 3~6px，此前已实测诊断）；
   - 第 18/24/28 轮 val 召回崩到 0.33~0.51、精度反升——BatchNorm + batch_size=2 的统计噪声特征；
   - `wire_positive_weight=8` + 阈值钉死 0.5：预测系统性偏宽（val R 0.87 > P 0.71），从我校准过。
2. **标签与数据**：仅 132/300 页过结构门槛（缺一条路径整页出局）；0 页人工批准；`source_group` 为空，工程级去重形同虚设，val 分数可能含泄漏。
3. **拓扑层**（`src/pcb_parser/wire_graph.py`）：框架完整（保护性遮蔽→骨架→路口保守拆分→断口登记→NET_CAND→引脚吸附），但三个关键开关全关：
   - `junction_model_available=false`：JunctionCNN 试点太弱（13 测试例仅 5 例敢答、答对 4），未上岗；
   - `automatic_gap_filling=false`：分割断裂把网络劈成两半，目前只登记不补救；
   - `automatic_same_label_merge=false`：**netlabel 同名并网未做——官方评测按电气网络计分，这是 NetF1 的第一大头，目前整块为 0**。
4. **评测口径**：只有 tile 像素 P/R/Dice/IoU，官方 NetF1 恒为 null；无任何"断裂率/错并/错拆"的拓扑级读数。

## 核心原则

- **先造尺子，再改东西**：没有 NetF1 代理指标之前，任何"改进"都无法裁决。
- **免费的分先捡**（纯逻辑，不训练），训练类修复其次，模型类升级最后。
- 每个自动合并动作必须留 `decision_source` 与哈希链，保守可审计的风格不变。

---

## 阶段 0：先造尺子（全部 CPU，无训练）

### 0.1 离线 NetF1 评测器
`scripts/eval_net_proposals.py`：
- 输入：`data/standardized/<case>/*_reference*.json`（赛事结构参考，由标注转换，作伪 Golden）+ 整链产物（`partial_submission.json` 或逐页图）。
- 输出：逐案例 NetF1/组件命中/引脚命中，并把误差归因到四类：**错并网络 / 错拆网络 / 标签未并 / 引脚未吸附**。
- 坐标系注意：伪 Golden 来自 CVJsonStd（左上原点），输出要求左下原点；评测在左上原点对抗，仅在最终导出时翻转。
### 0.2 骨架级质量读数
- 对 28 个 val 页跑现有 best.pt：断裂率（对标签骨架做 Hausdorff/端点对匹配）、漏段率；
- U-Net 阈值 0.30~0.70 扫描曲线（像素连通性 + 0.1 的 NetF1 各报一条），确定"预测偏宽但连续"的最优工作点，替代 0.5 拍脑袋值。
### 0.3 交叉画风审计
- 全 300 页 `keyPointPaths` 线段两两求交统计：裸交叉 / 圆点连接 / 跳线弧 的比例与位置；同时输出 Dot/Bridge/Flat 的自动标签草稿（为 2.5 铺路）。
- 若 Bridge 占比 <1%，三分类可降级为"圆点 vs 裸交叉"二分类。

**验收**：10 个 val 页的误差归因表 + 阈值扫描图 + 交叉比例统计。产出即 v1 基线，此后每个改动都对比它。

---

## 阶段 1：免费的逻辑分（无训练，优先做）

按预期收益排序：

1. **1.1 netlabel 同名并网**（`automatic_same_label_merge` 打开）
   - 在 `network_membership` 之后加合并器：引脚 `pinname`/网络标签文本（来自现有 OCR/pin_text 结果）同名 → NET_CAND 合并；同名多段先合并集合、保留 `merged_label` 审计字段；GND 类标签特判。
   - 边界规则：若同名 net 的两个成员 net 各自已含 ≥1 引脚且交叉为空（无共享图形节点）才允许纯逻辑合并，防 OCR 串名。
2. **1.2 圆点规则上岗**（0.3 的产物 + 形态学）
   - 检测"小而填充率近圆"的 blob（半径≈1.5~3×线宽的实心圆）；4 度路口圆点在中心 → 该路口 `observed_connected`，不拆；
   - 这直接补上"保守拆分把该连的拆了"这一整类错误，且不需要任何模型。
3. **1.3 gap 有条件自动合闸**
   - 对已有 `gap_candidates`：沿线 U-Net 概率均值 ≥ 高阈值（从 0.2 的扫描里取）+ 两端方向夹角 ≤ 现有 30° + 每个端点只有一个候选配对 → 自动连边并标 `auto_connected`；其余保持 review。
4. **1.4 逐类别 pin_attach 半径**
   - IC/电源/器件本体大的类用 8~10px，两脚小件用 5~6px（参考 CircuitVision 的分级思想），改 `wire_graph_v1.json` 一处查表。

**验收**：0.1 评测器上 NetF1 相对基线提升、"错并"不升（错并可接受 +1~2%，错拆必须下降）；1.2/1.3 各报启用前后消融行。

---

## 阶段 2：标签与训练修复（GPU 重训 U-Net）

1. **2.1 标签重栅格化**：按 0.3 顺带实测的每页渲染线宽画监督（预期 IoU 0.66→0.85 量级）；新增 **ignore 通道**：缺路径/存疑区域不作负样本。改 `prepare_unet_dataset.py` + `annotation_targets`（targets 从 2 通道扩为含 ignore）。
2. **2.2 放宽结构门槛**：`missing_physical_paths/out_of_bounds` 等从"整页排除"降级为"页内 ignore"，训练页 103 → 目标 ≥250；回填 `source_group`（用 `data/manifest.jsonl` 的来源字段即可），train/val 改**按工程分组**划分，先查泄漏再重划。
3. **2.3 结构改造**：DoubleConv 的 BatchNorm→GroupNorm（8 组）；AdamW 加 cosine 日程 + 模型 EMA（decay 0.999）。其余配方（512 切片/384 stride/60ep/batch2）不动。
4. **2.4 校准**：pos_weight 在 {2,4,8} × 阈值 {0.3~0.7} 网格上以 **0.2 的骨架级指标**选点（不以像素 Dice 选点），推理阈值写进 `unet_inference_v3.json` 并入哈希链。
5. **2.5 交叉分类器上岗**：用 0.3 自动标签训练三类（或两类）JunctionCNN——样本从 32 张人工裁剪跳到 300 页×每页几十点；案例级 held-out 测试；**上岗门槛：测试集敢答率 ≥80% 且答对率 ≥97%**，达标才置 `junction_model_available=true`。规则（1.2）先顶着，模型上线后规则降级为兜底。

**验收**：val 曲线无 18/24/28 型崩盘；骨架断裂率相对阶段 1 末再降；NetF1 代理指标提升 ≥ 阶段 1 的一半。

---

## 阶段 3：可选增强（有明确证据再排期）

- 3.1 若 YOLO 侧加 3 类小目标比 2.5 的裁剪分类器更准（同一评测器裁决），并入检测模型省一次推理；
- 3.2 官方管线第 3 阶段（多智能体校正）对接 `review_queue.json`：LLM 只做语义（位号/值/极性/类别混淆），**不改拓扑**——沿用 CircuitVision 的边界教训。

---

## 里程碑与依赖

| 步骤 | 依赖 | 预估 | 产出 |
|---|---|---|---|
| 环境：修复本机 Git Bash（`spawn bash.exe ENOENT`） | — | 重装 Git for Windows | 一切脚本的前提 |
| 0.1~0.3 | 环境 | 1 天 | v1 基线 + 阈值曲线 + 交叉统计 |
| 1.1~1.4 | 0.1 | 1~2 天，无训练 | 免费分落袋 |
| 2.1~2.4 | 0.2/0.3 结论 | 1 天改码 + 1.5 天训练（87min×2） | U-Net v2 |
| 2.5 | 0.3 标签 + 2.3 教训 | 1 天 | junction 上岗或维持规则 |

**红线**：Dataset300 原件只读；所有新标签/新产物写新版本目录（`segmentation_unet_v2/` 等）；采纳判据统一用 0.1 的 NetF1 代理 + 0.2 的拓扑级读数，像素 Dice 只作健康检查。
