# v3 优化运行手册（本地副本）

本轮按「召回卡住为主、合并有损耗」的诊断实施优化。原项目行为全部保留，v3 以新增配置/模式的形式并存，可随时回退。

## 变更清单

| 文件 | 变更 | 目的 |
|---|---|---|
| `src/pcb_parser/fusion.py` | 新增：类感知 WBF 融合 + 逐类置信度过滤 | 双路结果合并，救回 SAHI 损失的精度 |
| `tests/test_fusion.py` | 新增：8 个单元测试（含链式合并回归） | 锁定融合行为 |
| `scripts/train_detector.py` | 支持可选 `initial_architecture`（yaml 架构 + model.load 权重迁移） | 可换更大骨干 / P2 头 |
| `configs/detector_training_v3.json` | 新增：imgsz 1536、amp fp16、epochs 120、mosaic 0.35、scale 0.4、hsv_v 0.25、lr0 3e-4 热启动 | 训练侧主力优化 |
| `configs/detector_training_v3_big.json` | 新增：yolov8s-p2.yaml（P2 小目标头 + s 级骨干）+ COCO yolov8s 初始化 | 容量实验（第二顺位） |
| `scripts/download_yolo_weights.py` | 新增：下载 COCO yolov8s.pt 等并记录哈希 | v3_big 前置 |
| `scripts/run_baseline.py` | 新增 `fused` 模式、逐类置信度过滤、`sahi.skip_if_max_side_below` 小图跳过 | 推理侧 |
| `scripts/evaluate_detector.py` | `--mode fused`、`--thresholds` 逐类阈值文件 | 评测口径扩展 |
| `scripts/tune_class_thresholds.py` | 新增：单次低阈值推理 + 离线逐类阈值扫描（与评测同协议） | 替代拍脑袋全局阈值 0.25 |
| `scripts/error_breakdown.py` | 新增：按类/尺寸/来源拆解 P&R、FN/FP 归因、最差案例表 | 定位漏检来源 |
| `configs/baseline_v3.json` | 新增：v3 推理配置（overlap 0.25、IOS 0.45、fusion 块、小图阈值 1600） | 训练完成后启用 |

## 执行顺序

```powershell
$py = '.\.venv-gpu\Scripts\python.exe'

# 0) 对现有 v2 结果做误差拆解，确认优化方向（需要已有 evaluate 输出 JSON）
& $py scripts\error_breakdown.py --eval-json <v2_eval_output.json>

# 1) 训练 v3 主力配置（约 25-40 分钟 @120 epochs，OOM 则加 --batch 2 或 --imgsz 1280）
& $py scripts\train_detector.py --config configs\detector_training_v3.json --run-name detector_dataset300_v3_nano

# 2)（可选）容量实验：先下载 COCO 权重，再训 P2/s 级
& $py scripts\download_yolo_weights.py --model yolov8s.pt
& $py scripts\train_detector.py --config configs\detector_training_v3_big.json --run-name detector_dataset300_v3_s_p2

# 3) 用胜出的权重调逐类阈值（单次推理 + 离线扫描，模式与最终推理一致）
& $py scripts\tune_class_thresholds.py --weights weights\detector_dataset300_v3.pt --config configs\baseline_v3.json --mode fused --output configs\class_thresholds.json

# 4) 三种模式统一评测对比（--thresholds 使用上一步产物）
& $py scripts\evaluate_detector.py --weights weights\detector_dataset300_v3.pt --config configs\baseline_v3.json --mode fused --thresholds configs\class_thresholds.json --output runs\eval_v3_fused.json
& $py scripts\evaluate_detector.py --weights weights\detector_dataset300_v3.pt --config configs\baseline_v3.json --mode whole --output runs\eval_v3_whole.json

# 5) 对胜出配置再做一次误差拆解，决定下一步（数据/类别/网络侧）
& $py scripts\error_breakdown.py --eval-json runs\eval_v3_fused.json
```

## 判据与预期

- 基线：v2 整图 F1 0.8008 / 切片 0.8132（59 例验证集，IoU 0.5，conf 0.25）。
- v3_nano 预期：整图与切片各 +6~10 点（训练分辨率 + fp16 + 更长训练 + 增广放宽）；fused + 逐类阈值再 +2~4 点，合计应到 **0.86~0.90**。
- v3_s_p2 若优于 v3_nano，以其为新基线重跑第 3-4 步；两者都完成后可再用 balanced_v2 权重做一次类别行迁移微调。
- 采用前必须确认：`configs/class_thresholds.json` 的 `micro_f1_with_thresholds` > `micro_f1_at_global_threshold`，否则逐类阈值不值得用。

## 注意

- `detections.json` 里新增 `members`（融合成员数）、`class_confidence_dropped`、`sahi_skipped_small_image` 字段，仅调试用，下游绑定/导出不受影响。
- WBF 输出的 confidence 是成员均值，整体置信度分布会略低于单路，逐类阈值因此更必要。
- 官方比赛 F1（key 级匹配）仍未接入，以上均为类别+框的本地协议；OCR 绑定、引脚、导线、网络四模块进度不变。
- P2 头只在 v3_big 中启用；如训练不稳（NaN/不收敛），优先回退 `configs/detector_training_v3.json`。

## v3_nano 实测结果（2026-10-11，59 例验证集，IoU 0.5）

| 模式 | v2 基线 | v3_nano | Δ |
|---|---|---|---|
| whole | F1 0.8008 (P .8422/R .7632) | **0.8686** (P .8557/R .8818) | +6.8 |
| sliced | F1 0.8132 (P .7906/R .8371) | **0.8586** (P .8249/R .8952) | +4.5 |
| fused+逐类阈值 | —（新模式） | **0.8772** (P .8602/R .8948) | +7.6 vs v2 最优 |

- 逐类阈值：首版扫描脚本曾把置信度阈值误用作 IoU 匹配阈值（已修复），其报告的 0.9142 为虚高值，作废；修正后重调，阈值真实贡献 +2.0（0.8562→0.8760）
- 推理置信度消融（同权重同阈值协议）：whole/sliced 均 0.2 → F1 0.8772；均 0.05 → 0.8679；whole 0.2+sliced 0.05 → 0.8690。对称 0.2 最优，不对称无收益（SAHI 会在 conf<0.1 时强制切换 NMS/IOU）
- tiny 桶召回 48.9% → **82.2%**；small 87.5→89.7；medium 93.5；large 90.9
- 类别：crystal 0.118→**0.849**(R=1.0)，voltage_source 0.250→0.835，ground 0.710→0.904，led 0.936；仍弱：mosfet 0.560(R .438)、pin_port 0.326、fuse/optocoupler/esd/current_source/raw_block F1=0（个位数支持量）
- 已完成的补充实验：runs/eval_v3_fused_lowconf.json（conf 0.05）与 runs/eval_v3_fused_asym.json（不对称），均未超过 0.8772，见上条消融结论
- 产物：weights/detector_dataset300_v3_nano.pt、configs/class_thresholds.json、runs/eval_v3_*.json

## 最终采纳（2026-10-11）

**v3_nano + fused(对称 conf 0.2, WBF IoU 0.55) + 修复版逐类阈值 = F1 0.8830（P .8798 / R .8863）**，确认评测 runs/eval_v3_fused_v2thr.json。相对 v2 最优 +7.0、相对 v2 整图 +8.2。检测阶段收官，下一阶段：导线 U-Net 分割 → 骨架 → NetworkX 连通性 → 飞线归并（赛题权重 0.45 尚未开始挣）；数据侧目标 mosfet(R .438)/pin_port(F1 .326) 走合成渲染。

## U-Net 导线分割 v1（2026-10-11 深夜）

`runs/training/unet_wire_pin_v1`：60 轮 / 87 分钟，草稿监督（-AllowDraft，103 训练页）。最佳第 56 轮：val 像素 **P 0.776 / R 0.813 / Dice 0.794 / IoU 0.659**（固定 0.5 阈值，scope=选中标签，非官方 PinF1/NetF1）。曲线到第 56 轮仍在涨。产物 best.pt/last.pt + history。下一步：run_unet.py 出预测图目检 → 骨架提取 → NetworkX 连通 → 飞线归并；tip 热图需点级 5px 匹配评估。y11s stage-2 暂停于 20/60（best.pt 已存）。

## yolo11s 最终裁决（2026-10-12）

y11s（30+30 分段续训，共 60 轮）训练协议 mAP50 0.7280 > v3_nano 0.6876，但工作点 fused F1 **0.8741** < 0.8830（P .8766/R .8716）——mAP 赢、F1 输：v8n 三代热启动累计 460+ 轮，工作点置信度标定更准。按预定规则 **v3_nano（v8n）保持采纳**。y11s checkpoint 保留（yolo11s_stage1/stage22.pt）：可选后续 = 继续喂轮次，或 v8n+y11s 双模型 WBF 集成（检测线最后一个便宜实验）。

## U-Net IoU 偏低根因诊断（2026-10-12）

**结论：IoU 0.659 是标签线宽造成的天花板，不是模型质量问题。**

实测（28 页验证集中 10 页抽样）：
- 标签是**固定 3px 描边**（`configs/unet_wire_pin.json: label_line_width=3`，status=draft_rasterized_source_geometry_not_certified），距离变换实测中位 2.0px
- 图上**实际渲染的导线宽 2~4px，中位 3.0px，是标签的 1.5×**（部分案例 2×）
- IoU 对线宽的敏感度实测：对标签做 +1px 膨胀 → IoU 掉到 **0.599**；-1px 腐蚀 → **0.336**；整体错位 1px → **0.607**
- 即：在 2~3px 细线上，任何 1 像素级的宽度/中心线分歧就吃掉 35~40% 的 IoU

因此一个"完美分割可见导线"的模型（预测实际渲染宽度 3px），对标 2px 标签的 IoU ≈ 2/3 = **0.67**——与实测 0.659 几乎一致。旁证：验证集 P 0.776 < R 0.813，说明模型预测比标签更宽，与"标签偏细"一致。

官方 93.70% 不可直接对照：官方 mask 与被渲染图像**同源同宽**（同一工程文件导出）且经人工修正，标签-图像天然一致；本项目的标签是沿源几何中心线以固定 3px 重绘的草稿。

**处理建议**：① 比赛真正考核的是赛题 NetLineF1（两点线段集合）与连通性，依赖**骨架/拓扑**而非像素宽度——应改用骨架级或下游连通性指标评估，不要把像素 IoU 当质量指标；② 若报告需要更高 IoU，按每页实测线宽重新栅格化标签（正是交接文档列为待办的"复核线宽"项），预期 0.66 → 0.85+；③ 阈值 0.5 是任取值，可扫描。
