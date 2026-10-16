# YOLO11n与现有YOLOv8n替换实验

本次两组11n均未超过当前v8n，建议保留现有部署。即使将v8n也仅用同一30页重新校准阈值，11n仍分别低1.62和3.14个百分点；固定conf0.2时也分别低0.41和2.61个百分点。融合推理中位耗时相近，当前没有替换收益的证据。

本次回答新11n方案能否替换现有部署v8n。新增v8n重训已取消，部分59轮对照权重只归档，不用于正式比较。原protocol.json保留，范围修订见scope_revision.json。

新11n的两个种子使用同一418张训练样本、30张校准页、29张评分页。评分页不用于新模型选checkpoint或调阈值。现有v8n有多阶段PCB训练历史，并曾使用全部59页做开发；它作为实际部署参照，不称为同初始化架构对照。

现有部署工作点：本次29页F1=90.2808%，P=90.3198%，R=90.2418%；原59页F1=88.3043%已复现。

| 模型 | 评分集固定conf0.2 F1 | 同法30页校准阈值F1 | 校准后P | 校准后R | fused中位ms |
|---|---:|---:|---:|---:|---:|
| v8n_adopted | 87.2787% | 88.9563% | 89.2268% | 88.6874% | 116.6 |
| 11n_seed20261012_best | 86.8704% | 87.3374% | 87.7178% | 86.9603% | 117.7 |
| 11n_seed20261012_last | 86.8064% | 87.3523% | 88.5537% | 86.1831% | 114.1 |
| 11n_seed20261015_best | 84.6691% | 85.8134% | 86.2249% | 85.4059% | 113.7 |
| 11n_seed20261015_last | 84.6217% | 85.4040% | 85.9266% | 84.8877% | 114.8 |

## 新11n best对现有部署工作点的差异

- seed 20261012：-2.943个百分点；条件分组bootstrap95%区间[-5.245, -1.179]。
- seed 20261015：-4.467个百分点；条件分组bootstrap95%区间[-6.931, -1.978]。

两次11n best的平均差：-3.705个百分点。未按评分集挑选更高的种子或best/last来作为唯一结果。

## 限制

- Old Dataset300 pages; not independent new-engineering-source test
- Only final-stage recipe tested; not full historical three-stage retraining
- Two seeds describe limited training variation
- Labels partly reviewed; scores are bbox/category proxies, not official ComponentF1
- Historical adopted v8n used all 59 pages for model and threshold selection
- No retrained v8n control used; comparison does not isolate architecture from task-training history.
- Existing v8n checkpoint and deployment thresholds historically used all 59 validation pages. The 29 pages are a protected score subset for the new 11n runs, not an untouched test for the incumbent.

原部署权重与配置未替换。全部原始预测、阈值、逐类/逐页计数和whole/sliced结果保留在evaluation/与comparison.json。

## 训练与评分集AP

| 模型/种子 | 实际轮数 | 校准最佳轮 | 校准mAP50-95 | 参数量 | 训练分钟 | 进程累计峰值MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| 11n_seed20261012 | 120 | 102 | 49.8150% | 2596665 | 41.6 | 4982 |
| 11n_seed20261015 | 120 | 114 | 50.5050% | 2596665 | 43.1 | 4982 |

显存列为训练进程截至该组结束的累计PyTorch分配峰值，未逐模型重置，不能用于比较模型独立显存占用。

评分集AP采用整图推理、conf0.001、NMS IoU0.7、max_det1000；与前面的融合固定阈值F1口径不同。

| 模型 | 评分集mAP50 | 评分集mAP50-95 |
|---|---:|---:|
| v8n_adopted | 61.8522% | 42.9197% |
| 11n_seed20261012_best | 56.6677% | 37.5370% |
| 11n_seed20261012_last | 57.0429% | 38.3146% |
| 11n_seed20261015_best | 61.2324% | 41.9360% |
| 11n_seed20261015_last | 60.7040% | 41.9562% |

## 逐类差异（11n两个种子best平均；v8n为现有部署工作点）

| 类别 | 评分集GT数 | v8n F1 | 11n平均F1 | 差值（百分点） |
|---|---:|---:|---:|---:|
| ground | 283 | 95.1872% | 94.9772% | -0.210 |
| capacitor | 232 | 87.8151% | 81.6039% | -6.211 |
| resistor | 213 | 96.9412% | 95.0233% | -1.918 |
| ic | 132 | 94.8148% | 93.6004% | -1.214 |
| voltage_source | 114 | 89.9563% | 86.7546% | -3.202 |
| net_input | 36 | 76.0563% | 77.9133% | +1.857 |
| led | 21 | 87.1795% | 60.6618% | -26.518 |
| switch | 17 | 66.6667% | 66.6463% | -0.020 |
| other | 14 | 61.5385% | 16.6667% | -44.872 |
| net_short | 14 | 100.0000% | 86.7631% | -13.237 |
| inductor | 13 | 92.3077% | 81.3172% | -10.990 |
| diode | 13 | 66.6667% | 67.8030% | +1.136 |
| buzzer | 11 | 84.2105% | 72.1362% | -12.074 |
| crystal | 7 | 80.0000% | 89.5238% | +9.524 |
| pin_port | 6 | 36.3636% | 0.0000% | -36.364 |
| opamp | 5 | 88.8889% | 91.6667% | +2.778 |
| mosfet | 4 | 0.0000% | 20.0000% | +20.000 |
| net_output | 4 | 0.0000% | 0.0000% | +0.000 |
| motor | 4 | 100.0000% | 100.0000% | +0.000 |
| battery | 4 | 40.0000% | 0.0000% | -40.000 |
| raw_m | 4 | 100.0000% | 33.3333% | -66.667 |
| fuse | 3 | 0.0000% | 0.0000% | +0.000 |
| transistor | 2 | 100.0000% | 43.9560% | -56.044 |
| testpoint | 1 | 0.0000% | 0.0000% | +0.000 |
| raw_block | 1 | 0.0000% | 0.0000% | +0.000 |
| current_source | 0 | 0.0000% | 0.0000% | +0.000 |
| net_bidirectional | 0 | 0.0000% | 0.0000% | +0.000 |
| transformer | 0 | 0.0000% | 0.0000% | +0.000 |
| jumper | 0 | 0.0000% | 0.0000% | +0.000 |
| circle_header | 0 | 0.0000% | 0.0000% | +0.000 |
| antenna | 0 | 0.0000% | 0.0000% | +0.000 |
| esd | 0 | 0.0000% | 0.0000% | +0.000 |
| optocoupler | 0 | 0.0000% | 0.0000% | +0.000 |
| raw_dc | 0 | 0.0000% | 0.0000% | +0.000 |
| connector | 0 | 0.0000% | 0.0000% | +0.000 |

这里只评价检测器，没有声称元件身份/文本、引脚或网络得分已经提高。零GT及极少GT类别的F1不能支持泛化结论。
