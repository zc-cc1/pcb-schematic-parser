# U-Net 标签修正与对比度增强重训

本轮已完成两组各16轮微调，起点为冻结的v1第57轮权重。普通组采用修正后的训练标签，增强组采用相同标签并对训练输入做对比度增强。两组的初始权重、训练划分、随机种子、切片顺序、优化器和学习率计划一致。103页训练、26页去重验证；验证和PNG推理使用原图。

4页中的9个训练区域经过agent视觉检查。新标签副本补充0073中可见的CC2/VOUT导线，修正0088中偏移的导线路径和两个电容tip。原始数据不变，完整人工批准页仍为0。修正规则见 `configs/unet_retraining_v2_label_corrections.json`，标签来源和哈希见 `data/segmentation_unet_v2_reviewed_draft/summary.json`。

训练配置见 `configs/unet_retraining_v2.json`：原尺寸512切片、stride384、batch2、AdamW、学习率0.0001余弦衰减到0.00002。增强包含30%原样输入，其余随机gamma 0.55～1.8和笔画对比度0.75～1.2，不改变坐标、标签或白色padding。损失函数保持v1的导线加权BCE+Dice和tip BCE，本轮没有同时加入新连通损失。

| 模型 | 导线Dice | 导线召回率 | 连通参考路径 | 引脚坐标F1 |
|---|---:|---:|---:|---:|
| 原v1 | 79.46% | 87.62% | 1150/1548 | 70.17% |
| 普通微调，第7轮best | 79.10% | 86.52% | 1024/1548 | 69.05% |
| 增强微调，第10轮best | 79.52% | 86.39% | 1030/1548 | 69.87% |

训练各自完成16轮，best按原图验证损失选择。两组均未通过预先记录的采用条件。X1线段响应提高，但末端仍断开；DOUT出现更多漏分割。因此保留原默认模型，两个新权重保存为实验候选。检查规则和完整逐页结果见 `runs/unet_retrain_v2/evaluation/report.md`，结论见同目录 `conclusion.md`。

19项相关测试通过。增强模型从正式PNG入口对0036推理，其导线/tip概率数组与评估输出完全一致，分割掩膜差异0像素。记录在 `runs/unet_retrain_v2/fresh_png_verification.json`。

候选模型可以单独运行：

```powershell
& .\.venv-gpu\Scripts\python.exe scripts/run_unet.py --image data/standardized/0036/0036.png --weights runs/unet_retrain_v2/contrast/best.pt --inference-config runs/unet_retrain_v2/contrast/inference_config.json --device cuda:0 --output runs/unet_retrain_v2/new_png_review
```

准备、训练和评估脚本分别为 `scripts/prepare_unet_retraining_v2.py`、`scripts/retrain_unet_v2.py`、`scripts/evaluate_unet_retraining_v2.py`。版本化标签和训练/评估目录拒绝覆盖已有结果；再次训练需使用新实验目录和对应复核记录。

这些指标基于草稿标签，参考路径检查只评价局部连通；验证图参与过模型选择和后续开发，不是独立工程测试，也不是正式PinF1/NetF1。下一步应扩大有效标签复核，处理更多漏标/错位，再验证端点权重或连通约束。
