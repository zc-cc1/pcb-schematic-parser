# U-Net 引脚推理优化 v2

优化使用已完成训练的第 57 轮 `best.pt`，没有重新训练模型。推理依据 U-Net 概率图，不使用原图颜色阈值、形态学或霍夫线检测。

采用设置：引脚概率阈值从 0.5 改为 0.8，局部峰值窗口从 9×9 改为 5×5，使用对数概率的二次曲线做亚像素定位。坐标始终恢复到原图左上角坐标系。引脚归属、编号、名称和 nets 尚未输出。

导线仍使用原图尺度、0.5 概率阈值。多尺度导线候选在开发图上略有改善，但在保留验收图上 Dice 下降 2.12 个百分点，未采用。支持空间峰值去重的代码已加入，当前校准配置选择半径 0，不额外合并峰值。

## 结果与口径

按 RGB 像素去重后为 26 张图片，固定分成 13 张开发图和 13 张保留验收图，只在开发图搜索参数。原模型曾用这些页选择 checkpoint，因此不是独立工程测试集；标签尚未完整人工复核。

| 指标 | v1 | v2 |
|---|---:|---:|
| 13 张保留验收图的引脚坐标 F1 | 63.89% | 72.58% |
| 26 张去重图片的引脚 Precision | 53.82% | 71.36% |
| 26 张去重图片的引脚 Recall | 71.58% | 69.02% |
| 26 张去重图片的引脚坐标 F1 | 61.44% | 70.17% |
| 26 张去重图片的导线像素 Dice | 79.46% | 79.46% |

引脚按 ≤5px 欧氏距离一对一匹配，仅评价位置，不是正式 PinF1。阈值提升减少了误检，但损失少量召回。导线像素 Dice 不能代表网连接准确率。

## 使用

在项目根目录运行：

```powershell
.\scripts\run_unet_optimized.ps1 -Image 'data/standardized/0002/0002.png' -Device 'cuda:0'
```

也可指定其他 PNG 和输出目录：

```powershell
.\scripts\run_unet_optimized.ps1 -Image 'C:\path\schematic.png' -Device 'cpu' -Output 'runs/new_image_v2'
```

包装脚本优先使用 `weights/unet_wire_pin_v1.pt`，若没有该文件则使用 `runs/training/unet_wire_pin_v1/best.pt`。配置会检查 checkpoint 的 SHA256，拒绝把这次校准结果误用于不同模型。

输出 `probabilities.npz`、`wire_probability.png`、`pin_probability.png`、`wire_mask.png`、`pin_candidates.json` 和 `summary.json`。PNG 推理只读图像和权重，不读标注。

原入口不加 `--inference-config` 时继续使用 v1 参数，便于对照：

```powershell
& .\.venv-gpu\Scripts\python.exe scripts/run_unet.py --image data/standardized/0002/0002.png --weights runs/training/unet_wire_pin_v1/best.pt --device cpu --output runs/compare_v1
```

完整采用结果见 `runs/unet_inference_v2/adopted_report.md`；失败的导线候选及冻结配置也保留在实验记录中。下一步需要针对电容、地符号的 tip 偏移复核监督，关联 YOLO 元件归属，再加入 OCR 引脚编号、名称和网络构建。
