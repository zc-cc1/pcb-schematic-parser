# Dataset300 检测器首轮微调实测

## 本轮完成的工作

已从300例源数据导出检测标签，去掉3张完全重复图后保留297张：训练238张、验证59张。
检测类别共35类；训练框10218个，验证框2462个。
看图复核了411个独立元件裁剪图，记录83项人工修正或排除；详情在 `configs/detection_label_overrides.json`。
修正例子包括：ANT中误标的运放、Fuse中误标的电阻/开关、Crystal中误标的电容，以及SEG中的标题文字与连接器。复核证据和原始裁剪图保存在 `artifacts/detection_data_review/`。

0001和0002被固定在验证集。完全重复图、pHash相似图和可用同源文件名线索按组隔离；原始工程来源尚不完整，不能证明已隔离所有同源工程。
检测训练只使用类别和location框，左上坐标归一化成YOLO格式；位号、Name、value、引脚与网络字段未作为检测标签。10个越界框已裁到图片范围，源文件保持可追溯。

原始数据的全部标注尚未人工认证。本次指标来自经局部人工修正的原始检测标签，属于实验指标。

## 训练配置与实际运行

以HR-DONK电路检测权重初始化，在RTX 5060上完成20轮；总耗时4.19分钟。
1024输入、batch=4、2个数据加载进程、AdamW、初始学习率0.001、FP32；细节在 `configs/detector_training.json` 和运行 `args.yaml`。
PyTorch分配显存峰值2710 MiB；这不是整张显卡的总占用。
按类别名称复制了原权重10类在三个检测分支的30个分类输出行，并逐行验证一致；新增类别输出行单独初始化。
最佳权重：`C:\Users\Pc\Desktop\vision\pcb_parser_competition\weights\detector_dataset300_v1.pt`。SHA256：`4353e9be8dd0b4dfb295038fc2465fe7a7f28f23bb91e127ee048ac629642294`。

## 同一验证集的前后比较

两个模型使用完全相同的验证案例、标签版本、1024整图输入、confidence=0.25，以及类别名称匹配、IoU≥0.5的一对一框匹配。

| 指标 | 原始权重 | 微调权重 |
|---|---:|---:|
| Precision | 0.5363 | 0.8474 |
| Recall | 0.3387 | 0.7149 |
| F1 | 0.4152 | 0.7755 |

F1变化：+0.3603。这是类别+框的本地微平均F1，不是比赛要求位号、型号、数值和像素容差的ComponentF1，也不包含引脚/网络分数。

| 类别 | 验证框数 | 原始F1 | 微调F1 |
|---|---:|---:|---:|
| resistor | 399 | 0.5270 | 0.8803 |
| capacitor | 485 | 0.5068 | 0.8642 |
| ic | 282 | 0.3132 | 0.8304 |
| ground | 601 | 0.5276 | 0.8632 |
| voltage_source | 266 | 0.0000 | 0.5797 |
| crystal | 14 | 0.0000 | 0.0000 |
| inductor | 29 | 0.5143 | 0.7692 |
| opamp | 10 | 0.5333 | 0.8182 |
| mosfet | 16 | 0.0000 | 0.0000 |

验证集没有以下类别，因此本轮无法评价它们的泛化：current_source, net_bidirectional, jumper, circle_header, antenna, connector。

晶振与MOSFET在当前confidence=0.25下均没有有效检出，不能把类别表扩展理解为这些类别已具备可靠识别能力。后续需要复核更多稀有类标签、针对性切片增样或调整训练策略。

训练器保存的mAP50/mAP50-95使用Ultralytics默认AP计算口径，不与上面的固定阈值F1混用。最佳checkpoint的训练器指标：

- metrics/precision(B): 0.5246
- metrics/recall(B): 0.3473
- metrics/mAP50(B): 0.3982
- metrics/mAP50-95(B): 0.2537

## 两例整图/切片复查

| 案例 | 模式 | 候选数 | 暂绑定位号数 | 检测中位耗时 ms |
|---|---|---:|---:|---:|
| 0002 | whole | 13 | 7 | 16.7 |
| 0002 | sliced | 14 | 7 | 63.2 |
| 0001 | whole | 70 | 30 | 20.3 |
| 0001 | sliced | 102 | 28 | 143.9 |

预测框、OCR、绑定调试、overlay和局部赛事结构JSON保存在 `runs/s2_finetuned_quick_dev/`。
两例已多次用于调试，属于开发验证案例；更多候选不直接代表更高准确率。Name/value、引脚和网络尚未实现，JSON仍为局部结果。

已保存预测的本地IoU匹配比较：

| 案例 | 模型 | 模式 | TP | FP | FN | F1 |
|---|---|---|---:|---:|---:|---:|
| 0002 | before | whole | 7 | 1 | 5 | 0.7000 |
| 0002 | before | sliced | 7 | 1 | 5 | 0.7000 |
| 0002 | after | whole | 10 | 3 | 2 | 0.8000 |
| 0002 | after | sliced | 11 | 3 | 1 | 0.8462 |
| 0001 | before | whole | 9 | 48 | 82 | 0.1216 |
| 0001 | before | sliced | 23 | 84 | 68 | 0.2323 |
| 0001 | after | whole | 56 | 14 | 35 | 0.6957 |
| 0001 | after | sliced | 64 | 38 | 27 | 0.6632 |

0002的新模型切片F1为0.8462，高于整图0.8000；0001切片F1为0.6632，低于整图0.6957。当前保留两种模式用于调试，后续要改善重复框/碎片框合并，再决定何时启用SAHI。
可视化确认0002的IC和竖向电容已有检测框，但X2晶振被误分类为电容；0001的Y1晶振也仍有漏检。

## 再次运行

项目根目录下：

```powershell
.\.venv-gpu\Scripts\python.exe scripts/prepare_detection_dataset.py
.\.venv-gpu\Scripts\python.exe scripts/train_detector.py --run-name detector_dataset300_v2
.\.venv-gpu\Scripts\python.exe scripts/run_baseline.py --config configs/baseline_finetuned.json --image data/standardized/0002/0002.png --image data/standardized/0001/0001.png --devices cuda:0 --modes whole sliced --output runs/s2_finetuned_quick_dev
```

改变标签/划分后要同时重新评估原始和微调权重，比较工具会拒绝混用不同manifest。

## 后续优先事项

优先针对晶振、MOSFET等稀有类做切片增样与类别均衡训练，并改善SAHI的重复框/碎片框合并；同时补足标签和同源工程复核。
随后推进OCR的位号/型号/数值绑定，再接入引脚、导线骨架和网络规则。
