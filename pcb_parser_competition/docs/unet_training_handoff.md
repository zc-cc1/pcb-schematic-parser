# U-Net训练交接：给接手的agent

目标工程：`C:\Users\Pc\Desktop\新建文件夹\pcb_parser_competition`。U-Net已经合并到该YOLO工程的scripts/src/configs/tests目录；原有YOLO脚本、配置、权重与训练结果保持原样。不要把YOLO权重当作U-Net权重。

## 运行入口

在PowerShell中执行：

```powershell
Set-Location -LiteralPath 'C:\Users\Pc\Desktop\新建文件夹\pcb_parser_competition'
# 检查环境、开源结构哈希与数据路径，不跑GPU训练
.\scripts\run_unet_training.ps1 -Mode Check
# 若修改标签生成配置，可重新准备数据；必须同时复核标签
.\scripts\run_unet_training.ps1 -Mode Prepare
# 用CPU短跑验证链路
.\scripts\run_unet_training.ps1 -Mode Smoke
# GPU空闲后，运行现有草稿标签实验，记录为draft监督
.\scripts\run_unet_training.ps1 -Mode Train -AllowDraft -Device 'cuda:0' -RunName unet_wire_pin_v1
# 如果已完成标签复核并登记approved_cases，则去掉-AllowDraft
```

默认60epochs、batch2、512切片、stride384、AMP、AdamW lr0.0003。可加`-Epochs 80 -BatchSize 1`，或修改`configs/unet_wire_pin.json`。`-Epochs`在续训时表示目标总轮数。训练输出和YOLO分开，位于 `runs/training/<RunName>/`。

检查时GPU已有其他训练占满显存，请串行安排，不要终止其他任务。脚本在空闲显存不足4GiB时拒绝启动；此门槛不保证所有机器/批量都不会OOM。可在空闲后用batch1开始测量。

## 数据状态与口径

300页标签来自原始标注的实际wire路径和pin tip，不是传统图像分割的输出。132页通过当前结构完整性检查，其中103页训练、28页验证、1页排除。逻辑飞线、deleted记录被排除，缺路径不按端点虚构直线。

完整像素/接点人工批准页目前为0；导线以3px描绘源路径，仍需复核线宽、漏线与tip位置。`-AllowDraft`明确运行草稿监督实验，不把草稿称为正式Golden。默认不加该参数时仅使用`configs/unet_label_reviews.json`的approved_cases。

沿用既有case划分，工程级去重尚未确认，不能把验证分数说成独立工程泛化或比赛成绩。当前双输出为导线mask与pin-tip热图；引脚归属、编号、名称、交叉连接分类及完整nets仍待实现。

数据清单和metadata已在目标工程重新生成，全部图像/标签路径指向本目标工程。Python沿用工程原有`.venv-gpu`目录联接；它共享vision工程的GPU环境，不复制或修改环境包。

## 日志、续训与回传结果

训练每20个batch打印一次进度，每轮写入：

- `history.json`：train/val loss及验证导线像素Precision、Recall、Dice、IoU（固定0.5概率阈值）。
- `epochs.jsonl`：追加的逐轮日志；续训时同一轮可能有重跑记录，以history/checkpoint为准。
- `best.pt`：按val loss选择的模型。
- `last.pt`：含模型、优化器、scaler、轮数和历史的续训checkpoint。
- `summary.json`：训练完成后的配置相关信息、checkpoint哈希、训练/验证病例及耗时。

像素指标针对选择的标签；草稿实验的指标不能当成PinF1或NetF1。tip热图仍需点级5px匹配评估，不能拿导线Dice代表引脚识别准确率。

续训示例：

```powershell
.\scripts\run_unet_training.ps1 -Mode Train -AllowDraft -Device 'cuda:0' -RunName unet_wire_pin_v1 -Epochs 100 -Resume 'runs/training/unet_wire_pin_v1/last.pt'
```

使用同一数据manifest、训练类型、切片设置和输出目录续训；恢复模型/优化器状态，不承诺中断前后的随机顺序逐位一致。不要把cpu_smoke_only或partial_training_debug checkpoint用于正式识别。任何带`--max-train-steps/--max-val-steps`的运行都是部分调试。

使用已训练的正式权重进行图片推理：

```powershell
& .\.venv-gpu\Scripts\python.exe scripts/run_unet.py --image data/standardized/0002/0002.png --weights runs/training/unet_wire_pin_v1/best.pt --device cpu --output runs/unet_wire_pin_v1_predictions/0002
```

跑完后把summary.json、history.json、best.pt和几张实际预测的wire_mask/pin_probability图回传给原聊天。正式PNG预测入口不会读取同页标注。

开源来源/模型配置详见 `docs/unet_wire_pin_plan.md`、`src/pcb_parser/vendor/pytorch_unet/provenance.json`及LICENSE。U-Net结构固定于milesial/Pytorch-UNet提交21d7850f2af30a9695bbeea75f3136aa538cfc4a，保留GPL-3.0许可记录。
