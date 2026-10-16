# U-Net训练交接：给接手的agent

## 最新进展：节点切分已试验，保留v1解码

用户继续要求核对分段并实现预测连接点切分。已新增`src/pcb_parser/wire_segment_nodes.py`、评估脚本`scripts/evaluate_wire_segment_nodes_v2.py`，PNG入口增加可选`--node-config`。原始2266参考段及全部旧标签/预测/权重保留。分段审计发现267段≤5px几何靠近源pin接入位置；同一edge严格共线重叠合并的敏感性解释为2085段，未被用来选参数或替换主评分，不能据此宣称模型提升或源标注错误。

第一轮`runs/wire_segment_nodes_v2_20261015`：端点吸附3px使开发组原始端点代理F1 53.20%→52.04%，未采用。第二轮`runs/wire_segment_nodes_v2_split_only_20261015`（评估脚本加--split-only）：保持端点、只按预测pin切分，开发53.20%→53.30%，另一13页51.96%→51.94%（TP+1、FP+4），未通过检查条件，最终仍采用v1。当前26页端点proxy仍52.43%，不是官方分数。configs/wire_segment_nodes_v2.json因此enabled=false；实验配置在第二轮experimental_config.json，候选逐页保存在rejected_candidate.json。

48项wire测试通过，0170真实PNG/CPU推理用实验配置产生2处切分，无参考读取。原始v1默认解码未改变，也未重训或替换权重。详情与用法见docs/wire_segment_nodes_v2.md；检查记录为第二轮completion_checks.json。切勿把只改参考解释的分数当成识别改进，或因启用实验功能而宣称分割模型已经提升。

## 当前可运行的两点线段解码 v1

用户已要求继续按线段端点/折点处理，已实现`src/pcb_parser/wire_segments.py`、PNG入口`scripts/run_wire_segments.py`及13开发/13检查的对照脚本`scripts/evaluate_wire_segments_v1.py`。配置`configs/wire_segments_v1.json`选择4px最大缺口，组件预测主体限制新增桥接、预测tip保护，原mask不改写。无组件预测时CLI关闭跨缺口合并。输出两点线段、来源审计、左下H-y的edges候选；尚未归网，不能当完整nets。

最终采用有组件保护的实验目录`runs/wire_segments_v1_guarded_20261015`，旧无主体保护实验`runs/wire_segments_v1_20261015`保留审计。修订组26页端点几何代理F1=50.66%→52.43%，TP1075→1086、FP903→791、FN1191→1180；另一13页50.62%→51.96%，并非独立测试。按同一解码器原v6=50.57%、对照微调=52.04%。正式NetLineF1仍未计算；不能把proxy当比赛成绩。

已通过34项wire测试（16项新增），0007/0022 PNG真实CPU推理跑通。旧0036电感内部误合并被组件保护阻止。微调组仅1处缺口实际通过合并，主要收益是降低噪声及多余碎段；0208缺响应仍未解决。CPU/GPU少量阈值边界像素可能导致分段变化，fresh_png与冻结GPU评估分开记录。此阶段未重新训练或覆盖默认权重。用法与限制见`docs/wire_segments_v1.md`；检查记录见实验目录completion_checks.json。

## 最新结论：按赛题线段端点评估，不以像素连续性单独否决模型

2026-10-15用户要求重新核对0803Rev1导线要求。原PDF共30页，SHA256为45eedbc195ad71bbb03044c4606ea85ccf3eccebbb3d9b3304d6e325e1acc5e8。第8页线段在matched net pair内按两端默认5px匹配，第20页要求折线拆成两点edge、数量准确，第30页要求放在正确net下。NetLineF1权重0.10；0.35属于NetHypergraphF1。不能把2px严格路径连续性当成赛事要求，也不能仅因该统计下降就否决微调模型。

两组现已各20轮完成并完成28/26页整图评估，26页原v6 IoU/Dice=72.7127%/84.2008%，control_v6=74.4812%/85.3745%，reviewed_v7=74.4407%/85.3479%。实验summary中baseline_v6推荐是旧采用门槛的历史结果，不等于按赛事几何标准仍最优；没有自动替换默认权重。

新检查不要求预测像素连续，使用相同26页的2266段非零物理源参考：修订组沿线5px采样覆盖93.56%，1890段有≥90%沿线与两端几何证据，203段几乎无响应。现有mask-only骨架/RDP解码全图一对一5px端点代理F1=50.66%（原v6=48.87%、control=50.54%），不是官方NetLineF1：未按matched net限制，参考未认证，分段口径和完整官方规则未核对。修订组52条原严格检查失败的单段路径已有端点几何匹配结果，说明旧指标会低估可表达几何。0208修订组仅51个正像素、66段参考中65段几乎无响应，是真实灰线漏检方向的明确反例；不能宣称所有导线均已表达。后续优先评估线段解码、碎段合并/折点和端点精度，同时处理缺少响应的图；net归属仍须正确。

报告`runs/unet_competition_requirements_20261015/report.md`；逐段证据、纯预测解码结果和保护哈希`geometry_audit.json`；复现脚本`scripts/audit_unet_competition_geometry.py`。评估只读源参考，推理/解码不读真值。下方队列状态段为此前启动记录，当前训练已经完成，不要据旧段再次启动。

## v7继续优化：已排入资源队列

2026-10-15用户要求继续优化。新版`data/segmentation_unet_v7_train_review_20261015/training_manifest.jsonl`在v6上仅修订4张训练页0025、0073、0088、0093；新增3986、删除938个导线像素，移动0088的2个pin。全量300行、126份派生源标签重生成、ROI限制、回滚、冻结28页验证及43项U-Net测试均通过。完整512/batch2的CPU暖启动训练链路试跑通过。局部修订报告：`runs/unet_v7_label_review_20261015/report.md`。

实验目录`runs/training/unet_v7_finetune_comparison_20261015`已有队列工作进程：从v6第60轮相同权重，control_v6与reviewed_v7各20轮，学习率3e-5余弦至3e-6、相同种子/图像/批次顺序/损失。唯一组间差异是4页训练标签。用户已要求先等YOLO训练完成：队列须确认`experiments/y11s_matched_control_20261015`控制器状态completed、training_record状态trained且PID21660已退出，之后才检查≥4096MiB空闲显存。训练、评估、失败、记录缺失或进程仍在均不能放行；最多等待12小时，不终止YOLO。准确状态读progress.json，实际PID读worker.json，日志worker.log/worker.err.log；不要重复启动。完成后会自动评估原v6与两组候选的28/26页整图IoU/Dice、局部连通与pin坐标F1，写evaluation/report.md。默认权重不会自动替换。

原仅检查显存的队列曾在YOLO运行中提前放行；已停止自己的U-Net进程13908并保留control_v6第7轮完整last.pt（历史、优化器、调度器及随机状态均核验通过），等待条件满足后从第8轮恢复。reviewed_v7尚未训练。新队列已配置严格的YOLO完成门槛，7项门槛测试通过；scheduling_change.json记录修订与断点哈希，旧日志和协议保存在*_before_yolo_gate文件。不要删除现有断点或从头重复已完成轮次。

配置`configs/unet_v7_finetune_comparison_20261015.json`；入口`scripts/finetune_unet_v7.py --config configs/unet_v7_finetune_comparison_20261015.json --allow-draft --queue`。如异常退出，先确认旧工作进程已结束；原命令会从组内last.pt恢复并跳过已完成组，独占文件锁可阻止重复并发。不要把cpu_smoke.json当作GPU训练结果。

## 2026-10-15本地标签修订 v6

当前副本工程位于`C:\Users\Pc\Desktop\vision\pcb_parser_competition`。标签修订最新训练清单为`data/segmentation_unet_v6_corrected_relations_20261015/training_manifest.jsonl`，171训练+28原验证，使用时显式传入`--manifest`和`--allow-draft`。以下原始交接段落记录v1历史配置，默认入口仍指向v1，不能据此认为会自动使用v6。

此前已恢复679条几何路径；v6对剩余10条错误关联执行原图复核后的撤销/重新归属，同时恢复9条旧线路、补2条真实连接、校正5条已有线路和7个缺失pin。新增1168个导线正像素、删除117个错误正像素，6页取消4425像素ignore。3条非导线轮廓仍排除，其他3页仍保留675像素ignore。10条原关联审计保留为deleted，待查队列归零不等于整页独立认证。0101/0102各1个其他无归属根级pin仍未处理，继续排除联合训练。

全量300行核验、122份源标签重生成、修订可回滚、双训练读取器和43项U-Net测试均通过。原28页验证保持原样。标签修订阶段未更新权重，不能宣称Dice/IoU提升。报告见`runs/unet_relation_repair_v6/report.md`，检查结果见同目录`final_checks.json`。复现脚本为`scripts/repair_unet_relations_v6.py`与`scripts/verify_unet_relations_v6.py`。

2026-10-15已通过512切块/batch2的GPU试跑，并启动独立v6重训，输出目录为`runs/training/unet_wire_pin_v6_corrected_relations_20261015`。从头初始化，60轮、171训练/28原验证、1719/150切块、AdamW lr0.0003、AMP。启动参数、PID和清单哈希见该目录`launch.json`；日志为`train.log`和`train.err.log`；每轮完成后生成`history.json`、`last.pt`和`best.pt`。训练期间不要修改该清单及引用标签；默认权重尚未替换。试跑目录`unet_wire_pin_v6_preflight_20261015`仅有1个训练batch和1个验证batch，属于partial_training_debug，不用于质量评估或正式推理。

v6现已完成60/60轮、51600个训练batch，用时8775.42秒（约2小时26分）。最佳验证损失位于第60轮，best.pt的验证导线像素Precision=0.79706054、Recall=0.85954191、Dice=0.82712293、IoU=0.70520855，固定阈值0.5。完成结果见该运行目录summary.json；completion_checks.json确认权重可在CPU加载、模型张量有限、清单和权重哈希匹配、错误日志为空。训练进程已结束；默认权重仍未替换。这些是草稿标签下的切块像素指标，不是官方NetLineF1或比赛成绩。

随后统一整页评估口径：v6在28页整图IoU72.7798%、Dice84.2457%；与旧v1相同26页整图IoU72.7130%、Dice84.2010%，旧v1对应65.9218%/79.4613%。不能把旧整页指标和新切块指标直接相减。固定阈值0.5；探索阈值0.55只提高IoU约0.0232个百分点，未采用。局部路径连通1150/1548→1218/1548。0025 KEY4在y369、x[50,140)的90px真实导线仍为0标签且90px均参与背景损失；源JSON附近也无对应坐标，说明结构检查通过不代表像素完整。详细对比、论文来源及改进建议见`runs/unet_v6_paper_comparison_20261015/report.md`；未修改标签或权重，未再次训练。

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
