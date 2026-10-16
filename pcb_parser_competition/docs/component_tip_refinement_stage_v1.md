# tip双尺度试验与被动器件编号数据

当前默认推理仍为 `run_pin_output_rules.ps1`。本轮增加 `run_component_tip_refinement.ps1` 独立试验，没有替换原模型或训练。原U-Net热图native/1.5倍双尺度确认缺失tip候选，保留所有旧tip/文字，新增点的编号与名称保留未知。

```powershell
.\scripts\run_component_tip_refinement.ps1 -Image data/standardized/0149/0149.png -Device cpu
```

该入口需要现有PyTorch环境和U-Net权重。原PNG全链路运行，不读源标注、CAD或同页参考JSON。产物在 `runs/component_tip_predictions/<图片名>`，包括refinement_audit、引脚叠加图、物理/逻辑网络候选和output_rules。原学习导线概率不变；新增pin保护区可能改变导线图，因此只保留为实验。

26页草稿归属＋位置联合代理F1为65.12%→65.21%，新增5点中3点符合参考的类别/归属/5px位置，精确率略降，暂不采用默认。电源检测无owned tip数量64→63不是准确率。模型误分类为resistor的F2、+1V8 tip语义和标注缺失需要单独核验。

300页原标注审计发现4722个R/C元件、4563个有效双tip对象，原pinNumberByOCR/VLM没有一组合格1/2对。另存人工覆盖不包含在此数量。按CAD源ID生成的编号方向不一致，不能作为可靠编号训练监督。

40个未批准训练划分复核样例在 `data/passive_number_review_v1/review_queue.json`。A/B只表示源ID；number不可见或约定未知时保持null，批准训练目标当前0。该队列由离线审计脚本生成，PNG推理不读取它。

接下来的训练应使用可见编号＋核验tip/归属，或取得可靠的元件库与官方编号规则。当前默认模型和所有原标注保持原样；结果与转移验证见 `runs/s13_component_tips/report.md`。
