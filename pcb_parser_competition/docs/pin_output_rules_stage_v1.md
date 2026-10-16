# 引脚字段与网络候选出口

本阶段新增入口 `scripts/run_pin_output_rules.ps1`，调用现有YOLO/SAHI、U-Net及RapidOCR管线后生成保守部分JSON和独立候选JSON。上游模型、标注、已有编号/名称、tip点及原物理图不改动。

输入为PNG，或PNG加完整性校验通过的模型pin/OCR/网络缓存；输出为JSON、逐网络核验图、字段证据与电源缺口队列。默认输出目录为 `runs/pin_output_predictions/<图片名>`。

```powershell
.\scripts\run_pin_output_rules.ps1 -Image data/standardized/0002/0002.png -Device cuda:0
```

保守出口 `partial_submission.json` 只补已接受唯一普通标签的名称。候选出口 `submission_candidate.json` 另采用“已有编号作被动器件名称”的开发约定；需要至少2视图确认同一位置完整数字、置信度≥0.98、编号分配间隔≥15、合法编号范围及已确认元件身份。两端无文字时保留未知编号；不用空间排序猜1/2，不为MOSFET/IC/二极管套用数字名称。

赛题PDF第25页明确示例RUN_1的pinname为RUN，R1的pinname为1；该页未给出所有被动器件名称的通用规定。电容、电感和晶振的扩展在候选profile中标明provisional，官方Golden未验证。自动格式/约定补全只统计出口覆盖，不计OCR准确率。未知pinname不直接填空字符串。

电源检查读取模型框、图边和接入关系；若电源框6px范围内有网络边却没有所属且已接入的tip，则扣留该网络并给出旁路提示。该检查为保守几何门槛，需要人工复核，未创建缺失tip或合并GND/VCC。

候选几何出口把原骨架路径压缩成两点edges，保持路径端点与连通关系、偏离≤1个原图像素；不跨断线造连接。保守出口保留原edges，便于比较。所有model网络成员必须有完整字段，不能删掉未决成员使hyperGraph通过。

26页旧开发数据：原88个完整字段pin；保守93、net0；候选108、net1。0002中 `(C86.1,IC2.R8,X2.3)` 是目前唯一导出的候选网络，原图局部连接已视觉核验；完整原理图、官方pinname/坐标/NetLine评分尚未完成。实现与验证记录见 `runs/s12_pin_output_rules/report.md`。

主要旁路文件：`pin_field_changes.json`（字段证据）、`power_review_queue.json`（电源缺口）、`candidate_edge_geometry.json`（压缩偏离）、两份coverage与 `network_review_manifest.json`。离线视觉核验在 `data/reviews/pin_output_rules_v1.json`，不作为推理输入。

下一步应补充训练/独立核验用的被动器件编号和电源tip数据，区分源CAD的自动编号与图像可恢复的物理编号，并完善电源身份与逻辑作用域。上述候选出口目前不能当成已就绪的赛事提交。
