# 导线导出更新 v2（2026-10-20）

批处理入口默认启用以下修复：

- 引脚处分割导线时，保留骨架真实路径、折角和路径长度；切点来自路径最近点，不再将整段折线拉成端点连线。
- 对已经通过同名电源合并检查的模型网络，按实际物理分支导出合格导线。电源符号、已有引脚号、唯一导线归属及标签必须一致；不补画跨页或跨区域的长连线。
- 当同一电源符号的其他引脚候选都明确指向同一主候选时，恢复被重复候选阻断的主接触点。归属不明、多主候选、未附着导线等情况保留原结果。
- 对已有导线网络，恢复唯一、已编号且明确附着的电源接触点，也允许其他候选均明确指向唯一主候选的情况。电源类型和值来自已有器件记录；该成员恢复无需上游已形成同名电源合并组。新物理分支仍沿用原有合格电源组检查。

新增模块为 `src/pcb_parser/power_wire_export.py`，由 `agent.finalize_power_wire_export` 在 target 格式结果写出前调用。它只读取本张图的 `engineering_graph_proposals.json` 与 `engineering_pin_text_result.json`，并核对图像哈希、尺寸和坐标系。新分支经过原有导线清理；已有器件、引脚字段与导线几何保持原值，已有网络只允许增加有明确证据的电源接触成员。

每页处理日志新增 `power_wire_export` 摘要，详细来源记录在 `logs/<case_id>_power_wire_export.json`。如果该导出步骤发生异常，保留原有效预测，并在该页记录 `power_wire_export_error`。

审计版本为 `observed-physical-power-wires-v2`。`existing_power_contacts` 记录本轮恢复的已有网络接触成员及其候选、号码和恢复理由，`existing_power_contact_count` 为其数量；原有 `primary_contacts` / `primary_contact_count` 保留并包含唯一候选与显式重复主候选的恢复总数。

`PCB_POWER_WIRE_EXPORT=0` 可关闭新增电源导出；路径切分修复仍保留。`PCB_WIRE_CLEANUP=0` 同时关闭新增分支的导线清理。sample/project 格式保持原流程。

已有 13 页缓存通过生产函数验证：从 v1 到 v2，平均导线 F1 为 0.5568 → 0.6064，7 页提升、6 页不变；连接关系 F1 为 0.5345 → 0.5926，10 页提升、3 页不变。两项均无退步。这是经历史报告校准的本地分数。

13 页结果与此前验证候选完全一致；器件和引脚指标、已有线段保持不变，已有网络成员不丢失。输入不变、重复执行以及 65 次网络、物理分支、线段、附着记录和候选顺序打乱检查均通过。

15 项专项测试：

```powershell
python -B -m unittest discover -s tests -p test_wire_export_regression.py -v
python -B overlay_check.py
```

Dockerfile.overlay 已同步复制两个导线模块，overlay_check 会检查新模块存在且可导入。此次未构建 Docker 镜像，未重新运行模型或端到端评测。
