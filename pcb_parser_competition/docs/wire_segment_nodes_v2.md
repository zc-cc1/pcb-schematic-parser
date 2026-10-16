# 预测连接点切分与参考分段审计

此次按用户要求完成两项工作：核对高错误页0018、0170、0007的源分段表示；实现只依赖预测tip及线段几何的端点校正/连接点切分。没有重新训练或更改源JSON、标签、旧预测与权重。

源分段审计在`runs/wire_segment_nodes_v2_20261015/segmentation_audit.json`。原2266段中有267段≤5px且靠近源edge声明的pin位置；它们常是接入小折线，不等于267条独立漏线。0170/E_1还存在大幅重叠的线段。仅在同一edge内合并严格共线重叠、保留相接分段和所有独立短段，参考变为2085段。这只是参考解释敏感性分析，未经官方Golden确认，不改主评分，也不算模型提升。

第一轮实现端点吸附、内侧tip切分与保守T节点切分。在13页开发组上，端点吸附3px使原始端点代理F1从53.20%降为52.04%，5px吸附为51.91%；因此未采用。结果在`runs/wire_segment_nodes_v2_20261015/`。

第二轮单独验证切分，不移动现有端点：开发组53.20%→53.30%，另一13页51.96%→51.94%，增加1个TP但增加4个FP。未通过预先确定的检查条件，保留v1；当前采用结果仍为26页52.43%。这两组都是历史开发验证页，不是独立测试，分数也不是官方NetLineF1。结果在`runs/wire_segment_nodes_v2_split_only_20261015/`，被拒绝的候选仍按页保存在`rejected_candidate.json`。

模块`src/pcb_parser/wire_segment_nodes.py`不接收标注或GT端点。它支持高置信度tip端点吸附、内侧pin投影切分、间断T型支路投影；包含近邻歧义拒绝、最小片段长度、器件主体保护和疑似十字交叉拒绝。每个移动/切分记录原点、目标点、预测pin索引或几何来源。它只改变几何候选，不推断电气连通或net成员。

PNG入口新增`--node-config`。`configs/wire_segment_nodes_v2.json`保存最终采用条件，当前enabled=false，以保持原有解码。实验配置保存在第二轮输出目录`experimental_config.json`，enabled=true且标记未采用。0170已从原PNG真实CPU推理，使用实验配置产生2个切分；没有读取参考答案。使用实验配置时应另设输出目录：

```powershell
& ./.venv-gpu/Scripts/python.exe scripts/run_wire_segments.py --image data/standardized/0170/0170.png --weights runs/training/unet_v7_finetune_comparison_20261015/reviewed_v7/best.pt --config configs/wire_segments_v1.json --components runs/s6_pin_text_v1/predictions/0170/pin_text_result.json --node-config runs/wire_segment_nodes_v2_split_only_20261015/experimental_config.json --device cpu --output runs/node_experiment_example/0170
```

48项wire测试通过：原34项、10项节点测试和4项参考解释测试。完整检查见第二轮目录completion_checks.json。下一步应分别处理器件内部误检、真正缺少响应的灰线和更可靠的线段节点监督；当前tip吸附没有显示稳定收益，不能直接扩大修改范围。
