# 两点线段解码 v1

将U-Net的原图尺度概率图转换成两点线段候选，评价端点位置和分段数量；像素连续性不再作为硬门槛。当前仍是几何层，不负责net身份、hyperGraph或逻辑飞线。

入口：`scripts/run_wire_segments.py`。例如在工程根目录执行：

```powershell
& ./.venv-gpu/Scripts/python.exe scripts/run_wire_segments.py --image data/standardized/0007/0007.png --weights runs/training/unet_v7_finetune_comparison_20261015/reviewed_v7/best.pt --config configs/wire_segments_v1.json --components runs/s6_pin_text_v1/predictions/0007/pin_text_result.json --device cpu --output runs/wire_segments_example/0007
```

`--components`只接受与当前PNG的图像哈希、尺寸和左上原点匹配的组件预测。它用于阻止跨过器件主体的缺口合并；不能传入数据集真值框。没有组件预测时仍可拟合和导出线段，但CLI关闭缺口合并。模型、推理入口和解码器不读取源标注，源JSON只在独立评估脚本中使用。

主要产物：

- `segments_top_left.json`：内部左上坐标的两点候选、每段来源和跨越的缺口。
- `candidate_edges_bottom_left.json`：按现有H-y约定转为左下坐标的edges候选，尚未归网，不是完整参赛JSON。
- `segments_overlay.png`：绿色线段、橙红色端点。
- `probabilities.npz`、`wire_mask.png`与`pin_candidates.json`：保留模型证据。
- `summary.json`：输入/模型/配置/组件预测哈希、真实推理耗时及范围。

模块`src/pcb_parser/wire_segments.py`支持独立概率图、现成组件内部阻塞区域和仅限制新增桥接的区域。拟合保留折点与分叉，删除≤2px短毛刺；仅当固定的方向、偏移、缺口距离、预测像素支持和保护条件通过时，合并同一直线碎段。原始概率图和掩码不原地改写。

评估：`scripts/evaluate_wire_segments_v1.py`，输出`runs/wire_segments_v1_guarded_20261015/`。3套固定参数先在既有13张开发页比较，选择4px最大缺口配置，再检查另13页；后者此前已参与历史验证，不是独立测试。参考为未经完全认证的源几何，按5px端点一对一全图匹配，不按net限制，分段口径仍需官方Golden核验。

修订模型在26页的几何代理F1为50.66%→52.43%，TP1075→1086，FP903→791，FN1191→1180；另一组13页50.62%→51.96%。收益主要是减少噪声和额外碎段，新增缺口合并只有1处。0208灰线无响应仍未解决。初版未加器件保护的实验在`runs/wire_segments_v1_20261015/`保留作审计；其中0036电感内部的误合并已被器件保护拦截。

16项新解码测试及原18项wire测试通过，共34项；0007与0022均通过PNG真实CPU推理并导出。CPU/GPU数值差异可能使接近0.5的像素跨阈值，导致离散分段变化；当前比较表基于冻结GPU预测，fresh_png目录单独记录真实CPU输出，不能将两者宣称逐位一致。未重新训练或替换默认权重。
