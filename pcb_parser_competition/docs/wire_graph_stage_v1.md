# U-Net 导线图与电气网络阶段 v1

本阶段把 U-Net 导线概率转换为中心线与 NetworkX 图。分割输入始终来自冻结的 U-Net；图转换不从原图颜色或霍夫直线提取导线。中心线转换使用 scikit-image 的 skeletonize，图连通使用 NetworkX。

## 输入输出与运行

输入为 PNG、现有 YOLO/SAHI、U-Net 和 OCR 模型；也可传入带图片/模型/配置/文件哈希的引脚文字与导线概率缓存。

输出包含：

- `graph_proposals.json`：压缩节点、分段折线、网络候选、引脚接入、交叉和断线候选。
- `wire_graph_overlay.png`：同色为图连通假设，橙色为交叉待判，粉色为未补连的断线候选。绿色引脚代表唯一接入候选，不代表参考证实正确。
- `guarded_wire_mask.png`、`skeleton.png`、`body_excluded_mask.png`：分割后图转换的诊断图。
- `review_queue.json`：未接入或歧义引脚、交叉和断线候选。
- `partial_submission.json`、`export_coverage.json`：结构合法的部分 components/pins/nets 及未能导出原因。该文件不是完整比赛答案。
- `summary.json`：输入指纹、版本、图规模、配置与运行信息。

在桌面“新建文件夹”的项目根目录：

```powershell
.\scripts\run_wire_graph.ps1 -Image 'data/standardized/0002/0002.png' -Device 'cuda:0'
```

命令从 PNG 运行检测、U-Net、文字与图阶段，默认输出到 `runs/wire_graph_predictions/0002`。支持 `-Device cpu` 和 `-Output`。

固定模型缓存的调试入口：

```powershell
.\.venv-gpu\Scripts\python.exe scripts/run_wire_graph.py --image data/standardized/0002/0002.png --pin-text runs/s6_pin_text_v1/predictions/0002/pin_text_result.json --wire-cache runs/s7_wire_graph_v1/inputs/0002/wire_probability.npz --output runs/wire_graph_debug/0002
```

转移目录可复算离线结果：

```powershell
.\.venv-gpu\Scripts\python.exe scripts/evaluate_wire_graph.py --reuse-references
```

参考快照、原始 CVJsonStd 标注只由离线评估器读取，PNG 入口不导入评估器。

## 拓扑处理与边界

1. 保持已采用的 U-Net 原分辨率概率与 0.5 阈值。
2. 用预测元件框隔离内部区域，保护已归属的 tip 邻域，减少通过元件符号错误短接两端。所有 pin 原坐标保留。
3. 把预测掩膜转换为中心线；邻接时避免 L 弯形成斜对角捷径。压缩度为 2 的路径，保留闭环和折线。
4. T 形按模型掩膜暂作连通。四向/多向结点没有学习分类头，本轮明确标为待判；默认按可靠直通方向分成独立路径，保留全部连接的对照解释。
5. 引脚在固定 6px 接入半径内选择网络；两个候选距离相近时保留歧义。保存投影点和间距，不将投影点替换为 tip。
6. 相向且邻近的线端记录为断线候选，检查是否落入元件区域，不自动补线。
7. 未归属引脚仍可保留导线几何上下文，不升级为有编号、位号的正式引脚。零/单引脚线段碎片保留以便调试。

最终网络需要物理连通、junction/crossing 语义、同名普通网络标签，以及 VCC/GND 的专门规则。v1 没有自动同名标签合并或电源符号合并。

部分导出只接纳位号、编号、名称齐全的 pin；不会把未知名称猜成空字符串。含未决 pin、交叉或断线候选的网络不按删掉难点后的剩余 pin 集合导出。坐标出口采用已有本地 H-y 约定，尚待正式 Golden 验证。

## 26 页开发实测

同一批旧验证图：1477 个引脚候选全部保留，1395 个唯一接入图；400 个多引脚网络候选，另有线段碎片。60 个交叉、55 个断线候选待判。

在已匹配位置与归属的 910 个引脚间，默认连通配对 P=99.48%、R=60.74%、F1=75.43%。计入全部预测与参考引脚对后，代理 P=50.80%、R=30.63%、F1=38.22%。这些是 pin-pair 代理指标，不是官方 NetF1，不能用 99.48% 当作整体网络准确率。

原始参考采用 edgeId 分组、回指和重合 tip；本轮禁用标签名合并，但原始分组本身的物理/逻辑区别未独立认证。源漏 pin、tip/base 偏移和错误元件仍会影响指标。

本轮部分导出共 194 个元件、32 个完整字段 pin、0 个满足全部保守条件的 net。完整 nets 尚未完成；正式编号/名称缺失仍是主要瓶颈之一。详见 `runs/s7_wire_graph_v1/report.md` 与 `evaluation.json`。

## 复核与后续学习

`data/topology_review_v1/annotations.json` 提供 60 个交叉、55 个断线的原图裁剪、预测坐标、图片哈希与空标签；联系图位于同目录。没有自动给它们生成训练类别，也没有修改既有 U-Net 训练标注。

这批裁剪来自已使用的验证图，不是自采新增独立数据，不能据此申报数据采集附加分。后续训练 junction/crossing 分类或扩展 U-Net 通道前，需要核对标注，并按原理图来源分组划分；一旦训练使用这些页，旧页指标不能继续充当独立验收。

下一步优先核对交叉和真实断裂，完善学习监督；同时处理元件位号后缀和引脚编号/名称覆盖，再增加网络标签与电源语义，完成正式 nets/edges 出口。

## 依赖与来源

本项目环境已安装 scikit-image 0.26.0，NetworkX 3.7 已存在。新环境在安装原项目依赖后执行：

```powershell
.\.venv-gpu\Scripts\python.exe -m pip install -r requirements-topology.txt
```

骨架与图转换的接口依据：[scikit-image skeletonize](https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.skeletonize)、[NetworkX connected_components](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.components.connected_components.html)。
