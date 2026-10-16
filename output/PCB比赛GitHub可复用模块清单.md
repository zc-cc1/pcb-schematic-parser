# PCB 比赛：GitHub 可复用模块清单

检索日期：2026-10-11。面向 PNG 原理图 → components / pins / nets 比赛 JSON。

核查范围：仓库说明、文件目录、部分原始源码、许可证文件及权重入口。本轮没有安装运行这些项目，也没有在比赛数据上测 F1。下面的“优先”是接入顺序建议，不代表已验证性能；权重链接存在不等于已经下载验证。

## 1. 推荐组合

首版建议组合：

```text
元件检测：Ultralytics + 电路类别权重/必要微调
大图切片：SAHI
文字识别：RapidOCR
导线几何：图像掩码 + scikit-image + Skan
网络组织：NetworkX + 本项目电气规则
引脚先验：官方元件库；KiCad 格式库可用 kiutils 读取
导出评测：本项目 JSON 适配器 + 官方评测器
```

HR-DONK 可用作检测/OCR 封装的审计对象；CircuitSchematicImageInterpreter 可用作传统导线、交点方法的对照。Netlistify 是导线学习模型和连接分组的进阶候选，先解决许可证、权重和外部元件框输入。

多引脚 IC 的 `pin number + pinname + tip 像素坐标` 仍是本项目必须验证和补齐的核心。本轮没有确认到一个能够直接按本赛题契约输出这些字段的现成模块。

## 2. 模块总表

| 编号 | 模块/仓库 | 具体可复用部分 | 输入 → 输出 | 许可证核查 | 建议 |
|---|---|---|---|---|---|
| M01 | [RapidOCR](https://github.com/RapidAI/RapidOCR) | 文字检测、识别和推理后端 | 图像/局部图 → 文字框、文本、置信度 | Apache-2.0；README 另列 OCR 模型来源及许可 | 优先接入 |
| M02 | [SAHI](https://github.com/obss/sahi) | `get_sliced_prediction`、切片及结果合并 | 大图 + 检测器 → 原图中的检测结果 | MIT，已读 LICENSE | 优先接入 |
| M03 | [Ultralytics](https://github.com/ultralytics/ultralytics) | 目标检测、分类、分割与关键点训练/推理框架 | 图像 + 对应权重 → 框/类别等 | AGPL-3.0，已读 LICENSE；另有企业许可选项 | 用于检测；电路能力须靠对应权重验证 |
| M04 | [CircuitSchematicImageInterpreter](https://github.com/C-R-Kelly/CircuitSchematicImageInterpreter) | `wireScanHough`、`junctionDetection`、导线与交点对象 | 图像骨架 → 水平/垂直导线、交点 | MIT，已读 LICENSE | 传统 CV 基线和方法复用 |
| M05 | [scikit-image](https://github.com/scikit-image/scikit-image) | 骨架化、形态学与图像几何处理 | 导线掩码 → 细化图和几何候选 | LICENSE.txt：主要 BSD-3-Clause，含第三方许可说明 | 导线基础工具 |
| M06 | [Skan](https://github.com/jni/skan) | `skeleton_to_csgraph`、`Skeleton`、`summarize` | 细化图 → 像素图、分支、端点/连接点信息 | 仓库标明 BSD-3-Clause | 导线图结构模块 |
| M07 | [NetworkX](https://github.com/networkx/networkx) | 连通分量、图遍历、二部图组织和结构检查 | 带电气证据的节点/边 → 网络集合 | BSD-3-Clause，已读 LICENSE.txt | 网络数据结构与计算 |
| M08 | [HR-DONK/circuit-recognition](https://github.com/HR-DONK/circuit-recognition) | 检测、OCR、缓存及流水线封装候选 | 图像 → 检测/OCR 等中间结果 | 项目 MIT；Ultralytics 等依赖另行按其许可处理 | 延续前述底座审计 |
| M09 | [Netlistify](https://github.com/NYCU-AI-EDA/Netlistify) | 导线推理、`build_connection*`、切片与方向识别 | 图像 + 外部元件框/权重 → 连接分组/SPICE | 在已查看仓库页面中未确认项目许可证 | 技术候选，许可和运行依赖确认后再直接复用 |
| M10 | [kiutils](https://github.com/mvnmgrx/kiutils) | `.kicad_sym` / `.kicad_sch` 解析和符号对象 | KiCad 文件 → 结构化符号和引脚信息 | GPL-3.0，仓库标明 | 元件库可读时使用；不直接识别 PNG |
| M11 | [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) | OCR 训练、微调、检测与识别 | 图像 → 文字结果/训练权重 | Apache-2.0，仓库许可说明 | RapidOCR 不能满足小字时的备选 |
| M12 | [CI2N / Image2Net 数据与 benchmark](https://github.com/LAD021/ci2n_datasets) | 方向/镜像、交叉点数据；`validation/calc_ged.py` 等验证脚本 | 标注图或网表对 → 训练材料/图编辑距离诊断 | Apache-2.0，已读 LICENSE；数据来源说明另需保留 | 辅助训练及拓扑诊断 |

源码项目的许可证、预训练权重许可、数据来源和依赖许可分别登记。外围项目标成 MIT 不能替代对其内部 Ultralytics 等依赖的许可检查。

## 3. 优先模块的接入细节

### M01 RapidOCR：位号、值、型号、引脚文字

依据：[仓库与 API 示例](https://github.com/RapidAI/RapidOCR)、[模型许可记录](https://github.com/RapidAI/RapidOCR/blob/main/MODEL_LICENSES.md)。模型许可文件在 README 中有链接；具体所用模型仍需按该记录核验。

当前 README 示例使用 `from rapidocr import RapidOCR`，依赖 `rapidocr` 和选用的推理后端。前述 HR-DONK 说明使用旧的 `rapidocr-onnxruntime`；两个版本的结果格式不能默认相同。

接入时实现统一 OCR 适配器，固定输出 `text_raw、bbox、confidence、region_id、transform_id`。元件位号、Name/value、pin number、pinname 和网络标签分别绑定，不让 OCR 直接决定电气连接。字符串的大小写、空格和单位保留原文。

首轮验证：同一中等例与困难例，保存所有文字框和文本；检查小字是否被过滤、旋转文字是否漏识别，单独统计需要评分的字段精确匹配率。

### M02 SAHI：解决大图中的小元件检测

已检查接口：[sahi/predict.py](https://github.com/obss/sahi/blob/main/sahi/predict.py)，其中 `get_sliced_prediction()` 支持切片宽高、重叠比例、合并方式和匹配阈值。

它负责切片检测与结果合并，不提供电路识别权重，也不负责引脚/OCR/网络。接入后仍保存切片偏移和缩放信息，确认结果已回原图坐标。

首轮验证：固定检测器，比较原图与切片；看小目标召回、相邻元件被误合并、重复框及耗时。切片边长 1024、重叠 128 px 可以作为起始实验参数，不作为默认最优值。对同类密集器件尤其检查合并策略。

### M03 Ultralytics：复用检测框架

依据：[仓库](https://github.com/ultralytics/ultralytics)、[许可证](https://github.com/ultralytics/ultralytics/blob/main/LICENSE)。

可复用训练、推理和结果对象；通用目标检测权重不等于电路专用权重。先验证候选电路模型的类别表和实际结果，再决定是否微调。

目标是得到元件框及类别候选。位号、Name/value、引脚编号和名称仍需独立模块。框检测成功不能直接换算为比赛 ComponentF1。

用于引脚关键点属于另一个训练任务：需要对应引脚标注和权重，不能用普通检测模型自动产生准确引脚。

### M04 CircuitSchematicImageInterpreter：快速建立导线/交点基线

已检查：[actions.py](https://github.com/C-R-Kelly/CircuitSchematicImageInterpreter/blob/main/CircuitSchematicImageInterpreter/actions.py)、[LICENSE](https://github.com/C-R-Kelly/CircuitSchematicImageInterpreter/blob/main/LICENSE)。

`wireScanHough(image, minWireLength=10, borderSize=15)` 使用 `image.binarySkeleton`，返回水平/垂直导线。`junctionDetection(HorizWires, VertWires)` 生成交点对象。

适合建立无需训练的干净图基线；函数依赖项目自身的 image 和 wire 类，抽取时要连同需要的数据结构适配。方向范围主要针对水平/垂直线；骨架坐标和库对象存在 row/column、y/x 表示，不能直接当作比赛 x/y 输出。

交点算法只是几何候选；有连接点与无连接交叉必须另做图像判定。图纸边框、文字和器件轮廓也不能未经筛选纳入导线图。

### M05 + M06 scikit-image / Skan：从 wire mask 到分支图

依据：[scikit-image](https://github.com/scikit-image/scikit-image)、[Skan 源码](https://github.com/jni/skan/blob/main/src/skan/csr.py)、[Skan 入门说明](https://github.com/jni/skan/blob/main/doc/getting_started/getting_started.md)。

建议流程：提取导线掩码 → 骨架化 → `Skeleton`/`summarize` 或 `skeleton_to_csgraph` → 端点、分支和连接点候选。

Skan 能减少手写像素路径追踪的工作，但本身不认识电气语义。无连接交叉、文字粘连和器件内部线都需要在图构建前或之后处理。其坐标必须检查 row/column 顺序。

一个骨架分支可能包含多个折点，不能把整个分支的首尾直接作为一条比赛 `edge_i`；须按折点及官方标注分段。为避免一像素角点形成伪连接，首轮用直线、折线、T 节点、连通十字和不连通交叉验证。

### M07 NetworkX：网络组织，不替代连接识别

依据：[仓库](https://github.com/networkx/networkx)、[许可证](https://github.com/networkx/networkx/blob/main/LICENSE.txt)。

把 pin、wire 端点和有效连接点建成图，依据实际线段/接触证据建立边，再求连通分量。之后按核验过的同名标签规则加入逻辑连接。

也可用“器件引脚节点—网络节点”二部图保存超图语义，避免把不同 net 简单压成相同的器件两两连接。每个 net 同时保留 pin set 和所属 segment IDs。

不能因为两个节点距离接近就直接加电气边；数据结构模块不会判断无连接交叉、飞线或引脚编号正确性。

## 4. Netlistify：值得研究，但要处理几个实际缺口

已确认有真实源码文件：[inference.py](https://github.com/NYCU-AI-EDA/Netlistify/blob/main/inference.py)、[slice.py](https://github.com/NYCU-AI-EDA/Netlistify/blob/main/slice.py)、[collect_connection.py](https://github.com/NYCU-AI-EDA/Netlistify/blob/main/collect_connection.py)。README 有预训练模型的外部下载链接；本轮未下载权重。

优先研究导线模型与连接分组，而不是直接运行整仓库作为正式方案。

| 观察到的代码事实 | 接入时要做什么 |
|---|---|
| `get_box()` 读取 `labels/*.txt`，`inference()` 调用它取元件类别和框 | 接入实际检测器的预测结果；核实这些标签不是评测答案 |
| 推理脚本使用固定 CUDA 设备和模型路径 | 把设备、权重路径和模型加载改为配置；避免导入即加载重量模型 |
| 输出为 SPICE，类别和端口表示面向特定电路 | 增加正式 JSON、实际位号、文本、引脚和线段归属适配 |
| `build_connection` / `build_connection_v2` 根据端点距离分组 | 加入交叉语义与同名标签，保留原始线段及 net 归属 |
| 切片代码含归一化坐标与取整/过滤逻辑 | 检查原图回映射精度，避免小线段被删或分段改变 |
| 已查看根目录/README 中未确认项目许可证 | 直接复制或提交其代码前明确使用许可；当前先作为方法和接口研究对象 |

`build_connection` 会进一步处理点的重复/唯一性，直接拿它的末端点集合不能覆盖比赛要求的全部原始 edges。要在适配中保留输入 segment 信息。

这次对 Netlistify 的判断比仅阅读 README 更保守：它提供了可研究的导线与连接代码，但需要外部元件框、模型资源及格式改造，不能据“全流程推理”描述认定已经满足纯 PNG 比赛入口。

## 5. 引脚先验：kiutils + 合法符号库

已检查：[kiutils README](https://github.com/mvnmgrx/kiutils)、[symbol.py](https://github.com/mvnmgrx/kiutils/blob/master/src/kiutils/symbol.py)。可读取 `.kicad_sym` 符号库及 `.kicad_sch`；仓库标明 GPL-3.0。

使用方式：在允许的元件库中检索符号，读取引脚编号、名称、位置与方向，再结合图像中的型号、旋转/镜像和实际引脚区域做对齐。库坐标必须转换到图像像素，不能直接输出。

先使用赛事提供的元件库。如果它并非 KiCad 格式，需实现其格式的解析器，kiutils 就不是直接入口。不同 EDA 库中的同型号符号可能有不同绘制形状、拆分单元、隐藏引脚和布局，不能只凭型号就认定所有引脚坐标正确。

备用资料：[KiCad 符号库仓库](https://github.com/KiCad/kicad-symbols)及[官方库许可](https://www.kicad.org/libraries/license/)。库数据许可与解析器许可是两项不同内容；官方库页面说明 CC-BY-SA-4.0 带设计输出例外，重新分发库集合仍需遵守对应条款。本轮该 GitHub 根页未成功读取，不把库文件或具体型号覆盖列为已核实。

## 6. 训练与诊断资源：CI2N / PaddleOCR

### CI2N 是数据和 benchmark，不是完整 Image2Net 源码

仓库明确定位：[LAD021/ci2n_datasets](https://github.com/LAD021/ci2n_datasets)。可利用器件方向/镜像、交叉点/跳线标注及网表验证材料。

已确认验证脚本目录含 [calc_ged.py](https://github.com/LAD021/ci2n_datasets/blob/main/validation/calc_ged.py)、`utils_ged.py`、`device_def.toml` 等。它们可帮助做拓扑差异诊断，但其 NED/GED 不是本赛题的四项 F1，不能作为正式评测替代。复杂图的编辑距离耗时也要单独控制。

数据适配时按源电路分组，记录类别、方向和交叉标签的转换；避免与开发集同源泄漏。

### PaddleOCR 作为识别/微调备选

[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) 是 Apache-2.0 项目，可作为 OCR 对照及微调来源。首版优先把一个 OCR 后端跑稳，只有开发集证实当前后端在特定小字/符号上不足时再替换或微调。

使用所选模型对应的检测/识别推理接口，不把通用文档 Markdown 输出作为引脚和电路连接结果。运行框架和镜像需与赛事平台要求对齐。

## 7. 搜索到但不作为首版代码底座的项目

| 项目 | 核验结果 | 建议用途 |
|---|---|---|
| [PCBnet](https://github.com/ECNUACRush/PCBnet) | 前述仓库说明未公开完整源码；不能据示意目录推断推理可用 | 官方数据及方法参考，进入实现前重新确认资源状态 |
| [AMSnet 2.0 项目页](https://amsnet2-0.github.io/AMSNet2.0.github.io/) | 页面描述 U-Net 与交点拆分/合并路线；本轮页面没有核实到可运行 GitHub 推理库链接 | 方法参考，避免把网站模板源码当识别系统 |
| [SINA 论文](https://arxiv.org/html/2607.01609v1) | 作者代码脚注指向匿名托管平台；不应把任意同名 GitHub 仓库称为作者官方实现 | 后续考察 PCB 引脚与交叉识别方法 |
| [stockedge/circuit-llm-bench](https://github.com/stockedge/circuit-llm-bench) | README 描述第三方 SINA 实现；本轮源码子目录未成功读取，许可证也未确认 | 暂列待审计，避免认定已经可复用 |
| [手绘 YOLOv5 电路识别](https://github.com/rohithrr9/Hand-Drawn-Electrical-Circuit-Recognition-using-YOLOv5) | 前述 README 针对少数类别手绘电路 | 方法对照；不作为多引脚 PCB 的完整底座 |

## 8. 接入顺序和每个模块的最小验收

先准备一个中等和一个困难公开案例，固定它们的输入及允许用于离线评测的标注。每个候选先完成下列检查，再决定进入全开发集。

| 顺序 | 接入项 | 最小验证 | 留用依据 |
|---|---|---|---|
| 1 | RapidOCR 或已有 OCR 后端 | 画出文字框，保存原文/置信度及耗时 | 关键字段可识别、接口稳定，未丢失引脚小字 |
| 2 | 电路检测器 + SAHI | 相同权重下对比整图/切片，检查原图坐标 | 小目标收益、框误差、重复检测与耗时可接受 |
| 3 | 导线基线 + scikit-image/Skan | 先测 5 类几何小例，再测原图 wire mask | 端点/折点/交叉差异可解释，坐标正确 |
| 4 | 官方库解析；格式合适时用 kiutils | 一个已知 IC 符号，检查编号/名称/方向，再做像素对齐 | 库和图像符号一致，tip 误差能按标注测量 |
| 5 | NetworkX 网络层 | 物理连接、pin 直接接触、同名标签分别验证 | pin set 和 segment 归属正确，无无效引用 |
| 6 | Netlistify 导线模型或 CI2N 数据 | 先确认资源/许可，只在现有导线瓶颈上做对照 | 固定开发集官方指标改善，额外耗时在预算内 |

正式导出前统一坐标和引用：`pin_A5` ↔ `U1.A5`；全部线段恰有两个点；每条线段处于正确 net；保留需要的零长度线段。真实标注仅用于训练/诊断/评测，不进入实际推理输入。

本轮证据以检索日期的仓库页面和所列源码链接为准。进入复现时锁定实际 commit、权重哈希、依赖版本，并把真实运行结果补入仓库审计表。
