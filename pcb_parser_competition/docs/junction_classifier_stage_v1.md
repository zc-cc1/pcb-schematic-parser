# 交点学习模块 v1：训练、运行与调试

本阶段增加一个从 RGB 原图学习判断连接点的小型 PyTorch CNN，接到 U-Net → 骨架图 → NetworkX 的现有流程。模型参数 89,746；输入是以候选坐标为中心的两种视野（半径 8 和 24 原图像素，各缩放到 64×64），输出连接点/无连接跨线两类分数。对四个旋转方向都一致、最低分 ≥0.95、且图节点度数为 4 的候选才接受决策。跨线判断还必须能按现有骨架方向分成两对直通分支。

原图供 CNN 读取，不用颜色阈值、圆点模板或形态学规则判断连接语义。U-Net 仍提取导线像素，原引脚坐标保持不变。图变换仍用 scikit-image 和 NetworkX。没有新增需要下载的第三方模型；本 CNN 是当前项目新增的小模块，训练样本很少，属于开发试验。

## 当前实测与默认选择

完整对比在 [试验报告](../runs/s8_junction_classifier_v1/report.md)。26 张旧开发页：已匹配位置/归属的引脚对 F1 为 80.60%（v1 为 75.43%）；全量引脚对代理 F1 为 40.92%（v1 为 38.22%）。后者 FP 从 556 增到 614，不能仅看召回提升。这些都是原标注 edgeId 草稿下的开发观察，不能称赛事 NetF1。

分类器留出为 7 张原理图上的 13 个裁剪；直接预测 11/13 正确。接受的 5 个决策中 4 个正确，8 个保留待复核。0149 的连接圆点被高置信度判成跨线。样本量小，且这些原理图已经被上游模型开发使用过，不能作为整条流水线独立测试。

**继续保留 `run_wire_graph.ps1` 的 v1 默认配置。新入口显式运行 v2 试验；暂不替换默认。** 原 YOLO、U-Net 权重及训练标注没有改变。完整比赛 JSON 尚未完成，26 页严格部分网络仍为 0。

## 输入、输出与调试指标

| 阶段 | 输入 | 输出 | 检查点/实测 |
|---|---|---|---|
| 图像复核 | 60 个原图候选裁剪 | reviewed_annotations.json | 助手视觉开发参考：连接23，跨线35，伪影1，疑难1；不是官方 Golden |
| 分组 | case_id 和图像 SHA | train/calibration/test/excluded | 训练32；校准14中两类13；留出13；排除1。组间图像 SHA 不重叠 |
| 合成数据 | 程序绘图种子 | 1024 个训练样本 | 标签由绘图程序定义，只用于训练；不是新收集外部数据或附加分材料 |
| CNN 训练 | 训练组＋合成；校准组损失 | best.pt、history、evaluation.json | 30轮，采用18轮；24.1秒。不重新训练 YOLO/U-Net |
| 交点推理 | RGB、U-Net图候选、冻结CNN | junction_predictions.json | 原坐标；模型SHA；4次旋转分数；接受/拒绝原因；不读复核标签 |
| 图更新 | 同一U-Net概率与引脚预测＋CNN决策 | graph_proposals.json、overlay | 本次20个连接、21个跨线、19个待复核；55个断线不自动补 |
| 离线评估 | 图预测＋独立读取的参考快照 | report、evaluation | 分类正确率与覆盖率并报；连接FP/FN并报；official_NetF1=null |

## 在桌面项目中运行

项目位置：`C:/Users/Pc/Desktop/新建文件夹/pcb_parser_competition`。现有 `.venv-gpu`、requirements-gpu 与 requirements-topology 足够；无需再次下载 PyTorch。以下在项目目录执行。

只给一张 PNG 的完整模型流程：

```powershell
.\scripts\run_learned_wire_graph.ps1 -Image data/standardized/0002/0002.png -Device cuda:0 -Output runs/s8_junction_classifier_v1/png_runtime_check/0002
```

离线复现冻结 26 页模型输出（重用经过 SHA 校验的图像预测缓存）：

```powershell
.\.venv-gpu\Scripts\python.exe scripts/batch_learned_wire_graph.py --device cuda:0
.\.venv-gpu\Scripts\python.exe scripts/evaluate_learned_wire_graph.py
```

重新训练小 CNN（建议指定新输出目录保留原试验）：

```powershell
.\.venv-gpu\Scripts\python.exe scripts/train_junction_classifier.py --device cuda:0 --output runs/training/junction_classifier_next
```

训练代码与数据都已经合并，能在本机现有环境运行。默认 `freeze_junction_model.py` 只冻结 `runs/training/junction_classifier_v1/best.pt`；新实验应复制配置并写新的权重路径、SHA 和输出目录，不要覆盖正式上游权重。该脚本校准门槛只用于制作试验配置，不能代替独立可靠性验证。

```powershell
.\.venv-gpu\Scripts\python.exe -m unittest discover -s tests -p 'test_wire_graph*.py'
.\.venv-gpu\Scripts\python.exe -m unittest discover -s tests -p 'test_junction_classifier.py'
```

运行时只读取 PNG、模型预测及配置/权重；`reviewed_annotations.json` 和 `reference_connectivity.json` 只由训练或离线评估脚本读取。旧 `data/topology_review_v1/annotations.json` 的 115 条空标签原样保留；新复核记录另存 `data/junction_classifier_v1`。

## 下一步应解决的实际缺口

先增加真实新来源的原理图及交点复核，覆盖小圆点、浅色线、密集平行线、无圆点T连接、文字伪影与不同绘图规范。把本次0149错误作为开发失败案例；下一次若用它训练，必须换一组全新的留出页评估，不能再把本次留出分数称独立。

伪影和疑难点合计只有2个，暂没有训练第三类；CNN不会自动删除疑似文字像素。55个断线及完整引脚编号/名称覆盖仍未解决。网络标签和电源逻辑合并也还没有实现，因此最终nets尚不完整。
