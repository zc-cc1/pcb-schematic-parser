# PCBnet 原始基线查核

核查日期：2026-10-11。

找到与本赛题相关的 [官方仓库 ECNUACRush/PCBnet](https://github.com/ECNUACRush/PCBnet) 和 [论文 arXiv:2608.27923](https://arxiv.org/abs/2608.27923)。用户提到的“约95%”对应论文报告的元件检测 mAP 94.54%。论文表 I、表 III 分别报告导线 IoU 93.70%、连接恢复准确率84.47%、最终文本识别准确率98.57%。这些指标不能互换。[论文正文](https://arxiv.org/html/2608.27923v1)

论文 III-A 描述 YOLOv11 元件/文本区域检测、PaddleOCR、U-Net导线分割及骨架化；后续按几何关系估计引脚、构建连接并匹配文本，最后使用领域知识和多模态纠错。IV-A采用工程级8:1:1划分，实验硬件为RTX3090。公开正文未明确 YOLOv11 的 n/s/m/l/x 规格、输入尺寸、轮数和完整训练参数，也未明确该 mAP 使用哪个检测 IoU 口径。不能把 YOLO11s、1536输入或120轮当成已确认的论文原配置。[方法与实验](https://arxiv.org/html/2608.27923v1#S3)

截至本次核查，官方 main 分支归档只有 `README.md`，没有训练脚本或 `.pt/.pth/.onnx` 权重；[Releases](https://github.com/ECNUACRush/PCBnet/releases) 也没有发布包。README明确说明源码因知识产权限制暂不能发布。其列出的 models/、scripts/ 等目录和运行命令为计划内容，当前归档中不存在。公开搜索没有找到可验证为作者发布的独立权重或第三方完整复现；这不证明所有私有或未索引资源都不存在。[官方说明](https://github.com/ECNUACRush/PCBnet#dataset)

官方 Google Drive 链接指向 Dataset300.zip。README提示公开300样例不保证与论文实验样例完全一致。当前本地模型的81.32%是固定置信度下类别+框的微平均F1，数据划分与论文不同，不能直接与94.54%的mAP比较。此前使用的初始权重来自HR-DONK/circuit-recognition，采用YOLOv8n，与PCBnet作者的模型权重不同。

本次官方仓库快照保存在 `data/reference_sources/pcbnet_20261011/`。只下载并检查公开文档，没有执行上游代码或改动训练配置。归档大小3253字节、文件数1；SHA-256为 `471587d38e5c896a2bdbf237482511ade9aaebf055ed9ff979d8dc00d2122004`，分支为main，提交SHA未取得。逐项结果见 `source_check.json`。

要严格复现作者基线，还需要原始检测权重、模型规格、训练参数、类别表、工程划分及mAP评估脚本。现阶段可以依论文重建同类方法；这属于独立复现尝试，不能预先宣称拥有作者94.54%的现成基线。本次没有联系作者或发送任何外部消息。
