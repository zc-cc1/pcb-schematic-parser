# v100 分数导出优化版

本目录是从 pcb-parser:v100 镜像提取的完整应用，保留原检测、OCR、U-Net 和模型权重。

改动：
- 局部无效坐标、元件框、线段和引用单独清理，并记录原因。
- 后处理失败回退当前图片最近通过校验的结果；不读取旧运行最终结果。
- 复制诊断日志失败不会覆盖已经写好的预测。
- 最后阶段可补已知编号的 IC 端口、R/C/L 缺编号端口；保护既有字段，拒绝跨网络重复编号锚点。

PCB_SCORE_EXPORT_PROFILE 可设 none、ic、passive、combined。默认值和本次本地开发集结果见上一级 RESULTS.md。恢复候选仅依赖当前图预测，不读取参考答案。

运行（已安装依赖的 Python）：

    python agent.py --input INPUT_DIR --output NEW_OUTPUT_DIR --device cuda:0 --workers 1 --keep-work

Docker：使用上一级 Dockerfile，以已存在的 pcb-parser:v100 为基础镜像覆盖应用。

    docker build --pull=false -t pcb-parser:v101-score .

本机没有 Docker，因此本次验证的是 Windows Python/GPU 入口；未声称验证 Docker 构建。

本轮新增最终字段整理，默认 PCB_EXPORT_SEMANTICS=observed；旧标注兼容使用 legacy。详见 EXPORT_SEMANTICS.md。本轮源码覆盖包为 pcb-parser-v102-semantics-patch.zip，以 pcb-parser:v100 为基础镜像，建议构建标签 pcb-parser:v102-semantics。
