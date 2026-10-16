# 官方 Docker 线索核对

2026-10-16 用户提供 Q5 截图：8 月 28 日答复提到官方提供 Docker，选手需本地跑通后打包上传。这支持继续寻找官方容器及配套本地测试说明，但截图本身没有给出下载位置、版本和容器内容。

已做只读核查：

- 当前平台公开 API `/api/competitions/237/` 的 `docker_image` 为 `edarace-evaluator-base:ubuntu24`。
- 已下载的示例 `纯CV/Dockerfile` 引用 `edarace-agent-base:ubuntu24`，只复制 agent.py、run.sh 并设置选手程序入口。
- 当前公开资源 API `/api/competitions/237/resources/` 只列出赛题 PDF 和约 21 KB 的 2606sample_submission.zip，没有独立 Docker TAR 下载项。
- 本地 Docker 镜像列表中没有 `edarace*` 镜像。现有已测试镜像基于本地 Python/CUDA 环境构建。

以上两个镜像名不能直接证明包含完整评分器，也未给出可供选手访问的 registry。没有猜测镜像下载地址或修改 Docker 配置。

下一项需要的材料：官方 Docker 下载链接或完整 docker pull 命令，以及其配套本地测试命令。拿到后可检查是否包含 ingestion/scoring、公开样例、类别映射和本地评分入口，再决定是否重建算法镜像。

补充发现：`C:/Users/Pc/Desktop/新建文件夹/case_001_扩展示例.json` 和 `submission/case_001/case_001.json` 是独立 JSON 文件。它们使用 ic/resistor 和嵌套 pins、hyperGraph/edges，但来源尚未确认，且网络引用包含文件中未声明的元件/引脚，因此属于不完整示例，不能直接当作闭合的 GT 做整图评分。没有更改这些文件，也没有将它们放入推理镜像。

本次线索不改变已确认的 8094 事实：镜像运行并处理 10 张图成功；输出结构不兼容及内部评分为零已有报告依据。新候选包修复结构并本地通过，仍需完整官方评分链路验证。已更新反馈草稿，未发送。
