# 平台测试提交包

这是现有完整 S24 流程的源码包：YOLO v3 检测 → RapidOCR → 冻结 U-Net v1 → 引脚归属、公共引脚库与字段补全 → 工程整网出口。
保留项目默认 detector_dataset300_v3_nano.pt 和 unet_wire_pin_v1.pt；10月15日的 YOLO11、U-Net v7 与独立线段解码实验尚未接入该完整流程，本包没有擅自替换默认权重。

## 上传

上传 dist/submission.zip 到“我的提交”的当前阶段。ZIP 根目录直接包含 Dockerfile、agent.py、run.sh。
选择文件后平台可能立即开始提交。本包不使用大模型 API，无需密钥，Token 全部为 0。
GPU 自动选择 cuda:0，不更改 CUDA_VISIBLE_DEVICES；没有 GPU 时使用 CPU。

## 输出

- /output/final_results/<case_id>_Pred.json：默认按 sample README 的平铺 pins、nets.pins、wire_segments 布局导出。
- /output/project_results/<case_id>_Pred.json：保留项目原始嵌套 pins、hyperGraph、edges 布局，便于排查。
- /output/final_results/decision_log.json 与 /output/decision_log.json：同时写两处；官方 README 和平台网页对该位置存在不一致。
- /output/logs/：每张图的运行日志、覆盖统计和字段来源。

原项目的 type、Name/value、H-y 左下坐标及网络成员保持，适配器不重新编号或猜测元件。
官方示例 README 与正式 PDF 的 JSON 结构存在差异，尚无官方评分器验证。若平台明确要求 PDF 的嵌套结构，将 Dockerfile 的 PCB_OUTPUT_FORMAT=sample 改为 PCB_OUTPUT_FORMAT=project 后重新打包。

## 本地运行

Docker（在解压目录执行；输入/输出路径使用绝对路径）：

    docker build -t pcb-parser:test .
    docker run --rm --gpus all -v <输入绝对路径>:/input:ro -v <输出绝对路径>:/output:rw pcb-parser:test -input /input -output /output

CPU 测试可移除 --gpus all，并添加 -e PCB_DEVICE=cpu（放在镜像名称之前）。
基础镜像 edarace-agent-base:ubuntu24 来自官方示例；本机需能获取该镜像，平台是否预置须看构建日志。
构建时从 PyTorch 官方索引安装 CUDA 12.8 的 torch 2.10.0 / torchvision 0.25.0，推理所需 YOLO/U-Net/OCR 权重均已随包带入。

已安装同版本依赖的 Python 也能运行：

    python agent.py -input <输入目录> -output <输出目录> --device cpu

PNG 可位于子目录，保留文件名的前导零。不同目录若出现同名 PNG 会报错，避免覆盖结果。
每例生成独立日志；部分图片失败时输出空 JSON 并继续，其失败原因计入 decision_log。全部失败或预检失败返回非零退出码。
默认清理临时大概率图；需要保留时添加 --keep-work。平台可能限制中间文件下载，日志也会在失败时打印关键末尾信息。

## 验证边界

开发机未找到 Docker CLI，不能在此确认 Linux 镜像构建与平台运行成功。
本机将使用打包副本、仅 PNG 输入进行真实推理，并将结果记录在 dist/validation_report.json。
原有完整网覆盖仍有限，工程编号规则尚未由官方 Golden 确认。打包通过不代表取得官方识别成绩。
不含训练图片、参考 JSON、预测缓存、训练过程、Windows 虚拟环境或 API 密钥。
package_manifest.json 列出逐文件哈希、模型、配置与仅提交副本中的可移植性修改。
