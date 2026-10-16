# GPU 环境与 S1 检测/OCR 基线实测

结果生成时间（UTC）：2026-10-11T06:20:42.556250+00:00。

显卡：NVIDIA GeForce RTX 5060；显存 8123 MiB；Compute Capability [12, 0]。
PyTorch 2.10.0+cu128，torchvision 0.25.0+cu128，CUDA runtime 12.8。
CUDA 矩阵运算、卷积和 torchvision CUDA NMS 均已实际执行并通过检查。

独立解释器：`C:\Users\Pc\Desktop\vision\pcb_parser_competition\.venv-gpu\Scripts\python.exe`。使用此解释器即可运行，不需要手动激活环境。

## 实测结果

| 图像 | 模式 | 设备 | 检测数 | OCR 暂绑定位号数 | 推理中位耗时 ms | PyTorch 峰值分配显存 MiB |
|---|---|---|---:|---:|---:|---:|
| 0002 | whole | cpu | 8 | 4 | 64.7 | — |
| 0002 | sliced | cpu | 8 | 4 | 249.7 | — |
| 0002 | whole | cuda:0 | 8 | 4 | 17.0 | 36.0 |
| 0002 | sliced | cuda:0 | 8 | 4 | 62.0 | 81.4 |
| 0001 | whole | cpu | 57 | 26 | 142.0 | — |
| 0001 | sliced | cpu | 105 | 30 | 680.5 | — |
| 0001 | whole | cuda:0 | 57 | 26 | 23.3 | 108.2 |
| 0001 | sliced | cuda:0 | 107 | 30 | 136.7 | 130.9 |

每项先预热一次，再计时；耗时包括预处理、推理、后处理及结果转回 CPU，排除模型加载、图片读取和输出文件写入。GPU 每次计时前后同步。显存为 PyTorch 已分配峰值，不是系统全部占用。

SAHI 使用相同权重、输入尺寸、置信度、IoU 和 max_det；1024 切片，重叠 0.125，类别内 GREEDYNMM 合并，并增加一次整图预测。切片数量随图像尺寸变化。

- 0001 / whole：GPU 检测推理比 CPU 快约 **6.08 倍**；这是当前图像与参数的实测，不代表全流程或全部300例。
- 0001 / sliced：GPU 检测推理比 CPU 快约 **4.98 倍**；这是当前图像与参数的实测，不代表全流程或全部300例。
- 0002 / whole：GPU 检测推理比 CPU 快约 **3.80 倍**；这是当前图像与参数的实测，不代表全流程或全部300例。
- 0002 / sliced：GPU 检测推理比 CPU 快约 **4.03 倍**；这是当前图像与参数的实测，不代表全流程或全部300例。

OCR 当前使用 ONNX Runtime CPU，独立计时：

- 0002：54 个文字区域，中位耗时 805.3 ms。
- 0001：474 个文字区域，中位耗时 6306.7 ms。

## 输入、产物及边界

推理入口只读取 PNG、检测权重与配置，没有读取原始 CVJsonStd、转换参考 JSON 或其他标注。
每张图/设备/模式分别输出：

- `detections.json`：左上坐标检测框、类别、置信度及计时。
- `ocr.json`（每例共用）：OCR 文本、四边形与置信度。
- `overlay.png`：红框为检测，绿框为 OCR，保留原始像素尺寸。
- `binding_debug.json`：候选位号、距离及未绑定/冲突原因。
- `components_partial.json`：采用赛事结构的局部结果，底部坐标按 H-y；Name/value 尚未识别，pins/nets 为空。

检测数不是准确率或召回率；更多检测也可能包含误检或重复框。此轮没有官方 F1，没有完整引脚与连接重建。局部 JSON 仅通过本地结构/边界检查，不能作为最终完整提交。

人工可视化复核记录在 `docs/baseline_visual_review.md`。

## 依赖与模型来源

检测权重来源：[https://github.com/HR-DONK/circuit-recognition](https://github.com/HR-DONK/circuit-recognition)。
检测权重 SHA256：`418a2fb5b9244146224e1e993aa8d27037ca33613865a1194642ff39b3283daf`。
Git 提交：未取得（GitHub API 限流）；查询失败时通过权重内容哈希固定本次实际文件。
PyTorch wheel 从官方源下载并验证官方索引 SHA256；下载记录保存在 `artifacts/setup/wheels/torch_download.json`。
主要版本固定在 `requirements-gpu.txt`，全部实际安装版本固定在 `requirements-gpu.lock.txt`。

本地运行配置与赛事平台环境尚未核对。代码/模型来源许可分别记录：Ultralytics AGPL-3.0，HR-DONK 仓库 MIT，SAHI MIT，RapidOCR 及其默认 PaddleOCR 衍生模型 Apache-2.0。
参考：[PyTorch 安装版本](https://pytorch.org/get-started/previous-versions/)、[RapidOCR 模型来源](https://rapidai.github.io/RapidOCRDocs/main/en/model_licenses/)。

## 再次运行

在项目根目录执行：

```powershell
.\scripts\run_quick_dev.ps1
```

同时测 CPU/GPU：

```powershell
.\scripts\run_quick_dev.ps1 -Devices cpu,cuda:0 -RunName s1_gpu_baseline
```

## 下一步

先按 overlay 复核 0002，再复核 0001 的典型漏检与误检；用已确认的元件类别/框制作检测训练集并进行微调。
OCR 优先检查小字号、竖排文字、位号与数值绑定。随后接入引脚识别、导线骨架和网络构建，并在官方评测器到位后确认坐标/匹配规则。
