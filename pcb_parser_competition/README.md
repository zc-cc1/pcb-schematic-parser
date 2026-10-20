# PCB 原理图解析比赛项目

当前阶段：已完成结构转换、PyTorch CUDA环境及检测/OCR基线，并完成两轮Dataset300检测微调。本地整图检测F1从原始权重0.4152到第一轮0.7755，再到第二轮0.8008；第二轮SAHI/IOS切片F1为0.8132。稀有类仍有较多漏检，OCR语义绑定、引脚、导线骨架和网络构建待推进，尚无官方 F1。

已完成首批外部原理图收集：OLIMEX、SparkFun、Adafruit 共10个工程，8张原理图PNG、10份CAD源文件；保存来源、许可声明与内容哈希。与Dataset300做像素/pHash/工程名初筛未命中重复候选，工程级和官方PCBnet去重仍待复核。6页预留为训练候选，2页预留外部测试；已生成6页检测草稿和22个原生稀有符号复核条目。新数据尚未完成标注或用于训练，指标没有因此更新。详见 [外部数据收集报告及实施计划](docs/external_data_collection_report.md)。附加分最低数量与计分仍需单独的《数据提交说明》，不能把下载工程数量当成得分。

开发默认已采用第二轮权重：保留238张原训练页，新增180张训练局部图，在相同59张验证图上比较。晶振整图F1为0.1176、切片0.4000，MOSFET整图0.4800、切片0.4667；两类第一轮整图均为0。详见 [第二轮增样与微调报告](docs/detector_balanced_training_report.md)。这些指标仅评价类别和框，数据未完成全量人工认证，验证集已被用于多轮开发。

首轮微调使用238张训练图、59张验证图，0001/0002固定为验证。看图复核411个裁剪对象，记录83项人工修正或排除；训练20轮约4.2分钟。详见 [检测器微调实测报告](docs/detector_training_report.md)。

GPU 环境：`.venv-gpu/Scripts/python.exe`，PyTorch `2.10.0+cu128`，torchvision `0.25.0+cu128`。RTX 5060 已通过真实 CUDA 矩阵、卷积及 NMS 检查。详见 [GPU 与基线实测报告](docs/gpu_baseline_report.md)。

在本项目根目录运行两例微调模型的 GPU 推理：

```powershell
.\scripts\run_quick_dev.ps1
```

最新微调产物保存在 `runs/s2_balanced_quick_dev/`，第一轮产物在 `runs/s2_finetuned_quick_dev/`，原始权重基线在 `runs/s1_gpu_baseline/`。各版本均包含检测/OCR JSON、可视化、暂绑定位号和局部赛事结构 JSON。局部JSON仅含元件候选；Name/value 未实现，pins/nets 为空，不能当作最终完整提交。

第一轮检测数据与标签在 `data/detection_v1/`，第二轮派生训练数据在 `data/detection_balanced_v2/`。原始图片/JSON只读；类别修正只用于检测训练标签，先前的赛事结构参考JSON仍保留原转换语义。模型配置为 `configs/baseline_finetuned.json`，第二轮训练配置为 `configs/detector_training_balanced.json`，切片预算和人工修正在 `configs/detection_augmentation.json`。第一轮模型配置保留为 `configs/baseline_finetuned_v1.json`。

先查看 `docs/input_inventory.md`、`docs/requirements.md` 和 `docs/open_questions.md`。机器可读材料清单位于 `data/material_inventory.json`，本机环境记录位于 `docs/environment.json`。

已登记用户提供的 Dataset300：`data/manifest.jsonl` 共 300 条真实图像/原始标注记录；`data/splits/quick_dev.json` 固定 0002 与 0001 两个调试例。它们只有本地复杂度判断，尚无官方难度标签和完整赛事格式 Golden。详见 `docs/dataset_format.md` 和 `data/dataset_audit.json`。

300例的统一PNG和赛事结构参考JSON已写入 `data/standardized/<case_id>/`；已复查结构、引用和输出哈希，详情见 `docs/conversion_report.md`。4546个元件键为 `UNRESOLVED_*`，其位号或可表达性尚未确定；原始候选、来源和其他问题保存在逐例报告和复核队列，不将它们宣称为已确认Golden。

转换和重建报告（在项目根目录执行）：

```powershell
$pcbPython = 'C:\Users\Pc\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
& $pcbPython scripts/convert_dataset.py
& $pcbPython scripts/build_conversion_report.py
```

可以修改 `configs/annotation_overrides.json` 中已核验字段，或修改 `configs/conversion.json` 的坐标/类别/网络策略后重新转换。`--case-id 0002` 用于单例试验；生成合并报告前须全量重建，工具会检测混合版本和手动改动过的输出。

刷新材料清单（PowerShell，在项目根目录执行）：

```powershell
$pcbPython = 'C:\Users\Pc\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
& $pcbPython scripts/audit_dataset300.py --dataset-root 'C:\Users\Pc\Desktop\新建文件夹\Dataset300'
& $pcbPython scripts/intake_inventory.py --search-root 'C:\Users\Pc\Desktop\vision' --search-root 'C:\Users\Pc\Desktop\新建文件夹' --spec 'C:\Users\Pc\Desktop\赛题六：AI驱动的PCB原理图自动解析（芯粒CAD和制造浙江省工程研究中心&比昂芯0803Rev1.pdf'
```

有其他资料目录时重复传入 `--search-root`。该命令读取目录和候选 JSON 的结构，只写本项目的清单；不下载、复制、删除或改写资料。扫描范围和限制会写入清单，不把文件名扫描当作全盘搜寻。

目录约定：

```text
docs/              # 规格、问题、清单及环境记录
schemas/           # 本项目暂定格式校验规格；不是官方评测器
configs/           # 规则、输入路径和坐标决策
data/incoming/     # 后续接收的正式资料，原件保持不变
data/splits/       # 按源工程分组的案例划分
scripts/           # 材料清点工具
src/pcb_parser/    # 下一阶段实现五模块及适配器
artifacts/         # 分阶段预测与调试图
runs/              # 配置、逐例指标和运行日志
weights/           # 后续权重及哈希清单
submission/        # 正式 JSON
```

完整开发计划在工作区 `output/PCB五模块方案实施计划.md`。材料清点、原始数据登记、两例人工修正、全量结构转换、GPU检测/OCR基线、首轮检测微调及第二轮稀有类增样实验已完成。后续推进OCR位号/型号/数值绑定，继续改善两脚晶振等稀有类与切片误检，再接入引脚/导线/网络和官方评测器/元件库。官方 F1 保持未验证状态。

## v3 优化分支（2026-10-11 本地副本）

本副本在不动 v1/v2 行为的前提下加入 v3 优化：训练（imgsz 1536+amp+120 epochs、可选 yolov8s-P2 架构）、推理（whole/sliced/`fused` 三模式，类感知 WBF 双路融合、逐类置信度阈值、小图跳过 SAHI）、诊断（`error_breakdown.py` 按类/尺寸/来源拆解、`tune_class_thresholds.py` 阈值扫描）。GPU 虚拟环境通过 `.venv-gpu` 目录联接共享原项目。执行顺序与判据见 [docs/optimization_v3.md](docs/optimization_v3.md)。


## U-Net训练模块（合并交接）

已合并双输出U-Net导线分割/引脚tip热图训练模块，与现有YOLO源码、配置和权重并存。运行入口为 `scripts/run_unet_training.ps1`，接手agent先阅读 [U-Net训练交接说明](docs/unet_training_handoff.md)。

在本工程根目录执行 `./scripts/run_unet_training.ps1 -Mode Check` 检查路径与环境；显卡空闲后，可使用 `-Mode Train -AllowDraft -Device cuda:0` 启动草稿标签实验。未人工复核的标签不称为正式Golden；检查和CPU短跑不代表训练质量。详细配置与数据状态见交接说明。

## OCR 阶段（2026-10-18）

OCR 现状、六页误差预算、三条从 GT 反推出的标注约定（`Name == key`、无源器件 value 不写纯数字、位号字符串的可见天花板）、本轮三处改动及离线/端到端 A/B 见 [OCR 阶段现状与首轮修正](docs/ocr_stage_v1.md)；只读诊断脚本在 [`analysis_ocr/`](analysis_ocr/README.md)。

**必须知道**：线上跑的代码是 `D:\pcb5_submission`（从第 5 次评测镜像 cp 出来的 `/submission`），本仓库 `src/` 是较早的快照，缺少镜像独有的 15 个模块（`adapter.py`、`component_export.py`、`ic_geometry.py`、`ic_name_selection.py`、`led_geometry.py`、`text_recovery.py`、`targeted_pin_ocr.py`、`nearby_tip_fields.py`、`power_caption_ocr.py`、`whole_network_export.py`、`wire_segments.py` 等），并且 `component_text.py` 两边不同源。**用本仓库重建镜像会丢掉这些逻辑（第 6 次评测 0 分即同类失误）**；改动应先落在部署副本上，再同步回本仓库。
