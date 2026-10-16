# 外部原理图首批收集与使用计划

记录日期：2026-10-11。首批已下载 **10 个公开硬件工程、42 个文件（11.64 MiB）**，获得 **8 张真实原理图页面、10 份 CAD 源文件**。其中 9 份 CAD 已解析出 1300 个实例条目；条目包含电源标志等辅助对象，不能称为已验证的元件标注。

当前完成的是数据收集、图像生成、重复初筛和标注草稿。新增图像尚未训练；当前检测器与 59 张原验证页保持原状，不能宣称新增数据已提升准确率。

## 赛题附加分的适用范围

赛题 PDF 第 4 页允许自行收集数据用于训练优化；第 5 页说明数据贡献加分最高 10 分，按去重后的正确数据衡量。第 9 页把最低正确数据数量、具体计分和提交方式交由单独的《数据提交说明》规定，该材料目前尚未取得。

第 9 页“每队选取不超过 10 例”描述评估抽样上限，不表示下载 10 张图就能拿满 10 分，也不能据此推出最低提交数量。公开工程的来源、采集过程和新增标注应分别报告；是否被认可为可贡献数据、是否与 PCBnet 重复，仍需按该说明核验。当前没有提交或上传任何数据。

## 实际收集来源

| 工程 | PNG 页面 | CAD 格式 | 原生实例条目 | 稀有符号候选 | 上游硬件设计许可声明 | 当前用途 |
|---|---:|---|---:|---:|---|---|
| [OLIMEX/ESP32-EVB](https://github.com/OLIMEX/ESP32-EVB) | 1 | kicad_legacy | 261 | 6 | Apache-2.0 | 训练候选 |
| [OLIMEX/ESP32-POE](https://github.com/OLIMEX/ESP32-POE) | 1 | kicad_sexpression | 225 | 3 | Apache-2.0 | 训练候选 |
| [OLIMEX/ESP32-DevKit-LiPo](https://github.com/OLIMEX/ESP32-DevKit-LiPo) | 1 | kicad_legacy | 88 | 1 | Apache-2.0 | 训练候选 |
| [OLIMEX/RP2040-PICO30](https://github.com/OLIMEX/RP2040-PICO30) | 1 | kicad_legacy | 94 | 1 | CERN-OHL-S-2.0 | 训练候选 |
| [OLIMEX/RP2040-PICO-PC](https://github.com/OLIMEX/RP2040-PICO-PC) | 1 | kicad_legacy | 117 | 3 | CERN-OHL-S-2.0 | 训练候选 |
| [OLIMEX/STM32F4](https://github.com/OLIMEX/STM32F4) | 1 | 旧版 Eagle 二进制 | 未解析 | 未知 | CC-BY-SA-3.0-US | 训练候选 |
| [sparkfun/SparkFun_Thing_Plus-RP2040](https://github.com/sparkfun/SparkFun_Thing_Plus-RP2040) | 1 | eagle_xml | 150 | 2 | CC-BY-SA-4.0 | 外部测试候选 |
| [sparkfun/MicroMod_Processor-RP2040](https://github.com/sparkfun/MicroMod_Processor-RP2040) | 1 | eagle_xml | 81 | 1 | CC-BY-SA-4.0 | 外部测试候选 |
| [adafruit/Adafruit-Metro-RP2040-PCB](https://github.com/adafruit/Adafruit-Metro-RP2040-PCB) | 0 | eagle_xml | 162 | 3 | CC-BY-SA-3.0 | 待渲染 CAD |
| [adafruit/Adafruit-Feather-RP2040-PCB](https://github.com/adafruit/Adafruit-Feather-RP2040-PCB) | 0 | eagle_xml | 122 | 2 | CC-BY-SA_version_unspecified_in_README | 待渲染 CAD |

许可栏记录的是上游硬件设计声明。部分 OLIMEX 项目把文档与硬件分别声明：ESP32-POE、ESP32-DevKit-LiPo、RP2040-PICO30 的文档声明为 CC BY-SA 3.0，RP2040-PICO-PC 为 CC BY-SA 4.0；必须保留逐文件来源与声明，不能用软件 GPL 或一个项目级许可覆盖所有 PDF、CAD 和新增标注。Adafruit Feather 的 README 提到了 CC BY-SA，但引用的 license.txt 未取得，具体版本仍待确认。

首批包括 ESP32 的供电、以太网、PoE、CAN、继电器、USB-UART，以及 RP2040/STM32 的时钟、接口和电源电路。ESP32-EVB、PoE、DevKit-LiPo 存在相似子电路，后续按工程与共享电路复核，不把裁剪/不同 DPI 渲染视为新增独立样例。为增加风格覆盖，后续应优先补充模拟电路、电源驱动、分立 MOSFET 和两脚晶振工程。

Adafruit 两个工程目前只有 Eagle CAD，尚无本批可训练 PNG；没有把引脚分布图 PrettyPins 当成原理图。STM32-E407 的 PDF 已成功渲染，但旧版 Eagle 二进制 CAD 需要转换才能提取原生标注。另有两个可选文件缺失，详情记录在 audit.json 中。

## 已完成的核验

- 每个文件记录下载 URL、工程分组、分支、下载时间、大小和 SHA-256；原始 CAD、PDF、许可声明分别保存，渲染为 200 DPI PNG。
- GitHub API 本次未取得提交 SHA。清单以已下载内容的 SHA-256 固定版本，commit 字段为空，不把分支名当成不可变版本。
- 与 Dataset300 全部 300 页对比：精确 RGB 像素、64 位 pHash（距离 ≤5、宽高比相差 ≤8%）及已知工程名称文本；本批 0 页命中重复候选，批内也未命中。不同渲染、改版或局部相同电路仍可能避开检查，工程级和官方 PCBnet 去重未完成。
- 已预留 6 张 OLIMEX 页面为训练候选、2 张 SparkFun 页面为外部测试候选；同工程的所有版本、页面及裁剪继承同一划分。以后补齐 Adafruit 渲染也须先分组。外部测试候选未运行本次预测草稿，也未加入训练。
- 原检测清单 SHA-256 保持 `8261f5541e6036e994d8d1a8592eedf1bcc704ab1223e61b615e2a3ef3533812`，验证页数量仍为 59；Dataset300 原图和原始标注哈希已逐条核验。
- 原生清单筛出 15 个 MOSFET 候选和 7 个时钟符号候选，共 22 个。它们包括未渲染的 Adafruit CAD；晶振、振荡器、谐振器的最终类别需看图核对，未知位号例如 Q? 不自动当成有效赛事键。
- 当前 v2 权重在 RTX 5060 上为 6 张训练候选页生成 1220 个检测草稿框。模型草稿中可有误检和漏检，不能作为 Golden；OCR 本次未运行，完整 pins/nets 尚未生成。
- CAD 解析的 3 个针对性测试通过：实例与库定义分离、多行字段、电气 pinref 保留，以及二进制输入短错误处理。

## 接下来如何用于提升指标

| 阶段 | 输入 | 输出 | 本项目验收指标 |
|---|---|---|---|
| 完成检测标注 | 6 张训练候选图、CAD 清单、检测草稿 | 原图坐标的类别/框及复核记录 | 35 类映射明确；框有限且不越界；每页全量复核漏标，疑难项保留状态；15 个 MOSFET/7 个时钟候选逐个核实 |
| 扩充稀有类 | 当前首批及新增独立开源工程 | 有来源的独立样例与已复核对象 | 下一批内部目标：总计 50 个工程，晶振、MOSFET 各至少 50 个独立已核验对象；重复裁剪只计训练曝光，不计新数据贡献 |
| 数据拆分 | 已核验样例、工程/共享子电路信息 | 新训练清单与外部测试清单 | 原 59 页不变；训练与外部测试无同工程或已识别同电路泄漏；测试集不用于挑选裁剪和调参 |
| 第三轮训练与比较 | v2 权重、原训练样例、合格新增训练标注 | v3 权重和逐类对照报告 | 固定类别、置信度、IoU、SAHI 参数与原 59 页；同时在新增测试集比较 v2/v3；整体 F1 目标不低于 0.8132，晶振/MOSFET Recall 提升且 Precision 不明显下降。这是实验目标，不是已实测结果或赛事保证 |
| 完整赛事 JSON | 同页 CAD/图像、完整元件/引脚/网络复核 | 与官方字段和坐标约定一致的 JSON | 元件键、Name、bbox、pin 坐标及 net 引用完整一致；网络电气连通人工复核；官方资料到齐后用正式格式和评测器验收 |
| 数据贡献准备 | 去重后的完整正确案例、原始来源与许可声明 | 来源记录、标注说明、可审核贡献包 | 数量、可接受来源、正确性和许可发布范围按《数据提交说明》核验；当前 bonus_eligible=null，不报告得分 |

首批时钟样例数量很小，还不足以证明能修复两脚晶振的泛化问题。先扩大并核验目标符号，再训练 v3；直接把当前模型的错框回灌训练会延续原有偏差。当前模型的本地 SAHI F1 仍为 81.32%，没有新的实测提升。

## 文件与复现

- `configs/external_sources.json`：可扩展的数据来源清单；`scripts/collect_external_schematics.py`：下载、来源哈希、PDF 渲染、原生 CAD 清单。
- `data/external_collection_v1/raw/`：原始文件及逐文件 `.provenance.json`；`images/`：8 张 PNG；`manifest.jsonl`：来源与图像信息。
- `data/external_collection_v1/duplicate_review.json`：去重证据与最近邻；`review_manifest.jsonl`：预留划分；`rare_symbol_review.csv`：22 个符号复核队列。
- `data/external_collection_v1/annotation_tasks.json`：8 页本地标注任务，草稿和真值状态分开；其中 2 页外部测试任务没有预测草稿。它是本项目复核格式，尚非官方 Golden，也不是已确认的 Label Studio 导入格式。
- `runs/external_collection_v1_proposals/`：6 页 GPU 检测草稿与 overlay；`artifacts/external_data_review/collected_pages.png`：原理图总览。

在项目根目录复现：

```powershell
$pcbPdfPython = 'C:\Users\Pc\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
& $pcbPdfPython scripts/collect_external_schematics.py
& .venv-gpu/Scripts/python.exe scripts/review_external_collection.py
.\scripts\run_external_collection_proposals.ps1
# 报告工具核对草稿与模型的 SHA-256 后写复核任务。
& $pcbPdfPython scripts/build_external_collection_report.py
```

收集工具使用带 pdfplumber 的文档 Python，去重工具使用已安装 NumPy/scipy 的 GPU 环境；现有两个环境各自保持原状。重复运行会验证已下载原件哈希，不覆盖内容不同的原始文件。PNG 为派生文件，可由原 PDF 重新生成。
