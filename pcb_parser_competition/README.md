# PCB 原理图解析比赛项目

当前主要探测器（2026-10-16）：按用户决定，默认使用 **YOLO11s**，配置为 `configs/baseline_y11s.json`，权重为 `weights/detector_yolo11s_base_20261015.pt`（30页校准集选出的第76轮best）。检测、OCR绑定、引脚归属、提交构建及LED局部复检均使用这一模型与它自己的逐类阈值。精修候选未通过校准门槛，暂不设为默认。运行输出使用新的11s目录；v8n配置、旧缓存和旧提交归档保留作历史对照。现有28.92分属于v8n提交，11s尚无线上成绩。接入验证、运行与回退方法见 [YOLO11s主模型接入记录](docs/yolo11s_main_adoption_20261016.md)。下方为各阶段历史记录，其中旧默认路径与当时打包状态以本条更新为准。

IC 框定位修复（2026-10-16，源码改动尚未打包）：依据当前 PNG 四条直边和四角定位 IC 实框，缺边、框竞争及小符号保留原框。39 张公开图的 196 个已配对 IC 中，5 px 内数量 12→96、20 px 内 181→190，平均最大坐标误差 10.179→5.314 px；原通过案例未跌出对应阈值。这是开发子集诊断，不是官方 F1。92 项相关测试、官方运行时断网原始 PNG 完整入口与三例定位兼容性核对通过；几何步骤只改 IC bbox。线上 5 个框失败实例尚待新提交确认，旧 ZIP 不包含本次改动。详见[IC 框修复说明](dist/ic_bbox_fix_20261016/IC框修复说明.md)。

IC Name 候选优先级（2026-10-16，源码改动尚未打包）：按用户选择的“型号→已观察位号→其他说明”填最终 Target Name，完整已有 OCR 和逐器件候选/来源分别保留日志，保护其他字段及已有型号。26 页缓存回放中 29 个 IC 的 7 个空 Name 补齐、20 个已有型号保留；可对照官方训练标注的 15 个 IC 严格 Name 匹配仍为 12→12，不代表提分。80 项相关测试与官方镜像断网字段出口验证通过。此顺序也会使 LDO→L1、JTAG-20PIN→JTAG，说明仍保留为候选。详见[IC 名称候选与优先级](dist/ic_name_priority_20261016/IC名称候选与优先级.md)。

LED 框定位修复（2026-10-16，源码改动尚未打包）：方向由当前图像两侧引线确定，中心仅在两个局部冻结 YOLO 视野定位一致、不跨邻近 LED 且偏移超过 2.5 px 时修正。公开图 43 个已配对 LED 的 5 px 内数量由 40 增至 42；32 个受控初始偏移例子中 21 个恢复到 5 px 内，均未进一步变差。65 项相关测试和官方基础镜像断网 CPU 完整入口验证通过；9 个 LED 页的新增定位步骤耗时 2.77 秒。线上失败 PNG 不可获得，尚不能确认原三处失败已恢复，也未产生新官方分数。详见[LED 框修复说明](dist/led_bbox_fix_20261016/LED框修复说明.md)。

LED 文字绑定修复（2026-10-16，源码改动尚未打包）：已复现同一位置的 LED1/ED1、LED3/ED3 两种 OCR 读法分别被当作位号和型号的漏洞。针对高置信度完整位号与同字形残缺副本的冲突修复，保留独立型号和无证据的 null Name；55 项测试及 6 页真实 OCR/检测缓存回归通过。线上两例缺少原始 OCR，实际成因尚不能完全确定。详见[LED 文字绑定修复](dist/led_reference_binding_fix_20261016/LED文字绑定修复.md)。

电源出口修复（2026-10-16，尚未生成新镜像包）：针对 8422 的 28.92223 分结果，在平台边界补导出已检测的无位号电源/地，保留实际 OCR value 和原有引脚/网络，新增缺字段与设备环境日志。26 项测试及两张 PNG 容器验证通过；17 张同像素公开图的电源匹配诊断为 157/177，并非官方分数。现有 3.56 GB ZIP 仍是 8422 旧版，重建前不含本次改动。详见[电源出口修复与验证](dist/component_power_fix_20261016/电源出口修复与验证.md)。

最新复测包（2026-10-16）：复用官方 Ubuntu 24.04 / PyTorch 2.5.1 / CUDA 12.1 基础镜像，将公开数据修复版从 4.98 GB 缩至 3.56 GB，减少 1.42 GB（28.44%）。模型和算法文件保持，2 张 PNG 断网 CPU 验证及归档完整性核验通过；相比原包 GPU 结果仅坐标有最大 0.024 px 差异。官方运行时包含 A800 的 sm_80 支持但尚未实测 A800；不支持本机 RTX 5060 GPU，原 cu128 Docker 镜像保留。旧提交包和重复归档已按用户要求清理约 42.97 GB，最新 ZIP 保留。上传路径和验证范围见[最新上传说明](dist/上传说明.md)。

最新平台进度（2026-10-16）：提交8180已获官方20.449074分。用户补充的200训练案例、10个Golden Case与337条目元件库已完成审计；公开标注确认r/c/l的Name使用位号，r/c/l/d/led使用固定尺寸评分框。已修复平台适配出口，保留原始模型结果、value、IC字段、引脚和网络，16项适配器回归及两张PNG断网GPU测试通过。40张公开图的配对框几何诊断显示误差显著下降，该诊断不代表新官方成绩。电源/网络覆盖、IC Name版本差异仍待处理。详见[公开数据核对与修复说明](dist/official_public_dataset_audit_20261016/公开数据核对与修复说明.md)；上传文件以[最新上传说明](dist/上传说明.md)为准。

最新进度（S24，2026-10-13）：按完整OCR框的器件半边消解同行左右名称竞争，补0036/U1.4的CAGC；用唯一顶部L1＋独立LDO底部文字＋已有五个观测引脚恢复0018可见位号。26页工程完整net 3→5、完整pin266→272，新增`(C14.1,L1.4)`、`(C4.1,U1.4)`；两条新增网全部成员通过同一草稿归属/5px检查。原三条整网、1477个tip及S23几何/成员、保守JSON保持。382项回归、两页原始PNG新字段/整网/输出线段复现通过；0018未导出片段的1px骨架变化与候选ID排序交换另行记录。工程编号、官方身份和指标仍未验证，整体覆盖仍低。入口`scripts/run_observed_net_fields.ps1`，详见[可见字段补齐整网阶段](docs/observed_net_fields_stage_v1.md)。

最新进度（S23，2026-10-13）：用冻结U-Net的三尺度峰及模型导线路径修正0036的4个电容、8个端点，C1/C3关键点的同器件草稿距离约9px→2.58/1.81px。8点归属/5px匹配0→8，两条工程整网的所有成员坐标/归属匹配0→2；26页归属＋位置草稿代理F1为65.02%→65.59%，不是官方指标。工程整网仍3、pin仍266，所有成员/语义字段及保守JSON保持；局部坐标与8条边端点明确改变，未按参考或框边吸附。360项回归及0036原始PNG33.5秒验证通过。入口`scripts/run_capacitor_tip_geometry.ps1`，详见[电容端点定位阶段](docs/capacitor_tip_geometry_stage_v1.md)。

最新进度（S22，2026-10-13）：按用户选择增加独立工程出口，已识别R/C/L的合格双端点按左/上1、右/下2编号，名称取编号，并记录规则来源。26页工程完整net为3，新增0036两条三端网`(C1.2,L1.1,U1.2)`、`(C3.1,R1.2,U1.3)`；保守整网仍1且输出保持。撤回5处0170开放导线tip的被动元件归属，并修正反向边的端点检查；所有原点、边和成员保留。345项回归及0036原始PNG验证通过。两个电容tip距草稿参考约9px，编号约定和官方指标仍待验证。入口`scripts/run_engineering_networks.ps1`，详见[工程整网出口](docs/engineering_network_export_stage_v1.md)。

最新进度（S21，2026-10-13）：集中处理整网出口，打通首个本地完整网络 `(JTAG.15,RESET#.1)`；图上唯一JTAG标题与JTAG-20PIN底部OCR共同支持接口标识，未生成J1或后缀。26页保守整网出口0→1、候选1→2，完整pin123→132；标题身份策略未获官方Golden确认。新增逐网全成员引用证明、重复tip单接点拦截及元件端点漏成员检查。323项回归与0042原始PNG25.3秒验证通过。入口 `scripts/run_whole_networks.ps1`，详见 [整网输出阶段](docs/whole_network_export_stage_v1.md)。

最新进度（S20，2026-10-13）：新增同一物理引线近邻tip字段补全，0018/U3.8补入已有OCR名称INT，严格完整pin123、候选139，严格net仍0、候选仍1。方向由模型点/器件框判定，解决新PNG次级tip缺方向元数据的回放差异；保留所有原tip、编号、名称和成员。304项回归、0018原始PNG全流程54.2秒验证通过。独立入口 `scripts/run_nearby_tip_fields.ps1`；详见 [S20说明](docs/nearby_tip_fields_stage_v1.md)。

最新进度（S19，2026-10-13）：新增离线参考诊断，覆盖26页617个已输出字段，区分坐标超限、器件未匹配、草稿缺字段、字段差异和一对一冲突。S18的5个未匹配新增字段均为同器件坐标超过5px，其中U6编号3为5.147px；数码管草稿还存在字面错误。保留5px规则、原预测和参考，严格pin122/net0不变。278项回归通过。复跑：`.\.venv-gpu\Scripts\python.exe scripts/audit_pin_references.py`；详见 [S19说明](docs/pin_reference_audit_stage_v1.md)。

最新进度（S18，2026-10-13）：局部引脚OCR在26页旧开发集补回9个可见编号和2个名称，严格完整pin116→122、候选131→138；严格net仍0。已扣留IN截读成N的片段，修复字段扣留网络漏报电源成员风险的问题；仅字段未决网络从S17旧记录305修正为296。266项回归与0018原始PNG验证通过，保留独立试验入口 `scripts/run_targeted_pin_ocr.ps1`。详见 [S18说明](docs/targeted_pin_ocr_stage_v1.md)。

上一轮（S17）：撤回6处0402封装文字被绑定为被动器件编号的预测，保留原OCR证据并清除失效派生引用；按出口拦截条件和共享任务排序。旧队列中305个仅字段未决已在S18修正为296。入口 `scripts/run_network_completion.ps1`；详见 [S17说明](docs/network_completion_stage_v1.md)。

上一轮进度（S16）：电源文字局部OCR冻结YOLO/U-Net/引脚库，134个局部ROI补足28处电源文字证据；26页旧开发集的合格电源节点28→56、逻辑候选组3→7。同一电源草稿参考的全量连接对代理F1 25.41%→33.97%，条件FP也从3增到12；尚未通过独立测试或官方评测，保留试验入口。运行、限制与验证见 [S16说明](docs/power_caption_ocr_stage_v1.md) 和 [最新下游进度](docs/downstream_implementation_plan.md)。

```powershell
.\scripts\run_power_caption_ocr.ps1 -Image .\data\standardized\0018\0018.png -Device cpu
```

以下保留早期阶段的迁移、训练和数据准备记录；其中“待训练”“pins/nets为空”等描述属于对应历史阶段，当前状态以顶部最新记录为准。

最新下游决策：按用户要求，导线分割与引脚tip定位采用双输出U-Net。已固定GitHub源码版本、接入数据/训练/图片推理入口，生成300页标签草稿，其中132页通过结构完整性检查；8项新测试及CPU的4步训练/PNG推理链路验证通过。正式权重与识别指标尚未产生，GPU当前仍用于其他训练。路线与标签复核要求见 [U-Net导线与引脚方案](docs/unet_wire_pin_plan.md)。

当前阶段：已从用户桌面项目导入v3最终代码、权重和逐类阈值，默认冻结v3用于下游开发。源GPU记录59例检测F1为88.30%，本机CPU复测为88.24%。OCR与文字绑定已采用semantic_v2：增加局部复读、全局歧义判断和重复框归属处理。6张开发页共155个人工复核字段，位号绑定精确率97.18%、位号覆盖率82.14%、数值严格匹配率93.55%、型号严格匹配率88.89%。详见 [OCR优化实测报告](docs/component_text_optimization_report.md)。这些是开发集字段proxy，引脚、导线骨架与网络仍待实现，尚无官方F1。迁移证据见 [v3迁移与OCR首版报告](docs/v3_import_and_ocr_binding_report.md)。

后续按 [检测之后的实施计划](docs/downstream_implementation_plan.md) 推进：先完成OCR阶段覆盖率验收，再接入引脚、导线拓扑、NetworkX组网与完整JSON联合评测。图片推理入口读取PNG、检测预测和模型配置；配套标注只用于单独的离线评价。

在项目根目录运行当前完整的“v3检测→OCR→字段绑定”两例流程（CPU默认便于与其他GPU训练并行）：

```powershell
.\scripts\run_ocr_binding.ps1
# 只改绑定规则时，复用经过图像哈希/配置校验的预测缓存：
.\scripts\run_ocr_binding.ps1 -ReuseDetections -ReuseTexts
# 显卡空闲后可将检测部分改为GPU，OCR仍使用CPU：
.\scripts\run_ocr_binding.ps1 -Device 'cuda:0'
```

修改 `configs/baseline_v3_adopted.json` 中 `component_text` 配置调试OCR分块/方向、局部复读和绑定规则；版本快照为 `configs/component_text_v2.json`，兼容默认配置须同步。基础OCR或局部复读参数发生变化时须重新识别，不能使用旧的完整文字缓存。产物在 `runs/s3_component_text_quick_dev/<case>/`：`base_texts.json`、`texts.json`、`recovery_debug.json`、`component_records.json`、`components_partial.json`、`binding_debug.json`、`overlay.png`。文字保留实际识别的原字符；不确定项保留为空并记录候选，pins/nets尚为空。

运行独立的开发例文字评价与回归：

```powershell
& .\.venv-gpu\Scripts\python.exe scripts/evaluate_component_text.py --baseline-run runs/s3_component_text_v1_archive/prediction
& .\.venv-gpu\Scripts\python.exe scripts/build_text_optimization_report.py
& .\.venv-gpu\Scripts\python.exe scripts/run_regression_tests.py
```

原两例评价使用101个本地人工复核字段，额外0007/0042/0131/0229的54字段及产物在 `runs/s3_component_text_crosspage_v2/`；汇总在 `runs/s3_component_text_quick_dev/optimization_aggregate.json`。6页均已用于开发，不能将成绩作为独立测试或赛事分数。47项解析/融合/文字回归及文档环境的3项CAD清点测试通过，共50项。

已完成首批外部原理图收集：OLIMEX、SparkFun、Adafruit 共10个工程，8张原理图PNG、10份CAD源文件；保存来源、许可声明与内容哈希。与Dataset300做像素/pHash/工程名初筛未命中重复候选，工程级和官方PCBnet去重仍待复核。6页预留为训练候选，2页预留外部测试；已生成6页检测草稿和22个原生稀有符号复核条目。新数据尚未完成标注或用于训练，指标没有因此更新。详见 [外部数据收集报告及实施计划](docs/external_data_collection_report.md)。附加分最低数量与计分仍需单独的《数据提交说明》，不能把下载工程数量当成得分。

第二轮历史实验保留238张原训练页，新增180张训练局部图，在相同59张验证图上比较。晶振整图F1为0.1176、切片0.4000，MOSFET整图0.4800、切片0.4667；两类第一轮整图均为0。详见 [第二轮增样与微调报告](docs/detector_balanced_training_report.md)。这些指标仅评价类别和框，数据未完成全量人工认证，验证集已被用于多轮开发。

首轮微调使用238张训练图、59张验证图，0001/0002固定为验证。看图复核411个裁剪对象，记录83项人工修正或排除；训练20轮约4.2分钟。详见 [检测器微调实测报告](docs/detector_training_report.md)。

GPU 环境：`.venv-gpu/Scripts/python.exe`，PyTorch `2.10.0+cu128`，torchvision `0.25.0+cu128`。RTX 5060 已通过真实 CUDA 矩阵、卷积及 NMS 检查。详见 [GPU 与基线实测报告](docs/gpu_baseline_report.md)。

历史基线检测/OCR性能入口（默认已改用v3 fused；完整文字语义输出使用上面的run_ocr_binding）：

```powershell
.\scripts\run_quick_dev.ps1
```

v3固定检测产物在 `runs/s3_v3_detector_quick_dev/`，最新文字字段产物在 `runs/s3_component_text_quick_dev/`。第二轮历史产物在 `runs/s2_balanced_quick_dev/`，第一轮在 `runs/s2_finetuned_quick_dev/`，原始权重基线在 `runs/s1_gpu_baseline/`。历史基线Name/value未实现；新文字入口已输出型号/数值，pins/nets仍为空，不能当作最终完整提交。

第一轮检测数据与标签在 `data/detection_v1/`，第二轮派生训练数据在 `data/detection_balanced_v2/`。原始图片/JSON只读；类别修正只用于检测训练标签，先前的赛事结构参考JSON仍保留原转换语义。默认配置为 `configs/baseline_v3_adopted.json`，兼容入口 `configs/baseline_finetuned.json` 同步指向v3。第一轮和第二轮配置分别保留为 `configs/baseline_finetuned_v1.json`、`configs/baseline_finetuned_v2.json`。v3来源证据和原有文件备份在 `data/local_imports/v3_20261011/`。

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

完整开发计划在工作区 `output/PCB五模块方案实施计划.md`。材料清点、全量结构转换、GPU基线和前两轮微调实验保留；本轮已冻结v3并完成OCR优化与6页开发评价。下一轮接入二端器件引脚tip定位和导线分析，同时追踪复杂蓝色页面的文字覆盖率与歧义；正式评测器/元件库尚待取得。官方F1保持未验证状态。
