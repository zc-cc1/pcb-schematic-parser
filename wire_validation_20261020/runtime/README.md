# 平台测试提交包

这是现有完整 S24 流程的源码包：YOLO v3 检测 → RapidOCR → 冻结 U-Net v1 → 引脚归属、公共引脚库与字段补全 → 工程整网出口。
保留项目默认 detector_dataset300_v3_nano.pt 和 unet_wire_pin_v1.pt；10月15日的 YOLO11、U-Net v7 与独立线段解码实验尚未接入该完整流程，本包没有擅自替换默认权重。

## 上传

优先在本机构建、测试并导出 submission_target.zip，再上传该预构建镜像ZIP；此源码ZIP用于本机构建。
选择文件后平台可能立即开始提交。本包不使用大模型 API，无需密钥，Token 全部为 0。
GPU 默认自动选择：先执行真实 CUDA 矩阵运算，成功后选择 cuda:0；未开放GPU、驱动或架构不兼容时记录具体错误并回退 CPU。不更改 CUDA_VISIBLE_DEVICES，也不强制要求 GPU 以免平台缺设备时整批失败。

## 输出

- /output/final_results/<case_id>_Pred.json：默认使用实际 2026-10-08 评分协议的嵌套 pins、hyperGraph、edges 布局。
- /output/project_results/<case_id>_Pred.json：保留项目原始嵌套 pins、hyperGraph、edges 布局，便于排查。
- /output/final_results/decision_log.json 与 /output/decision_log.json：同时写两处；官方 README 和平台网页对该位置存在不一致。
- /output/logs/：每张图的运行日志、覆盖统计和字段来源。

value、H-y 坐标系、引脚坐标和网络成员保持。project_results 保留全部原始模型/OCR结果。
依据2026-10-16官方公开标注，r/c/l的Name使用已识别位号；已确认的重复位号实例保留原始基础名称。
r/c/l/d/led的评分框采用公开训练集一致使用的尺寸模板，保留模型预测中心与长轴方向。
模板短边/长边分别为r:8/20、c:16/20、l:6/34、d和led:14/20像素；不是按测试图片查表回填答案。
IC、二极管、LED的已有Name保持，200训练标注与10个Golden Case在IC Name等字段上存在版本差异。
赛题PDF第13、16、18、19页的JSON截图确认了components、嵌套pins和hyperGraph/edges结构，以及Name字段的大写N。
PDF示例使用ic/resistor等长类别名，与本次实际评分报告及平台页面的r编码存在冲突；不能把示例当作完整的当前类别表。
resistor/capacitor/inductor/diode/ic/led 对应 r/c/l/d/box/led；提交8180的145个匹配元件和公开标注已验证这六类，不代表覆盖全部类别。
sample/project 仅作显式调试选项；平台部署默认 target。显存不少于24 GiB时自动最多3例并行，小显存GPU和CPU默认串行；可用 --workers 1/2/3 覆盖。每例运行目录与配置缓存隔离。

## 本地运行

Docker（在解压目录执行；输入/输出路径使用绝对路径）：

    docker build -t pcb-parser:test .
    docker run --rm --gpus all -v <输入绝对路径>:/input:ro -v <输出绝对路径>:/output:rw pcb-parser:test -input /input -output /output

CPU 测试可移除 --gpus all，并添加 -e PCB_DEVICE=cpu（放在镜像名称之前）。
基础镜像 edarace-agent-base:ubuntu24 来自官方示例；本机需能获取该镜像，平台是否预置须看构建日志。
复用用户提供的官方 edarace-agent-base:ubuntu24 中 CUDA 12.1 的 torch 2.5.1 / torchvision 0.20.1，不叠加安装第二套CUDA，推理所需 YOLO/U-Net/OCR 权重均已随包带入。

已安装同版本依赖的 Python 也能运行：

    python agent.py -input <输入目录> -output <输出目录> --device cpu

PNG 可位于子目录，保留文件名的前导零。不同目录若出现同名 PNG 会报错，避免覆盖结果。
每例生成独立日志；部分图片失败时输出空 JSON 并继续，其失败原因计入 decision_log。全部失败或预检失败返回非零退出码。
默认清理临时大概率图；需要保留时添加 --keep-work。平台可能限制中间文件下载，日志也会在失败时打印关键末尾信息。

## 验证边界

上一版镜像8422已获28.92223分。本次包含无位号电源/地出口、LED文字与框、IC Name优先级和IC实框定位修复，具体构建/测试记录随新归档保存，尚需平台复测；不能沿用上一版分数或把离线诊断当成官方成绩。
原有完整网覆盖仍有限，工程编号规则尚未由官方 Golden 确认。打包通过不代表取得官方识别成绩。
不含训练图片、参考 JSON、预测缓存、训练过程、Windows 虚拟环境或 API 密钥。
package_manifest.json 列出逐文件哈希、模型、配置与仅提交副本中的可移植性修改。

## 电源元件出口修复

当前图像已检测的地/电源不再因缺位号而全部丢弃，供电类别转换为官方v。技术检测ID仅用于JSON唯一标识，评分忽略电源key/Name。value只保留已识别文字，不能确认时保持null；不根据符号类别统一猜GND。原有元件、引脚和网络保持。project_results保留原流程输出，component_results保留补齐电源后的转换输入。logs包含电源导出证据与缺失字段附近OCR，runtime_environment.json记录实际设备选择。

## LED 文字与框修复

同一字形的高置信度完整LEDn与截断EDn读法冲突时，不再将截断副本当作型号。LED评分框方向由当前图像两侧引线确定；中心仅在两个局部YOLO视野定位一致、不跨邻近LED归属、且偏移超过2.5px时修正，证据不足保留原坐标。固定公开尺寸仍为14×20或20×14。其余元件、文字、引脚、网络不受框修复影响。logs/<case>_led_geometry.json记录前后框、局部检测候选、方向证据与耗时；project_results与component_results保留几何修复前结果。公开数据诊断和受控偏移测试不代表线上失败框已恢复。

## IC Name 候选优先级

保留全部已有OCR行及观察读法到logs/<case>_ocr_texts.json；按可信型号、实际观察到的自身位号、其他器件说明选择IC Name，来源、附近候选和拒绝原因见logs/<case>_ic_name_selection.json。已绑定型号保留；不把引脚/电源标签、其他器件位号或技术检测ID用作名称，不从字典补造字符。该步骤仅作用于最终Target的IC Name，project_results、component_results及原有引脚/网络保持。型号缺失时U1会优先于2.4GHz、T_V等说明，这是用户选择的排序，尚不保证适合线上不同标注版本。

## IC 实框定位

最终Target的IC bbox从当前PNG中的四条直边和四角证据定位，替代检测框的普遍外扩；不按型号查固定尺寸，也不按线上误差猜平移。小于40px短边、缺边、多框竞争或不连续边界保留原框。logs/<case>_ic_geometry.json记录前后框、四边支持、候选和拒绝原因及耗时。原始project_results、component_results及Name、引脚、网络均不被几何步骤修改。公开配对诊断不是新官方分数，线上五个失败实例尚需新版复测。

## 官方运行时小包

本运行时面向比赛A800（sm_80），不支持本机RTX5060（sm_120）的GPU算子；本机需使用 --device cpu 验证。原cu128 Docker镜像保留用于RTX5060本地GPU测试。模型权重相同，结果仍须重新验证。
