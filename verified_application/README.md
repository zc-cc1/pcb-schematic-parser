# verified_application — PCB 原理图解析推理包

完整推理流程源码包:YOLO 元件检测 → RapidOCR → 冻结双头 U-Net(导线掩膜 + 引脚尖端)→ 引脚归属、公共引脚库与字段补全 → 工程整网出口。

保留项目默认权重 `detector_dataset300_v3_nano.pt`(元件检测)与 `unet_wire_pin_v1.pt`(导线/引脚);后续的 YOLO11、U-Net v7 与独立线段解码实验未接入本流程,未替换默认权重。

## 输出

- `/output/final_results/<case_id>_Pred.json`:默认输出,嵌套 pins、hyperGraph、edges 布局
- `/output/project_results/<case_id>_Pred.json`:保留原始模型/OCR 结果,便于排查
- `/output/final_results/decision_log.json` 与 `/output/decision_log.json`:同时写两处,兼容不同消费路径
- `/output/logs/`:每张图的运行日志、覆盖统计、字段来源与各修复步骤证据

value、H-y 坐标系、引脚坐标和网络成员保持不变。

- r/c/l 的 Name 使用已识别位号;已确认的重复位号实例保留原始基础名称
- r/c/l/d/led 的输出框采用与训练数据一致的尺寸模板,保留模型预测中心与长轴方向(模板短边/长边 r:8/20、c:16/20、l:6/34、d 与 led:14/20 像素)
- IC、二极管、LED 已有 Name 保持;型号缺失时按「可信型号 → 自身位号 → 其他器件说明」优先级选择,来源与拒绝理由写入 `logs/<case>_ic_name_selection.json`
- 类型别名:r/c/l/d/box/led 对应 resistor/capacitor/inductor/diode/ic/led
- 电源/地:已检测的无位号电源不再丢弃,供电类别转换为 `v` 表示;技术检测 ID 仅用于 JSON 唯一标识,评分忽略电源 key/Name
- sample/project 仅作显式调试选项,默认 target

## 本地运行

Docker(在解压目录执行;输入/输出路径使用绝对路径):

    docker build -t pcb-parser:test .
    docker run --rm --gpus all -v <输入绝对路径>:/input:ro -v <输出绝对路径>:/output:rw pcb-parser:test -input /input -output /output

CPU 测试可移除 `--gpus all`,并添加 `-e PCB_DEVICE=cpu`(放在镜像名称之前)。

已安装同版本依赖的 Python 也能运行:

    python agent.py -input <输入目录> -output <输出目录> --device cpu

基础镜像 `edarace-agent-base:ubuntu24`(CUDA 12.1,torch 2.5.1 / torchvision 0.20.1);推理所需 YOLO/U-Net/OCR 权重均已随包带入,不叠加安装第二套 CUDA。

PNG 可位于子目录,保留文件名的前导零;不同目录若出现同名 PNG 会报错,避免覆盖结果。每例生成独立日志;部分图片失败时输出空 JSON 并继续,失败原因计入 decision_log;全部失败或预检失败返回非零退出码。

GPU 自动选择:先执行真实 CUDA 矩阵运算,成功后选择 cuda:0;驱动或架构不兼容时记录具体错误并回退 CPU,不更改 `CUDA_VISIBLE_DEVICES`。显存不少于 24 GiB 时自动最多 3 例并行,小显存 GPU 和 CPU 默认串行;可用 `--workers 1/2/3` 覆盖,每例运行目录与配置缓存隔离。

本运行时面向 sm_80 GPU;sm_120 等新架构请使用 `--device cpu`。默认清理临时大概率图,需要保留时添加 `--keep-work`,关键末尾日志也会在失败时打印。

## 已知边界

- 完整整网覆盖仍有限;工程编号规则尚未经参考标注确认
- 打包通过不代表任何特定评测结果;各文档中的分数均为本地口径,详见 `docs/` 各说明

## 说明

- `package_manifest.json` 列出逐文件哈希、模型、配置与可移植性修改
- 本包不含训练图片、参考 JSON、预测缓存、训练过程或 API 密钥
- 电源出口、LED 文字与框、IC Name 优先级、IC 实框定位等修复的证据与前后对比均写入 `logs/`
