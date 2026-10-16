# YOLO11s 主模型接入记录

2026-10-16，按用户“把yolo11s作为主要模型使用”的决定，项目默认推理及后续提交构建已切换为 YOLO11s。没有启动训练，没有向平台提交。

## 采用的权重

- 配置：`configs/baseline_y11s.json`。
- 权重：`weights/detector_yolo11s_base_20261015.pt`。
- SHA256：`5c63d9510681d90b7cb54a36438a63d54d479ab6763a89d9b118afd06f3425d5`。
- 来源：`../experiments/y11s_matched_control_20261015`，30页校准集选出的第76轮best。
- 使用该模型30页校准所得的逐类阈值 `configs/class_thresholds_y11s_base.json`，没有复用v8n阈值。
- 精修候选未通过原定校准推荐门槛，继续保留为独立候选，不设为主模型。

## 已接入的位置

1. `configs/project.json` 的主模型登记；旧v8n的模型及指标移至 `previous_detector_v8n`，OCR旧指标明确标记测量模型。
2. `run_baseline.py`、`run_component_text.py`、`run_pin_ownership.py` 的默认配置，以及检测、OCR、引脚归属和当前完整流程的PowerShell入口。检测默认使用fused模式。
3. 平台适配器的LED局部YOLO复检，改为读取同一11s配置；不再隐式加载v8n。电源导出、IC/LED文字和几何步骤继续保留，U-Net仍为现有v1。
4. 提交构建按配置递归收集依赖和实际模型权重，自动带入11s逐类阈值文件。新源码包不包含旧v8n检测权重。
5. `decision_log.detector` 记录实际配置/权重哈希和校准轮次；LED复检日志记录实际权重哈希。

旧v8n配置、权重、历史实验与既有提交目录未被替换。检测/OCR及当前完整流程的默认输出改用新的11s目录。旧模型检测缓存不能直接用于新模型引脚归属流程。

## 验证

106项相关回归检查通过，覆盖平台结构/设备选择、电源出口、LED/IC几何、IC Name、文字绑定和引脚归属；增加了替换权重被拒绝及实际模型来源记录的检查。

使用已有 `pcb-parser:method1-348b97c5` 镜像中的官方运行时 **PyTorch 2.5.1+cu121**，断网、CPU、单进程运行。新源码以只读目录挂载；输入仅为两张PNG，未挂载标注。两张图都经过原始PNG检测、OCR、U-Net、引脚/网络及最终Target出口：

| 图片 | 完整处理成功 | 耗时 | 输出元件/引脚/网络 |
|---|---|---:|---|
| 0036 | 是 | 43.224秒 | 15 / 14 / 1 |
| 0147 | 是 | 67.295秒 | 53 / 32 / 1 |

两例实际检测缓存均记录11s主模型哈希；0147的3个LED复检条目也记录相同哈希。最终JSON结构与引用检查通过。新包111个清单文件在运行前后哈希一致，源码ZIP中的文件与清单逐项一致，ZIP全量CRC通过。

额外用0036的旧v8n检测缓存尝试新默认引脚归属入口，按预期报 `Detector cache uses a different checkpoint`，防止把旧预测当作11s结果。

这些是接入与兼容性验证，输出数量不能代表准确率。两张开发图不能证明线上成绩提高，也没有验证完整13图的时间预算、A800端到端运行或11s的新独立镜像构建。

详细机器记录：`dist/y11s_adoption_20261016/verification.json`；原始日志与最终JSON：`dist/y11s_adoption_20261016/container-output/`；修改前快照：`dist/y11s_adoption_20261016/before/`。

## 当前可用的源码包

- `dist/submission_y11s_main_source.zip`，94,544,036字节。
- SHA256：`f47ec6619198238a144d84a993a8c7fd35667e7108b4514485bc0c2af4441624`。
- 源码目录：`submission/platform_test_20261016_y11s_main_1`。
- 包含代码、11s/OCR/U-Net等运行资产及Dockerfile；这是源码构建ZIP，不是仅含submission.tar的预构建镜像ZIP。

本机可使用GPU环境中的Python直接运行该源码包，命令在项目根目录执行，输入为待解析PNG文件夹：

```powershell
& .\.venv-gpu\Scripts\python.exe -X utf8 .\submission\platform_test_20261016_y11s_main_1\agent.py -input <PNG文件夹绝对路径> -output <新输出文件夹绝对路径> --device cpu --workers 1
```

后续重新收集当前源码，默认构建器也将使用11s：

```powershell
& .\.venv-gpu\Scripts\python.exe -X utf8 scripts/build_platform_submission.py --runtime-profile official-cu121
```

`dist/local_image_build/build_test_export.ps1` 默认源码ZIP也已改为11s包；显式指定旧包仍可做历史复现。当前没有重新生成几GB的预构建镜像归档。

## 历史结果和回退

28.92223分属于v8n提交8422，不能标作11s成绩。原有IC/LED修复版v8n镜像和83MB的Dockerfile候选仍是旧模型，不会随工作目录配置自动更新，上传前应核对 `package_manifest.detector` 和真实权重哈希。

检测/OCR需要历史v8n对照时，显式传 `--config configs/baseline_v3_adopted.json` 并使用独立输出目录；引脚归属历史对照显式传 `--detector-config configs/pin_ownership_detector_v3.json`。完整历史平台流程使用之前保留的v8n源码快照。新的平台适配器与打包默认是11s，不能只换一个文件名就声称已回退。
