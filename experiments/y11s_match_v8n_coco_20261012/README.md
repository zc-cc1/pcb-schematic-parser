# YOLO11s 按 YOLOv8n 末阶段配置训练

用户选择：从 COCO 预训练 YOLO11s 重新训练，而不是续训现有 YOLO11s。

训练参数逐项复制自已采纳 v8n 的 `effective_training`：最多120轮、imgsz1536、batch4、workers2、CUDA0、AMP、AdamW、lr0=0.0003、lrf=0.1、weight_decay=0.0005、warmup3、patience25、seed20261012、deterministic=true、mosaic0.35、close_mosaic15、scale0.4、translate0.1、禁用旋转/翻转、hsv_v0.25、max_det1000。patience25与原配置一致，因此框架可在长期不改善时提前停止。除模型与初始化权重、输出路径外，不主动修改训练配置。

原训练数据418张，验证59张。477个样本的图像与标签SHA256已经逐项核验，manifest和dataset.yaml哈希与原v8n训练记录一致，来源分组无train/val交叉。COCO YOLO11s使用现存官方权重，SHA256已记录；AMP检查使用现存yolo11n权重，避免额外下载。

## 文件

- `experiment_config.json`：冻结配置、输入哈希、软件版本、比较限制。
- `run_experiment.py`：训练、进度记录及完成后自动评测的完整管线。
- `status.json`：实时阶段、已完成轮数、mAP、更新时间。
- `training_record.json`、`effective_args.json`：实际参数及运行记录。
- `stdout.log`、`stderr.log`：后台日志。
- `runs/y11s_coco_matched_v8n/weights/best.pt`、`last.pt`：新权重。
- 完成后生成 `comparison.md`、`comparison.json` 及 `evaluation/` 下预测缓存。

输出独立存放，原项目及已采纳检测器不被替换。第一次初始化的设备类型校验误把CUDA整数0与等价字符串"0"判为不同；在开始任何epoch前终止，修正后重新启动；相关初始化日志保留为`init_check_*`，空训练目录保留为`runs/y11s_init_check_failed`。

## 对比口径

训练完成后用同一59页、IoU0.5最大基数匹配、whole+sliced融合配置重新推理v8n、新y11s best和last；首先核对v8n原工作点0.883043能否复现，再分别报告conf0.2全局结果及相同方法的逐类阈值重调结果。重调固定推理conf0.2，避免原脚本在conf0.05时触发SAHI算法切换。包含逐类/逐页计数、P/R/F1、20000次配对页bootstrap和预热后推理耗时。

这对齐的是最后一阶段的训练参数。v8n有任务热启动历史，新y11s从COCO初始化，因此不能将差异全部归因于架构。阈值在同一验证集上选择和计分，成绩是探索性的本地bbox/category F1，非独立测试成绩、非官方ComponentF1。新模型不自动替换部署模型。

## 恢复

若中断时last.pt仍保留optimizer，停止原进程后可使用同一Python执行 `run_experiment.py --resume`，自动沿用训练checkpoint中的配置；程序会验证与冻结训练参数一致。

如果训练已结束而评测中断，执行 `run_experiment.py --evaluate-only`。完成后的导出best/last通常已剥离optimizer，不能按中断训练直接resume。
