# 当前模型权重

开发默认权重：`detector_dataset300_balanced_v2.pt`，35类；从第一轮权重继续20轮约5.1分钟。相同59例验证集整图F1为0.8008、SAHI/IOS切片F1为0.8132。晶振/MOSFET开始检出但召回仍有限。SHA256：`57cbbf13d93b6d0e617fa0e580cc8c7dd7b7582b3f0bc7bb8a707cba6293b1bf`。

第二轮记录在 `runs/training/detector_dataset300_balanced_v2_record.json`，详细结果见 `docs/detector_balanced_training_report.md`；当前推理配置为 `configs/baseline_finetuned.json`，第一轮推理配置保留为 `configs/baseline_finetuned_v1.json`。

首轮微调权重：`detector_dataset300_v1.pt`，35类；训练20轮约4.2分钟。本地验证集整图F1为0.7755，晶振/MOSFET等类别仍需改善。SHA256：`4353e9be8dd0b4dfb295038fc2465fe7a7f28f23bb91e127ee048ac629642294`。

训练来源与指标记录在 `runs/training/detector_dataset300_v1_record.json`，最佳/末轮checkpoint在训练运行目录；推理配置为 `configs/baseline_finetuned.json`。

已下载并实际运行 `circuit_yolov8n_v3.pt`（16,779,529 字节），来源为 HR-DONK/circuit-recognition。

SHA256：`418a2fb5b9244146224e1e993aa8d27037ca33613865a1194642ff39b3283daf`。

详细来源和下载时间记录在 `baseline_provenance.json`，MIT 仓库许可副本在 `HR-DONK_LICENSE.txt`。GitHub API 的提交查询被限流，本次通过文件 SHA256 固定实际权重；Ultralytics 框架 AGPL-3.0 与仓库许可分别记录。

模型类别表：resistor、capacitor、inductor、diode、transistor、opamp、ic、ground、voltage_source、current_source。晶振等比赛类别未包含在这10类中。

RapidOCR 3.9.2 内置 PP-OCRv6_det_small、PP-OCRv6_rec_small 和 ch_ppocr_mobile_v2.0_cls_mobile ONNX 模型；实际路径与 SHA256 在运行 summary.json 中。模型来源及许可见 RapidOCR 官方记录。

真实推理结果在 `runs/s1_gpu_baseline/`，经过验证的依赖版本在 `requirements-gpu.lock.txt`。

候选电路权重不等于已覆盖比赛中的全部器件类型。
