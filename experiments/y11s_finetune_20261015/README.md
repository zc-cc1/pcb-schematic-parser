# YOLO11s小学习率无Mosaic精修

从原11s第76轮best开始新精修阶段。418张训练、30页校准、29页评分不变；1536、batch4、AdamW等保持，lr0=0.00005，最多40轮、patience12、warmup1、mosaic0、close_mosaic0。初始模型、图像和标注均另存副本。上一轮第101轮早停，尚未执行原定106轮开始的无Mosaic阶段，本次检验精修是否有收益。

以相同整图/切片/WBF和阈值政策对比父11s。推荐门槛只使用30页校准集；29页评分只报告结果，属于已查看的旧开发数据。保留best和last，last仅作诊断。GPU空闲后以run_finetune.py --launch-if-idle启动。

YOLO11s独立推理配置：pcb_parser_competition/configs/baseline_y11s.json；微调结束另写候选配置，不覆盖旧v8n和已有下游缓存。
