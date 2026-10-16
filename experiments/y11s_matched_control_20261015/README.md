# YOLO11s匹配训练实验

等待GPU空闲后启动一个种子20261012。训练418张，校准30页，评分29页；1536、batch4、AMP、AdamW、lr0=0.0003、最多120轮、patience25及其他参数复制上一组冻结配方。独立保存数据副本、权重与报告；不改部署模型。若显存不足，不自动更改batch或分辨率。

使用run_y11s.py --launch-if-idle检查并启动；完成后生成comparison.json和comparison.md，包含与现有v8n、同配方v8n及11n的P/R/F1、整图AP和逐类差异。
