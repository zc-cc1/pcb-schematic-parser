# 引脚归属模块 v1 实测

输入为冻结 YOLO v3 元件预测和 U-Net v2 引脚预测。模型、坐标、阈值均不变；新增的是元件归属关系。所有候选保留，含歧义和无元件附近的点。
配置在本轮首次评估前固定，未针对验收页继续调参。26 页仍属于已参与开发的旧验证集，参考为草稿。

| 范围 | 原引脚坐标 F1 | 已分配候选的坐标 F1 | 元件归属＋坐标联合 F1 |
|---|---:|---:|---:|
| 13 张开发图 | 66.26% | 61.88% | 58.43% |
| 13 张保留验收图 | 72.58% | 71.39% | 69.10% |
| 全部 26 张 | 70.17% | 67.84% | 65.12% |

候选状态：{'assigned': 1268, 'no_component_nearby': 128, 'ambiguous_owner': 60, 'deep_inside_large_component': 21}。
已定位且已分配候选的归属一致性：95.99%；正确归属且定位的覆盖率（原始坐标 TP 为分母）：86.34%。

联合指标同时要求预测元件类别/框正确、引脚归属正确、引脚点≤5px；没有评价编号和 pinname，因此不是正式 PinF1。已分配候选的成绩不表示剩余候选可以删除。
引脚编号、名称和电气网络仍待处理；当前只输出候选关系，不把未知编号换成排序编号。

标注复核队列包含 423 条未匹配点；这是复核建议，没有改变参考标注或 approved_cases。

## 逐页结果

| 图片 | 分配/全部候选 | 联合 F1 | 归属图 |
|---|---:|---:|---|
| 0002 | 28/29 | 39.29% | [查看](<C:/Users/Pc/Desktop/新建文件夹/pcb_parser_competition/runs/s5_pin_ownership_v1/0002/ownership_overlay.png>) |
| 0007 | 59/78 | 10.71% | [查看](<C:/Users/Pc/Desktop/新建文件夹/pcb_parser_competition/runs/s5_pin_ownership_v1/0007/ownership_overlay.png>) |
| 0012 | 94/107 | 86.15% | [查看](<C:/Users/Pc/Desktop/新建文件夹/pcb_parser_competition/runs/s5_pin_ownership_v1/0012/ownership_overlay.png>) |
| 0018 | 116/132 | 84.68% | [查看](<C:/Users/Pc/Desktop/新建文件夹/pcb_parser_competition/runs/s5_pin_ownership_v1/0018/ownership_overlay.png>) |
| 0022 | 25/37 | 50.00% | [查看](<C:/Users/Pc/Desktop/新建文件夹/pcb_parser_competition/runs/s5_pin_ownership_v1/0022/ownership_overlay.png>) |
| 0026 | 61/80 | 78.12% | [查看](<C:/Users/Pc/Desktop/新建文件夹/pcb_parser_competition/runs/s5_pin_ownership_v1/0026/ownership_overlay.png>) |
| 0029 | 72/82 | 63.35% | [查看](<C:/Users/Pc/Desktop/新建文件夹/pcb_parser_competition/runs/s5_pin_ownership_v1/0029/ownership_overlay.png>) |
| 0033 | 39/42 | 69.88% | [查看](<C:/Users/Pc/Desktop/新建文件夹/pcb_parser_competition/runs/s5_pin_ownership_v1/0033/ownership_overlay.png>) |
| 0036 | 28/30 | 43.08% | [查看](<C:/Users/Pc/Desktop/新建文件夹/pcb_parser_competition/runs/s5_pin_ownership_v1/0036/ownership_overlay.png>) |
| 0042 | 36/50 | 67.53% | [查看](<C:/Users/Pc/Desktop/新建文件夹/pcb_parser_competition/runs/s5_pin_ownership_v1/0042/ownership_overlay.png>) |
| 0049 | 69/72 | 82.76% | [查看](<C:/Users/Pc/Desktop/新建文件夹/pcb_parser_competition/runs/s5_pin_ownership_v1/0049/ownership_overlay.png>) |
| 0096 | 43/58 | 37.62% | [查看](<C:/Users/Pc/Desktop/新建文件夹/pcb_parser_competition/runs/s5_pin_ownership_v1/0096/ownership_overlay.png>) |
| 0105 | 85/90 | 77.17% | [查看](<C:/Users/Pc/Desktop/新建文件夹/pcb_parser_competition/runs/s5_pin_ownership_v1/0105/ownership_overlay.png>) |
| 0131 | 27/28 | 72.41% | [查看](<C:/Users/Pc/Desktop/新建文件夹/pcb_parser_competition/runs/s5_pin_ownership_v1/0131/ownership_overlay.png>) |
| 0148 | 64/73 | 49.70% | [查看](<C:/Users/Pc/Desktop/新建文件夹/pcb_parser_competition/runs/s5_pin_ownership_v1/0148/ownership_overlay.png>) |
| 0149 | 28/30 | 77.19% | [查看](<C:/Users/Pc/Desktop/新建文件夹/pcb_parser_competition/runs/s5_pin_ownership_v1/0149/ownership_overlay.png>) |
| 0170 | 239/260 | 74.29% | [查看](<C:/Users/Pc/Desktop/新建文件夹/pcb_parser_competition/runs/s5_pin_ownership_v1/0170/ownership_overlay.png>) |
| 0208 | 0/0 | 0.00% | [查看](<C:/Users/Pc/Desktop/新建文件夹/pcb_parser_competition/runs/s5_pin_ownership_v1/0208/ownership_overlay.png>) |
| 0213 | 24/29 | 48.15% | [查看](<C:/Users/Pc/Desktop/新建文件夹/pcb_parser_competition/runs/s5_pin_ownership_v1/0213/ownership_overlay.png>) |
| 0227 | 18/30 | 69.77% | [查看](<C:/Users/Pc/Desktop/新建文件夹/pcb_parser_competition/runs/s5_pin_ownership_v1/0227/ownership_overlay.png>) |
| 0236 | 22/23 | 79.07% | [查看](<C:/Users/Pc/Desktop/新建文件夹/pcb_parser_competition/runs/s5_pin_ownership_v1/0236/ownership_overlay.png>) |
| 0240 | 21/23 | 77.27% | [查看](<C:/Users/Pc/Desktop/新建文件夹/pcb_parser_competition/runs/s5_pin_ownership_v1/0240/ownership_overlay.png>) |
| 0264 | 21/25 | 73.91% | [查看](<C:/Users/Pc/Desktop/新建文件夹/pcb_parser_competition/runs/s5_pin_ownership_v1/0264/ownership_overlay.png>) |
| 0267 | 20/21 | 61.54% | [查看](<C:/Users/Pc/Desktop/新建文件夹/pcb_parser_competition/runs/s5_pin_ownership_v1/0267/ownership_overlay.png>) |
| 0286 | 20/35 | 13.79% | [查看](<C:/Users/Pc/Desktop/新建文件夹/pcb_parser_competition/runs/s5_pin_ownership_v1/0286/ownership_overlay.png>) |
| 0287 | 9/13 | 76.19% | [查看](<C:/Users/Pc/Desktop/新建文件夹/pcb_parser_competition/runs/s5_pin_ownership_v1/0287/ownership_overlay.png>) |
