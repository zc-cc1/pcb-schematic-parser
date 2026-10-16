# 综合项目建议结构

```text
project/
├─ main.hdev                 # 批量运行入口
├─ procedures/               # 初始化、定位、检测、输出、释放
├─ config/                   # 可版本化参数
├─ models/                   # 模型文件与来源说明
├─ data/
│  ├─ development/
│  ├─ tuning/
│  └─ test/
├─ output/
│  ├─ logs/
│  ├─ failed/
│  └─ visualizations/
└─ report/                   # 流程图、参数表、测试报告
```

本目录只提供学习结构；创建自己的数据子目录时保持原图只读，不把输出写回输入目录。

## 推荐 Procedure

- `initialize_system`：读取配置、模型和标定参数。
- `check_image_quality`：检查尺寸、过暗、过亮和清晰度。
- `locate_part`：模板定位并返回状态、姿态和分数。
- `transform_inspection_rois`：根据位姿变换 ROI。
- `inspect_dimension`：输出尺寸与公差状态。
- `inspect_defect`：输出缺陷 Region、特征和等级。
- `inspect_presence` 或 `recognize_text`：第二项检测功能。
- `combine_results`：融合子结果，保留原因码。
- `save_result`：日志、可视化和失败样本。
- `dispose_system`：释放所有句柄。

