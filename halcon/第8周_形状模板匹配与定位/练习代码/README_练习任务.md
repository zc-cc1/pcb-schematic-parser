# 第 8 周练习任务

1. `01_create_shape_model.hdev`：模板 ROI、模型创建和轮廓显示。
2. `02_find_shape_model.hdev`：0/1/多实例结果处理。
3. `03_follow_roi_with_pose.hdev`：模型原点、刚性变换和 ROI 跟随。
4. `04_parameter_scan.hdev`：角度、分数、层级、重叠率、贪婪度单变量扫描。
5. `05_matching_stress_test.hdev`：六类共 60 张带真值压力测试。
6. `06_week_project_part_localization.hdev`：批量定位与检测 ROI 变换。

模型文件、参数和模板图片必须一同版本化记录；不要只保存一个无法追溯来源的模型文件。

## 已完成验收

- 以上 6 个程序均已用 HALCON 20.11.1.0 的 `hrun` 严格模式执行通过。
- 第 5 天实测：60/60 正确、误匹配 0、漏检 0。
- 第 6 天实测：12 个样例覆盖正常、无匹配、低分、多匹配和 ROI 越界。
- 程序采用固定随机种子，便于重复实验和比较参数变化。
