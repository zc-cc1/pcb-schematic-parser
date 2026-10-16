# 第 5 周算子卡片

| 算子 | 作用 | 关键点 | 常见问题 |
|---|---|---|---|
| `edges_sub_pix` | 提取亚像素边缘 XLD | 滤波器、Alpha、低/高阈值 | 阈值低导致噪声，平滑大导致偏移 |
| `segment_contours_xld` | 分割轮廓几何片段 | 模式、平滑、阈值 | 过度分割或分割不足 |
| `select_contours_xld` | 按轮廓特征筛选 | 长度、方向等 | 条件过紧丢失有效片段 |
| `union_adjacent_contours_xld` | 连接相邻轮廓 | 距离与属性模式 | 错误连接不同物体边缘 |
| `fit_line_contour_xld` | 鲁棒拟合直线 | 算法、迭代与裁剪参数 | 输入片段不直导致结果漂移 |
| `fit_circle_contour_xld` | 拟合圆/圆弧 | 算法、最大迭代 | 弧长太短时半径不稳定 |
| `gen_contour_polygon_xld` | 由点生成多边形 XLD | 行列坐标元组 | 点序错误导致交叉轮廓 |
| `get_contour_xld` | 获取轮廓点坐标 | Row、Column | 点数多，注意元组处理成本 |

## 记录要求

每次拟合至少记录：输入轮廓长度、点数、拟合参数、残差或跨图波动、耗时和失败图片。

