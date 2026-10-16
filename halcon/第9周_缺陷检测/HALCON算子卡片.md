# 第 9 周算子卡片

| 算子/流程 | 作用 | 关键点 | 常见问题 |
|---|---|---|---|
| `abs_diff_image` | 两图绝对差分 | 图像需同尺寸、同类型并已对齐 | 轻微错位形成双边假缺陷 |
| `sub_image` | 带比例与偏移的图像相减 | Mult、Add | 灰度范围和类型需核对 |
| `inspect_shape_model` | 检查模板形状层级 | 观察模型边缘与金字塔 | 误把定位模型直接当缺陷模型 |
| `create_variation_model` | 创建灰度变化模型 | 图像尺寸与类型 | 训练样本覆盖不足 |
| `train_variation_model` | 加入正常样本 | 只用合格且对齐图像 | 混入异常样本扩大容差 |
| `prepare_variation_model` | 准备比较阈值 | 绝对/变化阈值 | 阈值过大漏检小缺陷 |
| `compare_ext_variation_model` | 与变化模型比较 | 输出亮/暗变化区域 | 光照漂移导致大面积误报 |
| `clear_variation_model` | 释放变化模型 | 生命周期结束后释放 | 循环中重复创建未释放 |

分割、形态学、`connection`、`select_shape` 和定位变换仍是本周主流程的一部分，需要复用前几周卡片。

