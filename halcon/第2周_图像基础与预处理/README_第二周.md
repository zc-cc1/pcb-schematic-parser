# HALCON 第 2 周：图像基础与预处理

适配环境：HALCON 20.11 Steady

本周目标：理解图像通道、定义域、直方图、对比度和噪声，能够根据问题选择基础预处理方法，而不是机械地堆叠滤波算子。

## 本周安排

| 天数 | 主题 | 主要成果 |
|---|---|---|
| 第 1 天 | 彩色图、灰度图与 RGB 通道 | 分解、观察并重新组合三个通道 |
| 第 2 天 | ROI、定义域与裁剪 | 区分 `reduce_domain`、`crop_domain` 和真实裁剪 |
| 第 3 天 | 直方图与对比度增强 | 根据灰度分布增强低对比度图像 |
| 第 4 天 | 均值、高斯和中值滤波 | 比较三种滤波方法的效果和代价 |
| 第 5 天 | 光照校正与锐化 | 处理亮度不均并增强局部细节 |
| 第 6 天 | 预处理对比项目 | 对多张图像记录效果、参数和耗时 |
| 第 7 天 | 复盘验收 | 独立设计一条合理的预处理流程 |

## 第 1 天：彩色图、灰度图与 RGB 通道

### 核心概念

- 灰度图通常只有一个通道，每个像素保存一个强度值。
- RGB 彩色图通常有三个通道：红色 R、绿色 G、蓝色 B。
- 单独显示 R、G、B 通道时，它们看起来是灰度图；越亮表示对应颜色分量越强。
- `decompose3` 将三通道图像分解成三个单通道图像。
- `compose3` 将三个单通道图像重新组合成彩色图。
- `rgb1_to_gray` 将 RGB 图像转换为单通道灰度图。

### 今天的操作

打开 `练习代码/01_color_gray_channels.hdev`：

1. 完整运行一次；每次遇到 `stop ()` 点击继续。
2. 在图标变量窗口依次双击 `ColorImage`、`ImageR`、`ImageG`、`ImageB`、`GrayImage`。
3. 在控制变量窗口查看 `ChannelCount`、`MeanR`、`MeanG`、`MeanB`。
4. 判断示例图片中哪个通道的平均强度最高。
5. 比较 `ColorImage` 和 `ReconstructedColor`，确认分解后可以重新组合。

### 练习一：交换通道

将：

```text
compose3 (ImageR, ImageG, ImageB, ReconstructedColor)
```

改成：

```text
compose3 (ImageB, ImageG, ImageR, ReconstructedColor)
```

重新运行并观察颜色变化。交换 R 和 B 只改变颜色表达，不改变图像宽高。

### 练习二：使用其他彩色图

把：

```text
read_image (ColorImage, 'brycecanyon1')
```

改为：

```text
read_image (ColorImage, 'patras')
```

重新观察三个通道及其平均值。

### 练习三：验证灰度图通道数

单独读取：

```text
read_image (TestGrayImage, 'fabrik')
count_channels (TestGrayImage, TestChannelCount)
```

预期：`TestChannelCount = 1`。

不要对单通道 `TestGrayImage` 直接调用 `decompose3`，因为它没有三个通道可供分解。

### 第 1 天验收

能够回答：

1. 为什么单独的 R 通道显示出来通常是灰度图？
2. R 通道越亮代表什么？
3. `decompose3` 和 `compose3` 分别做什么？
4. `rgb1_to_gray` 是否等同于随便选取一个通道？
5. 如何在处理前确认图像通道数量？

五题能清楚回答四题，即可进入第 2 天。

## 本周方法要求

每次预处理实验都记录：

- 原图问题是什么。
- 使用了哪个算子。
- 为什么选择这个算子。
- 关键参数是多少。
- 处理后改善了什么。
- 是否损失了边缘或细节。
- 单张处理耗时是多少。

预处理的目标是让后续分割、测量或匹配更稳定，并不是单纯让图片“更好看”。
