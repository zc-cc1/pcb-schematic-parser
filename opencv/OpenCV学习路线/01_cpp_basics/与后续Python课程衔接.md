# 与第2周Python课程的衔接

本次只将第1周换成C++。第2～16周示例仍是Python；图像概念通用，但语法与内存行为需要按语言理解。没有Python基础时，请先阅读[旧版第一周讲义](../历史版本/01_python_numpy/讲义.md)，运行其中示例后再继续第2周。

| 含义 | 本周C++ | 后续Python |
| --- | --- | --- |
| 图像容器 | cv::Mat | NumPy数组 |
| 高、宽 | rows、cols | shape[:2] |
| 像素 | at<cv::Vec3b>(y,x) | image[y,x] |
| ROI | image(cv::Rect(x,y,w,h)) | image[y:y+h,x:x+w] |
| 独立复制 | clone() | copy() |
| 打印 | std::cout | print |
| 函数 | 返回类型与参数类型 | def与缩进 |
| 运行 | 编译后运行exe | python脚本 |

特别注意：C++的uint8_t算术先进行整数提升；NumPy的uint8数组加法可以直接在8位结果里回绕。不要把两种语言的中间计算行为混为一谈。

若学习目标是全程C++，需要将后续周的示例和讲义也改写后再按完整C++路线推进；目前不将剩余Python代码称为C++课程。
