# 第 1 周算子卡片

## read_image

- 作用：从文件读取图像。
- 主要输出：`Image`，图标变量。
- 主要输入：文件名或路径。
- 练习：分别读取 `fabrik`、`clip`、`monkey`。
- 常见问题：文件不存在、格式不支持、路径写错。

## get_image_size

- 作用：获得图像宽度与高度。
- 图标输入：`Image`。
- 控制输出：`Width`、`Height`。
- 注意：宽度对应列方向，高度对应行方向。

## dev_open_window

- 作用：打开 HDevelop 图形窗口。
- 重要参数：窗口左上角位置、窗口宽高、背景颜色。
- 控制输出：`WindowHandle`。
- 注意：窗口尺寸不等于原图分辨率。

## dev_display

- 作用：显示图像、区域或 XLD。
- 输入对象不同，显示含义也不同。
- 注意：显示不会改变对象本身。

## dev_set_part

- 作用：设置图形窗口中显示的图像坐标范围。
- 常用写法：`dev_set_part (0, 0, Height - 1, Width - 1)`。
- 注意：HALCON 图像坐标通常写作行、列，即 `Row, Column`。

## dev_set_color

- 作用：设置后续区域、轮廓或文字的显示颜色。
- 示例颜色：`'red'`、`'green'`、`'yellow'`、`'white'`。
- 注意：不修改图像像素。

## dev_set_draw

- 作用：设置区域显示方式。
- `'margin'`：只显示边界。
- `'fill'`：填充显示。

## dev_set_line_width

- 作用：设置轮廓或区域边界的显示线宽。
- 注意：只影响显示。

## disp_text

- 作用：在窗口或图像坐标位置显示文字。
- 重要参数：窗口句柄、内容、坐标系、行、列、颜色。
- 本周统一使用窗口坐标模式 `'window'`。

## gen_rectangle1

- 作用：根据左上角和右下角坐标生成轴对齐矩形区域。
- 输出：Region 图标变量。
- 坐标顺序：`Row1, Column1, Row2, Column2`。

## gen_circle

- 作用：根据圆心和半径生成圆形区域。
- 坐标顺序：`Row, Column, Radius`。

## reduce_domain

- 作用：将图像有效定义域限制到指定区域。
- 输入：原图与 Region。
- 输出：新的图像对象。
- 注意：图像矩阵尺寸通常没有改变，只改变有效区域。

## tuple_length

- 作用：获得控制元组的元素数量。
- HDevelop 也可使用表达式 `|Tuple|`。

## list_image_files

- 作用：列出目录中可读取的图像文件。
- 输出：文件路径元组。
- 常用扩展名参数：`'default'`。

## dev_get_exception_data

- 作用：读取捕获到的 HDevelop 异常信息。
- 常用数据项：`'error_code'`、`'error_msg'`。
