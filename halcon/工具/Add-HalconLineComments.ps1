param(
    [string]$CourseRoot = (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path))
)

$operatorComments = @{
    'abs_diff_image' = '计算两幅图像对应像素的绝对灰度差，用于突出变化或缺陷。'
    'add_noise_white' = '向图像加入白噪声，用来模拟噪声干扰并进行压力测试。'
    'affine_trans_contour_xld' = '使用仿射矩阵变换XLD轮廓，使轮廓跟随目标位姿。'
    'affine_trans_image' = '使用仿射矩阵变换整幅图像，得到旋转或平移后的图像。'
    'affine_trans_point_2d' = '使用二维仿射矩阵变换点坐标，计算点的新位置。'
    'affine_trans_region' = '使用仿射矩阵变换区域，使检测ROI跟随目标移动。'
    'angle_ll' = '计算两条直线之间的夹角。'
    'area_center' = '计算区域面积及其重心行列坐标，并写入输出变量。'
    'boundary' = '提取区域边界，得到只表示边缘的区域对象。'
    'calibrate_cameras' = '根据已经采集的标定观测执行相机标定，并返回重投影误差。'
    'circularity' = '计算区域圆度，数值越接近1通常越接近圆形。'
    'clear_bar_code_model' = '释放一维码模型句柄，避免句柄资源泄漏。'
    'clear_calib_data' = '释放标定数据模型句柄及其内部资源。'
    'clear_data_code_2d_model' = '释放二维码模型句柄，结束后不再占用模型资源。'
    'clear_ocr_class_mlp' = '释放OCR的MLP分类器句柄。'
    'clear_shape_model' = '释放形状模板句柄，避免模型资源泄漏。'
    'clip_region' = '把区域裁剪到指定矩形范围，去除ROI以外的部分。'
    'close_file' = '关闭已经打开的文件句柄，确保数据写入并释放文件资源。'
    'close_measure' = '释放一维测量句柄，结束测量对象的生命周期。'
    'closing_circle' = '使用圆形结构元素做闭运算，以连接小间隙并填补小孔。'
    'concat_obj' = '把当前图形对象追加到对象数组中。'
    'connection' = '把一个区域按连通性拆分成多个互不相连的区域。'
    'copy_obj' = '复制指定图形对象，得到独立的输出对象引用。'
    'count_channels' = '统计图像通道数量，用于判断灰度图或彩色图。'
    'count_obj' = '统计图形对象数组中的对象数量。'
    'count_seconds' = '读取当前高精度时间，用于计算程序运行耗时。'
    'create_bar_code_model' = '创建一维码识别模型并返回模型句柄。'
    'create_calib_data' = '创建相机标定数据模型，用于保存相机、标定板和观测结果。'
    'create_data_code_2d_model' = '创建指定类型的二维码识别模型并返回句柄。'
    'create_pose' = '根据平移、旋转和姿态约定创建三维位姿。'
    'create_shape_model' = '根据模板图像创建形状匹配模型并返回模型句柄。'
    'dev_clear_window' = '清空当前活动图形窗口中的显示内容。'
    'dev_close_window' = '关闭当前HDevelop图形窗口，防止重复运行时窗口不断累积。'
    'dev_display' = '把当前图像、区域或XLD对象显示到活动图形窗口。'
    'dev_get_exception_data' = '从捕获的异常中读取指定的错误信息字段。'
    'dev_open_window' = '按照给定位置和尺寸打开HDevelop图形窗口，并取得窗口句柄。'
    'dev_set_color' = '设置后续图形对象的显示颜色。'
    'dev_set_colored' = '设置多对象循环配色方式，便于区分不同区域或轮廓。'
    'dev_set_draw' = '设置区域按填充或轮廓方式显示。'
    'dev_set_line_width' = '设置后续轮廓和区域边界的显示线宽。'
    'dev_set_part' = '设置图像坐标与窗口显示区域的对应范围。'
    'dev_update_off' = '关闭HDevelop自动更新，避免逐条执行时产生不必要的刷新。'
    'difference' = '计算两个区域的差集，保留第一个区域中不属于第二个区域的部分。'
    'dilation_circle' = '使用圆形结构元素膨胀区域，使区域扩大并连接邻近部分。'
    'disp_text' = '在指定窗口位置显示说明文字、参数或检测结果。'
    'distance_pl' = '计算点到直线的最短距离。'
    'distance_pp' = '计算两个点之间的欧氏距离。'
    'do_ocr_multi_class_mlp' = '使用预训练MLP分类器识别每个字符区域，并返回字符和置信度。'
    'edges_sub_pix' = '以亚像素精度提取图像边缘，输出XLD轮廓。'
    'elliptic_axis' = '计算等效椭圆的主轴、次轴和方向，用于描述区域形状。'
    'erosion_circle' = '使用圆形结构元素腐蚀区域，使区域收缩并去除细小突出。'
    'fill_up' = '填充区域内部完全封闭的孔洞。'
    'find_bar_code' = '在图像中查找指定码制的一维码，并返回区域和解码字符串。'
    'find_calib_object' = '在当前图像中检测标定板，并把观测结果写入标定数据模型。'
    'find_data_code_2d' = '在图像中查找二维码，返回符号轮廓、结果句柄和解码内容。'
    'find_shape_model' = '在搜索图像中查找形状模型，返回位置、角度和匹配分数。'
    'fit_circle_contour_xld' = '用XLD轮廓点拟合圆，并输出圆心、半径及轮廓端点。'
    'fit_line_contour_xld' = '用XLD轮廓点拟合直线，并输出直线端点和方向。'
    'fnew_line' = '向文本文件写入换行符，开始下一条日志记录。'
    'fwrite_string' = '把指定字符串写入已经打开的文本文件。'
    'gauss_filter' = '使用高斯滤波平滑图像，在抑制噪声的同时尽量保留结构。'
    'gen_arrow_contour_xld' = '生成箭头XLD轮廓，用于可视化方向。'
    'gen_cam_par_area_scan_division' = '生成面阵相机初始参数，并采用division畸变模型。'
    'gen_circle' = '根据圆心和半径生成圆形区域。'
    'gen_circle_contour_xld' = '根据圆心、半径和角度范围生成圆弧XLD轮廓。'
    'gen_contour_polygon_xld' = '根据一组行列坐标生成折线XLD轮廓。'
    'gen_contour_region_xld' = '把区域边界转换为XLD轮廓。'
    'gen_cross_contour_xld' = '在指定坐标生成十字XLD，用于标记位置或特征点。'
    'gen_ellipse' = '根据中心、方向和半轴长度生成椭圆区域。'
    'gen_empty_obj' = '创建空的图形对象，作为无结果状态或对象收集器的初值。'
    'gen_image_const' = '创建指定尺寸和类型的常量图像，作为合成图像背景。'
    'gen_measure_rectangle2' = '创建矩形一维测量对象，并返回后续测量使用的句柄。'
    'gen_rectangle1' = '根据左上角和右下角坐标生成轴对齐矩形区域。'
    'gen_rectangle2' = '根据中心、方向和两个半边长生成可旋转矩形区域。'
    'get_calib_data' = '从标定数据模型读取相机参数、位姿或其他标定结果。'
    'get_calib_data_observ_contours' = '读取指定标定图中的标定板或标志点轮廓。'
    'get_calib_data_observ_points' = '读取标定图中检测到的图像点、索引和标定板位姿。'
    'get_contour_global_attrib_xld' = '读取XLD轮廓的全局属性。'
    'get_image_size' = '读取图像宽度和高度，供窗口、ROI或边界计算使用。'
    'get_shape_model_contours' = '提取形状模板内部保存的模型轮廓，用于可视化匹配结果。'
    'hom_mat2d_identity' = '创建二维单位变换矩阵，作为后续旋转和平移的起点。'
    'hom_mat2d_invert' = '求二维变换矩阵的逆矩阵，用于撤销已有位姿变换。'
    'hom_mat2d_rotate' = '在当前二维矩阵上加入绕指定中心的旋转变换。'
    'hom_mat2d_translate' = '在当前二维矩阵上加入行列方向的平移变换。'
    'image_points_to_world_plane' = '把图像行列坐标投影到指定世界平面，并按给定单位输出X/Y坐标。'
    'image_to_world_plane' = '依据相机参数和位姿把整幅图像校正到世界平面。'
    'inspect_bright_parts' = '调用本程序的亮目标检测过程，并接收区域、状态和面积结果。'
    'inspect_image' = '调用本程序的工程化检测过程，并接收区域、状态码和特征结果。'
    'intersection' = '计算两个区域的交集，只保留二者共同覆盖的部分。'
    'intersection_lines' = '计算两条无限直线的交点，并判断两线是否重合。'
    'mean_image' = '使用邻域均值滤波图像，以降低局部噪声和灰度波动。'
    'measure_pairs' = '沿测量矩形搜索成对边缘，并返回边缘位置、灰度幅度和宽度。'
    'measure_pos' = '沿测量矩形搜索满足极性的单边缘，并返回位置和边缘幅度。'
    'open_file' = '按指定模式打开文本文件，并返回文件句柄。'
    'opening_circle' = '使用圆形结构元素做开运算，以去除小区域和细小连接。'
    'opening_rectangle1' = '使用矩形结构元素做开运算，按指定方向分离或去除结构。'
    'orientation_region' = '计算区域主方向角，用于姿态归一化。'
    'paint_region' = '用指定灰度值把区域绘制到图像中，生成合成目标或缺陷。'
    'partition_rectangle' = '按照给定字符宽高把矩形区域切分成多个候选区域。'
    'read_image' = '读取指定图像文件或HALCON示例图像，并保存到输出图像变量。'
    'read_ocr_class_mlp' = '读取预训练OCR MLP分类器并返回分类器句柄。'
    'reduce_domain' = '把图像定义域限制到ROI；图像矩阵尺寸不变，但后续仅处理域内像素。'
    'region_to_bin' = '把区域转换为二值图像，并为区域内外分别设置灰度值。'
    'rgb1_to_gray' = '按加权方式把三通道彩色图转换为单通道灰度图。'
    'segment_contours_xld' = '把XLD轮廓按直线或圆弧结构分段。'
    'select_contours_xld' = '根据长度等轮廓特征筛选XLD轮廓。'
    'select_obj' = '从对象数组中取出指定序号的单个图形对象。'
    'select_shape' = '按照面积、宽高或其他形状特征筛选连通区域。'
    'select_shape_std' = '使用标准形状条件筛选区域，例如保留最大区域。'
    'set_calib_data_calib_object' = '向标定数据模型设置标定板描述文件。'
    'set_calib_data_cam_param' = '向标定数据模型设置相机初始参数。'
    'set_origin_pose' = '平移位姿原点，使世界平面校正使用新的参考原点。'
    'shape_trans' = '把区域转换成指定标准外形，例如最小外接矩形。'
    'sort_region' = '按字符阅读顺序对区域排序，保证OCR输出顺序正确。'
    'text_line_orientation' = '估计文本行方向，供后续旋转校正使用。'
    'threshold' = '按灰度上下限对图像做阈值分割，生成满足条件的区域。'
    'tuple_regexp_test' = '使用正则表达式检查字符串格式是否满足业务规则。'
    'tuple_sort' = '对数值元组升序排序，供中位数或分位耗时计算使用。'
    'union1' = '把对象数组中的所有区域合并为一个区域。'
    'union2' = '合并两个区域，得到它们的并集。'
    'vector_angle_to_rigid' = '根据参考位姿和当前位姿建立二维刚性变换矩阵。'
}

function Get-VariableMeaning([string]$name) {
    switch -Regex ($name) {
        '^(TP|TN|FP|FN)$' { return '混淆矩阵计数' }
        'Status(Code|Text|Log)?$' { return '检测状态' }
        '(Time|Times|T0|T1|P95)' { return '运行耗时' }
        '(Count|Counts|Success|Failures|Index|Indices|^I$|^K$|^L$)' { return '计数或索引' }
        '(Score|Scores|Confidence)' { return '置信度或匹配分数' }
        '(Area|Areas)' { return '面积特征' }
        '(Row|Rows|Column|Columns|Col)' { return '行列坐标' }
        '(Angle|Phi|Orientation)' { return '方向角' }
        '(Width|Height|Length|Radius|Distance|Ra|Rb)' { return '几何尺寸' }
        '(HomMat|Move|Transform|AlignMat|Inverse)' { return '二维变换矩阵' }
        '(CamPar|CamParam|Pose)' { return '相机参数或位姿' }
        '(ModelID|Handle|CalibDataID)' { return '资源句柄' }
        '(Threshold|Min|Max|Lower|Upper|Sigma|Polarity|Transition)' { return '算法参数' }
        '(String|Text|Message|Class|Grade|Name|File|Project)' { return '文本或标识信息' }
        '(OK|NG|Valid|Created|Truth|Predicted)' { return '布尔判定结果' }
        '(Image)' { return '图像数据' }
        '(Region|ROI|Contour|XLD|Object|Defect|Part)' { return '图形对象或检测结果' }
        default { return '控制变量' }
    }
}

function Get-AssignmentComment([string]$line) {
    if ($line -notmatch '^\s*([A-Za-z_][A-Za-z0-9_]*)\s*:=\s*(.*)$') {
        return $null
    }
    $name = $matches[1]
    $rhs = $matches[2].Trim()
    $meaning = Get-VariableMeaning $name

    if ($rhs -eq '[]') { return "把${meaning}变量 $name 初始化为空元组，供后续收集结果。" }
    if ($rhs -match ('^\[' + [regex]::Escape($name) + '\s*,')) { return "把本次结果追加到${meaning}变量 $name 中。" }
    if ($rhs -match ('^' + [regex]::Escape($name) + '\s*\+\s*1$')) { return "把${meaning}变量 $name 加1，记录一次新的事件。" }
    if ($rhs -match '^(true|false)$') { return "设置${meaning}变量 $name 的初始真假状态。" }
    if ($rhs -match "^'.*'$" ) { return "设置${meaning}变量 $name 的字符串值。" }
    if ($rhs -match '^[+-]?\d+(\.\d+)?$') { return "设置${meaning}变量 $name 的数值初值或阈值。" }
    if ($rhs -match '^\[') { return "设置${meaning}变量 $name 的元组数据，供后续批量处理。" }
    if ($rhs -match '(==|!=|&gt;=|&lt;=|>=|<=|\band\b|\bor\b|\bnot\b)') { return "计算判断条件，并把真假结果保存到 $name。" }
    if ($rhs -match '^(mean|max|min|sum|sqrt|abs|rad|deg|floor|max2|min2)\(') { return "根据当前数据计算${meaning}，并把结果保存到 $name。" }
    if ($rhs -match '\[[^\]]+\]') { return "从元组中读取指定元素或组合数据，并保存到 $name。" }
    return "根据当前参数和中间结果计算${meaning}变量 $name。"
}

function Get-LineComment([string]$encodedLine) {
    $line = [System.Net.WebUtility]::HtmlDecode($encodedLine).Trim()

    if ($line -match '^\*') { return '此行已被禁用，仅作为备选代码或提示保留。' }
    if ($line -match '^for\s+') { return '按照给定起点、终点和步长开始循环处理。' }
    if ($line -match '^endfor$') { return '结束当前for循环，继续执行循环后的代码。' }
    if ($line -match '^if\s*\(') { return '判断括号中的条件；条件为真时执行当前分支。' }
    if ($line -match '^elseif\s*\(') { return '前一条件不成立时继续判断这个补充条件。' }
    if ($line -eq 'else') { return '前面的条件均不成立时执行这个分支。' }
    if ($line -eq 'endif') { return '结束当前条件分支。' }
    if ($line -eq 'try') { return '开始受保护代码块，运行错误将交给catch分支处理。' }
    if ($line -match '^catch\s*\(') { return '捕获try代码块抛出的HALCON异常，并保存异常元组。' }
    if ($line -eq 'endtry') { return '结束异常处理代码块。' }
    if ($line -match '^return\s*\(') { return '结束当前局部过程并把输出参数返回给调用方。' }
    if ($line -match '^stop\s*\(') { return '暂停程序，便于在HDevelop中观察当前中间结果。' }

    $assignment = Get-AssignmentComment $line
    if ($assignment) { return $assignment }

    if ($line -match '^([A-Za-z_][A-Za-z0-9_]*)\s*\(') {
        $operator = $matches[1]
        if ($operatorComments.ContainsKey($operator)) { return $operatorComments[$operator] }
        return "调用 $operator 算子或局部过程，完成本行指定的数据处理。"
    }
    return '执行当前HDevelop控制语句，推进本步骤的处理流程。'
}

$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
$files = Get-ChildItem -LiteralPath $CourseRoot -Recurse -Filter '*.hdev' |
    Where-Object { $_.FullName -match '第(?:[4-9]|1[0-2])周' }

foreach ($file in $files) {
    $text = [System.IO.File]::ReadAllText($file.FullName, [System.Text.Encoding]::UTF8)
    $text = [regex]::Replace($text, '<c>\* \[逐行\].*?</c>\s*', '', 'Singleline')
    $text = [regex]::Replace($text, '<l>(.*?)</l>', {
        param($match)
        $comment = Get-LineComment $match.Groups[1].Value
        $escapedComment = [System.Security.SecurityElement]::Escape($comment)
        return '<c>* [逐行] ' + $escapedComment + '</c>' + [Environment]::NewLine + '<l>' + $match.Groups[1].Value + '</l>'
    }, 'Singleline')
    [System.IO.File]::WriteAllText($file.FullName, $text, $utf8NoBom)
}

Write-Output ("Commented HDevelop files: {0}" -f $files.Count)
