"""
赛题六工具函数模块

提供辅助功能：
1. 图像预处理
2. 坐标转换
3. 结果后处理
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict


def load_image(image_path: str) -> np.ndarray:
    """
    加载图像

    Args:
        image_path: 图像路径

    Returns:
        numpy数组格式的图像
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    return image


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    预处理图像（可选）

    Args:
        image: 原始图像

    Returns:
        预处理后的图像
    """
    # 示例：转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 示例：自适应二值化
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, 2
    )

    return binary


def validate_prediction(prediction: Dict) -> bool:
    """
    验证预测结果格式

    Args:
        prediction: 预测结果字典

    Returns:
        是否有效
    """
    # 检查必需字段
    required_keys = ["components", "pins", "nets"]
    for key in required_keys:
        if key not in prediction:
            return False

    # 检查类型
    if not isinstance(prediction["components"], dict):
        return False
    if not isinstance(prediction["pins"], dict):
        return False
    if not isinstance(prediction["nets"], dict):
        return False

    return True


def merge_close_components(components: Dict, threshold: float = 10.0) -> Dict:
    """
    合并距离很近的重复元件（可选的后处理）

    Args:
        components: 元件字典
        threshold: 距离阈值（像素）

    Returns:
        合并后的元件字典
    """
    # 简单示例，实际可以使用更复杂的聚类算法
    merged = {}

    for comp_id, comp_data in components.items():
        center = comp_data.get("center", [0, 0])

        # 检查是否与已有元件重复
        is_duplicate = False
        for merged_id, merged_data in merged.items():
            merged_center = merged_data.get("center", [0, 0])
            distance = np.sqrt(
                (center[0] - merged_center[0])**2 +
                (center[1] - merged_center[1])**2
            )

            if distance < threshold:
                # 重复，跳过
                is_duplicate = True
                break

        if not is_duplicate:
            merged[comp_id] = comp_data

    return merged


def calculate_iou(bbox1: List[float], bbox2: List[float]) -> float:
    """
    计算两个边界框的IoU

    Args:
        bbox1: [x1, y1, x2, y2]
        bbox2: [x1, y1, x2, y2]

    Returns:
        IoU值
    """
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2

    # 计算交集
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
        return 0.0

    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)

    # 计算并集
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = area1 + area2 - inter_area

    if union_area == 0:
        return 0.0

    return inter_area / union_area


def format_component_id(comp_type: str, index: int) -> str:
    """
    格式化元件ID

    Args:
        comp_type: 元件类型
        index: 索引

    Returns:
        格式化的ID（如 R1, C2, U3）
    """
    type_prefix = {
        "Resistor": "R",
        "Capacitor": "C",
        "IC": "U",
        "Inductor": "L",
        "Diode": "D",
        "Transistor": "Q",
    }

    prefix = type_prefix.get(comp_type, "X")
    return f"{prefix}{index}"


def simplify_wire_path(points: List[List[float]], tolerance: float = 2.0) -> List[List[float]]:
    """
    简化导线路径（Douglas-Peucker算法）

    Args:
        points: 路径点列表 [[x1,y1], [x2,y2], ...]
        tolerance: 简化容差

    Returns:
        简化后的路径点
    """
    if len(points) <= 2:
        return points

    # 转换为numpy数组
    points_array = np.array(points, dtype=np.float32)

    # 使用OpenCV的approxPolyDP
    epsilon = tolerance
    simplified = cv2.approxPolyDP(points_array, epsilon, closed=False)

    return simplified.reshape(-1, 2).tolist()
