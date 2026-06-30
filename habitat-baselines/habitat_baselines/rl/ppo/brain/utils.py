# -*- coding: utf-8 -*-
# ==============================================================================
# 文件: utils.py
# 描述: Brain模块工具函数
# ==============================================================================

"""
Brain模块工具函数
=================

提供行人检测、Brain决策相关的辅助函数。
"""

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch


def compute_pedestrian_distance(
    bbox: List[float],
    image_height: int,
    image_width: int,
    camera_height: float = 1.6,
    camera_fov: float = 90.0,
    avg_person_height: float = 1.7,
) -> float:
    """
    根据边界框估算行人与机器人的距离

    基于针孔相机模型，利用行人边界框高度估算距离。

    Args:
        bbox: 边界框坐标 [x1, y1, x2, y2]
        image_height: 图像高度(像素)
        image_width: 图像宽度(像素)
        camera_height: 相机高度(米)，默认1.6m(典型移动机器人高度)
        camera_fov: 相机垂直视场角(度)，默认90度
        avg_person_height: 假设的行人平均身高(米)，默认1.7m

    Returns:
        估算的距离(米)

    Note:
        这是一个近似估算，实际距离受多种因素影响：
        - 行人实际身高差异
        - 地平面不平整
        - 相机内参标定误差
    """
    _, y1, _, y2 = bbox
    bbox_height_pixels = y2 - y1

    if bbox_height_pixels <= 0:
        return float('inf')

    # 焦距(像素) = 图像高度 / (2 * tan(FOV/2))
    focal_length = image_height / (2 * np.tan(np.radians(camera_fov / 2)))

    # 距离 = (实际高度 * 焦距) / 像素高度
    distance = (avg_person_height * focal_length) / bbox_height_pixels

    # 考虑相机高度的影响(简单的三角校正)
    # 当行人很远时，边界框底边通常接近地平线
    # 这里的校正是经验性的
    if distance > 10:
        distance = distance * (camera_height / avg_person_height)

    return max(0.5, min(distance, 50.0))  # 限制在0.5-50米范围


def analyze_pedestrian_motion(
    current_bbox: List[float],
    previous_bbox: Optional[List[float]],
    current_distance: float,
    previous_distance: Optional[float],
    time_delta: float = 0.25,
) -> Dict[str, Any]:
    """
    分析行人的运动趋势

    Args:
        current_bbox: 当前边界框
        previous_bbox: 上一帧边界框(如果有)
        current_distance: 当前估算距离
        previous_distance: 上一帧估算距离(如果有)
        time_delta: 时间间隔(秒)

    Returns:
        运动分析结果字典
    """
    result = {
        "motion": "unknown",  # approaching, retreating, stationary, lateral
        "velocity": 0.0,       # 相对速度(m/s)
        "lateral_direction": None,  # left, right, center
        "threat_level": "low",      # low, medium, high
    }

    if previous_distance is not None and time_delta > 0:
        distance_delta = previous_distance - current_distance
        velocity = distance_delta / time_delta
        result["velocity"] = velocity

        if velocity > 0.3:
            result["motion"] = "approaching"
            result["threat_level"] = "high" if current_distance < 3 else "medium"
        elif velocity < -0.3:
            result["motion"] = "retreating"
            result["threat_level"] = "low"
        else:
            result["motion"] = "stationary"

    if current_bbox is not None:
        cx = (current_bbox[0] + current_bbox[2]) / 2
        # 假设图像宽度归一化到0-1
        if cx < 0.35:
            result["lateral_direction"] = "left"
        elif cx > 0.65:
            result["lateral_direction"] = "right"
        else:
            result["lateral_direction"] = "center"

    return result


def calculate_collision_risk(
    pedestrian_distance: float,
    pedestrian_velocity: float,
    robot_velocity: float,
    pedestrian_lateral_dir: str,
    pedestrian_bbox_area: float,
    safe_distance: float = 2.0,
    warning_distance: float = 5.0,
) -> Tuple[str, float]:
    """
    计算碰撞风险等级

    Args:
        pedestrian_distance: 行人距离(米)
        pedestrian_velocity: 行人接近速度(m/s)
        robot_velocity: 机器人移动速度(m/s)
        pedestrian_lateral_dir: 行人相对位置 (left/center/right)
        pedestrian_bbox_area: 边界框相对面积
        safe_distance: 安全距离阈值(米)
        warning_distance: 警告距离阈值(米)

    Returns:
        (风险等级, 风险分数(0-1))
    """
    # 基础风险分数
    if pedestrian_distance < safe_distance:
        base_risk = 0.9
    elif pedestrian_distance < warning_distance:
        base_risk = 0.5 + 0.4 * (warning_distance - pedestrian_distance) / (warning_distance - safe_distance)
    else:
        base_risk = 0.2

    # 速度调整
    relative_velocity = robot_velocity + pedestrian_velocity
    if pedestrian_velocity > 0:  # 正在接近
        velocity_factor = min(1.0, relative_velocity / 3.0)
        base_risk = min(1.0, base_risk + velocity_factor * 0.2)

    # 位置调整
    if pedestrian_lateral_dir == "center":
        base_risk = min(1.0, base_risk + 0.2)
    elif pedestrian_lateral_dir == "left":
        base_risk = min(1.0, base_risk + 0.05)

    # 面积调整(面积越大说明越近)
    if pedestrian_bbox_area > 0.15:
        base_risk = min(1.0, base_risk + 0.1)

    # 风险等级
    if base_risk >= 0.8:
        risk_level = "high"
    elif base_risk >= 0.5:
        risk_level = "medium"
    else:
        risk_level = "low"

    return risk_level, min(1.0, base_risk)


def format_pedestrian_status(
    pedestrian_info: Dict[str, Any],
    include_details: bool = True,
) -> str:
    """
    格式化行人状态为可读字符串

    Args:
        pedestrian_info: 行人检测信息字典
        include_details: 是否包含详细信息

    Returns:
        格式化的状态字符串
    """
    if not pedestrian_info.get("pedestrian_detected", False):
        return "未检测到行人 ✓"

    count = pedestrian_info.get("pedestrian_count", 0)
    warning = pedestrian_info.get("warning_level", "safe")

    if include_details:
        details = pedestrian_info.get("pedestrian_info", "")
        warning_emoji = "🔴" if warning == "danger" else ("🟡" if warning == "caution" else "🟢")
        return f"{warning_emoji} 检测到{count}个行人 [{warning.upper()}]\n{details}"
    else:
        return f"检测到{count}个行人 [{warning}]"


def batch_filter_detections(
    detections: List[Dict[str, Any]],
    min_confidence: float = 0.3,
    max_count: int = 5,
    min_area: float = 0.001,
) -> List[Dict[str, Any]]:
    """
    批量过滤和排序检测结果

    Args:
        detections: 检测结果列表
        min_confidence: 最小置信度
        max_count: 最大保留数量
        min_area: 最小边界框面积

    Returns:
        过滤后的检测列表
    """
    if not detections:
        return []

    # 过滤
    filtered = [
        d for d in detections
        if d.get("confidence", 0) >= min_confidence
        and d.get("relative_area", 0) >= min_area
    ]

    # 按面积排序(大的优先)
    filtered.sort(key=lambda x: x.get("relative_area", 0), reverse=True)

    # 限制数量
    return filtered[:max_count]


def compute_safe_action_score(
    action: str,
    pedestrian_info: Dict[str, Any],
    current_step: int,
    max_steps: int,
) -> float:
    """
    计算给定动作的安全分数

    Args:
        action: 动作名称
        pedestrian_info: 行人信息
        current_step: 当前步数
        max_steps: 最大步数

    Returns:
        安全分数 (0-1, 越高越安全)
    """
    if not pedestrian_info.get("pedestrian_detected", False):
        return 0.8  # 无行人时基础分

    warning_level = pedestrian_info.get("warning_level", "safe")
    pedestrian_count = pedestrian_info.get("pedestrian_count", 0)

    base_score = 0.5

    # 根据警告等级调整
    if warning_level == "danger":
        base_score = 0.2
        if action in ["STOP", "PAUSE"]:
            base_score = 0.9
        elif action == "TURN_LEFT" or action == "TURN_RIGHT":
            base_score = 0.6
    elif warning_level == "caution":
        base_score = 0.5
        if action in ["STOP", "PAUSE"]:
            base_score = 0.8
        elif action == "FORWARD":
            base_score = 0.4
    else:
        base_score = 0.8
        if action == "FORWARD":
            base_score = 0.9

    # 时间压力调整
    time_ratio = current_step / max_steps if max_steps > 0 else 0
    if time_ratio > 0.9:  # 接近最大步数
        base_score *= 0.8  # 降低安全分数的权重

    return min(1.0, max(0.0, base_score))


class BrainStats:
    """
    Brain模块统计收集器
    ====================

    用于收集和记录Brain模块的运行时统计信息。
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """重置所有统计"""
        self.num_inferences = 0
        self.num_pedestrian_detections = 0
        self.num_override_policy = 0
        self.total_pedestrian_latency_ms = 0.0
        self.total_brain_latency_ms = 0.0
        self.warning_counts = {"safe": 0, "caution": 0, "danger": 0}
        self.action_counts = {}

    def record_inference(
        self,
        pedestrian_latency_ms: float,
        brain_latency_ms: Optional[float] = None,
        pedestrian_detected: bool = False,
        override_policy: bool = False,
        warning_level: str = "safe",
        action_taken: Optional[str] = None,
    ):
        """记录一次推理"""
        self.num_inferences += 1
        self.total_pedestrian_latency_ms += pedestrian_latency_ms

        if pedestrian_detected:
            self.num_pedestrian_detections += 1

        if brain_latency_ms is not None:
            self.total_brain_latency_ms += brain_latency_ms

        if override_policy:
            self.num_override_policy += 1

        if warning_level in self.warning_counts:
            self.warning_counts[warning_level] += 1

        if action_taken and action_taken in self.action_counts:
            self.action_counts[action_taken] += 1
        elif action_taken:
            self.action_counts[action_taken] = 1

    def get_summary(self) -> Dict[str, Any]:
        """获取统计摘要"""
        avg_ped_latency = (
            self.total_pedestrian_latency_ms / self.num_inferences
            if self.num_inferences > 0 else 0
        )
        avg_brain_latency = (
            self.total_brain_latency_ms / self.num_inferences
            if self.num_inferences > 0 else 0
        )

        return {
            "total_inferences": self.num_inferences,
            "pedestrian_detection_rate": (
                self.num_pedestrian_detections / self.num_inferences
                if self.num_inferences > 0 else 0
            ),
            "override_policy_rate": (
                self.num_override_policy / self.num_inferences
                if self.num_inferences > 0 else 0
            ),
            "avg_pedestrian_latency_ms": avg_ped_latency,
            "avg_brain_latency_ms": avg_brain_latency,
            "warning_distribution": self.warning_counts,
            "action_distribution": self.action_counts,
        }

    def print_summary(self):
        """打印统计摘要"""
        summary = self.get_summary()
        print("\n========== Brain模块统计 ==========")
        print(f"总推理次数: {summary['total_inferences']}")
        print(f"行人检测率: {summary['pedestrian_detection_rate']*100:.1f}%")
        print(f"策略覆盖次数: {summary['override_policy_rate']*100:.1f}%")
        print(f"平均行人检测延迟: {summary['avg_pedestrian_latency_ms']:.2f}ms")
        print(f"平均Brain推理延迟: {summary['avg_brain_latency_ms']:.2f}ms")
        print(f"警告分布: {summary['warning_distribution']}")
        print(f"动作分布: {summary['action_distribution']}")
        print("=" * 40)
