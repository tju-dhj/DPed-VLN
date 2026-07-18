#!/usr/bin/env python3
"""
NaVid Action Parser
基于 agent_navid.py:extract_result() 实现
支持单步动作解析和多步动作序列解析
"""

import re
from typing import List, Optional, Tuple


class NaVidActionParser:
    """
    NaVid 动作解析器

    NaVid 输出自然语言动作词:
    - "stop" → action 0
    - "forward X" → action 1 (X = 距离 cm)
    - "left X" → action 2 (X = 角度)
    - "right X" → action 3 (X = 角度)
    """

    def __init__(self, forward_step: int = 25, turn_step: int = 15):
        """
        Args:
            forward_step: 前进一步的距离 (cm)，默认 25cm
            turn_step: 转弯一步的角度 (deg)，默认 15°
        """
        self.forward_step = forward_step
        self.turn_step = turn_step
        self.action_map = {
            "stop": 0,
            "forward": 1,
            "left": 2,
            "right": 3,
        }

    def parse_action(self, text: str) -> Tuple[int, int]:
        """
        解析单步动作文本

        Args:
            text: 模型输出的动作文本，例如 "forward 25" 或 "stop"

        Returns:
            (action_id, num_repeats): 动作 ID 和重复次数
        """
        text = text.strip().lower()

        if "stop" in text:
            return 0, 1

        elif "forward" in text:
            match = re.search(r'-?\d+', text)
            if match is None:
                return 1, 1  # 默认前进 1 步
            distance = float(match.group())
            num_repeats = min(3, max(1, int(distance / self.forward_step)))
            return 1, num_repeats

        elif "left" in text:
            match = re.search(r'-?\d+', text)
            if match is None:
                return 2, 1  # 默认左转 1 步
            angle = float(match.group())
            num_repeats = min(3, max(1, int(angle / self.turn_step)))
            return 2, num_repeats

        elif "right" in text:
            match = re.search(r'-?\d+', text)
            if match is None:
                return 3, 1  # 默认右转 1 步
            angle = float(match.group())
            num_repeats = min(3, max(1, int(angle / self.turn_step)))
            return 3, num_repeats

        # 无法解析，默认 stop
        return 0, 1

    def parse_action_sequence(self, text: str) -> List[Tuple[int, int]]:
        """
        解析多步动作序列

        支持格式:
        - 空格分隔: "forward forward left stop"
        - 分号分隔: "forward; forward; left; stop"
        - 编号列表: "1. forward 2. left 3. stop"

        Args:
            text: 模型输出的多步动作文本

        Returns:
            [(action_id, num_repeats), ...]: 动作序列列表
        """
        text = text.strip()

        # 尝试分号分隔
        if ";" in text:
            parts = [p.strip() for p in text.split(";") if p.strip()]
        # 尝试空格分隔（Uni-NaVid 格式）
        else:
            # 先清理编号
            text = re.sub(r'\d+\.\s*', '', text)
            # 按空格分割
            parts = text.split()

        results = []
        for part in parts:
            part = part.strip().lower()
            if not part:
                continue
            action_id, num = self.parse_action(part)
            results.append((action_id, num))

        return results if results else [(0, 1)]


# 动作 ID 到名称的映射
ACTION_ID_TO_NAME = {
    0: "stop",
    1: "move forward",
    2: "turn left",
    3: "turn right",
}

# 动作名称到 ID 的映射
ACTION_NAME_TO_ID = {
    "stop": 0,
    "move forward": 1,
    "turn left": 2,
    "turn right": 3,
    "forward": 1,
    "left": 2,
    "right": 3,
}
