#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Action Parser for StreamVLN
将StreamVLN生成的语言指令解析为Habitat3离散动作
Parses language instructions from StreamVLN into Habitat3 discrete actions
"""

import re
from typing import Optional, Tuple


class StreamVLNActionParser:
    """
    解析StreamVLN生成的语言指令为离散动作
    
    动作映射:
    - 0: STOP
    - 1: MOVE_FORWARD (with distance in cm)
    - 2: TURN_LEFT (with degree)
    - 3: TURN_RIGHT (with degree)
    
    Habitat3 Falcon框架的动作速度:
    - MOVE_FORWARD: 25 cm per step
    - TURN_LEFT: 15 degrees per step
    - TURN_RIGHT: 15 degrees per step
    """
    
    # 基础动作步长
    FORWARD_STEP = 25  # cm per action
    TURN_STEP = 15  # degrees per action
    
    # 正则表达式模式 - 支持多种自然语言表达方式
    PATTERNS = {
        0: re.compile(r"\b(stop|halt|end|finish|complete|arrived|reached)\b", re.IGNORECASE),
        1: re.compile(r"\b(move forward|go forward|advance|proceed forward|walk forward|step forward|move ahead|go ahead)\b", re.IGNORECASE),
        2: re.compile(r"\b(turn left|rotate left|turn to the left|turn towards left|rotate to left|go left|steer left)\b", re.IGNORECASE),
        3: re.compile(r"\b(turn right|rotate right|turn to the right|turn towards right|rotate to right|go right|steer right)\b", re.IGNORECASE),
    }
    
    # 距离和角度提取模式
    DISTANCE_PATTERN = re.compile(r"(?:move forward|go forward|advance|proceed forward|forward)\s*(\d+)\s*(?:cm|centimeter|centimeters|meter|meters|m)?", re.IGNORECASE)
    LEFT_ANGLE_PATTERN = re.compile(r"(?:turn left|rotate left|turn to the left|left)\s*(\d+)\s*(?:degree|degrees|deg|°)?", re.IGNORECASE)
    RIGHT_ANGLE_PATTERN = re.compile(r"(?:turn right|rotate right|turn to the right|right)\s*(\d+)\s*(?:degree|degrees|deg|°)?", re.IGNORECASE)
    
    def __init__(self, forward_step: int = 25, turn_step: int = 15):
        """
        初始化动作解析器
        
        Args:
            forward_step: 每次前进的距离（厘米）
            turn_step: 每次转向的角度（度）
        """
        self.forward_step = forward_step
        self.turn_step = turn_step
    
    def parse_action(self, instruction: str) -> Tuple[int, int]:
        """
        解析语言指令为动作和重复次数
        
        Args:
            instruction: StreamVLN生成的语言指令
            
        Returns:
            tuple: (action_id, num_repeats)
                - action_id: 0=STOP, 1=MOVE_FORWARD, 2=TURN_LEFT, 3=TURN_RIGHT
                - num_repeats: 该动作需要重复的次数
        """
        # 首先识别基础动作类型
        action_id = self._map_string_to_action(instruction)
        
        if action_id is None:
            # 如果无法识别，默认返回MOVE_FORWARD 1次
            return 1, 1
        
        if action_id == 0:  # STOP
            return 0, 1
        
        elif action_id == 1:  # MOVE_FORWARD
            distance = self._extract_distance(instruction)
            num_repeats = max(1, distance // self.forward_step)
            return 1, num_repeats
        
        elif action_id == 2:  # TURN_LEFT
            degree = self._extract_left_angle(instruction)
            num_repeats = max(1, degree // self.turn_step)
            return 2, num_repeats
        
        elif action_id == 3:  # TURN_RIGHT
            degree = self._extract_right_angle(instruction)
            num_repeats = max(1, degree // self.turn_step)
            return 3, num_repeats
        
        return 1, 1  # 默认
    
    def _map_string_to_action(self, s: str) -> Optional[int]:
        """
        将字符串映射到动作ID
        
        Args:
            s: 输入字符串
            
        Returns:
            action_id or None if no match
        """
        # 按优先级检查：先检查STOP，然后检查其他动作
        if self.PATTERNS[0].search(s):
            return 0
        
        # 检查左转（必须在右转之前检查）
        if self.PATTERNS[2].search(s):
            return 2
        
        # 检查右转
        if self.PATTERNS[3].search(s):
            return 3
        
        # 检查前进
        if self.PATTERNS[1].search(s):
            return 1
        
        return None
    
    def _extract_distance(self, instruction: str) -> int:
        """
        从指令中提取移动距离
        
        Args:
            instruction: 语言指令
            
        Returns:
            distance in cm (default forward_step)
        """
        try:
            match = self.DISTANCE_PATTERN.search(instruction)
            if match:
                distance = int(match.group(1))
                # 将距离规范化到最近的有效步长
                if (distance % 25) != 0:
                    distance = min([25, 50, 75], key=lambda x: abs(x - distance))
                return distance
        except:
            pass
        return self.forward_step  # 默认
    
    def _extract_left_angle(self, instruction: str) -> int:
        """
        从指令中提取左转角度
        
        Args:
            instruction: 语言指令
            
        Returns:
            degree (default turn_step)
        """
        try:
            match = self.LEFT_ANGLE_PATTERN.search(instruction)
            if match:
                degree = int(match.group(1))
                # 将角度规范化到最近的有效步长
                if (degree % 15) != 0:
                    degree = min([15, 30, 45], key=lambda x: abs(x - degree))
                return degree
        except:
            pass
        return self.turn_step  # 默认
    
    def _extract_right_angle(self, instruction: str) -> int:
        """
        从指令中提取右转角度
        
        Args:
            instruction: 语言指令
            
        Returns:
            degree (default turn_step)
        """
        try:
            match = self.RIGHT_ANGLE_PATTERN.search(instruction)
            if match:
                degree = int(match.group(1))
                # 将角度规范化到最近的有效步长
                if (degree % 15) != 0:
                    degree = min([15, 30, 45], key=lambda x: abs(x - degree))
                return degree
        except:
            pass
        return self.turn_step  # 默认


