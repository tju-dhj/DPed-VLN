#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Action Parser for NaVILA
将NaVILA生成的语言指令解析为Habitat3离散动作
Parses language instructions from NaVILA into Habitat3 discrete actions
"""

import re
from typing import Optional, Tuple


class NaVILAActionParser:
    """
    解析NaVILA生成的语言指令为离散动作
    
    动作映射:
    - 0: STOP
    - 1: MOVE_FORWARD (with distance in cm)
    - 2: TURN_LEFT (with degree)
    - 3: TURN_RIGHT (with degree)
    
    Habitat3 Falcon框架的动作速度:
    - MOVE_FORWARD: 25 cm per step
    - TURN_LEFT: 10 degrees per step (用户配置为15度也支持)
    - TURN_RIGHT: 10 degrees per step
    """
    
    # 基础动作步长
    FORWARD_STEP = 25  # cm per action
    TURN_STEP = 15  # degrees per action (适配NaVILA的15度步长)
    
    # 正则表达式模式 - 放宽模式以支持多种自然语言表达方式
    # 支持多种变体：move forward, go forward, advance, turn left, rotate left等
    PATTERNS = {
        0: re.compile(r"\b(stop|halt|end|finish|complete|arrived|reached)\b", re.IGNORECASE),
        1: re.compile(r"\b(move forward|go forward|advance|proceed forward|walk forward|step forward|move ahead|go ahead)\b", re.IGNORECASE),
        2: re.compile(r"\b(turn left|rotate left|turn to the left|turn towards left|rotate to left|go left|steer left)\b", re.IGNORECASE),
        3: re.compile(r"\b(turn right|rotate right|turn to the right|turn towards right|rotate to right|go right|steer right)\b", re.IGNORECASE),
    }
    
    # 距离和角度提取模式 - 支持更多格式变体
    # 支持: "move forward 25 cm", "forward 25cm", "advance 25 centimeters"等
    DISTANCE_PATTERN = re.compile(r"(?:move forward|go forward|advance|proceed forward|forward)\s*(\d+)\s*(?:cm|centimeter|centimeters|meter|meters|m)?", re.IGNORECASE)
    # 支持: "turn left 45 degree", "rotate left 45 degrees", "left 45 deg"等
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
            instruction: NaVILA生成的语言指令
            
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
        
        支持多种格式，包括：
        - "The next action is turn left 45 degree."
        - "is turn left"
        - "turn left"
        - "move forward 25 cm"
        等
        
        Args:
            s: 输入字符串
            
        Returns:
            action_id or None if no match
        """
        # 按优先级检查：先检查STOP（避免误判），然后检查其他动作
        # STOP优先级最高，避免将"stop"误判为其他动作
        if self.PATTERNS[0].search(s):
            return 0
        
        # 检查其他动作，按顺序匹配（第一个匹配的返回）
        # 注意：需要避免将"turn left"误判为"turn right"的一部分
        # 先检查更具体的模式（如"turn left"），再检查通用模式
        
        # 检查左转（必须在右转之前检查，避免误判）
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
            distance in cm (default 25)
        """
        try:
            match = self.DISTANCE_PATTERN.search(instruction)
            if match:
                distance = int(match.group(1))
                # 将距离规范化到最近的有效步长（参考navila_trainer.py的实现）
                if (distance % 25) != 0:
                    # 找最接近的有效距离：25, 50, 75 cm
                    distance = min([25, 50, 75], key=lambda x: abs(x - distance))
                return distance
        except:
            pass
        return self.forward_step  # 默认25cm
    
    def _extract_left_angle(self, instruction: str) -> int:
        """
        从指令中提取左转角度
        
        Args:
            instruction: 语言指令
            
        Returns:
            degree (default 15)
        """
        try:
            match = self.LEFT_ANGLE_PATTERN.search(instruction)
            if match:
                degree = int(match.group(1))
                # 将角度规范化到最近的有效步长（参考navila_trainer.py的实现）
                if (degree % 15) != 0:
                    # 找最接近的有效角度：15, 30, 45 度
                    degree = min([15, 30, 45], key=lambda x: abs(x - degree))
                return degree
        except:
            pass
        return self.turn_step  # 默认15度
    
    def _extract_right_angle(self, instruction: str) -> int:
        """
        从指令中提取右转角度

        Args:
            instruction: 语言指令

        Returns:
            degree (default 15)
        """
        try:
            match = self.RIGHT_ANGLE_PATTERN.search(instruction)
            if match:
                degree = int(match.group(1))
                # 将角度规范化到最近的有效步长（参考navila_trainer.py的实现）
                if (degree % 15) != 0:
                    # 找最接近的有效角度：15, 30, 45 度
                    degree = min([15, 30, 45], key=lambda x: abs(x - degree))
                return degree
        except:
            pass
        return self.turn_step  # 默认15度

    # ── Multi-action sequence parsing ──

    # Sub-actions that can appear in a sequence (simpler than full NL)
    _ACTION_TEXT_TO_ID = {
        "move forward 25 cm": 1,
        "move forward": 1,
        "forward": 1,
        "turn left 15 degrees": 2,
        "turn left": 2,
        "left": 2,
        "turn right 15 degrees": 3,
        "turn right": 3,
        "right": 3,
        "stop": 0,
        "wait": -1,  # special: skip but don't count as action
        "pause": -1,
        "move backward 25 cm": -2,
        "move backward": -2,
        "backward": -2,
    }

    def parse_action_sequence(self, text: str) -> list:
        """Parse a multi-action sequence string into a list of discrete action IDs.

        Accepts formats like:
          - "move forward 25 cm; turn left 15 degrees; stop"
          - "1. move forward 25 cm\n2. turn left 15 degrees\n3. stop"
          - Mixed formats

        Returns:
            list[int]: action IDs, e.g. [1, 2, 0] for forward, left, stop.
            Empty list if nothing can be parsed.
        """
        if not text or not isinstance(text, str):
            return []

        # Step 1: Normalize: split on newlines, strip numbering, then split on "; "
        text = text.strip()
        candidates = []

        # Try numbered list format first: "1. move forward\n2. turn left"
        if re.match(r'^\d+[\.\)]\s', text):
            # Split on numbered markers
            parts = re.split(r'\n\s*\d+[\.\)]\s*|\n', text)
            for p in parts:
                p = re.sub(r'^\d+[\.\)]\s*', '', p).strip()
                if p:
                    candidates.append(p)
        else:
            # Semicolon-separated format
            for part in text.split(';'):
                part = part.strip().strip('"').strip("'")
                if part:
                    candidates.append(part)

        if not candidates:
            return []

        # Step 2: Parse each action text
        actions = []
        for action_text in candidates:
            action_text = action_text.strip().lower()
            action_id = self._match_action_text(action_text)
            if action_id is not None and action_id >= 0:
                actions.append(action_id)
            elif action_id == -1:
                # wait/pause: skip
                continue
            elif action_id == -2:
                # backward: not supported as discrete action, skip
                continue
            # else fallback: try the single-action parser
            elif len(action_text) > 3:
                parsed_id, _ = self.parse_action(action_text)
                if parsed_id is not None:
                    actions.append(parsed_id)

        return actions

    def _match_action_text(self, text: str):
        """Match a short action text to its ID. Returns None if no match."""
        text = text.strip().lower()
        # Exact match first
        if text in self._ACTION_TEXT_TO_ID:
            return self._ACTION_TEXT_TO_ID[text]
        # Partial match
        for key, val in self._ACTION_TEXT_TO_ID.items():
            if key in text:
                return val
        # Regex match for simple patterns
        if re.search(r'\b(?:move\s+)?forward\b', text, re.IGNORECASE):
            return 1
        if re.search(r'\bturn\s+left\b', text, re.IGNORECASE):
            return 2
        if re.search(r'\bturn\s+right\b', text, re.IGNORECASE):
            return 3
        if re.search(r'\bstop\b', text, re.IGNORECASE):
            return 0
        return None
