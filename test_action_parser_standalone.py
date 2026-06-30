#!/usr/bin/env python3

"""
独立测试动作解析器（不依赖habitat）
Standalone test for action parser (no habitat dependency)
"""

import re
from typing import Optional, Tuple


class NaVILAActionParser:
    """动作解析器（独立版本用于测试）"""
    
    FORWARD_STEP = 25
    TURN_STEP = 15
    
    PATTERNS = {
        0: re.compile(r"\bstop\b", re.IGNORECASE),
        1: re.compile(r"\bis move forward\b", re.IGNORECASE),
        2: re.compile(r"\bis turn left\b", re.IGNORECASE),
        3: re.compile(r"\bis turn right\b", re.IGNORECASE),
    }
    
    DISTANCE_PATTERN = re.compile(r"move forward (\d+) cm", re.IGNORECASE)
    LEFT_ANGLE_PATTERN = re.compile(r"turn left (\d+) degree", re.IGNORECASE)
    RIGHT_ANGLE_PATTERN = re.compile(r"turn right (\d+) degree", re.IGNORECASE)
    
    def parse_action(self, instruction: str) -> Tuple[int, int]:
        action_id = self._map_string_to_action(instruction)
        
        if action_id is None:
            return 1, 1
        
        if action_id == 0:
            return 0, 1
        elif action_id == 1:
            distance = self._extract_distance(instruction)
            num_repeats = max(1, distance // self.FORWARD_STEP)
            return 1, num_repeats
        elif action_id == 2:
            degree = self._extract_left_angle(instruction)
            num_repeats = max(1, degree // self.TURN_STEP)
            return 2, num_repeats
        elif action_id == 3:
            degree = self._extract_right_angle(instruction)
            num_repeats = max(1, degree // self.TURN_STEP)
            return 3, num_repeats
        
        return 1, 1
    
    def _map_string_to_action(self, s: str) -> Optional[int]:
        for action, pattern in self.PATTERNS.items():
            if pattern.search(s):
                return action
        return None
    
    def _extract_distance(self, instruction: str) -> int:
        try:
            match = self.DISTANCE_PATTERN.search(instruction)
            if match:
                distance = int(match.group(1))
                if distance % self.FORWARD_STEP != 0:
                    valid_distances = [25, 50, 75]
                    distance = min(valid_distances, key=lambda x: abs(x - distance))
                return distance
        except:
            pass
        return self.FORWARD_STEP
    
    def _extract_left_angle(self, instruction: str) -> int:
        try:
            match = self.LEFT_ANGLE_PATTERN.search(instruction)
            if match:
                degree = int(match.group(1))
                if degree % self.TURN_STEP != 0:
                    valid_degrees = [15, 30, 45]
                    degree = min(valid_degrees, key=lambda x: abs(x - degree))
                return degree
        except:
            pass
        return self.TURN_STEP
    
    def _extract_right_angle(self, instruction: str) -> int:
        try:
            match = self.RIGHT_ANGLE_PATTERN.search(instruction)
            if match:
                degree = int(match.group(1))
                if degree % self.TURN_STEP != 0:
                    valid_degrees = [15, 30, 45]
                    degree = min(valid_degrees, key=lambda x: abs(x - degree))
                return degree
        except:
            pass
        return self.TURN_STEP


def test_action_parser():
    """测试动作解析器"""
    print("=" * 60)
    print("测试NaVILA动作解析器")
    print("Testing NaVILA Action Parser")
    print("=" * 60)
    
    parser = NaVILAActionParser()
    
    test_cases = [
        ("The next action is stop", 0, 1, "停止"),
        ("The next action is move forward 25 cm", 1, 1, "前进25cm"),
        ("The next action is move forward 50 cm", 1, 2, "前进50cm"),
        ("The next action is move forward 75 cm", 1, 3, "前进75cm"),
        ("The next action is turn left 15 degree", 2, 1, "左转15度"),
        ("The next action is turn left 30 degree", 2, 2, "左转30度"),
        ("The next action is turn left 45 degree", 2, 3, "左转45度"),
        ("The next action is turn right 15 degree", 3, 1, "右转15度"),
        ("The next action is turn right 30 degree", 3, 2, "右转30度"),
        ("The next action is turn right 45 degree", 3, 3, "右转45度"),
    ]
    
    passed = 0
    failed = 0
    
    for instruction, expected_action, expected_repeats, description in test_cases:
        action, repeats = parser.parse_action(instruction)
        if action == expected_action and repeats == expected_repeats:
            print(f"✓ {description}: 动作={action}, 重复={repeats}")
            passed += 1
        else:
            print(f"✗ {description}: 期望(动作={expected_action}, 重复={expected_repeats}), "
                  f"实际(动作={action}, 重复={repeats})")
            failed += 1
    
    print("-" * 60)
    print(f"通过: {passed}/{len(test_cases)}")
    print(f"失败: {failed}/{len(test_cases)}")
    print("=" * 60)
    
    if failed == 0:
        print("✓✓✓ 所有测试通过！")
        print("✓✓✓ All tests passed!")
        return True
    else:
        print("✗✗✗ 部分测试失败！")
        return False


if __name__ == "__main__":
    success = test_action_parser()
    exit(0 if success else 1)
