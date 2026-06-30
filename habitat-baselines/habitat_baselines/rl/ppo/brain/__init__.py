# -*- coding: utf-8 -*-
"""
Brain模块 - 外接大脑系统
=========================

本模块提供基于行人检测和多模态大模型的"外接大脑"功能，用于优化机器人的导航决策。

主要组件：
- PedestrianDetector: 行人检测器
- InstructionBrain: 指令优化Brain（检测到行人时优化导航指令）
- BrainManager: 统一管理器

支持的视觉语言模型（VLM）：
- Qwen系列: qwen3_vl_2b/4b/8b/32b, qwen2_5_vl_3b/7b/72b
- LLaVA系列: llava_v1_5_7b, llava_v1_6_7b, llava_next_7b/34b
- GLM-V系列（智谱AI）: glm_4_6v, glm_4_6v_flash, glm_4_5v, glm_4_1v_9b_thinking

使用示例：
```python
from habitat_baselines.rl.ppo.brain import InstructionBrain

# 初始化
brain = InstructionBrain(
    model_type="glm_4_6v_flash",
    device="cuda"
)

# 优化指令
result = brain.optimize_instruction(
    original_instruction="走到厨房",
    current_frame=current_frame,
    pedestrian_info=pedestrian_info
)
```
"""

from .pedestrian_detection import (
    DetectorType,
    PedestrianDetection,
    PedestrianDetectionManager,
    PedestrianDetector,
    DetectionResult,
)

from .instruction_brain import (
    InstructionBrain,
    InstructionOptimizationResult,
    InstructionModifier,
    FrameRecord,
    EpisodeRecord,
    BrainModelType,
)

__all__ = [
    # 行人检测
    "PedestrianDetector",
    "PedestrianDetection",
    "PedestrianDetectionManager",
    "DetectorType",
    "DetectionResult",
    # Instruction Brain
    "InstructionBrain",
    "InstructionOptimizationResult",
    "InstructionModifier",
    "FrameRecord",
    "EpisodeRecord",
    "BrainModelType",
]

__version__ = "2.1.0"
