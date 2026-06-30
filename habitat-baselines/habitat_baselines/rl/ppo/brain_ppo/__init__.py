# -*- coding: utf-8 -*-
# ==============================================================================
# 文件: __init__.py (brain_ppo子模块)
# 描述: BrainPPO训练器和评估器的导出
# ==============================================================================

"""
BrainPPO子模块
===============

导出Brain增强的PPO训练器和评估器。

使用方式：
```python
from habitat_baselines.rl.ppo.brain_ppo import BrainPPOTrainer, BrainPPOEvaluator
```
"""

from .brain_ppo_trainer import BrainPPOTrainer
from .brain_ppo_evaluator import BrainPPOEvaluator

__all__ = [
    "BrainPPOTrainer",
    "BrainPPOEvaluator",
]
