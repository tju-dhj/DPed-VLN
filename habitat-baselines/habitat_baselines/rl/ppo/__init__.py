#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from habitat_baselines.rl.ppo.cpc_aux_loss import CPCA
from habitat_baselines.rl.ppo.policy import (
    Net,
    NetPolicy,
    PointNavBaselinePolicy,
    Policy,
)
from habitat_baselines.rl.ppo.ppo import GRPO, PPO

# 延迟导入 StreamVLNEvaluator，避免在 StreamVLN 模块不可用时导致导入失败
try:
    from habitat_baselines.rl.ppo.streamvln_evaluator import StreamVLNEvaluator
except (ImportError, ModuleNotFoundError):
    StreamVLNEvaluator = None

# 导入 DPed_pro_expert，注册 expert_data_collector_6action trainer
try:
    from habitat_baselines.rl.ppo import DPed_pro_expert
except (ImportError, ModuleNotFoundError) as e:
    import warnings
    warnings.warn(f"Failed to import DPed_pro_expert: {e}")

# 导入 dped_trainer_server，注册 dped_trainer_server trainer (Flask HTTP server)
try:
    from habitat_baselines.rl.ppo import dped_trainer_server
except (ImportError, ModuleNotFoundError) as e:
    import warnings
    warnings.warn(f"Failed to import dped_trainer_server: {e}")

# 导入 dped_brain_trainer_server，注册 dped_brain_trainer_server trainer (Flask HTTP server + Brain)
try:
    from habitat_baselines.rl.ppo import dped_brain_trainer_server
except (ImportError, ModuleNotFoundError) as e:
    import warnings
    warnings.warn(f"Failed to import dped_brain_trainer_server: {e}")

try:
    from habitat_baselines.rl.ppo import stg_dyn_vln_trainer
except (ImportError, ModuleNotFoundError) as e:
    import warnings
    warnings.warn(f"Failed to import stg_dyn_vln_trainer: {e}")
    
__all__ = [
    "PPO",
    "GRPO",
    "Policy",
    "NetPolicy",
    "Net",
    "PointNavBaselinePolicy",
    "CPCA",
    "StreamVLNEvaluator",
]
