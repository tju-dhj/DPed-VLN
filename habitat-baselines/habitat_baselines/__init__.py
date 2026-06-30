#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import sys
import os

# 动态获取当前文件的路径，而不是硬编码
current_file = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file)

# 从当前文件位置计算项目根目录
# __file__ 是 habitat-baselines/habitat_baselines/__init__.py
# project_root 应该是 habitat-baselines/
# falcon_root 应该是 DPed_pro/
if "habitat-baselines" in current_dir:
    # current_dir 是 .../DPed_pro/habitat-baselines/habitat_baselines
    project_root = os.path.dirname(current_dir)  # .../DPed_pro/habitat-baselines
    falcon_root = os.path.dirname(project_root)  # .../DPed_pro
else:
    # 如果路径结构不符合预期，使用当前工作目录
    falcon_root = os.getcwd()
    project_root = os.path.join(falcon_root, "habitat-baselines")

# 获取habitat-lab目录（通常与habitat-baselines同级）
habitat_lab_dir = os.path.join(falcon_root, "habitat-lab")

# 动态添加路径到 sys.path
if project_root not in sys.path:
    sys.path.insert(0, project_root)
if habitat_lab_dir not in sys.path and os.path.exists(habitat_lab_dir):
    sys.path.insert(0, habitat_lab_dir)
if falcon_root not in sys.path:
    sys.path.append(falcon_root)

# 调试信息
print(f"[habitat_baselines.__init__] 使用路径:")
print(f"  project_root: {project_root}")
print(f"  falcon_root: {falcon_root}")
print(f"  habitat_lab_dir: {habitat_lab_dir}")

# 尝试导入基础类，如果失败则延迟导入
try:
    from habitat_baselines.common.base_il_trainer import BaseILTrainer
except (ImportError, ModuleNotFoundError) as e:
    # 静默处理导入错误，避免在导入时中断
    BaseILTrainer = None

try:
    from habitat_baselines.common.base_trainer import BaseRLTrainer, BaseTrainer
except (ImportError, ModuleNotFoundError) as e:
    # 静默处理导入错误，避免在导入时中断
    BaseRLTrainer = None
    BaseTrainer = None

try:
    from habitat_baselines.common.rollout_storage import RolloutStorage
except (ImportError, ModuleNotFoundError) as e:
    # 静默处理导入错误，避免在导入时中断
    RolloutStorage = None

# 延迟导入trainer，避免在__init__时就失败
try:
    from habitat_baselines.il.trainers.eqa_cnn_pretrain_trainer import (
        EQACNNPretrainTrainer,
    )
except ImportError as e:
    print(f"Warning: Could not import EQACNNPretrainTrainer: {e}")
    EQACNNPretrainTrainer = None

try:
    from habitat_baselines.il.trainers.pacman_trainer import PACMANTrainer
except ImportError as e:
    print(f"Warning: Could not import PACMANTrainer: {e}")
    PACMANTrainer = None

try:
    from habitat_baselines.il.trainers.vqa_trainer import VQATrainer
except ImportError as e:
    print(f"Warning: Could not import VQATrainer: {e}")
    VQATrainer = None
from habitat_baselines.rl.ppo.ppo_trainer import PPOTrainer
from habitat_baselines.rl.ver.ver_trainer import VERTrainer
from habitat_baselines.rl.ppo.orca_trainer import ORCANoTrainer
from habitat_baselines.rl.ppo.falcon_trainer import FalconTrainer
from habitat_baselines.rl.ppo.DPed_pro_expert import ExpertDataCollector6Action
from habitat_baselines.rl.ppo.collect_data import Collect_data
from habitat_baselines.rl.ppo.expert_data_collector import ExpertDataCollector
from habitat_baselines.rl.ppo.expert_data_collector_v3 import ExpertDataCollectorV3
from habitat_baselines.rl.ppo.dynamic_vln_trainer import DynamicVLNTrainer
from habitat_baselines.rl.ppo.trajectory_visualizer import TrajectoryVisualizer
from habitat_baselines.version import VERSION as __version__  # noqa: F401

# Brain模块Trainer导入
from habitat_baselines.rl.ppo.instruction_brain_ppo_trainer import InstructionBrainPPOTrainer
from habitat_baselines.rl.ppo.instruction_brain_ppo_evaluator import InstructionBrainPPOEvaluator
from habitat_baselines.rl.ppo.dped_trainer_server import DPedTrainerServer
from habitat_baselines.rl.ppo.stg_dyn_vln_trainer import STGDynamicVLNTrainer


__all__ = [
    "BaseTrainer",
    "BaseRLTrainer",
    "BaseILTrainer",
    "PPOTrainer",
    "FalconTrainer",
    "RolloutStorage",
    "EQACNNPretrainTrainer",
    "PACMANTrainer",
    "VQATrainer",
    "VERTrainer",
    "ORCANoTrainer",
    "Collect_data",
    "ExpertDataCollector",
    "ExpertDataCollectorV3",
    "ExpertDataCollector6Action",
    "DynamicVLNTrainer",
    "TrajectoryVisualizer",
    # Brain模块Trainer
    "InstructionBrainPPOTrainer",
    "InstructionBrainPPOEvaluator",
    # DPedTrainerServer (Flask HTTP server trainer)
    "DPedTrainerServer",
    "STGDynamicVLNTrainer",
]
