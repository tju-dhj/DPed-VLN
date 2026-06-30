#!/usr/bin/env python3

"""
改进的GT Action传感器

这个模块提供了与模仿学习框架兼容的gt_action传感器实现，
支持单步动作获取和轨迹级别的动作序列。

主要改进：
- 支持单步动作获取（用于在线训练）
- 支持轨迹级动作序列（用于离线训练）
- 兼容DAgger和IL训练框架
- 提供动作空间适配

作者: 基于原始GT_ActionSensor改进
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Union
from gym import spaces

from habitat.core.registry import registry
from habitat.core.simulator import Sensor, SensorTypes
from habitat.core.spaces import ActionSpace


@registry.register_sensor(name="ImprovedGT_ActionSensor")
class ImprovedGT_ActionSensor(Sensor):
    """
    改进的GT Action传感器
    
    支持两种模式：
    1. 单步模式：返回当前步骤的专家动作
    2. 轨迹模式：返回整个轨迹的专家动作序列
    
    兼容DAgger和IL训练框架
    """
    cls_uuid: str = "gt_action"
    
    def __init__(
        self,
        action_space_size: int = 4,
        mode: str = "single_step",  # "single_step" or "trajectory"
        max_trajectory_length: int = 100,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.action_space_size = action_space_size
        self.mode = mode
        self.max_trajectory_length = max_trajectory_length
        self.current_step = 0
        self.trajectory_actions = None
        
    def _get_uuid(self, *args, **kwargs):
        return self.cls_uuid
    
    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.SEMANTIC
    
    def _get_observation_space(self, *args, **kwargs):
        if self.mode == "single_step":
            # 单步模式：返回单个动作
            return spaces.Box(
                low=0,
                high=self.action_space_size - 1,
                shape=(1,),
                dtype=int,
            )
        else:
            # 轨迹模式：返回动作序列
            return spaces.Box(
                low=0,
                high=self.action_space_size - 1,
                shape=(self.max_trajectory_length,),
                dtype=int,
            )
    
    def reset(self, episode=None, **kwargs):
        """重置传感器状态"""
        self.current_step = 0
        self.trajectory_actions = None
        
        if episode and hasattr(episode, 'gt_action'):
            self.trajectory_actions = episode.gt_action
            if isinstance(self.trajectory_actions, list):
                self.trajectory_actions = torch.tensor(
                    self.trajectory_actions, dtype=torch.long
                )
    
    def get_observation(self, *args, episode=None, **kwargs):
        """获取观察数据"""
        if self.mode == "single_step":
            return self._get_single_step_action(episode)
        else:
            return self._get_trajectory_actions(episode)
    
    def _get_single_step_action(self, episode=None):
        """获取当前步骤的专家动作"""
        obs = torch.zeros(1, dtype=torch.long)
        
        if self.trajectory_actions is not None:
            if self.current_step < len(self.trajectory_actions):
                obs[0] = self.trajectory_actions[self.current_step]
            else:
                # 如果超出轨迹长度，返回停止动作
                obs[0] = 0  # 假设0是停止动作
        
        self.current_step += 1
        return obs
    
    def _get_trajectory_actions(self, episode=None):
        """获取整个轨迹的专家动作"""
        obs = torch.zeros(self.max_trajectory_length, dtype=torch.long)
        
        if self.trajectory_actions is not None:
            traj_len = min(len(self.trajectory_actions), self.max_trajectory_length)
            obs[:traj_len] = self.trajectory_actions[:traj_len]
        
        return obs


@registry.register_sensor(name="GT_ActionExpertSensor")
class GT_ActionExpertSensor(Sensor):
    """
    专家动作传感器 - 专门用于DAgger训练
    
    在DAgger训练中，这个传感器提供专家动作用于：
    1. 数据收集时的动作选择
    2. 监督学习的标签
    """
    cls_uuid: str = "expert_action"
    
    def __init__(
        self,
        action_space_size: int = 4,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.action_space_size = action_space_size
        self.current_step = 0
        self.trajectory_actions = None
        
    def _get_uuid(self, *args, **kwargs):
        return self.cls_uuid
    
    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.SEMANTIC
    
    def _get_observation_space(self, *args, **kwargs):
        return spaces.Box(
            low=0,
            high=self.action_space_size - 1,
            shape=(1,),
            dtype=int,
        )
    
    def reset(self, episode=None, **kwargs):
        """重置传感器状态"""
        self.current_step = 0
        self.trajectory_actions = None
        
        if episode and hasattr(episode, 'gt_action'):
            self.trajectory_actions = episode.gt_action
            if isinstance(self.trajectory_actions, list):
                self.trajectory_actions = torch.tensor(
                    self.trajectory_actions, dtype=torch.long
                )
    
    def get_observation(self, *args, episode=None, **kwargs):
        """获取专家动作"""
        obs = torch.zeros(1, dtype=torch.long)
        
        if self.trajectory_actions is not None:
            if self.current_step < len(self.trajectory_actions):
                obs[0] = self.trajectory_actions[self.current_step]
            else:
                # 如果超出轨迹长度，返回停止动作
                obs[0] = 0
        
        self.current_step += 1
        return obs


class GT_ActionDataProcessor:
    """
    GT Action数据处理器
    
    用于处理专家轨迹数据，支持：
    1. 数据格式转换
    2. 轨迹分割和批处理
    3. 动作序列验证
    """
    
    def __init__(self, action_space_size: int = 4):
        self.action_space_size = action_space_size
    
    def process_trajectory(
        self,
        observations: Dict[str, Any],
        gt_actions: Union[List[int], torch.Tensor],
        prev_actions: Optional[Union[List[int], torch.Tensor]] = None,
    ) -> Dict[str, Any]:
        """
        处理单个轨迹数据
        
        Args:
            observations: 观察数据字典
            gt_actions: 专家动作序列
            prev_actions: 前一步动作序列（可选）
            
        Returns:
            处理后的轨迹数据
        """
        # 转换动作格式
        if isinstance(gt_actions, list):
            gt_actions = torch.tensor(gt_actions, dtype=torch.long)
        
        if prev_actions is None:
            # 生成前一步动作序列
            prev_actions = torch.cat([
                torch.tensor([0], dtype=torch.long),  # 初始动作
                gt_actions[:-1]  # 除了最后一个动作的所有动作
            ])
        elif isinstance(prev_actions, list):
            prev_actions = torch.tensor(prev_actions, dtype=torch.long)
        
        # 验证动作空间
        self._validate_actions(gt_actions)
        self._validate_actions(prev_actions)
        
        # 处理观察数据
        processed_obs = {}
        for key, value in observations.items():
            if isinstance(value, torch.Tensor):
                processed_obs[key] = value
            elif isinstance(value, np.ndarray):
                processed_obs[key] = torch.from_numpy(value)
            else:
                # 尝试转换为tensor
                processed_obs[key] = torch.tensor(value)
        
        return {
            'observations': processed_obs,
            'gt_actions': gt_actions,
            'prev_actions': prev_actions,
            'trajectory_length': len(gt_actions),
        }
    
    def _validate_actions(self, actions: torch.Tensor):
        """验证动作的有效性"""
        if actions.min() < 0 or actions.max() >= self.action_space_size:
            raise ValueError(
                f"Actions out of range. Expected [0, {self.action_space_size-1}], "
                f"got [{actions.min()}, {actions.max()}]"
            )
    
    def batch_trajectories(
        self,
        trajectories: List[Dict[str, Any]],
        max_length: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        批处理多个轨迹
        
        Args:
            trajectories: 轨迹列表
            max_length: 最大轨迹长度（用于填充）
            
        Returns:
            批处理后的数据
        """
        if not trajectories:
            return {}
        
        # 计算最大长度
        if max_length is None:
            max_length = max(traj['trajectory_length'] for traj in trajectories)
        
        batch_size = len(trajectories)
        
        # 初始化批处理数据
        batch_data = {
            'observations': {},
            'gt_actions': torch.zeros(batch_size, max_length, dtype=torch.long),
            'prev_actions': torch.zeros(batch_size, max_length, dtype=torch.long),
            'masks': torch.zeros(batch_size, max_length, dtype=torch.bool),
        }
        
        # 获取所有观察键
        all_obs_keys = set()
        for traj in trajectories:
            all_obs_keys.update(traj['observations'].keys())
        
        # 初始化观察数据
        for key in all_obs_keys:
            # 假设所有观察都有相同的形状（除了第一个维度）
            sample_obs = trajectories[0]['observations'][key]
            if len(sample_obs.shape) > 1:
                batch_shape = (batch_size, max_length) + sample_obs.shape[1:]
            else:
                batch_shape = (batch_size, max_length)
            
            batch_data['observations'][key] = torch.zeros(
                batch_shape, dtype=sample_obs.dtype
            )
        
        # 填充数据
        for i, traj in enumerate(trajectories):
            traj_len = traj['trajectory_length']
            
            # 填充动作数据
            batch_data['gt_actions'][i, :traj_len] = traj['gt_actions']
            batch_data['prev_actions'][i, :traj_len] = traj['prev_actions']
            batch_data['masks'][i, :traj_len] = True
            
            # 填充观察数据
            for key, value in traj['observations'].items():
                if len(value.shape) > 1:
                    batch_data['observations'][key][i, :traj_len] = value
                else:
                    batch_data['observations'][key][i, :traj_len] = value
        
        return batch_data


def create_gt_action_sensor_config(
    action_space_size: int = 4,
    mode: str = "single_step",
    max_trajectory_length: int = 100,
) -> Dict[str, Any]:
    """
    创建GT Action传感器配置
    
    Args:
        action_space_size: 动作空间大小
        mode: 传感器模式 ("single_step" 或 "trajectory")
        max_trajectory_length: 最大轨迹长度
        
    Returns:
        传感器配置字典
    """
    return {
        "type": "ImprovedGT_ActionSensor",
        "uuid": "gt_action",
        "action_space_size": action_space_size,
        "mode": mode,
        "max_trajectory_length": max_trajectory_length,
    }


def create_expert_action_sensor_config(
    action_space_size: int = 4,
) -> Dict[str, Any]:
    """
    创建专家动作传感器配置（用于DAgger）
    
    Args:
        action_space_size: 动作空间大小
        
    Returns:
        传感器配置字典
    """
    return {
        "type": "GT_ActionExpertSensor",
        "uuid": "expert_action",
        "action_space_size": action_space_size,
    }


# 示例使用
if __name__ == "__main__":
    # 创建传感器
    sensor = ImprovedGT_ActionSensor(
        action_space_size=4,
        mode="single_step"
    )
    
    # 创建数据处理器
    processor = GT_ActionDataProcessor(action_space_size=4)
    
    # 示例轨迹数据
    sample_observations = {
        'rgb': torch.randn(10, 3, 224, 224),
        'depth': torch.randn(10, 1, 224, 224),
    }
    sample_gt_actions = [1, 2, 0, 1, 3, 0, 2, 1, 0, 0]
    
    # 处理轨迹
    processed_traj = processor.process_trajectory(
        sample_observations,
        sample_gt_actions
    )
    
    print("Processed trajectory:")
    print(f"GT Actions: {processed_traj['gt_actions']}")
    print(f"Prev Actions: {processed_traj['prev_actions']}")
    print(f"Trajectory Length: {processed_traj['trajectory_length']}")
