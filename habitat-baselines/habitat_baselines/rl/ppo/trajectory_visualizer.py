#!/usr/bin/env python3

"""
轨迹可视化Trainer

功能：
1. 从指定路径加载已收集的数据（action和trajectories）
2. 机器人执行gt_actions（专家动作）
3. 记录每一帧的top_down_views（俯视图）
4. 绘制机器人行人轨迹并保存
"""

import os
import json
import numpy as np
import cv2
from typing import Dict, Any, Optional, List
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.patches as mpatches

import hydra
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from habitat import VectorEnv, logger
from habitat.config import read_write
from habitat.utils.visualizations import maps
from habitat.utils.visualizations.utils import observations_to_image

from habitat_baselines.common.habitat_env_factory import HabitatVectorEnvFactory
from habitat_baselines.common.base_trainer import BaseRLTrainer
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.utils.common import batch_obs, inference_mode
from habitat_baselines.utils.timing import g_timer


@baseline_registry.register_trainer(name="trajectory_visualizer")
class TrajectoryVisualizer(BaseRLTrainer):
    """
    轨迹可视化Trainer
    
    从已收集的数据中加载action和trajectories，重新执行并可视化
    """
    
    def __init__(self, config=None):
        super().__init__(config)
        
        # 获取轨迹可视化配置
        vis_config = config.trajectory_visualization
        
        # 数据路径
        self.data_folder = Path(vis_config.data_folder)
        self.episode_id = vis_config.episode_id
        
        # 输出路径
        self.output_folder = Path(vis_config.output_folder)
        self.output_folder.mkdir(parents=True, exist_ok=True)
        
        # 功能开关
        self.save_top_down_views = vis_config.save_top_down_views
        self.save_trajectory_plot = vis_config.save_trajectory_plot
        self.save_video = vis_config.save_video
        
        # 可视化配置
        self.draw_robot_trajectory = vis_config.draw_robot_trajectory
        self.draw_human_trajectories = vis_config.draw_human_trajectories
        self.draw_goal = vis_config.draw_goal
        self.trajectory_line_width = vis_config.trajectory_line_width
        self.trajectory_point_size = vis_config.trajectory_point_size
        
        # 颜色配置（从扁平化的标量读取）
        self.robot_color = np.array([
            vis_config.robot_color_r,
            vis_config.robot_color_g,
            vis_config.robot_color_b
        ]) / 255.0
        
        # 重建human_colors列表
        self.human_colors = []
        for i in range(6):  # 最多6个行人
            if hasattr(vis_config, f'human_color_{i}_r'):
                color = np.array([
                    getattr(vis_config, f'human_color_{i}_r'),
                    getattr(vis_config, f'human_color_{i}_g'),
                    getattr(vis_config, f'human_color_{i}_b')
                ]) / 255.0
                self.human_colors.append(color)
        
        self.goal_color = np.array([
            vis_config.goal_color_r,
            vis_config.goal_color_g,
            vis_config.goal_color_b
        ]) / 255.0
        
        # 图像配置
        self.top_down_view_size = [
            vis_config.top_down_view_width,
            vis_config.top_down_view_height
        ]
        self.video_fps = vis_config.video_fps
        
        # 加载数据
        self.gt_actions = self._load_actions()
        self.trajectories = self._load_trajectories()
        
        logger.info(f"加载了 {len(self.gt_actions)} 个动作")
        logger.info(f"加载了 {len(self.trajectories)} 帧轨迹数据")
        
        # 存储每一帧的top_down_views
        self.top_down_views = []
        self.robot_positions = []
        self.human_positions_list = []
        
    def _load_actions(self) -> List[int]:
        """加载专家动作序列"""
        action_file = self.data_folder / "action" / f"{self.episode_id}.json"
        if not action_file.exists():
            raise FileNotFoundError(f"动作文件不存在: {action_file}")
        
        with open(action_file, 'r') as f:
            actions = json.load(f)
        
        return actions
    
    def _load_trajectories(self) -> List[Dict]:
        """加载轨迹数据"""
        trajectory_file = self.data_folder / "trajectories" / f"{self.episode_id}.json"
        if not trajectory_file.exists():
            raise FileNotFoundError(f"轨迹文件不存在: {trajectory_file}")
        
        with open(trajectory_file, 'r') as f:
            trajectories = json.load(f)
        
        return trajectories
    
    def _save_top_down_view(self, top_down_map: np.ndarray, step: int):
        """保存单帧top_down视图"""
        if not self.save_top_down_views:
            return
        
        # 创建输出目录
        output_dir = self.output_folder / "top_down_views"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 调整大小
        if top_down_map.shape[:2] != tuple(reversed(self.top_down_view_size)):
            top_down_map = cv2.resize(
                top_down_map,
                tuple(reversed(self.top_down_view_size)),
                interpolation=cv2.INTER_NEAREST
            )
        
        # 保存图像
        output_path = output_dir / f"step_{step:04d}.png"
        cv2.imwrite(str(output_path), cv2.cvtColor(top_down_map, cv2.COLOR_RGB2BGR))
    
    def _draw_trajectory_on_map(
        self,
        top_down_map: np.ndarray,
        robot_positions: List[np.ndarray],
        human_positions_list: List[List[np.ndarray]],
        goal_position: Optional[np.ndarray] = None,
        map_info: Optional[Dict] = None
    ) -> np.ndarray:
        """
        在top_down_map上绘制轨迹
        
        Args:
            top_down_map: 俯视图（可能是原始地图或已经处理过的地图）
            robot_positions: 机器人位置列表
            human_positions_list: 每个行人的位置列表
            goal_position: 目标位置（可选）
            map_info: 地图信息字典，包含fog_of_war_mask等（可选）
        
        Returns:
            绘制了轨迹的俯视图
        """
        # 如果top_down_map是字典，提取地图数据
        if isinstance(top_down_map, dict):
            map_data = top_down_map.get("map", top_down_map)
            if "fog_of_war_mask" in top_down_map:
                # 使用maps工具处理地图
                from habitat.utils.visualizations import maps as habitat_maps
                map_data = habitat_maps.colorize_topdown_map(
                    map_data,
                    top_down_map["fog_of_war_mask"]
                )
            top_down_map = map_data
        
        # 确保是RGB格式
        if len(top_down_map.shape) == 2:
            top_down_map = cv2.cvtColor(top_down_map, cv2.COLOR_GRAY2RGB)
        elif top_down_map.shape[2] == 1:
            top_down_map = cv2.cvtColor(top_down_map, cv2.COLOR_GRAY2RGB)
        
        # 复制地图
        map_with_trajectory = top_down_map.copy()
        
        # 注意：top_down_map中的位置已经是像素坐标，不需要转换
        # 如果需要绘制世界坐标，需要使用maps工具进行坐标转换
        # 这里我们假设地图已经包含了agent位置标记，只需要添加轨迹线
        
        # 绘制机器人轨迹（使用简单的点连接方式）
        if self.draw_robot_trajectory and len(robot_positions) > 1:
            # 在地图上绘制轨迹点（使用相对位置）
            # 由于top_down_map已经包含了agent位置，我们只需要标记轨迹点
            for i, pos in enumerate(robot_positions):
                # 这里需要根据实际地图的坐标系统进行调整
                # 暂时跳过，因为top_down_map已经包含了agent位置
                pass
        
        # 绘制行人轨迹（同样需要坐标转换）
        if self.draw_human_trajectories:
            for human_idx, human_positions in enumerate(human_positions_list):
                if len(human_positions) > 0:
                    # 暂时跳过，因为需要坐标转换
                    pass
        
        return map_with_trajectory
    
    def _create_trajectory_plot(
        self,
        robot_positions: List[np.ndarray],
        human_positions_list: List[List[np.ndarray]],
        goal_position: Optional[np.ndarray] = None
    ):
        """
        创建轨迹图（使用matplotlib）
        
        Args:
            robot_positions: 机器人位置列表
            human_positions_list: 每个行人的位置列表
            goal_position: 目标位置（可选）
        """
        if not self.save_trajectory_plot:
            return
        
        fig, ax = plt.subplots(figsize=(12, 12))
        
        # 绘制机器人轨迹
        if self.draw_robot_trajectory and len(robot_positions) > 0:
            robot_points = np.array(robot_positions)
            ax.plot(
                robot_points[:, 0],
                robot_points[:, 2],
                color=self.robot_color,
                linewidth=self.trajectory_line_width,
                label='Robot',
                marker='o',
                markersize=self.trajectory_point_size,
                markevery=max(1, len(robot_points) // 20)  # 每20个点标记一次
            )
            # 标记起点
            ax.plot(
                robot_points[0, 0],
                robot_points[0, 2],
                'o',
                color=self.robot_color,
                markersize=10,
                label='Robot Start'
            )
        
        # 绘制行人轨迹
        if self.draw_human_trajectories:
            for human_idx, human_positions in enumerate(human_positions_list):
                if len(human_positions) > 0:
                    color = self.human_colors[human_idx % len(self.human_colors)]
                    human_points = np.array(human_positions)
                    ax.plot(
                        human_points[:, 0],
                        human_points[:, 2],
                        color=color,
                        linewidth=self.trajectory_line_width,
                        label=f'Human {human_idx}',
                        marker='s',
                        markersize=self.trajectory_point_size,
                        markevery=max(1, len(human_points) // 20)
                    )
                    # 标记起点
                    ax.plot(
                        human_points[0, 0],
                        human_points[0, 2],
                        's',
                        color=color,
                        markersize=8,
                        label=f'Human {human_idx} Start'
                    )
        
        # 绘制目标点
        if self.draw_goal and goal_position is not None:
            ax.plot(
                goal_position[0],
                goal_position[2],
                'o',
                color=self.goal_color,
                markersize=15,
                label='Goal'
            )
        
        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Z (meters)')
        ax.set_title('Robot and Human Trajectories')
        ax.legend()
        ax.grid(True)
        ax.set_aspect('equal')
        
        # 保存图像
        output_path = self.output_folder / "trajectory_plot.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"轨迹图已保存到: {output_path}")
    
    def _create_video(self, frames: List[np.ndarray]):
        """创建视频"""
        if not self.save_video or len(frames) == 0:
            return
        
        output_path = self.output_folder / "trajectory_video.mp4"
        
        # 获取帧尺寸
        height, width = frames[0].shape[:2]
        
        # 创建视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(
            str(output_path),
            fourcc,
            self.video_fps,
            (width, height)
        )
        
        # 写入每一帧
        for frame in frames:
            # 转换颜色空间（RGB -> BGR）
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        
        out.release()
        logger.info(f"视频已保存到: {output_path}")
    
    def train(self) -> None:
        """执行轨迹可视化"""
        logger.info("开始轨迹可视化...")
        
        # 创建环境
        env_factory = HabitatVectorEnvFactory()
        envs = env_factory.construct_envs(
            self.config,
            workers_ignore_signals=True,
        )
        
        # 重置环境
        observations = envs.reset()
        
        # 检查动作空间类型
        # 获取第一个环境的动作空间来确定格式
        from gym import spaces
        use_numpy_array = False
        try:
            action_space_dict = envs.action_spaces[0]
            # action_spaces[0] 可能是字典（多智能体）或直接是空间对象（单智能体）
            if isinstance(action_space_dict, dict):
                # 多智能体情况：从字典中获取agent_0的动作空间
                action_space = action_space_dict.get("agent_0")
            else:
                # 单智能体情况：直接使用
                action_space = action_space_dict
            
            if action_space is not None:
                if isinstance(action_space, spaces.Box):
                    logger.info("动作空间是Box类型，需要使用np.ndarray")
                    use_numpy_array = True
                else:
                    logger.info(f"动作空间是{type(action_space)}类型，使用标量值")
                    use_numpy_array = False
            else:
                # 如果无法获取，默认假设是Box类型（更安全）
                logger.warning("无法获取动作空间，默认使用np.ndarray格式")
                use_numpy_array = True
        except Exception as e:
            logger.warning(f"无法获取动作空间类型: {e}，默认使用np.ndarray格式")
            # 默认使用numpy数组格式，因为错误信息显示需要np.ndarray
            use_numpy_array = True
        
        # 获取初始状态
        episode_id = 0
        episode_steps = 0
        done = False
        
        # 初始化轨迹数据
        robot_positions = []
        human_positions_list = [[] for _ in range(6)]  # 假设最多6个行人
        
        # 获取初始位置
        if len(self.trajectories) > 0:
            initial_state = self.trajectories[0]
            robot_pos = np.array(initial_state["robot"]["position"])
            robot_positions.append(robot_pos)
            
            for human in initial_state.get("pedestrians", []):
                human_id = human["id"]
                if human_id < len(human_positions_list):
                    human_pos = np.array(human["position"])
                    human_positions_list[human_id].append(human_pos)
        
        # 存储视频帧
        video_frames = []
        
        # 执行动作序列
        logger.info(f"开始执行 {len(self.gt_actions)} 个动作...")
        
        for step, action in enumerate(tqdm(self.gt_actions, desc="执行动作")):
            if done:
                break
            
            # 根据动作空间类型转换动作格式
            # 注意：在多进程环境中，动作会被pickle序列化，需要确保是numpy数组
            if use_numpy_array:
                # Box类型需要np.ndarray，且必须是一维数组
                # 根据continuous_vector_action_to_hab_dict_v3，它使用action[0]来获取动作索引
                # 需要确保是形状为(1,)的一维数组
                # 使用float32类型，因为Box动作空间通常是float类型
                # 使用copy()确保是独立的数组，避免在多进程传递时出现问题
                action_value = np.array([float(action)], dtype=np.float32).copy()
                # 确保是numpy数组且形状正确
                if not isinstance(action_value, np.ndarray):
                    raise TypeError(f"action_value应该是np.ndarray，但得到{type(action_value)}")
                if action_value.ndim != 1:
                    raise ValueError(f"action_value应该是一维数组，但维度是{action_value.ndim}")
                if action_value.shape[0] != 1:
                    raise ValueError(f"action_value形状应该是(1,)，但得到{action_value.shape}")
                # 确保是C连续的数组（在多进程传递时更可靠）
                if not action_value.flags['C_CONTIGUOUS']:
                    action_value = np.ascontiguousarray(action_value)
            else:
                # Discrete类型直接使用整数
                action_value = int(action)
            
            # 执行动作
            # 关键发现：expert_data_collector_v3_2 使用 np.array([actions[env_idx]])
            # 直接传递numpy数组，而不是字典
            # 这意味着对于Box类型的动作空间，应该直接传递numpy数组
            # 
            # 但expert_data_collector_v2 使用 {"agent_0": expert_action} 字典格式
            # 这可能是因为v2使用的是Discrete动作空间
            # 
            # 解决方案：根据动作空间类型选择传递方式
            # Box类型：直接传递numpy数组（与expert_data_collector_v3_2一致）
            # Discrete类型：使用字典格式（与expert_data_collector_v2一致）
            if use_numpy_array:
                # Box类型：直接传递numpy数组
                # 确保是numpy数组且格式正确
                action_to_pass = action_value.copy() if isinstance(action_value, np.ndarray) else np.array([float(action)], dtype=np.float32)
                if not isinstance(action_to_pass, np.ndarray):
                    raise TypeError(f"动作必须是np.ndarray，但得到{type(action_to_pass)}")
                # 直接传递numpy数组（与expert_data_collector_v3_2一致）
                envs.async_step_at(0, action_to_pass)
            else:
                # Discrete类型：使用字典格式（与expert_data_collector_v2一致）
                step_action_dict = {"agent_0": action_value}
                envs.async_step_at(0, step_action_dict)
            
            # 等待步骤完成
            # wait_step_at返回的是单个环境的输出，需要解包
            result = envs.wait_step_at(0)
            # result的格式可能是(obs, reward, done, info)元组
            if isinstance(result, tuple) and len(result) == 4:
                observations_list, rewards_list, dones_list, infos_list = result
                # 转换为列表格式以保持一致性
                observations = [observations_list]
                rewards = [rewards_list]
                dones = [dones_list]
                infos = [infos_list]
            else:
                # 如果返回格式不同，尝试直接使用
                observations = [result[0]] if isinstance(result, (list, tuple)) else [result]
                rewards = [result[1]] if isinstance(result, (list, tuple)) and len(result) > 1 else [0.0]
                dones = [result[2]] if isinstance(result, (list, tuple)) and len(result) > 2 else [False]
                infos = [result[3]] if isinstance(result, (list, tuple)) and len(result) > 3 else [{}]
            
            done = dones[0]
            episode_steps += 1
            
            # 获取当前状态
            info = infos[0]
            obs = observations[0]
            
            # 获取top_down_map
            top_down_map = None
            if "top_down_map" in info:
                top_down_map = info["top_down_map"]
                if isinstance(top_down_map, dict):
                    top_down_map = top_down_map.get("map", None)
            
            # 获取当前位置（从info或observations中获取）
            robot_pos = None
            if "agent_position" in info:
                robot_pos = np.array(info["agent_position"])
            elif "agent_0_position" in info:
                robot_pos = np.array(info["agent_0_position"])
            elif "agent_0_localization_sensor" in obs:
                loc_sensor = obs["agent_0_localization_sensor"]
                if isinstance(loc_sensor, np.ndarray) and len(loc_sensor) >= 3:
                    robot_pos = loc_sensor[:3]  # 取前3个元素作为位置
            
            if robot_pos is not None:
                robot_positions.append(robot_pos)
            
            # 从轨迹数据中获取行人位置
            if step < len(self.trajectories):
                traj_data = self.trajectories[step]
                for human in traj_data.get("pedestrians", []):
                    human_id = human["id"]
                    if human_id < len(human_positions_list):
                        human_pos = np.array(human["position"])
                        human_positions_list[human_id].append(human_pos)
            
            # 保存top_down_view
            if top_down_map is not None:
                # 获取地图信息（如果top_down_map是字典）
                map_info = None
                if isinstance(top_down_map, dict):
                    map_info = top_down_map
                    top_down_map = map_info.get("map", top_down_map)
                
                # 使用maps工具处理地图（如果需要）
                try:
                    from habitat.utils.visualizations import maps as habitat_maps
                    if isinstance(map_info, dict) and "fog_of_war_mask" in map_info:
                        top_down_map = habitat_maps.colorize_topdown_map(
                            top_down_map,
                            map_info["fog_of_war_mask"]
                        )
                    # 绘制agent位置
                    if len(robot_positions) > 0:
                        current_pos = robot_positions[-1]
                        # 注意：这里需要根据实际的地图坐标系统进行转换
                        # 暂时使用maps工具提供的函数
                        top_down_map = habitat_maps.draw_agent(
                            image=top_down_map,
                            agent_center_coord=None,  # 让maps自动从info中获取
                            agent_rotation=None,
                            agent_radius_px=min(top_down_map.shape[0:2]) // 24,
                        )
                except Exception as e:
                    logger.warning(f"处理top_down_map时出错: {e}")
                
                # 绘制轨迹（简化版本，主要依赖matplotlib绘制）
                map_with_trajectory = self._draw_trajectory_on_map(
                    top_down_map,
                    robot_positions,
                    human_positions_list,
                    map_info=map_info
                )
                
                # 保存单帧
                self._save_top_down_view(map_with_trajectory, step)
                self.top_down_views.append(map_with_trajectory)
                
                # 创建视频帧（包含RGB和top_down_map）
                rgb_frame = None
                if "agent_0_overhead_front_rgb" in obs:
                    rgb_frame = obs["agent_0_overhead_front_rgb"]
                    if not isinstance(rgb_frame, np.ndarray):
                        rgb_frame = rgb_frame.cpu().numpy()
                    if rgb_frame.dtype != np.uint8:
                        rgb_frame = (rgb_frame * 255.0).astype(np.uint8)
                
                # 拼接RGB和top_down_map
                if rgb_frame is not None:
                    # 调整top_down_map大小以匹配RGB高度
                    map_height = rgb_frame.shape[0]
                    map_width = int(map_height * top_down_map.shape[1] / top_down_map.shape[0])
                    map_resized = cv2.resize(
                        map_with_trajectory,
                        (map_width, map_height),
                        interpolation=cv2.INTER_NEAREST
                    )
                    combined_frame = np.concatenate([rgb_frame, map_resized], axis=1)
                    video_frames.append(combined_frame)
        
        # 创建轨迹图
        goal_position = None
        if len(self.trajectories) > 0 and "goal" in self.trajectories[-1]:
            goal_position = np.array(self.trajectories[-1]["goal"])
        
        self._create_trajectory_plot(
            robot_positions,
            human_positions_list,
            goal_position
        )
        
        # 创建视频
        if len(video_frames) > 0:
            self._create_video(video_frames)
        
        logger.info("轨迹可视化完成！")
        logger.info(f"输出目录: {self.output_folder}")
        
        # 关闭环境
        envs.close()
    
    def _eval_checkpoint(
        self,
        checkpoint_path: str,
        writer: Any,
        checkpoint_index: int = 0,
    ) -> None:
        """评估checkpoint（本trainer不需要）"""
        pass

