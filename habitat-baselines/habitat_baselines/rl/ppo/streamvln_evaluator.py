#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
StreamVLN Evaluator for Falcon Framework
专门处理StreamVLN策略的评估器，支持语言指令到动作的转换
保留StreamVLN的多轮对话机制
"""

import copy
import gc
import json
import os
import sys
import textwrap
import traceback
from collections import defaultdict
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import tqdm
from PIL import Image, ImageDraw, ImageFont

from habitat import logger
from habitat.tasks.rearrange.rearrange_sensors import GfxReplayMeasure
from habitat.tasks.rearrange.utils import write_gfx_replay
from habitat.utils.visualizations.utils import (
    observations_to_image,
    overlay_frame,
    overlay_text_to_image,
)
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
)
from habitat_baselines.rl.ppo.evaluator import Evaluator, pause_envs
from habitat_baselines.rl.ppo.falcon_evaluator import FALCONEvaluator
from habitat_baselines.utils.common import (
    batch_obs,
    generate_video,
    get_action_space_info,
    inference_mode,
)
from habitat_baselines.utils.info_dict import extract_scalars_from_info
# StreamVLN相关导入 - 延迟导入以避免循环依赖
STREAMVLN_AVAILABLE = False
STREAMVLN_IMPORT_ERROR: Optional[str] = None
StreamVLNActionParser = None

try:
    from habitat_baselines.rl.ddppo.policy import streamvln_policy as _streamvln_policy_module

    _streamvln_policy_module._maybe_extend_streamvln_sys_path()
    try:
        _streamvln_policy_module._ensure_streamvln_transformers_compatibility()
    except Exception:
        pass

    # StreamVLN相关导入
    from habitat_baselines.rl.ddppo.policy.streamvln.action_parser import StreamVLNActionParser

    STREAMVLN_AVAILABLE = True
    STREAMVLN_IMPORT_ERROR = None
except Exception as e:  # 不只捕获 ImportError，依赖缺失/版本冲突也会抛别的异常
    STREAMVLN_AVAILABLE = False
    STREAMVLN_IMPORT_ERROR = "".join(
        traceback.format_exception(type(e), e, e.__traceback__)
    )
    # 只在调试时打印，避免干扰正常使用
    import logging
    logger_temp = logging.getLogger(__name__)
    logger_temp.debug(f"StreamVLN modules not available. Details:\n{STREAMVLN_IMPORT_ERROR}")


def _find_rgb_key(batch: Dict[str, Any], config=None) -> Optional[str]:
    """
    查找可用的RGB key
    
    优先顺序（仿照NaVILA的实现）：
    1. "rgb" (经过obs_transforms后的通用key)
    2. agent_0_overhead_front_rgb (overhead传感器，第一视角)
    3. agent_0_articulated_agent_jaw_rgb (Falcon默认的RGB key)
    4. agent_0_third_rgb (第三视角RGB)
    5. 其他包含"rgb"的agent_0相关key
    
    Args:
        batch: 观察批次
        config: 配置对象（可选，用于检查配置的传感器）
        
    Returns:
        RGB key，如果找不到则返回None
    """
    # 按优先级尝试不同的RGB key
    rgb_keys = [
        "rgb",  # 通用key（可能经过obs_transforms后存在）
        "agent_0_overhead_front_rgb",  # overhead传感器（第一视角）
        "agent_0_articulated_agent_jaw_rgb",  # Falcon默认的RGB key
        "agent_0_third_rgb",  # 第三视角RGB
    ]
    
    # 先尝试预定义的keys
    for key in rgb_keys:
        if key in batch:
            return key
    
    # 如果都没找到，尝试查找任何包含"rgb"的agent_0相关key
    for key in batch.keys():
        if "agent_0" in key and "rgb" in key.lower():
            return key
    
    return None


def _load_overlay_font(size: int = 18) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def _annotate_frame_with_instruction(
    frame: np.ndarray, debug_info: Optional[Dict[str, Any]]
) -> np.ndarray:
    if not debug_info:
        return frame
    instruction = debug_info.get("instruction")
    if not instruction:
        return frame
    img = Image.fromarray(frame)
    width, height = img.size
    overlay_height = max(60, height // 8)
    overlay = Image.new("RGBA", (width, overlay_height), (0, 0, 0, 180))
    img.paste(overlay, (0, height - overlay_height), overlay)
    draw = ImageDraw.Draw(img)
    font = _load_overlay_font(size=20)
    wrapped_text = textwrap.fill(str(instruction), width=60)
    draw.text((16, height - overlay_height + 12), wrapped_text, fill=(255, 255, 255), font=font)
    return np.array(img)


def _write_instruction_log(
    video_dir: str,
    video_name: str,
    instruction_log: List[Dict[str, Any]],
) -> None:
    if not instruction_log:
        return
    json_path = os.path.join(video_dir, f"{video_name}.json")
    serializable = [
        {
            "step": entry.get("step"),
            "instruction": entry.get("instruction"),
            "model_output": entry.get("model_output"),
            "action": entry.get("action_name") or entry.get("action_id"),
            "repeats": entry.get("repeats"),
            "repeat_index": entry.get("repeat_index"),
            "from_queue": entry.get("from_queue"),
        }
        for entry in instruction_log
    ]
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)


def _build_video_basename(
    scene_id: str,
    episode_id: Union[int, str],
    checkpoint_idx: int,
    metrics: Dict[str, float],
    keys_to_include_in_name: Optional[List[str]] = None,
) -> str:
    if keys_to_include_in_name:
        use_keys = [
            k
            for k in metrics
            if any(to_include in k for to_include in keys_to_include_in_name)
        ]
    else:
        use_keys = list(metrics.keys())
    metric_str = "-".join(f"{k}={metrics[k]:.2f}" for k in use_keys)
    return f"scene={scene_id}-episode={episode_id}-ckpt={checkpoint_idx}-" + metric_str


def format_streamvln_debug_overlay(
    debug_info: Optional[Dict[str, Any]],
    width: int = 120,
) -> Optional[str]:
    """格式化StreamVLN调试信息用于视频叠加"""
    if not debug_info:
        return None
    lines: List[str] = []
    instruction = debug_info.get("instruction")
    model_output = debug_info.get("model_output")
    action_name = debug_info.get("action_name")
    repeats = debug_info.get("repeats", 1)
    repeat_index = debug_info.get("repeat_index", 1)
    if instruction:
        lines.append(f"Instr: {instruction[:width]}")
    if model_output:
        lines.append(f"LLM: {model_output[:width]}")
    if action_name is not None:
        if repeats > 1:
            lines.append(f"Action: {action_name} ({repeat_index}/{repeats})")
        else:
            lines.append(f"Action: {action_name}")
    if not lines:
        return None
    return "\n".join(lines)


def _tensor_like_to_numpy(data: Any) -> Optional[np.ndarray]:
    """将tensor-like数据转换为numpy数组"""
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    if isinstance(data, np.ndarray):
        return data
    if isinstance(data, (list, tuple)):
        try:
            return np.array(data)
        except Exception:
            return None
    return None


def _decode_text_instruction(data: Any) -> Optional[str]:
    """解码文本指令，支持多种数据格式"""
    if data is None:
        return None
    if isinstance(data, str):
        text = data.strip()
        return text if text else None
    if isinstance(data, (list, tuple)):
        # Assume list of tokens or characters
        try:
            chars = []
            for item in data:
                if isinstance(item, str) and item:
                    chars.append(item)
                elif isinstance(item, (int, np.integer)):
                    if item == 0 and chars:
                        break
                    if 32 <= int(item) < 127:
                        chars.append(chr(int(item)))
                else:
                    chars.append(str(item))
            text = "".join(chars).strip()
            return text if text else None
        except Exception:
            pass
    arr = _tensor_like_to_numpy(data)
    if arr is None:
        return None
    arr = np.asarray(arr).astype(np.int32).flatten()
    chars: List[str] = []
    for val in arr:
        if val == 0:
            if chars:
                break
            continue
        char_code = int(val) % 256
        if 32 <= char_code <= 126:
            chars.append(chr(char_code))
    text = "".join(chars).strip()
    return text if text else None


def extract_streamvln_instruction(
    observations: Dict[str, Any],
    instruction_sensor_uuid: Optional[str] = None,
    episode_instruction: Optional[str] = None,
) -> str:
    """从观察中提取导航指令，仿照NaVILA的实现"""
    if episode_instruction and isinstance(episode_instruction, str) and episode_instruction.strip():
        return episode_instruction.strip()

    DEFAULT_INSTRUCTION_KEYS = [
        "agent_0_falcon_instruction",
        "falcon_instruction",
        "instruction",
        "instruction_sensor",
    ]
    
    keys_to_try: List[str] = []
    if instruction_sensor_uuid:
        keys_to_try.append(instruction_sensor_uuid)
    keys_to_try.extend(DEFAULT_INSTRUCTION_KEYS)

    for key in keys_to_try:
        if key and key in observations:
            text = _decode_text_instruction(observations[key])
            if text:
                return text

    return "Navigate to the goal location"


class StreamVLNEvaluator(FALCONEvaluator):
    """
    StreamVLN专用评估器
    
    继承自FALCONEvaluator，但使用StreamVLN的语言模型直接生成动作，
    而不是通过policy网络。保留StreamVLN的多轮对话机制。
    """
    
    # 动作ID到名称的映射
    ACTION_ID_TO_NAME = {
        0: "stop",
        1: "move forward",
        2: "turn left",
        3: "turn right",
    }
    
    def evaluate_agent(
        self,
        agent,
        envs,
        config,
        checkpoint_index,
        step_id,
        writer,
        device,
        obs_transforms,
        env_spec,
        rank0_keys,
    ):
        """
        评估StreamVLN agent
        
        该方法重写父类方法，直接使用StreamVLN模型生成动作指令。
        保留StreamVLN的多轮对话机制。
        """
        # 如果 StreamVLN 模块不可用：降级到普通 Falcon evaluator，避免直接崩溃
        if not STREAMVLN_AVAILABLE:
            logger.warning(
                "StreamVLN modules unavailable, fallback to FALCONEvaluator.evaluate_agent(). "
                "To enable StreamVLN, fix import error:\n%s",
                STREAMVLN_IMPORT_ERROR or "Unknown import error",
            )
            return super().evaluate_agent(
                agent=agent,
                envs=envs,
                config=config,
                checkpoint_index=checkpoint_index,
                step_id=step_id,
                writer=writer,
                device=device,
                obs_transforms=obs_transforms,
                env_spec=env_spec,
                rank0_keys=rank0_keys,
            )
        # 获取StreamVLN配置
        streamvln_config = config.habitat_baselines.rl.policy.agent_0
        streamvln_model_path = streamvln_config.get("model_path", None)
        
        if streamvln_model_path is None or not os.path.exists(streamvln_model_path):
            raise ValueError(
                f"StreamVLN model path is not provided or does not exist: {streamvln_model_path}"
            )
        
        # logger.info(f"Using StreamVLN model from {streamvln_model_path}")
        
        # 从agent中获取StreamVLN网络
        # 参考 navila_evaluator.py 的处理方式，支持 MultiAgentAccessMgr
        streamvln_net = None
        target_agent = agent
        
        # 如果是MultiAgentAccessMgr，从_agents[0]获取agent_0
        if hasattr(agent, '_agents') and len(agent._agents) > 0:
            target_agent = agent._agents[0]
            # logger.info(f"Multi-agent detected, using agent_0 from _agents[0]")
        
        if hasattr(target_agent, 'actor_critic'):
            actor_critic = target_agent.actor_critic
            # NetPolicy有net属性，直接访问
            if hasattr(actor_critic, 'net'):
                streamvln_net = actor_critic.net
                # logger.info(f"Found net in agent.actor_critic.net: {type(streamvln_net).__name__}")
            else:
                logger.warning(f"actor_critic does not have 'net' attribute. Type: {type(actor_critic).__name__}")
        else:
            logger.warning(f"target_agent does not have 'actor_critic' attribute. Type: {type(target_agent).__name__}")
        
        if streamvln_net is None:
            raise ValueError(
                f"Could not find StreamVLN net in agent structure. "
                f"Agent type: {type(agent).__name__}, "
                f"target_agent type: {type(target_agent).__name__}, "
                f"has actor_critic: {hasattr(target_agent, 'actor_critic') if target_agent else False}"
            )
        
        # 保存 target_agent 供后续使用
        self._target_agent = target_agent
        
        instruction_sensor_uuid = streamvln_config.get("instruction_sensor_uuid", None)
        # 初始化动作解析器（如果可用）
        if StreamVLNActionParser is not None:
            action_parser = StreamVLNActionParser(
                forward_step=streamvln_config.get("forward_step", 25),
                turn_step=streamvln_config.get("turn_step", 15),
            )
        else:
            action_parser = None
            logger.warning("StreamVLNActionParser not available, action parsing may be limited")
        
        num_frames = streamvln_config.get("num_frames", 32)
        num_history = streamvln_config.get("num_history", 8)
        
        success_cal = 0
        observations = envs.reset()
        observations = envs.post_step(observations)
        batch = batch_obs(observations, device=device)
        batch = apply_obs_transforms_batch(batch, obs_transforms)
        
        current_episode_reward = torch.zeros(envs.num_envs, 1, device="cpu")
        
        stats_episodes: Dict[Any, Any] = {}
        ep_eval_count: Dict[Any, int] = defaultdict(lambda: 0)
        
        # 动作队列（每个环境一个）
        action_queues = [[] for _ in range(envs.num_envs)]
        streamvln_step_debug: List[Optional[Dict[str, Any]]] = [None for _ in range(envs.num_envs)]
        
        if len(config.habitat_baselines.eval.video_option) > 0:
            rgb_frames: List[List[np.ndarray]] = [
                [observations_to_image({k: v[env_idx] for k, v in batch.items()}, {})]
                for env_idx in range(config.habitat_baselines.num_environments)
            ]
        else:
            rgb_frames = None
        
        if len(config.habitat_baselines.eval.video_option) > 0:
            os.makedirs(config.habitat_baselines.video_dir, exist_ok=True)
        
        number_of_eval_episodes = config.habitat_baselines.test_episode_count
        evals_per_ep = config.habitat_baselines.eval.evals_per_ep
        if number_of_eval_episodes == -1:
            number_of_eval_episodes = sum(envs.number_of_episodes)
        else:
            total_num_eps = sum(envs.number_of_episodes)
            if total_num_eps < number_of_eval_episodes and total_num_eps > 1:
                logger.warn(
                    f"Config specified {number_of_eval_episodes} eval episodes, "
                    f"dataset only has {total_num_eps}. Evaluating with {total_num_eps} instead."
                )
                number_of_eval_episodes = total_num_eps
            else:
                assert evals_per_ep == 1
        
        assert number_of_eval_episodes > 0, (
            "You must specify a number of evaluation episodes with test_episode_count"
        )
        
        pbar = tqdm.tqdm(total=number_of_eval_episodes * evals_per_ep)
        actions_record = defaultdict(list)
        max_episode_steps = getattr(
            config.habitat_baselines.eval, "max_steps_per_episode", -1
        )
        episode_steps = [0 for _ in range(envs.num_envs)]
        episode_instruction_logs: List[List[Dict[str, Any]]] = [
            [] for _ in range(envs.num_envs)
        ]
        
        # 内存管理配置
        clear_cache_interval = getattr(
            config.habitat_baselines.eval, "clear_cache_interval", 10
        )  # 每10步清理一次缓存
        clear_cache_on_episode_end = getattr(
            config.habitat_baselines.eval, "clear_cache_on_episode_end", True
        )  # episode结束时清理缓存
        step_count = 0
        
        while (
            len(stats_episodes) < (number_of_eval_episodes * evals_per_ep)
            and envs.num_envs > 0
        ):
            current_episodes_info = envs.current_episodes()
            
            # 为每个环境生成动作
            # 参考 navila_evaluator.py，需要获取动作空间信息并构建正确的动作格式
            action_shape, discrete_actions = get_action_space_info(
                self._target_agent.actor_critic.policy_action_space
            )
            hidden_state_lens = self._target_agent.actor_critic.hidden_state_shape_lens
            action_space_lens = self._target_agent.actor_critic.policy_action_space_shape_lens
            if len(action_space_lens) == 0:
                raise ValueError("policy_action_space_shape_lens is empty, cannot map agent actions.")
            
            # 处理 action_space_lens[0] 可能是 Discrete 对象的情况
            # 参考 pop_play_wrappers.py 的处理方式
            from gym import spaces
            agent0_space = action_space_lens[0]
            if isinstance(agent0_space, spaces.Discrete):
                # Discrete 空间的宽度是 1（每个动作是标量）
                agent0_width = 1
            elif isinstance(agent0_space, spaces.Box):
                # Box 空间的宽度是 shape 的乘积
                agent0_width = int(np.prod(agent0_space.shape))
            elif isinstance(agent0_space, (int, np.integer)):
                # 如果已经是数字，直接使用
                agent0_width = int(agent0_space)
            else:
                # 尝试转换为数字
                try:
                    agent0_width = int(agent0_space)
                except (TypeError, ValueError):
                    raise ValueError(
                        f"Unsupported action space type for agent_0: {type(agent0_space)}. "
                        f"Expected Discrete, Box, or int, got {agent0_space}."
                    )
            
            if agent0_width != 1:
                raise ValueError(
                    f"StreamVLN expects agent_0 action length 1, but got {agent0_width}. "
                    "Please ensure agent_0 policy is discrete."
                )
            agent0_start = 0
            agent0_end = agent0_start + agent0_width
            
            # 构建基础动作数组（用于多代理场景）
            # 对于 StreamVLN，我们直接构建动作数组，而不是调用 actor_critic.act()
            # 因为 StreamVLN 直接生成动作，不需要通过 policy 的 act() 方法
            n_agents = len(config.habitat.simulator.agents)
            
            # 计算总动作维度
            total_action_dim = sum(
                int(np.prod(space.shape)) if isinstance(space, spaces.Box) 
                else 1 if isinstance(space, spaces.Discrete)
                else int(space) if isinstance(space, (int, np.integer))
                else 1
                for space in action_space_lens
            )
            
            # 初始化基础动作为零（其他代理使用默认动作）
            base_actions = np.zeros(
                (config.habitat_baselines.num_environments, total_action_dim),
                dtype=np.float32
            )
            
            # 为每个环境生成动作（仅覆盖agent_0的离散动作，其余代理沿用actor输出）
            streamvln_actions: List[int] = []
            for i in range(envs.num_envs):
                # 如果动作队列中有动作，直接使用
                if len(action_queues[i]) > 0:
                    queued_entry = action_queues[i].pop(0)
                    if isinstance(queued_entry, dict):
                        action = int(queued_entry.get("action", 0))
                        streamvln_step_debug[i] = queued_entry.get("debug")
                    else:
                        action = int(queued_entry)
                        streamvln_step_debug[i] = None
                    action_name = (
                        streamvln_step_debug[i].get("action_name")
                        if streamvln_step_debug[i]
                        else str(action)
                    )
                    logger.info(f"Env {i}: Using queued action {action_name} (queue_len={len(action_queues[i])})")
                    if streamvln_step_debug[i]:
                        episode_instruction_logs[i].append(dict(streamvln_step_debug[i]))
                else:
                    # 否则，使用StreamVLN生成新动作
                    action, debug_info = self._generate_streamvln_action(
                        batch, i, current_episodes_info[i], streamvln_net, 
                        action_parser, action_queues[i], device, config=config,
                        instruction_sensor_uuid=instruction_sensor_uuid,
                    )
                    streamvln_step_debug[i] = debug_info
                    if streamvln_step_debug[i]:
                        episode_instruction_logs[i].append(dict(streamvln_step_debug[i]))
                
                streamvln_actions.append(action)
                base_actions[i, agent0_start:agent0_end] = float(action)
            
            for i in range(envs.num_envs):
                action_name = self.ACTION_ID_TO_NAME.get(streamvln_actions[i], str(streamvln_actions[i]))
                logger.info(
                    "[StreamVLN][eval] Env %d executing action: %s (%d)",
                    i,
                    action_name,
                    streamvln_actions[i],
                )
            
            step_data = [base_actions[i].copy() for i in range(envs.num_envs)]
            
            # 执行动作
            outputs = envs.step(step_data)
            observations, rewards_l, dones, infos = [list(x) for x in zip(*outputs)]
            
            # ✅ 完全按照 streamvln_eval.py line 345-350 的逻辑
            # 在执行动作后，递增 step_id 并检查是否需要重置
            # 参考代码：
            #   observations = env.step(action)
            #   step_id += 1
            #   if step_id % self.num_frames == 0:
            #       self.model.reset_for_env(idx)
            #       output_ids = None
            #       past_key_values = None
            #       time_ids = []
            
            for i in range(envs.num_envs):
                # 递增 step_id（注意：step_id 在 streamvln_net 内部管理）
                streamvln_net.step_id += 1
                logger.info(f"[StreamVLN][env {i}] step_id 递增至 {streamvln_net.step_id}")
                
                # 检查是否需要重置
                if streamvln_net.step_id % num_frames == 0:
                    streamvln_net.model.reset_for_env(i)
                    streamvln_net.output_ids = None
                    streamvln_net.past_key_values = None
                    streamvln_net.time_ids = []
                    logger.info(
                        f"[StreamVLN][env {i}] ★★★ Reset at step {streamvln_net.step_id} "
                        f"(step_id % num_frames == 0, num_frames={num_frames}) ★★★"
                    )
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            
            # 记录动作
            for i in range(envs.num_envs):
                episode_steps[i] += 1
                if (
                    max_episode_steps > 0
                    and episode_steps[i] >= max_episode_steps
                    and not dones[i]
                ):
                    logger.info(
                        "[StreamVLN][eval] Env %d reached max steps (%d). Forcing episode end.",
                        i,
                        max_episode_steps,
                    )
                    infos[i]["max_step_reached"] = True
                    dones[i] = True

                episode_key = (
                    current_episodes_info[i].scene_id,
                    current_episodes_info[i].episode_id,
                    ep_eval_count[
                        (current_episodes_info[i].scene_id, current_episodes_info[i].episode_id)
                    ]
                )
                record_entry: Dict[str, Any] = {
                    "type": "scalar",
                    "value": int(step_data[i]),
                }
                debug_info = streamvln_step_debug[i]
                if debug_info:
                    record_entry.update(
                        {
                            "action_name": debug_info.get("action_name"),
                            "instruction": debug_info.get("instruction"),
                            "model_output": debug_info.get("model_output"),
                            "repeats": debug_info.get("repeats"),
                            "repeat_index": debug_info.get("repeat_index"),
                            "from_queue": debug_info.get("from_queue"),
                        }
                    )
                actions_record[episode_key].append(record_entry)
            
            # 更新观察
            observations = envs.post_step(observations)
            batch = batch_obs(observations, device=device)
            batch = apply_obs_transforms_batch(batch, obs_transforms)
            
            # 重要：更新模型的历史状态（rgb_list, depth_list等）
            # 即使从队列中取出动作，执行后环境状态已经改变，需要更新模型的历史列表
            # 这样模型才能保持与环境状态同步
            for i in range(envs.num_envs):
                if not dones[i]:  # 只更新未结束的环境
                    # 获取当前观察（使用_find_rgb_key查找RGB传感器）
                    rgb_key = _find_rgb_key(batch, config)
                    if rgb_key is None:
                        # 如果找不到RGB，尝试使用depth作为fallback
                        rgb_key = 'agent_0_articulated_agent_jaw_depth' if 'agent_0_articulated_agent_jaw_depth' in batch else 'depth'
                    
                    rgb_obs = batch[rgb_key][i]
                    rgb_np = rgb_obs.cpu().numpy()
                    if rgb_np.dtype != np.uint8:
                        if rgb_np.max() <= 1.0:
                            rgb_np = (rgb_np * 255).astype(np.uint8)
                        else:
                            rgb_np = rgb_np.astype(np.uint8)
                    
                    # 获取深度图像
                    depth_key = 'agent_0_articulated_agent_jaw_depth' if 'agent_0_articulated_agent_jaw_depth' in batch else 'depth'
                    depth_np = None
                    if depth_key in batch:
                        depth_obs = batch[depth_key][i]
                        depth_np = depth_obs.cpu().numpy()
                        if len(depth_np.shape) == 2:
                            depth_np = depth_np.reshape(depth_np.shape[0], depth_np.shape[1], 1)
                        elif len(depth_np.shape) == 3 and depth_np.shape[2] > 1:
                            depth_np = depth_np[:, :, 0:1]
                    
                    # 调用 generate_action_sequence 但设置 run_model=False
                    # 这样只会更新历史列表（rgb_list, depth_list等），不会生成新动作
                    try:
                        streamvln_net.generate_action_sequence(
                            rgb=rgb_np,
                            depth=depth_np,
                            instruction="",  # 不需要指令，只是更新历史
                            env_idx=i,
                            run_model=False  # 关键：设置为False，只更新历史列表，不生成新动作
                        )
                    except Exception as e:
                        # 如果更新失败，记录警告但不中断评估
                        logger.warning(f"Failed to update model history for env {i} after action execution: {e}")
            
            rewards = torch.tensor(rewards_l, dtype=torch.float, device="cpu").unsqueeze(1)
            current_episode_reward += rewards
            next_episodes_info = envs.current_episodes()
            envs_to_pause = []
            n_envs = envs.num_envs
            
            # 定期清理GPU缓存
            step_count += 1
            if step_count % clear_cache_interval == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()
            
            for i in range(n_envs):
                if (
                    ep_eval_count[(next_episodes_info[i].scene_id, next_episodes_info[i].episode_id)]
                    == evals_per_ep
                ):
                    envs_to_pause.append(i)
                
                disp_info = {k: v for k, v in infos[i].items() if k not in rank0_keys}
                
                if len(config.habitat_baselines.eval.video_option) > 0:
                    frame = observations_to_image({k: v[i] for k, v in batch.items()}, disp_info)
                    overlay_text = format_streamvln_debug_overlay(streamvln_step_debug[i])
                    if overlay_text:
                        # overlay_text_to_image expects a list of strings
                        text_lines = overlay_text.split('\n')
                        frame = overlay_text_to_image(frame, text_lines)
                    frame = overlay_frame(frame, disp_info)
                    frame = _annotate_frame_with_instruction(
                        frame, streamvln_step_debug[i]
                    )
                    if dones[i]:
                        final_frame = observations_to_image(
                            {k: v[i] * 0.0 for k, v in batch.items()}, disp_info
                        )
                        final_frame = overlay_frame(final_frame, disp_info)
                        final_frame = _annotate_frame_with_instruction(
                            final_frame, streamvln_step_debug[i]
                        )
                        rgb_frames[i].append(final_frame)
                        rgb_frames[i].append(frame)
                    else:
                        rgb_frames[i].append(frame)
                
                # Episode结束
                if dones[i]:
                    pbar.update()
                    if "success" in disp_info:
                        success_cal += disp_info['success']
                        logger.info(
                            f"Till now Success Rate: {success_cal/(len(stats_episodes)+1):.4f}"
                        )
                    
                    episode_stats = {"reward": current_episode_reward[i].item()}
                    episode_stats.update(extract_scalars_from_info(infos[i]))
                    current_episode_reward[i] = 0
                    k = (current_episodes_info[i].scene_id, current_episodes_info[i].episode_id)
                    ep_eval_count[k] += 1
                    stats_episodes[(k, ep_eval_count[k])] = episode_stats
                    
                    # 重置该环境的队列和StreamVLN状态
                    action_queues[i] = []
                    streamvln_step_debug[i] = None
                    # 重置StreamVLN的episode state（保留多轮对话机制）
                    streamvln_net.reset_episode_state()
                    
                    # Episode结束时清理内存
                    if clear_cache_on_episode_end:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        gc.collect()
                    
                    if len(config.habitat_baselines.eval.video_option) > 0:
                        scene_id = current_episodes_info[i].scene_id.split('/')[-1].split('.')[0]
                        logger.info(
                            f"Scene ID: {scene_id}, Episode ID: {current_episodes_info[i].episode_id}"
                        )
                        
                        metrics_for_video = extract_scalars_from_info(disp_info)
                        keys_to_include = getattr(
                            config.habitat_baselines.eval,
                            "keys_to_include_in_name",
                            None,
                        )
                        video_name = ""
                        try:
                            video_name = generate_video(
                                video_option=config.habitat_baselines.eval.video_option,
                                video_dir=config.habitat_baselines.video_dir,
                                images=rgb_frames[i][:-1],
                                scene_id=scene_id,
                                episode_id=f"{current_episodes_info[i].episode_id}_{ep_eval_count[k]}",
                                checkpoint_idx=checkpoint_index,
                                metrics=metrics_for_video,
                                fps=config.habitat_baselines.video_fps,
                                tb_writer=writer,
                                keys_to_include_in_name=keys_to_include,
                            )
                        except RuntimeError as exc:
                            logger.warning(
                                "Skipping video generation for env %d due to ffmpeg error: %s. "
                                "请确认系统已安装 ffmpeg 或设置 IMAGEIO_FFMPEG_EXE。",
                                i,
                                exc,
                            )
                            video_name = _build_video_basename(
                                scene_id=scene_id,
                                episode_id=f"{current_episodes_info[i].episode_id}_{ep_eval_count[k]}",
                                checkpoint_idx=checkpoint_index,
                                metrics=metrics_for_video,
                                keys_to_include_in_name=keys_to_include,
                            )
                        if video_name:
                            _write_instruction_log(
                                config.habitat_baselines.video_dir,
                                video_name,
                                episode_instruction_logs[i],
                            )
                        rgb_frames[i] = rgb_frames[i][-1:]
                    
                    gfx_str = infos[i].get(GfxReplayMeasure.cls_uuid, "")
                    if gfx_str != "":
                        write_gfx_replay(
                            gfx_str, config.habitat.task, current_episodes_info[i].episode_id
                        )
                    episode_instruction_logs[i] = []
                    episode_steps[i] = 0
            
            # 暂停环境
            if envs_to_pause:
                # 同时暂停队列
                action_queues = [action_queues[i] for i in range(n_envs) if i not in envs_to_pause]
                streamvln_step_debug = [streamvln_step_debug[i] for i in range(n_envs) if i not in envs_to_pause]
                episode_instruction_logs = [episode_instruction_logs[i] for i in range(n_envs) if i not in envs_to_pause]
                episode_steps = [episode_steps[i] for i in range(n_envs) if i not in envs_to_pause]
                
                not_done_masks = torch.tensor(
                    [[not done] for done in dones], dtype=torch.bool, device="cpu"
                )
                (
                    envs, _, not_done_masks, current_episode_reward, _, batch, rgb_frames,
                ) = pause_envs(
                    envs_to_pause, envs, None, not_done_masks, 
                    current_episode_reward, None, batch, rgb_frames,
                )
        
        pbar.close()
        
        # 最终清理内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        # 聚合统计信息
        aggregated_stats = {}
        all_ks = set()
        for ep in stats_episodes.values():
            all_ks.update(ep.keys())
        for stat_key in all_ks:
            aggregated_stats[stat_key] = np.mean(
                [v[stat_key] for v in stats_episodes.values() if stat_key in v]
            )
        
        for k, v in aggregated_stats.items():
            logger.info(f"Average episode {k}: {v:.4f}")
        
        writer.add_scalar("eval_reward/average_reward", aggregated_stats["reward"], step_id)
        
        metrics = {k: v for k, v in aggregated_stats.items() if k != "reward"}
        for k, v in metrics.items():
            writer.add_scalar(f"eval_metrics/{k}", v, step_id)
        
        # 保存结果
        result_path = os.path.join("output/", "result.json")
        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        evalai_result = {
            "SR": round(aggregated_stats.get("success", 0), 4),
            "SPL": round(aggregated_stats.get("spl", 0), 4),
            "PSC": round(aggregated_stats.get("psc", 0), 4),
            "H-Coll": round(aggregated_stats.get("human_collision", 0), 4),
            "Total": round(
                0.4 * aggregated_stats.get("success", 0)
                + 0.3 * aggregated_stats.get("spl", 0)
                + 0.3 * aggregated_stats.get("psc", 0),
                4,
            ),
        }
        
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(evalai_result, f, indent=2, ensure_ascii=False)
        
        # 保存动作记录
        actions_output_path = os.path.join("output/", "actions.json")
        serializable_actions = {
            f"{scene_id}|{episode_id}|{eval_count}": actions
            for (scene_id, episode_id, eval_count), actions in actions_record.items()
        }
        with open(actions_output_path, "w", encoding="utf-8") as f:
            json.dump(serializable_actions, f, indent=2, ensure_ascii=False)
    
    def _generate_streamvln_action(
        self, batch, env_idx, current_episode, streamvln_net, 
        action_parser, action_queue, device, config=None, instruction_sensor_uuid=None
    ):
        """
        使用StreamVLN模型生成动作
        
        Args:
            batch: 观察批次
            env_idx: 环境索引
            current_episode: 当前episode信息
            streamvln_net: StreamVLN网络
            action_parser: 动作解析器
            action_queue: 动作队列（用于存储多步骤动作）
            device: 设备
            instruction_sensor_uuid: 指令传感器UUID（可选）
            
        Returns:
            action: 动作ID (0-3)
            debug_info: 调试信息字典
        """
        # 获取当前RGB（使用_find_rgb_key查找RGB传感器，仿照NaVILA的实现）
        rgb_key = _find_rgb_key(batch, config)
        if rgb_key is None:
            # 如果找不到RGB，尝试使用depth作为fallback
            rgb_key = 'agent_0_articulated_agent_jaw_depth' if 'agent_0_articulated_agent_jaw_depth' in batch else 'depth'
            if rgb_key not in batch:
                available_keys = list(batch.keys())
                raise ValueError(
                    f"No RGB key found in batch. Available keys: {available_keys}. "
                    f"Please ensure at least one RGB sensor is enabled in the config "
                    f"(e.g., agent_0_overhead_front_rgb or agent_0_articulated_agent_jaw_rgb)."
                )
        
        rgb_obs = batch[rgb_key][env_idx]
        rgb_np = rgb_obs.cpu().numpy()
        if rgb_np.dtype != np.uint8:
            if rgb_np.max() <= 1.0:
                rgb_np = (rgb_np * 255).astype(np.uint8)
            else:
                rgb_np = rgb_np.astype(np.uint8)
        
        # 获取深度图像（参考 streamvln_eval.py 的实现）
        depth_key = 'agent_0_articulated_agent_jaw_depth' if 'agent_0_articulated_agent_jaw_depth' in batch else 'depth'
        depth_np = None
        if depth_key in batch:
            depth_obs = batch[depth_key][env_idx]
            depth_np = depth_obs.cpu().numpy()
            # 如果深度图像是单通道，确保形状正确
            if len(depth_np.shape) == 2:
                depth_np = depth_np.reshape(depth_np.shape[0], depth_np.shape[1], 1)
            elif len(depth_np.shape) == 3 and depth_np.shape[2] > 1:
                # 如果是多通道，取第一个通道
                depth_np = depth_np[:, :, 0:1]
        
        # 获取指令
        env_observations: Dict[str, Any] = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                env_observations[key] = value[env_idx]
            else:
                env_observations[key] = value

        episode_instruction = getattr(current_episode, "instruction", None)
        if episode_instruction is not None and not isinstance(episode_instruction, str):
            episode_instruction = getattr(episode_instruction, "instruction_text", None) or str(episode_instruction)

        instruction = extract_streamvln_instruction(
            env_observations,
            instruction_sensor_uuid=instruction_sensor_uuid,
            episode_instruction=episode_instruction,
        )
        
        # 使用StreamVLN的generate_action_sequence方法生成动作序列
        # 这个方法保留了StreamVLN的多轮对话机制
        try:
            # 使用torch.no_grad()减少内存占用（评估模式下不需要梯度）
            with torch.no_grad():
                action_seq, llm_output = streamvln_net.generate_action_sequence(
                    rgb=rgb_np,
                    depth=depth_np,  # 传递深度图像
                    instruction=instruction,
                    env_idx=env_idx,
                    run_model=True
                )
            
            logger.info(f"StreamVLN output: {llm_output}")
            
            # 生成动作后立即清理中间变量
            del rgb_np
            if depth_np is not None:
                del depth_np
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # ✅ 完全按照 streamvln_eval.py line 342 的逻辑
            # 参考代码：action = action_seq.pop(0)
            # 注意：action_seq 已经是完整的动作序列，需要取第一个并将其余加入队列
            if len(action_seq) > 0:
                action = int(action_seq[0])
                # 将后续动作加入队列
                for next_action in action_seq[1:]:
                    action_queue.append(int(next_action))
                logger.info(
                    f"[StreamVLN][env {env_idx}] 生成动作序列长度={len(action_seq)}, "
                    f"执行第1个动作={action}, 队列剩余={len(action_queue)}"
                )
            else:
                # 如果没有生成动作，使用默认动作
                action = 0  # STOP
                logger.warning(f"[StreamVLN] No actions in sequence, using STOP")
            
            debug_info = {
                "instruction": instruction,
                "model_output": llm_output,
                "action_id": int(action),
                "action_name": self.ACTION_ID_TO_NAME.get(action, f"action_{action}"),
                "repeats": 1,
                "repeat_index": 1,
                "from_queue": False,
            }
            
            logger.info(
                "[StreamVLN][env %d] action=%s action_seq=%s queue_len=%d instruction=\"%s\" llm=\"%s\"",
                env_idx,
                debug_info["action_name"],
                action_seq,  # 打印动作序列
                len(action_queue),  # 打印队列长度
                instruction,  # 打印完整instruction
                llm_output,  # 打印完整llm_output
            )
            
            return action, debug_info
            
        except Exception as e:
            logger.warning(f"StreamVLN action generation failed: {e}")
            # 回退：使用默认动作
            action = 1  # MOVE_FORWARD
            debug_info = {
                "instruction": instruction,
                "model_output": f"Error: {str(e)}",
                "action_id": int(action),
                "action_name": self.ACTION_ID_TO_NAME.get(action, f"action_{action}"),
                "repeats": 1,
                "repeat_index": 1,
                "from_queue": False,
            }
            return action, debug_info

