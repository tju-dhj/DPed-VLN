#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
NaVILA Evaluator for Falcon Framework
专门处理NaVILA策略的评估器，支持语言指令到动作的转换
"""

import copy
import datetime
import json
import os
import textwrap
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

# NaVILA相关导入
try:
    from habitat_baselines.rl.ddppo.policy.navila.action_parser import NaVILAActionParser
    from habitat_baselines.rl.ddppo.policy.navila_policy import (
        ACTION_ID_TO_NAME,
        extract_navila_instruction,
        format_navila_debug_overlay,
        sample_and_pad_images,
    )
    
    from habitat_baselines.rl.ddppo.policy.navila.llava.constants import (
        IMAGE_TOKEN_INDEX,
    )
    from habitat_baselines.rl.ddppo.policy.navila.llava.conversation import (
        SeparatorStyle,
        conv_templates,
    )
    from habitat_baselines.rl.ddppo.policy.navila.llava.mm_utils import (
        KeywordsStoppingCriteria,
        process_images,
        tokenizer_image_token,
    )
    from habitat_baselines.rl.ddppo.policy.navila.llava.model.builder import (
        load_pretrained_model,
    )
    NAVILA_AVAILABLE = True
except ImportError as e:
    NAVILA_AVAILABLE = False
    print(f"Warning: NaVILA modules not available: {e}")


def _find_rgb_key(batch: Dict[str, Any], config=None) -> Optional[str]:
    """
    查找可用的RGB key
    
    优先顺序：
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
        # Fallback to default bitmap font if truetype not available
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
    with open(json_path, "w") as f:
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


class NaVILAEvaluator(FALCONEvaluator):
    """
    NaVILA专用评估器
    
    继承自FALCONEvaluator，但使用NaVILA的语言模型直接生成动作，
    而不是通过policy网络。
    """
    
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
        评估NaVILA agent
        
        该方法重写父类方法，直接使用LLAVA模型生成动作指令。
        """
        if not NAVILA_AVAILABLE:
            raise ImportError("NaVILA modules are required for NaVILAEvaluator")
        
        # 尝试从agent的policy中获取已经加载的模型，避免重复加载
        navila_config = config.habitat_baselines.rl.policy.agent_0
        tokenizer = None
        model = None
        image_processor = None
        context_len = None
        
        # 检查agent的policy是否是NaVILAPolicy，如果是，则复用已加载的模型
        try:
            # 访问路径：agent.actor_critic.net (NetPolicy.net -> NaVILANet)
            # 对于MultiAgentAccessMgr，需要从_agents[0]获取agent_0
            net = None
            target_agent = agent
            
            # 如果是MultiAgentAccessMgr，从_agents[0]获取agent_0
            if hasattr(agent, '_agents') and len(agent._agents) > 0:
                target_agent = agent._agents[0]
                logger.info(f"Multi-agent detected, using agent_0 from _agents[0]")
            
            if hasattr(target_agent, 'actor_critic'):
                actor_critic = target_agent.actor_critic
                # NetPolicy有net属性，直接访问
                if hasattr(actor_critic, 'net'):
                    net = actor_critic.net
                    logger.info(f"Found net in agent.actor_critic.net: {type(net).__name__}")
            
            # 如果找到了net，检查是否是NaVILANet（有model、tokenizer、image_processor属性）
            if net is not None:
                if hasattr(net, 'model') and hasattr(net, 'tokenizer') and hasattr(net, 'image_processor'):
                    # 找到了已加载的模型，复用它们
                    tokenizer = net.tokenizer
                    model = net.model
                    image_processor = net.image_processor
                    context_len = getattr(net, 'context_len', None)
                    logger.info("Reusing NaVILA model from policy (avoiding duplicate loading)")
                else:
                    logger.warning(f"Found net but missing required attributes. Net type: {type(net).__name__}, has model: {hasattr(net, 'model')}, has tokenizer: {hasattr(net, 'tokenizer')}, has image_processor: {hasattr(net, 'image_processor')}")
            else:
                logger.warning(f"Could not find net in agent structure. Agent type: {type(agent).__name__}, target_agent type: {type(target_agent).__name__}, has actor_critic: {hasattr(target_agent, 'actor_critic')}. Will load model separately.")
        except Exception as e:
            logger.warning(f"Could not reuse model from policy: {e}. Will load model separately.", exc_info=True)
        
        # 如果无法从policy获取模型，则重新加载
        if model is None or tokenizer is None or image_processor is None:
            model_path = navila_config.get("model_path", None)
            
            if model_path is None or not os.path.exists(model_path):
                raise ValueError(
                    f"NaVILA model path is not provided or does not exist: {model_path}"
                )
            
            logger.info(f"Loading NaVILA model from {model_path}")
            model_name = os.path.basename(os.path.normpath(model_path))
            tokenizer, model, image_processor, context_len = load_pretrained_model(
                model_path, model_name
            )
            model = model.to(device)
            model.eval()
        else:
            # 确保模型在正确的设备上
            # 检查模型是否使用了 device_map（分散在多个设备上）
            has_device_map = hasattr(model, 'hf_device_map') and model.hf_device_map is not None
            
            # 打印详细的设备信息用于调试
            logger.info("=" * 80)
            logger.info("DEBUG: Device information for NaVILA model:")
            logger.info(f"  Target device: {device}")
            logger.info(f"  Model has hf_device_map: {has_device_map}")
            if has_device_map:
                logger.info(f"  Model hf_device_map: {model.hf_device_map}")
            
            # 检查模型各个部分的设备
            # try:
            #     if hasattr(model, 'llm'):
            #         llm_device = next(model.llm.parameters()).device
            #         logger.info(f"  Model.llm device: {llm_device}")
            #     if hasattr(model, 'vision_tower'):
            #         vision_device = next(model.vision_tower.parameters()).device
            #         logger.info(f"  Model.vision_tower device: {vision_device}")
            #     if hasattr(model, 'mm_projector'):
            #         projector_device = next(model.mm_projector.parameters()).device
            #         logger.info(f"  Model.mm_projector device: {projector_device}")
            #     # 检查所有参数设备
            #     param_devices = set()
            #     for name, param in model.named_parameters():
            #         param_devices.add(str(param.device))
            #     logger.info(f"  All parameter devices: {param_devices}")
            # except Exception as e:
            #     logger.warning(f"  Could not check model device details: {e}")
            
            if has_device_map:
                # 如果模型使用了 device_map，需要将所有部分移动到单个设备
                # 这对于避免设备不匹配错误很重要
                logger.warning(f"Model uses device_map (distributed across devices). Moving to single device {device}.")
                # 尝试将模型移动到单个设备
                try:
                    # 对于使用 device_map 的模型，需要先禁用 device_map
                    # 注意：不能设置为 None，因为 transformers 代码会访问 .values()
                    # 应该设置为空字典 {}，表示没有设备映射
                    if hasattr(model, 'hf_device_map'):
                        model.hf_device_map = {}
                    model = model.to(device)
                    logger.info(f"  Model moved to device: {device}")
                except Exception as e:
                    logger.warning(f"Failed to move model to single device: {e}. Will try to ensure inputs are on correct device.")
            else:
                # 普通模型，直接移动
                try:
                    model_device = next(model.parameters()).device
                    logger.info(f"  Model device (from first parameter): {model_device}")
                    if model_device != device:
                        logger.info(f"  Moving model from {model_device} to {device}")
                        model = model.to(device)
                        model_device = next(model.parameters()).device
                        logger.info(f"  Model device after move: {model_device}")
                except (StopIteration, AttributeError):
                    # 如果无法获取设备，尝试直接移动
                    try:
                        logger.info(f"  Could not get model device from parameters, trying direct move to {device}")
                        model = model.to(device)
                    except Exception as e:
                        logger.warning(f"Could not move model to device {device}: {e}")
            logger.info("=" * 80)
        
        instruction_sensor_uuid = navila_config.get("instruction_sensor_uuid", None)
        # 初始化动作解析器
        action_parser = NaVILAActionParser(
            forward_step=navila_config.get("forward_step", 25),
            turn_step=navila_config.get("turn_step", 15),
        )
        
        num_video_frames = navila_config.get("num_video_frames", 8)
        
        success_cal = 0
        observations = envs.reset()
        observations = envs.post_step(observations)
        batch = batch_obs(observations, device=device)
        batch = apply_obs_transforms_batch(batch, obs_transforms)
        
        action_shape, discrete_actions = get_action_space_info(
            agent.actor_critic.policy_action_space
        )
        hidden_state_lens = agent.actor_critic.hidden_state_shape_lens
        action_space_lens = agent.actor_critic.policy_action_space_shape_lens
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
                f"NaVILA expects agent_0 action length 1, but got {agent0_width}. "
                "Please ensure agent_0 policy is discrete."
            )
        agent0_start = 0
        agent0_end = agent0_start + agent0_width

        current_episode_reward = torch.zeros(envs.num_envs, 1, device="cpu")
        test_recurrent_hidden_states = torch.zeros(
            (
                config.habitat_baselines.num_environments,
                *agent.actor_critic.hidden_state_shape,
            ),
            device=device,
        )
        prev_actions = torch.zeros(
            config.habitat_baselines.num_environments,
            *action_shape,
            device=device,
            dtype=torch.long if discrete_actions else torch.float,
        )
        not_done_masks = torch.zeros(
            config.habitat_baselines.num_environments,
            *agent.masks_shape,
            device=device,
            dtype=torch.bool,
        )
        
        stats_episodes: Dict[Any, Any] = {}
        ep_eval_count: Dict[Any, int] = defaultdict(lambda: 0)
        
        # 历史RGB帧缓存（每个环境一个）
        past_rgbs = [[] for _ in range(envs.num_envs)]
        
        # 动作队列（每个环境一个）
        action_queues = [[] for _ in range(envs.num_envs)]
        navila_step_debug: List[Optional[Dict[str, Any]]] = [None for _ in range(envs.num_envs)]
        
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
        
        while (
            len(stats_episodes) < (number_of_eval_episodes * evals_per_ep)
            and envs.num_envs > 0
        ):
            current_episodes_info = envs.current_episodes()
            
            space_lengths: Dict[str, Any] = {}
            n_agents = len(config.habitat.simulator.agents)
            if n_agents > 1:
                space_lengths = {
                    "index_len_recurrent_hidden_states": hidden_state_lens,
                    "index_len_prev_actions": action_space_lens,
                }
            with inference_mode():
                action_data = agent.actor_critic.act(
                    batch,
                    test_recurrent_hidden_states,
                    prev_actions,
                    not_done_masks,
                    deterministic=False,
                    **space_lengths,
                )
                if action_data.should_inserts is None:
                    test_recurrent_hidden_states = action_data.rnn_hidden_states
                    prev_actions.copy_(action_data.actions)  # type: ignore
                else:
                    agent.actor_critic.update_hidden_state(
                        test_recurrent_hidden_states, prev_actions, action_data
                    )

            base_actions = (
                action_data.env_actions.detach().cpu().numpy().astype(np.float32, copy=True)
            )

            # 为每个环境生成动作（仅覆盖agent_0的离散动作，其余代理沿用actor输出）
            navila_actions: List[int] = []
            for i in range(envs.num_envs):
                # 如果动作队列中有动作，直接使用
                if len(action_queues[i]) > 0:
                    queued_entry = action_queues[i].pop(0)
                    if isinstance(queued_entry, dict):
                        action = int(queued_entry.get("action", 0))
                        navila_step_debug[i] = queued_entry.get("debug")
                    else:
                        action = int(queued_entry)
                        navila_step_debug[i] = None
                    action_name = (
                        navila_step_debug[i].get("action_name")
                        if navila_step_debug[i]
                        else str(action)
                    )
                    # logger.info(f"Env {i}: Using queued action {action_name}")  # silenced: per-step verbose
                else:
                    # 否则，使用NaVILA生成新动作
                    action, debug_info = self._generate_navila_action(
                        batch,
                        i,
                        past_rgbs[i],
                        num_video_frames,
                        current_episodes_info[i],
                        model,
                        tokenizer,
                        image_processor,
                        action_parser,
                        action_queues[i],
                        device,
                        instruction_sensor_uuid=instruction_sensor_uuid,
                    )
                    navila_step_debug[i] = debug_info

                if navila_step_debug[i]:
                    episode_instruction_logs[i].append(dict(navila_step_debug[i]))
                navila_actions.append(action)
                base_actions[i, agent0_start:agent0_end] = float(action)
            
            for i in range(envs.num_envs):
                action_name = ACTION_ID_TO_NAME.get(navila_actions[i], str(navila_actions[i]))
                # logger.info(
                #     "[NaVILA][eval] Env %d executing action: %s (%d)",
                #     i,
                #     action_name,
                #     navila_actions[i],
                # )  # silenced: per-step verbose
            
            step_data = [base_actions[i].copy() for i in range(envs.num_envs)]
            
            # 执行动作
            outputs = envs.step(step_data)
            observations, rewards_l, dones, infos = [list(x) for x in zip(*outputs)]
            
            # 记录动作
            for i in range(envs.num_envs):
                episode_steps[i] += 1
                if (
                    max_episode_steps > 0
                    and episode_steps[i] >= max_episode_steps
                    and not dones[i]
                ):
                    # logger.info(
                    #     "[NaVILA][eval] Env %d reached max steps (%d). Forcing episode end.",
                    #     i,
                    #     max_episode_steps,
                    # )  # silenced: per-step verbose
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
                    "value": int(navila_actions[i]),
                }
                debug_info = navila_step_debug[i]
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
            not_done_masks = torch.tensor(
                [[not done] for done in dones],
                dtype=torch.bool,
                device="cpu",
            ).repeat(1, *agent.masks_shape)
            not_done_masks = not_done_masks.to(device=device)
            
            # 添加当前RGB到历史
            rgb_key = _find_rgb_key(batch, config)
            if rgb_key is not None:
                for i in range(envs.num_envs):
                    curr_rgb = Image.fromarray(
                        np.uint8(batch[rgb_key][i].cpu().numpy())
                    ).convert("RGB")
                    past_rgbs[i].append(curr_rgb)
            else:
                logger.warning(f"No RGB key found in batch. Available keys: {list(batch.keys())}")
            
            rewards = torch.tensor(rewards_l, dtype=torch.float, device="cpu").unsqueeze(1)
            current_episode_reward += rewards
            next_episodes_info = envs.current_episodes()
            envs_to_pause = []
            n_envs = envs.num_envs
            
            for i in range(n_envs):
                if (
                    ep_eval_count[(next_episodes_info[i].scene_id, next_episodes_info[i].episode_id)]
                    == evals_per_ep
                ):
                    envs_to_pause.append(i)
                
                disp_info = {k: v for k, v in infos[i].items() if k not in rank0_keys}
                
                if len(config.habitat_baselines.eval.video_option) > 0:
                    frame = observations_to_image({k: v[i] for k, v in batch.items()}, disp_info)
                    # 使用 overlay_frame 的 additional 参数来添加 NaVILA 调试信息
                    additional_lines = None
                    if navila_step_debug[i] is not None:
                        overlay_text = format_navila_debug_overlay(navila_step_debug[i])
                        if overlay_text:
                            # format_navila_debug_overlay 返回字符串，可能包含多行
                            # 将其拆分成行列表
                            if isinstance(overlay_text, str):
                                # 按换行符拆分，并过滤空行
                                additional_lines = [line.strip() for line in overlay_text.split('\n') if line.strip()]
                            else:
                                additional_lines = [str(overlay_text)]
                    frame = overlay_frame(frame, disp_info, additional=additional_lines)
                    frame = _annotate_frame_with_instruction(
                        frame, navila_step_debug[i]
                    )
                    if dones[i]:
                        final_frame = observations_to_image(
                            {k: v[i] * 0.0 for k, v in batch.items()}, disp_info
                        )
                        final_frame = overlay_frame(final_frame, disp_info)
                        final_frame = _annotate_frame_with_instruction(
                            final_frame, navila_step_debug[i]
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
                    
                    metrics_for_video = extract_scalars_from_info(infos[i])
                    episode_stats = {"reward": current_episode_reward[i].item()}
                    episode_stats.update(metrics_for_video)
                    current_episode_reward[i] = 0
                    k = (current_episodes_info[i].scene_id, current_episodes_info[i].episode_id)
                    ep_eval_count[k] += 1
                    stats_episodes[(k, ep_eval_count[k])] = episode_stats
                    
                    # 重置该环境的历史和队列
                    past_rgbs[i] = []
                    action_queues[i] = []
                    navila_step_debug[i] = None
                    
                    if len(config.habitat_baselines.eval.video_option) > 0:
                        scene_id = current_episodes_info[i].scene_id.split('/')[-1].split('.')[0]
                        # logger.info(
                        #     f"Scene ID: {scene_id}, Episode ID: {current_episodes_info[i].episode_id}"
                        # )  # silenced: per-episode verbose
                        
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
                # 同时暂停历史和队列
                active_indices = [i for i in range(n_envs) if i not in envs_to_pause]
                past_rgbs = [past_rgbs[i] for i in active_indices]
                action_queues = [action_queues[i] for i in active_indices]
                navila_step_debug = [navila_step_debug[i] for i in active_indices]
                episode_instruction_logs = [episode_instruction_logs[i] for i in active_indices]
                episode_steps = [episode_steps[i] for i in active_indices]
                
                (
                    envs,
                    test_recurrent_hidden_states,
                    not_done_masks,
                    current_episode_reward,
                    prev_actions,
                    batch,
                    rgb_frames,
                ) = pause_envs(
                    envs_to_pause,
                    envs,
                    test_recurrent_hidden_states,
                    not_done_masks,
                    current_episode_reward,
                    prev_actions,
                    batch,
                    rgb_frames,
                )
        
        pbar.close()
        
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
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_root = os.path.join("navila-output")
        os.makedirs(output_root, exist_ok=True)
        result_path = os.path.join(output_root, f"result_{timestamp}.json")
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
        
        evalai_result["generated_at"] = timestamp
        with open(result_path, "w") as f:
            json.dump(evalai_result, f, indent=2)
        
        # 保存动作记录
        actions_output_path = os.path.join(output_root, f"actions_{timestamp}.json")
        serializable_actions = {
            f"{scene_id}|{episode_id}|{eval_count}": actions
            for (scene_id, episode_id, eval_count), actions in actions_record.items()
        }
        with open(actions_output_path, "w") as f:
            json.dump(serializable_actions, f, indent=2)
    
    def _generate_navila_action(
        self, batch, env_idx, past_rgbs, num_video_frames,
        current_episode, model, tokenizer, image_processor, 
        action_parser, action_queue, device, instruction_sensor_uuid=None
    ):
        """
        使用NaVILA模型生成动作
        
        Args:
            batch: 观察批次
            env_idx: 环境索引
            past_rgbs: 历史RGB帧列表
            num_video_frames: 视频帧数
            current_episode: 当前episode信息
            model: LLAVA模型
            tokenizer: tokenizer
            image_processor: 图像处理器
            action_parser: 动作解析器
            action_queue: 动作队列（用于存储多步骤动作）
            device: 设备
            instruction_sensor_uuid: 指令传感器UUID（可选）
            
        Returns:
            action: 动作ID (0-3)
        """
        # 获取当前RGB
        rgb_key = _find_rgb_key(batch)
        if rgb_key is None:
            available_keys = list(batch.keys())
            raise ValueError(
                f"No RGB key found in batch. Available keys: {available_keys}. "
                f"Please ensure at least one RGB sensor is enabled in the config "
                f"(e.g., agent_0_overhead_front_rgb or agent_0_articulated_agent_jaw_rgb)."
            )
        
        curr_rgb = Image.fromarray(
            np.uint8(batch[rgb_key][env_idx].cpu().numpy())
        ).convert("RGB")
        
        # 构建视频序列
        past_and_current_rgbs = past_rgbs + [curr_rgb]
        sampled_frames = sample_and_pad_images(
            past_and_current_rgbs, num_frames=num_video_frames
        )
        
        # 获取指令（如果有的话）
        env_observations: Dict[str, Any] = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                env_observations[key] = value[env_idx]
            else:
                env_observations[key] = value

        episode_instruction = getattr(current_episode, "instruction", None)
        if episode_instruction is not None and not isinstance(episode_instruction, str):
            episode_instruction = getattr(episode_instruction, "instruction_text", None) or str(episode_instruction)

        instruction = extract_navila_instruction(
            env_observations,
            instruction_sensor_uuid=instruction_sensor_uuid,
            episode_instruction=episode_instruction,
        )
        
        # 构建提示
        interleaved_images = "<image>\n" * (len(sampled_frames) - 1)
        question = (
            f"Imagine you are a robot programmed for navigation tasks. You have been given a video "
            f'of historical observations {interleaved_images}, and current observation <image>\n. '
            f'Your assigned task is: "{instruction}" '
            f"Analyze this series of images to decide your next action, which could be turning left or right by a specific "
            f"degree, moving forward a certain distance, or stop if the task is completed."
        )
        
        # 构建对话
        conv_mode = "llama_3"
        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        
        # 处理图像
        # 确保图像在正确的设备上（与模型相同的设备）
        try:
            model_device = next(model.parameters()).device
        except (StopIteration, AttributeError):
            model_device = device
        
        # 打印设备信息用于调试
        # logger.info("=" * 80)
        # logger.info("DEBUG: Device information before processing inputs:")
        # logger.info(f"  Target device: {device}")
        # logger.info(f"  Model device: {model_device}")
        
        images_tensor = process_images(
            sampled_frames, image_processor, model.config
        )
        # logger.info(f"  Images tensor device (before move): {images_tensor.device}")
        images_tensor = images_tensor.to(model_device, dtype=torch.float16)
        # logger.info(f"  Images tensor device (after move): {images_tensor.device}")
        
        # Tokenize
        # 确保 input_ids 在正确的设备上
        input_ids = (
            tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
        )
        # logger.info(f"  Input IDs device (before move): {input_ids.device}")
        input_ids = input_ids.to(model_device)
        # logger.info(f"  Input IDs device (after move): {input_ids.device}")
        # logger.info("=" * 80)
        
        # 停止条件
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        
        # 生成输出
        # 启用KV缓存以加速推理（预期5-10倍加速）
        # KV缓存在单次生成过程中可以避免重复计算已生成token的key-value
        # 注意：如果遇到兼容性问题，可以回退到use_cache=False
        # 确保所有输入都在模型所在的设备上
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=images_tensor.half().to(model_device),  # 使用 model_device 而不是 device
                do_sample=False,
                temperature=0.0,
                max_new_tokens=32,
                use_cache=True,  # 启用KV缓存以加速推理（预期5-10倍加速）
                stopping_criteria=[stopping_criteria],
                pad_token_id=tokenizer.eos_token_id,
            )
        
        # 解码
        output_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        if output_text.endswith(stop_str):
            output_text = output_text[: -len(stop_str)].strip()
        
        # logger.info(f"NaVILA output: {output_text}")  # silenced: per-step verbose
        
        # 解析动作
        action, num_repeats = action_parser.parse_action(output_text)
        debug_info = {
            "instruction": instruction,
            "model_output": output_text,
            "action_id": int(action),
            "action_name": ACTION_ID_TO_NAME.get(action, f"action_{action}"),
            "repeats": int(max(1, num_repeats)),
            "repeat_index": 1,
            "from_queue": False,
        }
        # logger.info(
        #     "[NaVILA][env %d] action=%s repeats=%d instruction=\"%s\" llm=\"%s\"",
        #     env_idx,
        #     debug_info["action_name"],
        #     debug_info["repeats"],
        #     instruction[:200],
        #     output_text[:200],
        # )  # silenced: per-step verbose
        # 将后续动作加入队列
        if num_repeats > 1:
            for repeat_idx in range(2, num_repeats + 1):
                queued_debug = dict(debug_info)
                queued_debug["repeat_index"] = repeat_idx
                queued_debug["from_queue"] = True
                action_queue.append({"action": action, "debug": queued_debug})
            # logger.info(f"Added {num_repeats - 1} actions to queue. Queue length: {len(action_queue)}")  # silenced: per-step verbose
        
        return action, debug_info

#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
NaVILA Evaluator for Falcon Framework
专门处理NaVILA策略的评估器，支持语言指令到动作的转换
"""

import copy
import datetime
import json
import os
import textwrap
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

# NaVILA相关导入
try:
    from habitat_baselines.rl.ddppo.policy.navila.action_parser import NaVILAActionParser
    from habitat_baselines.rl.ddppo.policy.navila_policy import (
        ACTION_ID_TO_NAME,
        extract_navila_instruction,
        format_navila_debug_overlay,
        sample_and_pad_images,
    )
    
    from habitat_baselines.rl.ddppo.policy.navila.llava.constants import (
        IMAGE_TOKEN_INDEX,
    )
    from habitat_baselines.rl.ddppo.policy.navila.llava.conversation import (
        SeparatorStyle,
        conv_templates,
    )
    from habitat_baselines.rl.ddppo.policy.navila.llava.mm_utils import (
        KeywordsStoppingCriteria,
        process_images,
        tokenizer_image_token,
    )
    from habitat_baselines.rl.ddppo.policy.navila.llava.model.builder import (
        load_pretrained_model,
    )
    NAVILA_AVAILABLE = True
except ImportError as e:
    NAVILA_AVAILABLE = False
    print(f"Warning: NaVILA modules not available: {e}")


def _find_rgb_key(batch: Dict[str, Any], config=None) -> Optional[str]:
    """
    查找可用的RGB key
    
    优先顺序：
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
        # Fallback to default bitmap font if truetype not available
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
    with open(json_path, "w") as f:
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


class NaVILAEvaluator(FALCONEvaluator):
    """
    NaVILA专用评估器
    
    继承自FALCONEvaluator，但使用NaVILA的语言模型直接生成动作，
    而不是通过policy网络。
    """
    
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
        评估NaVILA agent
        
        该方法重写父类方法，直接使用LLAVA模型生成动作指令。
        """
        if not NAVILA_AVAILABLE:
            raise ImportError("NaVILA modules are required for NaVILAEvaluator")
        
        # 尝试从agent的policy中获取已经加载的模型，避免重复加载
        navila_config = config.habitat_baselines.rl.policy.agent_0
        tokenizer = None
        model = None
        image_processor = None
        context_len = None
        
        # 检查agent的policy是否是NaVILAPolicy，如果是，则复用已加载的模型
        try:
            # 访问路径：agent.actor_critic.net (NetPolicy.net -> NaVILANet)
            # 对于MultiAgentAccessMgr，需要从_agents[0]获取agent_0
            net = None
            target_agent = agent
            
            # 如果是MultiAgentAccessMgr，从_agents[0]获取agent_0
            if hasattr(agent, '_agents') and len(agent._agents) > 0:
                target_agent = agent._agents[0]
                logger.info(f"Multi-agent detected, using agent_0 from _agents[0]")
            
            if hasattr(target_agent, 'actor_critic'):
                actor_critic = target_agent.actor_critic
                # NetPolicy有net属性，直接访问
                if hasattr(actor_critic, 'net'):
                    net = actor_critic.net
                    logger.info(f"Found net in agent.actor_critic.net: {type(net).__name__}")
            
            # 如果找到了net，检查是否是NaVILANet（有model、tokenizer、image_processor属性）
            if net is not None:
                if hasattr(net, 'model') and hasattr(net, 'tokenizer') and hasattr(net, 'image_processor'):
                    # 找到了已加载的模型，复用它们
                    tokenizer = net.tokenizer
                    model = net.model
                    image_processor = net.image_processor
                    context_len = getattr(net, 'context_len', None)
                    logger.info("Reusing NaVILA model from policy (avoiding duplicate loading)")
                else:
                    logger.warning(f"Found net but missing required attributes. Net type: {type(net).__name__}, has model: {hasattr(net, 'model')}, has tokenizer: {hasattr(net, 'tokenizer')}, has image_processor: {hasattr(net, 'image_processor')}")
            else:
                logger.warning(f"Could not find net in agent structure. Agent type: {type(agent).__name__}, target_agent type: {type(target_agent).__name__}, has actor_critic: {hasattr(target_agent, 'actor_critic')}. Will load model separately.")
        except Exception as e:
            logger.warning(f"Could not reuse model from policy: {e}. Will load model separately.", exc_info=True)
        
        # 如果无法从policy获取模型，则重新加载
        if model is None or tokenizer is None or image_processor is None:
            model_path = navila_config.get("model_path", None)
            
            if model_path is None or not os.path.exists(model_path):
                raise ValueError(
                    f"NaVILA model path is not provided or does not exist: {model_path}"
                )
            
            logger.info(f"Loading NaVILA model from {model_path}")
            model_name = os.path.basename(os.path.normpath(model_path))
            tokenizer, model, image_processor, context_len = load_pretrained_model(
                model_path, model_name
            )
            model = model.to(device)
            model.eval()
        else:
            # 确保模型在正确的设备上
            # 检查模型是否使用了 device_map（分散在多个设备上）
            has_device_map = hasattr(model, 'hf_device_map') and model.hf_device_map is not None
            
            # 打印详细的设备信息用于调试
            logger.info("=" * 80)
            logger.info("DEBUG: Device information for NaVILA model:")
            logger.info(f"  Target device: {device}")
            logger.info(f"  Model has hf_device_map: {has_device_map}")
            if has_device_map:
                logger.info(f"  Model hf_device_map: {model.hf_device_map}")
            
            # 检查模型各个部分的设备
            # try:
            #     if hasattr(model, 'llm'):
            #         llm_device = next(model.llm.parameters()).device
            #         logger.info(f"  Model.llm device: {llm_device}")
            #     if hasattr(model, 'vision_tower'):
            #         vision_device = next(model.vision_tower.parameters()).device
            #         logger.info(f"  Model.vision_tower device: {vision_device}")
            #     if hasattr(model, 'mm_projector'):
            #         projector_device = next(model.mm_projector.parameters()).device
            #         logger.info(f"  Model.mm_projector device: {projector_device}")
            #     # 检查所有参数设备
            #     param_devices = set()
            #     for name, param in model.named_parameters():
            #         param_devices.add(str(param.device))
            #     logger.info(f"  All parameter devices: {param_devices}")
            # except Exception as e:
            #     logger.warning(f"  Could not check model device details: {e}")
            
            if has_device_map:
                # 如果模型使用了 device_map，需要将所有部分移动到单个设备
                # 这对于避免设备不匹配错误很重要
                logger.warning(f"Model uses device_map (distributed across devices). Moving to single device {device}.")
                # 尝试将模型移动到单个设备
                try:
                    # 对于使用 device_map 的模型，需要先禁用 device_map
                    # 注意：不能设置为 None，因为 transformers 代码会访问 .values()
                    # 应该设置为空字典 {}，表示没有设备映射
                    if hasattr(model, 'hf_device_map'):
                        model.hf_device_map = {}
                    model = model.to(device)
                    logger.info(f"  Model moved to device: {device}")
                except Exception as e:
                    logger.warning(f"Failed to move model to single device: {e}. Will try to ensure inputs are on correct device.")
            else:
                # 普通模型，直接移动
                try:
                    model_device = next(model.parameters()).device
                    logger.info(f"  Model device (from first parameter): {model_device}")
                    if model_device != device:
                        logger.info(f"  Moving model from {model_device} to {device}")
                        model = model.to(device)
                        model_device = next(model.parameters()).device
                        logger.info(f"  Model device after move: {model_device}")
                except (StopIteration, AttributeError):
                    # 如果无法获取设备，尝试直接移动
                    try:
                        logger.info(f"  Could not get model device from parameters, trying direct move to {device}")
                        model = model.to(device)
                    except Exception as e:
                        logger.warning(f"Could not move model to device {device}: {e}")
            logger.info("=" * 80)
        
        instruction_sensor_uuid = navila_config.get("instruction_sensor_uuid", None)
        # 初始化动作解析器
        action_parser = NaVILAActionParser(
            forward_step=navila_config.get("forward_step", 25),
            turn_step=navila_config.get("turn_step", 15),
        )
        
        num_video_frames = navila_config.get("num_video_frames", 8)
        
        success_cal = 0
        observations = envs.reset()
        observations = envs.post_step(observations)
        batch = batch_obs(observations, device=device)
        batch = apply_obs_transforms_batch(batch, obs_transforms)
        
        action_shape, discrete_actions = get_action_space_info(
            agent.actor_critic.policy_action_space
        )
        hidden_state_lens = agent.actor_critic.hidden_state_shape_lens
        action_space_lens = agent.actor_critic.policy_action_space_shape_lens
        if len(action_space_lens) == 0:
            raise ValueError("policy_action_space_shape_lens is empty, cannot map agent actions.")
        agent0_width = action_space_lens[0]
        if agent0_width != 1:
            raise ValueError(
                f"NaVILA expects agent_0 action length 1, but got {agent0_width}. "
                "Please ensure agent_0 policy is discrete."
            )
        agent0_start = 0
        agent0_end = agent0_start + agent0_width

        current_episode_reward = torch.zeros(envs.num_envs, 1, device="cpu")
        test_recurrent_hidden_states = torch.zeros(
            (
                config.habitat_baselines.num_environments,
                *agent.actor_critic.hidden_state_shape,
            ),
            device=device,
        )
        prev_actions = torch.zeros(
            config.habitat_baselines.num_environments,
            *action_shape,
            device=device,
            dtype=torch.long if discrete_actions else torch.float,
        )
        not_done_masks = torch.zeros(
            config.habitat_baselines.num_environments,
            *agent.masks_shape,
            device=device,
            dtype=torch.bool,
        )
        
        stats_episodes: Dict[Any, Any] = {}
        ep_eval_count: Dict[Any, int] = defaultdict(lambda: 0)
        
        # 历史RGB帧缓存（每个环境一个）
        past_rgbs = [[] for _ in range(envs.num_envs)]
        
        # 动作队列（每个环境一个）
        action_queues = [[] for _ in range(envs.num_envs)]
        navila_step_debug: List[Optional[Dict[str, Any]]] = [None for _ in range(envs.num_envs)]
        
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
        
        while (
            len(stats_episodes) < (number_of_eval_episodes * evals_per_ep)
            and envs.num_envs > 0
        ):
            current_episodes_info = envs.current_episodes()
            
            space_lengths: Dict[str, Any] = {}
            n_agents = len(config.habitat.simulator.agents)
            if n_agents > 1:
                space_lengths = {
                    "index_len_recurrent_hidden_states": hidden_state_lens,
                    "index_len_prev_actions": action_space_lens,
                }
            with inference_mode():
                action_data = agent.actor_critic.act(
                    batch,
                    test_recurrent_hidden_states,
                    prev_actions,
                    not_done_masks,
                    deterministic=False,
                    **space_lengths,
                )
                if action_data.should_inserts is None:
                    test_recurrent_hidden_states = action_data.rnn_hidden_states
                    prev_actions.copy_(action_data.actions)  # type: ignore
                else:
                    agent.actor_critic.update_hidden_state(
                        test_recurrent_hidden_states, prev_actions, action_data
                    )

            base_actions = (
                action_data.env_actions.detach().cpu().numpy().astype(np.float32, copy=True)
            )

            # 为每个环境生成动作（仅覆盖agent_0的离散动作，其余代理沿用actor输出）
            navila_actions: List[int] = []
            for i in range(envs.num_envs):
                # 如果动作队列中有动作，直接使用
                if len(action_queues[i]) > 0:
                    queued_entry = action_queues[i].pop(0)
                    if isinstance(queued_entry, dict):
                        action = int(queued_entry.get("action", 0))
                        navila_step_debug[i] = queued_entry.get("debug")
                    else:
                        action = int(queued_entry)
                        navila_step_debug[i] = None
                    action_name = (
                        navila_step_debug[i].get("action_name")
                        if navila_step_debug[i]
                        else str(action)
                    )
                    # logger.info(f"Env {i}: Using queued action {action_name}")  # silenced: per-step verbose
                else:
                    # 否则，使用NaVILA生成新动作
                    action, debug_info = self._generate_navila_action(
                        batch,
                        i,
                        past_rgbs[i],
                        num_video_frames,
                        current_episodes_info[i],
                        model,
                        tokenizer,
                        image_processor,
                        action_parser,
                        action_queues[i],
                        device,
                        instruction_sensor_uuid=instruction_sensor_uuid,
                    )
                    navila_step_debug[i] = debug_info

                if navila_step_debug[i]:
                    episode_instruction_logs[i].append(dict(navila_step_debug[i]))
                navila_actions.append(action)
                base_actions[i, agent0_start:agent0_end] = float(action)
            
            for i in range(envs.num_envs):
                action_name = ACTION_ID_TO_NAME.get(navila_actions[i], str(navila_actions[i]))
                # logger.info(
                #     "[NaVILA][eval] Env %d executing action: %s (%d)",
                #     i,
                #     action_name,
                #     navila_actions[i],
                # )  # silenced: per-step verbose
            
            step_data = [base_actions[i].copy() for i in range(envs.num_envs)]
            
            # 执行动作
            outputs = envs.step(step_data)
            observations, rewards_l, dones, infos = [list(x) for x in zip(*outputs)]
            
            # 记录动作
            for i in range(envs.num_envs):
                episode_steps[i] += 1
                if (
                    max_episode_steps > 0
                    and episode_steps[i] >= max_episode_steps
                    and not dones[i]
                ):
                    # logger.info(
                    #     "[NaVILA][eval] Env %d reached max steps (%d). Forcing episode end.",
                    #     i,
                    #     max_episode_steps,
                    # )  # silenced: per-step verbose
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
                    "value": int(navila_actions[i]),
                }
                debug_info = navila_step_debug[i]
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
            not_done_masks = torch.tensor(
                [[not done] for done in dones],
                dtype=torch.bool,
                device="cpu",
            ).repeat(1, *agent.masks_shape)
            not_done_masks = not_done_masks.to(device=device)
            
            # 添加当前RGB到历史
            rgb_key = _find_rgb_key(batch, config)
            if rgb_key is not None:
                for i in range(envs.num_envs):
                    curr_rgb = Image.fromarray(
                        np.uint8(batch[rgb_key][i].cpu().numpy())
                    ).convert("RGB")
                    past_rgbs[i].append(curr_rgb)
            else:
                logger.warning(f"No RGB key found in batch. Available keys: {list(batch.keys())}")
            
            rewards = torch.tensor(rewards_l, dtype=torch.float, device="cpu").unsqueeze(1)
            current_episode_reward += rewards
            next_episodes_info = envs.current_episodes()
            envs_to_pause = []
            n_envs = envs.num_envs
            
            for i in range(n_envs):
                if (
                    ep_eval_count[(next_episodes_info[i].scene_id, next_episodes_info[i].episode_id)]
                    == evals_per_ep
                ):
                    envs_to_pause.append(i)
                
                disp_info = {k: v for k, v in infos[i].items() if k not in rank0_keys}
                
                if len(config.habitat_baselines.eval.video_option) > 0:
                    frame = observations_to_image({k: v[i] for k, v in batch.items()}, disp_info)
                    # 使用 overlay_frame 的 additional 参数来添加 NaVILA 调试信息
                    additional_lines = None
                    if navila_step_debug[i] is not None:
                        overlay_text = format_navila_debug_overlay(navila_step_debug[i])
                        if overlay_text:
                            # format_navila_debug_overlay 返回字符串，可能包含多行
                            # 将其拆分成行列表
                            if isinstance(overlay_text, str):
                                # 按换行符拆分，并过滤空行
                                additional_lines = [line.strip() for line in overlay_text.split('\n') if line.strip()]
                            else:
                                additional_lines = [str(overlay_text)]
                    frame = overlay_frame(frame, disp_info, additional=additional_lines)
                    frame = _annotate_frame_with_instruction(
                        frame, navila_step_debug[i]
                    )
                    if dones[i]:
                        final_frame = observations_to_image(
                            {k: v[i] * 0.0 for k, v in batch.items()}, disp_info
                        )
                        final_frame = overlay_frame(final_frame, disp_info)
                        final_frame = _annotate_frame_with_instruction(
                            final_frame, navila_step_debug[i]
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
                    
                    metrics_for_video = extract_scalars_from_info(infos[i])
                    episode_stats = {"reward": current_episode_reward[i].item()}
                    episode_stats.update(metrics_for_video)
                    current_episode_reward[i] = 0
                    k = (current_episodes_info[i].scene_id, current_episodes_info[i].episode_id)
                    ep_eval_count[k] += 1
                    stats_episodes[(k, ep_eval_count[k])] = episode_stats
                    
                    # 重置该环境的历史和队列
                    past_rgbs[i] = []
                    action_queues[i] = []
                    navila_step_debug[i] = None
                    
                    if len(config.habitat_baselines.eval.video_option) > 0:
                        scene_id = current_episodes_info[i].scene_id.split('/')[-1].split('.')[0]
                        # logger.info(
                        #     f"Scene ID: {scene_id}, Episode ID: {current_episodes_info[i].episode_id}"
                        # )  # silenced: per-episode verbose
                        
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
                # 同时暂停历史和队列
                active_indices = [i for i in range(n_envs) if i not in envs_to_pause]
                past_rgbs = [past_rgbs[i] for i in active_indices]
                action_queues = [action_queues[i] for i in active_indices]
                navila_step_debug = [navila_step_debug[i] for i in active_indices]
                episode_instruction_logs = [episode_instruction_logs[i] for i in active_indices]
                episode_steps = [episode_steps[i] for i in active_indices]
                
                (
                    envs,
                    test_recurrent_hidden_states,
                    not_done_masks,
                    current_episode_reward,
                    prev_actions,
                    batch,
                    rgb_frames,
                ) = pause_envs(
                    envs_to_pause,
                    envs,
                    test_recurrent_hidden_states,
                    not_done_masks,
                    current_episode_reward,
                    prev_actions,
                    batch,
                    rgb_frames,
                )
        
        pbar.close()
        
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
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_root = os.path.join("navila-output")
        os.makedirs(output_root, exist_ok=True)
        result_path = os.path.join(output_root, f"result_{timestamp}.json")
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
        
        evalai_result["generated_at"] = timestamp
        with open(result_path, "w") as f:
            json.dump(evalai_result, f, indent=2)
        
        # 保存动作记录
        actions_output_path = os.path.join(output_root, f"actions_{timestamp}.json")
        serializable_actions = {
            f"{scene_id}|{episode_id}|{eval_count}": actions
            for (scene_id, episode_id, eval_count), actions in actions_record.items()
        }
        with open(actions_output_path, "w") as f:
            json.dump(serializable_actions, f, indent=2)
    
    def _generate_navila_action(
        self, batch, env_idx, past_rgbs, num_video_frames,
        current_episode, model, tokenizer, image_processor, 
        action_parser, action_queue, device, instruction_sensor_uuid=None
    ):
        """
        使用NaVILA模型生成动作
        
        Args:
            batch: 观察批次
            env_idx: 环境索引
            past_rgbs: 历史RGB帧列表
            num_video_frames: 视频帧数
            current_episode: 当前episode信息
            model: LLAVA模型
            tokenizer: tokenizer
            image_processor: 图像处理器
            action_parser: 动作解析器
            action_queue: 动作队列（用于存储多步骤动作）
            device: 设备
            instruction_sensor_uuid: 指令传感器UUID（可选）
            
        Returns:
            action: 动作ID (0-3)
        """
        # 获取当前RGB
        rgb_key = _find_rgb_key(batch)
        if rgb_key is None:
            available_keys = list(batch.keys())
            raise ValueError(
                f"No RGB key found in batch. Available keys: {available_keys}. "
                f"Please ensure at least one RGB sensor is enabled in the config "
                f"(e.g., agent_0_overhead_front_rgb or agent_0_articulated_agent_jaw_rgb)."
            )
        
        curr_rgb = Image.fromarray(
            np.uint8(batch[rgb_key][env_idx].cpu().numpy())
        ).convert("RGB")
        
        # 构建视频序列
        past_and_current_rgbs = past_rgbs + [curr_rgb]
        sampled_frames = sample_and_pad_images(
            past_and_current_rgbs, num_frames=num_video_frames
        )
        
        # 获取指令（如果有的话）
        env_observations: Dict[str, Any] = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                env_observations[key] = value[env_idx]
            else:
                env_observations[key] = value

        episode_instruction = getattr(current_episode, "instruction", None)
        if episode_instruction is not None and not isinstance(episode_instruction, str):
            episode_instruction = getattr(episode_instruction, "instruction_text", None) or str(episode_instruction)

        instruction = extract_navila_instruction(
            env_observations,
            instruction_sensor_uuid=instruction_sensor_uuid,
            episode_instruction=episode_instruction,
        )
        
        # 构建提示
        interleaved_images = "<image>\n" * (len(sampled_frames) - 1)
        question = (
            f"Imagine you are a robot programmed for navigation tasks. You have been given a video "
            f'of historical observations {interleaved_images}, and current observation <image>\n. '
            f'Your assigned task is: "{instruction}" '
            f"Analyze this series of images to decide your next action, which could be turning left or right by a specific "
            f"degree, moving forward a certain distance, or stop if the task is completed."
        )
        
        # 构建对话
        conv_mode = "llama_3"
        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        
        # 处理图像
        # 确保图像在正确的设备上（与模型相同的设备）
        try:
            model_device = next(model.parameters()).device
        except (StopIteration, AttributeError):
            model_device = device
        
        # 打印设备信息用于调试
        # logger.info("=" * 80)
        # logger.info("DEBUG: Device information before processing inputs:")
        # logger.info(f"  Target device: {device}")
        # logger.info(f"  Model device: {model_device}")
        
        images_tensor = process_images(
            sampled_frames, image_processor, model.config
        )
        # logger.info(f"  Images tensor device (before move): {images_tensor.device}")
        images_tensor = images_tensor.to(model_device, dtype=torch.float16)
        # logger.info(f"  Images tensor device (after move): {images_tensor.device}")
        
        # Tokenize
        # 确保 input_ids 在正确的设备上
        input_ids = (
            tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
        )
        # logger.info(f"  Input IDs device (before move): {input_ids.device}")
        input_ids = input_ids.to(model_device)
        # logger.info(f"  Input IDs device (after move): {input_ids.device}")
        # logger.info("=" * 80)
        
        # 停止条件
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        
        # 生成输出
        # 启用KV缓存以加速推理（预期5-10倍加速）
        # KV缓存在单次生成过程中可以避免重复计算已生成token的key-value
        # 注意：如果遇到兼容性问题，可以回退到use_cache=False
        # 确保所有输入都在模型所在的设备上
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=images_tensor.half().to(model_device),  # 使用 model_device 而不是 device
                do_sample=False,
                temperature=0.0,
                max_new_tokens=32,
                use_cache=True,  # 启用KV缓存以加速推理（预期5-10倍加速）
                stopping_criteria=[stopping_criteria],
                pad_token_id=tokenizer.eos_token_id,
            )
        
        # 解码
        output_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        if output_text.endswith(stop_str):
            output_text = output_text[: -len(stop_str)].strip()
        
        # logger.info(f"NaVILA output: {output_text}")  # silenced: per-step verbose
        
        # 解析动作
        action, num_repeats = action_parser.parse_action(output_text)
        debug_info = {
            "instruction": instruction,
            "model_output": output_text,
            "action_id": int(action),
            "action_name": ACTION_ID_TO_NAME.get(action, f"action_{action}"),
            "repeats": int(max(1, num_repeats)),
            "repeat_index": 1,
            "from_queue": False,
        }
        # logger.info(
        #     "[NaVILA][env %d] action=%s repeats=%d instruction=\"%s\" llm=\"%s\"",
        #     env_idx,
        #     debug_info["action_name"],
        #     debug_info["repeats"],
        #     instruction[:200],
        #     output_text[:200],
        # )  # silenced: per-step verbose
        # 将后续动作加入队列
        if num_repeats > 1:
            for repeat_idx in range(2, num_repeats + 1):
                queued_debug = dict(debug_info)
                queued_debug["repeat_index"] = repeat_idx
                queued_debug["from_queue"] = True
                action_queue.append({"action": action, "debug": queued_debug})
            # logger.info(f"Added {num_repeats - 1} actions to queue. Queue length: {len(action_queue)}")  # silenced: per-step verbose
        
        return action, debug_info
