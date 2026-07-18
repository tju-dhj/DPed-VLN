#!/usr/bin/env python3
"""
NaVid Evaluator for Falcon Framework
仿照 NaVILAEvaluator 实现，使用 NaVid (Vicuna-7B + EVA-CLIP ViT-G) 进行 zero-shot 和 fine-tuned 评估。
"""

import datetime
import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import tqdm
from PIL import Image

from habitat import logger
from habitat.utils.visualizations.utils import observations_to_image

from habitat_baselines.common.obs_transformers import apply_obs_transforms_batch
from habitat_baselines.rl.ppo.falcon_evaluator import FALCONEvaluator
from habitat_baselines.utils.common import (
    batch_obs,
    get_action_space_info,
    inference_mode,
)
from habitat_baselines.utils.info_dict import extract_scalars_from_info
from habitat_baselines.rl.ppo.navila_evaluator import _find_rgb_key, _load_eval_resume, _save_eval_resume

# NaVid imports
NAVID_AVAILABLE = False
try:
    from habitat_baselines.rl.ddppo.policy.navid.action_parser import (
        NaVidActionParser,
        ACTION_ID_TO_NAME,
    )
    from habitat_baselines.rl.ddppo.policy.navid_policy import (
        sample_and_pad_images,
        extract_navid_instruction,
    )
    from habitat_baselines.rl.ddppo.policy.navid.constants import (
        IMAGE_TOKEN_INDEX,
        DEFAULT_IMAGE_TOKEN,
        VIDEO_START_SPECIAL_TOKEN,
        VIDEO_END_SPECIAL_TOKEN,
        IMAGE_START_TOKEN,
        IMAGE_END_TOKEN,
        NAVIGATION_SPECIAL_TOKEN,
        IAMGE_SEPARATOR,
    )
    from habitat_baselines.rl.ddppo.policy.navid.conversation import (
        conv_templates,
        SeparatorStyle,
    )
    from habitat_baselines.rl.ddppo.policy.navid.mm_utils import (
        tokenizer_image_token,
        KeywordsStoppingCriteria,
    )
    from habitat_baselines.rl.ddppo.policy.navid.model.builder import (
        load_pretrained_model,
    )
    NAVID_AVAILABLE = True
except ImportError as e:
    print(f"Warning: NaVid modules not available: {e}")


class NaVidEvaluator(FALCONEvaluator):
    """NaVid 专用评估器"""

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
        if not NAVID_AVAILABLE:
            raise ImportError("NaVid modules are required for NaVidEvaluator")

        navid_config = config.habitat_baselines.rl.policy.agent_0
        tokenizer = None
        model = None
        image_processor = None

        # Try to reuse model from policy
        try:
            net = None
            target_agent = agent
            if hasattr(agent, '_agents') and len(agent._agents) > 0:
                target_agent = agent._agents[0]

            if hasattr(target_agent, 'actor_critic'):
                actor_critic = target_agent.actor_critic
                if hasattr(actor_critic, 'net'):
                    net = actor_critic.net

            if net is not None and hasattr(net, 'model') and hasattr(net, 'tokenizer'):
                tokenizer = net.tokenizer
                model = net.model
                image_processor = net.image_processor
                logger.info("Reusing NaVid model from policy")
        except Exception as e:
            logger.warning(f"Could not reuse model from policy: {e}")

        # Load model if not available
        if model is None or tokenizer is None:
            eval_model_path = config.habitat_baselines.eval_ckpt_path_dir
            model_path = navid_config.get("model_path", eval_model_path)

            if not os.path.exists(model_path):
                raise ValueError(f"NaVid model path not found: {model_path}")

            logger.info(f"Loading NaVid model from {model_path}")
            model_name = os.path.basename(os.path.normpath(model_path))
            tokenizer, model, image_processor, context_len = load_pretrained_model(
                model_path, None, model_name
            )
            model = model.to(device)
            model.eval()

        # Detect SFT checkpoint and setup action_head for forward-based eval
        from torch import nn
        self.action_head = None
        eval_model_path = config.habitat_baselines.eval_ckpt_path_dir
        if eval_model_path and os.path.exists(os.path.join(eval_model_path, 'adapter_model.bin')):
            model_path_check = eval_model_path
        else:
            model_path_check = navid_config.get("model_path", "")
        if model_path_check and os.path.exists(os.path.join(model_path_check, 'adapter_model.bin')):
            logger.info("[NaVid Eval] SFT checkpoint detected — using forward()+action_head (no text generation)")
            self.action_head = nn.Sequential(
                nn.Linear(4096, 512), nn.ReLU(), nn.Dropout(0.1), nn.Linear(512, 4)
            ).to(device)
            # Load action_head weights from the checkpoint
            adapter_path = os.path.join(model_path_check, 'adapter_model.bin')
            try:
                adapter_state = torch.load(adapter_path, map_location=device)
                # Extract action_head params (keys containing 'action_head')
                ah_state = {}
                for k, v in adapter_state.items():
                    if 'action_head' in k:
                        new_k = k.split('base_model.model.')[-1] if 'base_model.model.' in k else k
                        new_k = new_k.replace('net.model.', '').replace('base_model.model.', '')
                        ah_state[new_k] = v
                if ah_state:
                    self.action_head.load_state_dict(ah_state, strict=False)
                    logger.info(f"[NaVid Eval] Loaded action_head weights ({len(ah_state)} params)")
                else:
                    logger.warning("[NaVid Eval] No action_head weights in checkpoint — using untrained head")
            except Exception as e:
                logger.warning(f"[NaVid Eval] Failed to load action_head: {e} — using untrained head")
            self.action_head.eval()

        # Action parser
        action_parser = NaVidActionParser(
            forward_step=navid_config.get("forward_step", 25),
            turn_step=navid_config.get("turn_step", 15),
        )

        num_video_frames = navid_config.get("num_video_frames", 4)

        # Init env
        observations = envs.reset()
        observations = envs.post_step(observations)
        batch = batch_obs(observations, device=device)
        batch = apply_obs_transforms_batch(batch, obs_transforms)

        action_shape, discrete_actions = get_action_space_info(
            agent.actor_critic.policy_action_space
        )
        action_space_lens = agent.actor_critic.policy_action_space_shape_lens
        hidden_state_lens = agent.actor_critic.hidden_state_shape_lens

        from gym import spaces
        agent0_space = action_space_lens[0]
        if isinstance(agent0_space, spaces.Discrete):
            agent0_width = 1
        elif isinstance(agent0_space, spaces.Box):
            agent0_width = int(np.prod(agent0_space.shape))
        elif isinstance(agent0_space, (int, np.integer)):
            agent0_width = int(agent0_space)
        else:
            try:
                agent0_width = int(agent0_space)
            except (TypeError, ValueError):
                raise ValueError(
                    f"Unsupported action space type for agent_0: {type(agent0_space)}. "
                    f"Expected Discrete, Box, or int, got {agent0_space}."
                )
        agent0_start = 0
        agent0_end = agent0_start + agent0_width

        current_episode_reward = torch.zeros(envs.num_envs, 1, device="cpu")
        test_recurrent_hidden_states = torch.zeros(
            (config.habitat_baselines.num_environments, *agent.actor_critic.hidden_state_shape),
            device=device,
        )
        prev_actions = torch.zeros(
            config.habitat_baselines.num_environments, *action_shape,
            device=device, dtype=torch.long if discrete_actions else torch.float,
        )
        not_done_masks = torch.zeros(
            config.habitat_baselines.num_environments, *agent.masks_shape,
            device=device, dtype=torch.bool,
        )

        stats_episodes, ep_eval_count, total_completed = _load_eval_resume(config)
        success_cal = sum(float(v.get("success", 0.0)) for v in stats_episodes.values())

        # Action queues and RGB history
        past_rgbs = [[] for _ in range(envs.num_envs)]
        action_queues = [[] for _ in range(envs.num_envs)]

        # Video recording
        if len(config.habitat_baselines.eval.video_option) > 0:
            rgb_frames = [
                [observations_to_image({k: v[env_idx] for k, v in batch.items()}, {})]
                for env_idx in range(config.habitat_baselines.num_environments)
            ]
        else:
            rgb_frames = None

        number_of_eval_episodes = config.habitat_baselines.test_episode_count
        evals_per_ep = config.habitat_baselines.eval.evals_per_ep
        if number_of_eval_episodes == -1:
            number_of_eval_episodes = sum(envs.number_of_episodes)

        pbar = tqdm.tqdm(total=number_of_eval_episodes * evals_per_ep)
        actions_record = defaultdict(list)
        max_episode_steps = getattr(config.habitat_baselines.eval, "max_steps_per_episode", -1)
        episode_steps = [0 for _ in range(envs.num_envs)]

        # Main loop
        while (
            len(stats_episodes) < (number_of_eval_episodes * evals_per_ep)
            and envs.num_envs > 0
        ):
            current_episodes_info = envs.current_episodes()

            space_lengths = {}
            n_agents = len(config.habitat.simulator.agents)
            if n_agents > 1:
                space_lengths = {
                    "index_len_recurrent_hidden_states": hidden_state_lens,
                    "index_len_prev_actions": action_space_lens,
                }

            with inference_mode():
                action_data = agent.actor_critic.act(
                    batch, test_recurrent_hidden_states, prev_actions,
                    not_done_masks, deterministic=False, **space_lengths,
                )
                if action_data.should_inserts is None:
                    test_recurrent_hidden_states = action_data.rnn_hidden_states
                    prev_actions.copy_(action_data.actions)
                else:
                    agent.actor_critic.update_hidden_state(
                        test_recurrent_hidden_states, prev_actions, action_data
                    )

            base_actions = action_data.env_actions.detach().cpu().numpy().astype(np.float32, copy=True)

            # Generate NaVid actions
            navid_actions = []
            for i in range(envs.num_envs):
                if len(action_queues[i]) > 0:
                    queued = action_queues[i].pop(0)
                    action = queued.get("action", 0) if isinstance(queued, dict) else int(queued)
                else:
                    action, debug_info = self._generate_navid_action(
                        batch, i, past_rgbs[i], num_video_frames,
                        current_episodes_info[i], model, tokenizer, image_processor,
                        action_parser, action_queues[i], device,
                    )
                navid_actions.append(action)
                base_actions[i, agent0_start:agent0_end] = float(action)

            # Execute actions
            step_data = [base_actions[i].copy() for i in range(envs.num_envs)]
            outputs = envs.step(step_data)
            observations, rewards_l, dones, infos = [list(x) for x in zip(*outputs)]

            for i in range(envs.num_envs):
                episode_steps[i] += 1
                if max_episode_steps > 0 and episode_steps[i] >= max_episode_steps and not dones[i]:
                    infos[i]["max_step_reached"] = True
                    dones[i] = True

                episode_key = (
                    current_episodes_info[i].scene_id,
                    current_episodes_info[i].episode_id,
                    ep_eval_count.get(
                        (current_episodes_info[i].scene_id, current_episodes_info[i].episode_id), 0
                    ),
                )
                actions_record[episode_key].append({
                    "type": "scalar",
                    "value": int(navid_actions[i]),
                    "action_name": ACTION_ID_TO_NAME.get(navid_actions[i], str(navid_actions[i])),
                })

            # Update observations
            observations = envs.post_step(observations)
            batch = batch_obs(observations, device=device)
            batch = apply_obs_transforms_batch(batch, obs_transforms)
            not_done_masks = torch.tensor(
                [[not done] for done in dones], dtype=torch.bool, device="cpu",
            ).repeat(1, *agent.masks_shape).to(device=device)

            # Update RGB history
            rgb_key = _find_rgb_key(batch, config)
            if rgb_key is not None:
                for i in range(envs.num_envs):
                    curr_rgb = Image.fromarray(
                        np.uint8(batch[rgb_key][i].cpu().numpy())
                    ).convert("RGB")
                    past_rgbs[i].append(curr_rgb)
                    if len(past_rgbs[i]) > 50:
                        past_rgbs[i] = past_rgbs[i][-50:]

            rewards = torch.tensor(rewards_l, dtype=torch.float, device="cpu").unsqueeze(1)
            current_episode_reward += rewards

            # Handle episode end
            next_episodes_info = envs.current_episodes()
            for i in range(envs.num_envs):
                if ep_eval_count.get(
                    (next_episodes_info[i].scene_id, next_episodes_info[i].episode_id), 0
                ) >= evals_per_ep:
                    # Episode complete
                    ep_key = (
                        next_episodes_info[i].scene_id,
                        next_episodes_info[i].episode_id,
                    )
                    ep_eval_count[ep_key] = ep_eval_count.get(ep_key, 0) + 1

                if dones[i]:
                    # Collect metrics (use navila-compatible key format)
                    episode_key = (
                        current_episodes_info[i].scene_id,
                        current_episodes_info[i].episode_id,
                    )
                    disp_info = infos[i]
                    metrics = extract_scalars_from_info(disp_info)
                    eval_count = ep_eval_count.get(episode_key, 1)
                    stats_episodes[(episode_key, eval_count)] = {
                        "success": float(metrics.get("success", 0.0)),
                        "spl": float(metrics.get("spl", 0.0)),
                        "distance_to_goal": float(metrics.get("distance_to_goal", 0.0)),
                        "path_length": float(metrics.get("path_length", 0.0)),
                    }

                    if float(metrics.get("success", 0.0)) > 0.5:
                        success_cal += 1

                    pbar.update(1)
                    pbar.set_description(
                        f"SR: {success_cal}/{len(stats_episodes)} = "
                        f"{success_cal / max(1, len(stats_episodes)):.3f}"
                    )

                    # Persist eval progress after each completed episode
                    _save_eval_resume(config, stats_episodes, ep_eval_count)

                    # Reset state
                    past_rgbs[i] = []
                    action_queues[i] = []
                    episode_steps[i] = 0

        # Final eval resume save after all episodes complete
        _save_eval_resume(config, stats_episodes, ep_eval_count)

        # Final stats
        successes = [float(v.get("success", 0.0)) for v in stats_episodes.values()]
        spls = [float(v.get("spl", 0.0)) for v in stats_episodes.values()]
        n_total = len(stats_episodes)

        logger.info("=" * 60)
        logger.info(f"NaVid Evaluation Complete: {n_total} episodes")
        logger.info(f"  Success Rate (SR): {np.mean(successes):.4f} ({sum(s > 0.5 for s in successes)}/{n_total})")
        logger.info(f"  SPL: {np.mean(spls):.4f}")
        logger.info("=" * 60)

        return stats_episodes

    def _generate_navid_action(
        self, batch, env_idx, past_rgbs, num_video_frames,
        current_episode, model, tokenizer, image_processor,
        action_parser, action_queue, device,
    ):
        """使用 NaVid 模型生成动作"""
        # Get current RGB
        rgb_key = _find_rgb_key(batch)
        if rgb_key is None:
            return 0, {"error": "No RGB key found"}

        curr_rgb = Image.fromarray(
            np.uint8(batch[rgb_key][env_idx].cpu().numpy())
        ).convert("RGB")

        # Build video sequence
        past_and_current = past_rgbs + [curr_rgb]
        sampled = sample_and_pad_images(past_and_current, num_frames=num_video_frames)

        # Get instruction
        env_observations = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                env_observations[key] = value[env_idx]
            else:
                env_observations[key] = value

        episode_instruction = getattr(current_episode, "instruction", None)
        if episode_instruction is not None and not isinstance(episode_instruction, str):
            episode_instruction = getattr(episode_instruction, "instruction_text", None) or str(episode_instruction)

        instruction = extract_navid_instruction(
            env_observations,
            episode_instruction=episode_instruction,
        )
        if not hasattr(self, '_debug_inst_count'):
            self._debug_inst_count = 0
        if self._debug_inst_count < 3:
            self._debug_inst_count += 1
            logger.info(f"[DEBUG-INST #{self._debug_inst_count}] instruction='{instruction[:150]}'")

        # ===== SFT forward path: use action_head directly (no text generation) =====
        if self.action_head is not None:
            curr_rgb_tensor = image_processor.preprocess(curr_rgb, return_tensors='pt')['pixel_values']
            curr_rgb_tensor = curr_rgb_tensor.half().to(device)
            qs = DEFAULT_IMAGE_TOKEN + '\n' + instruction.replace('<image>', '')
            conv = conv_templates["vicuna_v1"].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
            input_ids = input_ids.unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids, images=[curr_rgb_tensor],
                    prompts=[[instruction.replace('<image>','').replace('\n',' ').strip()]],
                    output_hidden_states=True, return_dict=True)
                pooled = outputs.hidden_states[-1].mean(dim=1).float()
                logits = self.action_head(pooled)
                action = int(logits.argmax(dim=-1).item())
            return action, {"method": "action_head", "logits": logits.cpu().numpy().tolist()}

        # Build prompt (matching official agent_navid.py format exactly)
        nav_prompt = (
            f"Imagine you are a robot programmed for navigation tasks. "
            f"You have been given a video of historical observations "
            f"and an image of the current observation <image>. "
            f"Your assigned task is: '{instruction}'. "
            f"Analyze this series of images to decide your next move, which could involve "
            f"turning left or right by a specific degree or moving forward a certain distance."
        )

        # Conversation (vicuna_v1 - matching official agent)
        conv_mode = "vicuna_v1"
        conv = conv_templates[conv_mode].copy()
        # Official format: single <image> token for current frame, video frames via images tensor
        qs = DEFAULT_IMAGE_TOKEN + '\n' + nav_prompt.replace('<image>', '')
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        # Process images
        try:
            model_device = next(model.parameters()).device
        except (StopIteration, AttributeError):
            model_device = device

        # Process frames through image processor
        if len(sampled) == 1:
            images_tensor = image_processor.preprocess(sampled[0], return_tensors='pt')['pixel_values']
        else:
            batch_np = np.stack([np.array(img) for img in sampled])
            images_tensor = image_processor.preprocess(batch_np, return_tensors='pt')['pixel_values']
        images_tensor = images_tensor.to(model_device, dtype=torch.float16)

        # Tokenize with special tokens
        token_prompt = tokenizer_image_token(
            prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
        ).cuda()

        # Insert NaVid special tokens
        image_start_token = tokenizer(IMAGE_START_TOKEN, return_tensors="pt").input_ids[0][1:].cuda()
        image_end_token = tokenizer(IMAGE_END_TOKEN, return_tensors="pt").input_ids[0][1:].cuda()
        video_start_token = tokenizer(VIDEO_START_SPECIAL_TOKEN, return_tensors="pt").input_ids[0][1:].cuda()
        video_end_token = tokenizer(VIDEO_END_SPECIAL_TOKEN, return_tensors="pt").input_ids[0][1:].cuda()
        navigation_token = tokenizer(NAVIGATION_SPECIAL_TOKEN, return_tensors="pt").input_ids[0][1:].cuda()
        image_sep = tokenizer(IAMGE_SEPARATOR, return_tensors="pt").input_ids[0][1:].cuda()

        indices_to_replace = torch.where(token_prompt == -200)[0]
        new_list = []
        while indices_to_replace.numel() > 0:
            idx = indices_to_replace[0]
            new_list.append(token_prompt[:idx])
            new_list.append(video_start_token)
            new_list.append(image_sep)
            new_list.append(token_prompt[idx:idx + 1])
            new_list.append(video_end_token)
            new_list.append(image_start_token)
            new_list.append(image_end_token)
            new_list.append(navigation_token)
            token_prompt = token_prompt[idx + 1:]
            indices_to_replace = torch.where(token_prompt == -200)[0]
        if token_prompt.numel() > 0:
            new_list.append(token_prompt)
        input_ids = torch.cat(new_list, dim=0).unsqueeze(0).to(model_device)

        # Stopping criteria
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        # Generate (matching official agent_navid.py exactly)
        question_text = nav_prompt.replace(DEFAULT_IMAGE_TOKEN, '').replace('\n', '')
        imgs = [images_tensor.half().to(model_device)]

        with torch.inference_mode():
            # Debug: verify prompts are set correctly
            logger.info(f"[DEBUG-GEN] question_text[:100]='{question_text[:100]}', model.prompts={getattr(model, 'prompts', None)}")
            logger.info(f"[DEBUG-GEN] input_ids.shape={input_ids.shape}, images[0].shape={imgs[0].shape if isinstance(imgs, list) else imgs.shape}")
            
            model.update_prompt([[question_text]])
            logger.info(f"[DEBUG-GEN] after update_prompt: model.prompts={model.prompts}")
            
            output_ids = model.generate(
                input_ids,
                images=imgs,
                do_sample=True,
                temperature=0.2,
                max_new_tokens=128,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
                pad_token_id=tokenizer.eos_token_id,
            )

        input_token_len = input_ids.shape[1]
        output_text = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0].strip()
        if output_text.endswith(stop_str):
            output_text = output_text[:-len(stop_str)].strip()

        # Parse action
        action, num_repeats = action_parser.parse_action(output_text)
        debug_info = {
            "instruction": instruction,
            "model_output": output_text,
            "action_id": int(action),
            "action_name": ACTION_ID_TO_NAME.get(action, f"action_{action}"),
            "repeats": int(max(1, num_repeats)),
        }
        if self._debug_inst_count <= 5:
            logger.info(f"[DEBUG-ACT #{self._debug_inst_count}] raw_output='{output_text}'")
            logger.info(f"[DEBUG-ACT #{self._debug_inst_count}] action={action}({ACTION_ID_TO_NAME.get(action, '?')}), repeats={num_repeats}")
        if self._debug_inst_count <= 5:
            logger.info(f"[DEBUG-PROMPT #{self._debug_inst_count}] nav_prompt[:500]='{nav_prompt[:500]}'")
            logger.info(f"[DEBUG-PROMPT #{self._debug_inst_count}] images_tensor.shape={images_tensor.shape}")
            logger.info(f"[DEBUG-PROMPT #{self._debug_inst_count}] input_ids.shape={input_ids.shape}, len_sampled={len(sampled)}")

        # Queue repeats
        if num_repeats > 1:
            for _ in range(num_repeats - 1):
                action_queue.append({"action": action})

        return action, debug_info
