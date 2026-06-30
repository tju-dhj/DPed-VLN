import os
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np
import torch
import tqdm
import gc

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
from habitat_baselines.utils.common import (
    batch_obs,
    generate_video,
    get_action_space_info,
    inference_mode,
    is_continuous_action_space,
)
from habitat_baselines.utils.info_dict import extract_scalars_from_info

import json

def calculate_and_log_average_stats(stats_episodes, completed_episodes, logger):
    """
    计算并输出从开始到当前的平均评估结果
    
    Args:
        stats_episodes: 存储所有episode统计信息的字典
        completed_episodes: 已完成的episodes数量
        logger: logger对象用于输出日志
    """
    if len(stats_episodes) == 0:
        return
    
    aggregated_stats = {}
    all_ks = set()
    for ep in stats_episodes.values():
        all_ks.update(ep.keys())
    for stat_key in all_ks:
        aggregated_stats[stat_key] = np.mean(
            [v[stat_key] for v in stats_episodes.values() if stat_key in v]
        )
    
    logger.info(f"========== 已完成 {completed_episodes} 个episodes的平均评估结果 ==========")
    for k, v in aggregated_stats.items():
        logger.info(f"Average episode {k}: {v:.4f}")
    logger.info("=" * 60)

class FALCONEvaluator(Evaluator):
    """
    Only difference is record the success rate of each episode while evaluating.
    Similar to ORCAEvaluator.
    """
    
    def _load_eval_checkpoint(self, checkpoint_path):
        """
        加载已完成的评估checkpoint
        
        Returns:
            tuple: (stats_episodes, ep_eval_count, actions_record, completed_episodes_ids)
        """
        if not os.path.exists(checkpoint_path):
            logger.info(f"No evaluation checkpoint found at {checkpoint_path}, starting fresh evaluation.")
            return {}, defaultdict(lambda: 0), defaultdict(list), set()
        
        try:
            with open(checkpoint_path, 'r') as f:
                checkpoint_data = json.load(f)
            
            # 恢复stats_episodes
            stats_episodes = {}
            for key_str, stats in checkpoint_data.get('stats_episodes', {}).items():
                # key格式: "scene_id|episode_id|eval_count"
                parts = key_str.split('|')
                if len(parts) == 3:
                    scene_id, episode_id, eval_count = parts[0], parts[1], int(parts[2])
                    stats_episodes[((scene_id, episode_id), eval_count)] = stats
            
            # 恢复ep_eval_count
            ep_eval_count = defaultdict(lambda: 0)
            for key_str, count in checkpoint_data.get('ep_eval_count', {}).items():
                parts = key_str.split('|')
                if len(parts) == 2:
                    scene_id, episode_id = parts[0], parts[1]
                    ep_eval_count[(scene_id, episode_id)] = count
            
            # 恢复actions_record
            actions_record = defaultdict(list)
            for key_str, actions in checkpoint_data.get('actions_record', {}).items():
                parts = key_str.split('|')
                if len(parts) == 3:
                    scene_id, episode_id, eval_count = parts[0], parts[1], int(parts[2])
                    actions_record[(scene_id, episode_id, eval_count)] = actions
            
            # 获取已完成的episode IDs集合
            completed_episodes_ids = set()
            for key_str in checkpoint_data.get('stats_episodes', {}).keys():
                parts = key_str.split('|')
                if len(parts) == 3:
                    scene_id, episode_id = parts[0], parts[1]
                    completed_episodes_ids.add((scene_id, episode_id))
            
            logger.info(f"Loaded evaluation checkpoint: {len(stats_episodes)} completed episodes, "
                       f"{len(completed_episodes_ids)} unique episodes evaluated.")
            
            return stats_episodes, ep_eval_count, actions_record, completed_episodes_ids
            
        except Exception as e:
            logger.error(f"Error loading evaluation checkpoint: {e}")
            logger.info("Starting fresh evaluation.")
            return {}, defaultdict(lambda: 0), defaultdict(list), set()
    
    def _save_eval_checkpoint(self, checkpoint_path, stats_episodes, ep_eval_count, actions_record):
        """
        保存当前的评估checkpoint
        """
        try:
            # 转换为可序列化的格式
            checkpoint_data = {
                'stats_episodes': {},
                'ep_eval_count': {},
                'actions_record': {}
            }
            
            # 保存stats_episodes
            # Key格式: ((scene_id, episode_id), eval_count)
            for ((scene_id, episode_id), eval_count), stats in stats_episodes.items():
                key_str = f"{scene_id}|{episode_id}|{eval_count}"
                checkpoint_data['stats_episodes'][key_str] = stats
            
            # 保存ep_eval_count
            for (scene_id, episode_id), count in ep_eval_count.items():
                key_str = f"{scene_id}|{episode_id}"
                checkpoint_data['ep_eval_count'][key_str] = count
            
            # 保存actions_record
            for (scene_id, episode_id, eval_count), actions in actions_record.items():
                key_str = f"{scene_id}|{episode_id}|{eval_count}"
                checkpoint_data['actions_record'][key_str] = actions
            
            # 确保目录存在
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            
            # 写入文件
            with open(checkpoint_path, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
            
            logger.info(f"Saved evaluation checkpoint: {len(stats_episodes)} completed episodes.")
            
        except Exception as e:
            logger.error(f"Error saving evaluation checkpoint: {e}")

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
        # 设置checkpoint路径，根据配置文件名和数据集名称区分
        # 获取配置相关信息用于区分不同的评估任务
        config_name = getattr(config.habitat_baselines, 'eval_config_name', 'default')
        dataset_name = config.habitat.dataset.data_path.split('/')[-1].replace('.json.gz', '').replace('.json', '')
        
        # 创建包含配置信息的checkpoint目录
        checkpoint_dir = os.path.join(
            config.habitat_baselines.checkpoint_folder, 
            "eval_checkpoints",
            f"{config_name}_{dataset_name}"
        )
        checkpoint_path = os.path.join(checkpoint_dir, f"eval_progress_ckpt_{checkpoint_index}.json")
        
        logger.info(f"Evaluation checkpoint will be saved to: {checkpoint_path}")
        
        # 加载已完成的评估checkpoint
        stats_episodes, ep_eval_count, actions_record, completed_episodes_ids = self._load_eval_checkpoint(checkpoint_path)
        
        success_cal = 0 ## my added
        # 计算已完成的episodes中的成功数
        for stats in stats_episodes.values():
            if 'success' in stats:
                success_cal += stats['success']
        
        observations = envs.reset()
        observations = envs.post_step(observations)
        batch = batch_obs(observations, device=device)
        batch = apply_obs_transforms_batch(batch, obs_transforms)  # type: ignore

        action_shape, discrete_actions = get_action_space_info(
            agent.actor_critic.policy_action_space
        )

        current_episode_reward = torch.zeros(envs.num_envs, 1, device="cpu")

        test_recurrent_hidden_states = torch.zeros(
            (
                config.habitat_baselines.num_environments,
                *agent.actor_critic.hidden_state_shape,
            ),
            device=device,
        )

        hidden_state_lens = agent.actor_critic.hidden_state_shape_lens
        action_space_lens = agent.actor_critic.policy_action_space_shape_lens

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

        if len(config.habitat_baselines.eval.video_option) > 0:
            # Add the first frame of the episode to the video.
            rgb_frames: List[List[np.ndarray]] = [
                [
                    observations_to_image(
                        {k: v[env_idx] for k, v in batch.items()}, {}
                    )
                ]
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
            # if total_num_eps is negative, it means the number of evaluation episodes is unknown
            if total_num_eps < number_of_eval_episodes and total_num_eps > 1:
                logger.warn(
                    f"Config specified {number_of_eval_episodes} eval episodes"
                    ", dataset only has {total_num_eps}."
                )
                logger.warn(f"Evaluating with {total_num_eps} instead.")
                number_of_eval_episodes = total_num_eps
            else:
                assert evals_per_ep == 1
        assert (
            number_of_eval_episodes > 0
        ), "You must specify a number of evaluation episodes with test_episode_count"

        # 更新进度条：从已完成的episodes开始
        pbar = tqdm.tqdm(total=number_of_eval_episodes * evals_per_ep, initial=len(stats_episodes))
        agent.eval()
        
        # 内存清理计数器：每处理N个episode就清理一次内存
        # 设置为极大值以禁用定期清理，避免eval速度变慢
        memory_cleanup_interval = 99999999
        episodes_since_cleanup = 0
        
        # 用于跟踪已完成episodes数量，用于每50个episodes输出一次平均结果
        completed_episodes_count = len(stats_episodes)  # 从已完成的数量开始
        stats_report_interval = 50  # 每50个episodes输出一次平均评估结果
        
        # checkpoint保存间隔：每N个episodes保存一次
        checkpoint_save_interval = 10  # 每10个episodes保存一次checkpoint
        episodes_since_last_save = 0
        
        # 记录哪些env当前正在运行已完成的episode（需要快速跳过）
        envs_skipping = set()

        while (
            len(stats_episodes) < (number_of_eval_episodes * evals_per_ep)
            and envs.num_envs > 0
            ):
            current_episodes_info = envs.current_episodes()

            # 检查哪些env当前运行的episode已经在checkpoint中完成过
            # 对这些env发送STOP动作（action=0），让episode尽快结束，进入下一个episode
            # 不能用pause_envs，那会永久移除env，导致后续episode无法运行
            for i in range(envs.num_envs):
                episode_key = (
                    current_episodes_info[i].scene_id,
                    current_episodes_info[i].episode_id,
                )
                if ep_eval_count[episode_key] >= evals_per_ep:
                    envs_skipping.add(i)
                else:
                    envs_skipping.discard(i)

            space_lengths = {}
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
                    test_recurrent_hidden_states = (
                        action_data.rnn_hidden_states
                    )
                    prev_actions.copy_(action_data.actions)  # type: ignore
                else:
                    agent.actor_critic.update_hidden_state(
                        test_recurrent_hidden_states, prev_actions, action_data
                    )

            # NB: Move actions to CPU.  If CUDA tensors are
            # sent in to env.step(), that will create CUDA contexts
            # in the subprocesses.
            if hasattr(agent, '_agents') and agent._agents[0]._actor_critic.action_distribution_type == 'categorical':
                step_data = [a.numpy() for a in action_data.env_actions.cpu()]
            elif is_continuous_action_space(env_spec.action_space):
                # Clipping actions to the specified limits
                step_data = [
                    np.clip(
                        a.numpy(),
                        env_spec.action_space.low,
                        env_spec.action_space.high,
                    )
                    for a in action_data.env_actions.cpu()
                ]
            else:
                step_data = [a.item() for a in action_data.env_actions.cpu()]

            # 对正在跳过的env（已完成的episode），强制发送STOP动作(0)让episode尽快结束
            # 不能用pause_envs，那会永久移除env，导致后续episode无法运行
            for i in envs_skipping:
                if i < len(step_data):
                    if isinstance(step_data[i], np.ndarray):
                        step_data[i] = np.zeros_like(step_data[i])
                    else:
                        step_data[i] = 0

            outputs = envs.step(step_data)

            observations, rewards_l, dones, infos = [
                list(x) for x in zip(*outputs)
            ]

            for i in range(envs.num_envs):
                episode_key = (
                    current_episodes_info[i].scene_id,
                    current_episodes_info[i].episode_id,
                    ep_eval_count[
                        (current_episodes_info[i].scene_id, current_episodes_info[i].episode_id)
                    ]
                )

                action_value = step_data[i]
                if isinstance(action_value, np.ndarray):
                    stored_action = {
                        "type": "array",
                        "value": action_value.tolist()
                    }
                else:
                    stored_action = {
                        "type": "array",
                        "value": np.array(action_value).tolist()
                    }

                actions_record[episode_key].append(stored_action)

            # Note that `policy_infos` represents the information about the
            # action BEFORE `observations` (the action used to transition to
            # `observations`).
            policy_infos = agent.actor_critic.get_extra(
                action_data, infos, dones
            )
            for i in range(len(policy_infos)):
                infos[i].update(policy_infos[i])

            observations = envs.post_step(observations)
            batch = batch_obs(  # type: ignore
                observations,
                device=device,
            )
            batch = apply_obs_transforms_batch(batch, obs_transforms)  # type: ignore

            not_done_masks = torch.tensor(
                [[not done] for done in dones],
                dtype=torch.bool,
                device="cpu",
            ).repeat(1, *agent.masks_shape)

            rewards = torch.tensor(
                rewards_l, dtype=torch.float, device="cpu"
            ).unsqueeze(1)
            current_episode_reward += rewards
            next_episodes_info = envs.current_episodes()

            # 保存本轮step前的skipping状态快照，用于判断刚结束的episode是否需要跳过统计
            envs_skipping_this_step = set(envs_skipping)

            # 根据step后的next_episodes_info同步更新envs_skipping状态
            for i in range(envs.num_envs):
                next_ep_key = (
                    next_episodes_info[i].scene_id,
                    next_episodes_info[i].episode_id,
                )
                if ep_eval_count[next_ep_key] >= evals_per_ep:
                    envs_skipping.add(i)
                else:
                    envs_skipping.discard(i)

            envs_to_pause = []
            n_envs = envs.num_envs
            for i in range(n_envs):
                if (
                    ep_eval_count[
                        (
                            next_episodes_info[i].scene_id,
                            next_episodes_info[i].episode_id,
                        )
                    ]
                    == evals_per_ep
                    and i not in envs_skipping  # resume时跳过中的env不能被pause，等它跑到新episode
                ):
                    envs_to_pause.append(i)

                # Exclude the keys from `_rank0_keys` from displaying in the video
                disp_info = {
                    k: v for k, v in infos[i].items() if k not in rank0_keys
                }

                if len(config.habitat_baselines.eval.video_option) > 0:
                    # TODO move normalization / channel changing out of the policy and undo it here
                    frame = observations_to_image(
                        {k: v[i] for k, v in batch.items()}, disp_info
                    )
                    if not not_done_masks[i].any().item():
                        # The last frame corresponds to the first frame of the next episode
                        # but the info is correct. So we use a black frame
                        final_frame = observations_to_image(
                            {k: v[i] * 0.0 for k, v in batch.items()},
                            disp_info,
                        )
                        final_frame = overlay_frame(final_frame, disp_info)
                        rgb_frames[i].append(final_frame)
                        # The starting frame of the next episode will be the final element..
                        rgb_frames[i].append(frame)
                    else:
                        frame = overlay_frame(frame, disp_info)
                        rgb_frames[i].append(frame)

                # episode ended
                if not not_done_masks[i].any().item():
                    k = (
                        current_episodes_info[i].scene_id,
                        current_episodes_info[i].episode_id,
                    )

                    # 如果这个episode是已完成的（正在跳过），只清理临时数据，不记录统计
                    if i in envs_skipping_this_step:
                        current_episode_reward[i] = 0
                        continue

                    pbar.update()
                    episodes_since_cleanup += 1
                    completed_episodes_count += 1
                    episodes_since_last_save += 1

                    if "success" in disp_info:
                        success_cal += disp_info['success']
                        print(f"Till now Success Rate: {success_cal/completed_episodes_count}")
                    episode_stats = {
                        "reward": current_episode_reward[i].item()
                    }
                    episode_stats.update(extract_scalars_from_info(infos[i]))
                    current_episode_reward[i] = 0
                    ep_eval_count[k] += 1
                    # use scene_id + episode_id as unique id for storing stats
                    stats_episodes[(k, ep_eval_count[k])] = episode_stats
                    
                    # 每50个episodes输出一次从开始到当前的平均评估结果
                    if completed_episodes_count % stats_report_interval == 0:
                        calculate_and_log_average_stats(stats_episodes, completed_episodes_count, logger)
                    
                    # 定期保存checkpoint
                    if episodes_since_last_save >= checkpoint_save_interval:
                        self._save_eval_checkpoint(checkpoint_path, stats_episodes, ep_eval_count, actions_record)
                        episodes_since_last_save = 0
                    
                    # 定期内存清理
                    if episodes_since_cleanup >= memory_cleanup_interval:
                        logger.info(f"[Memory Cleanup] Processed {episodes_since_cleanup} episodes, cleaning up memory...")
                        
                        # GPU内存清理
                        torch.cuda.empty_cache()
                        
                        # CPU内存清理 - 强制Python释放内存
                        gc.collect()
                        
                        # 尝试释放内存回操作系统（Linux）
                        try:
                            import ctypes
                            libc = ctypes.CDLL("libc.so.6")
                            libc.malloc_trim(0)
                        except Exception:
                            pass
                        
                        episodes_since_cleanup = 0
                        
                        # # 记录内存使用情况
                        # if torch.cuda.is_available():
                        #     logger.info(f"[Memory Cleanup] GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
                        #     logger.info(f"[Memory Cleanup] GPU memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
                        
                        # # 记录系统内存使用
                        # try:
                        #     import psutil
                        #     process = psutil.Process()
                        #     mem_info = process.memory_info()
                        #     logger.info(f"[Memory Cleanup] System memory (RSS): {mem_info.rss / 1024**3:.2f} GB")
                        #     logger.info(f"[Memory Cleanup] System memory (VMS): {mem_info.vms / 1024**3:.2f} GB")
                        # except Exception:
                        #     pass

                    if len(config.habitat_baselines.eval.video_option) > 0:
                        # show scene and episode
                        scene_id = current_episodes_info[i].scene_id.split('/')[-1].split('.')[0]
                        print(f"This is Scene ID: {scene_id}, Episode ID: {current_episodes_info[i].episode_id}.") # for debug
                        
                        generate_video(
                            video_option=config.habitat_baselines.eval.video_option,
                            video_dir=config.habitat_baselines.video_dir,
                            # Since the final frame is the start frame of the next episode.
                            images=rgb_frames[i][:-1],
                            scene_id=f"{current_episodes_info[i].scene_id}".split('/')[-1].split('.')[0],
                            episode_id=f"{current_episodes_info[i].episode_id}_{ep_eval_count[k]}",
                            checkpoint_idx=checkpoint_index,
                            metrics=extract_scalars_from_info(disp_info),
                            fps=config.habitat_baselines.video_fps,
                            tb_writer=writer,
                            keys_to_include_in_name=config.habitat_baselines.eval_keys_to_include_in_name,
                        )

                        # Since the starting frame of the next episode is the final frame.
                        rgb_frames[i] = rgb_frames[i][-1:]

                    # gfx_str = infos[i].get(GfxReplayMeasure.cls_uuid, "")
                    # if gfx_str != "":
                    #     write_gfx_replay(
                    #         gfx_str,
                    #         config.habitat.task,
                    #         current_episodes_info[i].episode_id,
                    #     )

            not_done_masks = not_done_masks.to(device=device)
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

            # We pause the statefull parameters in the policy.
            # We only do this if there are envs to pause to reduce the overhead.
            # In addition, HRL policy requires the solution_actions to be non-empty, and
            # empty list of envs_to_pause will raise an error.
            if any(envs_to_pause):
                agent.actor_critic.on_envs_pause(envs_to_pause)

        # 最终内存清理
        logger.info("[Memory Cleanup] Evaluation loop completed, performing final cleanup...")
        torch.cuda.empty_cache()
        gc.collect()
        if torch.cuda.is_available():
            logger.info(f"[Memory Cleanup] Final GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            logger.info(f"[Memory Cleanup] Final GPU memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

        # 保存最终的checkpoint
        self._save_eval_checkpoint(checkpoint_path, stats_episodes, ep_eval_count, actions_record)
        logger.info("Final evaluation checkpoint saved.")

        pbar.close()
        assert (
            len(ep_eval_count) >= number_of_eval_episodes
        ), f"Expected {number_of_eval_episodes} episodes, got {len(ep_eval_count)}."

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

        writer.add_scalar(
            "eval_reward/average_reward", aggregated_stats["reward"], step_id
        )

        metrics = {k: v for k, v in aggregated_stats.items() if k != "reward"}
        for k, v in metrics.items():
            writer.add_scalar(f"eval_metrics/{k}", v, step_id)

        # ==== 保存 result.json ====
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

        with open(result_path, "w") as f:
            json.dump(evalai_result, f, indent=2)

        # ==== 保存 actions.json ====
        actions_output_path = os.path.join("output/", "actions.json")
        os.makedirs(os.path.dirname(actions_output_path), exist_ok=True)
        serializable_actions = {
            f"{scene_id}|{episode_id}|{eval_count}": actions
            for (scene_id, episode_id, eval_count), actions in actions_record.items()
        }
        with open(actions_output_path, "w") as f:
            json.dump(serializable_actions, f, indent=2)