#!/usr/bin/env python3

import sys
import os


sys.path.insert(0, "/share/home/u19666033/dhj/DPed_pro/habitat-lab")
sys.path.insert(0, "/share/home/u19666033/dhj/DPed_pro/habitat-baselines")
# ========== 修复完成 ==========

import gc
import random
import importlib
from collections import OrderedDict, defaultdict
from typing import Dict, Tuple

import lmdb
try:
    import msgpack_numpy as _mpn
    _use_msgpack_numpy = True
except Exception:
    _mpn = None
    _use_msgpack_numpy = False
import msgpack
import numpy as np
import torch
import tqdm
from gym import spaces
from habitat import logger
from habitat.config import read_write
from habitat_baselines.common.base_il_trainer import BaseILTrainer
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.habitat_env_factory import HabitatVectorEnvFactory
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from habitat_baselines.common.aux_losses import AuxLosses  # 导入VLN-CE风格的辅助损失管理
from habitat_baselines.utils.common import batch_obs
from habitat_baselines.rl.ddppo.policy.resnet_policy import PointNavResNetPolicy
from habitat_baselines.rl.ddppo.ddp_utils import (
    save_resume_state,
    is_slurm_batch_job,
    SAVE_STATE,
)


class ObservationsDict(dict):
    def pin_memory(self):
        for k, v in self.items():
            self[k] = v.pin_memory()
        return self


def collate_fn(batch):
    def _pad_helper(t, max_len, fill_val=0):
        pad_amount = max_len - t.size(0)
        if pad_amount == 0:
            return t
        pad = torch.full_like(t[0:1], fill_val).expand(
            pad_amount, *t.size()[1:]
        )
        return torch.cat([t, pad], dim=0)

    transposed = list(zip(*batch))
    observations_batch = list(transposed[0])
    prev_actions_batch = list(transposed[1])
    corrected_actions_batch = list(transposed[2])
    weights_batch = list(transposed[3])
    B = len(prev_actions_batch)

    new_observations_batch = defaultdict(list)
    for sensor in observations_batch[0]:
        for bid in range(B):
            new_observations_batch[sensor].append(
                observations_batch[bid][sensor]
            )
    observations_batch = new_observations_batch

    max_traj_len = max(ele.size(0) for ele in prev_actions_batch)
    for bid in range(B):
        for sensor in observations_batch:
            # 对于instruction相关的sensor，使用0填充而不是1.0，避免损坏字符串数据
            if 'instruction' in sensor.lower():
                fill_val = 0.0  # 指令传感器使用0填充
            else:
                fill_val = 1.0  # 其他传感器使用1.0填充（通常是图像等）
            observations_batch[sensor][bid] = _pad_helper(
                observations_batch[sensor][bid], max_traj_len, fill_val=fill_val
            )
        prev_actions_batch[bid] = _pad_helper(prev_actions_batch[bid], max_traj_len)
        corrected_actions_batch[bid] = _pad_helper(
            corrected_actions_batch[bid], max_traj_len
        )
        weights_batch[bid] = _pad_helper(weights_batch[bid], max_traj_len)

    for sensor in observations_batch:
        observations_batch[sensor] = torch.stack(
            observations_batch[sensor], dim=1
        )
        observations_batch[sensor] = observations_batch[sensor].view(
            -1, *observations_batch[sensor].size()[2:]
        )

    prev_actions_batch = torch.stack(prev_actions_batch, dim=1)
    corrected_actions_batch = torch.stack(corrected_actions_batch, dim=1)
    weights_batch = torch.stack(weights_batch, dim=1)
    not_done_masks = torch.ones_like(corrected_actions_batch, dtype=torch.uint8)
    not_done_masks[0] = 0

    observations_batch = ObservationsDict(observations_batch)

    return (
        observations_batch,
        prev_actions_batch.view(-1, 1),
        not_done_masks.view(-1, 1),
        corrected_actions_batch,
        weights_batch,
    )


def _block_shuffle(lst, block_size):
    blocks = [lst[i : i + block_size] for i in range(0, len(lst), block_size)]
    random.shuffle(blocks)
    return [ele for block in blocks for ele in block]


class IWTrajectoryDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        lmdb_features_dir,
        use_iw,
        inflection_weight_coef=1.0,
        lmdb_map_size=1e9,
        batch_size=1,
        preload_multiplier=100,
    ):
        super().__init__()
        # 支持多个LMDB目录（分布式训练时每个rank有独立的目录）
        if isinstance(lmdb_features_dir, list):
            self.lmdb_features_dirs = [d for d in lmdb_features_dir if os.path.exists(d)]
            if len(self.lmdb_features_dirs) == 0:
                raise ValueError(f"No valid LMDB directories found in: {lmdb_features_dir}")
        else:
            self.lmdb_features_dirs = [lmdb_features_dir]
        
        self.lmdb_map_size = lmdb_map_size
        # 限制预加载缓存，避免一次性将所有trajectory留在内存
        multiplier = max(preload_multiplier, 1)
        self.preload_size = max(int(batch_size * multiplier), batch_size)
        self._preload = []
        self.batch_size = batch_size

        if use_iw:
            self.inflec_weights = torch.tensor([1.0, inflection_weight_coef])
        else:
            self.inflec_weights = torch.tensor([1.0, 1.0])

        # 计算所有目录的总条目数，并记录每个目录的条目数和起始索引
        self.length = 0
        self.dir_info = []  # [(dir_path, start_idx, count), ...]
        for lmdb_dir in self.lmdb_features_dirs:
            try:
                with lmdb.open(
                    lmdb_dir,
                    map_size=int(self.lmdb_map_size),
                    readonly=True,
                    lock=False,
                ) as lmdb_env:
                    count = lmdb_env.stat()["entries"]
                    self.dir_info.append((lmdb_dir, self.length, count))
                    self.length += count
            except Exception as e:
                print(f"Warning: Failed to open LMDB directory {lmdb_dir}: {e}", flush=True)
        
        if self.length == 0:
            print(f"Warning: No entries found in any LMDB directories: {self.lmdb_features_dirs}", flush=True)
        else:
            print(f"IWTrajectoryDataset: Found {self.length} total entries across {len(self.dir_info)} LMDB directories", flush=True)
        
        # 分布式训练相关属性
        self._dist_rank = 0
        self._dist_world_size = 1

    def set_distributed_info(self, rank, world_size):
        """设置分布式训练信息，用于在__iter__中进行数据分片"""
        self._dist_rank = rank
        self._dist_world_size = world_size
        print(f"IWTrajectoryDataset: Set distributed info - rank={rank}, world_size={world_size}", flush=True)

    def _get_lmdb_dir_for_index(self, global_idx):
        """根据全局索引找到对应的LMDB目录和局部索引"""
        for lmdb_dir, start_idx, count in self.dir_info:
            if global_idx < start_idx + count:
                local_idx = global_idx - start_idx
                return lmdb_dir, local_idx
        # 如果索引超出范围，返回最后一个目录
        if self.dir_info:
            lmdb_dir, start_idx, count = self.dir_info[-1]
            return lmdb_dir, global_idx - start_idx
        return None, global_idx

    def _load_next(self):
        if len(self._preload) == 0:
            if len(self.load_ordering) == 0:
                raise StopIteration

            new_preload = []
            lengths = []
            loaded_indices = []  # 记录加载的LMDB索引，用于调试
            
            # 按目录分组要加载的索引，避免频繁打开/关闭LMDB
            indices_to_load = []
            for _ in range(self.preload_size):
                if len(self.load_ordering) == 0:
                    break
                global_idx = self.load_ordering.pop()
                lmdb_dir, local_idx = self._get_lmdb_dir_for_index(global_idx)
                indices_to_load.append((global_idx, lmdb_dir, local_idx))
            
            # 按目录分组
            dir_to_indices = {}
            for global_idx, lmdb_dir, local_idx in indices_to_load:
                if lmdb_dir not in dir_to_indices:
                    dir_to_indices[lmdb_dir] = []
                dir_to_indices[lmdb_dir].append((global_idx, local_idx))
            
            # 从每个目录加载数据
            for lmdb_dir, idx_pairs in dir_to_indices.items():
                if lmdb_dir is None:
                    continue
                with lmdb.open(
                    lmdb_dir,
                    map_size=int(self.lmdb_map_size),
                    readonly=True,
                    lock=False,
                ) as lmdb_env, lmdb_env.begin(buffers=True) as txn:
                    for global_idx, local_idx in idx_pairs:
                        loaded_indices.append(global_idx)
                        raw_bytes = txn.get(str(local_idx).encode())
                        if raw_bytes is None:
                            print(f"Warning: No data found for index {local_idx} in {lmdb_dir}", flush=True)
                            continue
                        if _use_msgpack_numpy:
                            data = _mpn.unpackb(raw_bytes, raw=False)
                        else:
                            def _decode_nd(obj):
                                if isinstance(obj, dict) and obj.get("__nd", False):
                                    arr = np.frombuffer(obj["data"], dtype=np.dtype(obj["dtype"]))
                                    return arr.reshape(tuple(obj["shape"]))
                                return obj
                            data = msgpack.unpackb(raw_bytes, object_hook=_decode_nd, raw=False)
                        
                        # 将数据和索引一起存储为元组 (data, idx)
                        new_preload.append((data, global_idx))
                        lengths.append(len(data[0]))

            # 如果所有索引都缺失，new_preload 可能为空
            if len(new_preload) == 0:
                # 如果还有更多索引要加载，继续尝试
                if len(self.load_ordering) > 0:
                    # 递归调用，尝试加载更多数据
                    return self._load_next()
                else:
                    # 没有更多数据了，抛出 StopIteration
                    raise StopIteration("No valid data found in LMDB (all indices missing)")
            
            sort_priority = list(range(len(lengths)))
            random.shuffle(sort_priority)
            sorted_ordering = list(range(len(lengths)))
            sorted_ordering.sort(key=lambda k: (lengths[k], sort_priority[k]))
            for idx in _block_shuffle(sorted_ordering, self.batch_size):
                self._preload.append(new_preload[idx])

        # 检查 _preload 是否为空（防止在并发情况下出现问题）
        if len(self._preload) == 0:
            raise StopIteration("Preload buffer is empty")
        
        # 返回数据和索引的元组
        return self._preload.pop()

    def __next__(self):
        # 循环加载直到找到有效的episode（非默认指令）
        max_skip_attempts = 100  # 最多跳过100个无效episode
        skip_count = 0
        
        while skip_count < max_skip_attempts:
            try:
                loaded_data = self._load_next()
                # 处理新的数据格式：loaded_data是 (data, lmdb_idx) 的元组
                # 其中data是 [obs, prev_actions, oracle_actions] 的列表
                if isinstance(loaded_data, tuple) and len(loaded_data) == 2:
                    data, lmdb_idx = loaded_data
                    obs, prev_actions, oracle_actions = data
                else:
                    # 兼容旧格式（直接是数据列表）
                    obs, prev_actions, oracle_actions = loaded_data
                    lmdb_idx = None
            except StopIteration:
                # 如果没有更多数据，重新抛出异常
                raise
            
            # 检查指令是否有效（不是默认指令或空指令）
            default_instructions = [
                'navigate to the target location.', 
                'navigate to target location', 
                'go to target',
                'navigate to the target location',  # 没有句号的变体
            ]
            has_valid_instruction = False
            
            # 检查第一个step的观察中的指令
            # obs是从LMDB加载的，格式是 transposed_ep[0]，包含整个trajectory的观察
            # 每个传感器可能是 [T, ...] 形状，需要从第一个时间步提取指令
            if isinstance(obs, dict) and len(obs) > 0:
                instruction_keys = ['falcon_instruction', 'agent_0_falcon_instruction']
                instruction = None
                
                for key in instruction_keys:
                    if key in obs:
                        instruction = obs[key]
                        break
                
                # 如果找到了instruction，检查格式
                if instruction is not None:
                    # 处理不同格式的指令数据
                    # instruction可能是：
                    # 1. numpy数组，shape可能是 [T, max_len] (字符串编码为uint8数组)
                    # 2. 列表或元组，可能是 [instruction_string] 或 [[...], [...]] (trajectory格式)
                    # 3. 字符串
                    instr_text = None
                    
                    if isinstance(instruction, np.ndarray):
                        # 如果是numpy数组，需要获取第一个时间步的数据
                        if instruction.ndim > 0:
                            # 获取第一个时间步（如果是trajectory格式 [T, ...]）
                            if instruction.ndim >= 1 and instruction.shape[0] > 0:
                                first_timestep = instruction[0]
                                
                                # 处理第一个时间步的数据
                                if isinstance(first_timestep, np.ndarray) or isinstance(first_timestep, np.generic):
                                    if first_timestep.ndim == 0:
                                        # 标量
                                        instr_text = str(first_timestep.item())
                                    elif first_timestep.ndim == 1:
                                        # 1D数组，可能是字符串编码为uint8数组
                                        try:
                                            # 尝试解码为字符串
                                            if first_timestep.dtype == np.uint8 or first_timestep.dtype == np.int8:
                                                # 找到非零部分的长度（字符串可能被零填充）
                                                non_zero_len = len(first_timestep)
                                                for i in range(len(first_timestep)):
                                                    if first_timestep[i] == 0:
                                                        non_zero_len = i
                                                        break
                                                if non_zero_len > 0:
                                                    instr_text = bytes(first_timestep[:non_zero_len]).decode('utf-8', errors='ignore').strip()
                                                else:
                                                    instr_text = None
                                            else:
                                                # 其他类型，取第一个元素
                                                instr_text = str(first_timestep[0]) if len(first_timestep) > 0 else None
                                        except Exception as e:
                                            # 解码失败，尝试其他方式
                                            try:
                                                instr_text = str(first_timestep[0]) if len(first_timestep) > 0 else None
                                            except:
                                                instr_text = None
                                    else:
                                        # 多维数组，展平后尝试解码
                                        flat_arr = first_timestep.flatten()
                                        try:
                                            if flat_arr.dtype == np.uint8 or flat_arr.dtype == np.int8:
                                                # 找到非零部分
                                                non_zero_indices = np.where(flat_arr != 0)[0]
                                                if len(non_zero_indices) > 0:
                                                    non_zero_len = non_zero_indices[-1] + 1
                                                    instr_text = bytes(flat_arr[:non_zero_len]).decode('utf-8', errors='ignore').strip()
                                                else:
                                                    instr_text = None
                                            else:
                                                instr_text = str(flat_arr[0]) if len(flat_arr) > 0 else None
                                        except:
                                            instr_text = str(flat_arr[0]) if len(flat_arr) > 0 else None
                                else:
                                    # 如果第一个时间步不是数组，直接转换为字符串
                                    instr_text = str(first_timestep)
                            else:
                                # 空数组，尝试直接处理
                                if instruction.size > 0:
                                    flat_arr = instruction.flatten()
                                    try:
                                        if flat_arr.dtype == np.uint8 or flat_arr.dtype == np.int8:
                                            non_zero_indices = np.where(flat_arr != 0)[0]
                                            if len(non_zero_indices) > 0:
                                                non_zero_len = non_zero_indices[-1] + 1
                                                instr_text = bytes(flat_arr[:non_zero_len]).decode('utf-8', errors='ignore').strip()
                                        if instr_text is None:
                                            instr_text = str(flat_arr[0])
                                    except:
                                        instr_text = None
                                else:
                                    instr_text = None
                        else:
                            # 0维数组
                            instr_text = str(instruction.item())
                    elif isinstance(instruction, (list, tuple)):
                        # 如果是列表/元组，可能是 [instruction_string] 或 [[...], [...]] (trajectory格式)
                        if len(instruction) > 0:
                            first_elem = instruction[0]
                            if isinstance(first_elem, str):
                                instr_text = first_elem
                            elif isinstance(first_elem, (list, tuple)) and len(first_elem) > 0:
                                # 嵌套列表，可能是字符串列表或数组列表
                                if isinstance(first_elem[0], str):
                                    instr_text = first_elem[0]
                                else:
                                    instr_text = str(first_elem[0])
                            elif isinstance(first_elem, np.ndarray):
                                # 如果是numpy数组，尝试解码
                                try:
                                    if first_elem.dtype == np.uint8 or first_elem.dtype == np.int8:
                                        flat_arr = first_elem.flatten()
                                        non_zero_indices = np.where(flat_arr != 0)[0]
                                        if len(non_zero_indices) > 0:
                                            non_zero_len = non_zero_indices[-1] + 1
                                            instr_text = bytes(flat_arr[:non_zero_len]).decode('utf-8', errors='ignore').strip()
                                    if instr_text is None:
                                        instr_text = str(first_elem.flat[0]) if first_elem.size > 0 else None
                                except:
                                    instr_text = str(first_elem.flat[0]) if first_elem.size > 0 else None
                            else:
                                instr_text = str(first_elem)
                    elif isinstance(instruction, str):
                        instr_text = instruction
                    else:
                        instr_text = str(instruction)
                    
                    # 检查是否是默认指令、空指令或无法解析
                    if instr_text and instr_text.strip():
                        instr_text_lower = instr_text.lower().strip()
                        # 检查是否是默认指令
                        is_default = False
                        for default_instr in default_instructions:
                            if instr_text_lower == default_instr.lower().strip():
                                is_default = True
                                break
                        
                        if not is_default:
                            has_valid_instruction = True
                            idx_info = f" (LMDB idx: {lmdb_idx})" if lmdb_idx is not None else ""
                            logger.info(f"Dataset loader: ✓ Found valid instruction: '{instr_text[:100] if len(instr_text) > 100 else instr_text}'{idx_info}")
                        else:
                            skip_count += 1
                            logger.warning(f"Dataset loader: ✗ Skipping episode with default instruction: '{instr_text[:100] if len(instr_text) > 100 else instr_text}' (skip {skip_count}/{max_skip_attempts})")
                            continue  # 跳过这个episode，加载下一个
                    else:
                        # 无法解析指令或指令为空，跳过这个episode
                        skip_count += 1
                        logger.warning(f"Dataset loader: Cannot parse instruction or instruction is empty, skipping episode (skip {skip_count}/{max_skip_attempts})")
                        continue  # 跳过这个episode，加载下一个
                else:
                    # 如果没有找到falcon_instruction，跳过这个episode（没有指令的episode不应该用于训练）
                    skip_count += 1
                    logger.warning(f"Dataset loader: No falcon_instruction found in obs, skipping episode (skip {skip_count}/{max_skip_attempts})")
                    continue  # 跳过这个episode，加载下一个
            else:
                # obs不是dict或为空，跳过这个episode
                skip_count += 1
                logger.warning(f"Dataset loader: Invalid obs format (not dict or empty), skipping episode (skip {skip_count}/{max_skip_attempts})")
                continue  # 跳过这个episode，加载下一个
            
            # 如果找到有效指令，退出循环
            if has_valid_instruction:
                break  # 找到有效episode，退出循环
        
        # 如果达到最大跳过次数仍未找到有效episode，抛出异常
        if not has_valid_instruction and skip_count >= max_skip_attempts:
            logger.error(f"Dataset loader: Skipped {max_skip_attempts} episodes but still no valid instruction found. "
                        f"This may indicate all episodes in the dataset have default/empty instructions.")
            # 抛出StopIteration，让DataLoader知道没有更多有效数据
            raise StopIteration("No valid episodes with non-default instructions found after skipping max attempts")
        
        # 转换为tensor
        for k, v in obs.items():
            obs[k] = torch.from_numpy(np.copy(v))
        prev_actions = torch.from_numpy(np.copy(prev_actions))
        oracle_actions = torch.from_numpy(np.copy(oracle_actions))

        # 参考VLN-CE：对于无效动作（-1），权重应该为0
        # 首先检查是否有无效动作
        valid_mask = oracle_actions != -1
        
        # 确保oracle_actions在有效范围内（0到num_actions-1）
        # 这里假设动作空间是Discrete(4)，需要根据实际情况调整
        num_actions = 4  # 默认值，可以从配置或动作空间获取
        valid_range_mask = (oracle_actions >= 0) & (oracle_actions < num_actions)
        valid_mask = valid_mask & valid_range_mask
        
        # 如果发现无效动作，记录警告
        if not valid_mask.all():
            invalid_count = (~valid_mask).sum().item()
            logger.debug(f"Dataset loader: Found {invalid_count} invalid actions in trajectory "
                        f"(out of {len(oracle_actions)}), action range: [{oracle_actions.min().item()}, {oracle_actions.max().item()}]")
            # 将无效动作替换为0（STOP），但权重会设为0
            oracle_actions = torch.where(
                valid_mask,
                oracle_actions,
                torch.zeros_like(oracle_actions)
            )
        
        # 计算inflection weights（动作变化点的权重）
        inflections = torch.cat(
            [
                torch.tensor([1], dtype=torch.long),
                (oracle_actions[1:] != oracle_actions[:-1]).long(),
            ]
        )
        
        # 获取inflection weights
        weights = self.inflec_weights[inflections]
        
        # 对于无效动作（-1或超出范围），权重设为0（参考VLN-CE）
        weights = weights * valid_mask.float()
        
        return (
            obs,
            prev_actions,
            oracle_actions,
            weights,
        )

    def __len__(self):
        """返回数据集长度，供DistributedSampler使用"""
        return self.length

    def __iter__(self):
        # 首先根据分布式rank进行分片（每个rank只处理自己的数据）
        if self._dist_world_size > 1:
            # 分布式训练：每个rank处理不同的数据分片
            per_rank = int(np.ceil(self.length / self._dist_world_size))
            dist_start = per_rank * self._dist_rank
            dist_end = min(dist_start + per_rank, self.length)
        else:
            dist_start = 0
            dist_end = self.length
        
        # 然后根据DataLoader worker进行进一步分片
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            start = dist_start
            end = dist_end
        else:
            dist_length = dist_end - dist_start
            per_worker = int(np.ceil(dist_length / worker_info.num_workers))
            start = dist_start + per_worker * worker_info.id
            end = min(start + per_worker, dist_end)

        # 完全随机shuffle，确保每个epoch的数据顺序都不同
        # 使用当前时间和rank作为随机种子，确保不同rank有不同的顺序
        import time
        random.seed(int(time.time() * 1000) % (2**32) + self._dist_rank)
        indices = list(range(start, end))
        random.shuffle(indices)
        
        # 使用reversed以便从后往前pop（更高效）
        self.load_ordering = list(reversed(indices))
        return self


@baseline_registry.register_trainer(name="dagger")
class DaggerTrainer(BaseILTrainer):
    @staticmethod
    def _ensure_original_episode_id(episode):
        """
        确保 episode 有 original_episode_id 属性。
        如果没有，尝试从 episode.__dict__ 获取或使用 episode_id 作为后备。
        这是一个运行时补丁，用于处理 dynamic_vlnce_dataset.py 修改未生效的情况。
        """
        # 首先尝试直接获取
        original_id = getattr(episode, 'original_episode_id', None)
        
        # 检查是否为空或无效
        if original_id is None or original_id == '' or original_id == 'N/A':
            # 尝试从 __dict__ 获取
            ep_dict = episode.__dict__ if hasattr(episode, '__dict__') else {}
            original_id = ep_dict.get('original_episode_id', None)
            
        # 如果仍然无效，使用 episode_id 作为后备
        if original_id is None or original_id == '' or original_id == 'N/A':
            original_id = getattr(episode, 'episode_id', 'N/A')
            # 尝试动态设置属性
            try:
                episode.original_episode_id = str(original_id)
            except AttributeError:
                pass  # 某些对象可能不允许动态设置属性
        
        return str(original_id) if original_id else 'N/A'

    @staticmethod
    def _extract_instruction_text_from_obs(instruction):
        """
        从observation中提取指令文本，处理多种格式：
        - torch.Tensor (uint8编码的字符串)
        - numpy.ndarray (uint8编码的字符串)
        - list/tuple (可能包含tensor或字符串)
        - str (直接是字符串)
        """
        import numpy as np
        
        if instruction is None:
            return ""
        
        # 如果是tensor，转换为numpy
        if isinstance(instruction, torch.Tensor):
            instruction = instruction.cpu().numpy()
        
        # 如果是list/tuple，取第一个元素
        if isinstance(instruction, (list, tuple)) and len(instruction) > 0:
            instruction = instruction[0]
            # 如果第一个元素还是tensor，继续转换
            if isinstance(instruction, torch.Tensor):
                instruction = instruction.cpu().numpy()
        
        # 如果是numpy数组，尝试解码为字符串
        if isinstance(instruction, np.ndarray):
            # 展平数组
            flat_arr = instruction.flatten()
            
            # 如果是uint8/int8，尝试解码为字符串
            if flat_arr.dtype == np.uint8 or flat_arr.dtype == np.int8:
                # 找到非零部分的长度
                non_zero_indices = np.where(flat_arr != 0)[0]
                if len(non_zero_indices) > 0:
                    non_zero_len = non_zero_indices[-1] + 1
                    try:
                        instr_text = bytes(flat_arr[:non_zero_len]).decode('utf-8', errors='ignore').strip()
                        return instr_text
                    except:
                        pass
                return ""
            else:
                # 其他类型，尝试转换为字符串
                try:
                    if flat_arr.size > 0:
                        return str(flat_arr[0])
                    return ""
                except:
                    return ""
        
        # 如果是字符串，直接返回
        if isinstance(instruction, str):
            return instruction.strip()
        
        # 其他类型，转换为字符串
        try:
            return str(instruction).strip()
        except:
            return ""
    
    def __init__(self, config=None):
        # 保存基础LMDB目录路径（用于后续训练时读取所有rank的数据）
        self.lmdb_features_dir_base = config.habitat_baselines.il.dagger.lmdb_features_dir.format(
            split=config.habitat.dataset.split
        )
        self.lmdb_features_dir = self.lmdb_features_dir_base
        
        # 在调用super().__init__()之前初始化分布式相关属性
        # 因为_make_dirs()可能会调用_is_rank0()，需要这些属性
        self._is_distributed = False
        self._dist_rank = 0
        self._dist_world_size = 1
        self._dist_local_rank = 0
        
        # 保存config，因为_init_distributed可能需要
        self.config = config
        
        # 检查是否启用分布式训练（在super().__init__之前）
        dist_cfg = getattr(config.habitat_baselines.il, "distributed", None)
        if dist_cfg is not None and getattr(dist_cfg, "enabled", False):
            # 先设置基本属性，避免在_init_distributed中访问未初始化的属性
            self._init_distributed(dist_cfg)
            # 分布式训练时，每个rank使用独立的LMDB目录，避免写入锁竞争
            self.lmdb_features_dir = os.path.join(self.lmdb_features_dir_base, f"rank_{self._dist_rank}")
            print(f"[Rank {self._dist_rank}] Using independent LMDB directory: {self.lmdb_features_dir}", flush=True)

        # 分布式模式下，尽早将 GPU 设备写回 config，确保后续 env/model 初始化在对应卡上
        if self._is_distributed:
            from habitat.config import read_write
            with read_write(self.config):
                self.config.habitat_baselines.torch_gpu_id = self._dist_local_rank
                if hasattr(self.config, "habitat") and hasattr(
                    self.config.habitat, "simulator"
                ):
                    sim_cfg = self.config.habitat.simulator
                    if hasattr(sim_cfg, "habitat_sim_v0"):
                        sim_cfg.habitat_sim_v0.gpu_device_id = self._dist_local_rank
        
        # 现在可以安全调用super().__init__()，它会调用_make_dirs()
        super().__init__(config)
        
        # 添加日志文件处理器（所有rank都添加，但只有rank 0会创建文件）
        # 注意：需要在super().__init__()之后调用，因为需要先创建目录
        if hasattr(config.habitat_baselines, 'log_file') and config.habitat_baselines.log_file:
            try:
                # 确保日志目录存在
                log_dir = os.path.dirname(config.habitat_baselines.log_file)
                if log_dir and not os.path.exists(log_dir):
                    os.makedirs(log_dir, exist_ok=True)
                # 所有rank都添加文件处理器，但可能只有rank 0能成功写入
                logger.add_filehandler(config.habitat_baselines.log_file)
                if self._is_distributed:
                    logger.info(f"[Rank {self._dist_rank}] Log file handler added: {config.habitat_baselines.log_file}")
                else:
                    logger.info(f"Log file handler added: {config.habitat_baselines.log_file}")
            except Exception as e:
                # 如果添加失败，记录警告但继续执行
                if self._is_distributed:
                    logger.warning(f"[Rank {self._dist_rank}] Failed to add log file handler: {e}")
                else:
                    logger.warning(f"Failed to add log file handler: {e}")
        
        # 根据分布式设置选择设备
        if self._is_distributed:
            # 分布式训练时，使用local_rank对应的GPU
            # 再次验证设备可用性
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available but distributed training is enabled")
            num_gpus = torch.cuda.device_count()
            if self._dist_local_rank >= num_gpus:
                raise RuntimeError(
                    f"local_rank ({self._dist_local_rank}) >= available GPU count ({num_gpus}). "
                    f"Please check CUDA_VISIBLE_DEVICES setting. "
                    f"Current CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}"
                )
            self.device = torch.device("cuda", self._dist_local_rank)
            torch.cuda.set_device(self._dist_local_rank)
            logger.info(f"Set device to cuda:{self._dist_local_rank} (total available: {num_gpus})")
        else:
            self.device = (
                torch.device("cuda", self.config.habitat_baselines.torch_gpu_id)
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        
        # Checkpoint相关初始化（类似BaseRLTrainer）
        self._last_checkpoint_percent = -1.0
        self._current_epoch = 0
        self._total_epochs = (
            config.habitat_baselines.il.dagger.iterations 
            * config.habitat_baselines.il.epochs
        )
        
        # 验证checkpoint配置（允许同时设置，但优先使用num_checkpoints）
        if (
            config.habitat_baselines.num_checkpoints == -1
            and config.habitat_baselines.checkpoint_interval == -1
        ):
            raise RuntimeError(
                "One of num_checkpoints and checkpoint_interval must be specified"
                " num_checkpoints: {} checkpoint_interval: {}".format(
                    config.habitat_baselines.num_checkpoints,
                    config.habitat_baselines.checkpoint_interval,
                )
            )
        
        # 如果同时设置了两个，优先使用num_checkpoints，并给出警告
        if (
            config.habitat_baselines.num_checkpoints != -1
            and config.habitat_baselines.checkpoint_interval != -1
        ):
            logger.warning(
                "Both num_checkpoints ({}) and checkpoint_interval ({}) are specified. "
                "Will use num_checkpoints (priority).".format(
                    config.habitat_baselines.num_checkpoints,
                    config.habitat_baselines.checkpoint_interval,
                )
            )

    def _init_distributed(self, dist_cfg):
        """初始化分布式训练环境"""
        import torch.distributed as dist
        import os
        
        # 获取分布式参数
        backend = getattr(dist_cfg, "backend", "nccl")
        init_method = getattr(dist_cfg, "init_method", "env://")
        
        # 从环境变量或配置中获取rank和world_size
        if getattr(dist_cfg, "rank", -1) >= 0:
            rank = dist_cfg.rank
        else:
            rank = int(os.environ.get("RANK", -1))
        
        if getattr(dist_cfg, "world_size", -1) > 0:
            world_size = dist_cfg.world_size
        else:
            world_size = int(os.environ.get("WORLD_SIZE", 1))
        
        if getattr(dist_cfg, "local_rank", -1) >= 0:
            local_rank = dist_cfg.local_rank
        else:
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
        
        if rank < 0 or world_size <= 1:
            logger.warning("Distributed training requested but rank/world_size not properly set. Disabling distributed training.")
            return
        
        # 初始化分布式进程组
        if not dist.is_initialized():
            dist.init_process_group(
                backend=backend,
                init_method=init_method,
                rank=rank,
                world_size=world_size
            )
        
        self._is_distributed = True
        self._dist_rank = dist.get_rank()
        self._dist_world_size = dist.get_world_size()
        
        # 优先使用torch.distributed.get_local_rank()获取local_rank，确保与CUDA设备映射一致
        # 如果dist已初始化，使用get_local_rank()；否则使用环境变量或配置值
        if dist.is_initialized():
            try:
                self._dist_local_rank = dist.get_local_rank()
            except (AttributeError, RuntimeError):
                # 如果get_local_rank()不可用，回退到环境变量或配置值
                self._dist_local_rank = local_rank
        else:
            self._dist_local_rank = local_rank
        
        # 验证设备是否可用
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')
            
            # 检查 local_rank 是否在有效范围内
            if self._dist_local_rank >= num_gpus:
                error_msg = (
                    f"Invalid local_rank ({self._dist_local_rank}) for {num_gpus} available GPU(s).\n"
                    f"  - Current process: rank={self._dist_rank}, local_rank={self._dist_local_rank}\n"
                    f"  - World size: {self._dist_world_size}\n"
                    f"  - Available GPUs: {num_gpus}\n"
                    f"  - CUDA_VISIBLE_DEVICES: {cuda_visible_devices}\n"
                    f"\n"
                    f"Possible solutions:\n"
                    f"  1. If you have {self._dist_world_size} GPUs, ensure CUDA_VISIBLE_DEVICES is set correctly\n"
                    f"  2. If you only have {num_gpus} GPU(s), reduce --nproc_per_node to {num_gpus} or less\n"
                    f"  3. Check that CUDA_VISIBLE_DEVICES is set before launching torchrun"
                )
                logger.error(error_msg)
                raise RuntimeError(error_msg)
            
            # 检查 world_size 是否超过可用GPU数量（仅在rank 0时警告）
            if self._dist_rank == 0 and self._dist_world_size > num_gpus:
                logger.warning(
                    f"World size ({self._dist_world_size}) > available GPUs ({num_gpus}). "
                    f"This may cause some processes to fail. "
                    f"CUDA_VISIBLE_DEVICES={cuda_visible_devices}"
                )
        
        logger.info(f"Initialized distributed training: rank={self._dist_rank}, world_size={self._dist_world_size}, local_rank={self._dist_local_rank}, available_gpus={torch.cuda.device_count() if torch.cuda.is_available() else 0}")
    
    def _wrap_model_for_distributed(self):
        """将模型包装为DistributedDataParallel"""
        import torch.distributed as dist
        from torch.nn.parallel import DistributedDataParallel as DDP
        
        dist_cfg = getattr(self.config.habitat_baselines.il, "distributed", None)
        find_unused_params = getattr(dist_cfg, "find_unused_parameters", False)
        broadcast_buffers = getattr(dist_cfg, "broadcast_buffers", True)
        gradient_as_bucket_view = getattr(dist_cfg, "gradient_as_bucket_view", False)
        
        self.policy = DDP(
            self.policy,
            device_ids=[self._dist_local_rank],
            output_device=self._dist_local_rank,
            find_unused_parameters=find_unused_params,
            broadcast_buffers=broadcast_buffers,
            gradient_as_bucket_view=gradient_as_bucket_view,
        )
        
        logger.info(f"Wrapped policy model with DistributedDataParallel")
    
    def _is_rank0(self) -> bool:
        """检查是否是rank 0进程"""
        if not self._is_distributed:
            return True
        import torch.distributed as dist
        return dist.get_rank() == 0
    
    def _get_policy(self):
        """获取实际的policy对象，如果是DDP包装则返回内部的module"""
        if self._is_distributed:
            from torch.nn.parallel import DistributedDataParallel as DDP
            if isinstance(self.policy, DDP):
                return self.policy.module
        return self.policy
    
    def _get_model_state_dict(self):
        """获取模型状态字典，如果是DDP模型则获取内部的module"""
        return self._get_policy().state_dict()
    
    def _make_dirs(self) -> None:
        # 分布式训练时，只有rank 0创建公共目录（checkpoint, results等）
        if self._is_rank0():
            self._make_ckpt_dir()
            if self.config.habitat_baselines.il.eval_save_results:
                self._make_results_dir()
        
        # 每个rank都需要创建自己的LMDB目录（分布式时每个rank独立写入）
        os.makedirs(self.lmdb_features_dir, exist_ok=True)
        if self._is_distributed:
            print(f"[Rank {self._dist_rank}] Created LMDB directory: {self.lmdb_features_dir}", flush=True)
    
    def percent_done(self) -> float:
        """计算训练进度百分比（基于总epoch数）"""
        if self._total_epochs == 0:
            return 0.0
        return self._current_epoch / self._total_epochs
    
    def should_checkpoint(self) -> bool:
        """判断是否应该保存checkpoint（类似BaseRLTrainer）"""
        needs_checkpoint = False
        if self.config.habitat_baselines.num_checkpoints != -1:
            checkpoint_every = (
                1 / self.config.habitat_baselines.num_checkpoints
            )
            if (
                self._last_checkpoint_percent + checkpoint_every
                < self.percent_done()
            ):
                needs_checkpoint = True
                self._last_checkpoint_percent = self.percent_done()
        else:
            # 使用checkpoint_interval（基于epoch数）
            # 确保_current_epoch > 0，避免在训练开始时保存checkpoint
            if self._current_epoch > 0:
                needs_checkpoint = (
                    self._current_epoch
                    % self.config.habitat_baselines.checkpoint_interval
                ) == 0

        return needs_checkpoint
    
    def _should_save_resume_state(self) -> bool:
        """判断是否应该保存resume state（类似BaseRLTrainer）"""
        # 检查是否有信号要求保存状态
        if SAVE_STATE.is_set():
            return True
        
        # 优先从il配置读取（IL训练器应该使用IL配置）
        # 如果没有，则尝试从rl.preemption读取（兼容性支持）
        save_interval = None
        save_state_batch_only = False
        
        # 使用OmegaConf的安全访问方式检查配置
        from omegaconf import OmegaConf
        
        # 首先尝试从IL配置读取
        if OmegaConf.is_config(self.config.habitat_baselines) and 'il' in self.config.habitat_baselines:
            il_config = self.config.habitat_baselines.il
            if OmegaConf.is_config(il_config) and 'save_resume_state_interval' in il_config:
                save_interval = il_config.save_resume_state_interval
                save_state_batch_only = il_config.get('save_state_batch_only', False)
            else:
                # IL配置中没有，使用默认值
                save_interval = 100
                save_state_batch_only = False
        else:
            # 默认值：每100个epoch保存一次
            save_interval = 100
            save_state_batch_only = False
        
        # 如果设置了save_state_batch_only，只在SLURM batch job中保存
        if save_state_batch_only and not is_slurm_batch_job():
            return False
        
        # 根据epoch数判断是否应该保存
        if save_interval > 0 and self._current_epoch > 0:
            return (self._current_epoch % save_interval) == 0
        
        return False

    def _get_spaces(self) -> Tuple[Dict, Dict]:
        """
        仿照 PPO 的 MultiAgentAccessMgr 方式提取 agent_0 的动作空间
        使用 update_dict_with_agent_prefix 和 create_action_space 来自动处理
        """
        # 分布式训练时，需要确保每个进程使用正确的GPU
        if self._is_distributed:
            # 临时设置config中的torch_gpu_id为local_rank
            from habitat.config import read_write
            with read_write(self.config):
                self.config.habitat_baselines.torch_gpu_id = self._dist_local_rank
                self.config.habitat.simulator.habitat_sim_v0.gpu_device_id = self._dist_local_rank
        
        envs = HabitatVectorEnvFactory().construct_envs(self.config)
        observation_space = envs.observation_spaces[0]
        action_space = envs.action_spaces[0]
        orig_action_space = envs.orig_action_spaces[0]
        
        logger.info("=" * 80)
        logger.info("Extracting Agent_0 Action Space (PPO-style)")
        logger.info("=" * 80)
        logger.info(f"Full action space type: {type(action_space)}")
        logger.info(f"Full orig_action_space keys: {list(orig_action_space.spaces.keys()) if isinstance(orig_action_space, spaces.Dict) else 'N/A'}")
        
        # 仿照 MultiAgentAccessMgr._get_agents() 的方式提取 agent_0 的动作空间
        try:
            from habitat_baselines.rl.multi_agent.utils import update_dict_with_agent_prefix
            from habitat.gym.gym_wrapper import create_action_space
            from habitat_baselines.common.env_spec import EnvironmentSpec
            
            # 1. 提取 agent_0 的观察空间
            agent_0_obs_space = spaces.Dict(
                update_dict_with_agent_prefix(observation_space, 0)
            )
            logger.info(f"Agent_0 observation space keys: {list(agent_0_obs_space.spaces.keys())}")
            
            # 2. 提取 agent_0 的原始动作空间（Dict 类型）
            agent_0_orig_action_space = spaces.Dict(
                update_dict_with_agent_prefix(orig_action_space.spaces, 0)
            )
            logger.info(f"Agent_0 original action space keys: {list(agent_0_orig_action_space.spaces.keys())}")
            
            # 3. 使用 create_action_space 将 agent_0 的 Dict 动作空间转换为 Box/Discrete
            agent_0_action_space = create_action_space(agent_0_orig_action_space)
            logger.info(f"Agent_0 action space (after create_action_space): {agent_0_action_space}")
            logger.info(f"  Type: {type(agent_0_action_space)}")
            if isinstance(agent_0_action_space, spaces.Discrete):
                logger.info(f"  Number of actions: {agent_0_action_space.n}")
            elif isinstance(agent_0_action_space, spaces.Box):
                logger.info(f"  Shape: {agent_0_action_space.shape}")
                logger.info(f"  Low: {agent_0_action_space.low}")
                logger.info(f"  High: {agent_0_action_space.high}")
            
            # 4. 创建 agent_0 专用的 EnvironmentSpec（仿照 MultiAgentAccessMgr）
            self._agent_0_env_spec = EnvironmentSpec(
                observation_space=agent_0_obs_space,
                action_space=agent_0_action_space,
                orig_action_space=agent_0_orig_action_space,
            )
            
            # 5. 保存 agent_0 的动作空间信息（用于后续处理）
            if isinstance(agent_0_action_space, spaces.Discrete):
                self._agent_0_info = {
                    'num_actions': agent_0_action_space.n,
                    'action_space_type': 'discrete',
                    'action_space': agent_0_action_space,
                }
            elif isinstance(agent_0_action_space, spaces.Box):
                self._agent_0_info = {
                    'num_actions': agent_0_action_space.shape[0] if len(agent_0_action_space.shape) > 0 else 1,
                    'action_space_type': 'continuous',
                    'action_space': agent_0_action_space,
                }
            else:
                self._agent_0_info = {
                    'num_actions': 4,  # 默认值
                    'action_space_type': 'unknown',
                    'action_space': agent_0_action_space,
                }
            
            logger.info("")
            logger.info(f"✓ Successfully extracted agent_0 action space:")
            logger.info(f"  - Number of actions: {self._agent_0_info['num_actions']}")
            logger.info(f"  - Action space type: {self._agent_0_info['action_space_type']}")
            logger.info(f"  - Action space: {self._agent_0_info['action_space']}")
            logger.info("")
            logger.info("Note: Returning full observation space for policy network (agent_0 actions only)")
            logger.info("=" * 80)
            
            # 重要：返回完整的观察空间（策略网络需要），但使用 agent_0 的动作空间
            # 这是因为策略网络需要处理完整的观察（包括其他 agent 的观察），
            # 但只输出 agent_0 的动作
            envs.close()
            return observation_space, agent_0_action_space
            
        except Exception as e:
            logger.error(f"Failed to extract agent_0 action space using PPO-style method: {e}")
            import traceback
            logger.error(traceback.format_exc())
            logger.warning("Falling back to original observation space and action space")
            
            # 回退到原始方式
            self._agent_0_info = {
                'num_actions': 4,
                'action_space_type': 'discrete',
                'action_space': spaces.Discrete(4),
            }
        envs.close()
        return observation_space, action_space

    def _initialize_policy(self, observation_space, action_space):
        # 从IL配置读取参数
        il_cfg = self.config.habitat_baselines.il
        model_cfg = il_cfg.model
        policy_cfg = il_cfg.policy.agent_0 if hasattr(il_cfg, "policy") and hasattr(il_cfg.policy, "agent_0") else None
        
        # 读取网络架构参数
        hidden_size = int(model_cfg.hidden_size)
        backbone = getattr(model_cfg, "backbone", "resnet18")
        rnn_type = getattr(model_cfg, "rnn_type", "LSTM")
        num_recurrent_layers = getattr(model_cfg, "num_recurrent_layers", 2)
        
        # 读取CLIP文本指令相关参数
        text_encoder_dim = getattr(model_cfg, "text_encoder_dim", 500)
        fusion_method = getattr(model_cfg, "fusion_method", "concat")
        text_instruction_path = None  # 如果需要，可以从配置读取
        
        # 读取辅助损失配置
        aux_cfg = il_cfg.auxiliary_losses if hasattr(il_cfg, "auxiliary_losses") else None
        
        # 读取策略配置
        # PointNavResNetPolicy需要policy_config来正确判断动作类型
        # 如果policy_cfg存在，直接使用；如果不存在，创建默认配置确保使用离散动作
        if policy_cfg is None or not hasattr(policy_cfg, "action_distribution_type"):
            from omegaconf import OmegaConf
            # 创建默认配置，显式指定使用离散动作（categorical）
            policy_config = OmegaConf.create({
                "action_distribution_type": "categorical"
            })
            if policy_cfg is not None and hasattr(policy_cfg, "name"):
                policy_config.name = policy_cfg.name
        else:
            policy_config = policy_cfg
        
        # 排除渲染用相机
        ignore_names = []
        if hasattr(self.config.habitat_baselines, "eval") and hasattr(self.config.habitat_baselines.eval, "extra_sim_sensors"):
            ignore_names = [
                sensor.uuid
                for sensor in self.config.habitat_baselines.eval.extra_sim_sensors.values()
            ]
        
        filtered_obs = spaces.Dict(
            OrderedDict(
                (k, v)
                for k, v in observation_space.items()
                if k not in ignore_names
            )
        )
        
        # 仿照 PPO 的方式：action_space 已经是 agent_0 专用的动作空间
        # 不需要手动转换，直接使用即可
        logger.info("=" * 80)
        logger.info("Initializing Policy with Agent_0 Action Space")
        logger.info("=" * 80)
        logger.info(f"Action space type: {type(action_space)}")
        if isinstance(action_space, spaces.Discrete):
            logger.info(f"  Number of discrete actions: {action_space.n}")
        elif isinstance(action_space, spaces.Box):
            logger.info(f"  Box shape: {action_space.shape}")
            logger.info(f"  Low: {action_space.low}")
            logger.info(f"  High: {action_space.high}")
        logger.info("=" * 80)
        
        # 创建PointNavResNetPolicy
        self.policy = PointNavResNetPolicy(
            observation_space=filtered_obs,
            action_space=action_space,
            hidden_size=hidden_size,
            rnn_type=rnn_type,
            num_recurrent_layers=num_recurrent_layers,
            backbone=backbone,
            normalize_visual_inputs="rgb" in observation_space.spaces,
            force_blind_policy=getattr(self.config.habitat_baselines, "force_blind_policy", False),
            policy_config=policy_config,
            aux_loss_config=aux_cfg,
            fuse_keys=None,
            text_instruction_path=text_instruction_path,
            text_encoder_dim=text_encoder_dim,
            fusion_method=fusion_method,
        ).to(self.device)
        
        # 如果需要加载预训练权重（基础预训练权重，如CLIP）
        pretrained_weights = getattr(model_cfg, "pretrained_weights", None)
        pretrained = getattr(model_cfg, "pretrained", False)
        
        if pretrained and pretrained_weights is not None and os.path.exists(pretrained_weights):
            try:
                checkpoint = torch.load(pretrained_weights, map_location="cpu")
                # 加载权重时，跳过critic相关的键（DAgger不需要）
                state_dict = checkpoint.get("state_dict", checkpoint)
                
                # 过滤掉critic相关的键和可能的其他不需要的键
                filtered_dict = {}
                for k, v in state_dict.items():
                    # 只保留策略网络相关的键
                    if not k.startswith("critic") and not k.startswith("critic_head"):
                        # 如果键名不匹配，尝试适配
                        filtered_dict[k] = v
                
                # 加载权重，允许部分匹配
                missing_keys, unexpected_keys = self.policy.load_state_dict(
                    filtered_dict, strict=False
                )
                if missing_keys:
                    logger.info(f"Missing keys when loading pretrained weights: {missing_keys[:10]}... (showing first 10)")
                if unexpected_keys:
                    logger.info(f"Unexpected keys when loading pretrained weights: {unexpected_keys[:10]}... (showing first 10)")
                logger.info(f"Loaded pretrained weights from {pretrained_weights}")
            except Exception as e:
                logger.warning(f"Failed to load pretrained weights from {pretrained_weights}: {e}")
                logger.warning("Continuing without pretrained weights")
        
        # 如果需要从第一阶段checkpoint加载权重（DAgger迭代阶段）
        dagger_cfg = self.config.habitat_baselines.il.dagger
        load_from_ckpt = getattr(dagger_cfg, "load_from_ckpt", False)
        ckpt_to_load = getattr(dagger_cfg, "ckpt_to_load", None)
        
        if load_from_ckpt and ckpt_to_load is not None and os.path.exists(ckpt_to_load):
            try:
                logger.info(f"Loading checkpoint from previous stage: {ckpt_to_load}")
                checkpoint = torch.load(ckpt_to_load, map_location="cpu")
                
                # 尝试多种可能的checkpoint格式
                state_dict = None
                if "state_dict" in checkpoint:
                    state_dict = checkpoint["state_dict"]
                elif "policy_state_dict" in checkpoint:
                    state_dict = checkpoint["policy_state_dict"]
                elif isinstance(checkpoint, dict) and any(k.startswith("net.") or k.startswith("actor.") for k in checkpoint.keys()):
                    state_dict = checkpoint
                else:
                    # 如果都不匹配，尝试直接使用checkpoint
                    state_dict = checkpoint
                
                # 过滤掉critic相关的键和可能的其他不需要的键
                filtered_dict = {}
                for k, v in state_dict.items():
                    # 处理可能的键名前缀（如"actor."或"net."）
                    new_key = k
                    if k.startswith("actor."):
                        new_key = k[6:]  # 移除"actor."前缀
                    elif k.startswith("net."):
                        new_key = k[4:]  # 移除"net."前缀
                    
                    # 只保留策略网络相关的键
                    if not new_key.startswith("critic") and not new_key.startswith("critic_head"):
                        filtered_dict[new_key] = v
                
                # 加载权重，允许部分匹配
                missing_keys, unexpected_keys = self.policy.load_state_dict(
                    filtered_dict, strict=False
                )
                if missing_keys:
                    logger.info(f"Missing keys when loading checkpoint: {missing_keys[:10]}... (showing first 10)")
                if unexpected_keys:
                    logger.info(f"Unexpected keys when loading checkpoint: {unexpected_keys[:10]}... (showing first 10)")
                logger.info(f"Successfully loaded checkpoint from {ckpt_to_load}")
                
                # 如果checkpoint中包含optimizer状态，可以选择加载（但通常第二阶段重新开始训练）
                if "optimizer_state_dict" in checkpoint:
                    logger.info("Checkpoint contains optimizer state, but will use fresh optimizer for new training stage")
            except Exception as e:
                logger.error(f"Failed to load checkpoint from {ckpt_to_load}: {e}")
                import traceback
                logger.error(traceback.format_exc())
                raise RuntimeError(f"Cannot load checkpoint from {ckpt_to_load}. This is required for second stage training.")
        
        # 优化器
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.policy.parameters()),
            lr=float(il_cfg.optim.lr),
            eps=float(il_cfg.optim.eps),
        )
        self.max_grad_norm = float(il_cfg.optim.max_grad_norm)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="none")
        
        # 如果启用分布式训练，包装模型为DistributedDataParallel
        if self._is_distributed:
            self._wrap_model_for_distributed()

    def _pause_envs(self, envs_to_pause, envs, *args):
        if envs_to_pause is None or len(envs_to_pause) == 0:
            return (envs, *args)
        envs.pause_at(envs_to_pause)
        args = list(args)
        for i, v in enumerate(args):
            if torch.is_tensor(v):
                args[i] = v[~torch.tensor(envs_to_pause)]
        return (envs, *args)

    def _update_dataset(self, data_it):
        if torch.cuda.is_available():
            with torch.cuda.device(self.device):
                torch.cuda.empty_cache()

        # 分布式训练时，每个进程采集一部分数据
        if self._is_distributed:
            total_update_size = self.config.habitat_baselines.il.dagger.update_size
            # 每个进程采集的数据量（向上取整，确保总数不少于update_size）
            per_rank_size = (total_update_size + self._dist_world_size - 1) // self._dist_world_size
            # 当前进程需要采集的数据量（最后一个进程可能少一些）
            if self._dist_rank == self._dist_world_size - 1:
                # 最后一个进程采集剩余的所有数据
                local_update_size = total_update_size - (self._dist_world_size - 1) * per_rank_size
            else:
                local_update_size = per_rank_size
            logger.info(f"[Rank {self._dist_rank}/{self._dist_world_size}] Creating new environments for data collection...")
            logger.info(f"[Rank {self._dist_rank}/{self._dist_world_size}] DAgger iteration {data_it}: Will collect {local_update_size} episodes (total: {total_update_size}, per rank: {per_rank_size})")
        else:
            local_update_size = self.config.habitat_baselines.il.dagger.update_size
            logger.info("Creating new environments for data collection...")
            logger.info(f"DAgger iteration {data_it}: Creating fresh environments for episode collection")
        
        # 在每次DAgger迭代时，确保环境采样不同的episode
        # 通过临时修改配置来影响episode采样
        # 注意：config.habitat.seed 是一个整数，不是对象
        import random
        import numpy as np
        from habitat.config import read_write
        original_seed = None
        if hasattr(self.config.habitat, 'seed'):
            original_seed = self.config.habitat.seed
            # 使用DAgger迭代次数和进程rank来改变随机种子，确保每次迭代和每个进程采样不同的episode
            with read_write(self.config):
                if self._is_distributed:
                    # 分布式训练时，每个进程使用不同的种子，避免采集相同数据
                    new_seed = (original_seed + data_it * 1000 + self._dist_rank * 100) % (2**31 - 1)
                else:
                    new_seed = (original_seed + data_it * 1000) % (2**31 - 1)
                self.config.habitat.seed = new_seed
            if self._is_distributed:
                logger.info(f"[Rank {self._dist_rank}] Temporarily modified seed for episode diversity: {original_seed} -> {self.config.habitat.seed} (data_it={data_it}, rank={self._dist_rank})")
            else:
                logger.info(f"Temporarily modified seed for episode diversity: {original_seed} -> {self.config.habitat.seed}")
            
            # 同时设置Python和NumPy的随机种子，确保环境采样的一致性
            random.seed(self.config.habitat.seed)
            np.random.seed(self.config.habitat.seed)
        
        envs = None
        try:
            # 分布式训练时，确保每个进程使用正确的GPU
            if self._is_distributed:
                from habitat.config import read_write
                with read_write(self.config):
                    self.config.habitat_baselines.torch_gpu_id = self._dist_local_rank
                    self.config.habitat.simulator.habitat_sim_v0.gpu_device_id = self._dist_local_rank
            
            envs = HabitatVectorEnvFactory().construct_envs(self.config)
            logger.info(f"Environments created successfully: {envs.num_envs} environments")
            
            # 确保环境正确初始化 - 先reset一次
            # 给多进程环境一些时间完成初始化（减少延迟以提高速度）
            import time
            time.sleep(0.1)  # 减少初始化延迟
            
            logger.info("Initializing environments with reset...")
            logger.info(f"DAgger iteration {data_it}: Resetting environments (should sample different episodes)")
            
            # 检查并跳过使用默认指令的episode（在开始收集之前）
            max_reset_attempts = 5  # 最多尝试重置5次（减少尝试次数以提高速度）
            reset_attempts = 0
            valid_episode_found = False
            
            # 确保每个环境采样不同的episode（通过多次reset直到获得不同的episode）
            max_episode_diversity_attempts = 10  # 最多尝试10次以获得不同的episode（减少尝试次数以提高速度）
            episode_diversity_attempts = 0
            unique_episode_ids = set()
            
            while reset_attempts < max_reset_attempts and not valid_episode_found:
                observations = envs.reset()
                initial_episodes = envs.current_episodes()
                
                # 检查episode多样性：使用(scene_id, episode_id)组合来判断唯一性
                # 因为episode_id现在是索引，不同场景可能有相同的索引
                def get_episode_key(ep):
                    """获取episode的唯一标识：使用scene_id + episode_id组合"""
                    scene_name = ep.scene_id.split('/')[-1].replace('.basis.glb', '') if hasattr(ep, 'scene_id') else 'unknown'
                    return f"{scene_name}_{ep.episode_id}"
                
                current_ep_keys = [get_episode_key(ep) for ep in initial_episodes]
                if len(set(current_ep_keys)) < len(current_ep_keys):
                    # 有重复的episode，尝试重新reset
                        if episode_diversity_attempts < max_episode_diversity_attempts:
                            episode_diversity_attempts += 1
                            logger.warning(f"Found duplicate episodes (scene_episode): {current_ep_keys}, attempting reset {episode_diversity_attempts}/{max_episode_diversity_attempts} to get diverse episodes...")
                            import time
                            time.sleep(0.05)  # 减少延迟以提高速度
                            continue
                        else:
                            logger.warning(f"After {max_episode_diversity_attempts} attempts, still have duplicate episodes: {current_ep_keys}. Continuing anyway...")
                
                # 参考expert_data_collector_v3.py：检查episode的geodesic_distance，筛选距离足够大的episode
                # 避免初始位置和目标点太近导致episode太短的问题
                min_geodesic_distance = 1.0  # 最小距离阈值（米），参考expert_data_collector使用5.0m，但DAgger可以更宽松
                all_episodes_valid_distance = True
                invalid_distance_episodes = []
                
                for env_idx, ep in enumerate(initial_episodes):
                    try:
                        # 获取geodesic_distance（起始位置到目标点的最短路径距离）
                        geodesic_distance = 0.0
                        if hasattr(ep, 'info') and ep.info:
                            geodesic_distance = ep.info.get("geodesic_distance", 0.0)
                        elif hasattr(ep, 'distance_to_goal'):
                            # 如果没有info，尝试从episode属性获取
                            geodesic_distance = ep.distance_to_goal
                        
                        # 检查距离是否有效
                        if (geodesic_distance <= min_geodesic_distance or 
                            np.isnan(geodesic_distance) or 
                            np.isinf(geodesic_distance) or
                            geodesic_distance <= 0.0):
                            all_episodes_valid_distance = False
                            scene_name = ep.scene_id.split('/')[-1].replace('.basis.glb', '') if hasattr(ep, 'scene_id') else 'unknown'
                            invalid_distance_episodes.append({
                                'env_idx': env_idx,
                                'episode_id': ep.episode_id,
                                'scene': scene_name,
                                'distance': geodesic_distance
                            })
                            if self._is_rank0():
                                logger.warning(f"Env {env_idx}: Episode {ep.episode_id} (scene: {scene_name}) has invalid geodesic_distance: {geodesic_distance:.3f}m <= {min_geodesic_distance}m, will reset to find new episode")
                        else:
                            if self._is_rank0() and reset_attempts == 0:
                                scene_name = ep.scene_id.split('/')[-1].replace('.basis.glb', '') if hasattr(ep, 'scene_id') else 'unknown'
                                logger.debug(f"Env {env_idx}: Episode {ep.episode_id} (scene: {scene_name}) has valid geodesic_distance: {geodesic_distance:.3f}m")
                    except Exception as e:
                        if self._is_rank0():
                            logger.warning(f"Failed to check geodesic_distance for env {env_idx}: {e}")
                        all_episodes_valid_distance = False
                        invalid_distance_episodes.append({
                            'env_idx': env_idx,
                            'episode_id': getattr(ep, 'episode_id', 'N/A'),
                            'scene': 'unknown',
                            'distance': 0.0,
                            'error': str(e)
                        })
                
                # 如果有episode距离太近，重置环境寻找新的episode
                if not all_episodes_valid_distance:
                    reset_attempts += 1
                    if self._is_rank0():
                        logger.info(f"Found {len(invalid_distance_episodes)} episode(s) with invalid geodesic_distance, resetting (attempt {reset_attempts}/{max_reset_attempts})...")
                        for invalid_ep in invalid_distance_episodes:
                            logger.debug(f"  - Env {invalid_ep['env_idx']}: Episode {invalid_ep['episode_id']} (scene: {invalid_ep['scene']}), distance: {invalid_ep['distance']:.3f}m")
                    continue
                
                # 检查初始episode的指令
                default_instructions = ['navigate to the target location.', 'navigate to target location', 'go to target']
                has_valid_instruction = False
                
                if len(observations) > 0 and isinstance(observations[0], dict):
                    first_obs = observations[0]
                    # 检查可能的指令传感器键名
                    instruction_keys = ['falcon_instruction', 'agent_0_falcon_instruction']
                    instruction = None
                    
                    # 调试：打印可用的键（仅在第一次或找不到指令时）
                    if reset_attempts == 0 or instruction is None:
                        available_keys = list(first_obs.keys())
                        logger.debug(f"Available observation keys: {available_keys[:30]}... (showing first 30)")
                        similar_keys = [k for k in available_keys if 'instruction' in k.lower() or 'instr' in k.lower()]
                        if similar_keys:
                            logger.info(f"Found keys that might be instructions: {similar_keys}")
                    
                    for key in instruction_keys:
                        if key in first_obs:
                            instruction = first_obs[key]
                            logger.debug(f"Found instruction using key: {key}, type: {type(instruction)}")
                            break
                    
                    if instruction is not None:
                        # 提取指令文本（处理tensor格式）
                        instr_text = self._extract_instruction_text_from_obs(instruction)
                        logger.debug(f"Extracted instruction text (first 100 chars): '{instr_text[:100]}'")
                        
                        # 检查是否是默认指令
                        instr_text_lower = instr_text.lower().strip()
                        if instr_text_lower and instr_text_lower not in [d.lower() for d in default_instructions]:
                            has_valid_instruction = True
                            valid_episode_found = True
                            logger.info(f"Found valid episode with instruction: {instr_text[:80]}...")
                        else:
                            if not instr_text_lower:
                                logger.warning(f"Episode has empty instruction, resetting to find new episode...")
                            else:
                                logger.warning(f"Episode uses default instruction: '{instr_text}', resetting to find new episode...")
                            reset_attempts += 1
                            continue
                    else:
                        # 如果没有找到falcon_instruction，打印调试信息
                        available_keys = list(first_obs.keys())
                        logger.warning(f"No falcon_instruction found in observations. Available keys: {available_keys[:30]}... (showing first 30)")
                        similar_keys = [k for k in available_keys if 'instruction' in k.lower() or 'instr' in k.lower()]
                        if similar_keys:
                            logger.warning(f"Found similar keys that might be instructions: {similar_keys}. Consider checking the sensor configuration.")
                        # 假设有效（可能是数据集没有指令，或者使用其他格式）
                        has_valid_instruction = True
                        valid_episode_found = True
                        logger.info("No falcon_instruction found, assuming valid episode (may need to check sensor configuration)")
                
                if has_valid_instruction:
                    valid_episode_found = True
            
            if not valid_episode_found:
                logger.warning(f"After {max_reset_attempts} attempts, still using default instruction episode. Continuing anyway...")
            
            initial_episodes = envs.current_episodes()
            logger.info(f"Environments initialized successfully")
            logger.info(f"Initial episode IDs in iteration {data_it}: {[ep.episode_id for ep in initial_episodes]}")
            
            # 恢复原始种子（如果需要）
            if original_seed is not None:
                with read_write(self.config):
                    self.config.habitat.seed = original_seed
                # logger.debug(f"Restored original seed: {original_seed}")
            
            # 再次短暂延迟，确保reset后的状态稳定（减少延迟以提高速度）
            time.sleep(0.1)
            
        except Exception as e:
            logger.error(f"Failed to create or initialize environments: {e}")
            import traceback
            logger.error(traceback.format_exc())
            if envs is not None:
                try:
                    envs.close()
                except:
                    pass
            raise

        # 尝试多种可能的GT action传感器名称
        expert_uuid_candidates = [
            "agent_0_falcon_gt_action",
            "falcon_gt_action",
            "gt_action_sensor"
        ]
        expert_uuid = None
        # 从第一个观测中确定实际使用的expert_uuid
        first_obs = batch_obs(observations, device=self.device)
        for candidate in expert_uuid_candidates:
            if candidate in first_obs:
                expert_uuid = candidate
                logger.info(f"Using expert action sensor: {expert_uuid}")
                break
        
        if expert_uuid is None:
            logger.warning(f"No expert action sensor found in observations. Available keys: {list(first_obs.keys())}")
            expert_uuid = "falcon_gt_action"  # 使用默认值作为后备

        # 获取实际的policy对象（处理DDP包装）
        policy = self._get_policy()
        rnn_states = torch.zeros(
            envs.num_envs,
            policy.num_recurrent_layers,
            policy.recurrent_hidden_size,
            device=self.device,
        )
        prev_actions = torch.zeros(
            envs.num_envs,
            1,
            device=self.device,
            dtype=torch.long,
        )
        not_done_masks = torch.zeros(envs.num_envs, 1, dtype=torch.uint8, device=self.device)

        # observations已经在上面reset时获取了
        batch = batch_obs(observations, self.device)
        
        # 打印当前episode信息（仅在第一个环境reset时）
        current_episode_list = envs.current_episodes()
        if len(current_episode_list) > 0:
            ep0 = current_episode_list[0]
            logger.info(f"=== Episode Info (First Reset) ===")
            logger.info(f"Episode ID (index): {ep0.episode_id}")
            
            # 使用辅助函数确保 original_episode_id 存在并获取
            original_id = self._ensure_original_episode_id(ep0)
            logger.info(f"Original Episode ID: {original_id}")
            
            # 打印gt_action信息（专家轨迹动作）
            gt_action = getattr(ep0, 'gt_action', None)
            gt_len = len(gt_action) if gt_action else 0
            logger.info(f"GT Action Length: {gt_len}")
            if gt_action and len(gt_action) > 0:
                logger.info(f"GT Action (first 5): {gt_action[:5]}")
            else:
                # 打印episode的所有属性，帮助调试
                ep_attrs = list(ep0.__dict__.keys()) if hasattr(ep0, '__dict__') else []
                logger.warning(f"GT Action is empty! Episode attrs: {ep_attrs[:10]}...")
            
            # 获取scene_id
            if hasattr(ep0, 'scene_id'):
                logger.info(f"Scene ID: {ep0.scene_id}")
            
            # 额外检查gt_action详细信息
            if gt_action is not None and len(gt_action) > 0:
                logger.info(f"gt_action type: {type(gt_action)}, first 5 values: {gt_action[:5]}")
            elif 'gt_action' in (ep0.__dict__ if hasattr(ep0, '__dict__') else {}):
                logger.info(f"gt_action exists in episode but is None (field is defined but value is None)")
            else:
                logger.warning(f"Episode does not have gt_action attribute (field not found in episode object)")
 
        episodes = [[] for _ in range(envs.num_envs)]
        skips = [False for _ in range(envs.num_envs)]
        dones = [False for _ in range(envs.num_envs)]

        p = self.config.habitat_baselines.il.dagger.p
        # 统一使用p^iteration衰减，不再区分第一轮和后续迭代
        # 第一轮（data_it=0）时，p^0 = 1.0，完全使用专家动作（随机均匀采样）
        # 后续迭代：p^iteration衰减，逐渐减少专家动作的使用
        beta = 0.0 if p == 0.0 else p ** data_it
        logger.info(f"DAgger iteration {data_it}: Using beta={beta} (p={p}^{data_it})")
        
        # 不再强制收集所有不重复的episode，统一收集update_size条
        ensure_unique_episodes = False
        
        # 获取force_diverse_episodes配置（默认False，允许策略交互生成新数据）
        force_diverse_episodes = getattr(self.config.habitat_baselines.il.dagger, 'force_diverse_episodes', False)
        
        if self._is_distributed:
            logger.info(f"[Rank {self._dist_rank}/{self._dist_world_size}] DAgger iteration {data_it}: force_diverse_episodes={force_diverse_episodes}, will collect {local_update_size} episodes")
        else:
            logger.info(f"DAgger iteration {data_it}: force_diverse_episodes={force_diverse_episodes}, will collect {self.config.habitat_baselines.il.dagger.update_size} episodes")

        collected_eps = 0
        lmdb_index = 0  # 独立的LMDB索引计数器，即使episode被跳过也递增，确保索引连续
        ep_ids_collected = None
        skipped_scenes_counter = defaultdict(int)  # 统计每个scene被跳过的次数
        
        # 辅助函数：获取episode的唯一标识（scene_name + episode_id）
        def get_episode_unique_key(ep):
            """获取episode的唯一标识：使用scene_name + episode_id组合"""
            scene_name = ep.scene_id.split('/')[-1].replace('.basis.glb', '') if hasattr(ep, 'scene_id') else 'unknown'
            return f"{scene_name}_{ep.episode_id}"
        
        # 如果需要跟踪已收集的episode（force_diverse_episodes=True），初始化集合
        if force_diverse_episodes:
            ep_ids_collected = set()
            # 将当前初始episode加入已收集集合
            initial_episodes = envs.current_episodes()
            for ep in initial_episodes:
                ep_ids_collected.add(get_episode_unique_key(ep))
            logger.info(f"Initial episodes: {[get_episode_unique_key(ep) for ep in initial_episodes]}")
            logger.info(f"Will try to collect diverse episodes (tracking {len(ep_ids_collected)} initial episode keys)")

        # 打印数据收集阶段信息
        if self._is_rank0():
            logger.info("=" * 80)
            logger.info("DAgger Data Collection Phase (Distributed)")
            logger.info("=" * 80)
            logger.info(f"Total target episodes: {self.config.habitat_baselines.il.dagger.update_size}")
            logger.info(f"World size: {self._dist_world_size}")
            logger.info(f"Episodes per rank: ~{local_update_size}")
            logger.info(f"Max steps per episode: 500")
            logger.info("=" * 80)
        else:
            logger.info(f"[Rank {self._dist_rank}] Starting data collection: {local_update_size} episodes")
        
        # 使用本地update_size作为目标数量
        initial_total = local_update_size
        
        # 分布式训练时，每个进程使用不同的LMDB起始索引
        # 先获取当前LMDB的总条目数，然后每个进程使用不同的偏移量
        with lmdb.open(
            self.lmdb_features_dir,
            map_size=int(self.config.habitat_baselines.il.dagger.lmdb_map_size),
            readonly=True,
        ) as temp_lmdb_env:
            base_start_id = temp_lmdb_env.stat()["entries"]
        
        if self._is_distributed:
            # 每个进程使用不同的起始索引，避免冲突
            # rank 0: base_start_id + 0 * per_rank_size
            # rank 1: base_start_id + 1 * per_rank_size
            # ...
            per_rank_size_for_index = (self.config.habitat_baselines.il.dagger.update_size + self._dist_world_size - 1) // self._dist_world_size
            start_id = base_start_id + self._dist_rank * per_rank_size_for_index
            logger.info(f"[Rank {self._dist_rank}] Using LMDB start_id: {start_id} (base: {base_start_id}, offset: {self._dist_rank * per_rank_size_for_index})")
        else:
            start_id = base_start_id
        
        # 在分布式训练中，每个rank独立显示自己的进度条
        if self._is_distributed:
            # 每个rank显示自己的进度
            desc = f"[Rank {self._dist_rank}/{self._dist_world_size}] Collecting episodes"
            show_pbar = True  # 每个rank都显示自己的进度条
            pbar_total = local_update_size
        else:
            desc = "Collecting episodes"
            show_pbar = True
            pbar_total = initial_total
        
        with tqdm.tqdm(
            total=pbar_total,
            dynamic_ncols=True,
            desc=desc,
            unit="episode",
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}',
            disable=not show_pbar,
            position=0,  # 确保进度条在固定位置
            leave=True,
            mininterval=0.3,  # 最小更新间隔0.3秒，减少更新频率但保持实时性
        ) as pbar, lmdb.open(
            self.lmdb_features_dir,
            map_size=int(self.config.habitat_baselines.il.dagger.lmdb_map_size),
        ) as lmdb_env, torch.no_grad():
            txn = lmdb_env.begin(write=True)

            # 添加计数器，防止无限循环（如果所有episode都被跳过）
            steps_since_last_save = 0
            max_steps_without_save = 1000  # 如果1000步都没有保存episode，记录警告

            # 只在开始时打印一次循环条件
            if collected_eps == 0:
                if self._is_distributed:
                    logger.info(f"[Rank {self._dist_rank}] Starting collection loop: local_update_size={local_update_size}, "
                              f"data_it={data_it}, force_diverse_episodes={force_diverse_episodes}")
                else:
                    logger.info(f"Starting collection loop: update_size={self.config.habitat_baselines.il.dagger.update_size}, "
                              f"data_it={data_it}, force_diverse_episodes={force_diverse_episodes}")
            # 统一收集local_update_size条episode（分布式时每个进程收集一部分）
            while collected_eps < local_update_size and envs.num_envs > 0:
                steps_since_last_save += 1
                last_collected_eps = collected_eps
                current_episodes = None
                envs_to_pause = None
                if force_diverse_episodes:
                    envs_to_pause = []
                    current_episodes = envs.current_episodes()

                # 检查是否超过最大步数限制（500步）或gt_actions结束，强制结束episode
                max_action_length = 500
                for i in range(envs.num_envs):
                    if dones[i]:
                        continue  # 已经done的环境跳过
                    
                    current_episode_step = len(episodes[i])
                    
                    # 检查gt_actions是否已结束
                    # 注意：应该使用实际的episode长度（找到第一个stop动作的位置），而不是整个列表的长度
                    current_ep_info = envs.current_episodes()
                    gt_action_length = 0
                    if i < len(current_ep_info):
                        ep = current_ep_info[i]
                        gt_actions_list = getattr(ep, 'gt_action', [])
                        if gt_actions_list:
                            # 找到第一个stop动作（0）的位置，后续都是填充
                            # 如果没有找到stop动作，使用整个列表的长度
                            gt_action_length = len(gt_actions_list)
                            for idx, act in enumerate(gt_actions_list):
                                if act == 0:
                                    gt_action_length = idx + 1  # 包含stop动作
                                    break
                    
                    # 如果gt_actions已结束（current_episode_step >= gt_action_length 且 gt_action_length > 0），强制结束
                    # 但只有在实际执行了足够步骤后才结束，避免过早结束
                    if gt_action_length > 0 and current_episode_step >= gt_action_length:
                        dones[i] = True
                        if i < len(skips):
                            skips[i] = False
                        logger.info(f"Env {i} reached gt_action_length {gt_action_length} (current_step={current_episode_step}), forcing episode end")
                    # 或者超过最大步数限制（200步）
                    elif current_episode_step >= max_action_length:
                        # 强制标记为完成，并保存episode
                        dones[i] = True
                        # 重要：强制结束的episode不应该被跳过，立即设置skips[i] = False
                        # 这样在后续的保存检查中就能正确保存
                        if i < len(skips):
                            skips[i] = False
                        logger.info(f"Env {i} reached max_action_length {max_action_length} (current_step={current_episode_step}), forcing episode end, skips[{i}]=False")

                # 注意：保存检查移到step()之后，在skips计算之后执行
                # 这里先跳过，稍后在step()之后处理

                # 如果需要跟踪已收集的episode，检查并暂停重复的环境
                if force_diverse_episodes and current_episodes is not None and ep_ids_collected is not None:
                    # 检查当前episode是否重复（使用scene_name + episode_id组合）
                    for i, ep in enumerate(current_episodes):
                        ep_key = get_episode_unique_key(ep)
                        if ep_key in ep_ids_collected:
                            if i not in envs_to_pause:
                                envs_to_pause.append(i)
                    
                    # 如果有需要暂停的环境，执行暂停
                    if envs_to_pause:
                        (
                            envs,
                            rnn_states,
                            not_done_masks,
                            prev_actions,
                            batch,
                        ) = self._pause_envs(
                            envs_to_pause,
                            envs,
                            rnn_states,
                            not_done_masks,
                            prev_actions,
                            batch,
                        )
                        # 如果所有环境都被暂停，退出收集循环
                        if envs.num_envs == 0:
                            logger.info("All environments paused (duplicate episodes), exiting collection loop")
                            break

                # 获取实际的policy对象（处理DDP包装）
                policy = self._get_policy()
                action_data = policy.act(
                    batch,
                    rnn_states,
                    prev_actions,
                    not_done_masks,
                    deterministic=False,
                )
                actions = action_data.actions
                
                # 调试信息：打印形状和动作值范围
                # if len(episodes[0]) == 0:  # 仅第一次打印
                #     logger.info(f"DEBUG: action_data.actions shape: {actions.shape}")
                #     logger.info(f"DEBUG: prev_actions shape: {prev_actions.shape}")
                #     logger.info(f"DEBUG: actions value range: min={actions.min().item()}, max={actions.max().item()}")
                #     logger.info(f"DEBUG: actions dtype: {actions.dtype}")
                #     # 如果是连续值，打印前几个值
                #     if actions.numel() > 0:
                #         logger.info(f"DEBUG: first few action values: {actions.flatten()[:min(10, actions.numel())].tolist()}")
                
                expert_actions_raw = batch.get(expert_uuid, None)
                expert_actions_for_where = None
                
                # 获取每个环境的当前episode步数（用于索引到正确的动作）
                # 注意：不同环境的步数可能不同
                current_steps = [len(episodes[i]) if i < len(episodes) else 0 for i in range(envs.num_envs)]
                current_step = current_steps[0] if len(current_steps) > 0 else 0  # 用于兼容旧代码
                
                # 调试：打印当前步数（仅在第一次或有问题时）
                # 移除频繁的DEBUG日志
                # if len(episodes[0]) == 0 or current_step >= 200:
                #     logger.debug(f"current_steps={current_steps}, current_step={current_step}, len(episodes[0])={len(episodes[0])}")
                
                if expert_actions_raw is not None:
                    if len(episodes[0]) == 0:  # 仅第一次打印
                        # logger.info(f"DEBUG: expert_actions_raw shape: {expert_actions_raw.shape}")
                        # 打印当前episode信息
                        current_ep_info = envs.current_episodes()
                        if len(current_ep_info) > 0:
                            ep = current_ep_info[0]
                            # 只在第一次或每100个episode打印一次
                            if collected_eps == 0 or collected_eps % 100 == 0:
                                logger.debug(f"Current episode ID: {ep.episode_id}")
                                if hasattr(ep, 'scene_id'):
                                    logger.debug(f"Scene ID: {ep.scene_id}")
                            # 检查原始gt_action
                            if hasattr(ep, 'gt_action'):
                                gt_act = getattr(ep, 'gt_action', None)
                                if gt_act is not None and isinstance(gt_act, (list, tuple)):
                                    # 只在DEBUG级别记录详细信息
                                    logger.debug(f"Episode gt_action length: {len(gt_act)}")
                        # 打印前几个时间步的动作值
                        # 移除频繁的DEBUG日志
                        # if expert_actions_raw.dim() == 3:
                        #     logger.debug(f"First 5 timesteps of expert_actions from sensor: {expert_actions_raw[0, :5, :].cpu().numpy()}")
                    
                    # 保存原始expert_actions用于后续处理（保存原始形状用于提取标量值）
                    expert_actions = expert_actions_raw.clone()
                    
                    # 确保expert_actions和actions形状匹配
                    if expert_actions.shape != actions.shape:
                        # 这是预期的形状转换（GT_ActionSensor返回[batch_size, 500]），改为debug级别
                        if current_step == 0:
                            logger.debug(f"expert_actions shape {expert_actions.shape} != actions shape {actions.shape}, reshaping to match (expected behavior)")
                        
                        # 特殊处理：如果expert_actions是[batch, seq_len]形状
                        # GT_ActionSensor返回形状为(500,)，batch_obs后变成(batch_size, 500)
                        # 其中500是序列长度，包含action_id序列（注意：vln_sensors中action为0后填充的也是0，但trainer中需要填充为-1）
                        # 填充到长度为500就可以了，超过500步应该使用填充值-1
                        if expert_actions.dim() == 2:
                            # 形状如[batch_size, 500]
                            # 注意：这里我们使用第一个环境的步数（current_step）来提取 expert action
                            # 因为 expert_actions 是 batch 级别的，所有环境共享同一个 batch
                            # 实际每个环境的 expert action 应该在后续循环中根据 current_episode_step 提取
                            max_action_length = expert_actions.shape[1]  # 通常是500
                            # 对于 batch 级别的 expert_actions_for_where，我们使用第一个环境的步数
                            # 但实际保存时会在每个环境的循环中使用正确的步数
                            # 重要：current_step 应该是当前步数（在添加当前step之前），所以应该是 len(episodes[0])
                            # 如果 current_step >= 500，说明已经达到最大长度，使用填充值
                            
                            # 调试：打印 current_step 的值
                            # if current_step == 0:
                            #     logger.debug(f"DEBUG: current_step={current_step}, max_action_length={max_action_length}, len(episodes[0])={len(episodes[0])}")
                            
                            if current_step >= max_action_length:
                                # 超过500步，使用填充值
                                expert_actions_for_where = torch.full_like(actions, -1, dtype=torch.long)
                                # if current_step < 10 or current_step % 10 == 0:
                                #     logger.debug(f"Env 0 Step {current_step} >= max_action_length {max_action_length}, using padding value -1")
                            else:
                                # 在有效范围内（0-499），直接提取对应时间步的action_id（无需取[:,0]，因为已经是1D）
                                step_idx = current_step
                                expert_action_id = expert_actions[:, step_idx:step_idx+1].view(actions.shape)
                                
                                # 打印当前使用的动作值（仅前10步或每10步打印）
                                if current_step < 10 or current_step % 10 == 0:
                                    action_value = expert_action_id[0, 0].item()
                                    logger.debug(f"Env 0 Step {current_step}: Using expert_action[{step_idx}] = {action_value}")
                                
                                expert_actions_for_where = expert_action_id
                        elif expert_actions.dim() == 1:
                            # 形状如[500]，需要unsqueeze以匹配actions形状
                            expert_actions_for_where = expert_actions.unsqueeze(-1) if expert_actions.shape[0] > 1 else expert_actions.view(actions.shape)
                        elif expert_actions.numel() == actions.numel():
                            # 如果元素总数相同，可以reshape
                            expert_actions_for_where = expert_actions.view(actions.shape)
                        elif expert_actions.dim() > actions.dim():
                            # 如果expert_actions维度更高，squeeze多余的维度
                            expert_actions_reshaped = expert_actions
                            while expert_actions_reshaped.dim() > actions.dim() and expert_actions_reshaped.shape[-1] == 1:
                                expert_actions_reshaped = expert_actions_reshaped.squeeze(-1)
                            if expert_actions_reshaped.shape != actions.shape:
                                # 如果还是不匹配，尝试展平后取前N个元素
                                expert_actions_for_where = expert_actions_reshaped.flatten()[:actions.numel()].view(actions.shape)
                            else:
                                expert_actions_for_where = expert_actions_reshaped
                        else:
                            # 扩展维度以匹配
                            expert_actions_reshaped = expert_actions
                            while expert_actions_reshaped.dim() < actions.dim():
                                expert_actions_reshaped = expert_actions_reshaped.unsqueeze(-1)
                            expert_actions_for_where = expert_actions_reshaped
                    else:
                        expert_actions_for_where = expert_actions
                    
                    if expert_actions_for_where is not None:
                        # 参考VLN-CE的处理方式：检查是否有无效动作（-1）
                        # 如果有-1，应该使用skips机制，而不是直接用于torch.where
                        skips_mask = expert_actions_for_where.long() == -1
                        
                        # 调试：打印 skips_mask 的信息
                        # if skips_mask.any():
                        #     logger.debug(f"DEBUG: skips_mask has {skips_mask.sum().item()} True values out of {skips_mask.numel()}")
                        
                        if skips_mask.any():
                            # 如果有无效动作，先处理skips（类似VLN-CE）
                            # 对于无效动作，使用STOP（0）而不是-1
                            expert_actions_for_where = torch.where(
                                skips_mask,
                                torch.zeros_like(expert_actions_for_where),
                                expert_actions_for_where
                            )
                            # 注意：skips_mask 会在后续用于权重计算
                        
                        # 只有在expert_actions不是-1时才用于DAgger的beta采样
                        # 对于无效动作（-1），应该使用策略动作，而不是expert动作
                        valid_expert_mask = ~skips_mask if skips_mask.any() else torch.ones_like(actions, dtype=torch.bool)
                        
                        # DAgger采样：以beta概率使用expert动作，否则使用策略动作
                        # 但对于无效动作（-1），始终使用策略动作
                        actions = torch.where(
                            (torch.rand_like(actions, dtype=torch.float) < beta) & valid_expert_mask,
                            expert_actions_for_where.long(),
                            actions,
                        )
                else:
                    expert_actions = None

                for i in range(envs.num_envs):
                    # 处理expert_actions: 根据实际形状提取标量动作值
                    # GT_ActionSensor返回的是(500,)形状，经过batch_obs变成(batch_size, 500)
                    # 需要根据当前步数提取对应时间步的action_id
                    current_episode_step = len(episodes[i])
                    
                    # 调试：打印当前步数（仅在第一次或有问题时）
                    # 移除频繁的DEBUG日志
                    # if i == 0 and (current_episode_step == 0 or current_episode_step >= 500):
                    #     logger.debug(f"Env {i}, len(episodes[{i}])={len(episodes[i])}, current_episode_step={current_episode_step}")
                    
                    # 初始化expert_action_value为默认值（如果后续没有设置，使用-1）
                    expert_action_value = -1
                    
                    if expert_actions is not None:
                        try:
                            expert_slice = expert_actions[i]  # 形状应该是[500]
                            # 调试信息：仅第一次打印形状信息
                            # if i == 0 and len(episodes[0]) == 0:
                            #     logger.debug(f"expert_actions shape: {expert_actions.shape}, expert_actions[{i}] shape: {expert_slice.shape}")
                            #     logger.info(f"DEBUG: expert_actions[{i}] first 10 timesteps: {expert_slice[:10].cpu().numpy()}")
                            
                            # 根据GT_ActionSensor的定义，形状是(seq_len,)，只包含action_id
                            # 填充到长度为500就可以了，超过500步应该直接使用填充值-1
                            if expert_slice.dim() == 1:
                                # 形状为[500]，只包含action_id
                                max_action_length = expert_slice.shape[0]  # 通常是500
                                
                                # 调试：仅在第一次或有问题时打印详细信息
                                if i == 0 and (current_episode_step == 0 or current_episode_step >= max_action_length):
                                    logger.debug(f"Env {i} extracting expert action: current_episode_step={current_episode_step}, max_action_length={max_action_length}, expert_slice.shape={expert_slice.shape}")
                                
                                # 如果当前步数 >= 500，直接使用填充值，不需要读取传感器
                                if current_episode_step >= max_action_length:
                                    expert_action_value = -1
                                    is_padding = True
                                    if current_episode_step < 10 or current_episode_step % 10 == 0:
                                        logger.info(f"Env {i} Step {current_episode_step}: >= max_action_length {max_action_length}, using padding value -1")
                                else:
                                    # 在有效范围内（0-499），读取传感器数据
                                    step_idx = current_episode_step
                                    if i == 0 and (step_idx < 5 or step_idx % 10 == 0):
                                        logger.debug(f"Env {i} Step {step_idx}: Extracting expert_action from expert_slice[{step_idx}]")
                                    expert_action_id = expert_slice[step_idx].item()
                                    
                                    # 检查是否是填充部分：
                                    # 注意：vln_sensors中GT_ActionSensor在episode结束后（action为0）填充剩余部分为0
                                    # 但在trainer中，我们需要将这些填充值识别出来并改为-1
                                    # 因此需要检查当前步数是否超出实际episode长度
                                    current_ep_info = envs.current_episodes()
                                    gt_action_length = 0
                                    if i < len(current_ep_info):
                                        ep = current_ep_info[i]
                                        gt_actions_list = getattr(ep, 'gt_action', [])
                                        if gt_actions_list:
                                            # 找到第一个stop动作（0）的位置，后续都是填充
                                            gt_action_length = len(gt_actions_list)
                                            for idx, act in enumerate(gt_actions_list):
                                                if act == 0:
                                                    gt_action_length = idx + 1  # 包含stop动作
                                                    break
                                    
                                    # 判断是否是填充部分（在stop之后或超出gt_action长度）
                                    is_padding = (
                                        expert_action_id == -1 or  # 已经是-1的填充
                                        (gt_action_length > 0 and current_episode_step >= gt_action_length)  # 超出实际长度
                                    )
                                    
                                    if is_padding:
                                        expert_action_value = -1
                                    else:
                                        expert_action_value = int(expert_action_id)
                            
                            elif expert_slice.dim() == 0:
                                # 标量
                                expert_action_value = expert_slice.item()
                            else:
                                # 其他情况，展平后取第一个
                                expert_action_value = expert_slice.flatten()[0].item()
                        except Exception as e:
                            # 如果提取失败，使用-1作为默认值
                            logger.warning(f"Failed to extract expert action for env {i}: {e}, shape: {expert_actions[i].shape if expert_actions is not None else 'N/A'}, using -1")
                            expert_action_value = -1
                    else:
                        expert_action_value = -1
                    
                    # 在添加step到episode之前，检查当前episode是否有效
                    # 如果这是第一个step，检查指令是否有效
                    if len(episodes[i]) == 0:
                        # 第一次收集step时，检查指令
                        default_instructions = ['navigate to the target location.', 'navigate to target location', 'go to target']
                        has_valid_instruction = False
                        
                        if isinstance(observations[i], dict):
                            first_obs = observations[i]
                            instruction_keys = ['falcon_instruction', 'agent_0_falcon_instruction']
                            instruction = None
                            
                            for key in instruction_keys:
                                if key in first_obs:
                                    instruction = first_obs[key]
                                    break
                            
                            if instruction is not None:
                                # 提取指令文本（使用统一的提取函数）
                                instr_text = self._extract_instruction_text_from_obs(instruction)
                                
                                # 检查是否是默认指令
                                instr_text_lower = instr_text.lower().strip()
                                if instr_text_lower and instr_text_lower not in [d.lower() for d in default_instructions]:
                                    has_valid_instruction = True
                                else:
                                    if not instr_text_lower:
                                        logger.warning(f"Env {i} episode has empty instruction, marking to skip")
                                    else:
                                        logger.warning(f"Env {i} episode uses default instruction: '{instr_text}', marking to skip")
                                    # 标记这个episode为跳过
                                    skips[i] = True
                                    dones[i] = True  # 强制结束，避免继续收集
                                    continue
                            else:
                                # 如果没有找到falcon_instruction，假设有效
                                has_valid_instruction = True
                    
                    episodes[i].append(
                        (
                            observations[i],
                            prev_actions[i].item(),
                            expert_action_value,
                        )
                    )

                # 注意：skips的计算移到step()之后，确保使用正确的dones状态
                # 这里先跳过，稍后在step()之后计算
                
                # 确保actions的形状与prev_actions匹配 [num_envs, 1]
                # if len(episodes[0]) == 0:  # 仅第一次打印
                #     logger.info(f"DEBUG: actions shape before reshape: {actions.shape}, prev_actions shape: {prev_actions.shape}")
                
                # 仿照 PPO 的方式：action_space 已经是 agent_0 专用的动作空间
                # 策略网络直接输出 agent_0 的动作，不需要从 Box 中提取
                # 处理动作格式：确保是离散动作值 [0, num_actions-1]
                
                from habitat_baselines.utils.common import is_continuous_action_space
                
                # 判断动作空间类型（仿照 PPO 的 _compute_actions_and_step_envs）
                policy = self._get_policy()  # 获取实际的policy对象
                if hasattr(self, '_agent_0_env_spec'):
                    agent_action_space = self._agent_0_env_spec.action_space
                elif hasattr(policy, 'action_space'):
                    agent_action_space = policy.action_space
                else:
                    # 回退：使用默认的 Discrete(4)
                    agent_action_space = spaces.Discrete(4)
                
                if is_continuous_action_space(agent_action_space):
                    # 连续动作空间：使用 clip 限制范围（仿照 PPO）
                    if isinstance(agent_action_space, spaces.Box):
                        actions = torch.clamp(
                            actions,
                            torch.tensor(agent_action_space.low, device=actions.device, dtype=actions.dtype),
                            torch.tensor(agent_action_space.high, device=actions.device, dtype=actions.dtype),
                        )
                else:
                    # 离散动作空间：转换为整数（仿照 PPO）
                    if actions.dtype == torch.float32 or actions.dtype == torch.float64:
                        # 如果是浮点数，可能是logits，使用argmax
                        if actions.dim() > 1 and actions.shape[-1] > 1:
                            actions = actions.argmax(dim=-1, keepdim=True)
                        else:
                            actions = actions.long()
                    else:
                        actions = actions.long()
                    
                    # 确保形状正确 [num_envs, 1]
                    if actions.dim() == 1:
                        actions = actions.unsqueeze(-1)
                    elif actions.dim() > 2:
                        actions = actions.view(envs.num_envs, -1)[:, 0:1]
                
                if actions.shape != prev_actions.shape:
                    logger.warning(f"actions shape {actions.shape} != prev_actions shape {prev_actions.shape}, reshaping")
                    # 如果actions是多维的，需要reshape为[num_envs, 1]
                    if actions.dim() > 2:
                        # 如果是3D或更高维，取第一个环境的第一个动作
                        actions = actions[:, 0, :].view(envs.num_envs, -1)
                    
                    if actions.dim() == 2 and actions.shape[1] > 1:
                        # 如果是[num_envs, action_dim]，只取第一个维度作为离散动作索引
                        # 这是agent_0的动作（在多agent环境中，Box空间的第一维是agent_0）
                        actions = actions[:, 0:1]  # 取第一列（agent_0的动作），保持2D形状
                    elif actions.dim() == 1:
                        # 如果是1D，需要unsqueeze
                        actions = actions.unsqueeze(-1)
                    
                    # 最终检查：确保形状正确
                    if actions.shape != prev_actions.shape:
                        logger.error(f"Cannot reshape actions from {actions.shape} to {prev_actions.shape}")
                        # 强制reshape（可能会有数据丢失）
                        actions = actions.view(prev_actions.shape)
                
                # 确保动作值在有效范围内（仿照 PPO 的方式）
                if hasattr(self, '_agent_0_info'):
                    num_actions = self._agent_0_info['num_actions']
                    if self._agent_0_info['action_space_type'] == 'discrete':
                        # 离散动作：限制在 [0, num_actions-1]
                        actions_clamped = torch.clamp(actions, 0, num_actions - 1).long()
                        if not torch.equal(actions_clamped, actions.long()):
                            logger.warning(f"Actions out of range [0, {num_actions-1}]: min={actions.min().item()}, max={actions.max().item()}, clamping to valid range")
                            actions = actions_clamped.float()
                    # 连续动作的 clip 已经在上面处理了
                
                prev_actions.copy_(actions)

                # 在step之前检查是否超过500步或gt_actions结束，如果超过则不执行step
                max_action_length = 500
                forced_done_envs = []  # 记录因超过500步或gt_actions结束而被强制标记为done的环境
                for i in range(envs.num_envs):
                    if dones[i]:
                        continue  # 已经done的环境跳过
                    
                    current_episode_step = len(episodes[i])
                    
                    # 检查gt_actions是否已结束
                    # 注意：应该使用实际的episode长度（找到第一个stop动作的位置），而不是整个列表的长度
                    current_ep_info = envs.current_episodes()
                    gt_action_length = 0
                    if i < len(current_ep_info):
                        ep = current_ep_info[i]
                        gt_actions_list = getattr(ep, 'gt_action', [])
                        if gt_actions_list:
                            # 找到第一个stop动作（0）的位置，后续都是填充
                            # 如果没有找到stop动作，使用整个列表的长度
                            gt_action_length = len(gt_actions_list)
                            for idx, act in enumerate(gt_actions_list):
                                if act == 0:
                                    gt_action_length = idx + 1  # 包含stop动作
                                    break
                    
                    # 如果gt_actions已结束（current_episode_step >= gt_action_length 且 gt_action_length > 0），强制结束
                    # 但只有在实际执行了足够步骤后才结束，避免过早结束
                    if gt_action_length > 0 and current_episode_step >= gt_action_length:
                        dones[i] = True
                        forced_done_envs.append(i)
                        logger.info(f"Env {i} reached gt_action_length {gt_action_length} (current_step={current_episode_step}), forcing episode end")
                    # 或者超过200步
                    elif current_episode_step >= max_action_length:
                        # 超过200步，标记为done
                        dones[i] = True
                        forced_done_envs.append(i)
                        logger.info(f"Env {i} reached max_action_length {max_action_length} (current_step={current_episode_step}), forcing episode end")
                
                # 注意：不要在这里重置所有环境！应该先保存episode，然后在step之后重置
                # 如果所有环境都done了，在step之后保存并重置

                # 将动作转换为numpy数组列表，环境期望numpy数组格式（仿照 PPO）
                step_actions = []
                for i in range(envs.num_envs):
                    if not dones[i]:  # 只为未done的环境准备动作
                        act = actions[i]
                        
                        # 仿照 PPO 的 _compute_actions_and_step_envs 中的处理方式
                        policy = self._get_policy()  # 获取实际的policy对象
                        if hasattr(self, '_agent_0_env_spec'):
                            agent_action_space = self._agent_0_env_spec.action_space
                        elif hasattr(policy, 'action_space'):
                            agent_action_space = policy.action_space
                        else:
                            # 回退：使用默认的 Discrete(4)
                            agent_action_space = spaces.Discrete(4)
                        
                        from habitat_baselines.utils.common import is_continuous_action_space
                        if is_continuous_action_space(agent_action_space):
                            # 连续动作：使用 clip（仿照 PPO）
                            act_np = act.cpu().numpy()
                            if isinstance(agent_action_space, spaces.Box):
                                act_np = np.clip(
                                    act_np,
                                    agent_action_space.low,
                                    agent_action_space.high,
                                )
                            step_actions.append(act_np)
                        else:
                            # 离散动作：提取标量值（仿照 PPO）
                            action_value = act.item()
                            if not isinstance(action_value, (int, np.integer)):
                                action_value = int(action_value)
                            # 确保动作值在有效范围内
                            if hasattr(self, '_agent_0_info') and self._agent_0_info['action_space_type'] == 'discrete':
                                num_actions = self._agent_0_info['num_actions']
                                if action_value < 0 or action_value >= num_actions:
                                    logger.warning(f"Env {i} action value {action_value} out of range [0, {num_actions-1}], using STOP (0)")
                                    action_value = 0
                            step_actions.append(np.array([action_value], dtype=np.int64))
                    else:
                        # done的环境不执行step，使用dummy动作（STOP）
                        step_actions.append(np.array([0], dtype=np.int64))
                
                outputs = envs.step(step_actions)
                observations, _, step_dones, _ = [list(x) for x in zip(*outputs)]
                
                # 更新batch（在step之后，observations已经改变）
                batch = batch_obs(observations, self.device)
                
                # 更新dones：如果之前已经标记为done（超过200步），保持done状态
                # 否则使用环境返回的done状态
                # 重要：强制环境至少执行最小步数（min_episode_steps），避免过早结束
                min_episode_steps = 5  # 最小episode步数，确保收集足够长的动作序列（降低到5，因为有些episode确实很短）
                
                # 在检查done之前，保存当前episode_id，用于检测环境是否重置
                current_ep_info_before = envs.current_episodes()
                episode_ids_before = [ep.episode_id if i < len(current_ep_info_before) else None for i, ep in enumerate(current_ep_info_before)]
                
                for i in range(envs.num_envs):
                    if i in forced_done_envs:
                        # 如果超过200步被强制标记为done，保持done状态
                        dones[i] = True
                    elif i < len(step_dones):
                        # 检查当前episode步数
                        current_steps = len(episodes[i]) if i < len(episodes) else 0
                        
                        # 检查环境是否已经重置（episode_id是否改变）
                        episode_id_after = None
                        if i < len(current_ep_info_before):
                            current_ep_info_after = envs.current_episodes()
                            if i < len(current_ep_info_after):
                                episode_id_after = current_ep_info_after[i].episode_id
                        
                        # 如果环境返回done=True但episode步数太少，直接跳过这个episode
                        # 不再强制继续收集，而是标记为skip并让环境正常重置
                        if step_dones[i] and current_steps < min_episode_steps:
                            # 获取scene信息和详细分析数据
                            scene_info = "unknown"
                            episode_id_info = "N/A"
                            gt_action_len = 0
                            distance_to_goal = "N/A"
                            if i < len(current_ep_info_before):
                                ep = current_ep_info_before[i]
                                episode_id_info = ep.episode_id if hasattr(ep, 'episode_id') else 'N/A'
                                if hasattr(ep, 'scene_id'):
                                    # 提取scene名称（只显示scene名称，不显示完整路径）
                                    scene_info = ep.scene_id.split('/')[-1].replace('.basis.glb', '')
                                # 获取gt_action长度
                                gt_actions_list = getattr(ep, 'gt_action', [])
                                if gt_actions_list:
                                    # 找到第一个stop动作（0）的位置
                                    gt_action_len = len(gt_actions_list)
                                    for idx, act in enumerate(gt_actions_list):
                                        if act == 0:
                                            gt_action_len = idx + 1
                                            break
                                # 获取初始位置和目标点的距离
                                if hasattr(ep, 'distance_to_goal'):
                                    distance_to_goal = f"{ep.distance_to_goal:.2f}m"
                                elif hasattr(ep, 'start_position') and hasattr(ep, 'goals') and ep.goals:
                                    try:
                                        import numpy as np
                                        start_pos = np.array(ep.start_position)
                                        goal_pos = np.array(ep.goals[0].position)
                                        dist = np.linalg.norm(start_pos - goal_pos)
                                        distance_to_goal = f"{dist:.2f}m"
                                    except:
                                        pass
                            
                            # 分析原因并记录详细日志
                            reason = "unknown"
                            if gt_action_len > 0:
                                if current_steps == 1 and gt_action_len > 10:
                                    reason = "初始位置太接近目标点（gt_action有{}步但只执行了1步）".format(gt_action_len)
                                elif current_steps < gt_action_len * 0.2:
                                    reason = "执行步数远少于gt_action长度（gt_action:{}步，执行:{}步）".format(gt_action_len, current_steps)
                                else:
                                    reason = "episode太短（gt_action:{}步，执行:{}步）".format(gt_action_len, current_steps)
                            
                            # 太短的episode直接跳过，不保存
                            logger.warning(f"Env {i} returned done=True after only {current_steps} step(s) < min_episode_steps={min_episode_steps}. "
                                         f"Skipping this episode. [Scene: {scene_info}, Episode ID: {episode_id_info}, "
                                         f"GT Action Length: {gt_action_len}, Distance to Goal: {distance_to_goal}, Reason: {reason}]")
                            skipped_scenes_counter[scene_info] += 1  # 统计跳过次数
                            dones[i] = True  # 让episode结束
                            skips[i] = True  # 标记为跳过，不保存
                            episodes[i] = []  # 清空数据，确保不会被保存
                        elif step_dones[i] and current_steps <= 1:
                            # 获取scene信息用于日志
                            scene_info = "unknown"
                            episode_id_info = "N/A"
                            if i < len(current_ep_info_before):
                                ep = current_ep_info_before[i]
                                episode_id_info = ep.episode_id if hasattr(ep, 'episode_id') else 'N/A'
                                if hasattr(ep, 'scene_id'):
                                    # 提取scene名称（只显示scene名称，不显示完整路径）
                                    scene_info = ep.scene_id.split('/')[-1].replace('.basis.glb', '')
                            # 即使达到最小步数，如果只有1步也记录警告并跳过
                            logger.warning(f"Env {i} returned done=True after only {current_steps} step(s). Skipping this episode. [Scene: {scene_info}, Episode ID: {episode_id_info}]")
                            skipped_scenes_counter[scene_info] += 1  # 统计跳过次数
                            dones[i] = True
                            skips[i] = True
                            episodes[i] = []
                        else:
                            # 正常情况，使用环境返回的状态
                            dones[i] = step_dones[i]
                
                # 在step之后，重新计算skips（使用最新的dones状态）
                # 获取每个环境的当前episode步数（用于索引到正确的动作）
                current_steps = [len(episodes[i]) if i < len(episodes) else 0 for i in range(envs.num_envs)]
                expert_actions_for_skips = batch.get(expert_uuid, None)
                skips_tensor = None
                if expert_actions_for_skips is not None:
                    if expert_actions_for_skips.shape == actions.shape:
                        skips_tensor = (expert_actions_for_skips.long() == -1)
                    elif expert_actions_for_skips.dim() == 2:
                        # 形状为(batch, 500)，根据当前步数提取对应时间步的action_id
                        max_action_length = expert_actions_for_skips.shape[1]  # 通常是500
                        # 检查每个环境的步数
                        skips_tensor = torch.zeros_like(actions, dtype=torch.bool)
                        for i in range(envs.num_envs):
                            env_step = current_steps[i] if i < len(current_steps) else 0
                            if env_step >= max_action_length:
                                # 超过500步，检查是否已经被强制结束（dones[i] = True）
                                if not dones[i]:
                                    # 如果episode还没结束但超过500步，标记为skip（这种情况不应该发生）
                                    skips_tensor[i] = True
                                    logger.warning(f"Env {i} exceeds max_action_length {max_action_length} (step={env_step}) but not done, skipping")
                                else:
                                    # 如果已经done，不应该跳过（应该保存）
                                    skips_tensor[i] = False
                                    logger.info(f"Env {i} exceeds max_action_length {max_action_length} (step={env_step}) but done=True, NOT skipping (will save)")
                            else:
                                # 在有效范围内（0-499），直接提取对应时间步的action_id（无需[:,0]因为已经是1D）
                                step_idx = env_step
                                expert_action_id = expert_actions_for_skips[i, step_idx]
                                skips_tensor[i] = (expert_action_id.long() == -1)
                    else:
                        # 如果形状不匹配，尝试reshape
                        try:
                            expert_actions_flat = expert_actions_for_skips.flatten()
                            if expert_actions_flat.numel() >= actions.numel():
                                expert_actions_reshaped = expert_actions_flat[:actions.numel()].view(actions.shape)
                                skips_tensor = (expert_actions_reshaped.long() == -1)
                            else:
                                skips_tensor = torch.zeros_like(actions, dtype=torch.bool)
                        except:
                            skips_tensor = torch.zeros_like(actions, dtype=torch.bool)
                else:
                    skips_tensor = torch.zeros_like(actions, dtype=torch.bool)
                
                # 将tensor转换为Python列表，用于后续的布尔检查
                skips_tensor_cpu = skips_tensor.squeeze(-1).to(device="cpu", non_blocking=True)
                skips = [bool(skip.item()) for skip in skips_tensor_cpu]
                
                # 重要修复：如果episode已经被强制结束（dones[i] = True），即使超过500步也不应该跳过
                for i in range(envs.num_envs):
                    if dones[i] and len(episodes[i]) > 0:
                        # 如果episode已经结束且有数据，强制设置为不跳过（应该保存）
                        if skips[i]:
                            logger.info(f"Env {i} episode done with {len(episodes[i])} steps, unmarking skip to allow saving (dones[{i}]=True, skips[{i}]=True->False)")
                        skips[i] = False
                    elif dones[i] and len(episodes[i]) == 0:
                        # episode已结束但为空，可能是已经被保存了
                        current_ep = envs.current_episodes()[i] if i < len(envs.current_episodes()) else None
                        ep_id = current_ep.episode_id if current_ep else 'N/A'
                        orig_id = getattr(current_ep, 'original_episode_id', 'N/A') if current_ep else 'N/A'
                        gt_len = len(getattr(current_ep, 'gt_action', [])) if current_ep else 0
                        logger.debug(f"Env {i} episode done but empty (len=0), index_id={ep_id}, orig_id={orig_id}, gt_len={gt_len}, dones[{i}]={dones[i]}, skips[{i}]={skips[i]}")
                    elif not dones[i] and len(episodes[i]) >= 500:
                        # episode还没结束但已经达到500步，这是异常情况
                        logger.warning(f"Env {i} episode not done but reached {len(episodes[i])} steps (>= 500), dones[{i}]={dones[i]}, skips[{i}]={skips[i]}")
                    
                # 当episode结束时，打印episode信息并在step之后重置环境
                for i, done in enumerate(dones):
                    if done and len(episodes[i]) > 0:
                        current_ep_info = envs.current_episodes()
                        if i < len(current_ep_info):
                            ep = current_ep_info[i]
                            # 使用辅助函数获取 original_episode_id
                            original_id = self._ensure_original_episode_id(ep)
                            
                            # 获取gt_action（专家轨迹动作）
                            gt_action = getattr(ep, 'gt_action', None)
                            gt_len = len(gt_action) if gt_action else 0
                            gt_first_5 = gt_action[:5] if gt_action and len(gt_action) >= 5 else (gt_action if gt_action else [])
                            
                            # 每个episode都打印基本信息（包含专家动作前5个值）
                            logger.info(f"=== Episode {i} Finished ===")
                            logger.info(f"Episode ID (index): {ep.episode_id}, Original ID: {original_id}")
                            if gt_len > 0:
                                logger.info(f"GT Action Length: {gt_len}, First 5 actions: {gt_first_5}")
                            else:
                                # gt_action为空时，打印更多调试信息
                                ep_dict = ep.__dict__ if hasattr(ep, '__dict__') else {}
                                logger.warning(f"GT Action is EMPTY! Episode class: {type(ep).__name__}")
                                logger.warning(f"Episode has gt_action attr: {hasattr(ep, 'gt_action')}, value type: {type(gt_action)}")
                            if hasattr(ep, 'scene_id'):
                                # 只显示scene名称，不显示完整路径
                                scene_name = ep.scene_id.split('/')[-1].replace('.basis.glb', '')
                                logger.info(f"Scene: {scene_name}")
                            logger.info(f"Total steps collected: {len(episodes[i])}")
                
                # 关键修复：在step之后，立即处理所有done的episode并重置环境
                # 这样可以确保位置及时更新，避免位置未更新就进入下一个episode
                envs_to_reset_immediately = []  # 需要立即重置的环境列表
                
                # 在step之后，保存所有done的episode（skips已经计算完成）
                for i in range(envs.num_envs):
                    if dones[i] and not skips[i] and len(episodes[i]) > 0:
                        ep = episodes[i]
                        episode_id = envs.current_episodes()[i].episode_id if i < len(envs.current_episodes()) else 'N/A'
                        
                        # 检查episode有效性：检查是否有有效的动作（非-1）
                        valid_actions_count = sum(1 for step in ep if step[2] != -1)  # step[2]是expert_action
                        
                        # 检查指令是否有效（不是默认指令）
                        current_ep_info = envs.current_episodes()
                        has_valid_instruction = False
                        default_instructions = ['navigate to the target location.', 'navigate to target location', 'go to target']
                        
                        if len(ep) > 0 and isinstance(ep[0][0], dict):
                            first_obs = ep[0][0]
                            # 检查可能的指令传感器键名
                            instruction_keys = ['falcon_instruction', 'agent_0_falcon_instruction']
                            instruction = None
                            
                            for key in instruction_keys:
                                if key in first_obs:
                                    instruction = first_obs[key]
                                    break
                            
                            if instruction is not None:
                                # 提取指令文本（使用统一的提取函数）
                                instr_text = self._extract_instruction_text_from_obs(instruction)
                                logger.debug(f"Episode {i}: Found instruction, type: {type(instruction)}, extracted text (first 100 chars): '{instr_text[:100]}'")
                                
                                # 检查是否是默认指令或空指令
                                instr_text_lower = instr_text.lower().strip()
                                logger.debug(f"Episode {i}: Extracted instruction text (first 100 chars): '{instr_text[:100]}'")
                                if instr_text_lower and instr_text_lower not in [d.lower() for d in default_instructions]:
                                    has_valid_instruction = True
                                    logger.debug(f"Episode {i} has valid instruction: {instr_text[:50]}...")
                                else:
                                    if not instr_text_lower:
                                        logger.warning(f"Episode {i} has empty instruction, will skip")
                                    else:
                                        logger.warning(f"Episode {i} uses default instruction: {instr_text}, will skip")
                                    # 标记为跳过
                                    skips[i] = True
                                    dones[i] = True
                                    continue
                            else:
                                # 如果没有找到falcon_instruction，跳过这个episode（没有指令的episode不应该用于训练）
                                # 添加调试信息：打印可用的键
                                available_keys = list(first_obs.keys()) if isinstance(first_obs, dict) else []
                                logger.warning(f"Episode {i} (ID: {episode_id}) has no falcon_instruction. "
                                             f"Available observation keys: {available_keys[:20]}... (showing first 20)")
                                # 检查是否有类似的键
                                similar_keys = [k for k in available_keys if 'instruction' in k.lower() or 'instr' in k.lower()]
                                if similar_keys:
                                    logger.warning(f"Found similar keys that might be instructions: {similar_keys}")
                                episodes[i] = []
                                # 立即重置环境以收集新的episode
                                try:
                                    # 使用单个索引而不是列表
                                    reset_obs = envs.reset_at(i)
                                    if reset_obs is not None:
                                        observations[i] = reset_obs
                                        dones[i] = False
                                        logger.info(f"Env {i} reset after missing instruction, ready to collect new episode")
                                except Exception as e:
                                    logger.warning(f"Failed to reset env {i} after missing instruction: {e}")
                                    import traceback
                                    logger.debug(traceback.format_exc())
                                continue
                        
                        # 如果episode无效（没有有效动作或使用默认指令），跳过保存并重置环境收集新的episode
                        if valid_actions_count == 0:
                            logger.warning(f"Episode {i} (ID: {episode_id}) has no valid actions (all -1), skipping save and will collect new episode")
                            episodes[i] = []
                            # 立即重置环境以收集新的episode
                            try:
                                # 使用单个索引而不是列表
                                reset_obs = envs.reset_at(i)
                                if reset_obs is not None:
                                    observations[i] = reset_obs
                                    dones[i] = False
                                    logger.info(f"Env {i} reset after invalid episode, ready to collect new episode")
                            except Exception as e:
                                logger.warning(f"Failed to reset env {i} after invalid episode: {e}")
                                import traceback
                                logger.debug(traceback.format_exc())
                            continue
                        
                        # 如果使用默认指令，应该跳过并收集新的episode
                        if not has_valid_instruction:
                            logger.warning(f"Episode {i} (ID: {episode_id}) uses default instruction, skipping save and will collect new episode")
                            episodes[i] = []
                            # 立即重置环境以收集新的episode
                            try:
                                # 使用单个索引而不是列表
                                reset_obs = envs.reset_at(i)
                                if reset_obs is not None:
                                    observations[i] = reset_obs
                                    dones[i] = False
                                    logger.info(f"Env {i} reset after default instruction episode, ready to collect new episode")
                            except Exception as e:
                                logger.warning(f"Failed to reset env {i} after default instruction episode: {e}")
                            continue
                        
                        # 在保存之前检查episode是否重复
                        # 如果force_diverse_episodes=True，跟踪已收集的episode，跳过重复的
                        # 使用scene_name + episode_id组合作为唯一标识
                        should_skip_duplicate = False
                        current_ep_obj = envs.current_episodes()[i] if i < len(envs.current_episodes()) else None
                        ep_unique_key = get_episode_unique_key(current_ep_obj) if current_ep_obj else episode_id
                        
                        if force_diverse_episodes and ep_ids_collected is not None:
                            if ep_unique_key in ep_ids_collected:
                                should_skip_duplicate = True
                                logger.info(f"Episode {i} (key: {ep_unique_key}) already collected, skipping save")
                            else:
                                # 新的episode，添加到已收集集合
                                ep_ids_collected.add(ep_unique_key)
                                logger.debug(f"New unique episode collected: {ep_unique_key} (total unique: {len(ep_ids_collected)})")
                        
                        # 如果是重复的episode，跳过保存并重置环境
                        if should_skip_duplicate:
                            episodes[i] = []
                            # 立即重置环境以收集新的episode
                            try:
                                reset_obs = envs.reset_at(i)
                                if reset_obs is not None:
                                    observations[i] = reset_obs
                                    dones[i] = False
                                    logger.info(f"Env {i} reset after duplicate episode, ready to collect new episode")
                            except Exception as e:
                                logger.warning(f"Failed to reset env {i} after duplicate episode: {e}")
                            continue
                        
                        # 获取原始episode_id
                        current_ep = envs.current_episodes()[i] if i < len(envs.current_episodes()) else None
                        original_id = getattr(current_ep, 'original_episode_id', 'N/A') if current_ep else 'N/A'
                        gt_len = len(getattr(current_ep, 'gt_action', [])) if current_ep else 0
                        
                        # 只在每50个episode时打印详细信息，减少日志输出以提高速度
                        if collected_eps % 50 == 0:
                            logger.info(f"Saving episode {i}: {len(ep)} steps, {valid_actions_count} valid actions, index_id={episode_id}, orig_id={original_id}, gt_len={gt_len}")
                        else:
                            logger.debug(f"Saving episode {i}: {len(ep)} steps, index_id={episode_id}, orig_id={original_id}")
                        traj_obs = batch_obs(
                            [step[0] for step in ep], device=torch.device("cpu")
                        )
                        if expert_uuid in traj_obs:
                            del traj_obs[expert_uuid]
                        
                        # 递归转换所有 Tensor 为 numpy 数组
                        def _convert_tensor_to_numpy(obj):
                            """递归转换 Tensor 和嵌套结构中的 Tensor 为 numpy"""
                            if isinstance(obj, torch.Tensor):
                                numpy_obj = obj.detach().cpu().numpy()
                                if self.config.habitat_baselines.il.dagger.lmdb_fp16 and numpy_obj.dtype == np.float32:
                                    numpy_obj = numpy_obj.astype(np.float16)
                                return numpy_obj
                            elif isinstance(obj, dict):
                                return {k: _convert_tensor_to_numpy(v) for k, v in obj.items()}
                            elif isinstance(obj, (list, tuple)):
                                return type(obj)(_convert_tensor_to_numpy(item) for item in obj)
                            elif isinstance(obj, np.ndarray):
                                # 如果已经是 numpy 数组，检查是否需要转换精度
                                if self.config.habitat_baselines.il.dagger.lmdb_fp16 and obj.dtype == np.float32:
                                    return obj.astype(np.float16)
                                return obj
                            else:
                                return obj
                        
                        # 转换 traj_obs 中的所有 Tensor
                        traj_obs = _convert_tensor_to_numpy(traj_obs)

                        transposed_ep = [
                            traj_obs,
                            np.array([step[1] for step in ep], dtype=np.int64),
                            np.array([step[2] for step in ep], dtype=np.int64),
                        ]
                        if _use_msgpack_numpy:
                            packed = _mpn.packb(transposed_ep, use_bin_type=True)
                        else:
                            def _encode_nd(obj):
                                if isinstance(obj, np.ndarray):
                                    return {
                                        "__nd": True,
                                        "dtype": str(obj.dtype),
                                        "shape": obj.shape,
                                        "data": obj.tobytes(),
                                    }
                                elif isinstance(obj, torch.Tensor):
                                    # 如果还有 Tensor（理论上不应该），转换为 numpy
                                    numpy_obj = obj.detach().cpu().numpy()
                                    return {
                                        "__nd": True,
                                        "dtype": str(numpy_obj.dtype),
                                        "shape": numpy_obj.shape,
                                        "data": numpy_obj.tobytes(),
                                    }
                                return obj
                            packed = msgpack.packb(transposed_ep, default=_encode_nd, use_bin_type=True)
                        # 使用独立的 lmdb_index 确保索引连续，即使某些episode被跳过
                        txn.put(str(start_id + lmdb_index).encode(), packed)
                        collected_eps += 1
                        lmdb_index += 1  # 即使episode被跳过，索引也递增
                        steps_since_last_save = 0  # 重置计数器
                        
                        # 每个rank独立更新自己的进度条（不需要同步）
                        pbar.update(1)
                        postfix_info = {
                            'episodes': collected_eps,
                        }
                        if ep_ids_collected is not None:
                            postfix_info['unique'] = len(ep_ids_collected)
                        if len(ep) > 0:
                            postfix_info['steps'] = len(ep)
                        pbar.set_postfix(postfix_info)
                        
                        # 在分布式训练中，减少同步频率：每100个episode或在关键里程碑同步一次
                        # 主要用于统计，不影响各自进度显示
                        sync_interval = 10  # 每100个episode同步一次
                        should_sync = False
                        if self._is_distributed:
                            # 只在关键里程碑同步：每100个episode或完成时
                            if collected_eps % sync_interval == 0 or collected_eps >= local_update_size:
                                should_sync = True
                        
                        # 只在需要同步时进行all_gather（大幅减少同步次数，从每个episode减少到每100个episode）
                        if should_sync and self._is_distributed:
                            try:
                                import torch.distributed as dist
                                collected_eps_tensor = torch.tensor([collected_eps], dtype=torch.int64, device=self.device)
                                gathered_eps = [torch.zeros_like(collected_eps_tensor) for _ in range(self._dist_world_size)]
                                dist.all_gather(gathered_eps, collected_eps_tensor)
                                total_collected = sum([eps.item() for eps in gathered_eps])
                                
                                # 每100个episode或完成时，记录详细统计信息
                                rank_progress_str = ' '.join([f'R{i}:{eps.item()}' for i, eps in enumerate(gathered_eps)])
                                logger.info(f"[Rank {self._dist_rank}] Sync checkpoint: {collected_eps}/{local_update_size} (total across ranks: {total_collected}) [{rank_progress_str}]")
                            except Exception as e:
                                # 如果同步失败，记录警告但继续执行，不阻塞
                                logger.warning(f"[Rank {self._dist_rank}] all_gather failed: {e}, continuing without sync")
                        
                        # 每个rank每10个episode显示一次进度日志（独立显示，不需要同步）
                        if collected_eps % 10 == 0 or collected_eps == local_update_size:
                            if self._is_distributed:
                                if force_diverse_episodes and ep_ids_collected is not None:
                                    logger.info(f"[Rank {self._dist_rank}] Progress: {collected_eps}/{local_update_size} episodes collected ({len(ep_ids_collected)} unique)")
                                else:
                                    logger.info(f"[Rank {self._dist_rank}] Progress: {collected_eps}/{local_update_size} episodes collected")
                            else:
                                if force_diverse_episodes and ep_ids_collected is not None:
                                    logger.info(f"Progress: {collected_eps}/{self.config.habitat_baselines.il.dagger.update_size} episodes collected ({len(ep_ids_collected)} unique)")
                                else:
                                    logger.info(f"Progress: {collected_eps}/{self.config.habitat_baselines.il.dagger.update_size} episodes collected")
                        
                        # 检查是否达到目标数量
                        if collected_eps >= local_update_size:
                            if self._is_distributed:
                                logger.info(f"[Rank {self._dist_rank}] Reached target episode count ({collected_eps}/{local_update_size}), preparing to exit collection loop...")
                            else:
                                logger.info(f"Reached target episode count ({collected_eps}), exiting collection loop")
                            break
                        
                        if (
                            collected_eps
                            % self.config.habitat_baselines.il.dagger.lmdb_commit_frequency
                        ) == 0:
                            txn.commit()
                            txn = lmdb_env.begin(write=True)

                        # 注意：重复episode的检查已经在保存之前完成，这里不需要再次检查
                        # 保存后立即清空，避免重复保存
                        episodes[i] = []
                        # 标记需要立即重置的环境
                        envs_to_reset_immediately.append(i)
                    elif dones[i]:
                        # episode结束但被跳过或为空，记录日志
                        current_ep = envs.current_episodes()[i] if i < len(envs.current_episodes()) else None
                        episode_id = current_ep.episode_id if current_ep else 'N/A'
                        original_id = getattr(current_ep, 'original_episode_id', 'N/A') if current_ep else 'N/A'
                        gt_len = len(getattr(current_ep, 'gt_action', [])) if current_ep else 0
                        if skips[i]:
                            logger.info(f"Env {i} episode skipped (skips[i]=True), index_id={episode_id}, orig_id={original_id}, gt_len={gt_len}, steps={len(episodes[i])}")
                        elif len(episodes[i]) == 0:
                            # 获取scene信息
                            scene_info = "unknown"
                            if current_ep and hasattr(current_ep, 'scene_id'):
                                scene_info = current_ep.scene_id.split('/')[-1].replace('.basis.glb', '')
                            logger.info(f"Env {i} episode empty (len=0), index_id={episode_id}, orig_id={original_id}, gt_len={gt_len}, scene={scene_info}")
                        else:
                            logger.warning(f"Env {i} episode done but not saved! dones[{i}]={dones[i]}, skips[{i}]={skips[i]}, steps={len(episodes[i])}, index_id={episode_id}, orig_id={original_id}")
                        # 在清空episode之前，保存当前episode信息和agent位置（用于后续重置时比较）
                        if not hasattr(self, '_prev_episode_info'):
                            self._prev_episode_info = {}
                        if not hasattr(self, '_prev_agent_positions'):
                            self._prev_agent_positions = {}
                        if not hasattr(self, '_prev_episode_keys'):
                            self._prev_episode_keys = {}
                        
                        current_ep_for_save = envs.current_episodes()[i] if i < len(envs.current_episodes()) else None
                        if current_ep_for_save:
                            self._prev_episode_info[i] = current_ep_for_save
                            
                            # 保存episode的唯一标识（scene_id + episode_id）
                            scene_name = current_ep_for_save.scene_id.split('/')[-1].replace('.basis.glb', '') if hasattr(current_ep_for_save, 'scene_id') else 'unknown'
                            episode_key = f"{scene_name}_{current_ep_for_save.episode_id}"
                            self._prev_episode_keys[i] = episode_key
                            
                            # 保存agent的结束位置
                            try:
                                agent_state = envs.call_at(i, "get_agent_state")
                                if agent_state:
                                    self._prev_agent_positions[i] = np.array(agent_state.position)
                            except Exception as e:
                                logger.debug(f"Failed to save agent position for env {i}: {e}")
                                self._prev_agent_positions[i] = None
                        
                        # 清空episode（无论是否保存）
                        episodes[i] = []
                        # 标记需要立即重置的环境
                        envs_to_reset_immediately.append(i)
                
                # 关键修复：立即重置所有done的环境，确保位置及时更新
                # 在检查目标数量之前就重置，避免位置未更新就继续执行
                # 这样可以确保episode切换及时，位置更新及时
                for i in envs_to_reset_immediately:
                    if i < envs.num_envs and dones[i] and len(episodes[i]) == 0:
                        # episode已保存或跳过，立即重置环境以获取新的episode
                        # 在DAgger中，每次重置应该尽可能采样不同的episode
                        try:
                            reset_obs = envs.reset_at(i)
                            # 检查新episode是否与之前的不同（使用scene_name + episode_id组合）
                            if force_diverse_episodes and ep_ids_collected is not None:
                                new_ep_info = envs.current_episodes()
                                if i < len(new_ep_info):
                                    new_ep_key = get_episode_unique_key(new_ep_info[i])
                                    if new_ep_key in ep_ids_collected:
                                        # 遇到重复episode，记录日志
                                        logger.debug(f"Env {i} reset to duplicate episode {new_ep_key}, will try reset again if needed")
                                    else:
                                        # 遇到新的episode，添加到集合
                                        ep_ids_collected.add(new_ep_key)
                                        # 减少日志输出，只在每100个新episode时打印
                                        if len(ep_ids_collected) % 100 == 0:
                                            logger.debug(f"Env {i} reset to new episode {new_ep_id} (total unique: {len(ep_ids_collected)})")
                            # reset_at返回的可能是一个列表或单个观察
                            if reset_obs is None:
                                # 如果返回None，尝试使用reset()方法
                                logger.warning(f"reset_at returned None for env {i}, trying reset()")
                                all_reset = envs.reset()
                                if isinstance(all_reset, list) and len(all_reset) > i:
                                    observations[i] = all_reset[i]
                                    dones[i] = False
                                elif not isinstance(all_reset, list) and all_reset is not None:
                                    observations[i] = all_reset
                                    dones[i] = False
                            elif isinstance(reset_obs, list):
                                # 如果返回的是列表，取第一个元素
                                if len(reset_obs) > 0:
                                    observations[i] = reset_obs[0]
                                    dones[i] = False
                                else:
                                    logger.warning(f"reset_at returned empty list for env {i}, trying reset()")
                                    all_reset = envs.reset()
                                    if isinstance(all_reset, list) and len(all_reset) > i:
                                        observations[i] = all_reset[i]
                                        dones[i] = False
                            else:
                                # 如果返回的是单个观察（字典或其他类型）
                                observations[i] = reset_obs
                                dones[i] = False
                            
                            # 关键修复：验证episode是否真正切换，然后确保位置更新
                            # 参考评估器：重置后立即调用post_step
                            try:
                                observations = envs.post_step(observations)
                            except Exception as e:
                                logger.debug(f"post_step failed for env {i} after reset: {e}, continuing anyway")
                            
                            if not dones[i]:
                                import time
                                max_episode_switch_attempts = 5  # 最多尝试5次确保episode切换
                                episode_switched = False
                                position_updated = False
                                
                                # 获取上一个episode的标识和位置
                                prev_episode_key = None
                                prev_position = None
                                if hasattr(self, '_prev_episode_keys') and i in self._prev_episode_keys:
                                    prev_episode_key = self._prev_episode_keys[i]
                                if hasattr(self, '_prev_agent_positions') and i in self._prev_agent_positions:
                                    prev_position = self._prev_agent_positions[i]
                                
                                # 验证episode是否真正切换
                                for switch_check_idx in range(max_episode_switch_attempts):
                                    try:
                                        # 获取当前episode信息
                                        current_ep_info = envs.current_episodes()
                                        if i < len(current_ep_info):
                                            new_ep = current_ep_info[i]
                                            scene_name = new_ep.scene_id.split('/')[-1].replace('.basis.glb', '') if hasattr(new_ep, 'scene_id') else 'unknown'
                                            current_episode_key = f"{scene_name}_{new_ep.episode_id}"
                                            
                                            # 检查episode是否切换
                                            if prev_episode_key is None or current_episode_key != prev_episode_key:
                                                episode_switched = True
                                                logger.debug(f"Env {i} episode switched: {prev_episode_key} -> {current_episode_key}")
                                                
                                                # Episode已切换，现在检查位置是否更新
                                                if prev_position is not None:
                                                    try:
                                                        agent_state = envs.call_at(i, "get_agent_state")
                                                        if agent_state:
                                                            current_position = np.array(agent_state.position)
                                                            position_diff = np.linalg.norm(current_position - prev_position)
                                                            
                                                            # 如果位置差异足够大，说明已更新
                                                            if position_diff > 1.0:
                                                                position_updated = True
                                                                if not hasattr(self, '_position_update_status'):
                                                                    self._position_update_status = {}
                                                                self._position_update_status[i] = True
                                                                logger.debug(f"Env {i} episode switched and position updated, diff: {position_diff:.3f}m")
                                                                break
                                                            else:
                                                                # Episode切换了但位置没更新，执行dummy step
                                                                logger.debug(f"Env {i} episode switched but position not updated (diff: {position_diff:.3f}m), executing dummy step...")
                                                                try:
                                                                    envs.async_step_at(i, np.array([0], dtype=np.int64))
                                                                    dummy_outputs = envs.wait_step_at(i)
                                                                    if dummy_outputs is not None:
                                                                        dummy_obs, _, dummy_done, _ = dummy_outputs
                                                                        observations[i] = dummy_obs
                                                                        if dummy_done:
                                                                            dones[i] = True
                                                                            break
                                                                    try:
                                                                        observations = envs.post_step(observations)
                                                                    except Exception:
                                                                        pass
                                                                    time.sleep(0.2)
                                                                    
                                                                    # 再次检查位置
                                                                    agent_state = envs.call_at(i, "get_agent_state")
                                                                    if agent_state:
                                                                        current_position = np.array(agent_state.position)
                                                                        position_diff = np.linalg.norm(current_position - prev_position)
                                                                        if position_diff > 1.0:
                                                                            position_updated = True
                                                                            if not hasattr(self, '_position_update_status'):
                                                                                self._position_update_status = {}
                                                                            self._position_update_status[i] = True
                                                                            logger.debug(f"Env {i} position updated after dummy step, diff: {position_diff:.3f}m")
                                                                            break
                                                                except Exception as e:
                                                                    logger.debug(f"Failed to execute dummy step for env {i}: {e}")
                                                    except Exception as e:
                                                        logger.debug(f"Failed to check position after episode switch: {e}")
                                                else:
                                                    # 没有上一个位置，假设已更新
                                                    position_updated = True
                                                    if not hasattr(self, '_position_update_status'):
                                                        self._position_update_status = {}
                                                    self._position_update_status[i] = True
                                                
                                                break  # Episode已切换，退出循环
                                            else:
                                                # Episode未切换，需要重新重置
                                                logger.warning(f"Env {i} reset did not switch episode (still {current_episode_key}), retrying reset...")
                                                if switch_check_idx < max_episode_switch_attempts - 1:
                                                    # 重新重置
                                                    reset_obs = envs.reset_at(i)
                                                    if reset_obs is None:
                                                        all_reset = envs.reset()
                                                        if isinstance(all_reset, list) and len(all_reset) > i:
                                                            observations[i] = all_reset[i]
                                                        elif not isinstance(all_reset, list) and all_reset is not None:
                                                            observations[i] = all_reset
                                                    elif isinstance(reset_obs, list) and len(reset_obs) > 0:
                                                        observations[i] = reset_obs[0]
                                                    elif not isinstance(reset_obs, list):
                                                        observations[i] = reset_obs
                                                    
                                                    try:
                                                        observations = envs.post_step(observations)
                                                    except Exception:
                                                        pass
                                                    
                                                    time.sleep(0.2)  # 延迟后重试
                                                else:
                                                    logger.error(f"Env {i} failed to switch episode after {max_episode_switch_attempts} attempts, episode may be stuck")
                                                    dones[i] = True  # 标记为done，跳过这个环境
                                                    break
                                    except Exception as e:
                                        logger.warning(f"Failed to verify episode switch for env {i} (attempt {switch_check_idx + 1}): {e}")
                                        if switch_check_idx < max_episode_switch_attempts - 1:
                                            time.sleep(0.2)
                                
                                # 如果episode未切换或位置未更新，记录状态
                                if not episode_switched:
                                    if not hasattr(self, '_position_update_status'):
                                        self._position_update_status = {}
                                    self._position_update_status[i] = False
                                    logger.warning(f"Env {i} episode may not have switched properly after reset")
                                elif not position_updated and prev_position is not None:
                                    if not hasattr(self, '_position_update_status'):
                                        self._position_update_status = {}
                                    self._position_update_status[i] = False
                                    logger.warning(f"Env {i} episode switched but position may not be fully updated")
                                elif prev_position is None:
                                    # 没有上一个位置信息，假设位置已更新
                                    if not hasattr(self, '_position_update_status'):
                                        self._position_update_status = {}
                                    self._position_update_status[i] = True
                            
                            # 获取新episode信息，并检查指令有效性和geodesic_distance
                            if not dones[i]:
                                current_ep_info = envs.current_episodes()
                                if i < len(current_ep_info):
                                    new_ep = current_ep_info[i]
                                    
                                    # 参考expert_data_collector_v3.py：检查geodesic_distance
                                    # 但需要智能处理：区分"位置未更新"和"真的距离太近"
                                    min_geodesic_distance = 1.0  # 最小距离阈值（米）
                                    geodesic_distance = 0.0
                                    max_reset_attempts_for_distance = 3  # 最多尝试3次重置
                                    reset_count_for_distance = 0
                                    
                                    try:
                                        # 从observations中获取当前距离（更准确，因为已经执行了dummy step）
                                        if isinstance(observations[i], dict):
                                            pg = observations[i].get("agent_0_pointgoal_with_gps_compass", None)
                                            if pg is not None:
                                                try:
                                                    if hasattr(pg, "detach"):
                                                        pg_np = pg.detach().cpu().numpy()
                                                    elif hasattr(pg, "cpu"):
                                                        pg_np = pg.cpu().numpy()
                                                    else:
                                                        pg_np = np.array(pg)
                                                    if pg_np.ndim == 1:
                                                        current_dist_from_obs = float(pg_np[0])
                                                    elif pg_np.ndim == 2:
                                                        current_dist_from_obs = float(pg_np[0, 0])
                                                    else:
                                                        current_dist_from_obs = None
                                                except Exception:
                                                    current_dist_from_obs = None
                                            else:
                                                current_dist_from_obs = None
                                        else:
                                            current_dist_from_obs = None
                                        
                                        # 优先使用observations中的距离，如果没有则使用episode info
                                        if current_dist_from_obs is not None and current_dist_from_obs > 0:
                                            geodesic_distance = current_dist_from_obs
                                        elif hasattr(new_ep, 'info') and new_ep.info:
                                            geodesic_distance = new_ep.info.get("geodesic_distance", 0.0)
                                        elif hasattr(new_ep, 'distance_to_goal'):
                                            geodesic_distance = new_ep.distance_to_goal
                                        
                                        # 如果距离太近，需要智能判断：是否是位置未更新的问题
                                        # 关键：如果位置已经更新（通过之前的检查），但距离仍然很小，说明是真的距离很近
                                        # 应该接受这个episode，而不是继续重置
                                        position_was_updated = hasattr(self, '_position_update_status') and self._position_update_status.get(i, False)
                                        
                                        if (geodesic_distance <= min_geodesic_distance or 
                                            np.isnan(geodesic_distance) or 
                                            np.isinf(geodesic_distance) or
                                            geodesic_distance <= 0.0):
                                            scene_name = new_ep.scene_id.split('/')[-1].replace('.basis.glb', '') if hasattr(new_ep, 'scene_id') else 'unknown'
                                            
                                            # 如果位置已经更新，但距离仍然很小，说明是真的距离很近，接受这个episode
                                            if position_was_updated:
                                                logger.info(f"Env {i} reset to episode {new_ep.episode_id} (scene: {scene_name}) with small geodesic_distance: {geodesic_distance:.3f}m, "
                                                          f"but position was updated (diff > 1.0m), accepting this episode as valid short episode")
                                                # 继续使用这个episode，不重置
                                            else:
                                                # 位置未更新，需要检查是否是同一个scene
                                                # 获取上一个episode的信息（如果存在）
                                                prev_scene_name = None
                                                if hasattr(self, '_prev_episode_info') and i in self._prev_episode_info:
                                                    prev_ep = self._prev_episode_info[i]
                                                    if prev_ep and hasattr(prev_ep, 'scene_id'):
                                                        prev_scene_name = prev_ep.scene_id.split('/')[-1].replace('.basis.glb', '')
                                                
                                                # 如果是同一个scene且距离太近，可能是位置未更新的问题
                                                # 解决方案：等待一个step让环境状态更新，或者重试重置
                                                if prev_scene_name == scene_name and geodesic_distance <= min_geodesic_distance:
                                                    logger.info(f"Env {i} reset to episode {new_ep.episode_id} (scene: {scene_name}) with small geodesic_distance: {geodesic_distance:.3f}m. "
                                                              f"Same scene as previous episode, may be position not updated. Will retry reset (attempt {reset_count_for_distance + 1}/{max_reset_attempts_for_distance})...")
                                                
                                                # 重试重置，最多尝试max_reset_attempts_for_distance次
                                                while reset_count_for_distance < max_reset_attempts_for_distance:
                                                    reset_count_for_distance += 1
                                                    # 短暂延迟，让环境状态更新
                                                    import time
                                                    time.sleep(0.05)
                                                    
                                                    # 再次重置
                                                    reset_obs = envs.reset_at(i)
                                                    if reset_obs is None:
                                                        all_reset = envs.reset()
                                                        if isinstance(all_reset, list) and len(all_reset) > i:
                                                            observations[i] = all_reset[i]
                                                        elif not isinstance(all_reset, list) and all_reset is not None:
                                                            observations[i] = all_reset
                                                    elif isinstance(reset_obs, list) and len(reset_obs) > 0:
                                                        observations[i] = reset_obs[0]
                                                    elif not isinstance(reset_obs, list):
                                                        observations[i] = reset_obs
                                                    
                                                    # 重置后调用post_step确保环境状态完全更新
                                                    try:
                                                        observations = envs.post_step(observations)
                                                    except Exception as e:
                                                        logger.debug(f"post_step failed for env {i} during retry reset: {e}")
                                                    
                                                    # 执行dummy step强制更新位置状态
                                                    try:
                                                        envs.async_step_at(i, np.array([0], dtype=np.int64))
                                                        dummy_outputs = envs.wait_step_at(i)
                                                        if dummy_outputs is not None:
                                                            dummy_obs, _, dummy_done, _ = dummy_outputs
                                                            observations[i] = dummy_obs
                                                            if dummy_done:
                                                                dones[i] = True
                                                                break
                                                        try:
                                                            observations = envs.post_step(observations)
                                                        except Exception:
                                                            pass
                                                        time.sleep(0.1)
                                                    except Exception as e:
                                                        logger.debug(f"Failed to execute dummy step for env {i} during retry: {e}")
                                                    
                                                    # 重新获取episode信息
                                                    current_ep_info = envs.current_episodes()
                                                    if i < len(current_ep_info):
                                                        new_ep = current_ep_info[i]
                                                        
                                                        # 优先从observations获取距离（更准确）
                                                        current_dist_from_obs = None
                                                        if isinstance(observations[i], dict):
                                                            pg = observations[i].get("agent_0_pointgoal_with_gps_compass", None)
                                                            if pg is not None:
                                                                try:
                                                                    if hasattr(pg, "detach"):
                                                                        pg_np = pg.detach().cpu().numpy()
                                                                    elif hasattr(pg, "cpu"):
                                                                        pg_np = pg.cpu().numpy()
                                                                    else:
                                                                        pg_np = np.array(pg)
                                                                    if pg_np.ndim == 1:
                                                                        current_dist_from_obs = float(pg_np[0])
                                                                    elif pg_np.ndim == 2:
                                                                        current_dist_from_obs = float(pg_np[0, 0])
                                                                except Exception:
                                                                    pass
                                                        
                                                        # 重新检查geodesic_distance
                                                        if current_dist_from_obs is not None and current_dist_from_obs > 0:
                                                            geodesic_distance = current_dist_from_obs
                                                        elif hasattr(new_ep, 'info') and new_ep.info:
                                                            geodesic_distance = new_ep.info.get("geodesic_distance", 0.0)
                                                        elif hasattr(new_ep, 'distance_to_goal'):
                                                            geodesic_distance = new_ep.distance_to_goal
                                                        
                                                        # 如果距离现在有效了，退出重试循环
                                                        if (geodesic_distance > min_geodesic_distance and 
                                                            not np.isnan(geodesic_distance) and 
                                                            not np.isinf(geodesic_distance) and
                                                            geodesic_distance > 0.0):
                                                            logger.info(f"Env {i} after retry reset: episode {new_ep.episode_id} now has valid geodesic_distance: {geodesic_distance:.3f}m")
                                                            break
                                                
                                                    # 如果重试后仍然无效，记录日志但继续使用（可能是真的距离太近）
                                                    if geodesic_distance <= min_geodesic_distance:
                                                        logger.warning(f"Env {i} after {max_reset_attempts_for_distance} retry attempts, episode {new_ep.episode_id} still has small geodesic_distance: {geodesic_distance:.3f}m. "
                                                                     f"This may be a valid short episode. Continuing anyway...")
                                                else:
                                                    # 不同scene或距离真的太小，尝试重置一次
                                                    logger.info(f"Env {i} reset to episode {new_ep.episode_id} (scene: {scene_name}) with invalid geodesic_distance: {geodesic_distance:.3f}m <= {min_geodesic_distance}m, resetting again...")
                                                reset_obs = envs.reset_at(i)
                                                if reset_obs is None:
                                                    all_reset = envs.reset()
                                                    if isinstance(all_reset, list) and len(all_reset) > i:
                                                        observations[i] = all_reset[i]
                                                    elif not isinstance(all_reset, list) and all_reset is not None:
                                                        observations[i] = all_reset
                                                elif isinstance(reset_obs, list) and len(reset_obs) > 0:
                                                    observations[i] = reset_obs[0]
                                                elif not isinstance(reset_obs, list):
                                                    observations[i] = reset_obs
                                                
                                                # 重新获取episode信息
                                                current_ep_info = envs.current_episodes()
                                                if i < len(current_ep_info):
                                                    new_ep = current_ep_info[i]
                                                else:
                                                    # 如果仍然无法获取有效episode，标记为done跳过
                                                    dones[i] = True
                                                    continue
                                        else:
                                            scene_name = new_ep.scene_id.split('/')[-1].replace('.basis.glb', '') if hasattr(new_ep, 'scene_id') else 'unknown'
                                            logger.debug(f"Env {i} reset to episode {new_ep.episode_id} (scene: {scene_name}) with valid geodesic_distance: {geodesic_distance:.3f}m")
                                        
                                        # 保存当前episode信息，供下次重置时比较
                                        if not hasattr(self, '_prev_episode_info'):
                                            self._prev_episode_info = {}
                                        self._prev_episode_info[i] = new_ep
                                        
                                    except Exception as e:
                                        logger.warning(f"Failed to check geodesic_distance for env {i} after reset: {e}")
                                        # 如果检查失败，继续使用当前episode
                                    
                                    logger.info(f"Env {i} reset, new episode ID: {new_ep.episode_id}")
                                    
                                    # 检查新episode的指令
                                    default_instructions = ['navigate to the target location.', 'navigate to target location', 'go to target']
                                    if isinstance(observations[i], dict):
                                        new_obs = observations[i]
                                        instruction_keys = ['falcon_instruction', 'agent_0_falcon_instruction']
                                        instruction = None
                                        
                                        for key in instruction_keys:
                                            if key in new_obs:
                                                instruction = new_obs[key]
                                                break
                                        
                                        if instruction is not None:
                                            # 提取指令文本
                                            # 使用统一的指令提取函数
                                            instr_text = self._extract_instruction_text_from_obs(instruction)
                                            
                                            # 检查是否是默认指令
                                            instr_text_lower = instr_text.lower().strip()
                                            if instr_text_lower in [d.lower() for d in default_instructions]:
                                                logger.warning(f"Env {i} reset to episode with default instruction: '{instr_text}', will mark to skip")
                                                skips[i] = True
                                                dones[i] = True  # 强制结束，避免收集
                        except Exception as e:
                            logger.warning(f"Failed to reset env {i}: {e}, will try again next iteration")
                            import traceback
                            logger.debug(traceback.format_exc())
                
                # 关键修复：在所有环境重置完成后，统一更新batch
                # 这样可以确保下一个循环迭代使用正确的observations
                if len(envs_to_reset_immediately) > 0:
                    try:
                        batch = batch_obs(observations, self.device)
                    except Exception as e:
                        logger.debug(f"Failed to update batch after resetting {len(envs_to_reset_immediately)} environments: {e}")
                
                # 检查是否达到目标数量，如果达到则退出循环
                # 只在每200个episode时打印一次（减少日志输出以提高速度）
                if collected_eps % 200 == 0:
                    if self._is_distributed:
                        logger.debug(f"[Rank {self._dist_rank}] Loop check: collected_eps={collected_eps}, unique={len(ep_ids_collected) if ep_ids_collected is not None else 'N/A'}, envs.num_envs={envs.num_envs}")
                    else:
                        logger.debug(f"Loop check: collected_eps={collected_eps}, unique={len(ep_ids_collected) if ep_ids_collected is not None else 'N/A'}, envs.num_envs={envs.num_envs}")
                if collected_eps >= local_update_size:
                    if self._is_distributed:
                        logger.info(f"[Rank {self._dist_rank}] Collection complete: {collected_eps}/{local_update_size} episodes collected, exiting main loop...")
                    else:
                        logger.info(f"Collection complete: {collected_eps}/{self.config.habitat_baselines.il.dagger.update_size} episodes collected")
                    break
                
                # 如果所有环境都done了且episode都已保存，重置所有环境
                if all(dones) and all(len(episodes[i]) == 0 for i in range(envs.num_envs)):
                    logger.info("All environments done and episodes saved, resetting all environments")
                    try:
                        observations = envs.reset()
                        dones = [False] * envs.num_envs
                        # 清空所有episodes
                        for i in range(envs.num_envs):
                            episodes[i] = []
                        # 更新batch
                        batch = batch_obs(observations, self.device)
                    except Exception as e:
                        logger.warning(f"Failed to reset all environments: {e}")
                        import traceback
                        logger.debug(traceback.format_exc())
                
                # 更新not_done_masks（基于最新的dones状态）
                not_done_masks = torch.tensor(
                    [[0] if done else [1] for done in dones],
                    dtype=torch.uint8,
                    device=self.device,
                )

                # 检查是否长时间没有保存episode
                if steps_since_last_save >= max_steps_without_save:
                    logger.warning(f"Warning: {steps_since_last_save} steps without saving any episode. "
                                 f"Current progress: {collected_eps}/{self.config.habitat_baselines.il.dagger.update_size}. "
                                 f"Check if episodes are being skipped (skips={[skips[i] if i < len(skips) else 'N/A' for i in range(envs.num_envs)]})")
                    steps_since_last_save = 0  # 重置计数器，避免重复警告

            # 确保最后的事务已提交
            # LMDB事务会在with语句退出时自动提交，但为了安全起见，我们显式提交
            if self._is_distributed:
                logger.info(f"[Rank {self._dist_rank}] Exited collection loop, collected {collected_eps}/{local_update_size} episodes. Committing final transaction...")
            try:
                txn.commit()
                if self._is_distributed:
                    logger.info(f"[Rank {self._dist_rank}] Final LMDB transaction committed successfully ({collected_eps} episodes)")
                else:
                    logger.info("Final LMDB transaction committed successfully")
            except Exception as e:
                logger.warning(f"[Rank {self._dist_rank}] Error committing final transaction: {e}" if self._is_distributed else f"Error committing final transaction: {e}")
                import traceback
                logger.warning(traceback.format_exc())
            finally:
                # 强制同步，确保LMDB数据写入磁盘
                import os
                try:
                    os.fsync(lmdb_env.fd()) if hasattr(lmdb_env, 'fd') else None
                    if self._is_distributed:
                        logger.info(f"[Rank {self._dist_rank}] LMDB data synced to disk")
                except:
                    pass
            
            # 输出跳过的scene统计信息
            if skipped_scenes_counter:
                total_skipped = sum(skipped_scenes_counter.values())
                logger.info("=" * 80)
                logger.info(f"[Rank {self._dist_rank}] Skipped Episodes Statistics (Total: {total_skipped} skipped episodes)")
                logger.info("=" * 80)
                # 按跳过次数排序
                sorted_scenes = sorted(skipped_scenes_counter.items(), key=lambda x: x[1], reverse=True)
                for scene_name, skip_count in sorted_scenes:
                    logger.info(f"  Scene: {scene_name:30s} - Skipped: {skip_count:4d} episodes")
                logger.info("=" * 80)
            
            # 分布式训练时，在完成时进行最终统计（使用更安全的同步方式）
            # 注意：如果某个rank卡住了，这里可能会等待超时，所以添加详细的日志和异常处理
            if self._is_distributed:
                import torch.distributed as dist
                import time
                sync_start_time = time.time()
                logger.info(f"[Rank {self._dist_rank}] Starting final synchronization with other ranks (collected {collected_eps} episodes)...")
                logger.info(f"[Rank {self._dist_rank}] This rank has completed its work. Waiting for other ranks to finish...")
                try:
                    # 收集所有rank的最终统计信息
                    # 注意：如果某个rank还没完成循环，这里会等待，直到超时
                    collected_eps_tensor = torch.tensor([collected_eps], dtype=torch.int64, device=self.device)
                    gathered_eps = [torch.zeros_like(collected_eps_tensor) for _ in range(self._dist_world_size)]
                    logger.info(f"[Rank {self._dist_rank}] Calling all_gather to collect statistics from all ranks...")
                    logger.info(f"[Rank {self._dist_rank}] NOTE: If another rank is still collecting, this will wait. Check other rank logs if stuck here.")
                    dist.all_gather(gathered_eps, collected_eps_tensor)
                    sync_time = time.time() - sync_start_time
                    logger.info(f"[Rank {self._dist_rank}] all_gather completed successfully (took {sync_time:.2f}s)")
                    
                    total_collected = sum([eps.item() for eps in gathered_eps])
                    
                    # 每个rank都记录自己的完成信息和统计
                    rank_progress_str = ' '.join([f'R{i}:{eps.item()}' for i, eps in enumerate(gathered_eps)])
                    logger.info(f"[Rank {self._dist_rank}] Collection complete: {collected_eps}/{local_update_size} episodes")
                    logger.info(f"[Rank {self._dist_rank}] Total across all ranks: {total_collected} (target: {self.config.habitat_baselines.il.dagger.update_size}) [{rank_progress_str}]")
                    
                    # 只在rank 0打印汇总信息
                    if self._is_rank0():
                        logger.info("=" * 80)
                        logger.info("Distributed Data Collection Summary")
                        logger.info("=" * 80)
                        logger.info(f"Total episodes collected across all ranks: {total_collected} (target: {self.config.habitat_baselines.il.dagger.update_size})")
                        logger.info(f"Per-rank breakdown: {rank_progress_str}")
                        logger.info("=" * 80)
                    
                    # 使用barrier确保所有进程都完成了统计
                    logger.info(f"[Rank {self._dist_rank}] Calling barrier to synchronize all ranks...")
                    barrier_start = time.time()
                    dist.barrier()
                    barrier_time = time.time() - barrier_start
                    logger.info(f"[Rank {self._dist_rank}] Barrier passed, all ranks synchronized (took {barrier_time:.2f}s)")
                except Exception as e:
                    # 如果最终同步失败，记录警告但继续执行
                    sync_time = time.time() - sync_start_time
                    logger.error(f"[Rank {self._dist_rank}] Final sync failed after {sync_time:.2f}s: {e}", exc_info=True)
                    logger.warning(f"[Rank {self._dist_rank}] Continuing without full synchronization...")
                    logger.info(f"[Rank {self._dist_rank}] Collection complete: {collected_eps}/{local_update_size} episodes (sync failed, total unknown)")
                    logger.warning(f"[Rank {self._dist_rank}] If other ranks are still collecting, they should complete independently.")
            
            # 在with语句块内，确保所有LMDB操作完成
            # lmdb_env会在with语句退出时自动关闭

        # 参考PPO trainer的简单方式：直接关闭环境
        # PPO trainer在train()方法结束前只是简单地调用 self.envs.close()
        # 让Python的垃圾回收和with语句自动管理资源
        if envs is not None:
            try:
                envs.close()
            except Exception as e:
                logger.warning(f"Error closing environments: {e}")

    def train(self) -> None:
        if self.config.habitat_baselines.il.dagger.preload_lmdb_features:
            try:
                lmdb.open(self.lmdb_features_dir, readonly=True)
            except lmdb.Error as err:
                logger.error("Cannot open database for teacher forcing preload.")
                raise err
        else:
            with lmdb.open(
                self.lmdb_features_dir,
                map_size=int(self.config.habitat_baselines.il.dagger.lmdb_map_size),
            ) as lmdb_env, lmdb_env.begin(write=True) as txn:
                txn.drop(lmdb_env.open_db())

        with read_write(self.config):
            if self.config.habitat_baselines.il.dagger.p == 1.0:
                self.config.habitat.environment.iterator_options.max_scene_repeat_steps = -1
            
            # 确保每次DAgger迭代能采样到不同的episode
            # 注意：每次调用_update_dataset时都会创建新环境，应该会自动重新采样
            # config.habitat.seed 是一个整数，不是对象
            # 我们不需要在这里修改全局seed，而是在_update_dataset中为每次迭代修改

        observation_space, action_space = self._get_spaces()
        self._initialize_policy(observation_space, action_space)

        # 只有rank 0创建TensorBoard writer
        if self._is_rank0():
            writer_context = TensorboardWriter(
                self.config.habitat_baselines.tensorboard_dir,
                flush_secs=self.flush_secs,
                purge_step=0,
            )
        else:
            # 非rank 0进程使用一个空上下文管理器
            from contextlib import nullcontext
            writer_context = nullcontext()
        
        try:
            with writer_context as writer:
                # 打印训练总体信息
                total_iterations = self.config.habitat_baselines.il.dagger.iterations
                episodes_per_iteration = self.config.habitat_baselines.il.dagger.update_size
                epochs_per_iteration = self.config.habitat_baselines.il.epochs
                total_episodes = total_iterations * episodes_per_iteration
            
            logger.info("=" * 80)
            logger.info("DAgger Training Configuration")
            logger.info("=" * 80)
            logger.info(f"Total DAgger iterations: {total_iterations}")
            logger.info(f"Episodes per iteration: {episodes_per_iteration}")
            logger.info(f"Total episodes to collect: {total_episodes}")
            logger.info(f"Training epochs per iteration: {epochs_per_iteration}")
            logger.info(f"Batch size: {self.config.habitat_baselines.il.batch_size}")
            logger.info(f"Max steps per episode: 500")
            logger.info("=" * 80)
            logger.info("")
            
            # 重置epoch计数（用于checkpoint保存）
            self._current_epoch = 0
            self._last_checkpoint_percent = -1.0
            
            for dagger_it in range(self.config.habitat_baselines.il.dagger.iterations):
                # 分布式训练：同步所有进程
                if self._is_distributed:
                    import torch.distributed as dist
                    dist.barrier()
                
                if self._is_rank0():
                    logger.info("=" * 80)
                    logger.info(f"DAgger Iteration {dagger_it + 1}/{total_iterations}")
                    logger.info("=" * 80)
                
                step_id = 0
                if not self.config.habitat_baselines.il.dagger.preload_lmdb_features:
                    # 分布式训练时，所有进程都参与数据收集（每个进程收集一部分）
                    if self._is_distributed:
                        if self._is_rank0():
                            logger.info(f"Phase 1: Collecting {episodes_per_iteration} episodes across {self._dist_world_size} processes...")
                        self._update_dataset(dagger_it)
                        # _update_dataset内部已经包含了barrier同步，这里不需要再次同步
                        if self._is_rank0():
                            logger.info(f"Phase 1 completed: Collected {episodes_per_iteration} episodes across all processes")
                    else:
                        logger.info(f"Phase 1: Collecting {episodes_per_iteration} episodes...")
                        self._update_dataset(dagger_it)
                        logger.info(f"Phase 1 completed: Collected {episodes_per_iteration} episodes")
                else:
                    if self._is_rank0():
                        logger.info(f"Phase 1: Using preloaded LMDB features (skipping collection)")
                
                logger.info(f"Phase 2: Training for {epochs_per_iteration} epochs...")

                if torch.cuda.is_available():
                    with torch.cuda.device(self.device):
                        torch.cuda.empty_cache()
                gc.collect()

                # 分布式训练：使用DistributedSampler
                dataset_preload_multiplier = getattr(
                    self.config.habitat_baselines.il.dagger,
                    "dataset_preload_multiplier",
                    100,
                )

                # 分布式训练时，从所有rank的LMDB目录读取数据
                if self._is_distributed:
                    # 构建所有rank的LMDB目录列表
                    all_lmdb_dirs = [
                        os.path.join(self.lmdb_features_dir_base, f"rank_{r}")
                        for r in range(self._dist_world_size)
                    ]
                    # 只保留存在的目录
                    all_lmdb_dirs = [d for d in all_lmdb_dirs if os.path.exists(d)]
                    logger.info(f"[Rank {self._dist_rank}] Reading from {len(all_lmdb_dirs)} LMDB directories")
                    lmdb_dirs_for_dataset = all_lmdb_dirs
                else:
                    lmdb_dirs_for_dataset = self.lmdb_features_dir

                dataset = IWTrajectoryDataset(
                    lmdb_dirs_for_dataset,
                    self.config.habitat_baselines.il.use_iw,
                    inflection_weight_coef=self.config.habitat_baselines.il.inflection_weight_coef,
                    lmdb_map_size=self.config.habitat_baselines.il.dagger.lmdb_map_size,
                    batch_size=self.config.habitat_baselines.il.batch_size,
                    preload_multiplier=dataset_preload_multiplier,
                )
                
                # 注意：IWTrajectoryDataset是IterableDataset，不能使用DistributedSampler
                # 分布式训练时，需要设置dataset的rank和world_size，让其在__iter__中自己处理分片
                if self._is_distributed:
                    dataset.set_distributed_info(self._dist_rank, self._dist_world_size)
                    logger.info(f"[Rank {self._dist_rank}] Set distributed info for dataset: rank={self._dist_rank}, world_size={self._dist_world_size}")
                
                # 注意：drop_last=False，这样可以处理数据集较小的情况
                # 但如果batch_size > dataset.length，可能产生空的数据加载器
                actual_batches = max(1, dataset.length // dataset.batch_size) if dataset.length > 0 else 0
                if actual_batches == 0:
                    logger.warning(f"Dataset length ({dataset.length}) is smaller than batch size ({dataset.batch_size}), "
                                  f"no batches will be generated. Consider reducing batch_size or collecting more episodes.")
                
                # 从配置读取数据加载器参数，如果不存在则使用默认值
                dataloader_num_workers = getattr(
                    self.config.habitat_baselines.il, 'dataloader_num_workers', 0
                )  # 默认0以避免多进程LMDB访问问题
                pin_memory = getattr(
                    self.config.habitat_baselines.il, 'pin_memory', False
                )  # 默认False，如果支持可设为True以加速数据传输
                
                # IterableDataset不使用sampler，在__iter__中处理分布式分片
                diter = torch.utils.data.DataLoader(
                    dataset,
                    batch_size=self.config.habitat_baselines.il.batch_size,
                    shuffle=False,
                    collate_fn=collate_fn,
                    pin_memory=pin_memory,  # 从配置读取
                    drop_last=False,  # 改为False，允许最后一个不完整的batch
                    num_workers=dataloader_num_workers,  # 从配置读取
                )

                # 激活辅助损失管理器（参考VLN-CE）
                AuxLosses.activate()
                
                # 计算每个epoch的实际batch数（考虑drop_last=False的情况）
                # 向上取整：(length + batch_size - 1) // batch_size
                actual_batches_per_epoch = max(1, (dataset.length + dataset.batch_size - 1) // dataset.batch_size) if dataset.length > 0 else 1
                
                # 记录数据集信息（只有rank 0输出）
                if self._is_rank0():
                    logger.info("=" * 80)
                    logger.info("Training Dataset Information")
                    logger.info("=" * 80)
                    logger.info(f"Dataset length: {dataset.length}")
                    logger.info(f"Batch size: {self.config.habitat_baselines.il.batch_size}")
                    if self._is_distributed:
                        logger.info(f"Distributed training: {self._dist_world_size} processes")
                        logger.info(f"Expected batches per epoch per process: {actual_batches_per_epoch // self._dist_world_size if self._dist_world_size > 0 else actual_batches_per_epoch}")
                    else:
                        logger.info(f"Expected batches per epoch: {actual_batches_per_epoch} (drop_last=False)")
                    logger.info(f"Epochs per iteration: {epochs_per_iteration}")
                    logger.info("=" * 80)
                    logger.info("")
                
                try:
                    
                    for epoch in tqdm.trange(
                        self.config.habitat_baselines.il.epochs,
                        dynamic_ncols=True,
                        desc=f"Epoch",
                        unit="epoch",
                        leave=True,
                        disable=not self._is_rank0(),  # 只有rank 0显示进度条
                    ):
                        # 注意：IWTrajectoryDataset是IterableDataset，分布式分片在dataset内部处理
                        # 不使用DistributedSampler，所以不需要set_epoch
                        
                        # 使用tqdm.tqdm.write来避免与进度条冲突
                        if self._is_rank0():
                            tqdm.tqdm.write("=" * 80)
                            tqdm.tqdm.write(f"Epoch {epoch + 1}/{epochs_per_iteration}")
                            tqdm.tqdm.write("=" * 80)
                        
                        batch_count = 0
                        # 创建batch进度条，将loss信息显示在postfix中
                        # 使用mininterval和miniters来减少刷新频率，避免与日志冲突
                        pbar_batch = tqdm.tqdm(
                            diter,
                            total=actual_batches_per_epoch,
                            leave=False,
                            dynamic_ncols=True,
                            desc=f"  Batch",
                            unit="batch",
                            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}',
                            mininterval=0.5,  # 最少0.5秒更新一次
                            miniters=1,
                            file=sys.stdout  # 确保输出到stdout
                        )
                        
                        for batch in pbar_batch:
                            batch_count += 1
                            (
                                observations_batch,
                                prev_actions_batch,
                                not_done_masks,
                                corrected_actions_batch,
                                weights_batch,
                            ) = batch

                            # 保存原始的observations_batch用于调试和转换
                            original_obs_batch = observations_batch
                            
                            # 调试：检查batch中的指令数据（在转换之前）
                            if batch_count <= 2:
                                instruction_keys_in_batch = [k for k in original_obs_batch.keys() if 'instruction' in k.lower()]
                                if instruction_keys_in_batch:
                                    for key in instruction_keys_in_batch:
                                        instr_tensor = original_obs_batch[key]
                                        logger.info(f"Training batch {batch_count}: {key} shape: {instr_tensor.shape}, dtype: {instr_tensor.dtype}")
                                        # 尝试解析第一个时间步的指令（如果是trajectory格式）
                                        if instr_tensor.numel() > 0:
                                            first_timestep = instr_tensor[0] if instr_tensor.dim() > 0 else instr_tensor
                                            if isinstance(first_timestep, torch.Tensor):
                                                first_timestep_np = first_timestep.cpu().numpy()
                                                # 如果是uint8/int8，尝试解码为字符串
                                                if first_timestep_np.dtype in (np.uint8, np.int8) and first_timestep_np.size > 0:
                                                    # 找到非零部分
                                                    flat_arr = first_timestep_np.flatten()
                                                    non_zero_indices = np.where(flat_arr != 0)[0]
                                                    if len(non_zero_indices) > 0:
                                                        non_zero_len = non_zero_indices[-1] + 1
                                                        try:
                                                            instr_text = bytes(flat_arr[:non_zero_len]).decode('utf-8', errors='ignore').strip()
                                                            logger.info(f"Training batch {batch_count}: {key} decoded instruction: '{instr_text[:100]}'")
                                                        except:
                                                            logger.warning(f"Training batch {batch_count}: {key} failed to decode instruction")
                                                else:
                                                    logger.debug(f"Training batch {batch_count}: {key} first value: {first_timestep_np.flat[0] if first_timestep_np.size > 0 else 'empty'}")

                            # 转换observations到设备，但保持指令传感器的原始数据类型（通常是uint8/int8）
                            observations_batch = {}
                            for k, v in original_obs_batch.items():
                                if 'instruction' in k.lower():
                                    # 指令传感器保持原始数据类型（通常是uint8或int8），只移动到设备
                                    observations_batch[k] = v.to(device=self.device, non_blocking=True)
                                else:
                                    # 其他传感器转换为float32
                                    observations_batch[k] = v.to(device=self.device, dtype=torch.float32, non_blocking=True)

                            # 获取实际的policy对象（处理DDP包装）
                            policy = self._get_policy()
                            policy.train()
                            self.optimizer.zero_grad()
                            
                            # 在每个batch开始时清空辅助损失（参考VLN-CE）
                            AuxLosses.clear()
                                
                            # Build logits from features
                            features, _, aux_loss_state = policy.net(
                                observations_batch,
                                torch.zeros(
                                    observations_batch[next(iter(observations_batch))].shape[0],
                                    policy.num_recurrent_layers,
                                    policy.recurrent_hidden_size,
                                    device=self.device,
                                ),
                                prev_actions_batch.to(device=self.device, non_blocking=True),
                                not_done_masks.to(device=self.device, non_blocking=True),
                            )
                            dist = policy.action_distribution(features)
                                
                            # 重要：过滤无效动作（-1或其他超出范围的值）
                            # corrected_actions_batch可能包含-1，需要过滤
                            corrected_actions_flat = corrected_actions_batch.view(-1).to(self.device)
                            
                            # 获取动作数量（从策略配置或动作空间）
                            policy = self._get_policy()  # 获取实际的policy对象
                            if hasattr(self, '_agent_0_info'):
                                num_actions = self._agent_0_info['num_actions']
                            elif hasattr(policy, 'action_space') and hasattr(policy.action_space, 'n'):
                                num_actions = policy.action_space.n
                            else:
                                num_actions = 4  # 默认值
                            
                            # 创建有效掩码：动作值在[0, num_actions-1]范围内
                            valid_action_mask = (corrected_actions_flat >= 0) & (corrected_actions_flat < num_actions)
                            valid_action_mask = valid_action_mask.long()
                            
                            # 检查是否有无效动作
                            invalid_count = (valid_action_mask == 0).sum().item()
                            if invalid_count > 0:
                                logger.warning(f"Found {invalid_count} invalid actions in batch (out of {corrected_actions_flat.numel()}), "
                                             f"action range: [{corrected_actions_flat.min().item()}, {corrected_actions_flat.max().item()}], "
                                             f"expected range: [0, {num_actions-1}]")
                                # 将无效动作替换为0（STOP动作）以避免错误，但权重会设为0
                                corrected_actions_flat = torch.where(
                                    valid_action_mask.bool(),
                                    corrected_actions_flat,
                                    torch.zeros_like(corrected_actions_flat)
                                )
                            
                            # 计算交叉熵损失
                            ce = self.criterion(
                                dist.logits, corrected_actions_flat
                            )
                                
                            # 确保weights和valid_action_mask结合：无效动作的权重必须为0
                            iw = weights_batch.view(-1).to(self.device)
                            # 将无效动作的权重设为0
                            iw = iw * valid_action_mask.float()
                            
                            # 如果所有样本都被过滤掉，跳过这个batch
                            if iw.sum().item() == 0:
                                logger.warning("All samples in batch have invalid actions or zero weights, skipping this batch")
                                step_id += 1
                                continue
                            
                            action_loss = (ce * iw).sum() / iw.sum()  # 使用加权平均而不是mean

                            # ===== 切换到VLN-CE风格的辅助损失计算 =====
                            # 将Falcon原有的辅助损失模块转换为VLN-CE格式
                            aux_mask = (weights_batch.view(-1) > 0).to(self.device)
                            
                            # 如果policy有aux_loss_modules，计算并将它们注册到AuxLosses
                            # 检查observations_batch中是否包含辅助损失模块需要的键
                            # 注意：observations_batch中的键名可能是agent_0_xxx格式，需要映射
                            
                            # 创建传感器键名映射（从辅助损失期望的键名到实际observations中的键名）
                            sensor_mapping = {
                                'human_num_sensor': ['agent_0_human_num_sensor', 'human_num_sensor'],
                                'oracle_humanoid_future_trajectory': ['agent_0_oracle_humanoid_future_trajectory', 'oracle_humanoid_future_trajectory'],
                            }
                            
                            # 创建映射后的observations字典，供辅助损失模块使用
                            mapped_observations = {}
                            for key, value in observations_batch.items():
                                # 直接添加原始键
                                mapped_observations[key] = value
                                # 如果是agent_0_xxx格式，也添加不带前缀的版本（如果辅助损失期望这个格式）
                                if key.startswith('agent_0_'):
                                    base_key = key.replace('agent_0_', '', 1)
                                    if base_key not in mapped_observations:
                                        mapped_observations[base_key] = value
                            
                            batch_for_aux = dict(observations=mapped_observations)
                            
                            # 检查哪些辅助损失模块可以运行（有必要的传感器）
                            # 获取实际的policy对象（处理DDP包装）
                            policy = self._get_policy()
                            available_aux_modules = {}
                            aux_loss_modules = getattr(policy, 'aux_loss_modules', {})
                            for aux_name, aux_mod in aux_loss_modules.items():
                                can_run = True
                                # 根据辅助任务名称检查需要的传感器
                                if 'human_num' in aux_name.lower() or 'people_counting' in aux_name.lower():
                                    if 'human_num_sensor' not in mapped_observations:
                                        can_run = False
                                        logger.debug(f"Skipping {aux_name}: missing human_num_sensor")
                                if 'trajectory' in aux_name.lower() or 'future_trajectory' in aux_name.lower():
                                    if 'oracle_humanoid_future_trajectory' not in mapped_observations:
                                        can_run = False
                                        logger.debug(f"Skipping {aux_name}: missing oracle_humanoid_future_trajectory")
                                
                                if can_run:
                                    available_aux_modules[aux_name] = aux_mod
                            
                            if len(available_aux_modules) < len(aux_loss_modules):
                                logger.info(f"Only {len(available_aux_modules)}/{len(aux_loss_modules)} auxiliary loss modules can run "
                                          f"(missing sensors in observations)")
                            
                            for aux_name, aux_mod in available_aux_modules.items():
                                try:
                                    aux_out = aux_mod(aux_loss_state, batch_for_aux)
                                    if isinstance(aux_out, dict) and "loss" in aux_out:
                                        aux_loss_tensor = aux_out["loss"]
                                        # 获取权重alpha（从配置中读取，如果没有则使用默认值1.0）
                                        aux_cfg = self.config.habitat_baselines.il.auxiliary_losses
                                        alpha = 1.0
                                        if aux_cfg and hasattr(aux_cfg, aux_name):
                                            alpha = getattr(getattr(aux_cfg, aux_name), "loss_scale", 1.0)
                                        # 确保loss形状正确（应该是[batch_size]或[T, N]）
                                        if aux_loss_tensor.dim() == 0:
                                            # 标量，需要扩展到batch维度
                                            aux_loss_tensor = aux_loss_tensor.unsqueeze(0).expand(aux_mask.shape[0])
                                        elif aux_loss_tensor.dim() > 1:
                                            # 多维，展平
                                            aux_loss_tensor = aux_loss_tensor.view(-1)
                                        # 注册到AuxLosses
                                        AuxLosses.register_loss(aux_name, aux_loss_tensor, alpha=alpha)
                                except KeyError as e:
                                    logger.warning(f"Failed to compute auxiliary loss {aux_name}: missing key {e}, skipping")
                                except Exception as e:
                                    logger.warning(f"Failed to compute auxiliary loss {aux_name}: {e}, skipping")
                                    import traceback
                                    logger.debug(traceback.format_exc())
                            
                            # 使用AuxLosses.reduce计算总辅助损失（参考VLN-CE）
                            if AuxLosses.has_losses():
                                aux_total = AuxLosses.reduce(aux_mask)
                                if not isinstance(aux_total, torch.Tensor):
                                    aux_total = torch.tensor(aux_total, device=self.device)
                            else:
                                # 如果没有注册任何辅助损失，使用0
                                aux_total = torch.tensor(0.0, device=self.device)

                            loss = action_loss + aux_total
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                            self.optimizer.step()

                            # 详细记录训练损失，便于监控（每个batch都输出）
                            loss_val = loss.item()
                            action_loss_val = action_loss.item()
                            aux_loss_val = aux_total.item()
                            
                            # 更新进度条的postfix，显示loss信息
                            pbar_batch.set_postfix({
                                'loss': f'{loss_val:.4f}',
                                'act_loss': f'{action_loss_val:.4f}',
                                'aux_loss': f'{aux_loss_val:.4f}'
                            })
                            
                            # 使用格式化的输出，便于监控（每10个batch或前5个batch输出一次，避免日志过多）
                            # 使用tqdm.tqdm.write来避免与进度条冲突
                            if batch_count <= 5 or batch_count % 10 == 0:
                                tqdm.tqdm.write(
                                    f"[DAgger Iter {dagger_it+1}/{total_iterations}] "
                                    f"[Epoch {epoch+1}/{epochs_per_iteration}] "
                                    f"[Batch {batch_count}/{actual_batches_per_epoch}] "
                                    f"[Step {step_id}] "
                                    f"Loss: {loss_val:.6f} "
                                    f"(Action: {action_loss_val:.6f}, Aux: {aux_loss_val:.6f})"
                                )
                            
                            # 记录到TensorBoard
                            # 只有rank 0写入TensorBoard
                            if self._is_rank0():
                                writer.add_scalar(f"train_loss_iter_{dagger_it}", loss_val, step_id)
                                writer.add_scalar(f"train_action_loss_iter_{dagger_it}", action_loss_val, step_id)
                                writer.add_scalar(f"train_aux_loss_iter_{dagger_it}", aux_loss_val, step_id)
                                writer.add_scalar("train_loss_total", loss_val, step_id + dagger_it * 10000)
                                writer.add_scalar("train_action_loss_total", action_loss_val, step_id + dagger_it * 10000)
                                writer.add_scalar("train_aux_loss_total", aux_loss_val, step_id + dagger_it * 10000)
                            step_id += 1
                        
                        # 关闭batch进度条
                        pbar_batch.close()

                    # 更新当前epoch计数
                    self._current_epoch = dagger_it * self.config.habitat_baselines.il.epochs + epoch + 1
                    
                    # 在每个epoch结束后，根据配置决定是否保存检查点（类似RL训练器）
                    # 使用tqdm.tqdm.write来避免与进度条冲突
                    tqdm.tqdm.write("")
                    tqdm.tqdm.write("=" * 80)
                    tqdm.tqdm.write(f"Epoch {epoch + 1}/{epochs_per_iteration} completed!")
                    tqdm.tqdm.write(f"  Processed {batch_count} batches in this epoch")
                    tqdm.tqdm.write(f"  Total training steps so far: {step_id}")
                    tqdm.tqdm.write(f"  Training progress: {self.percent_done() * 100:.2f}% ({self._current_epoch}/{self._total_epochs} epochs)")
                    
                    # 只在should_checkpoint()返回True时保存checkpoint（只有rank 0保存）
                    if self.should_checkpoint() and self._is_rank0():
                        checkpoint_name = f"ckpt.{dagger_it * self.config.habitat_baselines.il.epochs + epoch + 1}.pth"
                        tqdm.tqdm.write(f"Saving checkpoint: {checkpoint_name}")
                        try:
                            self.save_checkpoint(
                                self._get_model_state_dict(), checkpoint_name
                            )
                            tqdm.tqdm.write(f"✓ Checkpoint saved successfully")
                        except Exception as e:
                            tqdm.tqdm.write(f"✗ Failed to save checkpoint: {e}")
                            import traceback
                            tqdm.tqdm.write(traceback.format_exc())
                    elif self.should_checkpoint() and not self._is_rank0():
                        # 非rank 0进程不保存checkpoint，但仍需同步等待
                        if self._is_distributed:
                            import torch.distributed as dist
                            dist.barrier()
                    else:
                        if self._is_rank0():
                            tqdm.tqdm.write(f"Skipping checkpoint (not at checkpoint interval)")
                    
                    # 保存resume state（类似RL训练器，用于SLURM作业恢复，只有rank 0保存）
                    if self._should_save_resume_state() and self._is_rank0():
                        tqdm.tqdm.write(f"Saving resume state...")
                        try:
                            resume_state = dict(
                                policy_state_dict=self._get_model_state_dict(),
                                optimizer_state_dict=self.optimizer.state_dict(),
                                config=self.config,
                                current_epoch=self._current_epoch,
                                current_dagger_iter=dagger_it,
                                current_epoch_in_iter=epoch,
                                total_epochs=self._total_epochs,
                                total_iterations=total_iterations,
                                step_id=step_id,
                                run_id=writer.get_run_id() if (writer is not None and hasattr(writer, 'get_run_id')) else None,
                            )
                            save_resume_state(
                                resume_state,
                                self.config,
                            )
                            tqdm.tqdm.write(f"✓ Resume state saved successfully")
                        except Exception as e:
                            tqdm.tqdm.write(f"✗ Failed to save resume state: {e}")
                            import traceback
                            tqdm.tqdm.write(traceback.format_exc())
                    elif self._should_save_resume_state() and not self._is_rank0():
                        # 非rank 0进程不保存resume state，但仍需同步等待
                        if self._is_distributed:
                            import torch.distributed as dist
                            dist.barrier()
                    
                    tqdm.tqdm.write("=" * 80)
                    tqdm.tqdm.write("")
                    
                    # 强制同步，确保数据写入磁盘
                    sys.stdout.flush()
                finally:
                    # 确保在所有情况下都清理资源
                    # 停用辅助损失管理器（参考VLN-CE）
                    AuxLosses.deactivate()
                    
                    # 参考PPO trainer：不需要复杂的资源清理
                    # Python的垃圾回收和with语句会自动管理资源
                    # 只需简单清理CUDA缓存即可
                    if torch.cuda.is_available():
                        with torch.cuda.device(self.device):
                            torch.cuda.empty_cache()
                
                logger.info(f"DAgger Iteration {dagger_it + 1}/{total_iterations} completed")
                
                # 参考PPO trainer：不需要在迭代之间做复杂清理
                # 让Python的垃圾回收和with语句自动管理资源
                # 如果需要在迭代之间清理，只做最基本的CUDA缓存清理
                if dagger_it < total_iterations - 1:  # 不是最后一轮
                    if torch.cuda.is_available():
                        with torch.cuda.device(self.device):
                            torch.cuda.empty_cache()
                    
            # 分布式训练：同步所有进程
            if self._is_distributed:
                import torch.distributed as dist
                dist.barrier()
            
            if self._is_rank0():
                logger.info("=" * 80)
                logger.info("All DAgger iterations completed!")
                logger.info("=" * 80)
                
                # 保存最终检查点，确保训练结果被保存（只有rank 0保存）
                final_checkpoint_name = f"ckpt.final.pth"
                logger.info(f"Saving final checkpoint: {final_checkpoint_name}")
                try:
                    self.save_checkpoint(self._get_model_state_dict(), final_checkpoint_name)
                    logger.info("Final checkpoint saved successfully")
                except Exception as e:
                    logger.error(f"Failed to save final checkpoint: {e}")
            
            # 确保所有数据写入磁盘
            sys.stdout.flush()
            
            # 分布式训练：正确清理进程组，避免资源泄漏
            if self._is_distributed:
                import torch.distributed as dist
                try:
                    # 确保所有进程都完成后再清理
                    if dist.is_initialized():
                        # 先同步所有进程
                        dist.barrier(timeout=torch.distributed.default_pg_timeout)
                        # 正确关闭进程组
                        dist.destroy_process_group()
                        if self._is_rank0():
                            logger.info("Distributed process group destroyed successfully")
                except Exception as e:
                    # 如果清理失败，记录警告但不影响训练结果
                    logger.warning(f"Error destroying process group: {e}")
                    import traceback
                    logger.debug(traceback.format_exc())
            
            # 清理CUDA缓存，释放内存
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            except Exception as e:
                logger.warning(f"Error clearing CUDA cache: {e}")
            
            # 强制垃圾回收，释放Python对象占用的内存
            # gc已在文件顶部导入，直接使用即可
            gc.collect()
            
            # 参考PPO trainer：不需要复杂的最终清理
            # PPO trainer在train()方法结束时只是让Python自然退出
            # 让with语句（TensorboardWriter）和Python的垃圾回收自动管理资源
        except Exception as e:
            # 如果训练过程中发生异常，也要确保清理资源
            logger.error(f"Training failed with error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            # 即使发生异常，也要尝试清理分布式进程组
            if self._is_distributed:
                import torch.distributed as dist
                try:
                    if dist.is_initialized():
                        dist.destroy_process_group()
                except:
                    pass
            
            # 重新抛出异常，让调用者知道训练失败
            raise


