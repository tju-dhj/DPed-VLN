#!/usr/bin/env python3

import re
import sys
import os
import gc
import glob
import json
import random
from collections import OrderedDict, defaultdict
from typing import Dict, Tuple, List, Any
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import tqdm
from gym import spaces
from habitat import logger
from habitat_baselines.common.base_il_trainer import BaseILTrainer
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.tensorboard_utils import TensorboardWriter, get_writer
from habitat_baselines.common.aux_losses import AuxLosses
from habitat_baselines.rl.ddppo.ddp_utils import (
    load_resume_state,
    save_resume_state,
)
from habitat_baselines.utils.common import (
    batch_obs,
    get_checkpoint_id,
    inference_mode,
)
from habitat_baselines.rl.ddppo.policy.resnet_policy import PointNavResNetPolicy

# 导入 habitat-lab 的几何工具，确保 GPS compass / heading 计算与模拟器完全一致
from habitat.tasks.utils import cartesian_to_polar
from habitat.utils.geometry_utils import quaternion_rotate_vector

# LoRA 支持
try:
    from peft import LoraConfig, get_peft_model, PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    LoraConfig = None
    get_peft_model = None
    PeftModel = None


class DirectFileDataset(torch.utils.data.IterableDataset):
    """直接从文件系统读取数据的Dataset，不经过模拟器"""
    
    def __init__(
        self,
        data_root: str,
        use_iw: bool = True,
        inflection_weight_coef: float = 1.0,
        batch_size: int = 1,
        max_episodes: int = -1,
        instruction_priority: List[str] = None,
        max_episode_length: int = 400,  # 最大episode长度，防止OOM
        use_mixed_instructions: bool = False,  # 是否启用混合指令模式
        rgb_data_roots: List[str] = None,  # 用于新版本数据集的图像根目录候选列表
        dataset_type: str = "directory",  # directory/json
        disable_rgb_decode: bool = False,
        forward_step_size: float = 0.25,  # 前进步长（米），与habitat-lab默认值一致
        turn_angle_deg: float = 10.0,  # 转弯角度（度），与habitat-lab默认值一致
        window_size: int = 0,  # 滑动窗口大小：0=整条episode，>=1=滑动窗口（如8/16/32）
        skip_episodes_with_empty_instruction: bool = True,  # 是否跳过空/默认指令的episode
        future_step: int = 4,  # 与 OracleHumanoidFutureTrajectorySensor / RL auxiliary loss 保持一致
    ):
        super().__init__()
        self.data_root = data_root
        self.batch_size = batch_size
        self.use_iw = use_iw
        self.inflection_weight_coef = inflection_weight_coef
        self.max_episode_length = max_episode_length  # 保存最大长度限制
        self.dataset_type = dataset_type
        self.disable_rgb_decode = disable_rgb_decode
        self.forward_step_size = forward_step_size
        self.turn_angle_deg = turn_angle_deg
        self.window_size = window_size if window_size >= 1 else 0  # 0 = 整条episode
        self.skip_episodes_with_empty_instruction = skip_episodes_with_empty_instruction
        self.future_step = int(future_step) if int(future_step) > 0 else 4
        self.rgb_data_roots = [r for r in (rgb_data_roots or []) if r]
        self._episode_rgb_cache: Dict[Tuple[str, str, str], str] = {}
        # 统计被过滤的 episode 数量
        self._skipped_empty_instruction = 0
        self._skipped_default_instruction = 0
        
        # 指令优先级列表（默认优先级）
        if instruction_priority is None:
            instruction_priority = ["instruction_vl_level_2", "instruction_level_2", "inst_navcomposer_v2"]
        self.instruction_priority = instruction_priority
        
        if use_iw:
            self.inflec_weights = torch.tensor([1.0, inflection_weight_coef])
        else:
            self.inflec_weights = torch.tensor([1.0, 1.0])
        
        # 扫描所有episode目录/文件
        if self.dataset_type == "json":
            self.episode_paths = self._scan_json_episodes(data_root, max_episodes)
        else:
            self.episode_paths = self._scan_episodes(data_root, max_episodes)
        
        # 检查是否启用混合指令模式（同时使用v1和v2）
        # 如果instruction_priority包含多个指令类型且use_mixed_instructions=True，则每个episode生成多个样本
        self.use_mixed_instructions = use_mixed_instructions and len(instruction_priority) > 1
        
        # 如果启用混合模式，每个episode会生成多个样本（每个指令类型一个）
        if self.use_mixed_instructions:
            self.episode_entries = []
            for episode_ref in self.episode_paths:
                for inst_type in instruction_priority:
                    if self.dataset_type == "json":
                        self.episode_entries.append((episode_ref, inst_type))
                        continue
                    episode_path = self._episode_ref_path(episode_ref)
                    inst_file = os.path.join(episode_path, inst_type, "0.txt")
                    if os.path.exists(inst_file):
                        self.episode_entries.append((episode_ref, inst_type))
            self.length = len(self.episode_entries)
        else:
            self.episode_entries = [(ep, None) for ep in self.episode_paths]
            self.length = len(self.episode_paths)
        
        # print(f"[DirectFileDataset] Found {self.length} episode entries in {data_root}")
        # print(f"[DirectFileDataset] Dataset type: {self.dataset_type}")
        # print(f"[DirectFileDataset] RGB data roots: {self.rgb_data_roots if self.rgb_data_roots else '[episode-local rgb/]'}")
        # print(f"[DirectFileDataset] Scanned {len(self.episode_paths)} unique episode directories")
        # print(f"[DirectFileDataset] Instruction priority: {self.instruction_priority}")
        # print(f"[DirectFileDataset] Max episode length: {self.max_episode_length} steps (to prevent OOM)")
        # print(f"[DirectFileDataset] Use mixed instructions: {self.use_mixed_instructions}")
        # if self.use_mixed_instructions:
        #     print(f"[DirectFileDataset] Mixed mode: Each episode generates {len(self.instruction_priority)} samples")
        # print(f"[DirectFileDataset] First 5 episode paths:")
        for i, ep_ref in enumerate(self.episode_paths[:5]):
            ep_path = self._episode_ref_path(ep_ref)
            ep_index = self._episode_ref_index(ep_ref)
            episode_id = os.path.basename(ep_path.rstrip('/'))
            print(f"  [{i}] episode_id={episode_id}, episode_index={ep_index}, path={ep_path}")
        if self.use_mixed_instructions and len(self.episode_entries) > 0:
            print(f"[DirectFileDataset] First 5 episode entries (with instruction types):")
            for i, (ep_ref, inst_type) in enumerate(self.episode_entries[:5]):
                ep_path = self._episode_ref_path(ep_ref)
                ep_index = self._episode_ref_index(ep_ref)
                episode_id = os.path.basename(ep_path.rstrip('/'))
                print(f"  [{i}] episode_id={episode_id}, episode_index={ep_index}, instruction_type={inst_type}, path={ep_path}")
        
        self._dist_rank = 0
        self._dist_world_size = 1
    
    def _scan_json_episodes(self, data_root: str, max_episodes: int = -1) -> List[Tuple[str, int]]:
        """扫描 JSON/JSON.GZ 数据根目录，展开文件内的每条 episode。"""
        episode_refs = []
        json_files = glob.glob(os.path.join(data_root, "*.json")) + glob.glob(os.path.join(data_root, "*.json.gz"))
        json_files = sorted(json_files)
        for json_path in json_files:
            if not os.path.isfile(json_path):
                continue
            try:
                episode_data = self._read_json_with_fallback(json_path)
                if isinstance(episode_data, dict) and 'episodes' in episode_data:
                    num_episodes = len(episode_data.get('episodes') or [])
                else:
                    num_episodes = 1
            except Exception:
                num_episodes = 1
            for episode_index in range(num_episodes):
                episode_refs.append((json_path, episode_index))
                if max_episodes > 0 and len(episode_refs) >= max_episodes:
                    break
            if max_episodes > 0 and len(episode_refs) >= max_episodes:
                break
        return episode_refs

    def _read_json_with_fallback(self, path: str):
        if path.endswith('.gz'):
            import gzip
            with gzip.open(path, 'rt', encoding='utf-8') as f:
                return json.load(f)
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _episode_ref_path(self, episode_ref: Any) -> str:
        if isinstance(episode_ref, (tuple, list)):
            return str(episode_ref[0])
        return str(episode_ref)

    def _episode_ref_index(self, episode_ref: Any):
        if isinstance(episode_ref, (tuple, list)) and len(episode_ref) > 1:
            return int(episode_ref[1])
        return None
    
    def _episode_basename(self, episode_path: str) -> str:
        episode_path = self._episode_ref_path(episode_path)
        name = Path(episode_path).name
        if name.endswith('.json.gz'):
            return name[:-8]
        if name.endswith('.json'):
            return name[:-5]
        return Path(episode_path).stem


    def _resolve_rgb_dir(self, episode_path: str, episode_id_str: str = None, scene_id: str = None) -> str:
        """在多个rgb_data_roots中寻找对应episode的rgb目录"""
        episode_path = self._episode_ref_path(episode_path)
        cache_key = (episode_path, str(episode_id_str or ""), str(scene_id or ""))
        if cache_key in self._episode_rgb_cache:
            return self._episode_rgb_cache[cache_key]

        episode_key = self._episode_basename(episode_path)
        search_roots = self.rgb_data_roots[:] if self.rgb_data_roots else [os.path.dirname(episode_path)]

        scene_base = ""
        scene_dir_name = ""
        if scene_id:
            scene_base = Path(str(scene_id)).stem
            scene_dir_name = scene_base.replace(".basis", "")
            if scene_dir_name.endswith(".glb"):
                scene_dir_name = Path(scene_dir_name).stem

        candidate_suffixes = []
        if scene_dir_name and episode_id_str:
            candidate_suffixes.extend(
                [
                    os.path.join(scene_dir_name, str(episode_id_str), "rgb"),
                    os.path.join(scene_dir_name, str(episode_id_str)),
                ]
            )
        if scene_base and episode_id_str:
            candidate_suffixes.extend(
                [
                    os.path.join(scene_base, str(episode_id_str), "rgb"),
                    os.path.join(scene_base, str(episode_id_str)),
                ]
            )
        if episode_id_str:
            candidate_suffixes.extend(
                [
                    os.path.join(f"{episode_key}.basis", str(episode_id_str), "rgb"),
                    os.path.join(f"{episode_key}.basis", str(episode_id_str)),
                ]
            )
        candidate_suffixes.extend(
            [
                os.path.join(episode_key, "rgb"),
                episode_key,
                os.path.join(f"{episode_key}.basis", "rgb"),
                f"{episode_key}.basis",
            ]
        )

        deduped_suffixes = []
        seen_suffixes = set()
        for suffix in candidate_suffixes:
            if suffix and suffix not in seen_suffixes:
                deduped_suffixes.append(suffix)
                seen_suffixes.add(suffix)
        candidate_suffixes = deduped_suffixes

        alt_keys = [episode_key]
        if scene_dir_name:
            alt_keys.append(scene_dir_name)
        if scene_base and scene_base not in alt_keys:
            alt_keys.append(scene_base)
        if episode_key.endswith('_v1') or episode_key.endswith('_v2'):
            alt_keys.append(episode_key.rsplit('_', 1)[0])

        # if getattr(self, '_debug_resolve_paths', False):
        #     print(f"[DirectFileDataset DEBUG] resolve episode_path={episode_path}")
        #     print(f"[DirectFileDataset DEBUG]   episode_key={episode_key}")
        #     print(f"[DirectFileDataset DEBUG]   episode_id_str={episode_id_str}")
        #     print(f"[DirectFileDataset DEBUG]   scene_base={scene_base}")
        #     print(f"[DirectFileDataset DEBUG]   scene_dir_name={scene_dir_name}")
        #     print(f"[DirectFileDataset DEBUG]   search_roots={search_roots}")
        #     print(f"[DirectFileDataset DEBUG]   candidate_suffixes={candidate_suffixes}")
        #     print(f"[DirectFileDataset DEBUG]   alt_keys={alt_keys}")

        for root in search_roots:
            for suffix in candidate_suffixes:
                candidate = os.path.join(root, suffix)
                # if getattr(self, '_debug_resolve_paths', False):
                    # print(f"[DirectFileDataset DEBUG]   try={candidate} exists={os.path.isdir(candidate)}")
                if os.path.isdir(candidate):
                    self._episode_rgb_cache[cache_key] = candidate
                    # if getattr(self, '_debug_resolve_paths', False):
                        # print(f"[DirectFileDataset DEBUG]   matched={candidate}")
                    return candidate

            for alt_key in alt_keys:
                matches = [p for p in glob.glob(os.path.join(root, "*")) if os.path.isdir(p) and alt_key in os.path.basename(p)]
                # if getattr(self, '_debug_resolve_paths', False):
                    # print(f"[DirectFileDataset DEBUG]   fuzzy alt_key={alt_key} matches={matches}")
                if len(matches) == 1:
                    if episode_id_str:
                        nested_rgb = os.path.join(matches[0], str(episode_id_str), 'rgb')
                        nested_dir = os.path.join(matches[0], str(episode_id_str))
                        if os.path.isdir(nested_rgb):
                            self._episode_rgb_cache[cache_key] = nested_rgb
                            return nested_rgb
                        if os.path.isdir(nested_dir):
                            self._episode_rgb_cache[cache_key] = nested_dir
                            return nested_dir
                    self._episode_rgb_cache[cache_key] = matches[0]
                    return matches[0]
                if len(matches) > 1:
                    for match in sorted(matches):
                        if not os.path.isdir(match):
                            continue
                        if episode_id_str:
                            nested_rgb = os.path.join(match, str(episode_id_str), 'rgb')
                            nested_dir = os.path.join(match, str(episode_id_str))
                            if os.path.isdir(nested_rgb):
                                self._episode_rgb_cache[cache_key] = nested_rgb
                                return nested_rgb
                            if os.path.isdir(nested_dir):
                                self._episode_rgb_cache[cache_key] = nested_dir
                                return nested_dir
                        self._episode_rgb_cache[cache_key] = match
                        return match

        legacy_rgb = os.path.join(os.path.dirname(episode_path), "rgb")
        if os.path.isdir(legacy_rgb):
            self._episode_rgb_cache[cache_key] = legacy_rgb
            return legacy_rgb

        raise FileNotFoundError(
            f"Unable to resolve rgb directory for episode={episode_path}. "
            f"Searched roots={search_roots}, episode_key={episode_key}, scene_base={scene_base}, episode_id_str={episode_id_str}, candidates={candidate_suffixes}"
        )



    def _get_episode_debug_info(self, episode_path: str) -> Tuple[str, str, str]:
        """读取 JSON 头部信息，用于调试输出 episode_id / scene_id / 文件名。"""
        episode_index = self._episode_ref_index(episode_path)
        episode_path = self._episode_ref_path(episode_path)
        episode_data = self._read_json_with_fallback(episode_path)
        if isinstance(episode_data, dict) and 'episodes' in episode_data:
            episodes = episode_data.get('episodes') or []
            if len(episodes) > 0:
                selected_index = episode_index if episode_index is not None else 0
                if selected_index >= len(episodes):
                    raise IndexError(
                        f"Episode index {selected_index} out of range for {episode_path}"
                    )
                if isinstance(episodes[selected_index], dict):
                    episode_data = episodes[selected_index]

        file_name = Path(episode_path).name
        if file_name.endswith('.json.gz'):
            file_name = file_name[:-8]
        elif file_name.endswith('.json'):
            file_name = file_name[:-5]

        episode_id = str(episode_data.get('episode_id') or episode_data.get('episode_uid') or file_name)
        scene_id = str(episode_data.get('scene_id') or episode_data.get('scene_name') or '')
        if episode_index is not None:
            file_name = f"{file_name}#{episode_index}"
        return episode_id, scene_id, file_name

    def _parse_info_human_waypoints(
        self, info: Dict[str, Any], max_humans: int = 6
    ) -> Tuple[int, int, np.ndarray]:
        """从 info 字段解析 human_{id}_waypoint_{idx}_{position|rotation} 数据。"""
        human_waypoint_pattern = re.compile(
            r"^human_(\d+)_waypoint_(\d+)_(position|rotation)$"
        )

        humans: Dict[int, Dict[int, Dict[str, Any]]] = defaultdict(dict)
        for key, value in info.items():
            match = human_waypoint_pattern.match(key)
            if match is None:
                continue

            human_id = int(match.group(1))
            waypoint_id = int(match.group(2))
            field_name = match.group(3)
            waypoint_entry = humans[human_id].setdefault(waypoint_id, {})
            waypoint_entry[field_name] = value

        human_ids = sorted(humans.keys())
        human_num = len(human_ids)
        waypoint_num = (
            max((len(waypoints) for waypoints in humans.values()), default=0)
            if human_num > 0
            else 0
        )

        trajectory = np.zeros((max_humans, waypoint_num, 4), dtype=np.float32)
        for output_human_idx, human_id in enumerate(human_ids[:max_humans]):
            waypoint_map = humans[human_id]
            for waypoint_id in sorted(waypoint_map.keys()):
                if waypoint_id >= waypoint_num:
                    continue
                waypoint = waypoint_map[waypoint_id]

                position = waypoint.get("position")
                if isinstance(position, (list, tuple)) and len(position) >= 3:
                    trajectory[output_human_idx, waypoint_id, :3] = np.asarray(
                        position[:3], dtype=np.float32
                    )

                rotation = waypoint.get("rotation")
                if isinstance(rotation, (int, float)):
                    trajectory[output_human_idx, waypoint_id, 3] = np.float32(rotation)
                elif isinstance(rotation, (list, tuple)) and len(rotation) > 0:
                    trajectory[output_human_idx, waypoint_id, 3] = np.float32(
                        rotation[0]
                    )

        return human_num, waypoint_num, trajectory

    def _load_episode(self, episode_path: str, instruction_type: str = None) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
        """从 JSON.GZ 文件加载完整 episode，RGB 从独立目录读取"""
        episode_index = self._episode_ref_index(episode_path)
        episode_path = self._episode_ref_path(episode_path)
        episode_data = self._read_json_with_fallback(episode_path)

        if isinstance(episode_data, dict) and 'episodes' in episode_data:
            if len(episode_data['episodes']) == 0:
                raise ValueError(f"No episodes found in {episode_path}")
            selected_index = episode_index if episode_index is not None else 0
            if selected_index >= len(episode_data['episodes']):
                raise IndexError(
                    f"Episode index {selected_index} out of range for {episode_path}"
                )
            episode_data = episode_data['episodes'][selected_index]

        episode_key = Path(episode_path).name
        if episode_key.endswith('.json.gz'):
            episode_key = episode_key[:-8]
        elif episode_key.endswith('.json'):
            episode_key = episode_key[:-5]

        scene_id = episode_data.get('scene_id') or episode_data.get('scene_name') or episode_key
        episode_id_str = str(episode_data.get('episode_id') or episode_data.get('episode_uid') or episode_key)
        action_seq = episode_data.get('gt_action')
        if action_seq is None:
            action_seq = episode_data.get('action')
        if action_seq is None:
            action_seq = episode_data.get('oracle_actions')
        # if action_seq is None and getattr(self, '_debug_resolve_paths', False):
            # print(f"[DirectFileDataset DEBUG]   available_action_keys={[k for k in episode_data.keys() if 'action' in k.lower()]}")
        if action_seq is None:
            raise ValueError(f"Missing actions in {episode_path}")
        actions = np.asarray(action_seq, dtype=np.int64)
        if len(actions) == 0:
            raise ValueError(f"No actions found in {episode_path}")

        instruction_text = ""
        instruction_candidates = [
            episode_data.get('instruction'),
            episode_data.get('instruction_text'),
            episode_data.get('instructions'),
        ]
        for cand in instruction_candidates:
            if isinstance(cand, str) and cand.strip():
                instruction_text = cand.strip()
                break
            if isinstance(cand, list) and len(cand) > 0 and isinstance(cand[0], str) and cand[0].strip():
                instruction_text = cand[0].strip()
                break

        # if getattr(self, '_debug_resolve_paths', False):
        #     print(f"[DirectFileDataset DEBUG] json_episode episode_path={episode_path}")
        #     print(f"[DirectFileDataset DEBUG]   episode_key={episode_key}, episode_id={episode_id_str}, scene_id={scene_id}")
        #     print(f"[DirectFileDataset DEBUG]   top_keys={list(episode_data.keys())[:20]}")
        #     print(f"[DirectFileDataset DEBUG]   action_len={len(actions)}, action_head={actions[:10].tolist()}")
        #     print(f"[DirectFileDataset DEBUG]   instruction_text={instruction_text[:120] if instruction_text else '(empty)'}")

        instruction_bytes = instruction_text.encode('utf-8')
        instruction_array = np.frombuffer(instruction_bytes, dtype=np.uint8)
        max_instr_len = 512
        if len(instruction_array) > max_instr_len:
            instruction_array = instruction_array[:max_instr_len]
        else:
            padded = np.zeros(max_instr_len, dtype=np.uint8)
            padded[:len(instruction_array)] = instruction_array
            instruction_array = padded

        num_steps = len(actions)
        if self.max_episode_length > 0 and num_steps > self.max_episode_length:
            num_steps = self.max_episode_length
        if num_steps == 0:
            raise ValueError(f"Invalid episode with 0 steps: {episode_path}")
        actions = actions[:num_steps]

        rgb_dir = self._resolve_rgb_dir(episode_path, episode_id_str, scene_id)
        # if getattr(self, '_debug_resolve_paths', False):
        #     print(f"[DirectFileDataset DEBUG]   rgb_dir={rgb_dir}")
        #     print(f"[DirectFileDataset DEBUG]   rgb_dir_parent={os.path.dirname(rgb_dir)}")

        rgb_patterns = [
            os.path.join(rgb_dir, "*.jpg"),
            os.path.join(rgb_dir, "*.jpeg"),
            os.path.join(rgb_dir, "*.png"),
            os.path.join(rgb_dir, "rgb", "*.jpg"),
            os.path.join(rgb_dir, "rgb", "*.jpeg"),
            os.path.join(rgb_dir, "rgb", "*.png"),
        ]
        rgb_files = []
        for pattern in rgb_patterns:
            rgb_files.extend(glob.glob(pattern))
        rgb_files = sorted(set(rgb_files), key=lambda x: int(os.path.basename(x).split('_')[0]) if os.path.basename(x).split('_')[0].isdigit() else os.path.basename(x))
        # if getattr(self, '_debug_resolve_paths', False):
            # print(f"[DirectFileDataset DEBUG]   rgb_patterns={rgb_patterns}")
            # print(f"[DirectFileDataset DEBUG]   rgb_count={len(rgb_files)}, rgb_head={[os.path.basename(p) for p in rgb_files[:10]]}")
        if len(rgb_files) == 0:
            raise ValueError(f"No RGB images found in {rgb_dir} for {episode_path}")
        rgb_files = rgb_files[:num_steps]
        num_steps = min(num_steps, len(rgb_files))
        actions = actions[:num_steps]

        if self.disable_rgb_decode:
            # 当禁用RGB解码时，仍然需要正确的RGB格式（uint8 [0-255]），而不是全零float32
            # 使用中等灰色填充，避免全零导致CLIP预处理器产生无效特征
            rgb_array = np.full((num_steps, 256, 256, 3), 128, dtype=np.uint8)
        else:
            # 使用uint8 [0-255]格式，与DPed RL配置一致
            rgb_array = np.empty((num_steps, 256, 256, 3), dtype=np.uint8)
            for i, rgb_file in enumerate(rgb_files[:num_steps]):
                with Image.open(rgb_file) as img:
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    rgb_array[i] = np.asarray(img, dtype=np.uint8)

        depth_dir = os.path.join(os.path.dirname(rgb_dir), 'depth')
        depth_files = sorted(glob.glob(os.path.join(depth_dir, '*.png')), key=lambda x: int(os.path.basename(x).split('_')[0])) if os.path.isdir(depth_dir) else []
        if len(depth_files) > 0:
            depth_array = np.empty((num_steps, 256, 256, 1), dtype=np.float32)
            for i, depth_file in enumerate(depth_files[:num_steps]):
                with Image.open(depth_file) as img:
                    arr = np.asarray(img)
                if np.issubdtype(arr.dtype, np.integer):
                    # RL/采集端保存的是 simulator depth([0,1]) * 1000。
                    arr = arr.astype(np.float32) / 1000.0
                else:
                    arr = arr.astype(np.float32)
                    if arr.max() > 1.0:
                        arr = arr / 1000.0
                if len(arr.shape) == 3:
                    arr = arr[:, :, 0]
                depth_array[i, :, :, 0] = np.clip(arr, 0.0, 1.0)
        else:
            depth_array = np.zeros((num_steps, 256, 256, 1), dtype=np.float32)

        start_position = np.asarray(episode_data.get('start_position', [0.0, 0.0, 0.0]), dtype=np.float64)
        start_rotation = np.asarray(episode_data.get('start_rotation', [0.0, 0.0, 0.0, 1.0]), dtype=np.float64)

        # ── 获取起始四元数（与模拟器完全一致的格式）──
        # episode JSON 中 start_rotation 是 [x, y, z, w]（与 habitat-sim 一致）
        # quaternion_from_coeff([x, y, z, w]) → np.quaternion(w, x, y, z)
        if start_rotation.shape[0] == 4:
            try:
                import quaternion
                # 与 habitat-lab quaternion_from_coeff 一致
                start_quat = quaternion.quaternion(
                    float(start_rotation[3]),  # w
                    float(start_rotation[0]),  # x
                    float(start_rotation[1]),  # y
                    float(start_rotation[2]),  # z
                )
                HAS_QUATERNION = True
            except ImportError:
                HAS_QUATERNION = False
                start_quat = None
        else:
            HAS_QUATERNION = False
            start_quat = None

        info = episode_data.get('info') or {}

        # ── 动态轨迹展开：使用四元数精确模拟 agent 运动（与 habitat-sim 完全一致）──
        turn_angle_rad = np.deg2rad(self.turn_angle_deg)

        pos_x = np.zeros(num_steps, dtype=np.float32)
        pos_y = np.zeros(num_steps, dtype=np.float32)
        pos_z = np.zeros(num_steps, dtype=np.float32)
        headings = np.zeros(num_steps, dtype=np.float32)

        if HAS_QUATERNION:
            # ── 四元数模式：与模拟器数学完全一致 ──
            import quaternion as quat_lib

            cur_q = quat_lib.quaternion(start_quat)  # 当前 agent 四元数 (world→agent)
            cur_pos = np.array([
                float(start_position[0]),
                float(start_position[1]),
                float(start_position[2]),
            ], dtype=np.float64)

            # 定义转弯四元数 (绕世界 y 轴旋转)
            # TURN_LEFT: 正角度 = 绕 +y 旋转
            turn_left_delta = quat_lib.from_rotation_vector(
                np.array([0.0, turn_angle_rad, 0.0])
            )
            # TURN_RIGHT: 负角度 = 绕 -y 旋转
            turn_right_delta = quat_lib.from_rotation_vector(
                np.array([0.0, -turn_angle_rad, 0.0])
            )

            for i in range(num_steps):
                if i > 0:
                    prev = int(actions[i - 1])
                    if prev == 1:  # MOVE_FORWARD
                        # Agent-local forward = (0, 0, -step_size)
                        # quat.inverse() = agent→world 旋转
                        world_delta = quaternion_rotate_vector(
                            cur_q.inverse(),
                            np.array([0.0, 0.0, -self.forward_step_size]),
                        )
                        cur_pos += world_delta
                    elif prev == 2:  # TURN_LEFT
                        # 世界系转弯：更新 agent 四元数
                        cur_q = turn_left_delta * cur_q
                    elif prev == 3:  # TURN_RIGHT
                        cur_q = turn_right_delta * cur_q
                    elif prev == 4:  # PAUSE (6-action space)
                        pass
                    elif prev == 5:  # MOVE_BACKWARD
                        world_delta = quaternion_rotate_vector(
                            cur_q.inverse(),
                            np.array([0.0, 0.0, self.forward_step_size]),
                        )
                        cur_pos += world_delta

                pos_x[i] = float(cur_pos[0])
                pos_y[i] = float(cur_pos[1])
                pos_z[i] = float(cur_pos[2])

                # 提取 heading：与 HeadingSensor 完全一致的公式
                # heading_sensor = _quat_to_xy_heading(cur_q.inverse())
                # = arctan2(world_forward_x, -world_forward_z)
                world_forward = quaternion_rotate_vector(
                    cur_q.inverse(),
                    np.array([0.0, 0.0, -1.0]),
                )
                headings[i] = float(np.arctan2(world_forward[0], -world_forward[2]))
        else:
            # ── Fallback：无四元数库时的近似计算（对大多数起始姿态正确）──
            # 注意：当起始 heading 接近 ±π/2 或 ±π 时，此近似可能有偏差
            cur_x = float(start_position[0])
            cur_y = float(start_position[1])
            cur_z = float(start_position[2])
            # 使用 start_rotation[1] 作为近似 heading（非万向节锁情况）
            cur_heading = float(start_rotation[1]) if start_rotation.shape[0] > 1 else 0.0
            for i in range(num_steps):
                if i > 0:
                    prev = int(actions[i - 1])
                    if prev == 1:    # MOVE_FORWARD
                        cur_x += self.forward_step_size * np.sin(cur_heading)
                        cur_z -= self.forward_step_size * np.cos(cur_heading)
                    elif prev == 2:  # TURN_LEFT
                        cur_heading += turn_angle_rad
                    elif prev == 3:  # TURN_RIGHT
                        cur_heading -= turn_angle_rad
                    elif prev == 5:  # MOVE_BACKWARD
                        cur_x -= self.forward_step_size * np.sin(cur_heading)
                        cur_z += self.forward_step_size * np.cos(cur_heading)
                    cur_heading = (cur_heading + np.pi) % (2 * np.pi) - np.pi
                pos_x[i] = cur_x
                pos_y[i] = cur_y
                pos_z[i] = cur_z
                headings[i] = cur_heading

        # localization_sensor: [x, y, z, heading] per step，与 RL LocalizationSensor 一致
        localization = np.stack([pos_x, pos_y, pos_z, headings], axis=-1).astype(np.float32)

        # starting_point_gps_compass: [distance_to_start, -bearing_to_start] per step
        # 与 FalconStartingPointGpsCompassSensor 一致：当前位置指向 episode.start_position。
        # 注意：必须使用 quaternion_rotate_vector(source_rotation.inverse(), direction)
        #       而非简化的三角函数近似
        gx = float(start_position[0])
        gz = float(start_position[2])
        if HAS_QUATERNION:
            # 精确四元数计算（与模拟器完全一致）
            gps_compass = np.zeros((num_steps, 2), dtype=np.float32)
            for i in range(num_steps):
                # 重建此步的 agent 四元数
                if i == 0:
                    step_q = quat_lib.quaternion(start_quat)
                else:
                    step_q = quat_lib.quaternion(start_quat)
                    for j in range(i):
                        prev = int(actions[j])
                        if prev == 2:
                            step_q = turn_left_delta * step_q
                        elif prev == 3:
                            step_q = turn_right_delta * step_q

                direction_world = np.array(
                    [gx - pos_x[i], 0.0, gz - pos_z[i]],
                    dtype=np.float64,
                )
                direction_agent = quaternion_rotate_vector(
                    step_q.inverse(), direction_world
                )
                rho, phi = cartesian_to_polar(
                    -direction_agent[2], direction_agent[0]
                )
                gps_compass[i] = np.array([rho, -phi], dtype=np.float32)
        else:
            # Fallback 三角函数近似
            dx = gx - pos_x
            dz = gz - pos_z
            c = np.cos(headings)
            s = np.sin(headings)
            x_local = dx * c + dz * s
            z_local = -dx * s + dz * c
            rho = np.sqrt(dx * dx + dz * dz)
            phi = np.arctan2(x_local, -z_local)
            gps_compass = np.stack([rho, -phi], axis=-1).astype(np.float32)

        human_num, waypoint_num, human_trajectory = self._parse_info_human_waypoints(info, max_humans=6)
        # DEBUG: 打印human数据解析结果（仅前5个episode）
        _dbg_counter = getattr(self, '_human_debug_count', 0)
        if _dbg_counter < 5:
            self._human_debug_count = _dbg_counter + 1
            # print(f"[DirectFileDataset HUMAN DEBUG] episode_path={Path(episode_path).name}")
            # print(f"[DirectFileDataset HUMAN DEBUG]   info_keys={sorted(info.keys()) if info else 'EMPTY'}")
            # print(f"[DirectFileDataset HUMAN DEBUG]   human_num={human_num}, waypoint_num={waypoint_num}")
            # print(f"[DirectFileDataset HUMAN DEBUG]   trajectory_shape={human_trajectory.shape}, non_zero={(human_trajectory != 0).any()}")
            # if human_num > 0:
                # print(f"[DirectFileDataset HUMAN DEBUG]   trajectory[0,0]={human_trajectory[0, 0]}")
                # print(f"[DirectFileDataset HUMAN DEBUG]   trajectory[0,1]={human_trajectory[0, 1]}")
        max_waypoints = self.future_step
        # 使用 -100 哨兵值填充缺失的human/waypoint槽位，与DPed保持一致
        # -100 表示"未记录到轨迹数据"，避免和真实坐标 (0,0,0,0) 混淆
        if waypoint_num == 0:
            human_trajectory = np.full((6, max_waypoints, 4), -100.0, dtype=np.float32)
        elif human_trajectory.shape[1] < max_waypoints:
            padded_trajectory = np.full((6, max_waypoints, 4), -100.0, dtype=np.float32)
            padded_trajectory[:, : human_trajectory.shape[1], :] = human_trajectory
            human_trajectory = padded_trajectory
        else:
            human_trajectory = human_trajectory[:, :max_waypoints, :]
        # 将超出 human_num 的槽位也设为 -100（DPed只有有效行人数量对应的数据）
        for h in range(human_num, 6):
            human_trajectory[h, :, :] = -100.0

        humanoid_future_trajectory = np.full(
            (num_steps, 6, self.future_step, 2),
            -100.0,
            dtype=np.float32,
        )
        valid_human_positions = (human_trajectory[:, :, :3] != -100.0).all(axis=-1)
        for t in range(num_steps):
            robot_xz = np.array([pos_x[t], pos_z[t]], dtype=np.float32)
            for h in range(6):
                valid_steps = valid_human_positions[h, : self.future_step]
                if not valid_steps.any():
                    continue
                abs_xz = human_trajectory[h, : self.future_step, :][:, [0, 2]].astype(np.float32)
                humanoid_future_trajectory[t, h, valid_steps, :] = (
                    abs_xz[valid_steps] - robot_xz
                )

        observations: Dict[str, np.ndarray] = {}
        # 使用与RL/DPed配置一致的key名称（无agent_0_前缀），匹配MultiAgentAccessMgr行为
        observations['overhead_front_rgb'] = rgb_array
        observations['overhead_front_depth'] = depth_array
        observations['falcon_instruction'] = np.tile(instruction_array, (num_steps, 1))
        observations['starting_point_gps_compass'] = gps_compass
        observations['localization_sensor'] = localization
        observations['human_num_sensor'] = np.full((num_steps, 1), human_num, dtype=np.float32)
        observations['oracle_humanoid_future_trajectory'] = humanoid_future_trajectory
        observations['falcon_gt_action'] = actions.astype(np.int64)

        prev_actions = np.zeros(num_steps, dtype=np.int64)
        prev_actions[1:] = actions[:-1]
        oracle_actions = actions.copy()
        return observations, prev_actions, oracle_actions

    
    def __len__(self):
        return self.length
    
    def __iter__(self):
        # 分布式分片
        if self._dist_world_size > 1:
            per_rank = int(np.ceil(self.length / self._dist_world_size))
            dist_start = per_rank * self._dist_rank
            dist_end = min(dist_start + per_rank, self.length)
        else:
            dist_start = 0
            dist_end = self.length
        
        # Worker分片
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            start = dist_start
            end = dist_end
        else:
            dist_length = dist_end - dist_start
            per_worker = int(np.ceil(dist_length / worker_info.num_workers))
            start = dist_start + per_worker * worker_info.id
            end = min(start + per_worker, dist_end)
        
        # 随机shuffle
        import time
        random.seed(int(time.time() * 1000) % (2**32) + self._dist_rank)
        indices = list(range(start, end))
        random.shuffle(indices)
        
        for idx in indices:
            if self.use_mixed_instructions:
                # 混合模式：使用episode_entries
                episode_path, instruction_type = self.episode_entries[idx]
            else:
                # 传统模式：使用episode_paths
                episode_path = self.episode_paths[idx]
                instruction_type = None
            
            try:
                # 提取episode_id用于调试
                episode_ref_path = self._episode_ref_path(episode_path)
                episode_id = os.path.basename(episode_ref_path.rstrip('/'))
                try:
                    debug_episode_id, debug_scene_id, debug_file_name = self._get_episode_debug_info(episode_path)
                    # if self._debug_count < 10:
                        # print(f"[DirectFileDataset DEBUG] Sample {self._debug_count}: file_name={debug_file_name}, episode_id={debug_episode_id}, scene_id={debug_scene_id}, instruction_type={instruction_type}")
                    episode_id = debug_episode_id
                except Exception as debug_e:
                    # if self._debug_count < 10:
                        # print(f"[DirectFileDataset DEBUG] Sample {self._debug_count}: failed to read debug info from json, fallback basename={episode_id}, err={debug_e}")
                    pass
                obs, prev_actions, oracle_actions = self._load_episode(episode_path, instruction_type)

                # ── 滑动窗口拆分 ──
                # window_size=0 → 整条 episode 作为一个 RNN 序列
                # window_size>0 → 拆成固定长度窗口，每个窗口独立训练
                num_steps_full = len(oracle_actions)
                if self.window_size > 0 and num_steps_full > self.window_size:
                    windows = []
                    for w_start in range(0, num_steps_full, max(1, self.window_size // 2)):
                        w_end = min(w_start + self.window_size, num_steps_full)
                        w_len = w_end - w_start
                        if w_len < max(2, self.window_size // 4):
                            continue
                        w_obs = {}
                        for k, v in obs.items():
                            if k in ('falcon_instruction',):
                                # 指令对每个窗口保持不变（完整episode指令）
                                w_obs[k] = np.tile(v[0:1], (w_len, 1))
                            elif k in ('human_num_sensor',):
                                w_obs[k] = v[w_start:w_end]
                            else:
                                w_obs[k] = v[w_start:w_end]
                        w_prev = np.zeros(w_len, dtype=np.int64)
                        w_prev[0] = 0 if w_start == 0 else int(oracle_actions[w_start - 1])
                        w_prev[1:] = oracle_actions[w_start:w_end - 1]
                        w_acts = oracle_actions[w_start:w_end].copy()
                        windows.append((w_obs, w_prev, w_acts))
                else:
                    windows = [(obs, prev_actions, oracle_actions)]

                for w_obs, w_prev, w_acts in windows:
                    yield from self._process_single_sample(w_obs, w_prev, w_acts, episode_id, instruction_type)

            except Exception as e:
                # logger.warning(f"Failed to load episode from {episode_path}: {e}")
                continue

    def _process_single_sample(self, obs, prev_actions, oracle_actions, episode_id, instruction_type):
        """处理单个样本（完整episode或窗口），转换为训练格式并yield"""
        instruction = obs.get('falcon_instruction', None)

        # ── 指令有效性检查（仅在 skip_episodes_with_empty_instruction=True 时跳过）──
        if self.skip_episodes_with_empty_instruction:
            if instruction is not None and len(instruction) > 0:
                first_instr = instruction[0]
                non_zero_mask = first_instr != 0
                if non_zero_mask.sum() == 0:
                    self._skipped_empty_instruction += 1
                    return  # 跳过空指令
                try:
                    instr_bytes = bytes(first_instr[first_instr != 0])
                    instr_text = instr_bytes.decode('utf-8', errors='ignore').strip()
                    default_instructions = [
                        'navigate to the target location.',
                        'navigate to target location',
                        'go to target',
                    ]
                    if instr_text.lower() in [d.lower() for d in default_instructions]:
                        self._skipped_default_instruction += 1
                        return  # 跳过默认指令
                except Exception:
                    pass

        # 转换为tensor（避免额外的 np.copy 峰值内存）
        for k, v in obs.items():
            obs[k] = torch.from_numpy(v)
        prev_actions = torch.from_numpy(prev_actions).long()
        oracle_actions = torch.from_numpy(oracle_actions).long()

        # 计算inflection weights
        inflections = torch.cat([
            torch.tensor([1], dtype=torch.long),
            (oracle_actions[1:] != oracle_actions[:-1]).long(),
        ])
        weights = self.inflec_weights[inflections]

        # 处理无效动作（-1）
        valid_mask = oracle_actions != -1
        num_actions = 4  # 默认值
        valid_range_mask = (oracle_actions >= 0) & (oracle_actions < num_actions)
        valid_mask = valid_mask & valid_range_mask
        weights = weights * valid_mask.float()

        # 替换无效动作为0
        oracle_actions = torch.where(
            valid_mask,
            oracle_actions,
            torch.zeros_like(oracle_actions)
        )

        yield obs, prev_actions, oracle_actions, weights


def collate_fn(batch):
    """与dagger_trainer相同的collate函数"""
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

    if B == 0:
        raise ValueError("Empty batch passed to collate_fn")

    # 过滤掉全零权重样本，避免在训练步骤里白跑并触发后续异常
    keep_indices = [i for i, w in enumerate(weights_batch) if w.sum().item() > 0]
    if len(keep_indices) == 0:
        raise ValueError("All samples in batch have zero weights")
    if len(keep_indices) != B:
        observations_batch = [observations_batch[i] for i in keep_indices]
        prev_actions_batch = [prev_actions_batch[i] for i in keep_indices]
        corrected_actions_batch = [corrected_actions_batch[i] for i in keep_indices]
        weights_batch = [weights_batch[i] for i in keep_indices]
        B = len(keep_indices)

    # 先按最长轨迹长度 pad，再统一 stack，避免重复构建中间大对象
    max_traj_len = max(ele.size(0) for ele in prev_actions_batch)

    new_observations_batch = defaultdict(list)
    for sensor in observations_batch[0]:
        for bid in range(B):
            new_observations_batch[sensor].append(observations_batch[bid][sensor])

    for bid in range(B):
        for sensor in new_observations_batch:
            fill_val = 0.0
            if 'instruction' in sensor.lower():
                fill_val = 0.0
            elif 'rgb' in sensor.lower() or 'depth' in sensor.lower():
                fill_val = 0.0
            new_observations_batch[sensor][bid] = _pad_helper(
                new_observations_batch[sensor][bid], max_traj_len, fill_val=fill_val
            )
        prev_actions_batch[bid] = _pad_helper(prev_actions_batch[bid], max_traj_len, fill_val=-1)
        corrected_actions_batch[bid] = _pad_helper(corrected_actions_batch[bid], max_traj_len, fill_val=-1)
        weights_batch[bid] = _pad_helper(weights_batch[bid], max_traj_len, fill_val=0.0)

    observations_batch = {}
    for sensor, tensors in new_observations_batch.items():
        stacked = torch.stack(tensors, dim=0)  # [B, max_traj_len, ...] 每个episode连续
        observations_batch[sensor] = stacked.reshape(-1, *stacked.size()[2:])

    prev_actions_batch = torch.stack(prev_actions_batch, dim=0)  # [B, max_traj_len]
    corrected_actions_batch = torch.stack(corrected_actions_batch, dim=0)  # [B, max_traj_len]
    weights_batch = torch.stack(weights_batch, dim=0)
    not_done_masks = torch.ones_like(corrected_actions_batch, dtype=torch.uint8)
    not_done_masks[:, 0] = 0  # 每个episode的第一步mask=0（RNN状态重置）
    # 填充步骤也设 mask=0，防止 RNN 学习填充数据
    not_done_masks[corrected_actions_batch < 0] = 0

    return (
        observations_batch,
        prev_actions_batch.reshape(-1, 1),
        not_done_masks.reshape(-1, 1),
        corrected_actions_batch,
        weights_batch,
    )


@baseline_registry.register_trainer(name="direct_il")
class DirectILTrainer(BaseILTrainer):
    """直接从文件系统读取数据进行模仿学习的Trainer，不经过模拟器"""
    
    def __init__(self, config=None):
        # 分布式训练相关属性（简化版，参考dagger_trainer）
        self._is_distributed = False
        self._dist_rank = 0
        self._dist_world_size = 1
        self._dist_local_rank = 0
        self.config = config
        
        # 检查是否启用分布式训练
        dist_cfg = getattr(config.habitat_baselines.il, "distributed", None)
        if dist_cfg is not None and getattr(dist_cfg, "enabled", False):
            self._init_distributed(dist_cfg)
        
        # 分布式模式下设置GPU设备
        if self._is_distributed:
            from habitat.config import read_write
            with read_write(self.config):
                self.config.habitat_baselines.torch_gpu_id = self._dist_local_rank
        
        super().__init__(config)
        
        # 添加日志文件处理器
        if hasattr(config.habitat_baselines, 'log_file') and config.habitat_baselines.log_file:
            try:
                log_dir = os.path.dirname(config.habitat_baselines.log_file)
                if log_dir and not os.path.exists(log_dir):
                    os.makedirs(log_dir, exist_ok=True)
                logger.add_filehandler(config.habitat_baselines.log_file)
            except Exception as e:
                # logger.warning(f"Failed to add log file handler: {e}")
                pass
        
        # 设置设备
        if self._is_distributed:
            self.device = torch.device("cuda", self._dist_local_rank)
            torch.cuda.set_device(self._dist_local_rank)
        else:
            self.device = (
                torch.device("cuda", self.config.habitat_baselines.torch_gpu_id)
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        
        # Checkpoint相关初始化
        self._last_checkpoint_percent = -1.0
        self._current_epoch = 0
        self._total_updates = 0
        epochs = config.habitat_baselines.il.epochs
        self._total_epochs = epochs

        # Eval infrastructure (lazy init)
        self._eval_envs = None
        self._eval_agent = None
        self._eval_env_spec = None
        self._eval_rank0_keys = None
        self._eval_obs_transforms = None

    def _init_envs(self, config=None, is_eval: bool = False):
        """初始化向量化环境（用于eval）"""
        import hydra
        from habitat_baselines.common.env_factory import VectorEnvFactory
        from habitat_baselines.common.env_spec import EnvironmentSpec

        if config is None:
            config = self.config

        env_factory: VectorEnvFactory = hydra.utils.instantiate(
            config.habitat_baselines.vector_env_factory
        )
        self._eval_envs = env_factory.construct_envs(
            config,
            workers_ignore_signals=False,
            enforce_scenes_greater_eq_environments=is_eval,
            is_first_rank=True,
        )
        self._eval_env_spec = EnvironmentSpec(
            observation_space=self._eval_envs.observation_spaces[0],
            action_space=self._eval_envs.action_spaces[0],
            orig_action_space=self._eval_envs.orig_action_spaces[0],
        )
        self._eval_rank0_keys = set(
            list(config.habitat.task.rank0_env0_measure_names)
            + list(config.habitat.task.rank0_measure_names)
        )

    def _create_agent(self):
        """创建适配的 agent wrapper（用于eval）。

        NaVidEvaluator 需要 agent.actor_critic 提供 action space 信息以及
        act/update_hidden_state 接口。对于 DirectIL zero-shot eval，我们
        创建一个最小化的 adapter 满足这些接口。
        """
        import types
        import numpy as np
        from gym import spaces

        # Build obs transforms (IL config may not have rl.policy)
        try:
            from habitat_baselines.common.obs_transformers import (
                get_active_obs_transforms,
            )
            self._eval_obs_transforms = get_active_obs_transforms(
                self.config, self._eval_env_spec
            )
        except Exception:
            self._eval_obs_transforms = []

        # 从环境获取 action space
        env_action_space = self._eval_env_spec.action_space
        # 解析 multi-agent action space
        agent0_space = None
        if hasattr(env_action_space, 'spaces') and 'agent_0' in env_action_space.spaces:
            agent0_space = env_action_space.spaces['agent_0']
        elif isinstance(env_action_space, spaces.Dict):
            for k, v in env_action_space.spaces.items():
                if 'agent_0' in k:
                    agent0_space = v
                    break
        if agent0_space is None:
            agent0_space = env_action_space

        # 确定 action 维度
        if isinstance(agent0_space, spaces.Discrete):
            action_dim = 1
        elif isinstance(agent0_space, spaces.Box):
            action_dim = int(np.prod(agent0_space.shape))
        else:
            action_dim = 1

        # 创建最小化 ActorCritic adapter
        class EvalActorCritic:
            def __init__(self, action_space, obs_space, device):
                self._action_space = action_space
                self._obs_space = obs_space
                self._device = device
                self.net = None  # model reuse not available for direct IL eval

            @property
            def policy_action_space(self):
                return self._action_space

            @property
            def policy_action_space_shape_lens(self):
                if isinstance(self._action_space, spaces.Dict):
                    return [int(np.prod(s.shape)) if isinstance(s, spaces.Box) else 1
                            for s in self._action_space.spaces.values()]
                return [action_dim]

            @property
            def hidden_state_shape_lens(self):
                return [1]  # minimal: no RNN hidden state needed

            @property
            def hidden_state_shape(self):
                return (1,)

            def act(self, batch, recurrent_hidden_states, prev_actions,
                    masks, deterministic=False, **kwargs):
                # 返回 dummy action data，只需 env_actions 不为空
                batch_size = 1
                try:
                    if isinstance(batch, dict):
                        for v in batch.values():
                            if hasattr(v, 'shape') and len(v.shape) >= 1:
                                batch_size = v.shape[0]
                                break
                except Exception:
                    pass
                dummy_env_actions = torch.zeros(
                    batch_size, action_dim, device=self._device
                )
                if isinstance(self._action_space, spaces.Discrete):
                    dummy_env_actions = dummy_env_actions.long()

                # Minimal return type
                class ActionData:
                    def __init__(self, env_actions, rnn_hidden_states):
                        self.env_actions = env_actions
                        self.actions = env_actions
                        self.rnn_hidden_states = rnn_hidden_states
                        self.should_inserts = None

                return ActionData(dummy_env_actions,
                                  torch.zeros(batch_size, 1, device=self._device))

            def update_hidden_state(self, recurrent_hidden_states, prev_actions, action_data):
                for i in range(len(recurrent_hidden_states)):
                    recurrent_hidden_states[i].copy_(action_data.rnn_hidden_states[i])
                for i in range(len(prev_actions)):
                    prev_actions[i].copy_(action_data.actions[i])

            def get_policy_info(self, batch, rnn_hidden_states, prev_actions, masks):
                return {}

        # 创建 EvalAgent adapter
        class EvalAgent:
            def __init__(self, ac, masks_shape):
                self.actor_critic = ac
                self._masks_shape = masks_shape
                self._agents = [self]

            @property
            def masks_shape(self):
                return self._masks_shape

        actor_critic = EvalActorCritic(agent0_space, self._eval_env_spec.observation_space, self.device)
        return EvalAgent(actor_critic, (1,))

    def _eval_checkpoint(
        self,
        checkpoint_path: str,
        writer: TensorboardWriter,
        checkpoint_index: int = 0,
    ) -> None:
        """评估单个 checkpoint（支持 zero-shot 和 fine-tuned）"""
        import hydra

        config = self.config
        should_load = config.habitat_baselines.eval.should_load_ckpt

        # 1. Load checkpoint (if available)
        if should_load and os.path.isfile(checkpoint_path):
            ckpt_dict = self.load_checkpoint(checkpoint_path, map_location="cpu")
            step_id = ckpt_dict.get("extra_state", {}).get("step", checkpoint_index)
            logger.info(f"Loaded checkpoint from {checkpoint_path}, step={step_id}")
        else:
            ckpt_dict = {"config": None}
            step_id = checkpoint_index
            logger.info(f"Zero-shot eval mode (no training checkpoint)")

        # 2. Init environments
        self._init_envs(config, is_eval=True)
        logger.info(f"Created {self._eval_envs.num_envs} eval environments")

        # 3. Create agent
        self._eval_agent = self._create_agent()
        logger.info("Agent created for eval")

        # 4. Adapt config: evaluator expects rl.policy, IL config uses il.policy
        # Rebuild config from YAML to bypass structured schema constraints
        from omegaconf import OmegaConf
        config_yaml = OmegaConf.to_yaml(config)
        config = OmegaConf.create(config_yaml)
        OmegaConf.set_struct(config, False)
        # Copy il.policy → rl.policy for evaluator compatibility
        il_policy = OmegaConf.create(
            OmegaConf.to_yaml(config.habitat_baselines.il.policy)
        )
        rl_cfg = OmegaConf.create({"policy": il_policy})
        config.habitat_baselines.rl = rl_cfg

        # 5. Create evaluator and evaluate
        evaluator = hydra.utils.instantiate(config.habitat_baselines.evaluator)
        logger.info(f"Evaluator: {type(evaluator).__name__}")

        evaluator.evaluate_agent(
            self._eval_agent,
            self._eval_envs,
            config,
            checkpoint_index,
            step_id,
            writer,
            self.device,
            self._eval_obs_transforms,
            self._eval_env_spec,
            self._eval_rank0_keys,
        )

        # 5. Cleanup
        self._eval_envs.close()
        logger.info("Eval complete, environments closed")

    def eval(self) -> None:
        """Override eval() to bypass checkpoint polling loop.
        DirectIL supports both zero-shot eval and checkpoint-based eval.
        """
        import time as _t0_time

        self._add_preemption_signal_handlers()

        resume_state = load_resume_state(self.config, filename_key="eval")
        if resume_state is not None:
            self.config = self._get_resume_state_config_or_new_config(
                resume_state["config"]
            )
            prev_ckpt_ind = resume_state["prev_ckpt_ind"]
        else:
            prev_ckpt_ind = -1

        self.device = (
            torch.device("cuda", self.config.habitat_baselines.torch_gpu_id)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

        with get_writer(self.config, flush_secs=self.flush_secs) as writer:
            ckpt_path = self.config.habitat_baselines.eval_ckpt_path_dir
            should_load = self.config.habitat_baselines.eval.should_load_ckpt

            if os.path.isfile(ckpt_path):
                # Single checkpoint file
                ckpt_idx = get_checkpoint_id(ckpt_path) or 0
                logger.info(f"Evaluating single checkpoint: {ckpt_path}")
                self._eval_checkpoint(ckpt_path, writer, checkpoint_index=ckpt_idx)
            elif should_load and os.path.isdir(ckpt_path):
                # Directory mode: evaluate all checkpoints in order
                logger.info(f"Polling checkpoint folder: {ckpt_path}")
                ckpt_files = sorted(
                    [f for f in os.listdir(ckpt_path)
                     if os.path.isfile(os.path.join(ckpt_path, f))
                     and "latest" not in f
                     and (f.endswith(".pth") or f.endswith(".ckpt") or f.endswith(".pt"))],
                    key=lambda f: os.path.getmtime(os.path.join(ckpt_path, f)),
                )
                if not ckpt_files:
                    logger.warning(
                        f"No checkpoint files found in {ckpt_path}, "
                        f"running in zero-shot mode"
                    )
                    self._eval_checkpoint(ckpt_path, writer, checkpoint_index=0)
                else:
                    for i, ckpt_file in enumerate(ckpt_files):
                        if i <= prev_ckpt_ind:
                            continue
                        full_path = os.path.join(ckpt_path, ckpt_file)
                        logger.info(f"Evaluating {full_path}")
                        self._eval_checkpoint(full_path, writer, checkpoint_index=i)
                        save_resume_state(
                            {"config": self.config, "prev_ckpt_ind": i},
                            self.config,
                            filename_key="eval",
                        )
            else:
                # Zero-shot: no checkpoint needed
                logger.info("Zero-shot evaluation (should_load_ckpt=False)")
                self._eval_checkpoint(ckpt_path, writer, checkpoint_index=0)

    def _init_distributed(self, dist_cfg):
        """初始化分布式训练环境（参考dagger_trainer）"""
        import torch.distributed as dist
        import os
        
        backend = getattr(dist_cfg, "backend", "nccl")
        init_method = getattr(dist_cfg, "init_method", "env://")
        
        rank = int(os.environ.get("RANK", -1)) if getattr(dist_cfg, "rank", -1) < 0 else dist_cfg.rank
        world_size = int(os.environ.get("WORLD_SIZE", 1)) if getattr(dist_cfg, "world_size", -1) <= 0 else dist_cfg.world_size
        local_rank = int(os.environ.get("LOCAL_RANK", 0)) if getattr(dist_cfg, "local_rank", -1) < 0 else dist_cfg.local_rank
        
        if rank < 0 or world_size <= 1:
            # logger.warning("Distributed training requested but rank/world_size not properly set. Disabling distributed training.")
            return
        
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
        
        if dist.is_initialized():
            try:
                self._dist_local_rank = dist.get_local_rank()
            except (AttributeError, RuntimeError):
                self._dist_local_rank = local_rank
        else:
            self._dist_local_rank = local_rank
        
        # logger.info(f"Initialized distributed training: rank={self._dist_rank}, world_size={self._dist_world_size}, local_rank={self._dist_local_rank}")
    
    def _wrap_model_for_distributed(self):
        """将模型包装为DistributedDataParallel"""
        import torch.distributed as dist
        from torch.nn.parallel import DistributedDataParallel as DDP
        
        dist_cfg = getattr(self.config.habitat_baselines.il, "distributed", None)
        find_unused_params = getattr(dist_cfg, "find_unused_parameters", False)
        broadcast_buffers = getattr(dist_cfg, "broadcast_buffers", True)
        
        self.policy = DDP(
            self.policy,
            device_ids=[self._dist_local_rank],
            output_device=self._dist_local_rank,
            find_unused_parameters=find_unused_params,
            broadcast_buffers=broadcast_buffers,
        )
        
        # logger.info(f"Wrapped policy model with DistributedDataParallel")
    
    def _is_rank0(self) -> bool:
        if not self._is_distributed:
            return True
        import torch.distributed as dist
        return dist.get_rank() == 0
    
    def _get_policy(self):
        if self._is_distributed:
            from torch.nn.parallel import DistributedDataParallel as DDP
            if isinstance(self.policy, DDP):
                return self.policy.module
        return self.policy
    
    def _get_model_state_dict(self):
        return self._get_policy().state_dict()
    
    def _make_dirs(self) -> None:
        if self._is_rank0():
            self._make_ckpt_dir()
            if self.config.habitat_baselines.il.eval_save_results:
                self._make_results_dir()
    
    def percent_done(self) -> float:
        if self._total_epochs == 0:
            return 0.0
        return self._current_epoch / self._total_epochs
    
    def should_checkpoint(self) -> bool:
        # Save every 500 updates (first save at update 500)
        if self._total_updates > 0 and self._total_updates % 500 == 0:
            return True
        # Also save at epoch boundaries
        if self._last_checkpoint_percent < self.percent_done():
            self._last_checkpoint_percent = self.percent_done()
            return True
        return False
    
    def save_checkpoint(
        self, 
        epoch: int,
        step_id: int,
        model_state_dict: OrderedDict,
        optimizer_state_dict: Dict,
        best_loss: float,
        file_name: str = None
    ) -> None:
        """保存完整的训练状态，支持断点续训
        
        Args:
            epoch: 当前epoch编号
            step_id: 当前step编号
            model_state_dict: 模型权重
            optimizer_state_dict: 优化器状态
            best_loss: 当前最佳loss
            file_name: checkpoint文件名（可选）
        """
        if file_name is None:
            file_name = f"ckpt.epoch_{epoch+1}.step_{step_id}.pth"
        
        checkpoint = {
            "state_dict": model_state_dict,
            "optimizer_state_dict": optimizer_state_dict,
            "epoch": epoch,
            "step_id": step_id,
            "best_loss": best_loss,
            "config": dict(self.config),  # 保存配置以便验证
        }
        
        checkpoint_dir = self.config.habitat_baselines.checkpoint_folder
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, file_name)
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"[CKPT] Saved {file_name} ({os.path.getsize(checkpoint_path)/1e6:.1f}MB)")
        
        # 同时保存为latest.pth，方便恢复
        latest_path = os.path.join(
            self.config.habitat_baselines.checkpoint_folder,
            "latest.pth"
        )
        torch.save(checkpoint, latest_path)
        # logger.info(f"✓ Updated latest checkpoint: {latest_path}")
    
    def load_checkpoint_for_resume(self) -> Dict:
        """加载checkpoint用于断点续训
        
        Returns:
            Dict包含恢复的训练状态，如果没有找到checkpoint则返回None
        """
        checkpoint_folder = self.config.habitat_baselines.checkpoint_folder
        
        # 按优先级查找checkpoint
        checkpoint_candidates = [
            os.path.join(checkpoint_folder, "latest.pth"),  # 最优先：latest
        ]
        
        # 查找所有ckpt.epoch_*.pth文件，按epoch编号排序
        import glob
        epoch_ckpts = glob.glob(os.path.join(checkpoint_folder, "ckpt.epoch_*.pth"))
        if epoch_ckpts:
            # 提取epoch编号并排序
            def extract_epoch(path):
                import re
                match = re.search(r'epoch_(\d+)', path)
                return int(match.group(1)) if match else -1
            epoch_ckpts.sort(key=extract_epoch, reverse=True)
            checkpoint_candidates.extend(epoch_ckpts[:3])  # 取最新的3个
        
        for ckpt_path in checkpoint_candidates:
            if os.path.exists(ckpt_path):
                try:
                    # logger.info(f"Loading checkpoint for resume: {ckpt_path}")
                    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                    
                    # 验证checkpoint结构
                    required_keys = ["state_dict", "optimizer_state_dict", "epoch", "step_id"]
                    if all(k in checkpoint for k in required_keys):
                        # logger.info(f"✓ Found valid checkpoint:")
                        # logger.info(f"  - Epoch: {checkpoint['epoch']}")
                        # logger.info(f"  - Step: {checkpoint['step_id']}")
                        # logger.info(f"  - Best loss: {checkpoint.get('best_loss', 'N/A')}")
                        return checkpoint
                    else:
                        # logger.warning(f"Checkpoint {ckpt_path} missing required keys, trying next...")
                        pass
                except Exception as e:
                    # logger.warning(f"Failed to load checkpoint {ckpt_path}: {e}")
                    continue
        
        # logger.info("No valid checkpoint found for resume, starting from scratch")
        return None
    
    def _get_spaces(self) -> Tuple[Dict, Dict]:
        """获取观察空间和动作空间（简化版，直接使用配置）"""
        # 直接从配置文件推断空间（因为不创建环境）
        # 观察空间：RGB, Depth, Instruction等（使用与DPed RL配置一致的key名称，无agent_0_前缀）
        # 注意：形状必须与_load_episode中实际生成的数据一致
        future_step = 4
        try:
            future_step = int(
                getattr(
                    self.config.habitat_baselines.rl.auxiliary_losses.future_trajectory_prediction,
                    "future_step",
                    future_step,
                )
            )
        except Exception:
            try:
                future_step = int(
                    getattr(
                        self.config.habitat_baselines.il.direct_il,
                        "future_step",
                        future_step,
                    )
                )
            except Exception:
                future_step = 4
        observation_space = spaces.Dict({
            'overhead_front_rgb': spaces.Box(
                low=0, high=255, shape=(256, 256, 3), dtype=np.uint8
            ),
            'overhead_front_depth': spaces.Box(
                low=0, high=1, shape=(256, 256, 1), dtype=np.float32
            ),
            'falcon_instruction': spaces.Box(
                low=0, high=255, shape=(512,), dtype=np.uint8
            ),
            'starting_point_gps_compass': spaces.Box(
                low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32
            ),
            'localization_sensor': spaces.Box(
                low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32
            ),
            'human_num_sensor': spaces.Box(
                low=0, high=np.inf, shape=(1,), dtype=np.float32
            ),
            'oracle_humanoid_future_trajectory': spaces.Box(
                low=-100.0, high=np.inf, shape=(6, future_step, 2), dtype=np.float32
            ),
            'falcon_gt_action': spaces.Box(
                low=0, high=3, shape=(), dtype=np.int64
            ),
        })
        
        # 动作空间：离散4个动作
        action_space = spaces.Discrete(4)
        
        self._agent_0_info = {
            'num_actions': 4,
            'action_space_type': 'discrete',
            'action_space': action_space,
        }
        
        return observation_space, action_space
    
    def _count_parameters(self, model):
        """统计模型参数量"""
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        return trainable_params, total_params
    
    def _print_model_structure(self, model):
        """打印模型结构和每个模块的参数量"""
        # logger.info("\n" + "="*80)
        # logger.info("MODEL STRUCTURE AND PARAMETER BREAKDOWN")
        # logger.info("="*80)
        
        # 获取主policy的net
        if hasattr(model, 'net'):
            net = model.net
        else:
            net = model
        
        # 详细统计每个子模块
        module_stats = {}
        
        for name, module in net.named_children():
            trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            total = sum(p.numel() for p in module.parameters())
            frozen = total - trainable
            module_stats[name] = {
                'trainable': trainable,
                'frozen': frozen,
                'total': total
            }
            
            # logger.info(f"\n[{name}]")
            # logger.info(f"  Trainable params: {trainable:,}")
            # logger.info(f"  Frozen params: {frozen:,}")
            # logger.info(f"  Total params: {total:,}")
            
            # 对于visual_encoder等复杂模块，进一步展开
            if hasattr(module, 'named_children'):
                for sub_name, sub_module in module.named_children():
                    sub_trainable = sum(p.numel() for p in sub_module.parameters() if p.requires_grad)
                    sub_total = sum(p.numel() for p in sub_module.parameters())
                    if sub_total > 0:
                        pass
                        # logger.info(f"    [{sub_name}]: {sub_trainable:,} / {sub_total:,} trainable")
        
        # 总体统计
        total_trainable, total_params = self._count_parameters(model)
        # logger.info("\n" + "-"*80)
        # logger.info(f"TOTAL TRAINABLE PARAMETERS: {total_trainable:,}")
        # logger.info(f"TOTAL FROZEN PARAMETERS: {total_params - total_trainable:,}")
        # logger.info(f"TOTAL PARAMETERS: {total_params:,}")
        # logger.info(f"Trainable ratio: {100.0 * total_trainable / total_params:.2f}%")
        # logger.info("="*80 + "\n")
        
        return module_stats, total_trainable, total_params
    
    def _initialize_policy(self, observation_space, action_space):
        """初始化策略网络（参考dagger_trainer）"""
        il_cfg = self.config.habitat_baselines.il
        model_cfg = il_cfg.model
        policy_cfg = il_cfg.policy.agent_0 if hasattr(il_cfg, "policy") and hasattr(il_cfg.policy, "agent_0") else None
        
        hidden_size = int(model_cfg.hidden_size)
        backbone = getattr(model_cfg, "backbone", "resnet18")
        rnn_type = getattr(model_cfg, "rnn_type", "LSTM")
        num_recurrent_layers = getattr(model_cfg, "num_recurrent_layers", 2)
        
        text_encoder_dim = getattr(model_cfg, "text_encoder_dim", 500)
        fusion_method = getattr(model_cfg, "fusion_method", "concat")
        
        # logger.info(f"\n{'='*80}")
        # logger.info(f"POLICY CONFIGURATION")
        # logger.info(f"{'='*80}")
        # logger.info(f"Hidden size: {hidden_size}")
        # logger.info(f"Backbone: {backbone}")
        # logger.info(f"RNN type: {rnn_type}")
        # logger.info(f"RNN layers: {num_recurrent_layers}")
        # logger.info(f"Text encoder dim: {text_encoder_dim}")
        # logger.info(f"Fusion method: {fusion_method}")
        # logger.info(f"{'='*80}\n")
        
        if policy_cfg is None or not hasattr(policy_cfg, "action_distribution_type"):
            from omegaconf import OmegaConf
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
        
        # logger.info(f"Observation space keys: {list(filtered_obs.spaces.keys())}")

        # 确定策略类型
        policy_name = getattr(policy_cfg, "name", "PointNavResNetPolicy") if policy_cfg is not None else "PointNavResNetPolicy"
        use_lora = getattr(model_cfg, "use_lora", False)

        if policy_name == "NaVILAPolicy":
            self._init_navilla_policy(filtered_obs, action_space, hidden_size, policy_cfg, model_cfg, il_cfg)
        elif policy_name == "StreamVLNPolicy":
            self._init_streamvln_policy(filtered_obs, action_space, hidden_size, policy_cfg, model_cfg, il_cfg)
        elif policy_name == "NaVidPolicy":
            self._init_navid_policy(filtered_obs, action_space, hidden_size, policy_cfg, model_cfg, il_cfg)
        else:
            # 默认使用 PointNavResNetPolicy
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
                aux_loss_config=None,
                fuse_keys=None,
                text_instruction_path=None,
                text_encoder_dim=text_encoder_dim,
                fusion_method=fusion_method,
                clip_visual_sensors=getattr(model_cfg, "clip_visual_sensors", None),
                clip_model_type=getattr(model_cfg, "clip_model_type", "longclip"),
            ).to(self.device)

            # 对 PointNavResNetPolicy 也支持 LoRA（如果模型包含 transformer 层）
            if use_lora and PEFT_AVAILABLE:
                self._apply_lora_to_resnet_policy()

        # 打印模型结构和参数统计
        self.module_stats, self.total_trainable, self.total_params = self._print_model_structure(self.policy)

        # ── 仿照 dynamic_vln_trainer 的 train_encoder 处理方式 ──
        self._is_static_encoder = not getattr(model_cfg, "train_encoder", True)
        if self._is_static_encoder:
            logger.info("[IL Trainer] train_encoder=False: encoder frozen")

        # 优化器
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.policy.parameters()),
            lr=float(il_cfg.optim.lr),
            eps=float(il_cfg.optim.eps),
        )
        self.max_grad_norm = float(il_cfg.optim.max_grad_norm)

        # 学习率调度器
        epochs = self.config.habitat_baselines.il.epochs
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs
        ) if getattr(il_cfg.optim, "use_lr_scheduler", True) else None

        # CrossEntropyLoss
        class_weight_enabled = getattr(il_cfg, "class_weight_enabled", True)
        if class_weight_enabled:
            class_weights_list = getattr(il_cfg, "class_weights", [3.0, 0.6, 1.0, 1.0])
            self.criterion = torch.nn.CrossEntropyLoss(
                reduction="none",
                weight=torch.tensor(class_weights_list, dtype=torch.float32, device=self.device),
            )
        else:
            self.criterion = torch.nn.CrossEntropyLoss(reduction="none")

    def _init_navilla_policy(self, observation_space, action_space, hidden_size, policy_cfg, model_cfg, il_cfg):
        """Initialize NaVILA policy with optional LoRA fine-tuning"""
        from habitat_baselines.rl.ddppo.policy.navila_policy import NaVILAPolicy

        model_path = getattr(model_cfg, "model_path", None)
        if model_path is None and policy_cfg is not None:
            model_path = getattr(policy_cfg, "model_path", None)
        num_video_frames = getattr(policy_cfg, "num_video_frames", 8) if policy_cfg else 8
        forward_step = getattr(policy_cfg, "forward_step", 25) if policy_cfg else 25
        turn_step = getattr(policy_cfg, "turn_step", 15) if policy_cfg else 15

        logger.info(f"[NaVILA-IL] Initializing NaVILAPolicy from {model_path}")

        self.policy = NaVILAPolicy(
            observation_space=observation_space,
            action_space=action_space,
            hidden_size=hidden_size,
            model_path=model_path,
            num_video_frames=num_video_frames,
            forward_step=forward_step,
            turn_step=turn_step,
            policy_config=policy_cfg,
        ).to(self.device)

        use_lora = getattr(model_cfg, "use_lora", False)
        if use_lora and PEFT_AVAILABLE:
            self._apply_lora_to_vlm_policy()

    def _init_navid_policy(self, observation_space, action_space, hidden_size, policy_cfg, model_cfg, il_cfg):
        """Initialize NaVid policy with optional LoRA fine-tuning"""
        from habitat_baselines.rl.ddppo.policy.navid_policy import NaVidPolicy

        model_path = getattr(model_cfg, "model_path", None)
        if model_path is None and policy_cfg is not None:
            model_path = getattr(policy_cfg, "model_path", None)
        num_video_frames = getattr(policy_cfg, "num_video_frames", 4) if policy_cfg else 4
        forward_step = getattr(policy_cfg, "forward_step", 25) if policy_cfg else 25
        turn_step = getattr(policy_cfg, "turn_step", 15) if policy_cfg else 15

        logger.info(f"[NaVid-IL] Initializing NaVidPolicy from {model_path}")

        self.policy = NaVidPolicy(
            observation_space=observation_space,
            action_space=action_space,
            hidden_size=hidden_size,
            model_path=model_path,
            num_video_frames=num_video_frames,
            forward_step=forward_step,
            turn_step=turn_step,
            policy_config=policy_cfg,
        ).to(self.device)

        use_lora = getattr(model_cfg, "use_lora", False)
        if use_lora and PEFT_AVAILABLE:
            self._apply_lora_to_vlm_policy()

    def _init_streamvln_policy(self, observation_space, action_space, hidden_size, policy_cfg, model_cfg, il_cfg):
        """Initialize StreamVLN policy with optional LoRA fine-tuning"""
        from habitat_baselines.rl.ddppo.policy.streamvln_policy import StreamVLNPolicy

        model_path = getattr(model_cfg, "model_path", None)
        if model_path is None and policy_cfg is not None:
            model_path = getattr(policy_cfg, "model_path", None)
        num_frames = getattr(policy_cfg, "num_frames", 32) if policy_cfg else 32
        num_history = getattr(policy_cfg, "num_history", 8) if policy_cfg else 8
        num_future_steps = getattr(policy_cfg, "num_future_steps", 4) if policy_cfg else 4
        model_max_length = getattr(policy_cfg, "model_max_length", 4096) if policy_cfg else 4096
        forward_step = getattr(policy_cfg, "forward_step", 25) if policy_cfg else 25
        turn_step = getattr(policy_cfg, "turn_step", 15) if policy_cfg else 15

        logger.info(f"[StreamVLN-IL] Initializing StreamVLNPolicy from {model_path}")

        self.policy = StreamVLNPolicy(
            observation_space=observation_space,
            action_space=action_space,
            hidden_size=hidden_size,
            model_path=model_path,
            num_frames=num_frames,
            num_history=num_history,
            num_future_steps=num_future_steps,
            model_max_length=model_max_length,
            device=str(self.device),
            forward_step=forward_step,
            turn_step=turn_step,
            policy_config=policy_cfg,
        ).to(self.device)

        use_lora = getattr(model_cfg, "use_lora", False)
        if use_lora and PEFT_AVAILABLE:
            self._apply_lora_to_vlm_policy()

    def _apply_lora_to_vlm_policy(self):
        """Apply LoRA adapter to VLM-based policy (NaVILA/StreamVLN).

        IMPORTANT: get_peft_model modifies the base model IN-PLACE (injecting lora_A/lora_B
        adapters and freezing original weights), and returns a PeftModel wrapper. We MUST
        assign the PeftModel wrapper BACK to the model hierarchy so that the policy's
        parameter enumeration and forward pass both see it.
        """
        if not PEFT_AVAILABLE:
            logger.warning("[LoRA] PEFT not available, skipping")
            return

        model_cfg = self.config.habitat_baselines.il.model
        lora_r = getattr(model_cfg, "lora_r", 16)
        lora_alpha = getattr(model_cfg, "lora_alpha", 32)
        lora_dropout = getattr(model_cfg, "lora_dropout", 0.05)
        target_modules = getattr(model_cfg, "lora_target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj",
             "gate_proj", "up_proj", "down_proj"])

        # Find the VLM's underlying language model (LLM)
        # NaviLLa: policy.net.model (LlavaLlamaModel) → .llm (LlamaForCausalLM)
        # StreamVLN: policy.net.model (StreamVLNForCausalLM) — IS already a causal LM
        vlm_model = None
        parent_obj = None   # the object that holds the attribute pointing to vlm_model
        parent_attr = None  # the attribute name on parent_obj, e.g. 'llm' or 'model'
        if hasattr(self.policy, 'net'):
            net = self.policy.net
            # Step 1: find the multimodal wrapper / causal LM
            wrapper = None
            wrapper_attr = None
            for attr in ['model', 'llava_model', 'llm', 'language_model']:
                wrapper = getattr(net, attr, None)
                if wrapper is not None:
                    wrapper_attr = attr
                    break
            if wrapper is not None:
                # Step 2: if wrapper IS already a causal LM (has prepare_inputs_for_generation), use it directly
                if hasattr(wrapper, 'prepare_inputs_for_generation'):
                    vlm_model = wrapper
                    parent_obj = net
                    parent_attr = wrapper_attr
                    logger.info(f"[LoRA] Using wrapper net.{wrapper_attr} as causal LM directly")
                else:
                    # Step 3: drill into multimodal wrapper to find the causal LM inside
                    for attr in ['llm', 'language_model']:
                        inner = getattr(wrapper, attr, None)
                        if inner is not None and hasattr(inner, 'prepare_inputs_for_generation'):
                            vlm_model = inner
                            parent_obj = wrapper
                            parent_attr = attr
                            logger.info(f"[LoRA] Found causal LM via net.{wrapper_attr}.{attr}")
                            break
                    # Step 4: fallback — use the wrapper itself
                    if vlm_model is None:
                        vlm_model = wrapper
                        parent_obj = net
                        parent_attr = wrapper_attr
                        logger.info(f"[LoRA] Fallback: using net.{wrapper_attr} (no prepare_inputs_for_generation found inside)")

        if vlm_model is None:
            logger.warning("[LoRA] Could not find VLM language model, skipping LoRA")
            return

        try:
            # Validate target modules exist before applying LoRA
            available_linears = []
            for n, m in vlm_model.named_modules():
                if isinstance(m, torch.nn.Linear):
                    available_linears.append(n)
            logger.info(f"[LoRA] Available Linear modules in target model: {available_linears}")
            valid_targets = [t for t in target_modules if any(n.endswith(t) for n in available_linears)]
            if not valid_targets:
                logger.warning(
                    f"[LoRA] None of the requested target_modules {target_modules} found in model. "
                    f"Available Linear modules: {available_linears}. "
                    f"Will auto-detect Qwen/LLaMA projection layers."
                )
                valid_targets = []
                for n in available_linears:
                    n_short = n.split('.')[-1]
                    if n_short not in valid_targets and n_short.endswith('_proj'):
                        valid_targets.append(n_short)
                if not valid_targets:
                    valid_targets = list(set(n.split('.')[-1] for n in available_linears if 'proj' in n.lower()))
            logger.info(f"[LoRA] Using target_modules: {valid_targets}")

            peft_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=valid_targets,
                lora_dropout=lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
            peft_model = get_peft_model(vlm_model, peft_config)

            # ── CRITICAL: assign PeftModel BACK to the model hierarchy ──
            if parent_obj is not None and parent_attr is not None:
                setattr(parent_obj, parent_attr, peft_model)
                logger.info(
                    f"[LoRA CRITICAL] Assigned PeftModel back: "
                    f"parent={type(parent_obj).__name__}, attr={parent_attr}, "
                    f"peft_model={type(peft_model).__name__}"
                )
            else:
                logger.error("[LoRA CRITICAL] Could not determine parent object — PeftModel wrapper is lost!")
                return

            logger.info(f"[LoRA] Applied to VLM: r={lora_r}, alpha={lora_alpha}, target_modules={valid_targets}")

            # ── Post-LoRA verification ──
            self._verify_lora_setup(peft_model, lora_r, lora_alpha)
        except Exception as e:
            logger.warning(f"[LoRA] Failed: {e}. Continuing without LoRA.")
            import traceback
            traceback.print_exc()

    def _verify_lora_setup(self, peft_model, lora_r, lora_alpha):
        """Verify that LoRA setup is correct: PeftModel, trainable params, lora_A/B present."""
        import torch
        from peft import PeftModel

        is_peft = isinstance(peft_model, PeftModel)
        logger.info(f"[LoRA VERIFY] peft_model is PeftModel = {is_peft}")
        logger.info(f"[LoRA VERIFY] peft_model type = {type(peft_model)}")

        trainable = [(n, p.numel()) for n, p in self.policy.named_parameters() if p.requires_grad]
        total_trainable = sum(x[1] for x in trainable)
        logger.info(f"[LoRA VERIFY] Policy trainable param tensors = {len(trainable)}")
        logger.info(f"[LoRA VERIFY] Policy trainable params = {total_trainable / 1e6:.2f}M")
        if len(trainable) > 0:
            logger.info(f"[LoRA VERIFY] First 20 trainable params: {[x[0] for x in trainable[:20]]}")

        lora_params = [
            n for n, p in self.policy.named_parameters()
            if ("lora_A" in n or "lora_B" in n)
        ]
        logger.info(f"[LoRA VERIFY] lora_A/lora_B param count = {len(lora_params)}")
        if len(lora_params) > 0:
            logger.info(f"[LoRA VERIFY] First 20 lora params: {lora_params[:20]}")

        # Also check peft_model internal params
        peft_trainable = [(n, p.numel()) for n, p in peft_model.named_parameters() if p.requires_grad]
        logger.info(f"[LoRA VERIFY] peft_model trainable params = {sum(x[1] for x in peft_trainable) / 1e6:.2f}M")

        # Safety assertions
        if not is_peft:
            logger.error("[LoRA VERIFY] CRITICAL: model is NOT PeftModel!")
        if len(trainable) == 0:
            logger.error("[LoRA VERIFY] CRITICAL: no trainable params in policy!")
        if len(lora_params) == 0:
            logger.error("[LoRA VERIFY] CRITICAL: no lora_A/lora_B params found!")

    def _lora_one_batch_probe(self, dataloader) -> None:
        """One-batch forward+backward probe to catch graph-disconnection early.

        Takes the first training batch, runs model forward, checks loss.requires_grad,
        and attempts loss.backward(). Reports detailed diagnostics.
        """
        import torch
        from peft import PeftModel

        logger.info("[LoRA PROBE] ======== Starting one-batch probe ========")
        try:
            first_iter = iter(dataloader)
            batch = next(first_iter, None)
            if batch is None:
                logger.error("[LoRA PROBE] No batch available for probe!")
                return
            obs_batch, prev_act, not_done, corr_act, weights = batch
        except Exception as e:
            logger.error(f"[LoRA PROBE] Failed to get batch: {e}")
            return

        # Print batch info
        logger.info(f"[LoRA PROBE] Batch loaded: obs_keys={list(obs_batch.keys())}")
        for k, v in obs_batch.items():
            if isinstance(v, torch.Tensor):
                logger.info(f"[LoRA PROBE]   {k}: shape={tuple(v.shape)}, dtype={v.dtype}")

        # Check policy type
        policy = self._get_policy()
        logger.info(f"[LoRA PROBE] policy type = {type(policy).__name__}")
        logger.info(f"[LoRA PROBE] policy.net type = {type(policy.net).__name__}")

        # Quick trainable param check (canonical)
        trainable_pre = [(n, p.numel()) for n, p in policy.named_parameters() if p.requires_grad]
        logger.info(f"[LoRA PROBE] Pre-probe trainable params = {sum(x[1] for x in trainable_pre) / 1e6:.2f}M "
                    f"({len(trainable_pre)} tensors)")

        # Move batch to device
        try:
            device = next(policy.parameters()).device
        except StopIteration:
            device = self.device

        obs_batch_dev = {}
        for k, v in obs_batch.items():
            if isinstance(v, torch.Tensor):
                obs_batch_dev[k] = v.to(device)
            else:
                obs_batch_dev[k] = v
        prev_act_dev = prev_act.to(device)
        not_done_dev = not_done.to(device)

        # Run forward (first timestep only)
        logger.info("[LoRA PROBE] Running forward pass...")
        policy.train()
        B_actual, T_actual = corr_act.shape

        obs_t = {}
        for k, v in obs_batch_dev.items():
            v_reshaped = v.view(B_actual, T_actual, *v.shape[1:])
            obs_t[k] = v_reshaped[:, 0].contiguous()
        prev_action_t = prev_act_dev.view(B_actual, T_actual, 1)[:, 0].contiguous()
        mask_t = not_done_dev.view(B_actual, T_actual, 1)[:, 0].contiguous()

        hidden_state = torch.zeros(
            B_actual, policy.num_recurrent_layers, policy.recurrent_hidden_size,
            device=device,
        )

        try:
            features_t, _, _ = policy.net(obs_t, hidden_state, prev_action_t, mask_t)
        except Exception as e:
            logger.error(f"[LoRA PROBE] Forward pass failed: {e}")
            import traceback
            traceback.print_exc()
            return

        logger.info(f"[LoRA PROBE] features_t shape={tuple(features_t.shape)}, "
                    f"requires_grad={features_t.requires_grad}")

        # Run action distribution
        dist = policy.action_distribution(features_t)
        logger.info(f"[LoRA PROBE] dist.logits shape={tuple(dist.logits.shape)}, "
                    f"requires_grad={dist.logits.requires_grad}, "
                    f"grad_fn={dist.logits.grad_fn}")

        # Create dummy labels and compute loss
        corr_act_flat = corr_act[:, 0].to(device)
        iw_t = weights[:, 0].to(device)
        valid_mask = (corr_act_flat >= 0) & (corr_act_flat < 4) & (iw_t > 0)
        if not valid_mask.any():
            logger.error("[LoRA PROBE] No valid actions in first timestep!")
            return
        corr_act_flat = torch.where(valid_mask, corr_act_flat, torch.zeros_like(corr_act_flat))
        ce = self.criterion(dist.logits, corr_act_flat)
        loss = (ce * iw_t).sum() / iw_t.sum()

        logger.info(f"[LoRA PROBE] loss={loss.item():.6f}, "
                    f"requires_grad={loss.requires_grad}, "
                    f"grad_fn={loss.grad_fn}")

        # Logits
        if hasattr(dist, 'logits'):
            logger.info(f"[LoRA PROBE] logits.requires_grad={dist.logits.requires_grad}, "
                        f"logits.grad_fn={dist.logits.grad_fn}")

        # Labels check
        labels_non_ignore = (corr_act_flat != -100).sum().item()
        logger.info(f"[LoRA PROBE] labels shape={tuple(corr_act_flat.shape)}, "
                    f"non-IGNORE_INDEX count={labels_non_ignore}")

        if not loss.requires_grad:
            logger.error("[LoRA PROBE] *** LOSS DOES NOT REQUIRE GRAD ***")
            if hasattr(dist, 'logits') and dist.logits.requires_grad:
                logger.warning("[LoRA PROBE] logits requires_grad=True but loss does not — "
                               "this is unusual, check loss computation")
            else:
                logger.error("[LoRA PROBE] Neither logits nor loss require grad — "
                             "check forward path for inference_mode/no_grad/detach")
        else:
            logger.info("[LoRA PROBE] ✓ loss.requires_grad=True")

        # Try backward
        logger.info("[LoRA PROBE] Attempting loss.backward()...")
        try:
            loss.backward()
            logger.info("[LoRA PROBE] ✓ loss.backward() succeeded!")
        except Exception as e:
            logger.error(f"[LoRA PROBE] *** loss.backward() FAILED: {e} ***")
            # Print full diagnostics
            trainable_diag = [(n, p.numel(), p.requires_grad, p.grad_fn if hasattr(p, 'grad_fn') else 'N/A')
                              for n, p in policy.named_parameters() if p.requires_grad]
            logger.error(f"[LoRA PROBE] Current trainable params ({len(trainable_diag)}):")
            for n, numel, rg, gf in trainable_diag[:20]:
                logger.error(f"[LoRA PROBE]   {n}: numel={numel}, requires_grad={rg}")
            return

        # Check LoRA grad
        lora_grad_info = []
        for n, p in policy.named_parameters():
            if ("lora_A" in n or "lora_B" in n) and p.requires_grad:
                grad_norm = p.grad.norm().item() if p.grad is not None else 0.0
                lora_grad_info.append((n, p.grad is not None, grad_norm))

        logger.info(f"[LoRA PROBE] LoRA grad check ({len(lora_grad_info)} lora params):")
        for n, has_grad, gnorm in lora_grad_info[:10]:
            logger.info(f"[LoRA PROBE]   {n}: grad_not_None={has_grad}, grad_norm={gnorm:.6f}")

        any_lora_grad = any(info[1] for info in lora_grad_info)
        if any_lora_grad:
            logger.info("[LoRA PROBE] ✓ At least one LoRA param has non-None grad!")
        else:
            logger.error("[LoRA PROBE] *** NO LoRA param has grad! ***")

        # Cleanup
        policy.zero_grad(set_to_none=True)
        logger.info("[LoRA PROBE] ======== Probe complete (grads cleared) ========")

    def _install_lora_forward_hook(self) -> None:
        """Install a forward hook on the first LoRA-enabled q_proj to monitor gradient flow."""
        import torch

        found = False
        for name, module in self.policy.named_modules():
            if name.endswith("q_proj") and hasattr(module, "lora_A"):
                found = True
                logger.info(f"[LoRA HOOK] Installing forward hook on: {name}")

                def _make_hook(mod_name):
                    def _hook(module, input, output):
                        if not hasattr(_hook, '_fired'):
                            _hook._fired = True
                            if isinstance(output, tuple):
                                out0 = output[0]
                            else:
                                out0 = output
                            logger.info(
                                f"[LoRA HOOK] q_proj '{mod_name}' output: "
                                f"requires_grad={out0.requires_grad}, "
                                f"grad_fn={out0.grad_fn}"
                            )
                        return output
                    _hook._fired = False
                    return _hook

                module.register_forward_hook(_make_hook(name))
                break

        if not found:
            logger.warning("[LoRA HOOK] No q_proj with lora_A found — LoRA may not be properly injected!")
            # Print all modules ending with q_proj for debugging
            qproj_names = [n for n, m in self.policy.named_modules() if n.endswith("q_proj")]
            logger.info(f"[LoRA HOOK] All q_proj modules found: {qproj_names[:10]}")
            for n in qproj_names[:5]:
                m = dict(self.policy.named_modules())[n]
                logger.info(f"[LoRA HOOK]   {n}: type={type(m).__name__}, "
                            f"has_lora_A={hasattr(m, 'lora_A')}, "
                            f"has_lora_B={hasattr(m, 'lora_B')}")

    def _apply_lora_to_resnet_policy(self):
        """Apply LoRA to transformer modules in ResNet policy"""
        if not PEFT_AVAILABLE:
            return

        model_cfg = self.config.habitat_baselines.il.model
        lora_r = getattr(model_cfg, "lora_r", 8)
        lora_alpha = getattr(model_cfg, "lora_alpha", 16)
        lora_dropout = getattr(model_cfg, "lora_dropout", 0.1)

        linear_modules = []
        for name, module in self.policy.named_modules():
            if isinstance(module, torch.nn.Linear) and any(
                t in name.lower() for t in ["text_encoder", "attention", "fusion", "proj"]
            ):
                linear_modules.append(name.split('.')[-1])

        if linear_modules:
            target_modules = list(set(linear_modules))
            try:
                peft_config = LoraConfig(
                    r=lora_r, lora_alpha=lora_alpha,
                    target_modules=target_modules,
                    lora_dropout=lora_dropout, bias="none",
                    task_type="FEATURE_EXTRACTION",
                )
                self.policy = get_peft_model(self.policy, peft_config)
                logger.info(f"[LoRA] Applied to ResNet: {target_modules}")
            except Exception as e:
                logger.warning(f"[LoRA] Failed on ResNet: {e}")
    def train(self) -> None:
        """训练主循环（直接从文件读取数据，不经过模拟器）"""
        observation_space, action_space = self._get_spaces()
        self._initialize_policy(observation_space, action_space)
        
        # ========== 尝试从checkpoint恢复训练 ==========
        resume_checkpoint = None
        start_epoch = 0
        start_step = 0
        best_loss_resumed = float('inf')
        
        if self._is_rank0():
            resume_checkpoint = self.load_checkpoint_for_resume()
        
        # 广播resume信息到所有进程（如果是分布式训练）
        if self._is_distributed:
            import torch.distributed as dist
            # 创建一个tensor来广播是否有checkpoint
            has_ckpt_tensor = torch.tensor([resume_checkpoint is not None], dtype=torch.bool).cuda()
            dist.broadcast(has_ckpt_tensor, src=0)
            has_checkpoint = has_ckpt_tensor.item()
        else:
            has_checkpoint = resume_checkpoint is not None
        
        # 如果有checkpoint，加载它
        if has_checkpoint and resume_checkpoint is not None:
            try:
                # 加载模型权重
                missing_keys, unexpected_keys = self.policy.load_state_dict(
                    resume_checkpoint["state_dict"], strict=False
                )
                # logger.info(f"✓ Loaded model weights from checkpoint")

                # 加载优化器状态
                self.optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
                # logger.info(f"✓ Loaded optimizer state from checkpoint")
                
                # 恢复训练进度
                start_epoch = resume_checkpoint["epoch"] + 1  # 从下一个epoch开始
                start_step = resume_checkpoint["step_id"]
                best_loss_resumed = resume_checkpoint.get("best_loss", float('inf'))

                # 恢复LR scheduler状态（将scheduler步进到当前epoch）
                if self.lr_scheduler is not None:
                    for _ in range(start_epoch):
                        self.lr_scheduler.step()
                
                # logger.info(f"{'='*80}")
                # logger.info(f"RESUMING TRAINING FROM CHECKPOINT")
                # logger.info(f"{'='*80}")
                # logger.info(f"Resuming from epoch: {start_epoch}")
                # logger.info(f"Total steps completed: {start_step}")
                # logger.info(f"Best loss so far: {best_loss_resumed:.4f}")
                # logger.info(f"{'='*80}\n")
            except Exception as e:
                logger.error(f"Failed to load checkpoint state: {e}")
                logger.warning("Starting training from scratch instead")
                start_epoch = 0
                start_step = 0
                best_loss_resumed = float('inf')
        # ================================================
        
        # TensorBoard writer
        if self._is_rank0():
            writer_context = TensorboardWriter(
                self.config.habitat_baselines.tensorboard_dir,
                flush_secs=self.flush_secs,
                purge_step=0,
            )
        else:
            from contextlib import nullcontext
            writer_context = nullcontext()
        
        try:
            with writer_context as writer:
                # 创建数据集
                data_root = self.config.habitat_baselines.il.direct_il.data_root
                max_episodes = getattr(self.config.habitat_baselines.il.direct_il, "max_episodes", -1)
                dataset_type = getattr(self.config.habitat_baselines.il.direct_il, "dataset_type", "directory")
                rgb_data_roots = getattr(self.config.habitat_baselines.il.direct_il, "rgb_data_roots", None)
                disable_rgb_decode = getattr(self.config.habitat_baselines.il.direct_il, "disable_rgb_decode", False)
                forward_step_size = getattr(self.config.habitat_baselines.il.direct_il, "forward_step_size", 0.25)
                turn_angle_deg = getattr(self.config.habitat_baselines.il.direct_il, "turn_angle_deg", 10.0)
                window_size = getattr(self.config.habitat_baselines.il.direct_il, "window_size", 0)
                skip_empty_inst = getattr(self.config.habitat_baselines.il.direct_il, "skip_episodes_with_empty_instruction", True)
                future_step = getattr(self.config.habitat_baselines.il.direct_il, "future_step", 4)
                try:
                    future_step = getattr(
                        self.config.habitat_baselines.rl.auxiliary_losses.future_trajectory_prediction,
                        "future_step",
                        future_step,
                    )
                except Exception:
                    pass
                # 从配置中读取指令优先级（如果未配置则使用默认值）
                instruction_priority = getattr(
                    self.config.habitat_baselines.il.direct_il, 
                    "instruction_priority", 
                    None
                )
                # 将 OmegaConf ListConfig 转换为 Python list（如果存在）
                if instruction_priority is not None:
                    from omegaconf import ListConfig
                    if isinstance(instruction_priority, ListConfig):
                        instruction_priority = list(instruction_priority)
                
                max_episode_length = getattr(self.config.habitat_baselines.il.direct_il, 'max_episode_length', 400)
                if max_episode_length <= 0:
                    max_episode_length = 400

                # 对于直接从文件读取的episode数据，过大的batch会显著放大内存峰值
                batch_size = self.config.habitat_baselines.il.batch_size
                if batch_size is None or batch_size <= 0:
                    batch_size = 1
                if max_episode_length >= 300 and batch_size > 4:
                    batch_size = 4

                # 检查是否启用混合指令模式
                use_mixed_instructions = getattr(
                    self.config.habitat_baselines.il.direct_il,
                    "use_mixed_instructions",
                    True,
                )

                dataset = DirectFileDataset(
                    data_root=data_root,
                    use_iw=self.config.habitat_baselines.il.use_iw,
                    inflection_weight_coef=self.config.habitat_baselines.il.inflection_weight_coef,
                    batch_size=batch_size,
                    max_episodes=max_episodes,
                    instruction_priority=instruction_priority,
                    max_episode_length=max_episode_length,
                    use_mixed_instructions=use_mixed_instructions,
                    rgb_data_roots=rgb_data_roots,
                    dataset_type=dataset_type,
                    disable_rgb_decode=disable_rgb_decode,
                    forward_step_size=forward_step_size,
                    turn_angle_deg=turn_angle_deg,
                    window_size=window_size,
                    skip_episodes_with_empty_instruction=skip_empty_inst,
                    future_step=future_step,
                )

                # ── 诊断信息：数据集统计 ──
                if self._is_rank0():
                    logger.info(f"[DirectFileDataset] Episodes loaded: {len(dataset)}")
                    logger.info(f"[DirectFileDataset] Window size: {dataset.window_size} "
                                f"(0=full episode, >0=sliding window)")
                    logger.info(f"[DirectFileDataset] Skip empty instruction: "
                                f"{dataset.skip_episodes_with_empty_instruction}")
                    logger.info(f"[DirectFileDataset] Forward step: {dataset.forward_step_size}m, "
                                f"Turn angle: {dataset.turn_angle_deg}deg")

                if len(dataset) == 0:
                    raise RuntimeError(
                        f"DirectFileDataset loaded 0 samples from {data_root}. "
                        "Please verify directory structure, instruction files, and episode files."
                    )
                
                # 分布式训练设置
                if self._is_distributed:
                    dataset.set_distributed_info(self._dist_rank, self._dist_world_size)
                
                # 数据加载器
                num_workers = getattr(self.config.habitat_baselines.il, 'dataloader_num_workers', 0)
                pin_memory = getattr(self.config.habitat_baselines.il, 'pin_memory', False)
                effective_batch_size = batch_size
                if effective_batch_size <= 0:
                    effective_batch_size = 1

                # logger.info(f"\nDataLoader Configuration:")
                # logger.info(f"  Batch size: {effective_batch_size}")
                # logger.info(f"  Num workers: {num_workers}")
                # logger.info(f"  Pin memory: {pin_memory}")
                # logger.info(f"  Shuffle: False (handled by dataset)")

                dataloader = torch.utils.data.DataLoader(
                    dataset,
                    batch_size=effective_batch_size,
                    shuffle=False,
                    collate_fn=collate_fn,
                    pin_memory=pin_memory,
                    drop_last=False,
                    num_workers=num_workers,
                )
                
                # 测试加载第一个batch
                if self._is_rank0():
                    # logger.info("\nTesting data loading with first batch...")
                    # logger.info(f"[DirectFileDataset DEBUG] dataset_type={dataset.dataset_type}, data_root={data_root}")
                    # logger.info(f"[DirectFileDataset DEBUG] episode_count={len(dataset.episode_paths)}, length={len(dataset)}, batch_size={effective_batch_size}, num_workers={num_workers}")
                    # logger.info(f"[DirectFileDataset DEBUG] rgb_data_roots={dataset.rgb_data_roots}")
                    setattr(dataset, '_debug_resolve_paths', True)
                    setattr(dataset, '_debug_count', 0)
                    try:
                        first_iter = iter(dataloader)
                        test_batch = next(first_iter, None)
                        if test_batch is None:
                            logger.error("[DirectFileDataset DEBUG] DataLoader returned None for first batch")
                            logger.error("[DirectFileDataset DEBUG] This usually means every sample was skipped in __iter__ or collate_fn filtered them all out.")
                            raise RuntimeError(
                                "DirectFileDataset/DataLoader produced no batch. "
                                "This usually means all episodes were filtered out or skipped."
                            )
                        obs_batch, prev_act, not_done, corr_act, weights = test_batch
                        # logger.info(f"[DirectFileDataset DEBUG] first batch loaded: obs_keys={list(obs_batch.keys())}")
                        # logger.info(f"[DirectFileDataset DEBUG] prev_act_shape={tuple(prev_act.shape)}, corr_act_shape={tuple(corr_act.shape)}, weights_shape={tuple(weights.shape)}")
                        # logger.info(f"[DirectFileDataset DEBUG] corr_act_minmax=[{corr_act.min().item()}, {corr_act.max().item()}]")
                        # logger.info(f"[DirectFileDataset DEBUG] weights_minmax=[{weights.min().item():.4f}, {weights.max().item():.4f}] sum={weights.sum().item():.2f}")
                        corr_act_flat = corr_act.view(-1)
                        valid_mask = (corr_act_flat >= 0) & (corr_act_flat < 4)
                        if valid_mask.any():
                            valid_actions = corr_act_flat[valid_mask].long()
                            action_hist = torch.bincount(valid_actions, minlength=4)
                        else:
                            action_hist = torch.zeros(4, dtype=torch.long, device=corr_act.device)
                        logger.info(f"[DirectFileDataset DEBUG] action_distribution={action_hist.cpu().numpy().tolist()}")
                    except Exception as e:
                        logger.error(f"✗ Failed to load first batch: {e}")
                        import traceback
                        traceback.print_exc()


                
                epochs = self.config.habitat_baselines.il.epochs
                # logger.info(f"\n{'='*80}")
                # logger.info(f"TRAINING CONFIGURATION")
                # logger.info(f"{'='*80}")
                # logger.info(f"Total episodes: {len(dataset)}")
                # logger.info(f"Total epochs: {epochs}")
                # logger.info(f"Batch size: {self.config.habitat_baselines.il.batch_size}")
                # logger.info(f"Batches per epoch: {len(dataloader)}")
                # logger.info(f"Total training steps: {epochs * len(dataloader)}")
                # logger.info(f"Use inflection weighting: {self.config.habitat_baselines.il.use_iw}")
                # logger.info(f"Inflection weight coef: {self.config.habitat_baselines.il.inflection_weight_coef}")
                # logger.info(f"{'='*80}\n")
                
                AuxLosses.activate()
                
                step_id = start_step  # 使用resume的step
                epoch_losses = []
                best_loss = best_loss_resumed  # 使用resume的best_loss
                
                # 用于统计的变量
                grad_norms = []
                action_distributions = []
                
                self._debug_minimal_mode = bool(getattr(self.config.habitat_baselines.il, "debug_minimal_mode", False))
                self._debug_print_batch_shapes = bool(getattr(self.config.habitat_baselines.il, "debug_print_batch_shapes", False))
                self._debug_print_every_batch = bool(getattr(self.config.habitat_baselines.il, "debug_print_every_batch", True))
                self._debug_dump_episode_info = bool(getattr(self.config.habitat_baselines.il, "debug_dump_episode_info", True))
                self._debug_max_episode_info = int(getattr(self.config.habitat_baselines.il, "debug_max_episode_info", 8))
                self._debug_step_prefix = "[DirectIL DEBUG]"

                def _debug_log(message: str):
                    if self._is_rank0():
                        logger.info(f"{self._debug_step_prefix} {message}")

                def _debug_tensor_summary(name, value):
                    if value is None:
                        return f"{name}=None"
                    if isinstance(value, torch.Tensor):
                        return (
                            f"{name}: shape={tuple(value.shape)}, dtype={value.dtype}, "
                            f"device={value.device}, min={value.min().item() if value.numel() > 0 else 'n/a'}, "
                            f"max={value.max().item() if value.numel() > 0 else 'n/a'}"
                        )
                    if isinstance(value, dict):
                        return f"{name}: dict(keys={list(value.keys())})"
                    return f"{name}: type={type(value).__name__}"

                def _debug_episode_info_from_batch(obs_batch, batch_idx):
                    episode_info = []
                    if not self._debug_dump_episode_info:
                        return episode_info
                    candidate_keys = [
                        "episode_id",
                        "episode_ids",
                        "episode_path",
                        "episode_paths",
                        "scene_id",
                        "scene_ids",
                        "instruction",
                        "instruction_text",
                        "falcon_instruction",
                    ]
                    for key in candidate_keys:
                        if key in obs_batch:
                            tensor = obs_batch[key]
                            if isinstance(tensor, torch.Tensor):
                                preview = tensor[: min(tensor.shape[0], self._debug_max_episode_info)]
                                episode_info.append(f"{key}={preview.detach().cpu().tolist() if preview.numel() > 0 else []}")
                            else:
                                episode_info.append(f"{key}={tensor}")
                    if not episode_info:
                        episode_info.append(f"batch_idx={batch_idx}")
                    return episode_info

                if self._debug_minimal_mode and self._is_rank0():
                    logger.info(f"{self._debug_step_prefix} minimal debug mode enabled")
                    logger.info(f"{self._debug_step_prefix} start_epoch={start_epoch}, epochs={epochs}, len(dataloader)={len(dataloader)}")
                    logger.info(f"{self._debug_step_prefix} device={self.device}, num_workers={getattr(dataloader, 'num_workers', 'n/a')}")

                # ── LoRA one-batch probe: 提前发现断图问题 ──
                _use_lora = getattr(self.config.habitat_baselines.il.model, "use_lora", False)
                _skip_probe = os.environ.get("LORA_SKIP_PROBE", "0") == "1"
                if _use_lora and self._is_rank0():
                    if _skip_probe:
                        logger.info("[LoRA] Skipping one-batch probe (LORA_SKIP_PROBE=1)")
                    else:
                        self._lora_one_batch_probe(dataloader)

                # ── LoRA forward hook: 监控第一个 q_proj 的输出版本 ──
                _skip_hook = os.environ.get("LORA_SKIP_HOOK", "0") == "1"
                if _use_lora and self._is_rank0():
                    if _skip_hook:
                        logger.info("[LoRA] Skipping forward hook (LORA_SKIP_HOOK=1)")
                    else:
                        self._install_lora_forward_hook()

                # ── 清理内存碎片，防止 malloc_consolidate crash ──
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                if self._is_rank0():
                    logger.info("[LoRA] Memory cleanup: gc + cuda empty_cache done")

                for epoch in (range(start_epoch, epochs) if self._debug_minimal_mode else tqdm.trange(
                    start_epoch,  # 从resume的epoch开始
                    epochs,
                    initial=start_epoch,  # 设置进度条初始值
                    dynamic_ncols=True,
                    desc="Epoch",
                    disable=not self._is_rank0(),
                )):
                    self._current_epoch = epoch + 1
                    if self._is_rank0():
                        logger.info(f"\n{'='*80}")
                        logger.info(f"EPOCH {epoch + 1}/{epochs}")
                        logger.info(f"{'='*80}")

                    if self._debug_minimal_mode and self._is_rank0():
                        logger.info(f"{self._debug_step_prefix} entering epoch loop: epoch={epoch}, step_id={step_id}")

                    epoch_loss = 0.0
                    epoch_steps = 0
                    epoch_action_acc = 0.0
                    epoch_valid_samples = 0
                    # Per-class 准确率累积（用于解决类别不均衡评估问题）
                    # 索引 0-3 对应 stop/forward/turn_left/turn_right
                    epoch_per_class_correct = [0.0, 0.0, 0.0, 0.0]
                    epoch_per_class_count = [0.0, 0.0, 0.0, 0.0]
                    # 预测分布累积
                    epoch_pred_counts = [0.0, 0.0, 0.0, 0.0]
                    epoch_target_counts = [0.0, 0.0, 0.0, 0.0]

                    batch_iterator = dataloader if self._debug_minimal_mode else tqdm.tqdm(
                        dataloader,
                        leave=False,
                        dynamic_ncols=True,
                        desc="Batch",
                        disable=not self._is_rank0(),
                    )

                    for batch_idx, batch in enumerate(batch_iterator):
                        if self._debug_minimal_mode and self._is_rank0() and self._debug_print_every_batch:
                            logger.info(f"{self._debug_step_prefix} epoch={epoch + 1}, batch={batch_idx}: fetched batch")
                        try:
                            (
                                observations_batch,
                                prev_actions_batch,
                                not_done_masks,
                                corrected_actions_batch,
                                weights_batch,
                            ) = batch

                            if self._debug_minimal_mode and self._is_rank0() and self._debug_print_batch_shapes:
                                logger.info(f"{self._debug_step_prefix} batch={batch_idx} raw observations keys={list(observations_batch.keys())}")
                                for k, v in observations_batch.items():
                                    logger.info(f"{self._debug_step_prefix} raw {_debug_tensor_summary(k, v)}")
                                logger.info(f"{self._debug_step_prefix} raw {_debug_tensor_summary('prev_actions_batch', prev_actions_batch)}")
                                logger.info(f"{self._debug_step_prefix} raw {_debug_tensor_summary('not_done_masks', not_done_masks)}")
                                logger.info(f"{self._debug_step_prefix} raw {_debug_tensor_summary('corrected_actions_batch', corrected_actions_batch)}")
                                logger.info(f"{self._debug_step_prefix} raw {_debug_tensor_summary('weights_batch', weights_batch)}")

                            if self._debug_minimal_mode and self._is_rank0():
                                episode_info = _debug_episode_info_from_batch(observations_batch, batch_idx)
                                logger.info(f"{self._debug_step_prefix} batch={batch_idx} episode_info={episode_info[:self._debug_max_episode_info]}")

                            if self._debug_minimal_mode and self._is_rank0():
                                logger.info(f"{self._debug_step_prefix} moving batch={batch_idx} to device={self.device}")

                            observations_batch = {
                                k: v.to(device=self.device, non_blocking=True)
                                for k, v in observations_batch.items()
                            }
                            prev_actions_batch = prev_actions_batch.to(self.device)
                            not_done_masks = not_done_masks.to(self.device)
                            corrected_actions_batch = corrected_actions_batch.to(self.device)
                            weights_batch = weights_batch.to(self.device)

                            if self._debug_minimal_mode and self._is_rank0() and self._debug_print_batch_shapes:
                                logger.info(f"{self._debug_step_prefix} batch={batch_idx} moved to device")
                                for k, v in observations_batch.items():
                                    logger.info(f"{self._debug_step_prefix} device {_debug_tensor_summary(k, v)}")
                                logger.info(f"{self._debug_step_prefix} device {_debug_tensor_summary('prev_actions_batch', prev_actions_batch)}")
                                logger.info(f"{self._debug_step_prefix} device {_debug_tensor_summary('not_done_masks', not_done_masks)}")
                                logger.info(f"{self._debug_step_prefix} device {_debug_tensor_summary('corrected_actions_batch', corrected_actions_batch)}")
                                logger.info(f"{self._debug_step_prefix} device {_debug_tensor_summary('weights_batch', weights_batch)}")

                            policy = self._get_policy()
                            policy.train()
                            self.optimizer.zero_grad()

                            AuxLosses.clear()

                            # ── 仿照 dynamic_vln_trainer: 预计算 visual features 注入 batch ──
                            # 仅适用于 PointNavResNetPolicy (有独立的 visual_encoder)
                            # NaVILA/StreamVLN 等 VLM 策略没有 visual_encoder，跳过此步骤
                            if self._is_static_encoder and hasattr(policy.net, 'visual_encoder'):
                                from habitat_baselines.rl.ddppo.policy.resnet_policy import PointNavResNetNet
                                visual_encoder = policy.net.visual_encoder
                                visual_encoder.eval()
                                with inference_mode():
                                    visual_feats = visual_encoder(observations_batch)
                                visual_feats = visual_feats.detach().clone()
                                observations_batch[
                                    PointNavResNetNet.PRETRAINED_VISUAL_FEATURES_KEY
                                ] = visual_feats

                            if self._debug_minimal_mode and self._is_rank0():
                                logger.info(f"{self._debug_step_prefix} batch={batch_idx} before policy.net")
                                if self._debug_print_batch_shapes:
                                    for k, v in observations_batch.items():
                                        logger.info(f"{self._debug_step_prefix} policy input {_debug_tensor_summary(k, v)}")
                                    logger.info(f"{self._debug_step_prefix} policy input {_debug_tensor_summary('prev_actions_batch', prev_actions_batch)}")
                                    logger.info(f"{self._debug_step_prefix} policy input {_debug_tensor_summary('not_done_masks', not_done_masks)}")

                            # ── 修复：按时间步迭代，正确传递 RNN hidden state ──
                            # collate_fn 将 obs/prev_actions/masks reshape 为 [B*T, ...]，
                            # 但 corrected_actions_batch 仍为 [B, T]，用它的 shape 还原 B 和 T
                            B_actual, T_actual = corrected_actions_batch.shape

                            hidden_state = torch.zeros(
                                B_actual,
                                policy.num_recurrent_layers,
                                policy.recurrent_hidden_size,
                                device=self.device,
                            )

                            all_features = []
                            for t in range(T_actual):
                                obs_t = {}
                                for k, v in observations_batch.items():
                                    v_reshaped = v.view(B_actual, T_actual, *v.shape[1:])
                                    obs_t[k] = v_reshaped[:, t].contiguous()
                                prev_action_t = prev_actions_batch.view(B_actual, T_actual, 1)[:, t].contiguous()
                                mask_t = not_done_masks.view(B_actual, T_actual, 1)[:, t].contiguous()

                                features_t, hidden_state, aux_loss_state = policy.net(
                                    obs_t,
                                    hidden_state,
                                    prev_action_t,
                                    mask_t,
                                )
                                all_features.append(features_t)

                            features = torch.cat(all_features, dim=0)  # [B*T, feature_dim]

                            if self._debug_minimal_mode and self._is_rank0():
                                logger.info(f"{self._debug_step_prefix} batch={batch_idx} after policy.net (T={T_actual} steps)")
                                logger.info(f"{self._debug_step_prefix} features shape={tuple(features.shape)}, dtype={features.dtype}, device={features.device}")
                                if isinstance(aux_loss_state, dict):
                                    logger.info(f"{self._debug_step_prefix} aux_loss_state keys={list(aux_loss_state.keys())}")
                                else:
                                    logger.info(f"{self._debug_step_prefix} aux_loss_state type={type(aux_loss_state).__name__}")

                            dist = policy.action_distribution(features)
                            if self._debug_minimal_mode and self._is_rank0():
                                logger.info(f"{self._debug_step_prefix} batch={batch_idx} dist.logits shape={tuple(dist.logits.shape)}, dtype={dist.logits.dtype}, device={dist.logits.device}")

                            corrected_actions_flat = corrected_actions_batch.view(-1)
                            iw = weights_batch.view(-1)
                            valid_mask = (corrected_actions_flat >= 0) & (corrected_actions_flat < 4) & (iw > 0)

                            if not valid_mask.all():
                                corrected_actions_flat = torch.where(
                                    valid_mask,
                                    corrected_actions_flat,
                                    torch.zeros_like(corrected_actions_flat)
                                )

                            ce = self.criterion(dist.logits, corrected_actions_flat)

                            if iw.sum().item() == 0:
                                if self._debug_minimal_mode and self._is_rank0():
                                    logger.warning(f"{self._debug_step_prefix} batch={batch_idx} all samples have zero weight, skipping")
                                continue

                            action_loss = (ce * iw).sum() / iw.sum()
                            loss = action_loss

                            if self._debug_minimal_mode and self._is_rank0():
                                logger.info(f"{self._debug_step_prefix} batch={batch_idx} before backward loss={loss.item():.6f}")

                            loss.backward()

                            if self._debug_minimal_mode and self._is_rank0():
                                logger.info(f"{self._debug_step_prefix} batch={batch_idx} after backward")

                            total_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                            grad_norms.append(total_norm.item())

                            self.optimizer.step()
                            self._total_updates += 1

                            with torch.no_grad():
                                pred_actions = dist.logits.argmax(dim=-1)
                                correct = (pred_actions == corrected_actions_flat) & valid_mask
                                accuracy = correct.float().sum() / valid_mask.sum()
                                epoch_action_acc += accuracy.item()
                                epoch_valid_samples += 1

                                # ── Per-class accuracy & prediction distribution ──
                                pred_flat = pred_actions[valid_mask]
                                tgt_flat = corrected_actions_flat[valid_mask]
                                for cls_idx in range(4):
                                    cls_mask = (tgt_flat == cls_idx)
                                    cls_count = cls_mask.sum().item()
                                    if cls_count > 0:
                                        cls_correct = (pred_flat[cls_mask] == cls_idx).float().sum().item()
                                        epoch_per_class_correct[cls_idx] += cls_correct
                                        epoch_per_class_count[cls_idx] += cls_count
                                    # 预测分布
                                    epoch_pred_counts[cls_idx] += (pred_flat == cls_idx).float().sum().item()
                                    epoch_target_counts[cls_idx] += cls_count

                            loss_val = loss.item()
                            epoch_loss += loss_val
                            epoch_steps += 1

                            log_interval = getattr(self.config.habitat_baselines, 'log_interval', 10)
                            if step_id % log_interval == 0 and self._is_rank0():
                                avg_grad_norm = np.mean(grad_norms[-10:]) if grad_norms else 0
                                # 计算当前 batch 的 macro accuracy 用于进度条
                                batch_macro_acc = np.mean([
                                    epoch_per_class_correct[c] / max(epoch_per_class_count[c], 1)
                                    for c in range(4)
                                ])
                                if self._debug_minimal_mode:
                                    logger.info(f"{self._debug_step_prefix} batch={batch_idx} loss={loss_val:.4f} acc={accuracy.item():.3f} macro_acc={batch_macro_acc:.3f} grad={avg_grad_norm:.2e}")
                                elif 'pbar_batch' in locals():
                                    pbar_batch.set_postfix({
                                        'loss': f'{loss_val:.4f}',
                                        'acc': f'{accuracy.item():.3f}',
                                        'macro': f'{batch_macro_acc:.3f}',
                                        'grad': f'{avg_grad_norm:.2e}'
                                    })
                                if writer is not None:
                                    writer.add_scalar("train_loss", loss_val, step_id)
                                    writer.add_scalar("train_accuracy", accuracy.item(), step_id)
                                    writer.add_scalar("train_macro_accuracy", batch_macro_acc, step_id)
                                    writer.add_scalar("grad_norm", total_norm.item(), step_id)

                            step_id += 1
                        except Exception as e:
                            logger.error(f"{self._debug_step_prefix} batch={batch_idx} failed: {e}")
                            if self._debug_dump_episode_info:
                                try:
                                    logger.error(f"{self._debug_step_prefix} current batch episode info: {_debug_episode_info_from_batch(observations_batch if 'observations_batch' in locals() else {}, batch_idx)}")
                                except Exception:
                                    pass
                            import traceback
                            traceback.print_exc()
                            raise

                    
                    # Epoch结束统计
                    if epoch_steps > 0 and self._is_rank0():
                        avg_epoch_loss = epoch_loss / epoch_steps
                        avg_epoch_acc = epoch_action_acc / epoch_valid_samples if epoch_valid_samples > 0 else 0
                        epoch_losses.append(avg_epoch_loss)

                        # ── Per-class accuracy ──
                        action_names = ["stop(0)", "forward(1)", "turn_left(2)", "turn_right(3)"]
                        per_class_acc = []
                        valid_classes = 0
                        for cls_idx in range(4):
                            if epoch_per_class_count[cls_idx] > 0:
                                cls_acc = epoch_per_class_correct[cls_idx] / epoch_per_class_count[cls_idx]
                                per_class_acc.append(cls_acc)
                                valid_classes += 1
                            else:
                                per_class_acc.append(0.0)
                        macro_acc = np.mean(per_class_acc) if valid_classes > 0 else 0.0

                        # ── Prediction distribution (ratio) ──
                        total_pred = sum(epoch_pred_counts)
                        total_target = sum(epoch_target_counts)
                        pred_ratios = [epoch_pred_counts[c] / max(total_pred, 1) for c in range(4)]
                        target_ratios = [epoch_target_counts[c] / max(total_target, 1) for c in range(4)]

                        # ── 关键指标: stop recall ──
                        stop_recall = per_class_acc[0] if epoch_per_class_count[0] > 0 else 0.0

                        logger.info(f"\n{'='*80}")
                        logger.info(f"EPOCH {epoch + 1} SUMMARY")
                        logger.info(f"{'='*80}")
                        logger.info(f"Average Loss:         {avg_epoch_loss:.4f}")
                        logger.info(f"Overall Accuracy:     {avg_epoch_acc:.4f} (majority baseline=0.4556)")
                        logger.info(f"Macro Accuracy:       {macro_acc:.4f} (random=0.25)")
                        logger.info(f"Stop Recall (CRITICAL): {stop_recall:.4f}")
                        logger.info(f"Per-class Accuracy:")
                        for cls_idx in range(4):
                            logger.info(f"  {action_names[cls_idx]:16s}: acc={per_class_acc[cls_idx]:.4f} "
                                        f"count={int(epoch_per_class_count[cls_idx])}")
                        logger.info(f"Prediction distribution (model output):")
                        for cls_idx in range(4):
                            logger.info(f"  {action_names[cls_idx]:16s}: pred={pred_ratios[cls_idx]:.4f} "
                                        f"target={target_ratios[cls_idx]:.4f}")
                        logger.info(f"Total Batches:        {epoch_steps}")
                        # 数据集过滤统计（skip_episodes_with_empty_instruction 相关）
                        skipped_empty = getattr(dataset, '_skipped_empty_instruction', 0)
                        skipped_default = getattr(dataset, '_skipped_default_instruction', 0)
                        if skipped_empty > 0 or skipped_default > 0:
                            logger.info(f"Filtered episodes:    empty_instruction={skipped_empty}, "
                                        f"default_instruction={skipped_default}")

                        if len(grad_norms) > 0:
                            logger.info(f"Grad Norm - Mean: {np.mean(grad_norms):.4e}, Std: {np.std(grad_norms):.4e}")
                            logger.info(f"Grad Norm - Min: {np.min(grad_norms):.4e}, Max: {np.max(grad_norms):.4e}")

                        if len(epoch_losses) > 1:
                            loss_change = epoch_losses[-1] - epoch_losses[-2]
                            loss_change_pct = 100.0 * loss_change / epoch_losses[-2]
                            logger.info(f"Loss change from prev epoch: {loss_change:+.4f} ({loss_change_pct:+.2f}%)")

                        if avg_epoch_loss < best_loss:
                            best_loss = avg_epoch_loss
                            logger.info(f"★ New best loss: {best_loss:.4f}")

                        logger.info(f"{'='*80}\n")

                        if writer is not None:
                            writer.add_scalar("epoch_loss", avg_epoch_loss, epoch + 1)
                            writer.add_scalar("epoch_accuracy", avg_epoch_acc, epoch + 1)
                            writer.add_scalar("epoch_macro_accuracy", macro_acc, epoch + 1)
                            writer.add_scalar("epoch_stop_recall", stop_recall, epoch + 1)
                            for cls_idx in range(4):
                                writer.add_scalar(f"per_class_acc/{action_names[cls_idx]}",
                                                  per_class_acc[cls_idx], epoch + 1)
                            for cls_idx in range(4):
                                writer.add_scalar(f"pred_ratio/{action_names[cls_idx]}",
                                                  pred_ratios[cls_idx], epoch + 1)

                        # 清空临时统计
                        grad_norms = []

                    # 学习率调度器步进（每个epoch结束后）
                    if self.lr_scheduler is not None:
                        self.lr_scheduler.step()
                        if self._is_rank0():
                            current_lr = self.optimizer.param_groups[0]['lr']
                            if writer is not None:
                                writer.add_scalar("learning_rate", current_lr, epoch + 1)

                    # Checkpoint保存（包含完整训练状态）
                    if self.should_checkpoint() and self._is_rank0():
                        checkpoint_name = f"ckpt.epoch_{epoch + 1}.step_{step_id}.pth"
                        # logger.info(f"Saving checkpoint: {checkpoint_name}")
                        self.save_checkpoint(
                            epoch=epoch,
                            step_id=step_id,
                            model_state_dict=self._get_model_state_dict(),
                            optimizer_state_dict=self.optimizer.state_dict(),
                            best_loss=best_loss,
                            file_name=checkpoint_name
                        )
                
                # 保存最终checkpoint
                if self._is_rank0():
                    final_checkpoint_name = "ckpt.final.pth"
                    # logger.info(f"Saving final checkpoint: {final_checkpoint_name}")
                    self.save_checkpoint(
                        epoch=epochs - 1,
                        step_id=step_id,
                        model_state_dict=self._get_model_state_dict(),
                        optimizer_state_dict=self.optimizer.state_dict(),
                        best_loss=best_loss,
                        file_name=final_checkpoint_name
                    )
                    
                    # 训练总结 - 已注释以加快训练
                    # logger.info(f"\n{'='*80}")
                    # logger.info(f"TRAINING COMPLETED")
                    # logger.info(f"{'='*80}")
                    # logger.info(f"Total epochs: {epochs}")
                    # logger.info(f"Total steps: {step_id}")
                    # logger.info(f"Best loss: {best_loss:.4f}")
                    # if len(epoch_losses) > 0:
                    #     logger.info(f"Final loss: {epoch_losses[-1]:.4f}")
                    #     logger.info(f"Loss improvement: {epoch_losses[0] - epoch_losses[-1]:.4f}")
                    # logger.info(f"{'='*80}\n")
                
                AuxLosses.deactivate()
        
        finally:
            # 清理分布式进程组
            if self._is_distributed:
                import torch.distributed as dist
                try:
                    if dist.is_initialized():
                        dist.barrier()
                        dist.destroy_process_group()
                except:
                    pass
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

