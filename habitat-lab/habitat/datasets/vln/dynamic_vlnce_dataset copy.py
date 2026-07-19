#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import gzip
import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import attr
import numpy as np

from habitat.core.dataset import Episode, Dataset
from habitat.core.registry import registry
from habitat.core.utils import DatasetFloatJSONEncoder
from habitat.datasets.pointnav.pointnav_dataset import PointNavDatasetV1
from habitat.datasets.utils import check_and_gen_physics_config
from habitat.tasks.nav.nav import NavigationGoal
from habitat.tasks.vln.vln import InstructionData, VLNEpisode

if TYPE_CHECKING:
    from omegaconf import DictConfig


@attr.s(auto_attribs=True, kw_only=True)
class DynamicVLNCEEpisode(VLNEpisode):
    r"""Specifies additional fields for Dynamic VLN-CE episodes.
    
    :property instruction: Natural language instruction for navigation
    :property gt_action: Ground truth action sequence for imitation learning
    :property instruction_tokens: Tokenized instruction for processing
    :property instruction_source: Source identifier for the instruction
    :property episode_id: Unique identifier for the episode (index for random sampling)
    :property original_episode_id: Original episode ID from dataset (for accessing collect_data)
    :property scene_id: Scene identifier
    :property start_position: Starting position [x, y, z]
    :property start_rotation: Starting rotation [x, y, z, w]
    :property goals: List of navigation goals
    :property trajectory_id: Trajectory identifier
    :property instruction_id: Instruction identifier
    :property path: List of waypoints in the trajectory
    :property distance_to_goal: Distance to the goal
    :property reference_path: Reference path for evaluation
    :property info: Additional episode information
    """
    
    # VLN specific fields
    instruction: str
    instruction_tokens: List[str] = []
    instruction_id: int = 0
    instruction_source: str = ""  # 指令来源标识
    
    # Dynamic VLN-CE specific fields
    gt_action: List[int] = []
    trajectory_id: str = ""
    original_episode_id: str = ""  # 保存原始episode_id，用于访问collect_data目录
    
    # Additional fields for compatibility
    path: List[List[float]] = []
    distance_to_goal: float = 0.0
    reference_path: List[List[float]] = []
    info: Dict[str, Any] = {}


@registry.register_dataset(name="DynamicVLNCE-v1")
class DynamicVLNCEDatasetV1(PointNavDatasetV1):
    r"""Class inherited from PointNavDataset that loads Dynamic VLN-CE dataset.
    
    This dataset extends the original Falcon dataset with VLN capabilities,
    including natural language instructions and ground truth actions for
    imitation learning.
    """
    
    episodes: List[DynamicVLNCEEpisode] = []  # type: ignore
    content_scenes_path: str = "{data_path}/{scene}.json.gz"
    instruction_vocab: Optional[Dict[str, int]] = None

    def to_json(self) -> str:
        """Serialize the dataset to JSON format."""
        result = DatasetFloatJSONEncoder().encode(self)
        return result

    def __init__(self, config: Optional["DictConfig"] = None) -> None:
        self.config = config
        self.episodes = []
        self.instruction_vocab = None

        if config is None:
            return

        if not self.check_config_paths_exist(config):
            raise ValueError(
                f"Requested DynamicVLNCEDataset config paths '{config.data_path}' or '{config.scenes_dir}' are not downloaded locally. Aborting."
            )

        check_and_gen_physics_config()
        
        # Load dataset files directly without calling parent __init__
        self._load_dataset_files(config)

    @classmethod
    def check_config_paths_exist(cls, config: "DictConfig") -> bool:
        """Check if dataset files exist for the given config."""
        import os
        import glob
        
        # For scene-based datasets, check if the directory exists and has files
        dataset_dir = os.path.dirname(config.data_path)
        if not os.path.exists(dataset_dir):
            return False
            
        # Check if there are any .json.gz files in the directory
        pattern = os.path.join(dataset_dir, "*.json.gz")
        files = glob.glob(pattern)
        return len(files) > 0 and os.path.exists(config.scenes_dir)

    @classmethod
    def get_scenes_to_load(cls, config: "DictConfig") -> List[str]:
        """Return list of scene ids for which dataset has separate files with episodes."""
        import os
        import glob
        
        dataset_dir = os.path.dirname(config.data_path)
        if not cls.check_config_paths_exist(config):
            raise FileNotFoundError(
                f"Could not find dataset directory `{dataset_dir}`"
            )

        # Get all scene files from the dataset directory
        pattern = os.path.join(dataset_dir, "*.json.gz")
        scene_files = glob.glob(pattern)
        
        # Extract scene IDs from filenames
        scene_ids = []
        for file_path in scene_files:
            filename = os.path.basename(file_path)
            scene_id = filename.replace('.json.gz', '')
            scene_ids.append(scene_id)
        
        return scene_ids

    def _load_dataset_files(self, config: "DictConfig") -> None:
        """Load dataset files for all scenes."""
        import os
        import glob
        
        dataset_dir = os.path.dirname(config.data_path)
        
        # Get all scene files from the dataset directory
        pattern = os.path.join(dataset_dir, "*.json.gz")
        scene_files = glob.glob(pattern)
        
        # print(f"Found {len(scene_files)} scene files in {dataset_dir}")
        
        # Load each scene file
        for scene_file in scene_files:
            # print(f"Loading scene file: {scene_file}")
            self._load_from_file(scene_file, config.scenes_dir)
            # print(f"Loaded {len(self.episodes)} episodes so far")
        
        # print(f"Total episodes loaded: {len(self.episodes)}")
        
        # Filter episodes based on content_scenes if specified
        if config.content_scenes and config.content_scenes != ["*"]:
            # Extract scene names from scene_ids for filtering
            scene_names = set()
            for ep in self.episodes:
                # Extract scene name from scene_id like "hm3d/train/00092-1mCzDx3EMom/1mCzDx3EMom.basis.glb"
                scene_id_parts = ep.scene_id.split('/')
                if len(scene_id_parts) >= 3:
                    scene_name = scene_id_parts[2].split('-')[1] if '-' in scene_id_parts[2] else scene_id_parts[2]
                    scene_names.add(scene_name)
            
            self.episodes = [
                ep for ep in self.episodes 
                if any(scene_name in ep.scene_id for scene_name in config.content_scenes)
            ]
            # print(f"After filtering: {len(self.episodes)} episodes")

    def _load_from_file(self, fname: str, scenes_dir: str) -> None:
        """Load episodes from a single scene file."""
        import gzip
        
        if fname.endswith(".json.gz"):
            with gzip.open(fname, "rt") as f:
                json_str = f.read()
                self.from_json(json_str, scenes_dir)
        elif fname.endswith(".json"):
            with open(fname, "r") as f:
                json_str = f.read()
                self.from_json(json_str, scenes_dir)
        else:
            raise ValueError(f"Unsupported file format: {fname}")

    def from_json(
        self, json_str: str, scenes_dir: Optional[str] = None
    ) -> None:
        """Load episodes from JSON string."""
        deserialized = json.loads(json_str)

        # num_episodes = len(deserialized.get('episodes', []))
        # print(f"[DynamicVLNCE] Loading {num_episodes} episodes from JSON")

        # Load instruction vocabulary if available
        if "instruction_vocab" in deserialized:
            self.instruction_vocab = deserialized["instruction_vocab"]
        else:
            self.instruction_vocab = None

        for i, episode in enumerate(deserialized["episodes"]):
            # Create DynamicVLNCEEpisode (instruction_source is now supported)
            dynamic_vlnce_episode = DynamicVLNCEEpisode(**episode)
            # 保存原始episode_id（用于访问collect_data目录）
            original_id = episode.get("episode_id", str(i))
            dynamic_vlnce_episode.original_episode_id = str(original_id)
            # 使用索引作为episode_id（方便随机采样）
            dynamic_vlnce_episode.episode_id = str(i)
            
            # 打印前5个episode的详细信息，方便调试
            # gt_action_from_json = episode.get("gt_action", [])
            # gt_action_len = len(gt_action_from_json)
            # ep_gt_action = getattr(dynamic_vlnce_episode, 'gt_action', None)
            # ep_gt_len = len(ep_gt_action) if ep_gt_action else 0
            # if i < 5:
            #     print(f"  [Dataset Load] Episode {i}: orig_id={original_id}, json_gt_len={gt_action_len}, ep_gt_len={ep_gt_len}, first_5={gt_action_from_json[:5] if gt_action_from_json else []}", flush=True)
            
            # Fix scene_id format - convert from hm3d/train/00092-1mCzDx3EMom/1mCzDx3EMom.basis.glb
            # to train/00092-1mCzDx3EMom/1mCzDx3EMom.basis.glb
            if dynamic_vlnce_episode.scene_id.startswith("hm3d/"):
                dynamic_vlnce_episode.scene_id = dynamic_vlnce_episode.scene_id[5:]  # Remove "hm3d/" prefix
            
            # Build complete scene path like pointnav_dataset.py does
            if scenes_dir is not None:
                if dynamic_vlnce_episode.scene_id.startswith("data/scene_datasets/"):
                    dynamic_vlnce_episode.scene_id = dynamic_vlnce_episode.scene_id[
                        len("data/scene_datasets/"):
                    ]
                dynamic_vlnce_episode.scene_id = os.path.join(scenes_dir, dynamic_vlnce_episode.scene_id)

            # Process instruction if it's a string
            if isinstance(dynamic_vlnce_episode.instruction, str):
                # Tokenize instruction (simple whitespace tokenization)
                dynamic_vlnce_episode.instruction_tokens = dynamic_vlnce_episode.instruction.split()
            elif isinstance(dynamic_vlnce_episode.instruction, dict):
                # Handle structured instruction data
                instruction_data = InstructionData(**dynamic_vlnce_episode.instruction)
                dynamic_vlnce_episode.instruction = instruction_data.text
                dynamic_vlnce_episode.instruction_tokens = instruction_data.tokens
                dynamic_vlnce_episode.instruction_id = instruction_data.instruction_id

            # Process goals
            for g_index, goal in enumerate(dynamic_vlnce_episode.goals):
                if isinstance(goal, dict):
                    dynamic_vlnce_episode.goals[g_index] = NavigationGoal(**goal)

            # Ensure gt_action is a list of integers
            if isinstance(dynamic_vlnce_episode.gt_action, str):
                # If gt_action is stored as string, try to parse it
                try:
                    dynamic_vlnce_episode.gt_action = json.loads(dynamic_vlnce_episode.gt_action)
                except:
                    dynamic_vlnce_episode.gt_action = []
            elif not isinstance(dynamic_vlnce_episode.gt_action, list):
                dynamic_vlnce_episode.gt_action = []

            self.episodes.append(dynamic_vlnce_episode)
        
        # 后处理：确保每个唯一的episode都有v1和v2两个版本
        # 如果数据集中已经有v1和v2版本，保持原样；如果没有，创建副本
        self._ensure_dual_instruction_episodes()

    def _ensure_dual_instruction_episodes(self) -> None:
        """确保每个唯一的episode都有v1和v2两个版本
        
        根据数据集类型智能处理：
        1. 如果数据集已经混合（episode_id已有_v1/_v2后缀），直接使用现有episode
        2. 如果是v2数据集（路径包含dynamic_dataset_final_v2），只保留v2版本
        3. 如果是mixed数据集（路径包含v1v2_mixed），应该已经有_v1/_v2后缀，直接使用
        4. 其他情况，创建v1和v2两个版本
        """
        import copy
        
        # 检测数据集类型
        dataset_type = "unknown"
        data_path = ""
        if self.config is not None:
            data_path = str(self.config.get("data_path", ""))
            if "v1v2_mixed" in data_path or "v1v2_mixed" in data_path.lower():
                dataset_type = "mixed"
            elif "dynamic_dataset_final_v2" in data_path or "final_v2" in data_path:
                dataset_type = "v2_only"
            elif "dynamic_dataset_final_v1" in data_path or "final_v1" in data_path:
                dataset_type = "v1_only"
        
        # 首先检查数据集是否已经混合（episode_id已经有_v1/_v2后缀）
        has_v1_v2_suffix = False
        for ep in self.episodes:
            orig_id = str(ep.original_episode_id)
            if orig_id.endswith('_v1') or orig_id.endswith('_v2'):
                has_v1_v2_suffix = True
                break
        
        # 如果数据集已经混合，直接使用现有episode（不需要创建副本）
        if has_v1_v2_suffix:
            # 只需要重新分配episode_id
            for i, ep in enumerate(self.episodes):
                ep.episode_id = str(i)
            
            v1_count = sum(1 for ep in self.episodes if str(ep.original_episode_id).endswith('_v1'))
            v2_count = sum(1 for ep in self.episodes if str(ep.original_episode_id).endswith('_v2'))
            # print(f"[DynamicVLNCE Dataset] Dataset already mixed: {len(self.episodes)} episodes ({v1_count} v1, {v2_count} v2)", flush=True)
            return
        
        # 如果数据集未混合，根据数据集类型处理
        if dataset_type == "v2_only":
            # v2数据集：只保留v2版本，不创建v1版本
            for ep in self.episodes:
                # 为每个episode添加v2标识
                ep.original_episode_id = f"{ep.original_episode_id}_v2"
                ep.instruction_source = "v2"
            
            # 重新分配episode_id
            for i, ep in enumerate(self.episodes):
                ep.episode_id = str(i)

            # print(f"[DynamicVLNCE Dataset] v2-only dataset: {len(self.episodes)} episodes (all v2)", flush=True)
            return
        
        elif dataset_type == "v1_only":
            # v1数据集：只保留v1版本，不创建v2版本
            for ep in self.episodes:
                # 为每个episode添加v1标识
                ep.original_episode_id = f"{ep.original_episode_id}_v1"
                ep.instruction_source = "v1"
            
            # 重新分配episode_id
            for i, ep in enumerate(self.episodes):
                ep.episode_id = str(i)

            # print(f"[DynamicVLNCE Dataset] v1-only dataset: {len(self.episodes)} episodes (all v1)", flush=True)
            return
        
        # 默认情况：为每个episode创建v1和v2两个版本
        # 按original_episode_id分组，去掉_v1/_v2后缀
        episode_groups = {}
        for ep in self.episodes:
            base_id = str(ep.original_episode_id).rstrip('_v1').rstrip('_v2')
            if base_id not in episode_groups:
                episode_groups[base_id] = {'v1': None, 'v2': None, 'base': None}
            
            orig_id = str(ep.original_episode_id)
            if orig_id.endswith('_v1'):
                episode_groups[base_id]['v1'] = ep
            elif orig_id.endswith('_v2'):
                episode_groups[base_id]['v2'] = ep
            else:
                episode_groups[base_id]['base'] = ep
        
        # 为每个基础episode确保有v1和v2版本
        new_episodes = []
        for base_id, group in episode_groups.items():
            v1_ep = group['v1']
            v2_ep = group['v2']
            base_ep = group['base']
            
            # 如果v1和v2都存在，直接添加
            if v1_ep is not None and v2_ep is not None:
                new_episodes.append(v1_ep)
                new_episodes.append(v2_ep)
            else:
                # 需要创建缺失的版本
                # 找到基础episode（优先使用base，否则使用v1或v2）
                source_ep = base_ep
                if source_ep is None:
                    source_ep = v1_ep if v1_ep is not None else v2_ep
                
                if source_ep is None:
                    continue  # 跳过无效的episode
                
                # 创建v1版本（如果缺失）
                if v1_ep is None:
                    v1_ep = copy.deepcopy(source_ep)
                    v1_ep.original_episode_id = f"{base_id}_v1"
                    v1_ep.instruction_source = "v1"
                    new_episodes.append(v1_ep)
                else:
                    new_episodes.append(v1_ep)
                
                # 创建v2版本（如果缺失）
                if v2_ep is None:
                    v2_ep = copy.deepcopy(source_ep)
                    v2_ep.original_episode_id = f"{base_id}_v2"
                    v2_ep.instruction_source = "v2"
                    new_episodes.append(v2_ep)
                else:
                    new_episodes.append(v2_ep)
        
        # 更新episodes列表
        self.episodes = new_episodes
        # 重新分配episode_id（使用连续索引）
        for i, ep in enumerate(self.episodes):
            ep.episode_id = str(i)
        
        # 打印统计信息
        # v1_count = sum(1 for ep in self.episodes if str(ep.original_episode_id).endswith('_v1'))
        # v2_count = sum(1 for ep in self.episodes if str(ep.original_episode_id).endswith('_v2'))
        # print(f"[DynamicVLNCE Dataset] After ensuring dual instructions: {len(self.episodes)} episodes ({v1_count} v1, {v2_count} v2)", flush=True)

        # 打印处理后的前5个episode，方便对比
        # if len(self.episodes) > 0:
        #     print(f"[DynamicVLNCE Dataset] First 5 episodes after processing:", flush=True)
        #     for i, ep in enumerate(self.episodes[:5]):
        #         orig_id = getattr(ep, 'original_episode_id', '')
        #         inst_source = getattr(ep, 'instruction_source', '')
        #         print(f"  [{i}] episode_id={ep.episode_id}, original_episode_id={orig_id}, instruction_source={inst_source}", flush=True)

    def to_binary(self) -> Dict[str, Any]:
        """
        Serialize the dataset to a pickle compatible Dict.
        """
        def access_idx(k, name_to_idx):
            if len(name_to_idx) == 0:
                name_to_idx[k] = 0
            if k not in name_to_idx:
                name_to_idx[k] = max(name_to_idx.values()) + 1
            return name_to_idx[k]

        def encode_name_dict(d, name_to_idx):
            ret_d = {}
            for k, v in d.items():
                ret_d[access_idx(k, name_to_idx)] = v
            return ret_d

        all_transforms: List[Any] = []
        name_to_idx: Dict[str, int] = {}
        all_eps = []

        for ep in self.episodes:
            # Convert episode to binary format
            ep_dict = {
                "episode_id": ep.episode_id,
                "original_episode_id": ep.original_episode_id,  # 保存原始episode_id
                "scene_id": access_idx(ep.scene_id, name_to_idx),
                "start_position": ep.start_position,
                "start_rotation": ep.start_rotation,
                "goals": ep.goals,
                "instruction": ep.instruction,
                "instruction_tokens": ep.instruction_tokens,
                "instruction_id": ep.instruction_id,
                "instruction_source": ep.instruction_source,
                "gt_action": ep.gt_action,
                "trajectory_id": ep.trajectory_id,
                "path": ep.path,
                "distance_to_goal": ep.distance_to_goal,
                "reference_path": ep.reference_path,
                "info": ep.info,
            }
            all_eps.append(ep_dict)

        return {
            "episodes": all_eps,
            "instruction_vocab": self.instruction_vocab,
            "name_to_idx": name_to_idx,
        }

    def get_instruction_vocab(self) -> Dict[str, int]:
        """Get instruction vocabulary."""
        if self.instruction_vocab is not None:
            return self.instruction_vocab
        
        # Build vocabulary from all instructions
        vocab = {"<pad>": 0, "<unk>": 1, "<sos>": 2, "<eos>": 3}
        vocab_idx = 4
        
        for episode in self.episodes:
            for token in episode.instruction_tokens:
                if token not in vocab:
                    vocab[token] = vocab_idx
                    vocab_idx += 1
        
        self.instruction_vocab = vocab
        return vocab

    def get_episodes_by_scene(self, scene_id: str) -> List[DynamicVLNCEEpisode]:
        """Get all episodes for a specific scene."""
        return [ep for ep in self.episodes if ep.scene_id == scene_id]

    def get_episodes_with_instruction(self, instruction_pattern: str) -> List[DynamicVLNCEEpisode]:
        """Get episodes containing a specific instruction pattern."""
        return [
            ep for ep in self.episodes 
            if instruction_pattern.lower() in ep.instruction.lower()
        ]

    def get_episodes_with_gt_actions(self) -> List[DynamicVLNCEEpisode]:
        """Get episodes that have ground truth actions."""
        return [ep for ep in self.episodes if len(ep.gt_action) > 0]

    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        total_episodes = len(self.episodes)
        episodes_with_gt = len(self.get_episodes_with_gt_actions())
        
        # Instruction length statistics
        instruction_lengths = [len(ep.instruction_tokens) for ep in self.episodes]
        avg_instruction_length = np.mean(instruction_lengths) if instruction_lengths else 0
        
        # Action sequence length statistics
        action_lengths = [len(ep.gt_action) for ep in self.episodes if ep.gt_action]
        avg_action_length = np.mean(action_lengths) if action_lengths else 0
        
        # Scene statistics
        unique_scenes = len(set(ep.scene_id for ep in self.episodes))
        
        return {
            "total_episodes": total_episodes,
            "episodes_with_gt_actions": episodes_with_gt,
            "unique_scenes": unique_scenes,
            "average_instruction_length": avg_instruction_length,
            "average_action_length": avg_action_length,
            "instruction_vocab_size": len(self.get_instruction_vocab()),
        }


@registry.register_dataset(name="DynamicVLNCE-v2")
class DynamicVLNCEDatasetV2(DynamicVLNCEDatasetV1):
    r"""Enhanced version of DynamicVLNCE dataset with additional features."""
    
    def __init__(self, config: Optional["DictConfig"] = None) -> None:
        super().__init__(config)
        
        # Post-process episodes for additional features
        self._post_process_episodes()

    def _post_process_episodes(self) -> None:
        """Post-process episodes to add additional features."""
        for episode in self.episodes:
            # Add trajectory length if not present
            if not episode.path and episode.goals:
                # Estimate path length from start to goal
                start_pos = np.array(episode.start_position)
                goal_pos = np.array(episode.goals[0].position)
                episode.distance_to_goal = float(np.linalg.norm(start_pos - goal_pos))
            
            # Ensure trajectory_id is set
            if not episode.trajectory_id:
                episode.trajectory_id = f"{episode.scene_id}_{episode.episode_id}"
            
            # Add instruction metadata
            if not episode.info:
                episode.info = {}
            
            episode.info.update({
                "instruction_length": len(episode.instruction_tokens),
                "action_sequence_length": len(episode.gt_action),
                "has_gt_actions": len(episode.gt_action) > 0,
            })

    def filter_episodes(self, 
                       min_instruction_length: int = 0,
                       max_instruction_length: int = float('inf'),
                       min_action_length: int = 0,
                       max_action_length: int = float('inf'),
                       scenes: Optional[List[str]] = None) -> List[DynamicVLNCEEpisode]:
        """Filter episodes based on criteria."""
        filtered_episodes = []
        
        for episode in self.episodes:
            # Filter by instruction length
            if not (min_instruction_length <= len(episode.instruction_tokens) <= max_instruction_length):
                continue
            
            # Filter by action length
            if not (min_action_length <= len(episode.gt_action) <= max_action_length):
                continue
            
            # Filter by scenes
            if scenes is not None and episode.scene_id not in scenes:
                continue
            
            filtered_episodes.append(episode)
        
        return filtered_episodes

    def create_subset(self, 
                     episode_ids: List[str],
                     subset_name: str = "subset") -> "DynamicVLNCEDatasetV2":
        """Create a subset of the dataset."""
        subset_episodes = [ep for ep in self.episodes if ep.episode_id in episode_ids]
        
        # Create new dataset instance
        subset_dataset = DynamicVLNCEDatasetV2(None)
        subset_dataset.episodes = subset_episodes
        subset_dataset.instruction_vocab = self.instruction_vocab
        subset_dataset.content_scenes_path = self.content_scenes_path
        
        return subset_dataset


def _try_register_dynamic_vlnce_dataset():
    """Try to register DynamicVLNCE dataset."""
    try:
        from habitat.datasets.vln.dynamic_vlnce_dataset import (  # noqa: F401 isort:skip
            DynamicVLNCEDatasetV1,
            DynamicVLNCEDatasetV2,
        )
    except ImportError as e:
        dynamic_vlnce_import_error = e

        @registry.register_dataset(name="DynamicVLNCE-v1")
        class DynamicVLNCEDatasetImportError(Dataset):
            def __init__(self, *args, **kwargs):
                raise dynamic_vlnce_import_error

        @registry.register_dataset(name="DynamicVLNCE-v2")
        class DynamicVLNCEDatasetV2ImportError(Dataset):
            def __init__(self, *args, **kwargs):
                raise dynamic_vlnce_import_error
