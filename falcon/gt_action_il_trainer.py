#!/usr/bin/env python3

"""
基于gt_action的模仿学习训练器

这个模块实现了基于ground truth action的模仿学习框架，
参考DAgger trainer的设计，但适配了Falcon项目的gt_action传感器。

主要特性：
- 支持预收集的专家轨迹数据训练
- 兼容Falcon的gt_action传感器
- 支持拐点权重和课程学习
- 集成TensorBoard日志记录

作者: 基于DAgger框架改进
"""

import gc
import os
import random
import warnings
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Any

import lmdb
import msgpack_numpy
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from habitat import logger
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
)
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from habitat_baselines.utils.common import batch_obs
from gym import spaces

from falcon.vln_sensors import GT_ActionSensor


class GT_ActionObservationsDict(dict):
    """支持内存固定的观察字典"""
    def pin_memory(self):
        for k, v in self.items():
            self[k] = v.pin_memory()
        return self


def gt_action_collate_fn(batch):
    """
    为gt_action数据定制的批处理函数
    
    每个样本: (obs, prev_actions, gt_actions, weights)
    """
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
    gt_actions_batch = list(transposed[2])
    weights_batch = list(transposed[3])
    B = len(prev_actions_batch)

    # 重组观察数据
    new_observations_batch = defaultdict(list)
    for sensor in observations_batch[0]:
        for bid in range(B):
            new_observations_batch[sensor].append(
                observations_batch[bid][sensor]
            )

    observations_batch = new_observations_batch

    # 计算最大轨迹长度并填充
    max_traj_len = max(ele.size(0) for ele in prev_actions_batch)
    for bid in range(B):
        for sensor in observations_batch:
            observations_batch[sensor][bid] = _pad_helper(
                observations_batch[sensor][bid], max_traj_len, fill_val=1.0
            )

        prev_actions_batch[bid] = _pad_helper(
            prev_actions_batch[bid], max_traj_len
        )
        gt_actions_batch[bid] = _pad_helper(
            gt_actions_batch[bid], max_traj_len
        )
        weights_batch[bid] = _pad_helper(weights_batch[bid], max_traj_len)

    # 堆叠数据
    for sensor in observations_batch:
        observations_batch[sensor] = torch.stack(
            observations_batch[sensor], dim=1
        )
        observations_batch[sensor] = observations_batch[sensor].view(
            -1, *observations_batch[sensor].size()[2:]
        )

    prev_actions_batch = torch.stack(prev_actions_batch, dim=1)
    gt_actions_batch = torch.stack(gt_actions_batch, dim=1)
    weights_batch = torch.stack(weights_batch, dim=1)
    
    # 创建未完成掩码
    not_done_masks = torch.ones_like(
        gt_actions_batch, dtype=torch.uint8
    )
    not_done_masks[0] = 0

    observations_batch = GT_ActionObservationsDict(observations_batch)

    return (
        observations_batch,
        prev_actions_batch.view(-1, 1),
        not_done_masks.view(-1, 1),
        gt_actions_batch,
        weights_batch,
    )


class GT_ActionTrajectoryDataset(torch.utils.data.Dataset):
    """
    基于gt_action的轨迹数据集
    
    支持从LMDB或内存中加载专家轨迹数据
    """
    
    def __init__(
        self,
        data_source: str,  # LMDB路径或数据列表
        use_iw: bool = True,
        inflection_weight_coef: float = 1.0,
        lmdb_map_size: int = 1e9,
        batch_size: int = 1,
        action_space_size: int = 4,  # 动作空间大小
    ):
        super().__init__()
        self.data_source = data_source
        self.use_iw = use_iw
        self.inflection_weight_coef = inflection_weight_coef
        self.lmdb_map_size = lmdb_map_size
        self.batch_size = batch_size
        self.action_space_size = action_space_size
        
        # 设置拐点权重
        if use_iw:
            self.inflec_weights = torch.tensor([1.0, inflection_weight_coef])
        else:
            self.inflec_weights = torch.tensor([1.0, 1.0])
        
        # 初始化数据
        self._init_data()
    
    def _init_data(self):
        """初始化数据源"""
        if os.path.isdir(self.data_source):
            # LMDB数据源
            with lmdb.open(
                self.data_source,
                map_size=int(self.lmdb_map_size),
                readonly=True,
                lock=False,
            ) as lmdb_env:
                self.length = lmdb_env.stat()["entries"]
                self.use_lmdb = True
        else:
            # 内存数据源（用于测试）
            self.trajectories = self._load_trajectories_from_memory()
            self.length = len(self.trajectories)
            self.use_lmdb = False
    
    def _load_trajectories_from_memory(self) -> List[Tuple]:
        """从内存加载轨迹数据（示例实现）"""
        # 这里应该加载你的实际轨迹数据
        # 返回格式: [(obs, prev_actions, gt_actions), ...]
        trajectories = []
        
        # 示例数据 - 实际使用时替换为你的数据加载逻辑
        for i in range(100):  # 假设有100个轨迹
            traj_len = random.randint(10, 50)
            obs = {
                'rgb': torch.randn(traj_len, 3, 224, 224),
                'depth': torch.randn(traj_len, 1, 224, 224),
                'instruction': torch.randn(traj_len, 512),
            }
            prev_actions = torch.randint(0, self.action_space_size, (traj_len,))
            gt_actions = torch.randint(0, self.action_space_size, (traj_len,))
            
            trajectories.append((obs, prev_actions, gt_actions))
        
        return trajectories
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        """获取单个轨迹样本"""
        if self.use_lmdb:
            return self._get_item_from_lmdb(idx)
        else:
            return self._get_item_from_memory(idx)
    
    def _get_item_from_lmdb(self, idx):
        """从LMDB获取数据"""
        with lmdb.open(
            self.data_source,
            map_size=int(self.lmdb_map_size),
            readonly=True,
            lock=False,
        ) as lmdb_env, lmdb_env.begin(buffers=True) as txn:
            data = msgpack_numpy.unpackb(
                txn.get(str(idx).encode()),
                raw=False,
            )
        
        obs, prev_actions, gt_actions = data
        
        # 转换为tensor
        for k, v in obs.items():
            obs[k] = torch.from_numpy(np.copy(v))
        
        prev_actions = torch.from_numpy(np.copy(prev_actions))
        gt_actions = torch.from_numpy(np.copy(gt_actions))
        
        # 计算拐点权重
        inflections = torch.cat([
            torch.tensor([1], dtype=torch.long),
            (gt_actions[1:] != gt_actions[:-1]).long(),
        ])
        
        return (
            obs,
            prev_actions,
            gt_actions,
            self.inflec_weights[inflections],
        )
    
    def _get_item_from_memory(self, idx):
        """从内存获取数据"""
        obs, prev_actions, gt_actions = self.trajectories[idx]
        
        # 计算拐点权重
        inflections = torch.cat([
            torch.tensor([1], dtype=torch.long),
            (gt_actions[1:] != gt_actions[:-1]).long(),
        ])
        
        return (
            obs,
            prev_actions,
            gt_actions,
            self.inflec_weights[inflections],
        )


@baseline_registry.register_trainer(name="gt_action_il")
class GT_ActionILTrainer:
    """
    基于gt_action的模仿学习训练器
    
    参考DAgger框架设计，但专门适配gt_action传感器
    """
    
    def __init__(self, config=None):
        self.config = config
        self.device = (
            torch.device("cuda", config.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.policy = None
        self.optimizer = None
        self.obs_transforms = []
        
        # 初始化数据路径
        self.lmdb_features_dir = config.IL.GT_ACTION_IL.lmdb_features_dir.format(
            split=config.TASK_CONFIG.DATASET.SPLIT
        )
        
        logger.info(f"GT Action IL Trainer initialized on device: {self.device}")
    
    def _make_dirs(self) -> None:
        """创建必要的目录"""
        os.makedirs(self.config.CHECKPOINT_FOLDER, exist_ok=True)
        os.makedirs(self.lmdb_features_dir, exist_ok=True)
        if self.config.EVAL.SAVE_RESULTS:
            os.makedirs(self.config.RESULTS_DIR, exist_ok=True)
    
    def _initialize_policy(
        self,
        config,
        load_from_ckpt: bool,
        observation_space: spaces.Space,
        action_space: spaces.Space,
    ) -> None:
        """初始化策略网络"""
        policy = baseline_registry.get_policy(config.MODEL.policy_name)
        self.policy = policy.from_config(
            config=config,
            observation_space=observation_space,
            action_space=action_space,
        )
        self.policy.to(self.device)
        
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=config.IL.lr
        )
        
        if load_from_ckpt:
            ckpt_path = config.IL.ckpt_to_load
            ckpt_dict = self.load_checkpoint(ckpt_path, map_location="cpu")
            self.policy.load_state_dict(ckpt_dict["state_dict"])
            if config.IL.is_requeue:
                self.optimizer.load_state_dict(ckpt_dict["optim_state"])
            logger.info(f"Loaded weights from checkpoint: {ckpt_path}")
        
        params = sum(param.numel() for param in self.policy.parameters())
        params_t = sum(
            p.numel() for p in self.policy.parameters() if p.requires_grad
        )
        logger.info(f"Agent parameters: {params}. Trainable: {params_t}")
        logger.info("Finished setting up policy.")
    
    def _get_spaces(self, config, envs=None):
        """获取观察空间和动作空间"""
        if envs is not None:
            observation_space = envs.observation_spaces[0]
            action_space = envs.action_spaces[0]
        else:
            env = get_env_class(config.ENV_NAME)(config=config)
            observation_space = env.observation_space
            action_space = env.action_space
        
        # 应用观察变换
        from habitat_baselines.common.obs_transformers import (
            get_active_obs_transforms,
            apply_obs_transforms_obs_space,
        )
        self.obs_transforms = get_active_obs_transforms(config)
        observation_space = apply_obs_transforms_obs_space(
            observation_space, self.obs_transforms
        )
        
        return observation_space, action_space
    
    def _update_agent(
        self,
        observations,
        prev_actions,
        not_done_masks,
        gt_actions,
        weights,
    ):
        """更新智能体参数"""
        T, N = gt_actions.size()
        
        # 初始化循环隐藏状态
        recurrent_hidden_states = torch.zeros(
            N,
            self.policy.net.num_recurrent_layers,
            self.config.MODEL.STATE_ENCODER.hidden_size,
            device=self.device,
        )
        
        # 前向传播
        outputs = self.policy.net(
            observations,
            recurrent_hidden_states,
            prev_actions,
            not_done_masks,
        )
        
        # 计算损失
        logits = outputs["logits"]
        logits = logits.view(T, N, -1)
        gt_actions = gt_actions.view(T, N)
        weights = weights.view(T, N)
        
        # 交叉熵损失
        action_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            gt_actions.view(-1),
            reduction="none",
        )
        
        # 应用权重
        action_loss = (action_loss * weights.view(-1)).mean()
        
        # 总损失
        total_loss = action_loss
        
        # 反向传播
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return total_loss.item(), action_loss.item(), 0.0
    
    def save_checkpoint(self, file_name: str) -> None:
        """保存检查点"""
        checkpoint = {
            "state_dict": self.policy.state_dict(),
            "optim_state": self.optimizer.state_dict(),
            "config": self.config,
        }
        torch.save(
            checkpoint, os.path.join(self.config.CHECKPOINT_FOLDER, file_name)
        )
    
    def load_checkpoint(self, checkpoint_path, *args, **kwargs) -> Dict:
        """加载检查点"""
        return torch.load(checkpoint_path, *args, **kwargs)
    
    def train(self) -> None:
        """主训练循环"""
        logger.info("Starting GT Action IL Training")
        
        # 创建目录
        self._make_dirs()
        
        # 获取空间
        observation_space, action_space = self._get_spaces(self.config)
        
        # 初始化策略
        self._initialize_policy(
            self.config,
            self.config.IL.load_from_ckpt,
            observation_space=observation_space,
            action_space=action_space,
        )
        
        # 创建数据集
        dataset = GT_ActionTrajectoryDataset(
            self.lmdb_features_dir,
            use_iw=self.config.IL.use_iw,
            inflection_weight_coef=self.config.IL.inflection_weight_coef,
            lmdb_map_size=self.config.IL.GT_ACTION_IL.lmdb_map_size,
            batch_size=self.config.IL.batch_size,
            action_space_size=action_space.n,
        )
        
        # 创建数据加载器
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config.IL.batch_size,
            shuffle=True,
            collate_fn=gt_action_collate_fn,
            pin_memory=False,
            drop_last=True,
            num_workers=2,
        )
        
        # 训练循环
        with TensorboardWriter(
            self.config.TENSORBOARD_DIR,
            flush_secs=30,
            purge_step=0,
        ) as writer:
            
            for epoch in tqdm.trange(self.config.IL.epochs, dynamic_ncols=True):
                epoch_loss = 0.0
                epoch_action_loss = 0.0
                
                for batch_idx, batch in enumerate(tqdm.tqdm(
                    dataloader,
                    leave=False,
                    dynamic_ncols=True,
                )):
                    (
                        observations_batch,
                        prev_actions_batch,
                        not_done_masks,
                        gt_actions_batch,
                        weights_batch,
                    ) = batch
                    
                    # 移动到设备
                    observations_batch = {
                        k: v.to(
                            device=self.device,
                            dtype=torch.float32,
                            non_blocking=True,
                        )
                        for k, v in observations_batch.items()
                    }
                    
                    prev_actions_batch = prev_actions_batch.to(
                        device=self.device, non_blocking=True
                    )
                    not_done_masks = not_done_masks.to(
                        device=self.device, non_blocking=True
                    )
                    gt_actions_batch = gt_actions_batch.to(
                        device=self.device, non_blocking=True
                    )
                    weights_batch = weights_batch.to(
                        device=self.device, non_blocking=True
                    )
                    
                    # 更新智能体
                    loss, action_loss, aux_loss = self._update_agent(
                        observations_batch,
                        prev_actions_batch,
                        not_done_masks,
                        gt_actions_batch,
                        weights_batch,
                    )
                    
                    epoch_loss += loss
                    epoch_action_loss += action_loss
                    
                    # 记录日志
                    if batch_idx % 10 == 0:
                        logger.info(f"Epoch {epoch}, Batch {batch_idx}")
                        logger.info(f"Loss: {loss:.4f}, Action Loss: {action_loss:.4f}")
                        
                        writer.add_scalar("train/loss", loss, epoch * len(dataloader) + batch_idx)
                        writer.add_scalar("train/action_loss", action_loss, epoch * len(dataloader) + batch_idx)
                
                # 保存检查点
                avg_loss = epoch_loss / len(dataloader)
                avg_action_loss = epoch_action_loss / len(dataloader)
                
                logger.info(f"Epoch {epoch} completed")
                logger.info(f"Average Loss: {avg_loss:.4f}")
                logger.info(f"Average Action Loss: {avg_action_loss:.4f}")
                
                writer.add_scalar("train/epoch_loss", avg_loss, epoch)
                writer.add_scalar("train/epoch_action_loss", avg_action_loss, epoch)
                
                # 保存检查点
                self.save_checkpoint(f"gt_action_il_epoch_{epoch}.pth")
        
        logger.info("GT Action IL Training completed!")


def create_gt_action_il_config():
    """创建GT Action IL训练配置"""
    config = {
        "IL": {
            "lr": 2.5e-4,
            "epochs": 10,
            "batch_size": 4,
            "use_iw": True,
            "inflection_weight_coef": 1.0,
            "load_from_ckpt": False,
            "is_requeue": False,
            "GT_ACTION_IL": {
                "lmdb_features_dir": "/path/to/gt_action_data/{split}",
                "lmdb_map_size": 1e9,
            }
        },
        "MODEL": {
            "policy_name": "PointNavResNetPolicy",
            "STATE_ENCODER": {
                "hidden_size": 512,
            }
        },
        "TASK_CONFIG": {
            "DATASET": {
                "SPLIT": "train",
            }
        },
        "CHECKPOINT_FOLDER": "/path/to/checkpoints",
        "TENSORBOARD_DIR": "/path/to/tensorboard",
        "TORCH_GPU_ID": 0,
    }
    return config


if __name__ == "__main__":
    # 示例使用
    config = create_gt_action_il_config()
    trainer = GT_ActionILTrainer(config)
    trainer.train()

