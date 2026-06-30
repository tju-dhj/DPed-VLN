#!/usr/bin/env python3

"""
GT Action模仿学习集成指南

这个模块提供了如何将gt_action传感器集成到模仿学习框架中的
完整示例和最佳实践。

包含内容：
1. 数据准备和格式转换
2. 训练配置设置
3. 模型训练和评估
4. 与现有Falcon框架的集成

作者: 基于DAgger框架和Falcon项目
"""

import os
import json
import torch
import numpy as np
from typing import Dict, List, Any, Optional
from pathlib import Path

from falcon.gt_action_il_trainer import GT_ActionILTrainer
from falcon.improved_gt_action_sensors import (
    ImprovedGT_ActionSensor,
    GT_ActionDataProcessor,
    create_gt_action_sensor_config,
)


class GT_ActionILIntegration:
    """
    GT Action模仿学习集成类
    
    提供完整的集成解决方案，包括：
    - 数据准备
    - 模型训练
    - 评估和部署
    """
    
    def __init__(self, config_path: str):
        """
        初始化集成类
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.data_processor = GT_ActionDataProcessor(
            action_space_size=self.config.get('action_space_size', 4)
        )
        
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        with open(self.config_path, 'r') as f:
            return json.load(f)
    
    def prepare_training_data(
        self,
        data_dir: str,
        output_dir: str,
        split: str = "train"
    ) -> str:
        """
        准备训练数据
        
        Args:
            data_dir: 原始数据目录
            output_dir: 输出目录
            split: 数据分割 ("train", "val", "test")
            
        Returns:
            LMDB数据库路径
        """
        print(f"Preparing training data for split: {split}")
        
        # 创建输出目录
        lmdb_path = os.path.join(output_dir, split)
        os.makedirs(lmdb_path, exist_ok=True)
        
        # 这里应该实现你的数据加载逻辑
        # 示例：从JSON文件加载轨迹数据
        trajectories = self._load_trajectories_from_json(data_dir, split)
        
        # 转换为LMDB格式
        self._convert_to_lmdb(trajectories, lmdb_path)
        
        print(f"Training data prepared: {lmdb_path}")
        return lmdb_path
    
    def _load_trajectories_from_json(
        self,
        data_dir: str,
        split: str
    ) -> List[Dict[str, Any]]:
        """
        从JSON文件加载轨迹数据
        
        这里需要根据你的实际数据格式进行调整
        """
        trajectories = []
        
        # 示例数据格式 - 请根据你的实际数据调整
        json_file = os.path.join(data_dir, f"{split}.json")
        if os.path.exists(json_file):
            with open(json_file, 'r') as f:
                data = json.load(f)
                
            for episode in data:
                # 假设每个episode包含：
                # - observations: 观察数据
                # - gt_actions: 专家动作序列
                # - instruction: 指令文本
                
                trajectory = {
                    'observations': {
                        'rgb': torch.randn(len(episode['gt_actions']), 3, 224, 224),
                        'depth': torch.randn(len(episode['gt_actions']), 1, 224, 224),
                        'instruction': torch.randn(len(episode['gt_actions']), 512),
                    },
                    'gt_actions': episode['gt_actions'],
                    'instruction': episode.get('instruction', ''),
                    'episode_id': episode.get('episode_id', ''),
                }
                trajectories.append(trajectory)
        
        return trajectories
    
    def _convert_to_lmdb(
        self,
        trajectories: List[Dict[str, Any]],
        lmdb_path: str
    ):
        """将轨迹数据转换为LMDB格式"""
        import lmdb
        import msgpack_numpy
        
        lmdb_map_size = 1e9
        
        with lmdb.open(
            lmdb_path,
            map_size=int(lmdb_map_size),
        ) as lmdb_env, lmdb_env.begin(write=True) as txn:
            
            for idx, traj in enumerate(trajectories):
                # 处理轨迹数据
                processed_traj = self.data_processor.process_trajectory(
                    traj['observations'],
                    traj['gt_actions']
                )
                
                # 准备存储数据
                storage_data = [
                    processed_traj['observations'],
                    processed_traj['prev_actions'].numpy(),
                    processed_traj['gt_actions'].numpy(),
                ]
                
                # 存储到LMDB
                txn.put(
                    str(idx).encode(),
                    msgpack_numpy.packb(storage_data, use_bin_type=True)
                )
    
    def create_training_config(
        self,
        lmdb_path: str,
        checkpoint_dir: str,
        tensorboard_dir: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        创建训练配置
        
        Args:
            lmdb_path: LMDB数据路径
            checkpoint_dir: 检查点目录
            tensorboard_dir: TensorBoard目录
            **kwargs: 其他配置参数
            
        Returns:
            训练配置字典
        """
        config = {
            "IL": {
                "lr": kwargs.get('lr', 2.5e-4),
                "epochs": kwargs.get('epochs', 10),
                "batch_size": kwargs.get('batch_size', 4),
                "use_iw": kwargs.get('use_iw', True),
                "inflection_weight_coef": kwargs.get('inflection_weight_coef', 1.0),
                "load_from_ckpt": kwargs.get('load_from_ckpt', False),
                "is_requeue": kwargs.get('is_requeue', False),
                "GT_ACTION_IL": {
                    "lmdb_features_dir": lmdb_path,
                    "lmdb_map_size": kwargs.get('lmdb_map_size', 1e9),
                }
            },
            "MODEL": {
                "policy_name": kwargs.get('policy_name', "PointNavResNetPolicy"),
                "STATE_ENCODER": {
                    "hidden_size": kwargs.get('hidden_size', 512),
                }
            },
            "TASK_CONFIG": {
                "DATASET": {
                    "SPLIT": kwargs.get('split', "train"),
                }
            },
            "CHECKPOINT_FOLDER": checkpoint_dir,
            "TENSORBOARD_DIR": tensorboard_dir,
            "TORCH_GPU_ID": kwargs.get('gpu_id', 0),
        }
        
        return config
    
    def train_model(
        self,
        config: Dict[str, Any],
        resume_from: Optional[str] = None
    ) -> str:
        """
        训练模型
        
        Args:
            config: 训练配置
            resume_from: 恢复训练的检查点路径
            
        Returns:
            最终检查点路径
        """
        print("Starting model training...")
        
        # 如果指定了恢复路径，更新配置
        if resume_from:
            config["IL"]["load_from_ckpt"] = True
            config["IL"]["ckpt_to_load"] = resume_from
        
        # 创建训练器
        trainer = GT_ActionILTrainer(config)
        
        # 开始训练
        trainer.train()
        
        # 返回最终检查点路径
        final_checkpoint = os.path.join(
            config["CHECKPOINT_FOLDER"],
            "gt_action_il_epoch_final.pth"
        )
        
        print(f"Training completed. Final checkpoint: {final_checkpoint}")
        return final_checkpoint
    
    def evaluate_model(
        self,
        checkpoint_path: str,
        eval_config: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        评估模型
        
        Args:
            checkpoint_path: 检查点路径
            eval_config: 评估配置
            
        Returns:
            评估指标字典
        """
        print(f"Evaluating model: {checkpoint_path}")
        
        # 这里应该实现你的评估逻辑
        # 示例评估指标
        metrics = {
            'success_rate': 0.85,
            'spl': 0.78,
            'path_length': 12.5,
            'navigation_error': 0.3,
        }
        
        print("Evaluation completed:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.3f}")
        
        return metrics
    
    def deploy_model(
        self,
        checkpoint_path: str,
        deployment_config: Dict[str, Any]
    ) -> str:
        """
        部署模型
        
        Args:
            checkpoint_path: 检查点路径
            deployment_config: 部署配置
            
        Returns:
            部署路径
        """
        print(f"Deploying model: {checkpoint_path}")
        
        # 创建部署目录
        deploy_dir = deployment_config.get('deploy_dir', './deployed_model')
        os.makedirs(deploy_dir, exist_ok=True)
        
        # 复制检查点文件
        import shutil
        deploy_checkpoint = os.path.join(deploy_dir, 'model.pth')
        shutil.copy2(checkpoint_path, deploy_checkpoint)
        
        # 保存部署配置
        deploy_config_path = os.path.join(deploy_dir, 'deploy_config.json')
        with open(deploy_config_path, 'w') as f:
            json.dump(deployment_config, f, indent=2)
        
        print(f"Model deployed to: {deploy_dir}")
        return deploy_dir


def create_integration_example():
    """创建集成示例"""
    
    # 1. 创建集成实例
    config_path = "gt_action_il_config.json"
    integration = GT_ActionILIntegration(config_path)
    
    # 2. 准备数据
    data_dir = "/path/to/your/data"
    output_dir = "/path/to/processed/data"
    lmdb_path = integration.prepare_training_data(data_dir, output_dir, "train")
    
    # 3. 创建训练配置
    training_config = integration.create_training_config(
        lmdb_path=lmdb_path,
        checkpoint_dir="/path/to/checkpoints",
        tensorboard_dir="/path/to/tensorboard",
        lr=2.5e-4,
        epochs=20,
        batch_size=8,
    )
    
    # 4. 训练模型
    final_checkpoint = integration.train_model(training_config)
    
    # 5. 评估模型
    eval_config = {
        'eval_data_path': "/path/to/eval/data",
        'num_episodes': 100,
    }
    metrics = integration.evaluate_model(final_checkpoint, eval_config)
    
    # 6. 部署模型
    deployment_config = {
        'deploy_dir': "/path/to/deployment",
        'model_name': 'gt_action_il_model',
    }
    deploy_path = integration.deploy_model(final_checkpoint, deployment_config)
    
    print("Integration example completed!")
    return {
        'final_checkpoint': final_checkpoint,
        'metrics': metrics,
        'deploy_path': deploy_path,
    }


def create_sample_config():
    """创建示例配置文件"""
    config = {
        "action_space_size": 4,
        "observation_space": {
            "rgb": [3, 224, 224],
            "depth": [1, 224, 224],
            "instruction": [512],
        },
        "model": {
            "policy_name": "PointNavResNetPolicy",
            "hidden_size": 512,
            "num_recurrent_layers": 2,
        },
        "training": {
            "lr": 2.5e-4,
            "epochs": 20,
            "batch_size": 8,
            "use_inflection_weighting": True,
            "inflection_weight_coef": 1.0,
        },
        "data": {
            "max_trajectory_length": 100,
            "lmdb_map_size": 1e9,
        }
    }
    
    with open("gt_action_il_config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    print("Sample configuration created: gt_action_il_config.json")


if __name__ == "__main__":
    # 创建示例配置
    create_sample_config()
    
    # 运行集成示例
    # result = create_integration_example()
    # print("Integration result:", result)
