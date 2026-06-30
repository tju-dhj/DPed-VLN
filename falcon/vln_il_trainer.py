#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
from collections import defaultdict, deque
import time
from habitat.utils import profiling_wrapper
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.ppo import PPO
from habitat_baselines.rl.ppo.ppo_trainer import PPOTrainer
from habitat_baselines.utils.common import (
    batch_obs,
    generate_video,
    linear_decay,
)
from habitat_baselines.rl.ppo.policy import PolicyActionData

from .vln_policy import VLNILPolicy, VLNHybridPolicy
from .vln_sensors import InstructionSensor, GT_ActionSensor


class VLNILTrainer(PPOTrainer):
    """
    Imitation Learning Trainer for VLN tasks.
    Extends PPO trainer with IL-specific capabilities.
    """
    
    def __init__(self, config):
        super().__init__(config)
        
        # IL-specific parameters
        self.il_loss_weight = config.habitat_baselines.rl.ppo.vln_il_policy.get("il_loss_weight", 1.0)
        self.il_batch_size = config.habitat_baselines.rl.ppo.vln_il_policy.get("il_batch_size", 32)
        self.il_learning_rate = config.habitat_baselines.rl.ppo.vln_il_policy.get("il_learning_rate", 3e-4)
        self.il_epochs = config.habitat_baselines.rl.ppo.vln_il_policy.get("il_epochs", 4)
        
        # IL-specific optimizers
        self.il_optimizer = torch.optim.Adam(
            self.actor_critic.parameters(),
            lr=self.il_learning_rate,
            eps=1e-5
        )
        
        # IL loss tracking
        self.il_losses = deque(maxlen=100)
        self.il_loss_epochs = deque(maxlen=100)
        
        # Expert data storage
        self.expert_data_buffer = []
        self.expert_data_size = config.habitat_baselines.rl.ppo.vln_il_policy.get("expert_data_size", 10000)
        
    def _collect_expert_data(self, rollouts):
        """
        Collect expert data from rollouts for IL training.
        """
        expert_data = []
        
        for step in rollouts:
            observations = step.observations
            actions = step.actions
            gt_actions = observations.get(GT_ActionSensor.cls_uuid)
            instructions = observations.get(InstructionSensor.cls_uuid)
            
            if gt_actions is not None and instructions is not None:
                expert_data.append({
                    'observations': observations,
                    'actions': actions,
                    'gt_actions': gt_actions,
                    'instructions': instructions,
                })
        
        # Add to expert data buffer
        self.expert_data_buffer.extend(expert_data)
        
        # Keep only the most recent expert data
        if len(self.expert_data_buffer) > self.expert_data_size:
            self.expert_data_buffer = self.expert_data_buffer[-self.expert_data_size:]
    
    def _compute_il_loss(self, batch):
        """
        Compute imitation learning loss for a batch of expert data.
        """
        observations = batch['observations']
        gt_actions = batch['gt_actions']
        
        # Get policy features
        rnn_hidden_states = torch.zeros(
            self.actor_critic.net.num_recurrent_layers,
            len(observations),
            self.actor_critic.net.recurrent_hidden_size,
            device=observations[0].device
        )
        
        prev_actions = torch.zeros(
            len(observations), 1, device=observations[0].device
        )
        
        masks = torch.ones(
            len(observations), 1, device=observations[0].device
        )
        
        # Compute IL loss using policy
        if isinstance(self.actor_critic, VLNILPolicy):
            il_loss = self.actor_critic.compute_il_loss(
                observations, rnn_hidden_states, prev_actions, masks, gt_actions
            )
        else:
            # Fallback to simple MSE loss
            features, _, _ = self.actor_critic.net(
                observations, rnn_hidden_states, prev_actions, masks
            )
            action_logits = self.actor_critic.action_distribution(features)
            il_loss = nn.MSELoss()(action_logits, gt_actions.float())
        
        return il_loss
    
    def _train_il_epoch(self):
        """
        Train one epoch of imitation learning.
        """
        if len(self.expert_data_buffer) < self.il_batch_size:
            return 0.0
        
        # Sample batch from expert data
        batch_indices = np.random.choice(
            len(self.expert_data_buffer),
            size=min(self.il_batch_size, len(self.expert_data_buffer)),
            replace=False
        )
        
        batch_data = [self.expert_data_buffer[i] for i in batch_indices]
        
        # Prepare batch
        batch = {
            'observations': [item['observations'] for item in batch_data],
            'gt_actions': torch.stack([item['gt_actions'] for item in batch_data]),
            'instructions': [item['instructions'] for item in batch_data],
        }
        
        # Compute IL loss
        il_loss = self._compute_il_loss(batch)
        
        # Backward pass
        self.il_optimizer.zero_grad()
        il_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 0.5)
        self.il_optimizer.step()
        
        return il_loss.item()
    
    def _update_il(self):
        """
        Update policy using imitation learning.
        """
        il_losses = []
        
        for epoch in range(self.il_epochs):
            il_loss = self._train_il_epoch()
            if il_loss > 0:
                il_losses.append(il_loss)
        
        if il_losses:
            avg_il_loss = np.mean(il_losses)
            self.il_losses.append(avg_il_loss)
            self.il_loss_epochs.append(self.num_updates_done)
            
            return avg_il_loss
        
        return 0.0
    
    def _update(self, rollouts):
        """
        Update policy using both RL and IL.
        """
        # Collect expert data from rollouts
        self._collect_expert_data(rollouts)
        
        # Standard RL update
        rl_loss = super()._update(rollouts)
        
        # IL update
        il_loss = self._update_il()
        
        return {
            'rl_loss': rl_loss,
            'il_loss': il_loss,
        }
    
    def _eval_checkpoint(
        self,
        checkpoint_path: str,
        writer,
        checkpoint_index: int = 0,
    ) -> None:
        """
        Evaluate checkpoint with VLN-specific metrics.
        """
        # Load checkpoint
        ckpt_dict = self.load_checkpoint(checkpoint_path, map_location="cpu")
        
        # Set to eval mode
        self.actor_critic.eval()
        
        # Run evaluation
        with torch.no_grad():
            eval_metrics = self._run_eval_episodes()
        
        # Log VLN-specific metrics
        if writer is not None:
            writer.add_scalar("eval/success_rate", eval_metrics.get("success_rate", 0.0), checkpoint_index)
            writer.add_scalar("eval/spl", eval_metrics.get("spl", 0.0), checkpoint_index)
            writer.add_scalar("eval/instruction_accuracy", eval_metrics.get("instruction_accuracy", 0.0), checkpoint_index)
        
        return eval_metrics
    
    def _run_eval_episodes(self) -> Dict[str, float]:
        """
        Run evaluation episodes and compute VLN metrics.
        """
        eval_metrics = {
            "success_rate": 0.0,
            "spl": 0.0,
            "instruction_accuracy": 0.0,
        }
        
        # TODO: Implement actual evaluation logic
        # This would involve running episodes and computing metrics
        
        return eval_metrics


class VLNHybridTrainer(VLNILTrainer):
    """
    Hybrid trainer that supports both RL and IL training.
    Can switch between training modes or use both simultaneously.
    """
    
    def __init__(self, config):
        super().__init__(config)
        
        # Hybrid training parameters
        self.rl_il_ratio = config.habitat_baselines.rl.ppo.vln_hybrid_policy.get("rl_il_ratio", 0.5)
        self.training_mode = config.habitat_baselines.rl.ppo.vln_hybrid_policy.get("training_mode", "hybrid")
        self.switch_frequency = config.habitat_baselines.rl.ppo.vln_hybrid_policy.get("switch_frequency", 1000)
        
        # Training mode tracking
        self.current_mode = "rl"
        self.mode_switch_counter = 0
    
    def _should_switch_mode(self):
        """
        Determine if we should switch training modes.
        """
        if self.training_mode == "hybrid":
            self.mode_switch_counter += 1
            if self.mode_switch_counter >= self.switch_frequency:
                self.mode_switch_counter = 0
                self.current_mode = "il" if self.current_mode == "rl" else "rl"
                return True
        return False
    
    def _update(self, rollouts):
        """
        Hybrid update that can use both RL and IL.
        """
        # Check if we should switch modes
        if self._should_switch_mode():
            if isinstance(self.actor_critic, VLNHybridPolicy):
                self.actor_critic.set_training_mode(self.current_mode)
        
        # Collect expert data
        self._collect_expert_data(rollouts)
        
        # Determine update strategy
        if self.training_mode == "rl":
            return super(VLNILTrainer, self)._update(rollouts)
        elif self.training_mode == "il":
            return self._update_il_only()
        else:  # hybrid
            return self._update_hybrid(rollouts)
    
    def _update_il_only(self):
        """
        Update using only imitation learning.
        """
        il_loss = self._update_il()
        return {"il_loss": il_loss}
    
    def _update_hybrid(self, rollouts):
        """
        Update using both RL and IL with adaptive weighting.
        """
        # RL update
        rl_loss = super(VLNILTrainer, self)._update(rollouts)
        
        # IL update
        il_loss = self._update_il()
        
        # Adaptive weighting based on current mode
        if self.current_mode == "rl":
            rl_weight = 1.0
            il_weight = 0.1
        else:
            rl_weight = 0.1
            il_weight = 1.0
        
        return {
            "rl_loss": rl_loss * rl_weight,
            "il_loss": il_loss * il_weight,
            "total_loss": rl_loss * rl_weight + il_loss * il_weight,
        }


# Register trainers
@baseline_registry.register_trainer(name="vln_il")
class VLNILTrainerWrapper(VLNILTrainer):
    """Wrapper for VLNILTrainer to be used with baseline registry."""
    pass


@baseline_registry.register_trainer(name="vln_hybrid")
class VLNHybridTrainerWrapper(VLNHybridTrainer):
    """Wrapper for VLNHybridTrainer to be used with baseline registry."""
    pass


class VLNDataLoader:
    """
    Data loader for VLN training data.
    Handles loading and preprocessing of VLN datasets.
    """
    
    def __init__(self, dataset_path: str, batch_size: int = 32):
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.dataset = self._load_dataset()
    
    def _load_dataset(self):
        """
        Load VLN dataset from filtered dataset.
        """
        import json
        import gzip
        import os
        
        dataset = []
        
        # Load all scene files
        for scene_file in os.listdir(self.dataset_path):
            if scene_file.endswith('.json.gz'):
                scene_path = os.path.join(self.dataset_path, scene_file)
                
                with gzip.open(scene_path, 'rt') as f:
                    scene_data = json.load(f)
                
                # Process episodes
                for episode in scene_data.get('episodes', []):
                    if 'instruction' in episode and 'gt_action' in episode:
                        dataset.append({
                            'episode_id': episode['episode_id'],
                            'scene_id': episode['scene_id'],
                            'instruction': episode['instruction'],
                            'gt_action': episode['gt_action'],
                            'start_position': episode['start_position'],
                            'goals': episode['goals'],
                        })
        
        return dataset
    
    def get_batch(self):
        """
        Get a batch of training data.
        """
        if len(self.dataset) < self.batch_size:
            return None
        
        # Sample batch
        batch_indices = np.random.choice(
            len(self.dataset),
            size=self.batch_size,
            replace=False
        )
        
        batch_data = [self.dataset[i] for i in batch_indices]
        return batch_data
    
    def __len__(self):
        return len(self.dataset)
