#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple, List
from gym import spaces
from omegaconf import DictConfig

from habitat_baselines.rl.ppo.policy import NetPolicy, PolicyActionData
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.utils.common import CategoricalNet, GaussianNet, get_num_actions
from habitat_baselines.rl.ppo.policy import CriticHead

from .vln_net import VLNNet, VLNILNet
from .vln_sensors import InstructionSensor, GT_ActionSensor


@baseline_registry.register_policy
class VLNPolicy(NetPolicy):
    """
    VLN Policy that extends the original PointNavBaselinePolicy with VLN capabilities.
    Maintains compatibility with existing RL training while adding VLN features.
    """
    
    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space: spaces.Space,
        hidden_size: int = 512,
        aux_loss_config: Optional[DictConfig] = None,
        use_dinov3: bool = True,
        use_clip_text: bool = True,
        freeze_visual_encoder: bool = False,
        freeze_language_encoder: bool = False,
        **kwargs,
    ):
        # Initialize VLN network
        vln_net = VLNNet(
            observation_space=observation_space,
            hidden_size=hidden_size,
            use_dinov3=use_dinov3,
            use_clip_text=use_clip_text,
            freeze_visual_encoder=freeze_visual_encoder,
            freeze_language_encoder=freeze_language_encoder,
        )
        
        # Initialize parent NetPolicy
        super().__init__(
            net=vln_net,
            action_space=action_space,
            aux_loss_config=aux_loss_config,
        )
        
        # Store VLN-specific parameters
        self.use_dinov3 = use_dinov3
        self.use_clip_text = use_clip_text
        self.freeze_visual_encoder = freeze_visual_encoder
        self.freeze_language_encoder = freeze_language_encoder
    
    @classmethod
    def from_config(
        cls,
        config: DictConfig,
        observation_space: spaces.Dict,
        action_space: spaces.Space,
        **kwargs,
    ):
        """Create VLNPolicy from configuration."""
        vln_config = config.habitat_baselines.rl.ppo.vln_policy
        
        return cls(
            observation_space=observation_space,
            action_space=action_space,
            hidden_size=config.habitat_baselines.rl.ppo.hidden_size,
            aux_loss_config=config.habitat_baselines.rl.auxiliary_losses,
            use_dinov3=vln_config.get("use_dinov3", True),
            use_clip_text=vln_config.get("use_clip_text", True),
            freeze_visual_encoder=vln_config.get("freeze_visual_encoder", False),
            freeze_language_encoder=vln_config.get("freeze_language_encoder", False),
        )
    
    def act(
        self,
        observations: Dict[str, torch.Tensor],
        rnn_hidden_states: torch.Tensor,
        prev_actions: torch.Tensor,
        masks: torch.Tensor,
        deterministic: bool = False,
    ) -> PolicyActionData:
        """
        Act method for VLN policy.
        Extends the original act method with VLN-specific processing.
        """
        # Get base features from VLN network
        features, rnn_hidden_states, aux_loss_state = self.net(
            observations, rnn_hidden_states, prev_actions, masks
        )
        
        # Get action distribution
        action_logits = self.action_distribution(features)
        value = self.critic(features)
        
        # Sample actions
        if deterministic:
            if isinstance(self.action_distribution, CategoricalNet):
                action = action_logits.argmax(dim=-1, keepdim=True)
            else:
                action = action_logits
        else:
            if isinstance(self.action_distribution, CategoricalNet):
                action = torch.multinomial(
                    torch.softmax(action_logits, dim=-1), 1
                )
            else:
                action = self.action_distribution.sample()
        
        return PolicyActionData(
            actions=action,
            rnn_hidden_states=rnn_hidden_states,
            policy_info=aux_loss_state,
        )
    
    def evaluate_actions(
        self,
        observations: Dict[str, torch.Tensor],
        rnn_hidden_states: torch.Tensor,
        prev_actions: torch.Tensor,
        masks: torch.Tensor,
        action: torch.Tensor,
        rnn_build_seq_info: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Evaluate actions for RL training.
        Extends the original evaluate_actions with VLN-specific features.
        """
        # Get features and auxiliary losses
        features, rnn_hidden_states, aux_loss_state = self.net(
            observations, rnn_hidden_states, prev_actions, masks, rnn_build_seq_info
        )
        
        # Get action distribution and value
        action_logits = self.action_distribution(features)
        value = self.critic(features)
        
        # Compute action log probabilities
        if isinstance(self.action_distribution, CategoricalNet):
            action_log_probs = torch.log_softmax(action_logits, dim=-1)
            action_log_probs = action_log_probs.gather(1, action)
        else:
            action_log_probs = self.action_distribution.log_probs(action)
        
        # Compute entropy
        if isinstance(self.action_distribution, CategoricalNet):
            entropy = -(torch.softmax(action_logits, dim=-1) * action_log_probs).sum(dim=-1)
        else:
            entropy = self.action_distribution.entropy()
        
        return value, action_log_probs, entropy, rnn_hidden_states, aux_loss_state


@baseline_registry.register_policy
class VLNILPolicy(VLNPolicy):
    """
    VLN Policy specifically designed for Imitation Learning (IL).
    Extends VLNPolicy with IL-specific capabilities.
    """
    
    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space: spaces.Space,
        hidden_size: int = 512,
        aux_loss_config: Optional[DictConfig] = None,
        use_dinov3: bool = True,
        use_clip_text: bool = True,
        freeze_visual_encoder: bool = False,
        freeze_language_encoder: bool = False,
        il_loss_weight: float = 1.0,
        **kwargs,
    ):
        # Initialize VLNIL network
        vln_il_net = VLNILNet(
            observation_space=observation_space,
            hidden_size=hidden_size,
            use_dinov3=use_dinov3,
            use_clip_text=use_clip_text,
            freeze_visual_encoder=freeze_visual_encoder,
            freeze_language_encoder=freeze_language_encoder,
            il_loss_weight=il_loss_weight,
        )
        
        # Initialize parent NetPolicy with VLNIL network
        super(VLNPolicy, self).__init__(
            net=vln_il_net,
            action_space=action_space,
            aux_loss_config=aux_loss_config,
        )
        
        # Store IL-specific parameters
        self.il_loss_weight = il_loss_weight
    
    @classmethod
    def from_config(
        cls,
        config: DictConfig,
        observation_space: spaces.Dict,
        action_space: spaces.Space,
        **kwargs,
    ):
        """Create VLNILPolicy from configuration."""
        vln_config = config.habitat_baselines.rl.ppo.vln_il_policy
        
        return cls(
            observation_space=observation_space,
            action_space=action_space,
            hidden_size=config.habitat_baselines.rl.ppo.hidden_size,
            aux_loss_config=config.habitat_baselines.rl.auxiliary_losses,
            use_dinov3=vln_config.get("use_dinov3", True),
            use_clip_text=vln_config.get("use_clip_text", True),
            freeze_visual_encoder=vln_config.get("freeze_visual_encoder", False),
            freeze_language_encoder=vln_config.get("freeze_language_encoder", False),
            il_loss_weight=vln_config.get("il_loss_weight", 1.0),
        )
    
    def act_il(
        self,
        observations: Dict[str, torch.Tensor],
        rnn_hidden_states: torch.Tensor,
        prev_actions: torch.Tensor,
        masks: torch.Tensor,
        gt_actions: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> PolicyActionData:
        """
        Act method for IL training.
        Includes ground truth action processing.
        """
        # Get features with IL processing
        features, rnn_hidden_states, aux_loss_state = self.net.forward_il(
            observations, rnn_hidden_states, prev_actions, masks, gt_actions
        )
        
        # Get action distribution
        action_logits = self.action_distribution(features)
        value = self.critic(features)
        
        # For IL, we can use ground truth actions if available
        if gt_actions is not None and not deterministic:
            action = gt_actions
        else:
            # Sample actions normally
            if deterministic:
                if isinstance(self.action_distribution, CategoricalNet):
                    action = action_logits.argmax(dim=-1, keepdim=True)
                else:
                    action = action_logits
            else:
                if isinstance(self.action_distribution, CategoricalNet):
                    action = torch.multinomial(
                        torch.softmax(action_logits, dim=-1), 1
                    )
                else:
                    action = self.action_distribution.sample()
        
        return PolicyActionData(
            actions=action,
            rnn_hidden_states=rnn_hidden_states,
            policy_info=aux_loss_state,
        )
    
    def compute_il_loss(
        self,
        observations: Dict[str, torch.Tensor],
        rnn_hidden_states: torch.Tensor,
        prev_actions: torch.Tensor,
        masks: torch.Tensor,
        gt_actions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute imitation learning loss.
        """
        # Get features with IL processing
        features, _, aux_loss_state = self.net.forward_il(
            observations, rnn_hidden_states, prev_actions, masks, gt_actions
        )
        
        # Get action predictions
        action_predictions = aux_loss_state.get("action_prediction")
        sequence_predictions = aux_loss_state.get("sequence_prediction")
        
        if action_predictions is not None:
            # Compute IL loss using the network's IL loss computation
            il_loss = self.net.compute_il_loss(
                action_predictions, gt_actions, sequence_predictions
            )
            return il_loss
        else:
            # Fallback to simple MSE loss
            action_logits = self.action_distribution(features)
            return torch.nn.MSELoss()(action_logits, gt_actions.float())
    
    def evaluate_actions_il(
        self,
        observations: Dict[str, torch.Tensor],
        rnn_hidden_states: torch.Tensor,
        prev_actions: torch.Tensor,
        masks: torch.Tensor,
        action: torch.Tensor,
        gt_actions: Optional[torch.Tensor] = None,
        rnn_build_seq_info: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Evaluate actions for IL training.
        """
        # Get features with IL processing
        features, rnn_hidden_states, aux_loss_state = self.net.forward_il(
            observations, rnn_hidden_states, prev_actions, masks, gt_actions, rnn_build_seq_info
        )
        
        # Get action distribution and value
        action_logits = self.action_distribution(features)
        value = self.critic(features)
        
        # Compute action log probabilities
        if isinstance(self.action_distribution, CategoricalNet):
            action_log_probs = torch.log_softmax(action_logits, dim=-1)
            action_log_probs = action_log_probs.gather(1, action)
        else:
            action_log_probs = self.action_distribution.log_probs(action)
        
        # Compute entropy
        if isinstance(self.action_distribution, CategoricalNet):
            entropy = -(torch.softmax(action_logits, dim=-1) * action_log_probs).sum(dim=-1)
        else:
            entropy = self.action_distribution.entropy()
        
        # Add IL loss to auxiliary losses
        if gt_actions is not None:
            il_loss = self.compute_il_loss(
                observations, rnn_hidden_states, prev_actions, masks, gt_actions
            )
            aux_loss_state["il_loss"] = il_loss
        
        return value, action_log_probs, entropy, rnn_hidden_states, aux_loss_state


class VLNHybridPolicy(VLNPolicy):
    """
    Hybrid VLN Policy that supports both RL and IL training.
    Can switch between RL and IL modes during training.
    """
    
    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space: spaces.Space,
        hidden_size: int = 512,
        aux_loss_config: Optional[DictConfig] = None,
        use_dinov3: bool = True,
        use_clip_text: bool = True,
        freeze_visual_encoder: bool = False,
        freeze_language_encoder: bool = False,
        il_loss_weight: float = 1.0,
        rl_il_ratio: float = 0.5,  # Ratio of RL vs IL training
        **kwargs,
    ):
        # Initialize with VLNIL network for hybrid capabilities
        vln_il_net = VLNILNet(
            observation_space=observation_space,
            hidden_size=hidden_size,
            use_dinov3=use_dinov3,
            use_clip_text=use_clip_text,
            freeze_visual_encoder=freeze_visual_encoder,
            freeze_language_encoder=freeze_language_encoder,
            il_loss_weight=il_loss_weight,
        )
        
        # Initialize parent NetPolicy
        super(VLNPolicy, self).__init__(
            net=vln_il_net,
            action_space=action_space,
            aux_loss_config=aux_loss_config,
        )
        
        # Store hybrid-specific parameters
        self.il_loss_weight = il_loss_weight
        self.rl_il_ratio = rl_il_ratio
        self.training_mode = "hybrid"  # "rl", "il", or "hybrid"
    
    def set_training_mode(self, mode: str):
        """Set training mode: 'rl', 'il', or 'hybrid'."""
        assert mode in ["rl", "il", "hybrid"], f"Invalid training mode: {mode}"
        self.training_mode = mode
    
    def act(
        self,
        observations: Dict[str, torch.Tensor],
        rnn_hidden_states: torch.Tensor,
        prev_actions: torch.Tensor,
        masks: torch.Tensor,
        deterministic: bool = False,
        gt_actions: Optional[torch.Tensor] = None,
    ) -> PolicyActionData:
        """
        Hybrid act method that can handle both RL and IL.
        """
        if self.training_mode == "il" and gt_actions is not None:
            # Use IL-specific act method
            return self.act_il(
                observations, rnn_hidden_states, prev_actions, masks, gt_actions, deterministic
            )
        else:
            # Use standard RL act method
            return super().act(
                observations, rnn_hidden_states, prev_actions, masks, deterministic
            )
    
    def evaluate_actions(
        self,
        observations: Dict[str, torch.Tensor],
        rnn_hidden_states: torch.Tensor,
        prev_actions: torch.Tensor,
        masks: torch.Tensor,
        action: torch.Tensor,
        gt_actions: Optional[torch.Tensor] = None,
        rnn_build_seq_info: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Hybrid evaluate_actions that can handle both RL and IL.
        """
        if self.training_mode == "il" and gt_actions is not None:
            # Use IL-specific evaluate method
            return self.evaluate_actions_il(
                observations, rnn_hidden_states, prev_actions, masks, action, gt_actions, rnn_build_seq_info
            )
        else:
            # Use standard RL evaluate method
            return super().evaluate_actions(
                observations, rnn_hidden_states, prev_actions, masks, action, rnn_build_seq_info
            )
