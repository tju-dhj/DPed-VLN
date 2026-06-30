#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple
from gym import spaces

from habitat_baselines.rl.models.rnn_state_encoder import build_rnn_state_encoder
from habitat_baselines.rl.models.simple_cnn import SimpleCNN
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat.tasks.nav.nav import (
    ImageGoalSensor,
    IntegratedPointGoalGPSAndCompassSensor,
    PointGoalSensor,
)

from .vln_sensors import (
    LanguageEncoder,
    DINOv3VisualEncoder,
    CrossModalFusion,
    InstructionSensor,
    GT_ActionSensor,
)


class VLNNet(nn.Module):
    """
    VLN Network that combines visual and language features for navigation.
    Extends the original PointNavBaselineNet with VLN capabilities.
    """
    
    def __init__(
        self,
        observation_space: spaces.Dict,
        hidden_size: int = 512,
        use_dinov3: bool = True,
        use_clip_text: bool = True,
        freeze_visual_encoder: bool = False,
        freeze_language_encoder: bool = False,
    ):
        super().__init__()
        
        self._hidden_size = hidden_size
        self.use_dinov3 = use_dinov3
        self.use_clip_text = use_clip_text
        self.freeze_visual_encoder = freeze_visual_encoder
        self.freeze_language_encoder = freeze_language_encoder
        
        # Initialize visual encoder
        if use_dinov3:
            self.visual_encoder = DINOv3VisualEncoder(hidden_size=hidden_size)
            if freeze_visual_encoder:
                for param in self.visual_encoder.parameters():
                    param.requires_grad = False
        else:
            # Fallback to SimpleCNN
            self.visual_encoder = SimpleCNN(observation_space, hidden_size)
        
        # Initialize language encoder
        if use_clip_text:
            self.language_encoder = LanguageEncoder(hidden_size=hidden_size)
            if freeze_language_encoder:
                for param in self.language_encoder.parameters():
                    param.requires_grad = False
        
        # Cross-modal fusion
        self.cross_modal_fusion = CrossModalFusion(
            visual_dim=hidden_size,
            language_dim=hidden_size,
            hidden_size=hidden_size
        )
        
        # Goal encoding (keep original goal handling)
        self._setup_goal_encoding(observation_space)
        
        # State encoder (RNN)
        self.state_encoder = build_rnn_state_encoder(
            (0 if self.is_blind else self._hidden_size) + self._n_input_goal,
            self._hidden_size,
        )
        
        # Additional VLN-specific layers
        self.vln_fusion_layer = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size)
        )
        
        self.train()
    
    def _setup_goal_encoding(self, observation_space: spaces.Dict):
        """Setup goal encoding similar to original PointNavBaselineNet."""
        if (
            IntegratedPointGoalGPSAndCompassSensor.cls_uuid
            in observation_space.spaces
        ):
            self._n_input_goal = observation_space.spaces[
                IntegratedPointGoalGPSAndCompassSensor.cls_uuid
            ].shape[0]
        elif PointGoalSensor.cls_uuid in observation_space.spaces:
            self._n_input_goal = observation_space.spaces[
                PointGoalSensor.cls_uuid
            ].shape[0]
        elif ImageGoalSensor.cls_uuid in observation_space.spaces:
            goal_observation_space = spaces.Dict(
                {"rgb": observation_space.spaces[ImageGoalSensor.cls_uuid]}
            )
            self.goal_visual_encoder = SimpleCNN(
                goal_observation_space, self._hidden_size
            )
            self._n_input_goal = self._hidden_size
        else:
            self._n_input_goal = 0
    
    @property
    def output_size(self):
        return self._hidden_size
    
    @property
    def is_blind(self):
        if self.use_dinov3:
            return False  # DINOv3 always processes visual input
        else:
            return self.visual_encoder.is_blind
    
    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers
    
    @property
    def recurrent_hidden_size(self):
        return self._hidden_size
    
    @property
    def perception_embedding_size(self):
        return self._hidden_size
    
    def _encode_visual(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode visual observations."""
        if self.use_dinov3:
            # Use DINOv3 for visual encoding
            rgb = observations.get("rgb", observations.get("color_sensor"))
            if rgb is not None:
                return self.visual_encoder(rgb)
            else:
                # Fallback to zero features if no RGB
                batch_size = next(iter(observations.values())).shape[0]
                return torch.zeros(batch_size, self._hidden_size, device=next(iter(observations.values())).device)
        else:
            # Use SimpleCNN
            return self.visual_encoder(observations)
    
    def _encode_language(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode language instructions."""
        if not self.use_clip_text:
            # Return zero features if not using language
            batch_size = next(iter(observations.values())).shape[0]
            return torch.zeros(batch_size, self._hidden_size, device=next(iter(observations.values())).device)
        
        # Get instruction from observations
        instruction = observations.get(InstructionSensor.cls_uuid)
        if instruction is not None and instruction.numel() > 0:
            # Instruction is already encoded by the sensor
            return instruction
        else:
            # Return zero features if no instruction
            batch_size = next(iter(observations.values())).shape[0]
            return torch.zeros(batch_size, self._hidden_size, device=next(iter(observations.values())).device)
    
    def _encode_goal(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode goal information."""
        if IntegratedPointGoalGPSAndCompassSensor.cls_uuid in observations:
            return observations[IntegratedPointGoalGPSAndCompassSensor.cls_uuid]
        elif PointGoalSensor.cls_uuid in observations:
            return observations[PointGoalSensor.cls_uuid]
        elif ImageGoalSensor.cls_uuid in observations:
            image_goal = observations[ImageGoalSensor.cls_uuid]
            return self.goal_visual_encoder({"rgb": image_goal})
        else:
            # Return zero goal encoding
            batch_size = next(iter(observations.values())).shape[0]
            return torch.zeros(batch_size, self._n_input_goal, device=next(iter(observations.values())).device)
    
    def forward(
        self,
        observations: Dict[str, torch.Tensor],
        rnn_hidden_states: torch.Tensor,
        prev_actions: torch.Tensor,
        masks: torch.Tensor,
        rnn_build_seq_info: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass of VLN network.
        
        Args:
            observations: Dictionary of observations
            rnn_hidden_states: RNN hidden states
            prev_actions: Previous actions
            masks: Action masks
            rnn_build_seq_info: RNN sequence building info
            
        Returns:
            Tuple of (output_features, rnn_hidden_states, aux_loss_state)
        """
        aux_loss_state = {}
        
        # Encode visual observations
        visual_features = self._encode_visual(observations)
        aux_loss_state["visual_features"] = visual_features
        
        # Encode language instructions
        language_features = self._encode_language(observations)
        aux_loss_state["language_features"] = language_features
        
        # Cross-modal fusion
        fused_features = self.cross_modal_fusion(visual_features, language_features)
        aux_loss_state["fused_features"] = fused_features
        
        # Encode goal
        goal_encoding = self._encode_goal(observations)
        
        # Combine fused features with goal encoding
        x = [fused_features, goal_encoding]
        x_out = torch.cat(x, dim=1)
        
        # Pass through RNN state encoder
        x_out, rnn_hidden_states = self.state_encoder(
            x_out, rnn_hidden_states, masks, rnn_build_seq_info
        )
        
        aux_loss_state["rnn_output"] = x_out
        
        return x_out, rnn_hidden_states, aux_loss_state


class VLNILNet(VLNNet):
    """
    VLN Network specifically designed for Imitation Learning (IL).
    Includes additional components for IL training.
    """
    
    def __init__(
        self,
        observation_space: spaces.Dict,
        hidden_size: int = 512,
        use_dinov3: bool = True,
        use_clip_text: bool = True,
        freeze_visual_encoder: bool = False,
        freeze_language_encoder: bool = False,
        il_loss_weight: float = 1.0,
    ):
        super().__init__(
            observation_space=observation_space,
            hidden_size=hidden_size,
            use_dinov3=use_dinov3,
            use_clip_text=use_clip_text,
            freeze_visual_encoder=freeze_visual_encoder,
            freeze_language_encoder=freeze_language_encoder,
        )
        
        self.il_loss_weight = il_loss_weight
        
        # IL-specific components
        self.action_predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 1)  # Single action prediction
        )
        
        # Sequence predictor for action sequences
        self.sequence_predictor = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        
        self.sequence_output = nn.Linear(hidden_size, 1)  # Action prediction
    
    def forward_il(
        self,
        observations: Dict[str, torch.Tensor],
        rnn_hidden_states: torch.Tensor,
        prev_actions: torch.Tensor,
        masks: torch.Tensor,
        gt_actions: Optional[torch.Tensor] = None,
        rnn_build_seq_info: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass for IL training.
        
        Args:
            observations: Dictionary of observations
            rnn_hidden_states: RNN hidden states
            prev_actions: Previous actions
            masks: Action masks
            gt_actions: Ground truth actions for IL
            rnn_build_seq_info: RNN sequence building info
            
        Returns:
            Tuple of (output_features, rnn_hidden_states, aux_loss_state)
        """
        # Get base VLN features
        x_out, rnn_hidden_states, aux_loss_state = self.forward(
            observations, rnn_hidden_states, prev_actions, masks, rnn_build_seq_info
        )
        
        # IL-specific processing
        if gt_actions is not None:
            # Predict actions for IL loss
            action_pred = self.action_predictor(x_out)
            aux_loss_state["action_prediction"] = action_pred
            aux_loss_state["gt_actions"] = gt_actions
            
            # Sequence prediction
            if gt_actions.dim() > 1:  # Sequence of actions
                sequence_output, _ = self.sequence_predictor(x_out.unsqueeze(1))
                sequence_pred = self.sequence_output(sequence_output)
                aux_loss_state["sequence_prediction"] = sequence_pred
        
        return x_out, rnn_hidden_states, aux_loss_state
    
    def compute_il_loss(
        self,
        action_predictions: torch.Tensor,
        gt_actions: torch.Tensor,
        sequence_predictions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute imitation learning loss.
        
        Args:
            action_predictions: Predicted actions
            gt_actions: Ground truth actions
            sequence_predictions: Predicted action sequences (optional)
            
        Returns:
            IL loss tensor
        """
        # Action prediction loss
        action_loss = nn.MSELoss()(action_predictions, gt_actions.float())
        
        total_loss = self.il_loss_weight * action_loss
        
        if sequence_predictions is not None:
            # Sequence prediction loss
            seq_loss = nn.MSELoss()(sequence_predictions, gt_actions.float())
            total_loss += self.il_loss_weight * seq_loss
        
        return total_loss


# Register the networks
@baseline_registry.register_net(name="VLNNet")
class VLNNetWrapper(VLNNet):
    """Wrapper for VLNNet to be used with baseline registry."""
    pass


@baseline_registry.register_net(name="VLNILNet")
class VLNILNetWrapper(VLNILNet):
    """Wrapper for VLNILNet to be used with baseline registry."""
    pass
