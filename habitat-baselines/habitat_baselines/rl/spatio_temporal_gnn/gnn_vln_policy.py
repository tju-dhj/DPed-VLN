#!/usr/bin/env python3
"""
VLN Policy with Spatio-Temporal Graph Neural Network.

This module integrates ST-GNN into the VLN policy network, providing
social-aware navigation capabilities while maintaining full compatibility
with the existing DPED-PRO framework.

Key Features:
- Drop-in replacement for existing policies
- Optional GNN processing (can be disabled)
- Seamless integration with existing visual encoders
- Supports both discrete and continuous action spaces

Author: DPED-PRO
Date: 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, List
from gym import spaces
import numpy as np

from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.ppo.policy import NetPolicy, PolicyActionData, CriticHead
from habitat_baselines.rl.models.rnn_state_encoder import build_rnn_state_encoder
from habitat_baselines.utils.common import CategoricalNet, GaussianNet, get_num_actions

from .st_gcn import GNNFeatureExtractor, SocialAwareFusion, SpatioTemporalGCN
from .graph_builder import DynamicSpatioTemporalGraph


@baseline_registry.register_policy
class VLNGNNPolicy(NetPolicy):
    """
    VLN Policy with Spatio-Temporal Graph Neural Network.
    
    This policy extends the standard VLN policy with:
    - Dynamic pedestrian detection and tracking
    - Spatio-temporal graph construction
    - ST-GCN for social-aware feature extraction
    - Multi-modal fusion with visual and language features
    
    The GNN components are optional and can be disabled for ablation studies.
    """
    
    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space,
        hidden_size: int = 512,
        gnn_hidden_dim: int = 128,
        gnn_output_dim: int = 128,
        use_gnn: bool = True,
        use_gat: bool = True,
        num_spatial_layers: int = 2,
        num_temporal_layers: int = 2,
        max_pedestrians: int = 10,
        max_history_len: int = 8,
        distance_threshold: float = 3.0,
        policy_config: Optional[Any] = None,
        aux_loss_config: Optional[Any] = None,
        **kwargs,
    ):
        """
        Initialize VLN GNN Policy.
        
        Args:
            observation_space: Observation space
            action_space: Action space
            hidden_size: Hidden dimension for main network
            gnn_hidden_dim: Hidden dimension for GNN
            gnn_output_dim: Output dimension for GNN features
            use_gnn: Whether to use GNN processing
            use_gat: Whether to use Graph Attention
            num_spatial_layers: Number of spatial GCN layers
            num_temporal_layers: Number of temporal TCN layers
            max_pedestrians: Maximum pedestrians to track
            max_history_len: Maximum temporal history length
            distance_threshold: Distance threshold for edges
            policy_config: Policy configuration
            aux_loss_config: Auxiliary loss configuration
        """
        # Initialize the GNN-augmented network
        net = VLNGNNNet(
            observation_space=observation_space,
            action_space=action_space,
            hidden_size=hidden_size,
            gnn_hidden_dim=gnn_hidden_dim,
            gnn_output_dim=gnn_output_dim,
            use_gnn=use_gnn,
            use_gat=use_gat,
            num_spatial_layers=num_spatial_layers,
            num_temporal_layers=num_temporal_layers,
            max_pedestrians=max_pedestrians,
            max_history_len=max_history_len,
            distance_threshold=distance_threshold,
            policy_config=policy_config,
        )
        
        # Initialize parent policy
        super().__init__(
            net=net,
            action_space=action_space,
            policy_config=policy_config,
            aux_loss_config=aux_loss_config,
        )
        
        # Store GNN configuration
        self.use_gnn = use_gnn
        self.gnn_hidden_dim = gnn_hidden_dim
        self.gnn_output_dim = gnn_output_dim
    
    @classmethod
    def from_config(cls, config, observation_space, action_space, **kwargs):
        """Create VLNGNNPolicy from configuration."""
        gnn_config = config.habitat_baselines.rl.get("gnn", {})
        
        return cls(
            observation_space=observation_space,
            action_space=action_space,
            hidden_size=config.habitat_baselines.rl.ppo.hidden_size,
            gnn_hidden_dim=gnn_config.get("hidden_dim", 128),
            gnn_output_dim=gnn_config.get("output_dim", 128),
            use_gnn=gnn_config.get("use_gnn", True),
            use_gat=gnn_config.get("use_gat", True),
            num_spatial_layers=gnn_config.get("num_spatial_layers", 2),
            num_temporal_layers=gnn_config.get("num_temporal_layers", 2),
            max_pedestrians=gnn_config.get("max_pedestrians", 10),
            max_history_len=gnn_config.get("max_history_len", 8),
            distance_threshold=gnn_config.get("distance_threshold", 3.0),
            policy_config=config.habitat_baselines.rl.policy.get("default", None),
            aux_loss_config=config.habitat_baselines.rl.auxiliary_losses,
        )


class VLNGNNNet(nn.Module):
    """
    VLN Network with GNN integration.
    
    This network extends the standard VLN network with spatio-temporal
    graph processing for social-aware navigation.
    """
    
    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space,
        hidden_size: int = 512,
        gnn_hidden_dim: int = 128,
        gnn_output_dim: int = 128,
        use_gnn: bool = True,
        use_gat: bool = True,
        num_spatial_layers: int = 2,
        num_temporal_layers: int = 2,
        max_pedestrians: int = 10,
        max_history_len: int = 8,
        distance_threshold: float = 3.0,
        policy_config: Optional[Any] = None,
        **kwargs,
    ):
        """
        Initialize VLN GNN Network.
        
        Args:
            observation_space: Observation space
            action_space: Action space
            hidden_size: Hidden dimension
            gnn_hidden_dim: GNN hidden dimension
            gnn_output_dim: GNN output dimension
            use_gnn: Whether to use GNN
            use_gat: Whether to use GAT
            num_spatial_layers: Number of spatial layers
            num_temporal_layers: Number of temporal layers
            max_pedestrians: Maximum pedestrians
            max_history_len: History length
            distance_threshold: Edge distance threshold
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.use_gnn = use_gnn
        self.gnn_output_dim = gnn_output_dim
        
        # GNN components (optional)
        if use_gnn:
            # GNN feature extractor
            self.gnn_extractor = GNNFeatureExtractor(
                robot_feature_dim=8,  # [x, y, z, sin_h, cos_h, goal_x, goal_y, timestep]
                pedestrian_feature_dim=14,  # Relative position, distance, velocity, etc.
                hidden_dim=gnn_hidden_dim,
                output_dim=gnn_output_dim,
                num_spatial_layers=num_spatial_layers,
                num_temporal_layers=num_temporal_layers,
                num_heads=4,
                dropout=0.1,
                use_gat=use_gat,
            )
            
            # Graph builder (per-environment state)
            self.graph_builder = None  # Will be created per environment
            
            # Social-aware fusion
            self.social_fusion = SocialAwareFusion(
                visual_dim=hidden_size,  # Main visual feature dim
                language_dim=hidden_size,
                gnn_dim=gnn_output_dim,
                hidden_dim=hidden_size,
                num_heads=4,
                dropout=0.1,
            )
        
        # Main network components (similar to PointNavResNetNet)
        self._setup_base_network(observation_space, action_space, policy_config)
    
    def _setup_base_network(
        self,
        observation_space: spaces.Dict,
        action_space,
        policy_config: Optional[Any],
    ):
        """Setup base network components."""
        from habitat.tasks.nav.nav import (
            IntegratedPointGoalGPSAndCompassSensor,
            EpisodicCompassSensor,
            EpisodicGPSSensor,
            PointGoalSensor,
            HeadingSensor,
            ProximitySensor,
            ImageGoalSensor,
        )
        
        # Determine action distribution type
        if policy_config is not None:
            self.action_distribution_type = getattr(
                policy_config, "action_distribution_type", "categorical"
            )
        else:
            self.action_distribution_type = "categorical"
        
        # Previous action embedding
        self._n_prev_action = 32
        if self.action_distribution_type == "categorical":
            self.prev_action_embedding = nn.Embedding(
                action_space.n + 1, self._n_prev_action
            )
        else:
            num_actions = get_num_actions(action_space)
            self.prev_action_embedding = nn.Linear(
                num_actions, self._n_prev_action
            )
        
        # 1D state fusion keys (exclude GNN-related sensors)
        exclude_keys = [
            "gnn_node_features",
            "gnn_edge_index", 
            "gnn_edge_weight",
            "gnn_node_mask",
            "pedestrian_detection",
        ]
        
        fuse_keys = [
            k for k in observation_space.spaces.keys()
            if len(observation_space.spaces[k].shape) == 1
            and k not in exclude_keys
        ]
        
        self._fuse_keys_1d = fuse_keys
        
        rnn_input_size = self._n_prev_action
        if len(self._fuse_keys_1d) > 0:
            rnn_input_size += sum(
                observation_space.spaces[k].shape[0]
                for k in self._fuse_keys_1d
            )
        
        # Goal encoding
        self._n_input_goal = 0
        if IntegratedPointGoalGPSAndCompassSensor.cls_uuid in observation_space.spaces:
            n_input_goal = observation_space.spaces[
                IntegratedPointGoalGPSAndCompassSensor.cls_uuid
            ].shape[0]
            self.tgt_embedding = nn.Linear(n_input_goal + 1, 32)
            rnn_input_size += 32
            self._n_input_goal = n_input_goal
        
        # GPS embedding
        if EpisodicGPSSensor.cls_uuid in observation_space.spaces:
            self.gps_embedding = nn.Linear(2, 32)
            rnn_input_size += 32
        
        # Compass embedding
        if EpisodicCompassSensor.cls_uuid in observation_space.spaces:
            self.compass_embedding = nn.Linear(2, 32)
            rnn_input_size += 32
        
        # Visual encoder (expected to be set externally)
        self.visual_encoder = None
        self.visual_fc = None
        self._visual_feature_size = 0
        
        # GNN feature integration
        if self.use_gnn:
            rnn_input_size += self.gnn_output_dim
        
        # State encoder (RNN)
        self.state_encoder = build_rnn_state_encoder(
            rnn_input_size,
            self.hidden_size,
            rnn_type="GRU",
            num_layers=1,
        )
        
        self._hidden_size = self.hidden_size
    
    @property
    def output_size(self):
        return self._hidden_size
    
    @property
    def is_blind(self):
        return self.visual_encoder is None or self.visual_encoder.is_blind
    
    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers
    
    @property
    def recurrent_hidden_size(self):
        return self._hidden_size
    
    @property
    def perception_embedding_size(self):
        return self._hidden_size
    
    def init_graph_builder(
        self,
        max_pedestrians: int = 10,
        max_history_len: int = 8,
        distance_threshold: float = 3.0,
        device: str = "cuda",
    ):
        """Initialize graph builder for a new environment."""
        if self.use_gnn:
            self.graph_builder = DynamicSpatioTemporalGraph(
                max_pedestrians=max_pedestrians,
                max_history_len=max_history_len,
                distance_threshold=distance_threshold,
                edge_distance_threshold=distance_threshold,
                device=device,
            )
    
    def reset_graph(self):
        """Reset graph state."""
        if self.graph_builder is not None:
            self.graph_builder.reset()
    
    def forward(
        self,
        observations: Dict[str, torch.Tensor],
        rnn_hidden_states: torch.Tensor,
        prev_actions: torch.Tensor,
        masks: torch.Tensor,
        rnn_build_seq_info: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass of VLN GNN Network.
        
        Args:
            observations: Observation dictionary
            rnn_hidden_states: RNN hidden states
            prev_actions: Previous actions
            masks: Episode masks
            rnn_build_seq_info: RNN sequence info
            
        Returns:
            Tuple of (output_features, rnn_hidden_states, aux_loss_state)
        """
        aux_loss_state = {}
        x = []
        
        # Visual features
        if not self.is_blind:
            if hasattr(self.visual_encoder, "forward"):
                visual_feats = self.visual_encoder(observations)
                if hasattr(self, "visual_fc") and self.visual_fc is not None:
                    visual_feats = self.visual_fc(visual_feats)
            else:
                visual_feats = observations.get("rgb_features", None)
                if visual_feats is None:
                    visual_feats = torch.zeros(
                        observations[list(observations.keys())[0]].shape[0],
                        self._hidden_size,
                        device=observations[list(observations.keys())[0]].device,
                    )
            
            aux_loss_state["visual_features"] = visual_feats
            x.append(visual_feats)
        
        # 1D state features
        if len(self._fuse_keys_1d) > 0:
            fuse_states = torch.cat(
                [observations[k] for k in self._fuse_keys_1d if k in observations],
                dim=-1
            ).float()
            x.append(fuse_states)
        
        # Goal features
        goal_features = self._encode_goal(observations)
        if goal_features is not None:
            x.append(goal_features)
        
        # GNN features (if enabled)
        gnn_features = None
        if self.use_gnn and self.graph_builder is not None:
            gnn_features = self._process_gnn(observations)
            if gnn_features is not None:
                x.append(gnn_features)
                aux_loss_state["gnn_features"] = gnn_features
        
        # Previous action
        if self.action_distribution_type == "categorical":
            prev_actions = prev_actions.squeeze(-1)
            start_token = torch.zeros_like(prev_actions)
            prev_actions = self.prev_action_embedding(
                torch.where(masks.view(-1) > 0, prev_actions + 1, start_token)
            )
        else:
            prev_actions = self.prev_action_embedding(
                masks * prev_actions.float()
            )
        x.append(prev_actions)
        
        # Concatenate and process through RNN
        out = torch.cat(x, dim=1)
        out, rnn_hidden_states = self.state_encoder(
            out, rnn_hidden_states, masks, rnn_build_seq_info
        )
        
        aux_loss_state["rnn_output"] = out
        
        return out, rnn_hidden_states, aux_loss_state
    
    def _encode_goal(self, observations: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
        """Encode goal features."""
        from habitat.tasks.nav.nav import IntegratedPointGoalGPSAndCompassSensor, EpisodicCompassSensor, EpisodicGPSSensor
        
        features = []
        
        # Integrated goal
        if IntegratedPointGoalGPSAndCompassSensor.cls_uuid in observations:
            goal_obs = observations[IntegratedPointGoalGPSAndCompassSensor.cls_uuid]
            if goal_obs.shape[1] == 2:
                goal_obs = torch.stack([
                    goal_obs[:, 0],
                    torch.cos(-goal_obs[:, 1]),
                    torch.sin(-goal_obs[:, 1]),
                ], -1)
            features.append(self.tgt_embedding(goal_obs))
        
        # GPS
        if EpisodicGPSSensor.cls_uuid in observations:
            gps = observations[EpisodicGPSSensor.cls_uuid]
            features.append(self.gps_embedding(gps))
        
        # Compass
        if EpisodicCompassSensor.cls_uuid in observations:
            compass = observations[EpisodicCompassSensor.cls_uuid]
            compass_enc = torch.stack([
                torch.cos(compass),
                torch.sin(compass),
            ], -1)
            features.append(self.compass_embedding(compass_enc))
        
        if len(features) == 0:
            return None
        
        return torch.cat(features, dim=-1)
    
    def _process_gnn(
        self,
        observations: Dict[str, torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """
        Process GNN features from observations.
        
        Expects observations to contain:
        - pedestrian_detection: List of detections or preprocessed graph data
        - Or GNN-related tensors if precomputed
        """
        device = observations[list(observations.keys())[0]].device
        
        # Check if precomputed GNN features exist
        if "gnn_features" in observations:
            return observations["gnn_features"]
        
        # Get pedestrian detection data
        pedestrian_data = observations.get("pedestrian_detection", None)
        
        if pedestrian_data is None or len(pedestrian_data) == 0:
            # No pedestrians, return zeros
            batch_size = observations[list(observations.keys())[0]].shape[0]
            return torch.zeros(batch_size, self.gnn_output_dim, device=device)
        
        # Extract features for GNN
        # This would be called from the sensor or trainer
        # For now, return placeholder
        batch_size = observations[list(observations.keys())[0]].shape[0]
        return torch.zeros(batch_size, self.gnn_output_dim, device=device)
    
    def get_gnn_features_from_observations(
        self,
        rgb: torch.Tensor,
        depth: torch.Tensor,
        robot_position: torch.Tensor,
        robot_heading: torch.Tensor,
        goal: Optional[torch.Tensor] = None,
        pedestrian_detections: Optional[List[Dict]] = None,
    ) -> torch.Tensor:
        """
        Get GNN features from raw observations.
        
        This method can be called externally to process observations
        through the GNN pipeline.
        
        Args:
            rgb: RGB image [B, H, W, C]
            depth: Depth image [B, H, W]
            robot_position: Robot position [B, 3]
            robot_heading: Robot heading [B]
            goal: Goal position [B, 2] (optional)
            pedestrian_detections: List of detections per sample
            
        Returns:
            GNN features [B, gnn_output_dim]
        """
        if not self.use_gnn or self.graph_builder is None:
            batch_size = rgb.shape[0]
            return torch.zeros(batch_size, self.gnn_output_dim, device=rgb.device)
        
        device = rgb.device
        batch_size = rgb.shape[0]
        gnn_features_list = []
        
        for b in range(batch_size):
            # Get robot info
            robot_pos = robot_position[b].cpu().numpy()
            robot_head = robot_heading[b].item()
            goal_pos = goal[b].cpu().numpy() if goal is not None else None
            
            # Get detections for this sample
            detections = []
            if pedestrian_detections is not None and b < len(pedestrian_detections):
                detections = pedestrian_detections[b]
            
            # Update graph
            graph_output = self.graph_builder.update(
                robot_position=robot_pos,
                robot_heading=robot_head,
                yolo_detections=detections,
                goal_position=goal_pos,
            )
            
            # Get GNN features
            with torch.no_grad():
                robot_feat = torch.from_numpy(
                    graph_output.robot_features
                ).float().to(device)
                
                ped_feats = torch.zeros(
                    self.graph_builder.max_pedestrians,
                    self.graph_builder.PEDESTRIAN_FEATURE_DIM,
                    device=device
                )
                
                num_ped = 0
                for i, node in enumerate(graph_output.pedestrian_nodes.values()):
                    if num_ped >= self.graph_builder.max_pedestrians:
                        break
                    ped_feats[num_ped] = torch.from_numpy(
                        self.graph_builder._build_pedestrian_features(
                            node, robot_pos
                        )
                    )
                    num_ped += 1
                
                # Process through GNN
                gnn_feat, _ = self.gnn_extractor(
                    robot_features=robot_feat,
                    pedestrian_features=ped_feats[:num_ped] if num_ped > 0 else ped_feats,
                    edge_index=graph_output.edge_index,
                    edge_weight=graph_output.edge_weights,
                    node_mask=graph_output.node_mask,
                )
                
                gnn_features_list.append(gnn_feat.squeeze(0))
        
        return torch.stack(gnn_features_list, dim=0)


class VLNGNNPolicyWithDetection(VLNGNNPolicy):
    """
    VLN GNN Policy with integrated pedestrian detection.
    
    This variant includes YOLO detection internally.
    """
    
    def __init__(self, *args, use_integrated_detection: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        
        if use_integrated_detection:
            from .yolo_detector import YOLOPedestrianDetector
            self.detector = YOLOPedestrianDetector(
                model_path="yolov8n.pt",
                confidence_threshold=0.5,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
        else:
            self.detector = None
    
    def detect_and_process(
        self,
        rgb_frame: np.ndarray,
        depth_frame: Optional[np.ndarray],
        robot_position: np.ndarray,
        robot_heading: float,
        goal: Optional[np.ndarray] = None,
    ) -> torch.Tensor:
        """
        Detect pedestrians and get GNN features.
        
        Args:
            rgb_frame: RGB frame
            depth_frame: Depth frame
            robot_position: Robot position
            robot_heading: Robot heading
            goal: Goal position
            
        Returns:
            GNN features
        """
        if self.detector is None:
            return torch.zeros(1, self.gnn_output_dim)
        
        # Detect pedestrians
        detections = self.detector.detect(rgb_frame, depth_frame)
        
        # Convert to tensor format
        rgb_t = torch.from_numpy(rgb_frame).float().unsqueeze(0) / 255.0
        depth_t = torch.from_numpy(depth_frame).float().unsqueeze(0) if depth_frame is not None else None
        robot_pos_t = torch.from_numpy(robot_position).float().unsqueeze(0)
        robot_head_t = torch.tensor([[robot_heading]], dtype=torch.float32)
        goal_t = torch.from_numpy(goal).float().unsqueeze(0) if goal is not None else None
        
        return self.get_gnn_features_from_observations(
            rgb=rgb_t,
            depth=depth_t,
            robot_position=robot_pos_t,
            robot_heading=robot_head_t,
            goal=goal_t,
            pedestrian_detections=[detections],
        )
