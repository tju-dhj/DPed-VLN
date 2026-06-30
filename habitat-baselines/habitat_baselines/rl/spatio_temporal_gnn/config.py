#!/usr/bin/env python3
"""
Configuration for Spatio-Temporal GNN module.

This module provides configuration classes and YAML config examples
for the GNN-based VLN policy.

Author: DPED-PRO
Date: 2024
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from omegaconf import OmegaConf


@dataclass
class GNNConfig:
    """Configuration for GNN module."""
    
    # Enable/disable GNN
    use_gnn: bool = True
    
    # Use Graph Attention Network instead of GCN
    use_gat: bool = True
    
    # Hidden dimension
    hidden_dim: int = 128
    
    # Output dimension
    output_dim: int = 128
    
    # Number of GNN layers
    num_spatial_layers: int = 2
    num_temporal_layers: int = 2
    
    # Number of attention heads (for GAT)
    num_heads: int = 4
    
    # Dropout
    dropout: float = 0.1
    
    # Pedestrian tracking
    max_pedestrians: int = 10
    max_history_len: int = 8
    distance_threshold: float = 3.0  # meters
    edge_distance_threshold: float = 3.0  # meters
    disappearance_threshold: int = 5  # frames


@dataclass
class DetectionConfig:
    """Configuration for pedestrian detection."""
    
    # YOLO model
    model_path: str = "yolov8n.pt"
    
    # Detection thresholds
    confidence_threshold: float = 0.5
    nms_threshold: float = 0.4
    max_detections: int = 20
    
    # Tracking
    use_tracking: bool = True
    iou_threshold: float = 0.3
    max_age: int = 30
    
    # Camera intrinsics
    fx: float = 525.0
    fy: float = 525.0
    cx: float = 319.5
    cy: float = 239.5


@dataclass
class FusionConfig:
    """Configuration for multi-modal fusion."""
    
    # Visual feature dimension
    visual_dim: int = 512
    
    # Language feature dimension
    language_dim: int = 512
    
    # GNN feature dimension
    gnn_dim: int = 128
    
    # Fusion hidden dimension
    hidden_dim: int = 256
    
    # Number of attention heads
    num_heads: int = 4
    
    # Dropout
    dropout: float = 0.1


@dataclass
class RewardShapingConfig:
    """Configuration for GNN-based reward shaping."""
    
    # Collision penalty
    collision_penalty: float = -0.5
    
    # Pedestrian proximity penalty
    pedestrian_proximity_penalty: float = -0.1
    pedestrian_proximity_threshold: float = 1.5  # meters
    
    # Safe distance reward
    safe_distance_reward: float = 0.05
    safe_distance_threshold: float = 3.0  # meters
    
    # Success bonus
    success_reward: float = 10.0
    
    # Social compliance reward
    social_compliance_reward: float = 0.01
    
    # Pedestrian avoidance reward (encourage staying away)
    avoidance_reward_coef: float = 0.02


@dataclass
class VLNGNNConfig:
    """Complete configuration for VLN GNN module."""
    
    gnn: GNNConfig = field(default_factory=GNNConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    reward: RewardShapingConfig = field(default_factory=RewardShapingConfig)
    
    # Integration settings
    integrate_with_existing_policy: bool = True
    freeze_gnn_until_step: int = 0  # 0 = no freezing


def get_default_config() -> VLNGNNConfig:
    """Get default VLN GNN configuration."""
    return VLNGNNConfig()


def config_to_dict(config: VLNGNNConfig) -> Dict[str, Any]:
    """Convert config to dictionary."""
    return OmegaConf.structured(config).to_container()


def config_to_yaml(config: VLNGNNConfig) -> str:
    """Convert config to YAML string."""
    return OmegaConf.to_yaml(OmegaConf.structured(config))


# Example YAML configuration
EXAMPLE_YAML_CONFIG = """
# VLN GNN Configuration Example
# Add this to your main config file under habitat_baselines.rl.gnn

gnn:
  # Enable GNN processing
  use_gnn: true
  
  # Use Graph Attention Network (recommended)
  use_gat: true
  
  # Network dimensions
  hidden_dim: 128
  output_dim: 128
  
  # GNN architecture
  num_spatial_layers: 2
  num_temporal_layers: 2
  num_heads: 4
  dropout: 0.1
  
  # Pedestrian tracking parameters
  max_pedestrians: 10
  max_history_len: 8
  distance_threshold: 3.0
  edge_distance_threshold: 3.0
  disappearance_threshold: 5

# Pedestrian detection
detection:
  model_path: "yolov8n.pt"
  confidence_threshold: 0.5
  nms_threshold: 0.4
  max_detections: 20
  use_tracking: true
  iou_threshold: 0.3
  max_age: 30

# Multi-modal fusion
fusion:
  visual_dim: 512
  language_dim: 512
  gnn_dim: 128
  hidden_dim: 256
  num_heads: 4
  dropout: 0.1

# Reward shaping
reward:
  collision_penalty: -0.5
  pedestrian_proximity_penalty: -0.1
  pedestrian_proximity_threshold: 1.5
  safe_distance_reward: 0.05
  safe_distance_threshold: 3.0
  success_reward: 10.0
  social_compliance_reward: 0.01
  avoidance_reward_coef: 0.02
"""


if __name__ == "__main__":
    # Print example config
    print(EXAMPLE_YAML_CONFIG)
