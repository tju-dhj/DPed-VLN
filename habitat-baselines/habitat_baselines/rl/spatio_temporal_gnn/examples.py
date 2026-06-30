#!/usr/bin/env python3
"""
Usage Examples for Spatio-Temporal GNN Module.

This script demonstrates how to use the GNN module in various scenarios:
1. Standalone usage
2. Integration with Habitat environment
3. Integration with existing PPO training

Author: DPED-PRO
Date: 2024
"""

import numpy as np
import torch

# =============================================================================
# Example 1: Standalone Graph Builder Usage
# =============================================================================

def example_graph_builder():
    """Example of using the graph builder directly."""
    from habitat_baselines.rl.spatio_temporal_gnn.graph_builder import (
        DynamicSpatioTemporalGraph,
        SimpleTracker,
    )
    
    # Create graph builder
    graph_builder = DynamicSpatioTemporalGraph(
        max_pedestrians=10,
        max_history_len=8,
        distance_threshold=5.0,
        edge_distance_threshold=3.0,
    )
    
    # Simulate robot state
    robot_position = np.array([0.0, 0.0, 0.0])
    robot_heading = 0.0
    
    # Simulate YOLO detections
    yolo_detections = [
        {
            'track_id': 1,
            'bbox': [100, 100, 200, 300],
            'confidence': 0.9,
            'depth': 3.0,
        },
        {
            'track_id': 2,
            'bbox': [300, 150, 400, 350],
            'confidence': 0.85,
            'depth': 4.5,
        },
    ]
    
    # Update graph
    graph_output = graph_builder.update(
        robot_position=robot_position,
        robot_heading=robot_heading,
        yolo_detections=yolo_detections,
    )
    
    print("Example 1: Graph Builder")
    print(f"  Number of pedestrians: {graph_output.num_pedestrians}")
    print(f"  Node features shape: {graph_output.node_features.shape}")
    print(f"  Edge index shape: {graph_output.edge_index.shape}")
    print()


# =============================================================================
# Example 2: Standalone ST-GCN Usage
# =============================================================================

def example_st_gcn():
    """Example of using the ST-GCN directly."""
    from habitat_baselines.rl.spatio_temporal_gnn.st_gcn import SpatioTemporalGCN
    
    # Create ST-GCN
    st_gcn = SpatioTemporalGCN(
        in_channels=14,  # Pedestrian feature dimension
        hidden_channels=128,
        out_channels=128,
        num_spatial_layers=2,
        num_temporal_layers=2,
        num_heads=4,
        use_gat=True,
    )
    
    # Simulate graph data
    num_nodes = 5
    node_features = torch.randn(num_nodes, 14)
    edge_index = torch.tensor([
        [0, 0, 1, 1, 2, 2, 3, 3, 4, 4],  # source
        [1, 2, 0, 2, 0, 1, 0, 4, 0, 3],   # target
    ])
    edge_weight = torch.ones(edge_index.size(1))
    
    # Forward pass
    graph_embedding, _ = st_gcn(
        node_features=node_features,
        edge_index=edge_index,
        edge_weight=edge_weight,
    )
    
    print("Example 2: ST-GCN")
    print(f"  Node features: {node_features.shape}")
    print(f"  Graph embedding: {graph_embedding.shape}")
    print()


# =============================================================================
# Example 3: YOLO Detector Usage
# =============================================================================

def example_yolo_detector():
    """Example of using the YOLO detector."""
    from habitat_baselines.rl.spatio_temporal_gnn.yolo_detector import YOLOPedestrianDetector
    
    # Create detector
    detector = YOLOPedestrianDetector(
        model_path="yolov8n.pt",
        confidence_threshold=0.5,
        use_tracking=True,
    )
    
    # Simulate RGB frame
    rgb_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Simulate depth frame
    depth_frame = np.random.uniform(0.5, 5.0, (480, 640)).astype(np.float32)
    
    # Detect pedestrians
    detections = detector.detect(rgb_frame, depth_frame)
    
    print("Example 3: YOLO Detector")
    print(f"  Number of detections: {len(detections)}")
    for i, det in enumerate(detections):
        print(f"    Detection {i}: track_id={det['track_id']}, "
              f"bbox={det['bbox']}, depth={det['depth']:.2f}m")
    print()


# =============================================================================
# Example 4: Integration with Habitat Environment
# =============================================================================

def example_habitat_integration():
    """
    Example of integrating GNN sensors with Habitat environment.
    
    Add this to your Habitat config:
    
    habitat:
      environment:
        sensors:
          - type: PedestrianDetectionSensor
            rgb_sensor: "rgb"
            depth_sensor: "depth"
    
    habitat_baselines:
      rl:
        gnn:
          use_gnn: true
          hidden_dim: 128
          output_dim: 128
    """
    print("Example 4: Habitat Integration")
    print("  See config.yaml for Habitat configuration example")
    print()


# =============================================================================
# Example 5: Custom Reward Shaping with GNN
# =============================================================================

def example_reward_shaping():
    """Example of custom reward shaping based on GNN features."""
    from habitat_baselines.rl.spatio_temporal_gnn.config import RewardShapingConfig
    
    config = RewardShapingConfig(
        collision_penalty=-0.5,
        pedestrian_proximity_penalty=-0.1,
        pedestrian_proximity_threshold=1.5,
        safe_distance_reward=0.05,
        success_reward=10.0,
    )
    
    # Example reward calculation
    base_reward = 0.0
    pedestrian_distances = [1.0, 2.5, 3.5]  # Example distances
    
    for dist in pedestrian_distances:
        if dist < config.pedestrian_proximity_threshold:
            base_reward += config.pedestrian_proximity_penalty
        elif dist > config.safe_distance_threshold:
            base_reward += config.safe_distance_reward
    
    print("Example 5: Reward Shaping")
    print(f"  Config: collision_penalty={config.collision_penalty}")
    print(f"  Pedestrian distances: {pedestrian_distances}")
    print(f"  Calculated reward: {base_reward:.3f}")
    print()


# =============================================================================
# Example 6: Running the Examples
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Spatio-Temporal GNN Module - Usage Examples")
    print("=" * 60)
    print()
    
    # Run examples
    try:
        example_graph_builder()
    except Exception as e:
        print(f"Example 1 (Graph Builder) Error: {e}")
        print()
    
    try:
        example_st_gcn()
    except Exception as e:
        print(f"Example 2 (ST-GCN) Error: {e}")
        print()
    
    try:
        example_yolo_detector()
    except Exception as e:
        print(f"Example 3 (YOLO Detector) Error: {e}")
        print()
    
    example_habitat_integration()
    example_reward_shaping()
    
    print("=" * 60)
    print("Done!")
    print("=" * 60)
