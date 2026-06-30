#!/usr/bin/env python3
"""
Spatio-Temporal Graph Neural Network Module for Dynamic Pedestrian-Aware VLN.

This module provides spatio-temporal graph construction and ST-GNN processing
for dynamic pedestrian detection and tracking in vision-language navigation.

Key Components:
- Dynamic graph builder for real-time pedestrian tracking
- ST-GCN for spatio-temporal feature extraction
- GNN-integrated policy for VLN with social awareness
- YOLO pedestrian detector integration
- Habitat sensors for environment integration

Quick Start:
    from habitat_baselines.rl.spatio_temporal_gnn import (
        DynamicSpatioTemporalGraph,
        SpatioTemporalGCN,
        VLNGNNPolicy,
        YOLOPedestrianDetector,
    )

Example Usage:
    # Create graph builder
    graph_builder = DynamicSpatioTemporalGraph(max_pedestrians=10)
    
    # Update with detections
    graph_output = graph_builder.update(
        robot_position=robot_pos,
        robot_heading=robot_heading,
        yolo_detections=detections,
    )
    
    # Process through ST-GCN
    gnn_features = st_gcn(graph_output.node_features, graph_output.edge_index)

Author: DPED-PRO
Date: 2024
"""

from .graph_builder import (
    DynamicSpatioTemporalGraph,
    PedestrianNode,
    SpatioTemporalGraphOutput,
    SimpleTracker,
)

from .st_gcn import (
    SpatioTemporalGCN,
    SpatialGraphConv,
    GraphAttentionLayer,
    TemporalConvLayer,
    SocialAwareFusion,
    GNNFeatureExtractor,
)

from .yolo_detector import (
    YOLOPedestrianDetector,
    DetectionResult,
    SimpleIoUTracker,
)

from .gnn_vln_policy import (
    VLNGNNPolicy,
    VLNGNNNet,
    VLNGNNPolicyWithDetection,
)

from .gnn_sensors import (
    PedestrianDetectionSensor,
    SpatioTemporalGraphSensor,
    GNNFeatureSensor,
)

from .config import (
    GNNConfig,
    DetectionConfig,
    FusionConfig,
    RewardShapingConfig,
    VLNGNNConfig,
    get_default_config,
)

__all__ = [
    # Graph Builder
    "DynamicSpatioTemporalGraph",
    "PedestrianNode",
    "SpatioTemporalGraphOutput",
    "SimpleTracker",
    
    # ST-GCN
    "SpatioTemporalGCN",
    "SpatialGraphConv",
    "GraphAttentionLayer",
    "TemporalConvLayer",
    "SocialAwareFusion",
    "GNNFeatureExtractor",
    
    # YOLO Detector
    "YOLOPedestrianDetector",
    "DetectionResult",
    "SimpleIoUTracker",
    
    # Policy
    "VLNGNNPolicy",
    "VLNGNNNet",
    "VLNGNNPolicyWithDetection",
    
    # Sensors
    "PedestrianDetectionSensor",
    "SpatioTemporalGraphSensor",
    "GNNFeatureSensor",
    
    # Config
    "GNNConfig",
    "DetectionConfig",
    "FusionConfig",
    "RewardShapingConfig",
    "VLNGNNConfig",
    "get_default_config",
]

__version__ = "1.0.0"
