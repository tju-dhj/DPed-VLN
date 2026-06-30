#!/usr/bin/env python3
"""
GNN-aware Sensors for Habitat Environment.

This module provides custom sensors for the Habitat environment that:
- Detect pedestrians using YOLO
- Build spatio-temporal graphs
- Extract GNN features for the policy network

Author: DPED-PRO
Date: 2024
"""

import numpy as np
import torch
from typing import Dict, Any, Optional, List
from gym import spaces

from habitat.core.simulator import  Sensor


from .yolo_detector import YOLOPedestrianDetector
from .graph_builder import DynamicSpatioTemporalGraph


class PedestrianDetectionSensor(Sensor):
    """
    Sensor that detects pedestrians using YOLO and provides detection results.
    
    This sensor:
    1. Receives RGB and Depth images from the simulator
    2. Runs YOLO detection for pedestrians
    3. Associates detections across frames using tracking
    4. Returns detection results as observation
    """
    
    uuid = "pedestrian_detection"
    observation_space = spaces.Dict({
        "num_detections": spaces.Box(low=0, high=20, shape=(1,), dtype=np.int32),
        "bboxes": spaces.Box(low=0, high=1, shape=(10, 4), dtype=np.float32),
        "confidences": spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32),
        "track_ids": spaces.Box(low=-1, high=1000, shape=(10,), dtype=np.int32),
        "depths": spaces.Box(low=0, high=10, shape=(10,), dtype=np.float32),
    })
    
    def __init__(
        self,
        sim: Any,
        config: Dict[str, Any],
        **kwargs,
    ):
        """
        Initialize pedestrian detection sensor.
        
        Args:
            sim: Habitat simulator instance
            config: Sensor configuration
        """
        self.sim = sim
        self.config = config
        
        # Detector
        self.detector = YOLOPedestrianDetector(
            model_path=config.get("model_path", "yolov8n.pt"),
            confidence_threshold=config.get("confidence_threshold", 0.5),
            nms_threshold=config.get("nms_threshold", 0.4),
            max_detections=config.get("max_detections", 20),
            use_tracking=config.get("use_tracking", True),
            device=config.get("device", "cuda" if torch.cuda.is_available() else "cpu"),
        )
        
        # Track history
        self.detection_history: List[Dict] = []
    
    def get_observation(
        self,
        observations: Dict[str, Any],
        *args: Any,
        **kwargs: Any,
    ) -> Dict[str, np.ndarray]:
        """
        Get pedestrian detection observation.
        
        Args:
            observations: Current observations from simulator
            
        Returns:
            Detection results dictionary
        """
        # Get RGB and depth
        rgb_key = self.config.get("rgb_sensor", "rgb")
        depth_key = self.config.get("depth_sensor", "depth")
        
        rgb = observations.get(rgb_key)
        depth = observations.get(depth_key)
        
        if rgb is None:
            # Return empty detection
            return self._empty_observation()
        
        # Ensure RGB is in correct format
        if isinstance(rgb, torch.Tensor):
            rgb = rgb.cpu().numpy()
        
        # Convert from CHW to HWC if needed
        if rgb.ndim == 3 and rgb.shape[0] == 3:
            rgb = rgb.transpose(1, 2, 0)
        
        # Convert to uint8 if needed
        if rgb.dtype != np.uint8:
            if rgb.max() <= 1.0:
                rgb = (rgb * 255).astype(np.uint8)
            else:
                rgb = rgb.astype(np.uint8)
        
        # Get depth
        depth_np = None
        if depth is not None:
            if isinstance(depth, torch.Tensor):
                depth = depth.cpu().numpy()
            if depth.ndim == 3:
                depth = depth[:, :, 0]  # Take first channel
            depth_np = depth.astype(np.float32)
        
        # Run detection
        detections = self.detector.detect(rgb, depth_np)
        
        # Format observation
        return self._format_observation(detections)
    
    def _empty_observation(self) -> Dict[str, np.ndarray]:
        """Return empty observation."""
        return {
            "num_detections": np.array([0], dtype=np.int32),
            "bboxes": np.zeros((10, 4), dtype=np.float32),
            "confidences": np.zeros(10, dtype=np.float32),
            "track_ids": np.full(10, -1, dtype=np.int32),
            "depths": np.zeros(10, dtype=np.float32),
        }
    
    def _format_observation(self, detections: List[Dict]) -> Dict[str, np.ndarray]:
        """Format detections into observation space."""
        num_det = len(detections)
        
        # Pad to max detections
        bboxes = np.zeros((10, 4), dtype=np.float32)
        confidences = np.zeros(10, dtype=np.float32)
        track_ids = np.full(10, -1, dtype=np.int32)
        depths = np.zeros(10, dtype=np.float32)
        
        h, w = 480, 640  # Default size, will be normalized
        
        for i, det in enumerate(detections[:10]):
            bbox = det['bbox']
            # Normalize to [0, 1]
            bboxes[i] = np.array([
                bbox[0] / w, bbox[1] / h,
                bbox[2] / w, bbox[3] / h,
            ], dtype=np.float32)
            confidences[i] = det.get('confidence', 1.0)
            track_ids[i] = det.get('track_id', i)
            depths[i] = det.get('depth', 0.0)
        
        return {
            "num_detections": np.array([num_det], dtype=np.int32),
            "bboxes": bboxes,
            "confidences": confidences,
            "track_ids": track_ids,
            "depths": depths,
        }
    
    def reset(self):
        """Reset sensor state."""
        self.detector.reset()
        self.detection_history.clear()


class SpatioTemporalGraphSensor(Sensor):
    """
    Sensor that builds spatio-temporal graphs from pedestrian detections.
    
    This sensor:
    1. Receives pedestrian detections from PedestrianDetectionSensor
    2. Receives robot state (position, heading)
    3. Builds and maintains a spatio-temporal graph
    4. Returns graph tensors for the policy network
    """
    
    uuid = "spatio_temporal_graph"
    
    def __init__(
        self,
        sim: Any,
        config: Dict[str, Any],
        **kwargs,
    ):
        """
        Initialize spatio-temporal graph sensor.
        
        Args:
            sim: Habitat simulator instance
            config: Sensor configuration
        """
        self.sim = sim
        self.config = config
        
        # Graph builder
        self.graph_builder = DynamicSpatioTemporalGraph(
            max_pedestrians=config.get("max_pedestrians", 10),
            max_history_len=config.get("max_history_len", 8),
            distance_threshold=config.get("distance_threshold", 5.0),
            edge_distance_threshold=config.get("edge_distance_threshold", 3.0),
            disappearance_threshold=config.get("disappearance_threshold", 5),
            device=config.get("device", "cuda" if torch.cuda.is_available() else "cpu"),
        )
        
        # Observation space (dynamic, but we use fixed max)
        max_nodes = config.get("max_pedestrians", 10) + 1  # +1 for robot
        max_edges = max_nodes * max_nodes
        
        self.observation_space = spaces.Dict({
            "node_features": spaces.Box(
                low=-100, high=100, shape=(max_nodes, 14), dtype=np.float32
            ),
            "edge_index": spaces.Box(
                low=0, high=max_nodes, shape=(2, max_edges), dtype=np.int64
            ),
            "edge_weights": spaces.Box(
                low=0, high=1, shape=(max_edges,), dtype=np.float32
            ),
            "node_mask": spaces.Box(
                low=0, high=1, shape=(max_nodes,), dtype=np.float32
            ),
            "num_pedestrians": spaces.Box(low=0, high=20, shape=(1,), dtype=np.int32),
        })
    
    def get_observation(
        self,
        observations: Dict[str, Any],
        *args: Any,
        **kwargs: Any,
    ) -> Dict[str, np.ndarray]:
        """
        Get spatio-temporal graph observation.
        
        Args:
            observations: Current observations from simulator
            
        Returns:
            Graph tensors
        """
        # Get robot state
        robot_position = self._get_robot_position()
        robot_heading = self._get_robot_heading()
        goal_position = self._get_goal_position()
        
        # Get pedestrian detections
        ped_det = observations.get("pedestrian_detection")
        
        detections = []
        if ped_det is not None:
            detections = self._parse_pedestrian_detection(ped_det)
        
        # Update graph
        graph_output = self.graph_builder.update(
            robot_position=robot_position,
            robot_heading=robot_heading,
            yolo_detections=detections,
            goal_position=goal_position,
        )
        
        # Format observation
        return self._format_graph_observation(graph_output)
    
    def _get_robot_position(self) -> np.ndarray:
        """Get robot position from simulator."""
        try:
            agent_state = self.sim.get_agent_state()
            position = agent_state.position
            return np.array([position[0], position[2], 0], dtype=np.float32)  # x, z, y -> x, y, z
        except:
            return np.zeros(3, dtype=np.float32)
    
    def _get_robot_heading(self) -> float:
        """Get robot heading from simulator."""
        try:
            agent_state = self.sim.get_agent_state()
            rotation = agent_state.rotation
            # Convert quaternion to heading
            heading = np.arctan2(2.0 * (rotation.y * rotation.w + rotation.x * rotation.z),
                                1.0 - 2.0 * (rotation.y * rotation.y + rotation.z * rotation.z))
            return float(heading)
        except:
            return 0.0
    
    def _get_goal_position(self) -> Optional[np.ndarray]:
        """Get goal position relative to robot."""
        try:
            if hasattr(self.sim, "get_goal_position"):
                goal_pos = self.sim.get_goal_position()
                robot_pos = self._get_robot_position()
                return goal_pos - robot_pos[:2]
        except:
            pass
        return None
    
    def _parse_pedestrian_detection(self, ped_det: Dict) -> List[Dict]:
        """Parse pedestrian detection observation."""
        detections = []
        
        num_det = int(ped_det.get("num_detections", [0])[0])
        
        if num_det == 0:
            return detections
        
        bboxes = ped_det["bboxes"][:num_det]
        confidences = ped_det["confidences"][:num_det]
        track_ids = ped_det["track_ids"][:num_det]
        depths = ped_det["depths"][:num_det]
        
        h, w = 480, 640  # Default, should match actual observation
        
        for i in range(num_det):
            bbox = bboxes[i]
            # Denormalize
            det = {
                'track_id': int(track_ids[i]),
                'bbox': [
                    bbox[0] * w, bbox[1] * h,
                    bbox[2] * w, bbox[3] * h,
                ],
                'confidence': float(confidences[i]),
                'depth': float(depths[i]),
            }
            detections.append(det)
        
        return detections
    
    def _format_graph_observation(
        self,
        graph_output,
    ) -> Dict[str, np.ndarray]:
        """Format graph output to observation."""
        max_nodes = self.config.get("max_pedestrians", 10) + 1
        max_edges = max_nodes * max_nodes
        
        # Pad tensors to fixed size
        node_features = np.zeros((max_nodes, 14), dtype=np.float32)
        edge_index = np.zeros((2, max_edges), dtype=np.int64)
        edge_weights = np.zeros(max_edges, dtype=np.float32)
        node_mask = np.zeros(max_nodes, dtype=np.float32)
        
        # Fill in actual values
        nf = graph_output.node_features.cpu().numpy()
        node_features[:nf.shape[0], :nf.shape[1]] = nf
        
        ei = graph_output.edge_index.cpu().numpy()
        if ei.shape[1] <= max_edges:
            edge_index[:, :ei.shape[1]] = ei
        
        ew = graph_output.edge_weights.cpu().numpy()
        edge_weights[:len(ew)] = ew
        
        nm = graph_output.node_mask.cpu().numpy()
        node_mask[:len(nm)] = nm
        
        return {
            "node_features": node_features,
            "edge_index": edge_index,
            "edge_weights": edge_weights,
            "node_mask": node_mask,
            "num_pedestrians": np.array([graph_output.num_pedestrians], dtype=np.int32),
        }
    
    def reset(self):
        """Reset sensor state."""
        self.graph_builder.reset()


class GNNFeatureSensor(Sensor):
    """
    Sensor that extracts GNN features for the policy network.
    
    This sensor combines:
    - Pedestrian detection (from PedestrianDetectionSensor)
    - Spatio-temporal graph (from SpatioTemporalGraphSensor)
    - GNN processing (ST-GCN)
    
    And outputs features ready for the policy network.
    """
    
    uuid = "gnn_features"
    
    def __init__(
        self,
        sim: Any,
        config: Dict[str, Any],
        **kwargs,
    ):
        """
        Initialize GNN feature sensor.
        
        Args:
            sim: Habitat simulator instance
            config: Sensor configuration
        """
        self.sim = sim
        self.config = config
        
        # Initialize components
        self.detector = YOLOPedestrianDetector(
            model_path=config.get("model_path", "yolov8n.pt"),
            confidence_threshold=config.get("confidence_threshold", 0.5),
            device=config.get("device", "cuda" if torch.cuda.is_available() else "cpu"),
        )
        
        self.graph_builder = DynamicSpatioTemporalGraph(
            max_pedestrians=config.get("max_pedestrians", 10),
            max_history_len=config.get("max_history_len", 8),
            distance_threshold=config.get("distance_threshold", 5.0),
            edge_distance_threshold=config.get("edge_distance_threshold", 3.0),
            device=config.get("device", "cuda" if torch.cuda.is_available() else "cpu"),
        )
        
        # GNN feature dimension
        self.feature_dim = config.get("feature_dim", 128)
        
        self.observation_space = spaces.Box(
            low=-10, high=10, shape=(self.feature_dim,), dtype=np.float32
        )
    
    def get_observation(
        self,
        observations: Dict[str, Any],
        *args: Any,
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Get GNN features observation.
        
        Args:
            observations: Current observations from simulator
            
        Returns:
            GNN features [feature_dim]
        """
        # Get RGB and depth
        rgb_key = self.config.get("rgb_sensor", "rgb")
        depth_key = self.config.get("depth_sensor", "depth")
        
        rgb = observations.get(rgb_key)
        depth = observations.get(depth_key)
        
        if rgb is None:
            return np.zeros(self.feature_dim, dtype=np.float32)
        
        # Process RGB
        if isinstance(rgb, torch.Tensor):
            rgb = rgb.cpu().numpy()
        if rgb.ndim == 3 and rgb.shape[0] == 3:
            rgb = rgb.transpose(1, 2, 0)
        if rgb.dtype != np.uint8:
            if rgb.max() <= 1.0:
                rgb = (rgb * 255).astype(np.uint8)
            else:
                rgb = rgb.astype(np.uint8)
        
        # Process depth
        depth_np = None
        if depth is not None:
            if isinstance(depth, torch.Tensor):
                depth = depth.cpu().numpy()
            if depth.ndim == 3:
                depth = depth[:, :, 0]
            depth_np = depth.astype(np.float32)
        
        # Get robot state
        robot_position = self._get_robot_position()
        robot_heading = self._get_robot_heading()
        goal_position = self._get_goal_position()
        
        # Detect pedestrians
        detections = self.detector.detect(rgb, depth_np)
        
        # Update graph
        graph_output = self.graph_builder.update(
            robot_position=robot_position,
            robot_heading=robot_heading,
            yolo_detections=detections,
            goal_position=goal_position,
        )
        
        # Extract features (simplified - actual GNN would be applied)
        features = self._extract_features(graph_output, robot_position, robot_heading)
        
        return features
    
    def _get_robot_position(self) -> np.ndarray:
        """Get robot position."""
        try:
            agent_state = self.sim.get_agent_state()
            position = agent_state.position
            return np.array([position[0], position[2], 0], dtype=np.float32)
        except:
            return np.zeros(3, dtype=np.float32)
    
    def _get_robot_heading(self) -> float:
        """Get robot heading."""
        try:
            agent_state = self.sim.get_agent_state()
            rotation = agent_state.rotation
            return float(np.arctan2(2.0 * (rotation.y * rotation.w + rotation.x * rotation.z),
                                   1.0 - 2.0 * (rotation.y * rotation.y + rotation.z * rotation.z)))
        except:
            return 0.0
    
    def _get_goal_position(self) -> Optional[np.ndarray]:
        """Get goal position."""
        try:
            if hasattr(self.sim, "get_goal_position"):
                goal_pos = self.sim.get_goal_position()
                robot_pos = self._get_robot_position()
                return goal_pos - robot_pos[:2]
        except:
            pass
        return None
    
    def _extract_features(
        self,
        graph_output,
        robot_position: np.ndarray,
        robot_heading: float,
    ) -> np.ndarray:
        """
        Extract GNN features from graph.
        
        This is a simplified feature extraction. For full GNN processing,
        use the SpatioTemporalGCN module.
        """
        features = np.zeros(self.feature_dim, dtype=np.float32)
        
        # Encode robot info
        import math
        features[0] = robot_position[0]
        features[1] = robot_position[1]
        features[2] = math.sin(robot_heading)
        features[3] = math.cos(robot_heading)
        
        # Encode pedestrian info
        num_ped = graph_output.num_pedestrians
        features[4] = num_ped
        
        # Encode closest pedestrian info
        if num_ped > 0:
            node_feats = graph_output.node_features.cpu().numpy()
            
            # Find closest pedestrian (distance is in column 3)
            if node_feats.shape[0] > 1:
                distances = node_feats[1:, 3]  # Skip robot node
                valid_distances = distances[distances > 0]
                
                if len(valid_distances) > 0:
                    closest_idx = np.argmin(valid_distances)
                    closest_ped = node_feats[1 + closest_idx]
                    
                    features[5] = closest_ped[0]  # rel_x
                    features[6] = closest_ped[1]  # rel_y
                    features[7] = closest_ped[3]  # distance
                    features[8] = closest_ped[4]  # direction
                    features[9] = closest_ped[5]  # speed
        
        # Encode graph structure
        if hasattr(graph_output, 'edge_index'):
            edge_idx = graph_output.edge_index.cpu().numpy()
            num_edges = edge_idx.shape[1]
            features[10] = num_edges
            features[11] = num_ped / 10.0  # Normalized pedestrian density
        
        # Encode temporal info
        features[12] = graph_output.timestep / 100.0
        
        return features
    
    def reset(self):
        """Reset sensor state."""
        self.detector.reset()
        self.graph_builder.reset()


def register_gnn_sensors():
    """Register GNN sensors with Habitat."""
    try:
        from habitat.core.registry import registry
        from habitat.core.simulator import Sensor
        from habitat.tasks.nav.nav import SimulatorTaskInfo
        
        # Register sensors
        registry.register_sensor_module(PedestrianDetectionSensor)
        registry.register_sensor_module(SpatioTemporalGraphSensor)
        registry.register_sensor_module(GNNFeatureSensor)
        
        return True
    except Exception as e:
        print(f"Warning: Could not register GNN sensors: {e}")
        return False
