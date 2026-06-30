#!/usr/bin/env python3
"""
Dynamic Spatio-Temporal Graph Builder for Pedestrian-Aware Navigation.

This module builds and maintains a dynamic spatio-temporal graph that represents
the spatial relationships and temporal evolution of pedestrians in the environment.

Key Features:
- Real-time pedestrian node creation/deletion based on YOLO detection
- Distance-based edge construction between pedestrians and robot
- Temporal history management for trajectory tracking
- Configurable thresholds for node appearance/disappearance

Author: DPED-PRO
Date: 2024
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import deque
import math


@dataclass
class PedestrianNode:
    """
    Represents a pedestrian node in the spatio-temporal graph.
    
    Attributes:
        track_id: Unique tracking ID across frames
        position_3d: 3D position in world coordinates [x, y, z]
        velocity: Velocity vector [vx, vy, vz]
        depth: Distance from robot camera
        bbox_2d: 2D bounding box [x1, y1, x2, y2]
        visible: Whether the pedestrian is currently visible
        last_seen_frame: Frame number when last detected
        confidence: Detection confidence score
    """
    track_id: int
    position_3d: np.ndarray
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    depth: float = 0.0
    bbox_2d: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    visible: bool = True
    last_seen_frame: int = 0
    confidence: float = 1.0


@dataclass
class SpatioTemporalGraphOutput:
    """
    Output from the spatio-temporal graph builder.
    
    Contains all necessary information for ST-GNN processing.
    """
    # Node features: [num_nodes, feature_dim]
    # Node types: 0=robot, 1+=pedestrian
    node_features: torch.Tensor
    
    # Edge indices: [2, num_edges]
    edge_index: torch.Tensor
    
    # Edge weights: [num_edges]
    edge_weights: torch.Tensor
    
    # Node mask: [num_nodes], True for valid nodes
    node_mask: torch.Tensor
    
    # Node types: [num_nodes], 0=robot, 1=pedestrian
    node_types: torch.Tensor
    
    # Robot features (first node)
    robot_features: torch.Tensor
    
    # Pedestrian count
    num_pedestrians: int
    
    # Timestep
    timestep: int


class DynamicSpatioTemporalGraph:
    """
    Dynamic Spatio-Temporal Graph Builder.
    
    Builds and maintains a dynamic graph that represents:
    - Robot position as the central node
    - Pedestrian positions as peripheral nodes
    - Distance-based edges between nodes
    - Temporal edges connecting history
    
    This graph is used as input to the ST-GCN for social-aware navigation.
    """
    
    # Node feature dimension (robot-centric features)
    ROBOT_FEATURE_DIM = 8  # [x, y, z, sin_heading, cos_heading, goal_x, goal_y, timestep]
    
    # Pedestrian feature dimension
    PEDESTRIAN_FEATURE_DIM = 14  # [rel_x, rel_y, rel_z, distance, direction, speed, 
                                  # vx, vy, depth, height, visible, confidence, 
                                  # time_since_seen, track_age]
    
    def __init__(
        self,
        max_pedestrians: int = 10,
        max_history_len: int = 8,
        distance_threshold: float = 5.0,
        edge_distance_threshold: float = 3.0,
        disappearance_threshold: int = 5,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize the dynamic spatio-temporal graph builder.
        
        Args:
            max_pedestrians: Maximum number of pedestrian nodes
            max_history_len: Maximum temporal history length
            distance_threshold: Distance for graph connectivity (meters)
            edge_distance_threshold: Distance for edge creation (meters)
            disappearance_threshold: Frames before removing disappeared nodes
            device: Computation device
        """
        self.max_pedestrians = max_pedestrians
        self.max_history_len = max_history_len
        self.distance_threshold = distance_threshold
        self.edge_distance_threshold = edge_distance_threshold
        self.disappearance_threshold = disappearance_threshold
        self.device = device
        
        # Node tracking
        self.next_track_id = 0
        self.track_to_node_idx: Dict[int, int] = {}  # track_id -> node_idx
        self.node_idx_to_track: Dict[int, int] = {}  # node_idx -> track_id
        
        # Pedestrian nodes
        self.pedestrian_nodes: Dict[int, PedestrianNode] = {}
        
        # Disappeared node tracking
        self.disappeared_frames: Dict[int, int] = {}
        
        # History for temporal modeling
        self.history_features: deque = deque(maxlen=max_history_len)
        self.current_timestep = 0
        
        # Camera intrinsics (can be set externally)
        self.camera_K = None
        
    def set_camera_intrinsics(
        self, 
        fx: float, 
        fy: float, 
        cx: float, 
        cy: float
    ):
        """
        Set camera intrinsic parameters for 3D projection.
        
        Args:
            fx, fy: Focal lengths
            cx, cy: Principal point
        """
        self.camera_K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float32)
    
    def update(
        self,
        robot_position: np.ndarray,
        robot_heading: float,
        yolo_detections: List[Dict],
        goal_position: Optional[np.ndarray] = None,
        depth_frame: Optional[np.ndarray] = None,
        rgb_shape: Optional[Tuple[int, int]] = None,
    ) -> SpatioTemporalGraphOutput:
        """
        Update the graph with new observations.
        
        Args:
            robot_position: Robot position in world coordinates [x, y, z]
            robot_heading: Robot heading angle (radians)
            yolo_detections: List of YOLO detections, each containing:
                - 'track_id': Unique tracking ID
                - 'bbox': [x1, y1, x2, y2] in pixel coordinates
                - 'confidence': Detection confidence
                - 'depth': (optional) Depth in meters
            goal_position: Goal position relative to robot [x, y]
            depth_frame: (optional) Depth image for 3D projection
            rgb_shape: (optional) RGB image shape (H, W)
            
        Returns:
            SpatioTemporalGraphOutput containing graph data
        """
        current_tracks = set()
        
        # Process YOLO detections
        for det in yolo_detections:
            track_id = det['track_id']
            current_tracks.add(track_id)
            
            if track_id in self.track_to_node_idx:
                # Update existing node
                self._update_pedestrian_node(
                    track_id, det, robot_position, robot_heading
                )
            else:
                # Create new node
                self._add_pedestrian_node(
                    track_id, det, robot_position, robot_heading
                )
        
        # Remove disappeared pedestrians
        self._remove_disappeared_pedestrians(current_tracks)
        
        # Build graph output
        output = self._build_graph_output(robot_position, robot_heading, goal_position)
        
        # Update history
        self._update_history(output.node_features)
        
        self.current_timestep += 1
        
        return output
    
    def _update_pedestrian_node(
        self,
        track_id: int,
        detection: Dict,
        robot_position: np.ndarray,
        robot_heading: float,
    ):
        """Update an existing pedestrian node."""
        node_idx = self.track_to_node_idx[track_id]
        node = self.pedestrian_nodes[node_idx]
        
        # Calculate new position
        bbox = detection['bbox']
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        
        # Get depth
        depth = detection.get('depth', node.depth)
        
        # Project to 3D
        pos_3d = self._project_to_3d(
            center_x, center_y, depth,
            robot_position, robot_heading
        )
        
        # Calculate velocity
        if node.position_3d is not None:
            velocity = pos_3d - node.position_3d
        else:
            velocity = np.zeros(3)
        
        # Update node
        node.position_3d = pos_3d
        node.velocity = velocity
        node.depth = depth
        node.bbox_2d = tuple(bbox)
        node.confidence = detection.get('confidence', 1.0)
        node.last_seen_frame = self.current_timestep
        node.visible = True
        
        # Reset disappearance counter
        if node_idx in self.disappeared_frames:
            del self.disappeared_frames[node_idx]
    
    def _add_pedestrian_node(
        self,
        track_id: int,
        detection: Dict,
        robot_position: np.ndarray,
        robot_heading: float,
    ):
        """Add a new pedestrian node."""
        if len(self.pedestrian_nodes) >= self.max_pedestrians:
            # Remove oldest disappeared node if exists
            self._remove_oldest_disappeared()
            
        if len(self.pedestrian_nodes) >= self.max_pedestrians:
            return  # Still full
        
        # Assign node index
        node_idx = len(self.pedestrian_nodes)
        
        # Calculate position
        bbox = detection['bbox']
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        depth = detection.get('depth', 2.0)  # Default 2m if not provided
        
        pos_3d = self._project_to_3d(
            center_x, center_y, depth,
            robot_position, robot_heading
        )
        
        # Create node
        node = PedestrianNode(
            track_id=track_id,
            position_3d=pos_3d,
            velocity=np.zeros(3),
            depth=depth,
            bbox_2d=tuple(bbox),
            visible=True,
            last_seen_frame=self.current_timestep,
            confidence=detection.get('confidence', 1.0)
        )
        
        self.pedestrian_nodes[node_idx] = node
        self.track_to_node_idx[track_id] = node_idx
        self.node_idx_to_track[node_idx] = track_id
    
    def _remove_disappeared_pedestrians(self, current_tracks: set):
        """Remove pedestrians that are no longer visible."""
        to_remove = []
        
        for node_idx, node in self.pedestrian_nodes.items():
            if node.track_id not in current_tracks:
                if node_idx not in self.disappeared_frames:
                    self.disappeared_frames[node_idx] = 1
                else:
                    self.disappeared_frames[node_idx] += 1
                
                if self.disappeared_frames[node_idx] >= self.disappearance_threshold:
                    to_remove.append(node_idx)
                    node.visible = False
        
        for node_idx in to_remove:
            self._remove_node(node_idx)
    
    def _remove_node(self, node_idx: int):
        """Remove a node from the graph."""
        if node_idx in self.pedestrian_nodes:
            track_id = self.node_idx_to_track[node_idx]
            
            del self.pedestrian_nodes[node_idx]
            del self.track_to_node_idx[track_id]
            del self.node_idx_to_track[node_idx]
            
            if node_idx in self.disappeared_frames:
                del self.disappeared_frames[node_idx]
    
    def _remove_oldest_disappeared(self):
        """Remove the oldest disappeared node to make room for new ones."""
        if not self.disappeared_frames:
            return
        
        oldest_idx = max(self.disappeared_frames, key=self.disappeared_frames.get)
        self._remove_node(oldest_idx)
    
    def _project_to_3d(
        self,
        u: float,
        v: float,
        depth: float,
        robot_position: np.ndarray,
        robot_heading: float,
    ) -> np.ndarray:
        """
        Project 2D pixel coordinates to 3D world coordinates.
        
        Uses a simplified ground plane assumption.
        
        Args:
            u, v: Pixel coordinates
            depth: Depth in meters
            robot_position: Robot position [x, y, z]
            robot_heading: Robot heading (radians)
            
        Returns:
            3D position [x, y, z]
        """
        if self.camera_K is None:
            # Default camera intrinsics
            fx, fy = 525.0, 525.0
            cx, cy = 319.5, 239.5
        else:
            fx = self.camera_K[0, 0]
            fy = self.camera_K[1, 1]
            cx = self.camera_K[0, 2]
            cy = self.camera_K[1, 2]
        
        # Camera coordinates (assuming ground plane at z=0)
        x_cam = (u - cx) * depth / fx
        z_cam = depth
        
        # Transform to world coordinates
        cos_h = math.cos(robot_heading)
        sin_h = math.sin(robot_heading)
        
        # Camera offset from robot base (typically forward and up)
        cam_forward = 0.1  # meters forward
        cam_height = 0.5   # meters up
        
        # World position (on ground plane)
        x_world = robot_position[0] + cos_h * z_cam - sin_h * x_cam
        y_world = robot_position[1] + sin_h * z_cam + cos_h * x_cam
        z_world = 0.0  # Ground plane
        
        return np.array([x_world, y_world, z_world], dtype=np.float32)
    
    def _build_graph_output(
        self,
        robot_position: np.ndarray,
        robot_heading: float,
        goal_position: Optional[np.ndarray],
    ) -> SpatioTemporalGraphOutput:
        """Build the graph output tensor."""
        num_nodes = 1 + len(self.pedestrian_nodes)  # robot + pedestrians
        
        # Node features
        all_features = []
        all_node_types = []
        
        # Robot features
        robot_feat = self._build_robot_features(
            robot_position, robot_heading, goal_position
        )
        all_features.append(robot_feat)
        all_node_types.append(0)  # Robot type
        
        # Pedestrian features
        for node_idx in sorted(self.pedestrian_nodes.keys()):
            node = self.pedestrian_nodes[node_idx]
            ped_feat = self._build_pedestrian_features(node, robot_position)
            all_features.append(ped_feat)
            all_node_types.append(1)  # Pedestrian type
        
        # Pad to max size if needed
        while len(all_features) < self.max_pedestrians + 1:
            all_features.append(np.zeros(self.PEDESTRIAN_FEATURE_DIM + self.ROBOT_FEATURE_DIM - self.PEDESTRIAN_FEATURE_DIM))
            all_node_types.append(-1)  # Invalid type
        
        # Stack features
        max_feat_len = max(len(f) for f in all_features)
        features_array = np.zeros((len(all_features), max_feat_len), dtype=np.float32)
        for i, f in enumerate(all_features):
            features_array[i, :len(f)] = f
        
        # Build edges
        edge_index, edge_weights = self._build_edges(
            features_array, robot_position
        )
        
        # Build masks
        node_mask = torch.zeros(len(all_features), dtype=torch.bool)
        for i in range(min(len(all_features), num_nodes)):
            node_mask[i] = True
        
        # Convert to tensors
        node_features = torch.from_numpy(features_array).float().to(self.device)
        edge_index_t = torch.from_numpy(edge_index).long().to(self.device)
        edge_weights_t = torch.from_numpy(edge_weights).float().to(self.device)
        node_types = torch.tensor(all_node_types).long().to(self.device)
        robot_features = node_features[0:1]
        
        return SpatioTemporalGraphOutput(
            node_features=node_features,
            edge_index=edge_index_t,
            edge_weights=edge_weights_t,
            node_mask=node_mask,
            node_types=node_types,
            robot_features=robot_features,
            num_pedestrians=len(self.pedestrian_nodes),
            timestep=self.current_timestep,
        )
    
    def _build_robot_features(
        self,
        robot_position: np.ndarray,
        robot_heading: float,
        goal_position: Optional[np.ndarray],
    ) -> np.ndarray:
        """Build robot node features."""
        features = [
            robot_position[0],  # x
            robot_position[1],  # y
            robot_position[2],  # z
            math.sin(robot_heading),  # heading sin
            math.cos(robot_heading),  # heading cos
        ]
        
        if goal_position is not None:
            features.extend([goal_position[0], goal_position[1]])
        else:
            features.extend([0.0, 0.0])
        
        # Timestep (normalized)
        features.append(self.current_timestep / 100.0)
        
        # Pad to ROBOT_FEATURE_DIM
        while len(features) < self.ROBOT_FEATURE_DIM:
            features.append(0.0)
        
        return np.array(features[:self.ROBOT_FEATURE_DIM], dtype=np.float32)
    
    def _build_pedestrian_features(
        self,
        node: PedestrianNode,
        robot_position: np.ndarray,
    ) -> np.ndarray:
        """Build pedestrian node features."""
        # Relative position
        rel_pos = node.position_3d - robot_position
        
        # Distance and direction
        distance = np.linalg.norm(rel_pos)
        direction = math.atan2(rel_pos[1], rel_pos[0]) if distance > 0 else 0.0
        
        # Speed
        speed = np.linalg.norm(node.velocity)
        
        # Time since last seen
        time_since_seen = self.current_timestep - node.last_seen_frame
        
        # Track age
        track_age = self.current_timestep - node.last_seen_frame + 1
        
        features = [
            rel_pos[0],  # relative x
            rel_pos[1],  # relative y
            rel_pos[2],  # relative z
            distance,  # distance
            direction,  # direction angle
            speed,  # speed
            node.velocity[0],  # vx
            node.velocity[1],  # vy
            node.depth,  # depth
            node.position_3d[2] if len(node.position_3d) > 2 else 0.0,  # height
            1.0 if node.visible else 0.0,  # visible
            node.confidence,  # confidence
            time_since_seen / 10.0,  # normalized time since seen
            track_age / 100.0,  # normalized track age
        ]
        
        return np.array(features[:self.PEDESTRIAN_FEATURE_DIM], dtype=np.float32)
    
    def _build_edges(
        self,
        features: np.ndarray,
        robot_position: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build graph edges based on distance.
        
        Returns:
            edge_index: [2, num_edges]
            edge_weights: [num_edges]
        """
        num_nodes = features.shape[0]
        
        # Robot is always node 0
        robot_pos = robot_position[:2]  # Use x, y only
        
        edge_list = []
        weight_list = []
        
        # Connect robot to pedestrians
        for i in range(1, num_nodes):
            # Check if this is a valid pedestrian node
            if features[i, 0] == 0 and features[i, 1] == 0:  # Zero features = padding
                continue
                
            # Get pedestrian position (relative features are at indices 0, 1)
            ped_pos = features[i, :2] + robot_pos  # Convert back to world coords
            
            # Calculate distance
            distance = np.linalg.norm(ped_pos - robot_pos)
            
            if distance < self.edge_distance_threshold:
                # Robot -> Pedestrian edge
                edge_list.append([0, i])
                weight_list.append(self._distance_to_weight(distance))
                
                # Bidirectional
                edge_list.append([i, 0])
                weight_list.append(self._distance_to_weight(distance))
        
        # Connect pedestrians to each other (social edges)
        for i in range(1, num_nodes):
            for j in range(i + 1, num_nodes):
                if features[i, 0] == 0 and features[i, 1] == 0:
                    continue
                if features[j, 0] == 0 and features[j, 1] == 0:
                    continue
                
                # Convert relative to world coords
                pos_i = features[i, :2] + robot_pos
                pos_j = features[j, :2] + robot_pos
                
                distance = np.linalg.norm(pos_i - pos_j)
                
                if distance < self.edge_distance_threshold:
                    edge_list.append([i, j])
                    weight_list.append(self._distance_to_weight(distance))
                    
                    edge_list.append([j, i])
                    weight_list.append(self._distance_to_weight(distance))
        
        # Add self-loops for all valid nodes
        for i in range(num_nodes):
            if features[i, 0] != 0 or features[i, 1] != 0 or i == 0:
                edge_list.append([i, i])
                weight_list.append(1.0)
        
        if len(edge_list) == 0:
            # Return empty edges
            return np.zeros((2, 1), dtype=np.int64), np.ones(1, dtype=np.float32)
        
        return np.array(edge_list, dtype=np.int64).T, np.array(weight_list, dtype=np.float32)
    
    def _distance_to_weight(self, distance: float) -> float:
        """Convert distance to edge weight using Gaussian kernel."""
        sigma = self.edge_distance_threshold / 2
        return math.exp(-(distance ** 2) / (2 * sigma ** 2))
    
    def _update_history(self, current_features: torch.Tensor):
        """Update temporal history with current frame."""
        # Detach to avoid gradient tracking
        self.history_features.append(current_features.detach().cpu())
    
    def get_temporal_features(self) -> Optional[torch.Tensor]:
        """
        Get temporal history features for TCN processing.
        
        Returns:
            Temporal features [seq_len, num_nodes, feature_dim] or None if no history
        """
        if len(self.history_features) == 0:
            return None
        
        # Stack history
        history = torch.stack(list(self.history_features))
        
        # Transpose to [seq_len, num_nodes, feature_dim]
        return history.transpose(0, 1)
    
    def reset(self):
        """Reset the graph state."""
        self.pedestrian_nodes.clear()
        self.track_to_node_idx.clear()
        self.node_idx_to_track.clear()
        self.disappeared_frames.clear()
        self.history_features.clear()
        self.current_timestep = 0
        self.next_track_id = 0


class SimpleTracker:
    """
    Simple IoU-based multi-object tracker for pedestrian tracking.
    
    This tracker associates YOLO detections across frames using
    Intersection over Union (IoU) matching.
    """
    
    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_age: int = 30,
    ):
        """
        Initialize the tracker.
        
        Args:
            iou_threshold: Minimum IoU for matching
            max_age: Maximum frames to keep lost tracks
        """
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        
        self.tracks: Dict[int, Dict] = {}  # track_id -> track info
        self.next_id = 0
        self.frame_count = 0
    
    def update(self, detections: List[Dict]) -> List[Dict]:
        """
        Update tracks with new detections.
        
        Args:
            detections: List of detections, each containing:
                - 'bbox': [x1, y1, x2, y2]
                - 'confidence': Detection confidence
                
        Returns:
            List of tracked detections with track_ids
        """
        self.frame_count += 1
        
        # If no existing tracks, create new ones
        if not self.tracks:
            for det in detections:
                track_id = self.next_id
                self.next_id += 1
                
                det['track_id'] = track_id
                self.tracks[track_id] = {
                    'bbox': det['bbox'],
                    'confidence': det['confidence'],
                    'age': 0,
                    'hits': 1,
                }
            return detections
        
        # Compute IoU matrix
        track_ids = list(self.tracks.keys())
        track_bboxes = [self.tracks[tid]['bbox'] for tid in track_ids]
        
        iou_matrix = self._compute_iou_matrix(track_bboxes, [d['bbox'] for d in detections])
        
        # Match using greedy algorithm
        matched_tracks = set()
        matched_detections = set()
        matches = []
        
        for _ in range(min(len(track_ids), len(detections))):
            best_iou = self.iou_threshold
            best_match = None
            
            for i, tid in enumerate(track_ids):
                if i in matched_tracks:
                    continue
                    
                for j, det in enumerate(detections):
                    if j in matched_detections:
                        continue
                    
                    if iou_matrix[i, j] > best_iou:
                        best_iou = iou_matrix[i, j]
                        best_match = (i, j, tid)
            
            if best_match is None:
                break
                
            i, j, tid = best_match
            matched_tracks.add(i)
            matched_detections.add(j)
            matches.append((tid, detections[j]))
        
        # Update matched tracks
        for tid, det in matches:
            self.tracks[tid]['bbox'] = det['bbox']
            self.tracks[tid]['confidence'] = det.get('confidence', 1.0)
            self.tracks[tid]['age'] = 0
            self.tracks[tid]['hits'] += 1
            det['track_id'] = tid
        
        # Age unmatched tracks
        to_remove = []
        for tid, track in self.tracks.items():
            if tid not in [m[0] for m in matches]:
                track['age'] += 1
                if track['age'] > self.max_age:
                    to_remove.append(tid)
        
        for tid in to_remove:
            del self.tracks[tid]
        
        # Create new tracks for unmatched detections
        for j, det in enumerate(detections):
            if j not in matched_detections:
                track_id = self.next_id
                self.next_id += 1
                
                det['track_id'] = track_id
                self.tracks[track_id] = {
                    'bbox': det['bbox'],
                    'confidence': det.get('confidence', 1.0),
                    'age': 0,
                    'hits': 1,
                }
        
        return detections
    
    def _compute_iou_matrix(
        self,
        boxes1: List[List[float]],
        boxes2: List[List[float]],
    ) -> np.ndarray:
        """Compute IoU matrix between two sets of boxes."""
        n = len(boxes1)
        m = len(boxes2)
        
        if n == 0 or m == 0:
            return np.zeros((n, m))
        
        iou_matrix = np.zeros((n, m))
        
        for i, box1 in enumerate(boxes1):
            for j, box2 in enumerate(boxes2):
                iou_matrix[i, j] = self._compute_iou(box1, box2)
        
        return iou_matrix
    
    @staticmethod
    def _compute_iou(box1: List[float], box2: List[float]) -> float:
        """Compute IoU between two boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        union = area1 + area2 - intersection
        
        return intersection / (union + 1e-6)
    
    def reset(self):
        """Reset the tracker."""
        self.tracks.clear()
        self.next_id = 0
        self.frame_count = 0
