#!/usr/bin/env python3
"""
YOLO Pedestrian Detector Module.

This module provides a wrapper for YOLO-based pedestrian detection
that integrates seamlessly with the spatio-temporal graph builder.

Features:
- YOLOv8 integration for real-time pedestrian detection
- Simple IoU-based tracking for multi-frame association
- Depth fusion for 3D position estimation
- Compatible with existing DPED-PRO observation pipeline

Author: DPED-PRO
Date: 2024
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    """
    Detection result from YOLO pedestrian detector.
    
    Attributes:
        bbox: Bounding box [x1, y1, x2, y2] in pixel coordinates
        confidence: Detection confidence score
        class_id: Class ID (0 for person)
        class_name: Class name
        track_id: Tracking ID (assigned by tracker)
        depth: Estimated depth in meters
    """
    bbox: Tuple[float, float, float, float]
    confidence: float
    class_id: int
    class_name: str
    track_id: int
    depth: float = 0.0


class YOLOPedestrianDetector:
    """
    YOLO-based Pedestrian Detector with tracking.
    
    This class wraps YOLO detection and provides:
    - Real-time pedestrian detection
    - Multi-object tracking across frames
    - Depth estimation from depth camera
    - Easy integration with spatio-temporal graph builder
    """
    
    # Default camera intrinsics (can be overridden)
    DEFAULT_K = np.array([
        [525.0, 0.0, 319.5],
        [0.0, 525.0, 239.5],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)
    
    # Person class ID in COCO dataset
    PERSON_CLASS_ID = 0
    
    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence_threshold: float = 0.5,
        nms_threshold: float = 0.4,
        max_detections: int = 20,
        use_tracking: bool = True,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize YOLO pedestrian detector.
        
        Args:
            model_path: Path to YOLO model weights
            confidence_threshold: Minimum confidence for detections
            nms_threshold: NMS IoU threshold
            max_detections: Maximum number of detections
            use_tracking: Whether to use multi-object tracking
            device: Device to run inference on
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.max_detections = max_detections
        self.use_tracking = use_tracking
        self.device = device
        
        # YOLO model (lazy loaded)
        self._yolo_model = None
        
        # Tracker
        self._tracker = SimpleIoUTracker(
            iou_threshold=0.3,
            max_age=30,
        )
        
        # Camera intrinsics
        self.K = self.DEFAULT_K.copy()
        
        # Track history
        self._detection_history: List[List[DetectionResult]] = []
    
    @property
    def yolo_model(self):
        """Lazy load YOLO model."""
        if self._yolo_model is None:
            self._load_yolo_model()
        return self._yolo_model
    
    def _load_yolo_model(self):
        """Load YOLO model."""
        try:
            from ultralytics import YOLO
            logger.info(f"Loading YOLO model from {self.model_path}")
            self._yolo_model = YOLO(self.model_path)
            self._yolo_model.to(self.device)
            logger.info("YOLO model loaded successfully")
        except ImportError:
            logger.warning("Ultralytics YOLO not installed. Using mock detector.")
            self._yolo_model = None
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            self._yolo_model = None
    
    def set_camera_intrinsics(
        self,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
    ):
        """
        Set camera intrinsic parameters.
        
        Args:
            fx, fy: Focal lengths
            cx, cy: Principal point
        """
        self.K = np.array([
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)
    
    def detect(
        self,
        rgb_frame: np.ndarray,
        depth_frame: Optional[np.ndarray] = None,
        return_dict: bool = True,
    ) -> List[Dict]:
        """
        Detect pedestrians in RGB frame.
        
        Args:
            rgb_frame: RGB image [H, W, 3] in uint8 format
            depth_frame: Depth image [H, W] in meters (optional)
            return_dict: Whether to return dict format
            
        Returns:
            List of detections, each containing:
                - 'bbox': [x1, y1, x2, y2]
                - 'confidence': float
                - 'class_id': int (0 for person)
                - 'track_id': int
                - 'depth': float (if depth provided)
        """
        # Run YOLO detection
        if self.yolo_model is not None:
            detections = self._detect_with_yolo(rgb_frame)
        else:
            detections = self._mock_detect(rgb_frame)
        
        # Get depths for detections
        if depth_frame is not None:
            detections = self._add_depths(detections, depth_frame)
        
        # Update tracker
        if self.use_tracking:
            detection_dicts = [
                {
                    'bbox': d.bbox,
                    'confidence': d.confidence,
                }
                for d in detections
            ]
            tracked = self._tracker.update(detection_dicts)
            
            # Update track_ids
            for det, track in zip(detections, tracked):
                det.track_id = track['track_id']
        
        # Store history
        self._detection_history.append(detections)
        if len(self._detection_history) > 10:
            self._detection_history.pop(0)
        
        # Return in dict format for graph builder
        if return_dict:
            return [
                {
                    'bbox': d.bbox,
                    'confidence': d.confidence,
                    'class_id': d.class_id,
                    'track_id': d.track_id,
                    'depth': d.depth,
                }
                for d in detections
            ]
        else:
            return detections
    
    def _detect_with_yolo(self, rgb_frame: np.ndarray) -> List[DetectionResult]:
        """Run YOLO detection."""
        results = self.yolo_model(
            rgb_frame,
            conf=self.confidence_threshold,
            iou=self.nms_threshold,
            max_det=self.max_detections,
            classes=[self.PERSON_CLASS_ID],  # Only detect persons
            verbose=False,
        )
        
        detections = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy()
            
            for bbox, conf, cls in zip(boxes, confs, classes):
                detections.append(DetectionResult(
                    bbox=tuple(float(x) for x in bbox),
                    confidence=float(conf),
                    class_id=int(cls),
                    class_name='person',
                    track_id=-1,  # Will be assigned by tracker
                    depth=0.0,  # Will be estimated from depth
                ))
        
        return detections
    
    def _mock_detect(self, rgb_frame: np.ndarray) -> List[DetectionResult]:
        """
        Mock detection for testing without YOLO.
        
        Returns random detections for testing.
        """
        h, w = rgb_frame.shape[:2]
        
        # Generate 0-3 random detections
        num_det = np.random.randint(0, 3)
        
        detections = []
        for i in range(num_det):
            # Random bbox
            x1 = np.random.uniform(0, w * 0.5)
            y1 = np.random.uniform(0, h * 0.5)
            x2 = x1 + np.random.uniform(w * 0.1, w * 0.3)
            y2 = y1 + np.random.uniform(h * 0.2, h * 0.4)
            x2 = min(x2, w)
            y2 = min(y2, h)
            
            detections.append(DetectionResult(
                bbox=(x1, y1, x2, y2),
                confidence=np.random.uniform(0.5, 0.95),
                class_id=0,
                class_name='person',
                track_id=-1,
                depth=np.random.uniform(1.0, 4.0),
            ))
        
        return detections
    
    def _add_depths(
        self,
        detections: List[DetectionResult],
        depth_frame: np.ndarray,
    ) -> List[DetectionResult]:
        """
        Add depth estimates to detections.
        
        Uses center point of bbox to estimate depth.
        """
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            
            # Clamp to frame bounds
            center_x = max(0, min(center_x, depth_frame.shape[1] - 1))
            center_y = max(0, min(center_y, depth_frame.shape[0] - 1))
            
            # Get depth
            depth = depth_frame[center_y, center_x]
            
            # Validate depth
            if np.isnan(depth) or np.isinf(depth) or depth <= 0:
                depth = 2.0  # Default depth
            
            det.depth = float(depth)
        
        return detections
    
    def get_3d_position(
        self,
        bbox: Tuple[float, float, float, float],
        depth: float,
        robot_position: Optional[np.ndarray] = None,
        robot_heading: float = 0.0,
    ) -> np.ndarray:
        """
        Project 2D detection to 3D world coordinates.
        
        Args:
            bbox: Bounding box [x1, y1, x2, y2]
            depth: Depth in meters
            robot_position: Robot position [x, y, z] (optional)
            robot_heading: Robot heading in radians
            
        Returns:
            3D position [x, y, z]
        """
        # Use center of bbox
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        
        # Project to camera coordinates
        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]
        
        # Camera coordinates (assuming ground plane)
        x_cam = (center_x - cx) * depth / fx
        z_cam = depth
        
        # Transform to world coordinates
        if robot_position is not None:
            import math
            cos_h = math.cos(robot_heading)
            sin_h = math.sin(robot_heading)
            
            x_world = robot_position[0] + cos_h * z_cam - sin_h * x_cam
            y_world = robot_position[1] + sin_h * z_cam + cos_h * x_cam
            z_world = 0.0  # Ground plane
        else:
            x_world = x_cam
            y_world = z_cam
            z_world = 0.0
        
        return np.array([x_world, y_world, z_world], dtype=np.float32)
    
    def reset(self):
        """Reset detector state."""
        self._tracker.reset()
        self._detection_history.clear()


class SimpleIoUTracker:
    """
    Simple IoU-based multi-object tracker.
    
    Tracks objects across frames using Intersection over Union matching.
    """
    
    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_age: int = 30,
    ):
        """
        Initialize tracker.
        
        Args:
            iou_threshold: Minimum IoU for matching
            max_age: Maximum frames to keep lost tracks
        """
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        
        self.tracks: Dict[int, Dict] = {}
        self.next_id = 0
        self.frame_count = 0
    
    def update(self, detections: List[Dict]) -> List[Dict]:
        """
        Update tracks with new detections.
        
        Args:
            detections: List of detections with 'bbox' and 'confidence'
            
        Returns:
            Detections with assigned 'track_id'
        """
        self.frame_count += 1
        
        if not self.tracks:
            # Create new tracks
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
        det_bboxes = [d['bbox'] for d in detections]
        
        iou_matrix = self._compute_iou_matrix(track_bboxes, det_bboxes)
        
        # Greedy matching
        matched_tracks = set()
        matched_detections = set()
        
        for _ in range(min(len(track_ids), len(detections))):
            best_iou = self.iou_threshold
            best_match = None
            
            for i, tid in enumerate(track_ids):
                if i in matched_tracks:
                    continue
                for j in range(len(detections)):
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
            detections[j]['track_id'] = tid
            
            # Update track
            self.tracks[tid]['bbox'] = detections[j]['bbox']
            self.tracks[tid]['confidence'] = detections[j].get('confidence', 1.0)
            self.tracks[tid]['age'] = 0
            self.tracks[tid]['hits'] += 1
        
        # Age unmatched tracks
        to_remove = []
        for tid, track in self.tracks.items():
            if tid not in [self.tracks.keys()[i] for i in matched_tracks]:
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
        """Compute IoU matrix."""
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
        """Reset tracker."""
        self.tracks.clear()
        self.next_id = 0
        self.frame_count = 0
