#!/usr/bin/env python3
"""
Script to generate videos from saved trajectory data.
Usage:
    python generate_trajectory_video.py --trajectory_dir /path/to/trajectory --output_dir /path/to/output
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image

# Import generate_video from habitat_baselines
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'habitat-baselines'))
from habitat_baselines.utils.common import generate_video
from habitat_baselines.common.tensorboard_utils import TensorboardWriter


def load_images_from_dir(rgb_dir: str) -> List[np.ndarray]:
    """Load RGB images from directory, sorted by step number.
    
    Args:
        rgb_dir: Path to directory containing RGB images (e.g., "0_0.jpg", "1_0.jpg", ...)
    
    Returns:
        List of numpy arrays representing images in order
    """
    images = []
    rgb_path = Path(rgb_dir)
    
    if not rgb_path.exists():
        raise ValueError(f"RGB directory does not exist: {rgb_dir}")
    
    # Get all image files and sort by step number
    image_files = sorted(
        rgb_path.glob("*.jpg"),
        key=lambda x: int(re.match(r'(\d+)_\d+\.jpg', x.name).group(1))
    )
    
    for img_file in image_files:
        img = Image.open(img_file)
        # Convert PIL image to numpy array (RGB format)
        img_array = np.array(img)
        images.append(img_array)
    
    return images


def load_metrics_from_dir(trajectory_dir: str) -> Dict[str, float]:
    """Load metrics from trajectory directory.
    
    Args:
        trajectory_dir: Path to trajectory directory
    
    Returns:
        Dictionary of metric names and values
    """
    metrics = {}
    traj_path = Path(trajectory_dir)
    
    # Try to load pedestrian_in_view data
    pedestrian_file = traj_path / "pedestrian_in_view" / "0.json"
    if pedestrian_file.exists():
        with open(pedestrian_file, 'r') as f:
            pedestrian_data = json.load(f)
            # Calculate average pedestrians in view
            if isinstance(pedestrian_data, list):
                metrics['avg_pedestrians_in_view'] = np.mean(pedestrian_data)
                metrics['max_pedestrians_in_view'] = np.max(pedestrian_data)
    
    # Try to load human_num data
    human_num_file = traj_path / "human_num" / "0.json"
    if human_num_file.exists():
        with open(human_num_file, 'r') as f:
            human_num_data = json.load(f)
            if isinstance(human_num_data, list) and len(human_num_data) > 0:
                # human_num is a list of lists, take the first element of each
                if isinstance(human_num_data[0], list):
                    human_nums = [item[0] if len(item) > 0 else 0 for item in human_num_data]
                    metrics['avg_human_num'] = np.mean(human_nums)
                    metrics['max_human_num'] = np.max(human_nums)
    
    # Try to load distance_to_goal data
    distance_file = traj_path / "distance_to_goal" / "0.json"
    if distance_file.exists():
        with open(distance_file, 'r') as f:
            distance_data = json.load(f)
            if isinstance(distance_data, list):
                metrics['initial_distance'] = distance_data[0] if len(distance_data) > 0 else 0.0
                metrics['final_distance'] = distance_data[-1] if len(distance_data) > 0 else 0.0
                metrics['min_distance'] = np.min(distance_data)
    
    return metrics


def generate_video_from_trajectory(
    trajectory_dir: str,
    output_dir: str,
    scene_id: Optional[str] = None,
    episode_id: Optional[str] = None,
    checkpoint_idx: int = 0,
    fps: int = 10,
    video_option: List[str] = None,
) -> str:
    """Generate video from saved trajectory data.
    
    Args:
        trajectory_dir: Path to trajectory directory (e.g., /path/to/aYhkzj2fEhP.basis/5045)
        output_dir: Directory to save the video
        scene_id: Scene ID (if None, extracted from trajectory_dir)
        episode_id: Episode ID (if None, extracted from trajectory_dir)
        checkpoint_idx: Checkpoint index for video naming
        fps: Frames per second for the video
        video_option: List of video options (default: ["disk"])
    
    Returns:
        Path to generated video file
    """
    if video_option is None:
        video_option = ["disk"]
    
    trajectory_path = Path(trajectory_dir)
    
    # Extract scene_id and episode_id from path if not provided
    if scene_id is None:
        # Extract from path like: .../aYhkzj2fEhP.basis/5045
        parts = trajectory_path.parts
        for i, part in enumerate(parts):
            if part.endswith('.basis'):
                scene_id = part.replace('.basis', '')
                break
        if scene_id is None:
            scene_id = trajectory_path.parent.name
    
    if episode_id is None:
        episode_id = trajectory_path.name
    
    # Load RGB images
    rgb_dir = trajectory_path / "rgb"
    if not rgb_dir.exists():
        raise ValueError(f"RGB directory does not exist: {rgb_dir}")
    
    print(f"Loading images from {rgb_dir}...")
    images = load_images_from_dir(str(rgb_dir))
    print(f"Loaded {len(images)} images")
    
    if len(images) == 0:
        raise ValueError(f"No images found in {rgb_dir}")
    
    # Load metrics
    metrics = load_metrics_from_dir(str(trajectory_path))
    print(f"Loaded metrics: {metrics}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a dummy TensorboardWriter (won't be used if video_option doesn't include "tensorboard")
    class DummyTensorboardWriter:
        def add_video_from_np_images(self, *args, **kwargs):
            pass
    
    tb_writer = DummyTensorboardWriter()
    
    # Generate video
    print(f"Generating video with {len(images)} frames at {fps} fps...")
    video_name = generate_video(
        video_option=video_option,
        video_dir=output_dir,
        images=images,
        scene_id=scene_id,
        episode_id=episode_id,
        checkpoint_idx=checkpoint_idx,
        metrics=metrics,
        tb_writer=tb_writer,
        fps=fps,
        verbose=True,
        keys_to_include_in_name=None,  # Include all metrics in name
    )
    
    print(f"Video saved: {os.path.join(output_dir, video_name)}")
    return os.path.join(output_dir, video_name)


def main():
    parser = argparse.ArgumentParser(
        description="Generate video from saved trajectory data"
    )
    parser.add_argument(
        "--trajectory_dir",
        type=str,
        required=True,
        help="Path to trajectory directory (e.g., /path/to/aYhkzj2fEhP.basis/5045)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the generated video",
    )
    parser.add_argument(
        "--scene_id",
        type=str,
        default=None,
        help="Scene ID (if not provided, extracted from trajectory_dir)",
    )
    parser.add_argument(
        "--episode_id",
        type=str,
        default=None,
        help="Episode ID (if not provided, extracted from trajectory_dir)",
    )
    parser.add_argument(
        "--checkpoint_idx",
        type=int,
        default=0,
        help="Checkpoint index for video naming",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="Frames per second for the video",
    )
    
    args = parser.parse_args()
    
    try:
        video_path = generate_video_from_trajectory(
            trajectory_dir=args.trajectory_dir,
            output_dir=args.output_dir,
            scene_id=args.scene_id,
            episode_id=args.episode_id,
            checkpoint_idx=args.checkpoint_idx,
            fps=args.fps,
        )
        print(f"\n✓ Successfully generated video: {video_path}")
    except Exception as e:
        print(f"\n✗ Error generating video: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

