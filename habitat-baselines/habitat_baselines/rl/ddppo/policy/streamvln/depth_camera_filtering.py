"""
Depth camera filtering utilities for StreamVLN.
This module provides depth image preprocessing functions.
"""

import numpy as np
import cv2
from typing import Optional


def filter_depth(depth: np.ndarray, blur_type: Optional[str] = None) -> np.ndarray:
    """
    Filter and preprocess depth images.
    
    Args:
        depth: Input depth image as numpy array (H, W)
        blur_type: Type of blur to apply. Options: None, 'gaussian', 'median', 'bilateral'
        
    Returns:
        Filtered depth image as numpy array (H, W)
    """
    # Make a copy to avoid modifying the original
    filtered_depth = depth.copy()
    
    # Handle NaN and inf values
    filtered_depth = np.nan_to_num(filtered_depth, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Apply blur if specified
    if blur_type == 'gaussian':
        filtered_depth = cv2.GaussianBlur(filtered_depth, (5, 5), 0)
    elif blur_type == 'median':
        filtered_depth = cv2.medianBlur(filtered_depth.astype(np.float32), 5)
    elif blur_type == 'bilateral':
        # Bilateral filter preserves edges while smoothing
        filtered_depth = cv2.bilateralFilter(
            filtered_depth.astype(np.float32), 
            d=5, 
            sigmaColor=0.1, 
            sigmaSpace=5
        )
    
    return filtered_depth
