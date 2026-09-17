import numpy as np
import cv2
from typing import Optional, Tuple


def sanitize_depth(
    depth: np.ndarray,
    encoding: str,
    unit_scale: float = 0.001,
    min_depth: float = 0.1,
    max_depth: float = 6.0,
) -> np.ndarray:
    """Convert a ROS depth image to finite 32-bit metres."""
    result = depth.astype(np.float32, copy=True)
    if encoding.lower() == '16uc1':
        result *= unit_scale
    invalid = ~np.isfinite(result) | (result < min_depth) | (result > max_depth)
    result[invalid] = 0.0
    return result


def dilate_mask(mask: np.ndarray, pixels: int) -> np.ndarray:
    if pixels <= 0 or not np.any(mask):
        return mask
    size = 2 * pixels + 1
    return cv2.dilate(mask.astype(np.uint8), np.ones((size, size), np.uint8))

def get_3d_point(u: int, v: int, depth_map: np.ndarray, intrinsics: np.ndarray, depth_scale: float = 0.001) -> Optional[np.ndarray]:
    """
    Converts 2D pixel (u, v) and depth value to 3D point (X, Y, Z).
    
    Args:
        u: Horizontal pixel coordinate.
        v: Vertical pixel coordinate.
        depth_map: Depth image (in meters or scaled).
        intrinsics: (3, 3) camera intrinsic matrix [[fx, 0, cx], [0, fy, cy], [0, 0, 1]].
        
    Returns:
        3D point as (3,) numpy array or None if depth is invalid.
    """
    depth = depth_map[v, u]
    if not np.isfinite(depth) or depth <= 0:
        return None

    if np.issubdtype(depth_map.dtype, np.integer):
        depth = float(depth) * depth_scale
    else:
        depth = float(depth)
    
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]
    
    X = (u - cx) * depth / fx
    Y = (v - cy) * depth / fy
    Z = depth
    
    return np.array([X, Y, Z])

def apply_semantic_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Blacks out dynamic objects in the image based on the mask.
    
    Args:
        image: Original RGB image.
        mask: Binary mask (1 for dynamic objects, 0 for static).
        
    Returns:
        Masked image.
    """
    masked_image = image.copy()
    masked_image[mask > 0] = 0
    return masked_image
