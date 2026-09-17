import numpy as np
from typing import Tuple, Optional
import rclpy

logger = rclpy.logging.get_logger('math_utils')

def compute_transform_svd(points_A: np.ndarray, points_B: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes optimal SE(3) transform (Kabsch algorithm) from A to B.
    
    Args:
        points_A: (N, 3) matrix of points in source frame.
        points_B: (N, 3) matrix of points in target frame.
        
    Returns:
        R: (3, 3) rotation matrix.
        t: (3, 1) translation vector.
    """
    if points_A.shape != points_B.shape or points_A.shape[1] != 3:
        raise ValueError(f"Shape mismatch: {points_A.shape} vs {points_B.shape}")
    
    if points_A.shape[0] < 3:
        raise ValueError("At least 3 points required.")

    # 1. Centroids
    centroid_A = np.mean(points_A, axis=0)
    centroid_B = np.mean(points_B, axis=0)

    # 2. Centering
    AA = points_A - centroid_A
    BB = points_B - centroid_B

    # 3. Covariance
    H = AA.T @ BB

    # 4. SVD
    U, S, Vt = np.linalg.svd(H)

    # 5. Rotation R
    R = Vt.T @ U.T

    # Special case for reflection
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    # 6. Translation t
    t = centroid_B.T - R @ centroid_A.T

    return R, t.reshape(3, 1)

def compute_transform_ransac(
    points_A: np.ndarray,
    points_B: np.ndarray,
    threshold: float = 0.05,
    iterations: int = 100,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[Optional[np.ndarray], int, float]:
    """
    Finds best SE(3) transform using RANSAC.
    """
    if points_A.shape[0] < 3:
        return None, 0, 0.0

    best_inlier_count = 0
    best_transform = None
    best_rmse = float('inf')

    rng = rng or np.random.default_rng()
    for _ in range(iterations):
        # Sample 3 random points
        idx = rng.choice(len(points_A), 3, replace=False)
        try:
            R, t = compute_transform_svd(points_A[idx], points_B[idx])
            
            # Project all points
            B_pred = (R @ points_A.T).T + t.T
            dist = np.linalg.norm(points_B - B_pred, axis=1)
            inliers = np.where(dist < threshold)[0]
            
            if len(inliers) > best_inlier_count:
                best_inlier_count = len(inliers)
                if best_inlier_count >= 3:
                    R_best, t_best = compute_transform_svd(points_A[inliers], points_B[inliers])
                    T = np.eye(4)
                    T[:3, :3] = R_best
                    T[:3, 3] = t_best.flatten()
                    
                    # Compute RMSE for inliers
                    B_pred_best = (R_best @ points_A[inliers].T).T + t_best.T
                    inlier_dist = np.linalg.norm(points_B[inliers] - B_pred_best, axis=1)
                    rmse = np.sqrt(np.mean(inlier_dist**2))
                    
                    best_transform = T
                    best_rmse = rmse
                    
        except (ValueError, np.linalg.LinAlgError):
            continue
            
    return best_transform, best_inlier_count, best_rmse
