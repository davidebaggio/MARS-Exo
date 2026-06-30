import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import Tuple


def estimate_gravity(accel_samples: np.ndarray) -> np.ndarray:
    if len(accel_samples) == 0:
        return np.array([0.0, 0.0, 1.0])
    gravity = np.mean(accel_samples, axis=0)
    norm = np.linalg.norm(gravity)
    if norm < 1e-6:
        return np.array([0.0, 0.0, 1.0])
    return gravity / norm


def detect_motion(
    gyro_samples: np.ndarray,
    accel_samples: np.ndarray,
    gyro_threshold: float = 0.3,
    accel_var_threshold: float = 0.5,
) -> bool:
    if len(gyro_samples) == 0 and len(accel_samples) == 0:
        return True
    if len(gyro_samples) > 0:
        gyro_mags = np.linalg.norm(gyro_samples, axis=1)
        if np.mean(gyro_mags) > gyro_threshold:
            return True
    if len(accel_samples) > 0:
        if np.var(accel_samples, axis=0).sum() > accel_var_threshold:
            return True
    return False


def estimate_gyro_bias(gyro_samples_at_rest: np.ndarray) -> np.ndarray:
    if len(gyro_samples_at_rest) == 0:
        return np.zeros(3)
    return np.mean(gyro_samples_at_rest, axis=0)


def gyro_integrate_rotvec(gyro_ang_vel: np.ndarray, dt: float) -> np.ndarray:
    return gyro_ang_vel * dt


def check_gravity_alignment(
    R_est: np.ndarray,
    g_head: np.ndarray,
    g_exo: np.ndarray,
    threshold_deg: float = 15.0,
) -> Tuple[bool, float]:
    g_head_in_exo = R_est @ g_head
    g_head_norm = g_head_in_exo / (np.linalg.norm(g_head_in_exo) + 1e-10)
    g_exo_norm = g_exo / (np.linalg.norm(g_exo) + 1e-10)
    cos_angle = np.clip(np.dot(g_head_norm, g_exo_norm), -1.0, 1.0)
    angle_deg = float(np.degrees(np.arccos(cos_angle)))
    return angle_deg < threshold_deg, angle_deg
