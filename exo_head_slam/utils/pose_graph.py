from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R


@dataclass
class PoseGraphKeyframe:
    id: int
    pose: np.ndarray  # 4x4 T_map_kf
    head_kp: Optional[np.ndarray] = None  # (N, 2)
    head_3d: Optional[np.ndarray] = None  # (N, 3)
    exo_kp: Optional[np.ndarray] = None
    exo_3d: Optional[np.ndarray] = None
    timestamp: float = 0.0


@dataclass
class PoseGraphEdge:
    src: int
    dst: int
    T_rel: np.ndarray  # 4x4 T_src_dst
    info: np.ndarray   # 6x6 information matrix
    etype: str = 'temporal'  # 'temporal' | 'cross_camera'


def se3_log(T: np.ndarray) -> np.ndarray:
    r = R.from_matrix(T[:3, :3])
    rotvec = r.as_rotvec()
    t = T[:3, 3]
    return np.concatenate([t, rotvec])


def se3_exp(v: np.ndarray) -> np.ndarray:
    t = v[:3]
    r = R.from_rotvec(v[3:6])
    T = np.eye(4)
    T[:3, :3] = r.as_matrix()
    T[:3, 3] = t
    return T


def compute_edge_residual(T_w_src: np.ndarray, T_w_dst: np.ndarray, T_rel: np.ndarray) -> np.ndarray:
    T_pred = np.linalg.inv(T_w_dst) @ T_w_src
    error_T = T_rel @ np.linalg.inv(T_pred)
    return se3_log(error_T)


class SlidingWindowPoseGraph:
    def __init__(self, window_size: int = 20):
        self.window_size = window_size
        self.keyframes: List[PoseGraphKeyframe] = []
        self.edges: List[PoseGraphEdge] = []
        self._next_id = 0

    def add_keyframe(self, kf: PoseGraphKeyframe) -> int:
        kf.id = self._next_id
        self._next_id += 1
        self.keyframes.append(kf)
        if len(self.keyframes) > self.window_size:
            removed = self.keyframes.pop(0)
            self.edges = [e for e in self.edges
                          if e.src != removed.id and e.dst != removed.id]
        return kf.id

    def add_edge(self, edge: PoseGraphEdge):
        self.edges.append(edge)

    def get_pose(self, kf_id: int) -> Optional[np.ndarray]:
        for kf in self.keyframes:
            if kf.id == kf_id:
                return kf.pose.copy()
        return None

    def latest_pose(self) -> Optional[np.ndarray]:
        if not self.keyframes:
            return None
        return self.keyframes[-1].pose.copy()

    def latest_kf_id(self) -> Optional[int]:
        if not self.keyframes:
            return None
        return self.keyframes[-1].id

    def _pack_params(self) -> np.ndarray:
        params = []
        for kf in self.keyframes:
            params.append(se3_log(kf.pose))
        return np.concatenate(params)

    def _unpack_params(self, x: np.ndarray):
        n = len(self.keyframes)
        for i in range(n):
            self.keyframes[i].pose = se3_exp(x[i*6:(i+1)*6])

    def _residual_func(self, x: np.ndarray) -> np.ndarray:
        n = len(self.keyframes)
        poses = [se3_exp(x[i*6:(i+1)*6]) for i in range(n)]
        residuals = []
        for edge in self.edges:
            src_idx = None
            dst_idx = None
            for i, kf in enumerate(self.keyframes):
                if kf.id == edge.src:
                    src_idx = i
                if kf.id == edge.dst:
                    dst_idx = i
            if src_idx is None or dst_idx is None:
                continue
            err = compute_edge_residual(poses[src_idx], poses[dst_idx], edge.T_rel)
            L = np.linalg.cholesky(edge.info)
            weighted = L @ err
            residuals.append(weighted)
        if not residuals:
            return np.array([])
        return np.concatenate(residuals)

    def optimize(self) -> bool:
        if len(self.keyframes) < 2 or len(self.edges) < 1:
            return False
        x0 = self._pack_params()
        n_fixed = 1
        fixed_mask = [True] * (n_fixed * 6) + [False] * (len(x0) - n_fixed * 6)

        def wrapped_res(x):
            x_full = x0.copy()
            x_full[~np.array(fixed_mask)] = x
            return self._residual_func(x_full)

        x_free = x0[~np.array(fixed_mask)]
        result = least_squares(wrapped_res, x_free, method='lm', max_nfev=200, ftol=1e-6, xtol=1e-6)
        if result.success:
            x_full = x0.copy()
            x_full[~np.array(fixed_mask)] = result.x
            self._unpack_params(x_full)
            return True
        return False
