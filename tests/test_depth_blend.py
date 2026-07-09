import numpy as np
import torch

from exo_head_slam.extrinsic_solver_node import ExtrinsicSolverNode


def test_blend_depth_only_fills_camera_holes():
    node = object.__new__(ExtrinsicSolverNode)
    node.conf_gate_combine = True

    raw = np.array([1.0, 0.0, 1.0], dtype=np.float32)
    pred = np.array([2.0, 2.0, 1.1], dtype=np.float32)
    raw_valid = np.array([True, False, True])
    conf = np.array([True, True, True])

    out = node.blend_depth(raw, pred, raw_valid, conf)

    assert np.allclose(out, [1.0, 2.0, 1.0])


def test_unproject_depth_map_identity_camera():
    node = object.__new__(ExtrinsicSolverNode)
    depth = torch.tensor([[[[1.0], [2.0]], [[3.0], [4.0]]]])
    extrinsic = torch.eye(4)[:3].unsqueeze(0)
    intrinsic = torch.eye(3).unsqueeze(0)

    points = node.unproject_depth_map_to_point_map(depth, extrinsic, intrinsic)

    expected = np.array([
        [
            [[0.0, 0.0, 1.0], [2.0, 0.0, 2.0]],
            [[0.0, 3.0, 3.0], [4.0, 4.0, 4.0]],
        ]
    ])
    assert np.allclose(points, expected)


if __name__ == '__main__':
    test_blend_depth_only_fills_camera_holes()
    test_unproject_depth_map_identity_camera()
