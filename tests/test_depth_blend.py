import numpy as np

from exo_head_slam.utils.vision_utils import (
    blend_depth_holes,
    transform_points,
    unproject_depth_np,
)


def test_blend_depth_only_fills_camera_holes():
    raw = np.array([1.0, 0.0, 1.0], dtype=np.float32)
    pred = np.array([2.0, 2.0, 1.1], dtype=np.float32)
    raw_valid = np.array([True, False, True])
    conf = np.array([True, True, True])

    out = blend_depth_holes(raw, pred, raw_valid, conf, gate_by_confidence=True)

    assert np.allclose(out, [1.0, 2.0, 1.0])


def test_unproject_depth_map_identity_camera():
    depth = np.array([[1.0, 2.0], [3.0, 4.0]])
    extrinsic = np.eye(4)[:3]
    intrinsic = np.eye(3)

    points = unproject_depth_np(depth, extrinsic, intrinsic)

    expected = np.array([
        [[0.0, 0.0, 1.0], [2.0, 0.0, 2.0]],
        [[0.0, 3.0, 3.0], [4.0, 4.0, 4.0]],
    ])
    assert np.allclose(points, expected)


def test_transform_points_into_exo_link():
    points = np.array([[0.0, 0.0, 1.0], [1.0, 2.0, 3.0]])
    exo_from_world = np.eye(4)
    exo_from_world[:3, :3] = [
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    exo_from_world[:3, 3] = [0.5, -1.0, 2.0]

    transformed = transform_points(points, exo_from_world)

    assert np.allclose(transformed, [[0.5, -1.0, 3.0], [-1.5, 0.0, 5.0]])


if __name__ == '__main__':
    test_blend_depth_only_fills_camera_holes()
    test_unproject_depth_map_identity_camera()
    test_transform_points_into_exo_link()
