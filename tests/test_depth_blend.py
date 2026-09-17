import numpy as np

from exo_head_slam.utils.vision_utils import (
    bidirectional_overlap,
    blend_depth_holes,
    depth_edge_mask,
    robust_metric_scale,
    sanitize_depth,
    transform_points,
    unproject_depth_np,
    voxel_keep_first,
)


def test_blend_depth_only_fills_camera_holes():
    raw = np.array([1.0, 0.0, 1.0], dtype=np.float32)
    pred = np.array([2.0, 2.0, 1.1], dtype=np.float32)
    raw_valid = np.array([True, False, True])
    conf = np.array([True, True, True])

    out = blend_depth_holes(raw, pred, raw_valid, conf, gate_by_confidence=True)

    assert np.allclose(out, [1.0, 2.0, 1.0])


def test_depth_sanitization_and_semantic_holes():
    raw = np.array([1000, 0, 7000], dtype=np.uint16)
    assert np.allclose(sanitize_depth(raw, '16UC1'), [1.0, 0.0, 0.0])

    raw = np.array([1.0, 0.0, np.nan], dtype=np.float32)
    out = blend_depth_holes(
        raw, np.full(3, 2.0), np.isfinite(raw) & (raw > 0.0),
        np.ones(3, dtype=bool), True, np.isfinite(raw),
    )
    assert np.allclose(out, [1.0, 2.0, 0.0])


def test_robust_scale_edges_overlap_and_voxel_priority():
    raw = np.full((32, 32), 2.0)
    predicted = np.ones((32, 32))
    predicted[0, 0] = 100.0
    confident = np.ones((32, 32), dtype=bool)
    scale, camera_scales, supports = robust_metric_scale(
        [raw, raw], [predicted, predicted], [confident, confident],
        min_support=100,
    )
    assert np.isclose(scale, 2.0)
    assert np.allclose(camera_scales, [2.0, 2.0])
    assert sum(supports) >= 100
    assert depth_edge_mask(np.array([[1.0, 2.0]])).all()

    intrinsic = np.array([[10.0, 0.0, 8.0], [0.0, 10.0, 8.0], [0.0, 0.0, 1.0]])
    identity = np.eye(4)[:3]
    overlap = bidirectional_overlap(
        (np.ones((16, 16)), np.ones((16, 16))),
        (identity, identity), (intrinsic, intrinsic),
        (np.ones((16, 16), bool), np.ones((16, 16), bool)), stride=2,
    )
    assert overlap == 1.0

    points, colors = voxel_keep_first(
        np.array([[0.001, 0.0, 1.0], [0.009, 0.0, 1.0]]),
        np.array([[1, 2, 3], [9, 9, 9]], dtype=np.uint8), 0.02,
    )
    assert len(points) == 1
    assert colors.tolist() == [[1, 2, 3]]


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
