import numpy as np
from scipy.spatial.transform import Rotation

from exo_head_slam.extrinsic_solver_node import combined_cloud_points
from exo_head_slam.utils.math_utils import compute_transform_ransac
from exo_head_slam.utils.vision_utils import dilate_mask, sanitize_depth


def test_ransac_and_combined_cloud_are_metric_and_repeatable():
    rng = np.random.default_rng(4)
    source = rng.normal(size=(100, 3))
    rotation = Rotation.from_euler('xyz', [0.1, -0.2, 0.3]).as_matrix()
    translation = np.array([0.4, -0.1, 0.7])
    target = source @ rotation.T + translation
    target[:20] = rng.normal(size=(20, 3))

    first = compute_transform_ransac(
        source, target, threshold=0.01, iterations=200,
        rng=np.random.default_rng(7),
    )
    second = compute_transform_ransac(
        source, target, threshold=0.01, iterations=200,
        rng=np.random.default_rng(7),
    )
    assert np.allclose(first[0], second[0])
    assert first[1] >= 80
    assert np.allclose(first[0][:3, :3], rotation)
    assert np.allclose(first[0][:3, 3], translation)

    depth = np.array([[1.0, np.nan], [0.0, 2.0]], dtype=np.float32)
    intrinsics = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    exo_from_head = np.eye(4)
    exo_from_head[0, 3] = 1.0
    cloud = combined_cloud_points(
        depth, depth, intrinsics, intrinsics,
        np.eye(4), np.eye(4), exo_from_head, stride=1,
    )
    assert cloud.shape == (4, 3)
    assert any(np.allclose(point, [1.0, 0.0, 1.0]) for point in cloud)
    assert any(np.allclose(point, [0.0, 0.0, 1.0]) for point in cloud)
    assert np.isfinite(cloud).all()


def test_depth_units_and_mask_dilation():
    depth = sanitize_depth(
        np.array([[1000, 0, 7000]], dtype=np.uint16), '16UC1',
        max_depth=6.0,
    )
    assert np.array_equal(depth, [[1.0, 0.0, 0.0]])
    mask = np.zeros((5, 5), dtype=np.uint8)
    mask[2, 2] = 1
    assert dilate_mask(mask, 1).sum() == 9
