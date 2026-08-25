import numpy as np

from exo_head_slam.cloud_map_evaluator_node import cloud_metrics, transform_matrix, voxel_downsample


def test_cloud_helpers():
    transform = transform_matrix([1, 2, 3], [0, 0, 0, 1])
    assert np.allclose(np.array([[0, 0, 0]]) @ transform[:3, :3].T + transform[:3, 3], [[1, 2, 3]])

    points = np.array([[0.01, 0, 0], [0.02, 0, 0], [1, 0, 0]])
    downsampled = voxel_downsample(points, 0.1)
    assert len(downsampled) == 2

    metrics = cloud_metrics(downsampled, downsampled, 0.1)
    assert metrics['chamfer'] == 0.0
    assert metrics['fscore'] == 1.0
