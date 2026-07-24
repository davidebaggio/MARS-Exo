from types import SimpleNamespace

import numpy as np
import pytest

from exo_head_slam.dense_global_map_node import clear_on_epoch, project_rgbd


def camera_info(width=2, height=2, fx=2.0, fy=2.0, cx=0.0, cy=0.0):
    return SimpleNamespace(
        width=width,
        height=height,
        k=[fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0],
    )


def test_project_rgbd_known_geometry():
    rgb = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
    depth = np.full((2, 2), 2.0, dtype=np.float32)

    points, colors = project_rgbd(rgb, depth, camera_info(), 1, 0.1, 10.0)

    assert np.allclose(points, [[0, 0, 2], [1, 0, 2], [0, 1, 2], [1, 1, 2]])
    assert np.array_equal(colors, rgb.reshape(-1, 3))


def test_epoch_change_clears_cloud():
    points = np.ones((2, 3), dtype=np.float32)
    colors = np.ones((2, 3), dtype=np.uint8)

    epoch, points, colors = clear_on_epoch(3, 4, points, colors)

    assert epoch == 4
    assert points.shape == colors.shape == (0, 3)


def test_same_epoch_preserves_cloud():
    points = np.ones((2, 3), dtype=np.float32)
    colors = np.ones((2, 3), dtype=np.uint8)

    _, kept_points, kept_colors = clear_on_epoch(3, 3, points, colors)

    assert kept_points is points
    assert kept_colors is colors


@pytest.mark.parametrize(
    ('rgb', 'depth', 'info'),
    [
        (np.zeros((2, 3, 3)), np.zeros((2, 2)), camera_info()),
        (np.zeros((2, 2, 3)), np.zeros((2, 2)), camera_info(width=3)),
        (np.zeros((2, 2, 3)), np.zeros((2, 2)), camera_info(fx=0.0)),
    ],
)
def test_invalid_projection_input_is_rejected(rgb, depth, info):
    with pytest.raises(ValueError):
        project_rgbd(rgb, depth, info, 1, 0.1, 10.0)
