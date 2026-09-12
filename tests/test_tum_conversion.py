import importlib.util
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


path = Path(__file__).parents[1] / 'data/TUM/add_pointclouds_to_bagfile.py'
spec = importlib.util.spec_from_file_location('tum_converter', path)
converter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(converter)


def test_groundtruth_interpolation_uses_translation_and_slerp():
    groundtruth = np.array([
        [0.0, 0.0, 0.0, 0.0, *Rotation.identity().as_quat()],
        [2.0, 2.0, 0.0, 0.0, *Rotation.from_euler('z', 90, degrees=True).as_quat()],
    ])

    pose = converter.interpolate_groundtruth(groundtruth, 1_000_000_000, max_gap=2.0)

    assert np.allclose(pose[1:4], [1.0, 0.0, 0.0])
    assert np.allclose(
        Rotation.from_quat(pose[4:8]).as_euler('zxy', degrees=True)[0], 45.0
    )


def test_identical_camera_poses_make_identity_pair_transform():
    pose = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    points = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 2.0]])

    cloud, pair_transform = converter.paired_groundtruth(points, pose, points, pose)

    expected = points @ converter.LINK_FROM_OPTICAL[:3, :3].T
    assert np.allclose(cloud, np.vstack([expected, expected]))
    assert np.allclose(pair_transform, np.eye(4))
