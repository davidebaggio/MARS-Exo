#!/usr/bin/env python3
import numpy as np
from scipy.spatial.transform import Rotation

from exo_head_slam.benchmark_evaluator_node import (
    cloud_metrics,
    first_pose_alignment,
    match_trajectories,
    trajectory_metrics,
    visualization_alignment,
)


def test_first_overlap_alignment_preserves_later_drift():
    times = np.linspace(0.0, 3.0, 31)
    estimated_positions = np.column_stack([
        times, np.sin(times), 0.2 * np.cos(2.0 * times),
    ])
    estimated_rotations = Rotation.from_euler('z', 0.1 * times)
    alignment = Rotation.from_euler('xyz', [0.2, -0.1, 0.3])
    translation = np.array([1.0, -2.0, 0.5])
    gt_positions = alignment.apply(estimated_positions) + translation
    gt_rotations = alignment * estimated_rotations

    estimated = list(zip(
        times, estimated_positions, estimated_rotations.as_quat()
    ))
    ground_truth = list(zip(times, gt_positions, gt_rotations.as_quat()))
    estimated[-1] = (
        estimated[-1][0],
        estimated[-1][1] + np.array([1.0, 0.0, 0.0]),
        estimated[-1][2],
    )
    matched = match_trajectories(estimated, ground_truth, np.eye(4))
    world_from_map = first_pose_alignment(matched)
    metrics, rotation, translation, aligned = trajectory_metrics(
        matched, world_from_map
    )

    assert np.allclose(aligned[0], matched['gt_positions'][0])
    assert metrics['ate_rotation_deg']['rmse'] < 1e-10
    assert metrics['ate_translation_m']['rmse'] > 0.1
    assert metrics['rpe_1s_translation_m']['rmse'] > 0.1
    assert np.allclose(
        estimated_positions @ rotation.T + translation,
        gt_positions,
    )


def test_first_predicted_pose_inside_gt_range_uses_interpolated_camera_pose():
    estimated = [
        (time, np.array([time, 0.0, 0.0]), Rotation.identity().as_quat())
        for time in (0.0, 2.5, 3.0, 3.5)
    ]
    ground_truth = [
        (time, np.array([0.0, time, 0.0]), Rotation.identity().as_quat())
        for time in (2.0, 4.0)
    ]
    waist_from_camera = np.eye(4)
    waist_from_camera[0, 3] = 0.5

    matched = match_trajectories(estimated, ground_truth, waist_from_camera)
    world_from_map = first_pose_alignment(matched)
    aligned_reference = world_from_map @ np.r_[
        matched['estimated_positions'][0], 1.0
    ]

    assert matched['times'][0] == 2.5
    assert np.allclose(matched['gt_positions'][0], [0.5, 2.5, 0.0])
    assert np.allclose(aligned_reference[:3], matched['gt_positions'][0])


def test_visualization_and_cloud_use_same_fixed_transform():
    world_from_map = np.eye(4)
    world_from_map[:3, :3] = Rotation.from_euler('z', 0.3).as_matrix()
    world_from_map[:3, 3] = [2.0, 3.0, 1.0]
    world_from_ground = visualization_alignment(world_from_map, 1.0)
    ground_from_map = np.eye(4)
    ground_from_map[2, 3] = 1.0
    assert np.allclose(world_from_ground @ ground_from_map, world_from_map)

    points = np.random.default_rng(0).normal(size=(100, 3))
    aligned_points = points @ world_from_map[:3, :3].T + world_from_map[:3, 3]
    mapping = cloud_metrics(aligned_points, aligned_points.copy())
    assert mapping['accuracy_rmse_m'] == 0.0
    assert mapping['completeness_rmse_m'] == 0.0
    assert mapping['fscore_05cm'] == 1.0


if __name__ == '__main__':
    test_first_overlap_alignment_preserves_later_drift()
    test_first_predicted_pose_inside_gt_range_uses_interpolated_camera_pose()
    test_visualization_and_cloud_use_same_fixed_transform()
    print('benchmark evaluator checks passed')
