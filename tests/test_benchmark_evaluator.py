#!/usr/bin/env python3
import numpy as np
from scipy.spatial.transform import Rotation
from nav_msgs.msg import Odometry

from exo_head_slam.benchmark_evaluator_node import (
    BenchmarkEvaluatorNode,
    cloud_metrics,
    match_trajectories,
    trajectory_metrics,
    similarity_trajectory_metrics,
    visualization_alignment,
)


def test_invalid_odometry_quaternion_is_ignored():
    message = Odometry()
    message.pose.pose.orientation.w = 0.0
    assert BenchmarkEvaluatorNode._pose(message) is None
    message.pose.pose.orientation.w = 2.0
    assert np.array_equal(BenchmarkEvaluatorNode._pose(message)[2], [0, 0, 0, 1])


def demo():
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
    matched = match_trajectories(estimated, ground_truth, np.eye(4))
    metrics, _, _, _ = trajectory_metrics(matched)

    assert metrics['ate_translation_m']['rmse'] < 1e-10
    assert metrics['ate_rotation_deg']['rmse'] < 1e-10
    assert metrics['rpe_1s_translation_m']['rmse'] < 1e-10

    scaled = dict(matched)
    scaled['estimated_positions'] = matched['estimated_positions'] * 3.0
    sim_metrics, scale, _, _, _ = similarity_trajectory_metrics(scaled)
    assert abs(scale - 1.0 / 3.0) < 1e-10
    assert sim_metrics['ate_translation_m']['rmse'] < 1e-10

    world_from_camera = np.eye(4)
    world_from_camera[:3, 3] = [2.0, 3.0, 1.0]
    world_from_ground = visualization_alignment(world_from_camera, 1.0)
    ground_from_map = np.eye(4)
    ground_from_map[2, 3] = 1.0
    assert np.allclose(world_from_ground @ ground_from_map, world_from_camera)

    points = np.random.default_rng(0).normal(size=(100, 3))
    mapping = cloud_metrics(points, points.copy())
    assert mapping['accuracy_rmse'] == 0.0
    assert mapping['completeness_rmse'] == 0.0
    assert mapping['fscore'] == 1.0
    assert mapping['fscore_02cm'] == 1.0


if __name__ == '__main__':
    demo()
    print('benchmark evaluator checks passed')
