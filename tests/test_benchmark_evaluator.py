#!/usr/bin/env python3
import numpy as np
from scipy.spatial.transform import Rotation
from nav_msgs.msg import Odometry
from rtabmap_msgs.msg import MapGraph

from exo_head_slam.benchmark_evaluator_node import (
    BenchmarkEvaluatorNode,
    cloud_metrics,
    match_trajectories,
    trajectory_metrics,
    similarity_trajectory_metrics,
    trajectory_from_graph,
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

    graph = MapGraph()
    graph.poses_id = [4]
    graph.poses = [Odometry().pose.pose]
    graph.poses[0].position.x = 2.0
    graph.poses[0].orientation.w = 1.0
    graph_trajectory = trajectory_from_graph(graph, {4: 12.0})
    assert graph_trajectory[0][0] == 12.0
    assert np.array_equal(graph_trajectory[0][1], [2.0, 0.0, 0.0])

    gap_times = np.r_[np.linspace(0.0, 1.0, 11), np.linspace(130.0, 131.0, 11)]
    gap_trajectory = list(zip(
        gap_times,
        np.column_stack([gap_times, np.zeros((len(gap_times), 2))]),
        Rotation.identity(len(gap_times)).as_quat(),
    ))
    gap_matched = match_trajectories(
        gap_trajectory, gap_trajectory, np.eye(4), max_interpolation_gap=1.0
    )
    gap_metrics, _, _, _ = trajectory_metrics(gap_matched)
    assert gap_metrics['tracked_segments'] == 2
    assert np.isclose(gap_metrics['duration_s'], 2.0)
    assert np.isclose(gap_metrics['elapsed_span_s'], 131.0)

    sparse_times = np.arange(0.0, 10.1, 2.0)
    dense_times = np.arange(0.0, 10.1, 0.1)
    sparse = list(zip(
        sparse_times,
        np.column_stack([sparse_times, np.zeros((len(sparse_times), 2))]),
        Rotation.identity(len(sparse_times)).as_quat(),
    ))
    dense = list(zip(
        dense_times,
        np.column_stack([dense_times, np.zeros((len(dense_times), 2))]),
        Rotation.identity(len(dense_times)).as_quat(),
    ))
    sparse_matched = match_trajectories(sparse, dense, np.eye(4))
    assert np.array_equal(sparse_matched['times'], sparse_times)
    assert trajectory_metrics(sparse_matched)[0]['ate_translation_m']['rmse'] < 1e-10


if __name__ == '__main__':
    demo()
    print('benchmark evaluator checks passed')
