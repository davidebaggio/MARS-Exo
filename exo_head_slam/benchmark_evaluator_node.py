import copy
import csv
import json
import os

import numpy as np
import rclpy
import tf2_ros
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation, Slerp
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_srvs.srv import Trigger
from tf2_msgs.msg import TFMessage


def _transform(position, quaternion):
    result = np.eye(4)
    result[:3, :3] = Rotation.from_quat(quaternion).as_matrix()
    result[:3, 3] = position
    return result


def rigid_alignment(source, target):
    """Return metric SE(3) mapping source points into target coordinates."""
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    u, _, vt = np.linalg.svd((source - source_center).T @ (target - target_center))
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    translation = target_center - rotation @ source_center
    return rotation, translation


def visualization_alignment(world_from_camera, map_start_z):
    viz_ground_from_map = np.eye(4)
    viz_ground_from_map[2, 3] = map_start_z
    return world_from_camera @ np.linalg.inv(viz_ground_from_map)


def match_trajectories(estimated, ground_truth, waist_from_camera):
    estimated = sorted(estimated, key=lambda row: row[0])
    ground_truth = sorted(ground_truth, key=lambda row: row[0])
    gt_times, unique = np.unique([row[0] for row in ground_truth], return_index=True)
    gt_positions = np.asarray([ground_truth[i][1] for i in unique])
    gt_rotations = Rotation.from_quat([ground_truth[i][2] for i in unique])

    camera_positions = []
    camera_rotations = []
    for position, rotation in zip(gt_positions, gt_rotations):
        world_from_camera = _transform(position, rotation.as_quat()) @ waist_from_camera
        camera_positions.append(world_from_camera[:3, 3])
        camera_rotations.append(Rotation.from_matrix(world_from_camera[:3, :3]).as_quat())
    camera_positions = np.asarray(camera_positions)

    valid_estimated = [
        row for row in estimated if gt_times[0] <= row[0] <= gt_times[-1]
    ]
    times = np.asarray([row[0] for row in valid_estimated])
    if len(times) < 3 or len(gt_times) < 2:
        raise ValueError('Not enough overlapping trajectory samples')

    gt_interp_positions = np.column_stack([
        np.interp(times, gt_times, camera_positions[:, axis]) for axis in range(3)
    ])
    gt_interp_rotations = Slerp(
        gt_times, Rotation.from_quat(camera_rotations)
    )(times).as_quat()

    return {
        'times': times,
        'estimated_positions': np.asarray([row[1] for row in valid_estimated]),
        'estimated_quaternions': np.asarray([row[2] for row in valid_estimated]),
        'gt_positions': gt_interp_positions,
        'gt_quaternions': gt_interp_rotations,
    }


def trajectory_metrics(matched):
    estimated = matched['estimated_positions']
    ground_truth = matched['gt_positions']
    align_rotation, align_translation = rigid_alignment(estimated, ground_truth)
    aligned = estimated @ align_rotation.T + align_translation

    aligned_rotations = Rotation.from_matrix(align_rotation) * Rotation.from_quat(
        matched['estimated_quaternions']
    )
    gt_rotations = Rotation.from_quat(matched['gt_quaternions'])

    translation_error = np.linalg.norm(aligned - ground_truth, axis=1)
    rotation_error = np.degrees(
        (gt_rotations.inv() * aligned_rotations).magnitude()
    )

    times = matched['times']
    end_indices = np.searchsorted(times, times + 1.0)
    start_indices = np.arange(len(times))
    valid = (end_indices < len(times))
    valid &= np.where(valid, np.abs(times[np.minimum(end_indices, len(times) - 1)] - times - 1.0) <= 0.1, False)
    start_indices = start_indices[valid]
    end_indices = end_indices[valid]

    if len(start_indices):
        estimated_delta = aligned_rotations[start_indices].inv().apply(
            aligned[end_indices] - aligned[start_indices]
        )
        gt_delta = gt_rotations[start_indices].inv().apply(
            ground_truth[end_indices] - ground_truth[start_indices]
        )
        rpe_translation = np.linalg.norm(estimated_delta - gt_delta, axis=1)
        estimated_relative_rotation = (
            aligned_rotations[start_indices].inv() * aligned_rotations[end_indices]
        )
        gt_relative_rotation = gt_rotations[start_indices].inv() * gt_rotations[end_indices]
        rpe_rotation = np.degrees(
            (gt_relative_rotation.inv() * estimated_relative_rotation).magnitude()
        )
    else:
        rpe_translation = rpe_rotation = np.array([])

    def stats(values):
        if not len(values):
            return None
        return {
            'rmse': float(np.sqrt(np.mean(values ** 2))),
            'mean': float(np.mean(values)),
            'median': float(np.median(values)),
            'max': float(np.max(values)),
        }

    return {
        'sample_count': int(len(times)),
        'duration_s': float(times[-1] - times[0]),
        'ate_translation_m': stats(translation_error),
        'ate_rotation_deg': stats(rotation_error),
        'rpe_1s_translation_m': stats(rpe_translation),
        'rpe_1s_rotation_deg': stats(rpe_rotation),
    }, align_rotation, align_translation, aligned


def voxel_downsample(points, voxel_size=0.05):
    # ponytail: one point per voxel; use centroids only if map-density bias appears.
    _, indices = np.unique(
        np.floor(points / voxel_size).astype(np.int64), axis=0, return_index=True
    )
    return points[indices]


def cloud_metrics(estimated, ground_truth):
    estimated = voxel_downsample(estimated)
    ground_truth = voxel_downsample(ground_truth)
    if not len(estimated) or not len(ground_truth):
        raise ValueError('Cannot evaluate an empty point cloud')
    estimated_to_gt = cKDTree(ground_truth).query(estimated, workers=-1)[0]
    gt_to_estimated = cKDTree(estimated).query(ground_truth, workers=-1)[0]

    result = {
        'estimated_points': int(len(estimated)),
        'ground_truth_points': int(len(ground_truth)),
        'accuracy_rmse_m': float(np.sqrt(np.mean(estimated_to_gt ** 2))),
        'completeness_rmse_m': float(np.sqrt(np.mean(gt_to_estimated ** 2))),
        'symmetric_mean_distance_m': float(
            (np.mean(estimated_to_gt) + np.mean(gt_to_estimated)) / 2.0
        ),
    }
    for threshold in (0.05, 0.10):
        precision = float(np.mean(estimated_to_gt <= threshold))
        recall = float(np.mean(gt_to_estimated <= threshold))
        result[f'fscore_{int(threshold * 100):02d}cm'] = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        )
    return result


def _points(message):
    values = point_cloud2.read_points_numpy(
        message, field_names=['x', 'y', 'z'], skip_nans=True
    )
    if values.dtype.names:
        values = np.column_stack([values[name] for name in ('x', 'y', 'z')])
    return np.asarray(values, dtype=np.float64).reshape(-1, 3)


class BenchmarkEvaluatorNode(Node):
    def __init__(self):
        super().__init__('benchmark_evaluator')
        self.declare_parameter('output_prefix', 'metrics/eval/benchmark')
        self.declare_parameter('map_start_z', 1.0)
        self.declare_parameter('estimated_odom_topic', '/exo_rtabmap/odom')
        self.declare_parameter('estimated_map_topic', '/exo_rtabmap/cloud_map')
        self.output_prefix = self.get_parameter('output_prefix').value
        self.map_start_z = float(self.get_parameter('map_start_z').value)
        estimated_odom_topic = self.get_parameter('estimated_odom_topic').value
        estimated_map_topic = self.get_parameter('estimated_map_topic').value

        self.estimated_trajectory = []
        self.gt_trajectory = []
        self.waist_from_camera = None
        self.estimated_map = None
        self.gt_map = None
        self.estimated_map_frame = None
        self.gt_map_frame = None
        self.first_gt_pose = None
        self.visualization_tf_sent = False
        self.static_broadcaster = tf2_ros.StaticTransformBroadcaster(self)
        self.gt_tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        transient_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.create_subscription(
            Odometry, estimated_odom_topic, self._estimated_odom, qos_profile_sensor_data
        )
        self.create_subscription(
            Odometry, '/exoskeleton/odom', self._gt_odom, qos_profile_sensor_data
        )
        self.create_subscription(
            PointCloud2, estimated_map_topic, self._estimated_map,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            PointCloud2, '/ground_truth/visible_map', self._gt_map, transient_qos
        )
        self.create_subscription(
            TFMessage, '/ground_truth/tf_static', self._gt_static, transient_qos
        )
        self.create_subscription(
            TFMessage, '/ground_truth/tf', self._gt_dynamic, qos_profile_sensor_data
        )
        self.create_service(Trigger, '~/finalize', self._finalize)

    @staticmethod
    def _pose(message):
        stamp = message.header.stamp
        pose = message.pose.pose
        return (
            stamp.sec + stamp.nanosec * 1e-9,
            np.array([pose.position.x, pose.position.y, pose.position.z]),
            np.array([
                pose.orientation.x, pose.orientation.y,
                pose.orientation.z, pose.orientation.w,
            ]),
        )

    def _estimated_odom(self, message):
        self.estimated_trajectory.append(self._pose(message))

    def _gt_odom(self, message):
        pose = self._pose(message)
        self.gt_trajectory.append(pose)
        if self.first_gt_pose is None:
            self.first_gt_pose = pose
        self._publish_visualization_tf()

    def _estimated_map(self, message):
        self.estimated_map = _points(message)
        self.estimated_map_frame = message.header.frame_id

    def _gt_map(self, message):
        self.gt_map = _points(message)
        self.gt_map_frame = message.header.frame_id

    def _gt_static(self, message):
        for transform in message.transforms:
            if (
                transform.header.frame_id == 'waist_link'
                and transform.child_frame_id == 'front_camera_link'
            ):
                value = transform.transform
                self.waist_from_camera = _transform(
                    np.array([
                        value.translation.x, value.translation.y, value.translation.z
                    ]),
                    np.array([
                        value.rotation.x, value.rotation.y,
                        value.rotation.z, value.rotation.w,
                    ]),
                )
                self._publish_visualization_tf()
        self.static_broadcaster.sendTransform([
            self._prefixed_gt_transform(transform) for transform in message.transforms
        ])

    def _gt_dynamic(self, message):
        self.gt_tf_broadcaster.sendTransform([
            self._prefixed_gt_transform(transform) for transform in message.transforms
        ])

    @staticmethod
    def _prefixed_gt_transform(transform):
        result = copy.deepcopy(transform)
        if result.header.frame_id != 'world':
            result.header.frame_id = f'gt_{result.header.frame_id}'
        result.child_frame_id = f'gt_{result.child_frame_id}'
        return result

    def _publish_visualization_tf(self):
        if (
            self.visualization_tf_sent
            or self.first_gt_pose is None
            or self.waist_from_camera is None
        ):
            return

        _, position, quaternion = self.first_gt_pose
        world_from_camera = _transform(position, quaternion) @ self.waist_from_camera
        world_from_viz_ground = visualization_alignment(
            world_from_camera, self.map_start_z
        )

        message = TransformStamped()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = 'world'
        message.child_frame_id = 'viz_ground'
        message.transform.translation.x = float(world_from_viz_ground[0, 3])
        message.transform.translation.y = float(world_from_viz_ground[1, 3])
        message.transform.translation.z = float(world_from_viz_ground[2, 3])
        rotation = Rotation.from_matrix(world_from_viz_ground[:3, :3]).as_quat()
        message.transform.rotation.x = float(rotation[0])
        message.transform.rotation.y = float(rotation[1])
        message.transform.rotation.z = float(rotation[2])
        message.transform.rotation.w = float(rotation[3])
        self.static_broadcaster.sendTransform(message)
        self.visualization_tf_sent = True

    def _finalize(self, _request, response):
        try:
            output = self.evaluate()
            response.success = True
            response.message = output
        except Exception as error:
            response.success = False
            response.message = str(error)
            self.get_logger().error(f'Benchmark finalization failed: {error}')
        return response

    def evaluate(self):
        if self.waist_from_camera is None:
            raise ValueError('Missing waist_link -> front_camera_link ground-truth TF')
        if self.estimated_map is None or self.gt_map is None:
            raise ValueError('Missing estimated or ground-truth map')
        if self.estimated_map_frame not in ('map', 'odom'):
            raise ValueError(f'Unexpected estimated map frame: {self.estimated_map_frame}')
        if self.gt_map_frame != 'world':
            raise ValueError(f'Unexpected ground-truth map frame: {self.gt_map_frame}')

        matched = match_trajectories(
            self.estimated_trajectory, self.gt_trajectory, self.waist_from_camera
        )
        trajectory, rotation, translation, aligned_positions = trajectory_metrics(matched)
        aligned_map = self.estimated_map @ rotation.T + translation
        mapping = cloud_metrics(aligned_map, self.gt_map)

        result = {
            'trajectory': trajectory,
            'map': mapping,
            'alignment': {
                'rotation': rotation.tolist(),
                'translation_m': translation.tolist(),
            },
        }
        output_dir = os.path.dirname(self.output_prefix)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        json_path = f'{self.output_prefix}.json'
        with open(json_path, 'w', encoding='utf-8') as stream:
            json.dump(result, stream, indent=2)

        csv_path = f'{self.output_prefix}_trajectory.csv'
        with open(csv_path, 'w', newline='', encoding='utf-8') as stream:
            writer = csv.writer(stream)
            writer.writerow([
                'timestamp', 'estimated_x', 'estimated_y', 'estimated_z',
                'gt_x', 'gt_y', 'gt_z',
            ])
            for time, estimated, gt in zip(
                matched['times'], aligned_positions, matched['gt_positions']
            ):
                writer.writerow([time, *estimated, *gt])

        self.get_logger().info(f'Benchmark written: {json_path}')
        return json_path


def main(args=None):
    rclpy.init(args=args)
    node = BenchmarkEvaluatorNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
