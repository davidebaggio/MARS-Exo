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
from rtabmap_msgs.msg import Info, MapData, MapGraph
from scipy.spatial.transform import Rotation, Slerp
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_srvs.srv import Trigger
from tf2_msgs.msg import TFMessage

from exo_head_slam.cloud_map_evaluator_node import (
    cloud_metrics as current_cloud_metrics,
    voxel_downsample,
)


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


def similarity_alignment(source, target):
    """Return Sim(3) mapping source points into target coordinates."""
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    centered_source = source - source_center
    centered_target = target - target_center
    u, singular_values, vt = np.linalg.svd(centered_source.T @ centered_target)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        singular_values[-1] *= -1
        rotation = vt.T @ u.T
    source_variance = np.sum(centered_source ** 2)
    if source_variance <= np.finfo(float).eps:
        raise ValueError('Cannot align a zero-motion trajectory')
    scale = singular_values.sum() / source_variance
    translation = target_center - scale * rotation @ source_center
    return float(scale), rotation, translation


def visualization_alignment(world_from_camera, map_start_z):
    viz_ground_from_map = np.eye(4)
    viz_ground_from_map[2, 3] = map_start_z
    return world_from_camera @ np.linalg.inv(viz_ground_from_map)


def trajectory_from_graph(graph, stamps):
    """Join final optimized RTAB poses to timestamps reported for node IDs."""
    trajectory = []
    for node_id, pose in zip(graph.poses_id, graph.poses):
        if node_id not in stamps:
            continue
        quaternion = np.array([
            pose.orientation.x, pose.orientation.y,
            pose.orientation.z, pose.orientation.w,
        ])
        norm = np.linalg.norm(quaternion)
        if not np.isfinite(norm) or norm < 1e-12:
            continue
        trajectory.append((
            stamps[node_id],
            np.array([pose.position.x, pose.position.y, pose.position.z]),
            quaternion / norm,
        ))
    return trajectory


def match_trajectories(
    estimated, ground_truth, waist_from_camera, max_interpolation_gap=None
):
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

    estimated_times, estimated_unique = np.unique(
        [row[0] for row in estimated], return_index=True
    )
    estimated_positions = np.asarray([estimated[i][1] for i in estimated_unique])
    estimated_rotations = Rotation.from_quat(
        [estimated[i][2] for i in estimated_unique]
    )
    if len(estimated_times) < 3 or len(gt_times) < 2:
        raise ValueError('Not enough overlapping trajectory samples')

    if max_interpolation_gap is None:
        valid = (estimated_times >= gt_times[0]) & (estimated_times <= gt_times[-1])
        times = estimated_times[valid]
        estimated_positions = estimated_positions[valid]
        estimated_quaternions = estimated_rotations[valid].as_quat()
        gt_matched_positions = np.column_stack([
            np.interp(times, gt_times, camera_positions[:, axis]) for axis in range(3)
        ])
        gt_matched_quaternions = Slerp(
            gt_times, Rotation.from_quat(camera_rotations)
        )(times).as_quat()
        segments = np.zeros(len(times), dtype=int)
        target_count = len(times)
    else:
        target_indices = np.flatnonzero(
            (gt_times >= estimated_times[0]) & (gt_times <= estimated_times[-1])
        )
        target_times = gt_times[target_indices]
        right = np.searchsorted(estimated_times, target_times, side='right')
        left = np.maximum(right - 1, 0)
        right = np.minimum(right, len(estimated_times) - 1)
        exact = np.isclose(target_times, estimated_times[left], atol=1e-6)
        gaps = estimated_times[right] - estimated_times[left]
        covered = exact | ((right > left) & (gaps <= max_interpolation_gap))
        target_indices = target_indices[covered]
        times = gt_times[target_indices]
        if len(times) < 3:
            raise ValueError('Fewer than three ground-truth samples are covered')
        estimated_positions = np.column_stack([
            np.interp(times, estimated_times, estimated_positions[:, axis])
            for axis in range(3)
        ])
        estimated_quaternions = Slerp(
            estimated_times, estimated_rotations
        )(times).as_quat()
        gt_matched_positions = camera_positions[target_indices]
        gt_matched_quaternions = np.asarray(camera_rotations)[target_indices]
        segments = np.cumsum(np.r_[
            0,
            (np.diff(target_indices) > 1)
            | (np.diff(times) > max_interpolation_gap),
        ])
        target_count = len(gt_times)

    observed_duration = float(np.sum(
        np.diff(times)[segments[1:] == segments[:-1]]
    ))
    return {
        'times': times,
        'estimated_positions': estimated_positions,
        'estimated_quaternions': estimated_quaternions,
        'gt_positions': gt_matched_positions,
        'gt_quaternions': gt_matched_quaternions,
        'segments': segments,
        'observed_duration_s': observed_duration,
        'target_count': int(target_count),
        'raw_estimated_count': int(len(estimated_times)),
    }


def _trajectory_metrics(matched, scale, align_rotation, align_translation):
    estimated = matched['estimated_positions']
    ground_truth = matched['gt_positions']
    aligned = scale * (estimated @ align_rotation.T) + align_translation

    aligned_rotations = Rotation.from_matrix(align_rotation) * Rotation.from_quat(
        matched['estimated_quaternions']
    )
    gt_rotations = Rotation.from_quat(matched['gt_quaternions'])

    translation_error = np.linalg.norm(aligned - ground_truth, axis=1)
    rotation_error = np.degrees(
        (gt_rotations.inv() * aligned_rotations).magnitude()
    )

    times = matched['times']
    tolerance = 0.1
    if len(times) > 1:
        tolerance = max(tolerance, 0.55 * np.median(np.diff(times)))
    start_indices = []
    end_indices = []
    segments = matched.get('segments', np.zeros(len(times), dtype=int))
    for start, desired in enumerate(times + 1.0):
        insertion = np.searchsorted(times, desired)
        candidates = [index for index in (insertion - 1, insertion)
                      if start < index < len(times)]
        if not candidates:
            continue
        end = min(candidates, key=lambda index: abs(times[index] - desired))
        if abs(times[end] - desired) <= tolerance and segments[start] == segments[end]:
            start_indices.append(start)
            end_indices.append(end)
    start_indices = np.asarray(start_indices, dtype=int)
    end_indices = np.asarray(end_indices, dtype=int)

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
        estimated_distance = np.linalg.norm(
            aligned[end_indices] - aligned[start_indices], axis=1
        )
        gt_distance = np.linalg.norm(
            ground_truth[end_indices] - ground_truth[start_indices], axis=1
        )
        moving = gt_distance > 1e-6
        scale_error = 100.0 * np.abs(
            estimated_distance[moving] / gt_distance[moving] - 1.0
        )
    else:
        rpe_translation = rpe_rotation = scale_error = np.array([])

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
        'target_sample_count': int(matched.get('target_count', len(times))),
        'raw_estimated_count': int(matched.get('raw_estimated_count', len(times))),
        'tracking_coverage': float(
            len(times) / matched.get('target_count', len(times))
        ),
        'tracking_lost_samples': int(
            matched.get('target_count', len(times)) - len(times)
        ),
        'tracked_segments': int(len(np.unique(segments))),
        'duration_s': float(matched.get('observed_duration_s', times[-1] - times[0])),
        'elapsed_span_s': float(times[-1] - times[0]),
        'ate_translation_m': stats(translation_error),
        'ate_rotation_deg': stats(rotation_error),
        'rpe_1s_translation_m': stats(rpe_translation),
        'rpe_1s_rotation_deg': stats(rpe_rotation),
        'rpe_1s_scale_error_percent': stats(scale_error),
    }, align_rotation, align_translation, aligned


def trajectory_metrics(matched):
    rotation, translation = rigid_alignment(
        matched['estimated_positions'], matched['gt_positions']
    )
    return _trajectory_metrics(matched, 1.0, rotation, translation)


def similarity_trajectory_metrics(matched):
    scale, rotation, translation = similarity_alignment(
        matched['estimated_positions'], matched['gt_positions']
    )
    metrics, _, _, aligned = _trajectory_metrics(
        matched, scale, rotation, translation
    )
    return metrics, scale, rotation, translation, aligned


def cloud_metrics(estimated, ground_truth, voxel_size=0.05):
    estimated = voxel_downsample(estimated, voxel_size)
    ground_truth = voxel_downsample(ground_truth, voxel_size)
    if not len(estimated) or not len(ground_truth):
        raise ValueError('Cannot evaluate an empty point cloud')
    result = {
        'estimated_points': int(len(estimated)),
        'ground_truth_points': int(len(ground_truth)),
        'voxel_size': voxel_size,
        'fscore_threshold': 0.10,
        **current_cloud_metrics(
            estimated, ground_truth, 0.10, extra_thresholds=(0.02, 0.05, 0.10, 0.20)
        ),
    }
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
        self.declare_parameter('waist_to_exo_translation', [0.07, 0.0, 0.0])
        self.declare_parameter('waist_to_exo_pitch_deg', 0.0)
        self.declare_parameter('ground_truth_odom_topic', '/exoskeleton/odom')
        self.declare_parameter('require_ground_truth_map', True)
        self.declare_parameter('ground_truth_is_camera_pose', False)
        self.output_prefix = self.get_parameter('output_prefix').value
        self.map_start_z = float(self.get_parameter('map_start_z').value)
        self.require_ground_truth_map = bool(
            self.get_parameter('require_ground_truth_map').value
        )

        self.estimated_trajectory = []
        self.rtab_stamps = {}
        self.optimized_graph = None
        self.gt_trajectory = []
        self.waist_from_camera = _transform(
            self.get_parameter('waist_to_exo_translation').value,
            Rotation.from_euler(
                'y', self.get_parameter('waist_to_exo_pitch_deg').value,
                degrees=True,
            ).as_quat(),
        )
        if self.get_parameter('ground_truth_is_camera_pose').value:
            self.waist_from_camera = np.linalg.inv(_transform(
                [0.0, 0.0, 0.0], [0.5, -0.5, 0.5, -0.5]
            ))
        self.estimated_map = None
        self.gt_map = None
        self.estimated_map_frame = None
        self.map_from_odom = None
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
            Odometry, '/exo_rtabmap/odom', self._estimated_odom, qos_profile_sensor_data
        )
        self.create_subscription(
            Info, '/info', self._rtab_info, 10
        )
        self.create_subscription(
            MapGraph, '/mapGraph', self._map_graph, transient_qos,
        )
        self.create_subscription(
            MapData, '/mapData', self._map_data, 10,
        )
        self.create_subscription(
            Odometry, self.get_parameter('ground_truth_odom_topic').value,
            self._gt_odom, qos_profile_sensor_data
        )
        self.create_subscription(
            PointCloud2, '/exo_rtabmap/cloud_map', self._estimated_map,
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
        self.create_subscription(
            TFMessage, '/tf', self._estimated_tf, qos_profile_sensor_data
        )
        self.create_service(Trigger, '~/finalize', self._finalize)

    @staticmethod
    def _pose(message):
        stamp = message.header.stamp
        pose = message.pose.pose
        position = np.array([pose.position.x, pose.position.y, pose.position.z])
        quaternion = np.array([
            pose.orientation.x, pose.orientation.y,
            pose.orientation.z, pose.orientation.w,
        ])
        norm = np.linalg.norm(quaternion)
        if not np.isfinite(position).all() or not np.isfinite(norm) or norm < 1e-12:
            return None
        return (
            stamp.sec + stamp.nanosec * 1e-9,
            position,
            quaternion / norm,
        )

    def _estimated_odom(self, message):
        pose = self._pose(message)
        if pose is not None:
            self.estimated_trajectory.append(pose)

    def _rtab_info(self, message):
        if message.ref_id > 0:
            stamp = message.header.stamp
            self.rtab_stamps[message.ref_id] = stamp.sec + stamp.nanosec * 1e-9

    def _map_graph(self, message):
        self.optimized_graph = message

    def _map_data(self, message):
        self.optimized_graph = message.graph
        for node in message.nodes:
            if node.id > 0 and node.stamp > 0.0:
                self.rtab_stamps[node.id] = node.stamp

    def _gt_odom(self, message):
        pose = self._pose(message)
        if pose is None:
            return
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

    def _estimated_tf(self, message):
        for transform in message.transforms:
            if transform.header.frame_id == 'map' and transform.child_frame_id == 'odom':
                value = transform.transform
                self.map_from_odom = _transform(
                    [value.translation.x, value.translation.y, value.translation.z],
                    [value.rotation.x, value.rotation.y, value.rotation.z, value.rotation.w],
                )

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
        has_maps = self.estimated_map is not None and self.gt_map is not None
        if self.require_ground_truth_map and not has_maps:
            raise ValueError('Missing estimated or ground-truth map')
        if has_maps and self.estimated_map_frame not in ('map', 'odom'):
            raise ValueError(f'Unexpected estimated map frame: {self.estimated_map_frame}')
        if has_maps and self.gt_map_frame != 'world':
            raise ValueError(f'Unexpected ground-truth map frame: {self.gt_map_frame}')

        odom_matched = match_trajectories(
            self.estimated_trajectory, self.gt_trajectory, self.waist_from_camera,
            max_interpolation_gap=1.0,
        )
        odometry, _, _, _ = trajectory_metrics(odom_matched)
        odometry['source_frame'] = 'odom'
        sim_odometry, _, _, _, _ = similarity_trajectory_metrics(odom_matched)

        estimated_trajectory = self.estimated_trajectory
        trajectory_frame = 'odom'
        if self.optimized_graph is not None:
            optimized = trajectory_from_graph(self.optimized_graph, self.rtab_stamps)
            if len(optimized) >= 3:
                estimated_trajectory = optimized
                trajectory_frame = 'map'
        matched = match_trajectories(
            estimated_trajectory, self.gt_trajectory, self.waist_from_camera,
            max_interpolation_gap=1.0 if trajectory_frame == 'odom' else None,
        )
        trajectory, rotation, translation, aligned_positions = trajectory_metrics(matched)
        trajectory['source_frame'] = trajectory_frame
        sim_trajectory, scale, sim_rotation, sim_translation, _ = (
            similarity_trajectory_metrics(matched)
        )
        mapping = sim_mapping = None
        if has_maps:
            estimated_map = self.estimated_map
            if self.estimated_map_frame != trajectory_frame:
                if self.map_from_odom is None:
                    raise ValueError('Missing final map -> odom transform')
                transform = self.map_from_odom
                if trajectory_frame == 'odom':
                    transform = np.linalg.inv(transform)
                estimated_map = estimated_map @ transform[:3, :3].T + transform[:3, 3]
            aligned_map = estimated_map @ rotation.T + translation
            mapping = cloud_metrics(aligned_map, self.gt_map)
            sim_map = scale * (estimated_map @ sim_rotation.T) + sim_translation
            sim_mapping = cloud_metrics(sim_map, self.gt_map)

        result = {
            'trajectory': trajectory,
            'odometry': odometry,
            'map': mapping,
            'alignment': {
                'rotation': rotation.tolist(),
                'translation_m': translation.tolist(),
            },
            'sim3_diagnostic': {
                'trajectory': sim_trajectory,
                'odometry': sim_odometry,
                'map': sim_mapping,
                'alignment': {
                    'scale': scale,
                    'rotation': sim_rotation.tolist(),
                    'translation_m': sim_translation.tolist(),
                },
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
    except Exception:
        if rclpy.ok():
            raise
    finally:
        if rclpy.ok():
            node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
