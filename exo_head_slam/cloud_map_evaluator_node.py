import csv
import os
import sys

import message_filters
import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2


F_SCORE_THRESHOLDS = (0.02, 0.05, 0.10)


def transform_matrix(translation, quaternion):
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_quat(quaternion).as_matrix()
    matrix[:3, 3] = translation
    return matrix


def voxel_downsample(points, voxel_size):
    if len(points) == 0:
        return points
    keys = np.floor(points / voxel_size).astype(np.int64)
    return points[np.unique(keys, axis=0, return_index=True)[1]]


def merge_voxels(voxels, points, voxel_size):
    keys = np.floor(points / voxel_size).astype(np.int64)
    for key, point in zip(map(tuple, keys), points):
        voxels.setdefault(key, point)


def cloud_metrics(estimated, ground_truth, threshold, extra_thresholds=()):
    est_to_gt = cKDTree(ground_truth).query(estimated, workers=-1)[0]
    gt_to_est = cKDTree(estimated).query(ground_truth, workers=-1)[0]
    precision = float(np.mean(est_to_gt <= threshold))
    recall = float(np.mean(gt_to_est <= threshold))
    result = {
        'accuracy_mean': float(np.mean(est_to_gt)),
        'accuracy_rmse': float(np.sqrt(np.mean(est_to_gt ** 2))),
        'completeness_mean': float(np.mean(gt_to_est)),
        'completeness_rmse': float(np.sqrt(np.mean(gt_to_est ** 2))),
        'chamfer': float((np.mean(est_to_gt) + np.mean(gt_to_est)) / 2.0),
        'fscore': 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0,
    }
    for extra in extra_thresholds:
        extra_precision = float(np.mean(est_to_gt <= extra))
        extra_recall = float(np.mean(gt_to_est <= extra))
        result[f'fscore_{round(extra * 100):02d}cm'] = (
            2.0 * extra_precision * extra_recall / (extra_precision + extra_recall)
            if extra_precision + extra_recall else 0.0
        )
    return result


class CloudMapEvaluatorNode(Node):
    def __init__(self, global_flag=False):
        super().__init__('cloud_map_evaluator')
        self.declare_parameter(
            'estimated_cloud_topic', '/lightglue/combined_pointcloud'
        )
        self.declare_parameter('ground_truth_cloud_topic', '/ground_truth/visible_cloud')
        self.declare_parameter('ground_truth_odom_topic', '/exoskeleton/odom')
        self.declare_parameter('metrics_csv_path', 'cloud_metrics.csv')
        self.declare_parameter('voxel_size', 0.1)
        self.declare_parameter('fscore_threshold', 0.1)
        self.declare_parameter('waist_to_exo_translation', [0.07, 0.0, 0.0])
        self.declare_parameter('waist_to_exo_pitch_deg', 0.0)
        self.declare_parameter('ground_truth_cloud_is_local', False)
        self.declare_parameter('ground_truth_is_camera_pose', False)
        self.declare_parameter('global', True)

        self.metrics_csv_path = self.get_parameter('metrics_csv_path').value
        root, extension = os.path.splitext(self.metrics_csv_path)
        for path in (self.metrics_csv_path, f'{root}_global{extension}',
                     f'{root}_global_maps.npz'):
            try:
                os.remove(path)
            except FileNotFoundError:
                pass
        self.voxel_size = float(self.get_parameter('voxel_size').value)
        self.fscore_threshold = float(self.get_parameter('fscore_threshold').value)
        self.global_mode = global_flag or self.get_parameter('global').value
        self.ground_truth_cloud_is_local = bool(
            self.get_parameter('ground_truth_cloud_is_local').value
        )
        self.estimated_voxels = {}
        self.ground_truth_voxels = {}
        self.last_ground_truth_msg = None
        self.waist_to_exo = transform_matrix(
            self.get_parameter('waist_to_exo_translation').value,
            Rotation.from_euler(
                'y', self.get_parameter('waist_to_exo_pitch_deg').value, degrees=True).as_quat(),
        )
        self.optical_from_exo = np.linalg.inv(transform_matrix(
            [0.0, 0.0, 0.0], [0.5, -0.5, 0.5, -0.5]
        )) if self.get_parameter('ground_truth_is_camera_pose').value else None
        estimated_sub = message_filters.Subscriber(
            self, PointCloud2, self.get_parameter('estimated_cloud_topic').value,
            qos_profile=qos_profile_sensor_data)
        ground_truth_sub = message_filters.Subscriber(
            self, PointCloud2, self.get_parameter('ground_truth_cloud_topic').value,
            qos_profile=qos_profile_sensor_data)
        odom_sub = message_filters.Subscriber(
            self, Odometry, self.get_parameter('ground_truth_odom_topic').value,
            qos_profile=qos_profile_sensor_data)
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [estimated_sub, ground_truth_sub, odom_sub], queue_size=200, slop=0.25)
        self.sync.registerCallback(self.evaluate)
        self.get_logger().info(
            f'Visible-cloud evaluator online (per-frame'
            f'{" + global" if self.global_mode else ""}). '
            f'voxel={self.voxel_size:.3f} m, '
            f'F-score threshold={self.fscore_threshold:.3f} m')

    @staticmethod
    def points(msg):
        return point_cloud2.read_points_numpy(
            msg, field_names=['x', 'y', 'z'], skip_nans=True).astype(np.float64)

    def evaluate(self, estimated_msg, ground_truth_msg, odom_msg):
        if self.ground_truth_cloud_is_local:
            if ground_truth_msg.header.frame_id != estimated_msg.header.frame_id:
                self.get_logger().error(
                    f'Cannot compare local frames {estimated_msg.header.frame_id!r} '
                    f'and {ground_truth_msg.header.frame_id!r}.')
                return
        elif ground_truth_msg.header.frame_id != odom_msg.header.frame_id:
            self.get_logger().error(
                f'Cannot compare frames {ground_truth_msg.header.frame_id!r} '
                f'and {odom_msg.header.frame_id!r}.')
            return
        pose = odom_msg.pose.pose
        world_from_pose = transform_matrix(
            [pose.position.x, pose.position.y, pose.position.z],
            [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w],
        )
        world_from_exo = world_from_pose @ (
            self.optical_from_exo if self.optical_from_exo is not None
            else self.waist_to_exo
        )
        estimated = self.points(estimated_msg)
        estimated = estimated[np.isfinite(estimated).all(axis=1)]
        ground_truth = self.points(ground_truth_msg)
        local_estimated = estimated
        local_ground_truth = ground_truth
        if not self.ground_truth_cloud_is_local:
            estimated = (
                estimated @ world_from_exo[:3, :3].T + world_from_exo[:3, 3]
            )
        if self.global_mode:
            global_estimated = (
                local_estimated @ world_from_exo[:3, :3].T + world_from_exo[:3, 3]
                if self.ground_truth_cloud_is_local else estimated
            )
            global_ground_truth = (
                local_ground_truth @ world_from_exo[:3, :3].T + world_from_exo[:3, 3]
                if self.ground_truth_cloud_is_local else ground_truth
            )
            merge_voxels(self.estimated_voxels, global_estimated, self.voxel_size)
            merge_voxels(self.ground_truth_voxels, global_ground_truth, self.voxel_size)
            self.last_ground_truth_msg = ground_truth_msg

        estimated = voxel_downsample(estimated, self.voxel_size)
        ground_truth = voxel_downsample(ground_truth, self.voxel_size)
        if len(estimated) == 0 or len(ground_truth) == 0:
            self.get_logger().warn('Skipping empty estimated or ground-truth visible cloud.')
            return
        metrics = cloud_metrics(
            estimated, ground_truth, self.fscore_threshold, F_SCORE_THRESHOLDS)
        self.write_metrics(ground_truth_msg, len(estimated), len(ground_truth), metrics)
        self.get_logger().info(
            f'Visible cloud evaluation: Chamfer={metrics["chamfer"]:.3f} m, '
            f'F-score={metrics["fscore"]:.3f}, points={len(estimated)}/{len(ground_truth)}')

    def finish_global(self):
        if not self.global_mode or self.last_ground_truth_msg is None:
            return
        estimated = np.asarray(list(self.estimated_voxels.values()))
        ground_truth = np.asarray(list(self.ground_truth_voxels.values()))
        if len(estimated) == 0 or len(ground_truth) == 0:
            if rclpy.ok():
                self.get_logger().warn('Skipping empty global estimated or ground-truth map.')
            return
        if rclpy.ok():
            self.get_logger().info(
                f'Finalizing global maps: {len(estimated)}/{len(ground_truth)} points...')
        metrics = cloud_metrics(
            estimated, ground_truth, self.fscore_threshold, F_SCORE_THRESHOLDS)
        root, extension = os.path.splitext(self.metrics_csv_path)
        self.write_metrics(
            self.last_ground_truth_msg, len(estimated), len(ground_truth), metrics,
            f'{root}_global{extension}')
        np.savez_compressed(
            f'{root}_global_maps.npz',
            predicted=estimated.astype(np.float32),
            ground_truth=ground_truth.astype(np.float32),
        )
        if rclpy.ok():
            self.get_logger().info(
                f'Global visible-map evaluation: Chamfer={metrics["chamfer"]:.3f} m, '
                f'F-score={metrics["fscore"]:.3f}, points={len(estimated)}/{len(ground_truth)}')

    def write_metrics(self, msg, estimated_points, ground_truth_points, metrics, path=None):
        path = path or self.metrics_csv_path
        dirname = os.path.dirname(path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        exists = os.path.exists(path) and os.path.getsize(path) > 0
        row = {
            'timestamp': msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
            'voxel_size': self.voxel_size,
            'fscore_threshold': self.fscore_threshold,
            'estimated_points': estimated_points,
            'ground_truth_points': ground_truth_points,
            **metrics,
        }
        with open(path, 'a', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=row.keys())
            if not exists:
                writer.writeheader()
            writer.writerow(row)


def main(args=None):
    args = list(sys.argv[1:] if args is None else args)
    global_flag = '--global' in args
    rclpy.init(args=[arg for arg in args if arg != '--global'])
    node = CloudMapEvaluatorNode(global_flag)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except Exception:
        if rclpy.ok():
            raise
    finally:
        node.finish_global()
        if rclpy.ok():
            node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
