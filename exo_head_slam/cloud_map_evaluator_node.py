import csv
import os

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


def cloud_metrics(estimated, ground_truth, threshold):
    est_to_gt = cKDTree(ground_truth).query(estimated, workers=-1)[0]
    gt_to_est = cKDTree(estimated).query(ground_truth, workers=-1)[0]
    precision = float(np.mean(est_to_gt <= threshold))
    recall = float(np.mean(gt_to_est <= threshold))
    return {
        'accuracy_mean': float(np.mean(est_to_gt)),
        'accuracy_rmse': float(np.sqrt(np.mean(est_to_gt ** 2))),
        'completeness_mean': float(np.mean(gt_to_est)),
        'completeness_rmse': float(np.sqrt(np.mean(gt_to_est ** 2))),
        'chamfer': float((np.mean(est_to_gt) + np.mean(gt_to_est)) / 2.0),
        'fscore': 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0,
    }


class CloudMapEvaluatorNode(Node):
    def __init__(self):
        super().__init__('cloud_map_evaluator')
        self.declare_parameter('vggt_cloud_topic', '/vggt/combined_pointcloud')
        self.declare_parameter('ground_truth_cloud_topic', '/ground_truth/visible_cloud')
        self.declare_parameter('ground_truth_odom_topic', '/exoskeleton/odom')
        self.declare_parameter('metrics_csv_path', 'cloud_metrics.csv')
        self.declare_parameter('voxel_size', 0.1)
        self.declare_parameter('fscore_threshold', 0.1)
        self.declare_parameter('waist_to_exo_translation', [0.07, 0.0, 0.0])
        self.declare_parameter('waist_to_exo_pitch_deg', 0.0)

        self.metrics_csv_path = self.get_parameter('metrics_csv_path').value
        self.voxel_size = float(self.get_parameter('voxel_size').value)
        self.fscore_threshold = float(self.get_parameter('fscore_threshold').value)
        self.waist_to_exo = transform_matrix(
            self.get_parameter('waist_to_exo_translation').value,
            Rotation.from_euler(
                'y', self.get_parameter('waist_to_exo_pitch_deg').value, degrees=True).as_quat(),
        )
        vggt_sub = message_filters.Subscriber(
            self, PointCloud2, self.get_parameter('vggt_cloud_topic').value,
            qos_profile=qos_profile_sensor_data)
        ground_truth_sub = message_filters.Subscriber(
            self, PointCloud2, self.get_parameter('ground_truth_cloud_topic').value,
            qos_profile=qos_profile_sensor_data)
        odom_sub = message_filters.Subscriber(
            self, Odometry, self.get_parameter('ground_truth_odom_topic').value,
            qos_profile=qos_profile_sensor_data)
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [vggt_sub, ground_truth_sub, odom_sub], queue_size=200, slop=0.25)
        self.sync.registerCallback(self.evaluate)
        self.get_logger().info(
            f'Visible-cloud evaluator online. voxel={self.voxel_size:.3f} m, '
            f'F-score threshold={self.fscore_threshold:.3f} m')

    @staticmethod
    def points(msg):
        return point_cloud2.read_points_numpy(
            msg, field_names=['x', 'y', 'z'], skip_nans=True).astype(np.float64)

    def evaluate(self, vggt_msg, ground_truth_msg, odom_msg):
        if ground_truth_msg.header.frame_id != odom_msg.header.frame_id:
            self.get_logger().error(
                f'Cannot compare frames {ground_truth_msg.header.frame_id!r} '
                f'and {odom_msg.header.frame_id!r}.')
            return
        pose = odom_msg.pose.pose
        world_to_waist = transform_matrix(
            [pose.position.x, pose.position.y, pose.position.z],
            [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w],
        )
        world_to_exo = world_to_waist @ self.waist_to_exo
        estimated = self.points(vggt_msg)
        estimated = estimated[np.isfinite(estimated).all(axis=1)]
        estimated = estimated @ world_to_exo[:3, :3].T + world_to_exo[:3, 3]
        estimated = voxel_downsample(estimated, self.voxel_size)
        ground_truth = voxel_downsample(self.points(ground_truth_msg), self.voxel_size)
        if len(estimated) == 0 or len(ground_truth) == 0:
            self.get_logger().warn('Skipping empty VGGT or ground-truth visible cloud.')
            return
        metrics = cloud_metrics(estimated, ground_truth, self.fscore_threshold)
        self.write_metrics(ground_truth_msg, len(estimated), len(ground_truth), metrics)
        self.get_logger().info(
            f'Visible cloud evaluation: Chamfer={metrics["chamfer"]:.3f} m, '
            f'F-score={metrics["fscore"]:.3f}, points={len(estimated)}/{len(ground_truth)}')

    def write_metrics(self, msg, estimated_points, ground_truth_points, metrics):
        dirname = os.path.dirname(self.metrics_csv_path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        exists = os.path.exists(self.metrics_csv_path) and os.path.getsize(self.metrics_csv_path) > 0
        row = {
            'timestamp': msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
            'voxel_size': self.voxel_size,
            'fscore_threshold': self.fscore_threshold,
            'estimated_points': estimated_points,
            'ground_truth_points': ground_truth_points,
            **metrics,
        }
        with open(self.metrics_csv_path, 'a', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=row.keys())
            if not exists:
                writer.writeheader()
            writer.writerow(row)


def main(args=None):
    rclpy.init(args=args)
    node = CloudMapEvaluatorNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if rclpy.ok():
            node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
