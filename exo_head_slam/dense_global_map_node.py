from __future__ import annotations

import numpy as np
import message_filters
import rclpy
from cv_bridge import CvBridge
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField
from std_msgs.msg import UInt32
from tf2_ros import Buffer, TransformListener


def pack_rgb(rgb: np.ndarray) -> np.ndarray:
    return (
        (rgb[:, 0].astype(np.uint32) << 16)
        | (rgb[:, 1].astype(np.uint32) << 8)
        | rgb[:, 2].astype(np.uint32)
    )


def voxel_downsample_cloud(points: np.ndarray, colors: np.ndarray, voxel_size: float, max_points: int):
    if len(points) == 0:
        return points, colors

    if voxel_size > 0.0:
        voxel_keys = np.floor(points / voxel_size).astype(np.int64)
        _, keep_idx = np.unique(voxel_keys, axis=0, return_index=True)
        keep_idx = np.sort(keep_idx)
        points = points[keep_idx]
        colors = colors[keep_idx]

    if max_points > 0 and len(points) > max_points:
        keep_idx = np.linspace(0, len(points) - 1, max_points, dtype=np.int64)
        points = points[keep_idx]
        colors = colors[keep_idx]

    return points, colors


def pointcloud2_from_xyzrgb(points: np.ndarray, colors: np.ndarray, frame_id: str, stamp) -> PointCloud2:
    data = np.zeros(len(points), dtype=[
        ('x', np.float32),
        ('y', np.float32),
        ('z', np.float32),
        ('rgb', np.uint32),
    ])
    data['x'] = points[:, 0]
    data['y'] = points[:, 1]
    data['z'] = points[:, 2]
    data['rgb'] = pack_rgb(colors)

    pcl_msg = PointCloud2()
    pcl_msg.header.frame_id = frame_id
    pcl_msg.header.stamp = stamp
    pcl_msg.height = 1
    pcl_msg.width = len(points)
    pcl_msg.is_dense = False
    pcl_msg.is_bigendian = False
    pcl_msg.fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1),
    ]
    pcl_msg.point_step = 16
    pcl_msg.row_step = pcl_msg.point_step * len(points)
    pcl_msg.data = data.tobytes()
    return pcl_msg


def transform_to_matrix(transform) -> np.ndarray:
    q = transform.transform.rotation
    t = transform.transform.translation
    x, y, z, w = q.x, q.y, q.z, q.w
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    matrix = np.eye(4, dtype=np.float32)
    matrix[0, 0] = 1.0 - 2.0 * (yy + zz)
    matrix[0, 1] = 2.0 * (xy - wz)
    matrix[0, 2] = 2.0 * (xz + wy)
    matrix[1, 0] = 2.0 * (xy + wz)
    matrix[1, 1] = 1.0 - 2.0 * (xx + zz)
    matrix[1, 2] = 2.0 * (yz - wx)
    matrix[2, 0] = 2.0 * (xz - wy)
    matrix[2, 1] = 2.0 * (yz + wx)
    matrix[2, 2] = 1.0 - 2.0 * (xx + yy)
    matrix[0, 3] = t.x
    matrix[1, 3] = t.y
    matrix[2, 3] = t.z
    return matrix


def project_rgbd(rgb: np.ndarray, depth: np.ndarray, camera_info: CameraInfo, downsample_factor: int,
                 min_depth: float, max_depth: float):
    if rgb.ndim != 3 or rgb.shape[2] != 3 or depth.ndim != 2 or rgb.shape[:2] != depth.shape:
        raise ValueError('RGB and depth dimensions must match')
    if (camera_info.width and camera_info.width != depth.shape[1]) or (
            camera_info.height and camera_info.height != depth.shape[0]):
        raise ValueError('CameraInfo dimensions do not match RGB-D images')
    if downsample_factor < 1:
        raise ValueError('downsample_factor must be at least 1')

    K = np.array(camera_info.k, dtype=np.float32).reshape(3, 3)
    if not np.all(np.isfinite(K)) or K[0, 0] <= 0.0 or K[1, 1] <= 0.0:
        raise ValueError('CameraInfo focal lengths must be finite and positive')

    if downsample_factor > 1:
        rgb = rgb[::downsample_factor, ::downsample_factor]
        depth = depth[::downsample_factor, ::downsample_factor]
        K[0, 0] /= downsample_factor
        K[1, 1] /= downsample_factor
        K[0, 2] /= downsample_factor
        K[1, 2] /= downsample_factor

    if depth.dtype != np.float32:
        depth = depth.astype(np.float32)

    valid = np.isfinite(depth) & (depth >= min_depth) & (depth <= max_depth)
    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)

    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    z = depth[valid]
    x = (u[valid] - K[0, 2]) * z / K[0, 0]
    y = (v[valid] - K[1, 2]) * z / K[1, 1]
    points = np.stack([x, y, z], axis=1).astype(np.float32)
    colors = rgb[valid].astype(np.uint8)
    return points, colors


def clear_on_epoch(current_epoch, new_epoch, points, colors):
    if current_epoch is None or current_epoch == new_epoch:
        return new_epoch, points, colors
    return (
        new_epoch,
        np.empty((0, 3), dtype=np.float32),
        np.empty((0, 3), dtype=np.uint8),
    )


class DenseGlobalMapNode(Node):
    def __init__(self):
        super().__init__('dense_global_map_accumulator')

        self.declare_parameter('rgb_topic', '/exo/masked/image_raw')
        self.declare_parameter('depth_topic', '/exo/masked/depth_raw')
        self.declare_parameter('camera_info_topic', '/camera/exo/color/camera_info')
        self.declare_parameter('map_frame_id', 'map')
        self.declare_parameter('voxel_size', 0.03)
        self.declare_parameter('max_points', 250000)
        self.declare_parameter('downsample_factor', 2)
        self.declare_parameter('min_depth', 0.1)
        self.declare_parameter('max_depth', 10.0)
        self.declare_parameter('lookup_timeout_sec', 0.1)
        self.declare_parameter('warn_interval_sec', 5.0)

        self.rgb_topic = self.get_parameter('rgb_topic').value
        self.depth_topic = self.get_parameter('depth_topic').value
        self.camera_info_topic = self.get_parameter('camera_info_topic').value
        self.map_frame_id = self.get_parameter('map_frame_id').value
        self.voxel_size = float(self.get_parameter('voxel_size').value)
        self.max_points = int(self.get_parameter('max_points').value)
        self.downsample_factor = int(self.get_parameter('downsample_factor').value)
        self.min_depth = float(self.get_parameter('min_depth').value)
        self.max_depth = float(self.get_parameter('max_depth').value)
        self.lookup_timeout_sec = float(self.get_parameter('lookup_timeout_sec').value)
        self.warn_interval_sec = float(self.get_parameter('warn_interval_sec').value)

        self.bridge = CvBridge()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=True)

        self.rgb_sub = message_filters.Subscriber(
            self, Image, self.rgb_topic, qos_profile=qos_profile_sensor_data
        )
        self.depth_sub = message_filters.Subscriber(
            self, Image, self.depth_topic, qos_profile=qos_profile_sensor_data
        )
        self.info_sub = message_filters.Subscriber(
            self, CameraInfo, self.camera_info_topic, qos_profile=qos_profile_sensor_data
        )
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub, self.info_sub],
            queue_size=20,
            slop=0.05,
        )
        self.sync.registerCallback(self.callback)

        cloud_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.cloud_pub = self.create_publisher(PointCloud2, '/orbslam/cloud_map', cloud_qos)
        self.epoch_sub = self.create_subscription(
            UInt32, '/orbslam/map_epoch', self.epoch_callback, cloud_qos
        )
        self._points = np.empty((0, 3), dtype=np.float32)
        self._colors = np.empty((0, 3), dtype=np.uint8)
        self._epoch = None
        self._pending_frame = None
        self._last_warn_sec = 0.0
        self.retry_timer = self.create_timer(0.02, self.retry_pending_frame)

        self.get_logger().info('Dense global map accumulator online')

    def callback(self, rgb_msg: Image, depth_msg: Image, info_msg: CameraInfo):
        transform_time = Time.from_msg(depth_msg.header.stamp)
        if not self.tf_buffer.can_transform(
                self.map_frame_id, depth_msg.header.frame_id, transform_time):
            self._pending_frame = (rgb_msg, depth_msg, info_msg)
            return
        self._pending_frame = None
        self.accumulate_frame(rgb_msg, depth_msg, info_msg, transform_time)

    def retry_pending_frame(self):
        if self._pending_frame is None:
            return
        rgb_msg, depth_msg, info_msg = self._pending_frame
        transform_time = Time.from_msg(depth_msg.header.stamp)
        if self.tf_buffer.can_transform(
                self.map_frame_id, depth_msg.header.frame_id, transform_time):
            self._pending_frame = None
            self.accumulate_frame(rgb_msg, depth_msg, info_msg, transform_time)

    def accumulate_frame(
            self, rgb_msg: Image, depth_msg: Image, info_msg: CameraInfo, transform_time: Time):
        try:
            rgb = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='rgb8')
            depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
        except Exception as exc:
            self.get_logger().warn(f'RGB-D conversion failed: {exc}')
            return

        try:
            tf = self.tf_buffer.lookup_transform(
                self.map_frame_id,
                depth_msg.header.frame_id,
                transform_time,
                timeout=Duration(seconds=self.lookup_timeout_sec),
            )
        except Exception as exc:
            self._warn_throttled(f'No TF {depth_msg.header.frame_id} -> {self.map_frame_id}: {exc}')
            return

        try:
            points_local, colors = project_rgbd(
                rgb, depth, info_msg, self.downsample_factor, self.min_depth, self.max_depth
            )
        except ValueError as exc:
            self._warn_throttled(f'Invalid RGB-D frame: {exc}')
            return
        if len(points_local) == 0:
            return

        t_map_cam = transform_to_matrix(tf)
        rot = t_map_cam[:3, :3]
        trans = t_map_cam[:3, 3]
        points_global = (points_local @ rot.T) + trans

        self._points = np.vstack([self._points, points_global.astype(np.float32)])
        self._colors = np.vstack([self._colors, colors.astype(np.uint8)])
        self._points, self._colors = voxel_downsample_cloud(
            self._points, self._colors, self.voxel_size, self.max_points
        )
        self.cloud_pub.publish(
            pointcloud2_from_xyzrgb(self._points, self._colors, self.map_frame_id, depth_msg.header.stamp)
        )

    def epoch_callback(self, msg: UInt32):
        old_epoch = self._epoch
        self._epoch, self._points, self._colors = clear_on_epoch(
            self._epoch, msg.data, self._points, self._colors
        )
        self._pending_frame = None
        if old_epoch is not None and old_epoch != self._epoch:
            self.get_logger().warn(f'Cleared dense map for ORB map epoch {self._epoch}')

    def _warn_throttled(self, msg: str):
        now_sec = self.get_clock().now().nanoseconds * 1e-9
        if now_sec - self._last_warn_sec < self.warn_interval_sec:
            return
        self._last_warn_sec = now_sec
        self.get_logger().warn(msg)


def main(args=None):
    rclpy.init(args=args)
    node = DenseGlobalMapNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
