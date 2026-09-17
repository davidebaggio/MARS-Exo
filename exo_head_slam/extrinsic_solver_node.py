import rclpy
import time
from collections import deque
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from geometry_msgs.msg import TransformStamped
from tf2_msgs.msg import TFMessage
import tf2_ros
import message_filters
from cv_bridge import CvBridge
import numpy as np
import csv
import os
from typing import Optional
from scipy.spatial.transform import Rotation
from .utils.math_utils import compute_transform_ransac
from .utils.vision_utils import get_3d_point

from .utils.matchers import ORBMatcher, LightGlueMatcher


def transform_matrix(transform):
    value = transform.transform if hasattr(transform, 'transform') else transform
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_quat([
        value.rotation.x, value.rotation.y, value.rotation.z, value.rotation.w,
    ]).as_matrix()
    matrix[:3, 3] = [
        value.translation.x, value.translation.y, value.translation.z,
    ]
    return matrix


def depth_cloud_points(depth, intrinsics, stride=2):
    """Unproject finite metric depth while retaining original pixel coordinates."""
    rows = np.arange(0, depth.shape[0], stride)
    columns = np.arange(0, depth.shape[1], stride)
    u, v = np.meshgrid(columns, rows)
    z = depth[::stride, ::stride]
    valid = np.isfinite(z) & (z > 0.1) & (z < 10.0)
    z = z[valid]
    return np.column_stack([
        (u[valid] - intrinsics[0, 2]) * z / intrinsics[0, 0],
        (v[valid] - intrinsics[1, 2]) * z / intrinsics[1, 1],
        z,
    ])


def apply_transform(points, output_from_input):
    return points @ output_from_input[:3, :3].T + output_from_input[:3, 3]


def combined_cloud_points(
    head_depth, exo_depth, head_intrinsics, exo_intrinsics,
    head_link_from_optical, exo_link_from_optical, exo_link_from_head_link,
    stride=2,
):
    head = depth_cloud_points(head_depth, head_intrinsics, stride)
    exo = depth_cloud_points(exo_depth, exo_intrinsics, stride)
    head = apply_transform(
        apply_transform(head, head_link_from_optical), exo_link_from_head_link
    )
    exo = apply_transform(exo, exo_link_from_optical)
    return np.vstack([head, exo])


class ExtrinsicSolverNode(Node):
    def __init__(self):
        super().__init__('extrinsic_solver_node')
        
        self.declare_parameter('head_rgb_topic', '')
        self.declare_parameter('head_depth_topic', '')
        self.declare_parameter('exo_rgb_topic', '')
        self.declare_parameter('exo_depth_topic', '')
        self.declare_parameter('head_camera_info_topic', '')
        self.declare_parameter('exo_camera_info_topic', '')
        self.declare_parameter('head_frame_id', '')
        self.declare_parameter('exo_frame_id', '')
        self.declare_parameter('min_3d_matches', 0)
        self.declare_parameter('matcher_type', 'orb')
        self.declare_parameter('lightglue_device', 'cpu')
        self.declare_parameter('lightglue_max_keypoints', 2048)
        self.declare_parameter('tf_filter_alpha', 0.1)
        
        # New parameters for Section 3 fixes
        self.declare_parameter('min_solver_interval', 0.2)
        self.declare_parameter('ransac_threshold', 0.05)
        self.declare_parameter('ransac_iterations', 100)
        self.declare_parameter('ransac_seed', 0)
        self.declare_parameter('max_ransac_rmse', 0.05)
        self.declare_parameter('min_inlier_ratio', 0.3)
        self.declare_parameter('max_trans_jump', 0.3)
        self.declare_parameter('max_rot_jump', 0.5)
        self.declare_parameter('validate_rig_geometry', True)
        self.declare_parameter('connect_uninitialized_tf', True)
        self.declare_parameter('combined_cloud_topic', '/lightglue/combined_pointcloud')
        self.declare_parameter('combined_cloud_stride', 2)
        
        # Evaluation metrics parameters
        self.declare_parameter('metrics_enabled', False)
        self.declare_parameter('metrics_csv_path', 'extrinsic_metrics.csv')
        self.declare_parameter('gt_parent_frame', '')
        self.declare_parameter('gt_child_frame', '')
        self.declare_parameter('gt_tf_topic', '')

        self.bridge = CvBridge()
        self.matcher = self.create_matcher()
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.gt_transforms = deque(maxlen=10_000)
        gt_tf_topic = str(self.get_parameter('gt_tf_topic').value)
        self.gt_tf_sub = (
            self.create_subscription(TFMessage, gt_tf_topic, self._ingest_gt_tf, 10_000)
            if gt_tf_topic else None
        )
        
        # Filtering state
        self.tf_filter_alpha = float(self.get_parameter('tf_filter_alpha').value)
        self.current_t: Optional[np.ndarray] = None
        self.current_q: Optional[np.ndarray] = None
        
        # Section 3 fixes configuration & state
        self.min_solver_interval = float(self.get_parameter('min_solver_interval').value)
        self.ransac_threshold = float(self.get_parameter('ransac_threshold').value)
        self.ransac_iterations = int(self.get_parameter('ransac_iterations').value)
        self.rng = np.random.default_rng(
            int(self.get_parameter('ransac_seed').value)
        )
        self.max_ransac_rmse = float(self.get_parameter('max_ransac_rmse').value)
        self.min_inlier_ratio = float(self.get_parameter('min_inlier_ratio').value)
        self.max_trans_jump = float(self.get_parameter('max_trans_jump').value)
        self.max_rot_jump = float(self.get_parameter('max_rot_jump').value)
        self.validate_rig_geometry = bool(
            self.get_parameter('validate_rig_geometry').value
        )
        self.connect_uninitialized_tf = bool(self.get_parameter('connect_uninitialized_tf').value)
        self.combined_cloud_stride = int(
            self.get_parameter('combined_cloud_stride').value
        )
        self.combined_cloud_pub = self.create_publisher(
            PointCloud2, self.get_parameter('combined_cloud_topic').value, 10
        )
        self.last_solver_time: Optional[float] = None
        
        # Metrics configuration
        self.metrics_enabled = bool(self.get_parameter('metrics_enabled').value)
        self.metrics_csv_path = str(self.get_parameter('metrics_csv_path').value)
        self.gt_parent_frame = str(self.get_parameter('gt_parent_frame').value)
        self.gt_child_frame = str(self.get_parameter('gt_child_frame').value)
        
        # ponytail: initialize metrics file overwriting on start to have clean data per run
        if self.metrics_enabled:
            try:
                dirname = os.path.dirname(self.metrics_csv_path)
                if dirname:
                    os.makedirs(dirname, exist_ok=True)
                with open(self.metrics_csv_path, mode='w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        'timestamp', 'num_2d_matches', 'num_3d_matches', 'inliers', 'inlier_ratio', 'rmse', 'status',
                        't_x', 't_y', 't_z', 'q_x', 'q_y', 'q_z', 'q_w',
                        'gt_t_x', 'gt_t_y', 'gt_t_z', 'gt_q_x', 'gt_q_y', 'gt_q_z', 'gt_q_w',
                        'error_t', 'error_r_deg', 'tf_fresh', 'cycle_time_sec'
                    ])
                self.get_logger().info(f"Logging metrics to {os.path.abspath(self.metrics_csv_path)}")
            except Exception as e:
                self.get_logger().error(f"Failed to initialize metrics CSV: {str(e)}")
                self.metrics_enabled = False
        
        # Intrinsics storage
        self.head_k: Optional[np.ndarray] = None
        self.exo_k: Optional[np.ndarray] = None
        self.head_frame_id = self.get_parameter('head_frame_id').value
        self.exo_frame_id = self.get_parameter('exo_frame_id').value
        self.min_3d_matches = int(self.get_parameter('min_3d_matches').value)
        self.head_camera_info_topic = self.get_parameter('head_camera_info_topic').value
        self.exo_camera_info_topic = self.get_parameter('exo_camera_info_topic').value
        self.head_rgb_topic = self.get_parameter('head_rgb_topic').value
        self.head_depth_topic = self.get_parameter('head_depth_topic').value
        self.exo_rgb_topic = self.get_parameter('exo_rgb_topic').value
        self.exo_depth_topic = self.get_parameter('exo_depth_topic').value
        
        # Subscriptions
        self.head_info_sub = self.create_subscription(
            CameraInfo, self.head_camera_info_topic, self.head_info_cb,
            qos_profile_sensor_data,
        )
        self.exo_info_sub = self.create_subscription(
            CameraInfo, self.exo_camera_info_topic, self.exo_info_cb,
            qos_profile_sensor_data,
        )
        # Subscribers
        self.head_rgb_sub = message_filters.Subscriber(
            self, Image, self.head_rgb_topic, qos_profile=qos_profile_sensor_data)
        self.head_depth_sub = message_filters.Subscriber(
            self, Image, self.head_depth_topic, qos_profile=qos_profile_sensor_data)
        self.exo_rgb_sub = message_filters.Subscriber(
            self, Image, self.exo_rgb_topic, qos_profile=qos_profile_sensor_data)
        self.exo_depth_sub = message_filters.Subscriber(
            self, Image, self.exo_depth_topic, qos_profile=qos_profile_sensor_data)
        
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.head_rgb_sub, self.head_depth_sub, self.exo_rgb_sub, self.exo_depth_sub],
            queue_size=10,
            slop=0.05
        )
        self.ts.registerCallback(self.solve_callback)
        
        self.get_logger().info(f"Extrinsic Solver Node ({self.matcher.name}) initialized.")

    def create_matcher(self):
        matcher_type = str(self.get_parameter('matcher_type').value).strip().lower()
        if matcher_type == 'lightglue':
            lightglue_matcher = LightGlueMatcher(
                device=str(self.get_parameter('lightglue_device').value),
                max_keypoints=int(self.get_parameter('lightglue_max_keypoints').value),
            )
            if lightglue_matcher.available:
                return lightglue_matcher

            raise RuntimeError(f'LightGlue unavailable: {lightglue_matcher.error}')

        if matcher_type not in ('orb', 'lightglue'):
            raise ValueError(f"Unknown matcher_type: {matcher_type}")

        return ORBMatcher()

    def head_info_cb(self, msg: CameraInfo):
        self.head_k = np.array(msg.k).reshape(3, 3)

    def exo_info_cb(self, msg: CameraInfo):
        self.exo_k = np.array(msg.k).reshape(3, 3)

    def solve_callback(self, h_rgb: Image, h_depth: Image, e_rgb: Image, e_depth: Image):
        if not rclpy.ok():
            return
        stamp_sec = h_rgb.header.stamp.sec + h_rgb.header.stamp.nanosec * 1e-9

        run_solver = (
            self.last_solver_time is None
            or stamp_sec < self.last_solver_time
            or stamp_sec - self.last_solver_time >= self.min_solver_interval
        )
        if run_solver:
            self.last_solver_time = stamp_sec

        if run_solver:
            cycle_start = time.perf_counter()
            self.get_logger().info("Solver received synced image quad. Processing extrinsic estimation...")
            num_2d = 0
            num_3d = 0
            inliers = 0
            ratio = 0.0
            rmse = 0.0
            status = 'UNKNOWN'
            
            try:
                if self.head_k is None or self.exo_k is None:
                    self.get_logger().warn("Waiting for CameraInfo...")
                    status = 'WAITING_FOR_CAMERA_INFO'
                else:
                    head_img = self.bridge.imgmsg_to_cv2(h_rgb, 'bgr8')
                    head_dep = self.bridge.imgmsg_to_cv2(h_depth, 'passthrough')
                    exo_img = self.bridge.imgmsg_to_cv2(e_rgb, 'bgr8')
                    exo_dep = self.bridge.imgmsg_to_cv2(e_depth, 'passthrough')

                    # Match
                    pts_head, pts_exo = self.matcher.match(head_img, exo_img)
                    num_2d = len(pts_head)

                    if len(pts_head) == 0 or len(pts_exo) == 0:
                        self.get_logger().warn("Nessun match 2D valido trovato.")
                        status = 'NO_2D_MATCHES'
                    else:
                        # Deproject
                        points_3d_head = []
                        points_3d_exo = []
                        
                        # Get optical -> link transforms
                        try:
                            t_h_link_opt = self.tf_buffer.lookup_transform(self.head_frame_id, h_rgb.header.frame_id, h_rgb.header.stamp, timeout=rclpy.duration.Duration(seconds=0.1))
                            t_e_link_opt = self.tf_buffer.lookup_transform(self.exo_frame_id, e_rgb.header.frame_id, e_rgb.header.stamp, timeout=rclpy.duration.Duration(seconds=0.1))
                            
                            T_h_link_opt = transform_matrix(t_h_link_opt)
                            T_e_link_opt = transform_matrix(t_e_link_opt)
                        except Exception as e:
                            self.get_logger().warn(f"Waiting for link-to-optical TFs: {str(e)}")
                            status = 'WAITING_FOR_LINK_TF'
                            self.log_metrics(
                                stamp_sec, num_2d, 0, 0, 0.0, 0.0, status,
                                time.perf_counter() - cycle_start,
                            )
                            if self.current_t is not None:
                                self.broadcast_transform(e_rgb.header.stamp)
                            elif self.connect_uninitialized_tf:
                                self.broadcast_startup_transform(e_rgb.header.stamp)
                            return

                        for p_h, p_e in zip(pts_head, pts_exo):
                            p3_h_opt = get_3d_point(int(p_h[0]), int(p_h[1]), head_dep, self.head_k)
                            p3_e_opt = get_3d_point(int(p_e[0]), int(p_e[1]), exo_dep, self.exo_k)
                            
                            if p3_h_opt is not None and p3_e_opt is not None:
                                # Transform to link frames
                                p3_h_link = (T_h_link_opt[:3, :3] @ p3_h_opt) + T_h_link_opt[:3, 3]
                                p3_e_link = (T_e_link_opt[:3, :3] @ p3_e_opt) + T_e_link_opt[:3, 3]
                                points_3d_head.append(p3_h_link)
                                points_3d_exo.append(p3_e_link)
                        
                        points_3d_head = np.array(points_3d_head)
                        points_3d_exo = np.array(points_3d_exo)
                        num_3d = len(points_3d_head)

                        if num_3d < self.min_3d_matches:
                            self.get_logger().warn(f"Insufficient 3D matches ({num_3d}/{self.min_3d_matches}). Skipping.")
                            status = 'INSUFFICIENT_3D_MATCHES'
                        else:
                            # Solve for HeadLink in ExoLink frame (T_exo_head)
                            T, inliers, rmse = compute_transform_ransac(
                                points_3d_head, points_3d_exo,
                                self.ransac_threshold, self.ransac_iterations,
                                self.rng,
                            )
                            
                            if T is None:
                                self.get_logger().warn(f"RANSAC failed to find a valid transform for {num_3d} matches.")
                                status = 'RANSAC_FAILED'
                            else:
                                ratio = inliers / num_3d
                                
                                # Confidence checks
                                if not (rmse <= self.max_ransac_rmse and ratio >= self.min_inlier_ratio and inliers >= self.min_3d_matches):
                                    self.get_logger().warn(
                                        f"Rejecting transform due to confidence checks: "
                                        f"RMSE={rmse:.3f} (max={self.max_ransac_rmse}), "
                                        f"ratio={ratio:.2f} (min={self.min_inlier_ratio}), "
                                        f"inliers={inliers} (min={self.min_3d_matches})"
                                    )
                                    status = 'REJECTED_CONFIDENCE_LIMITS'
                                else:
                                    new_t = T[:3, 3]
                                    new_q = Rotation.from_matrix(T[:3, :3]).as_quat()
                                    
                                    # Physical bounds check
                                    dist = np.linalg.norm(new_t)
                                    valid = True
                                    if self.validate_rig_geometry and not (0.2 <= dist <= 2.2):
                                        self.get_logger().warn(f"Rejecting transform: distance {dist:.2f} m out of bounds [0.2, 2.2]")
                                        status = 'REJECTED_DISTANCE_BOUNDS'
                                        valid = False
                                    elif self.validate_rig_geometry and not (0.1 <= new_t[2] <= 1.5):
                                        self.get_logger().warn(f"Rejecting transform: relative Z height {new_t[2]:.2f} m out of bounds [0.1, 1.5]")
                                        status = 'REJECTED_HEIGHT_BOUNDS'
                                        valid = False
                                    
                                    # Geometric consistency checks (jump limits)
                                    if valid and self.current_t is not None and self.max_trans_jump > 0:
                                        trans_jump = np.linalg.norm(new_t - self.current_t)
                                        if trans_jump > self.max_trans_jump:
                                            self.get_logger().warn(
                                                f"Rejecting transform due to translation jump: {trans_jump:.3f} m > {self.max_trans_jump} m"
                                            )
                                            status = 'REJECTED_TRANS_JUMP'
                                            valid = False
                                    
                                    if valid and self.current_q is not None and self.max_rot_jump > 0:
                                        dot_product = abs(np.dot(self.current_q, new_q))
                                        dot_product = min(1.0, max(0.0, dot_product))
                                        rot_jump = 2.0 * np.arccos(dot_product)
                                        if rot_jump > self.max_rot_jump:
                                            self.get_logger().warn(
                                                f"Rejecting transform due to rotation jump: {rot_jump:.3f} rad > {self.max_rot_jump} rad"
                                            )
                                            status = 'REJECTED_ROT_JUMP'
                                            valid = False

                                    if valid:
                                        status = 'SUCCESS'
                                        # Update EMA-filtered state
                                        if self.current_t is None:
                                            self.current_t = new_t
                                            self.current_q = new_q
                                            self.get_logger().info(
                                                f"Initialized extrinsic: t=[{self.current_t[0]:.3f}, {self.current_t[1]:.3f}, {self.current_t[2]:.3f}] "
                                                f"with {inliers} inliers (RMSE={rmse:.3f} m)"
                                            )
                                        else:
                                            self.current_t = (1.0 - self.tf_filter_alpha) * self.current_t + self.tf_filter_alpha * new_t
                                            
                                            if np.dot(self.current_q, new_q) < 0:
                                                new_q = -new_q
                                            self.current_q = (1.0 - self.tf_filter_alpha) * self.current_q + self.tf_filter_alpha * new_q
                                            self.current_q /= np.linalg.norm(self.current_q)
                                            self.get_logger().info(
                                                f"Updated extrinsic: t=[{self.current_t[0]:.3f}, {self.current_t[1]:.3f}, {self.current_t[2]:.3f}] "
                                                f"with {inliers} inliers (RMSE={rmse:.3f} m)"
                                            )
                                        
                                        current_transform = np.eye(4)
                                        current_transform[:3, :3] = Rotation.from_quat(
                                            self.current_q
                                        ).as_matrix()
                                        current_transform[:3, 3] = self.current_t
                                        self.publish_combined_cloud(
                                            head_dep, exo_dep,
                                            T_h_link_opt, T_e_link_opt,
                                            current_transform, e_rgb.header.stamp,
                                        )
            except Exception as e:
                if not rclpy.ok():
                    return
                self.get_logger().error(f"Solver callback failed: {str(e)}")
                status = f"ERROR_{type(e).__name__}"

            if not rclpy.ok():
                return
            self.log_metrics(
                stamp_sec, num_2d, num_3d, inliers, ratio, rmse, status,
                time.perf_counter() - cycle_start,
            )

        # ponytail: simple TF fallback - broadcast the last known good filtered transform 
        # with the current timestamp to keep the TF tree active even when solver fails or is throttled.
        if self.current_t is not None:
            self.broadcast_transform(e_rgb.header.stamp)
        elif self.connect_uninitialized_tf:
            self.broadcast_startup_transform(e_rgb.header.stamp)

    def _ingest_gt_tf(self, message):
        self.gt_transforms.extend(message.transforms)

    def publish_combined_cloud(
        self, head_depth, exo_depth, head_link_from_optical,
        exo_link_from_optical, exo_link_from_head_link, stamp,
    ):
        if not rclpy.ok():
            return
        points = combined_cloud_points(
            head_depth, exo_depth, self.head_k, self.exo_k,
            head_link_from_optical, exo_link_from_optical,
            exo_link_from_head_link, self.combined_cloud_stride,
        ).astype(np.float32)
        if not len(points):
            return
        message = PointCloud2()
        message.header.stamp = stamp
        message.header.frame_id = self.exo_frame_id
        message.height = 1
        message.width = len(points)
        message.is_dense = True
        message.is_bigendian = False
        message.fields = [
            PointField(name=name, offset=offset, datatype=PointField.FLOAT32, count=1)
            for name, offset in (('x', 0), ('y', 4), ('z', 8))
        ]
        message.point_step = 12
        message.row_step = 12 * len(points)
        message.data = points.tobytes()
        try:
            self.combined_cloud_pub.publish(message)
        except Exception:
            if rclpy.ok():
                raise

    def broadcast_transform(self, stamp):
        if not rclpy.ok() or self.current_t is None or self.current_q is None:
            return
            
        t_msg = TransformStamped()
        t_msg.header.stamp = stamp
        t_msg.header.frame_id = self.exo_frame_id
        t_msg.child_frame_id = self.head_frame_id
        
        t_msg.transform.translation.x = float(self.current_t[0])
        t_msg.transform.translation.y = float(self.current_t[1])
        t_msg.transform.translation.z = float(self.current_t[2])
        
        t_msg.transform.rotation.x = float(self.current_q[0])
        t_msg.transform.rotation.y = float(self.current_q[1])
        t_msg.transform.rotation.z = float(self.current_q[2])
        t_msg.transform.rotation.w = float(self.current_q[3])
        
        try:
            self.tf_broadcaster.sendTransform(t_msg)
        except Exception:
            if rclpy.ok():
                raise

    def broadcast_startup_transform(self, stamp):
        if not rclpy.ok():
            return
        # ponytail: temporary identity keeps RViz/debug clouds connected until the solver has a real extrinsic.
        t_msg = TransformStamped()
        t_msg.header.stamp = stamp
        t_msg.header.frame_id = self.exo_frame_id
        t_msg.child_frame_id = self.head_frame_id
        t_msg.transform.rotation.w = 1.0
        try:
            self.tf_broadcaster.sendTransform(t_msg)
        except Exception:
            if rclpy.ok():
                raise

    def log_metrics(
        self, timestamp: float, num_2d: int, num_3d: int, inliers: int,
        ratio: float, rmse: float, status: str, cycle_time_sec: float,
    ):
        if not self.metrics_enabled:
            return
            
        gt_t = [np.nan, np.nan, np.nan]
        gt_q = [np.nan, np.nan, np.nan, np.nan]
        error_t = np.nan
        error_r_deg = np.nan
        
        if (
            status == 'SUCCESS'
            and self.gt_parent_frame
            and self.gt_child_frame
        ):
            try:
                candidates = [
                    transform for transform in self.gt_transforms
                    if transform.header.frame_id == self.gt_parent_frame
                    and transform.child_frame_id == self.gt_child_frame
                ]
                if candidates:
                    gt_tf = min(candidates, key=lambda transform: abs(
                        transform.header.stamp.sec
                        + transform.header.stamp.nanosec * 1e-9 - timestamp
                    ))
                    gt_time = (
                        gt_tf.header.stamp.sec
                        + gt_tf.header.stamp.nanosec * 1e-9
                    )
                    if abs(gt_time - timestamp) > 0.05:
                        raise ValueError('No ground-truth transform within 0.05 s')
                else:
                    gt_tf = self.tf_buffer.lookup_transform(
                        self.gt_parent_frame, self.gt_child_frame,
                        rclpy.time.Time(),
                    )
                gt_t = [
                    gt_tf.transform.translation.x,
                    gt_tf.transform.translation.y,
                    gt_tf.transform.translation.z
                ]
                gt_q = [
                    gt_tf.transform.rotation.x,
                    gt_tf.transform.rotation.y,
                    gt_tf.transform.rotation.z,
                    gt_tf.transform.rotation.w
                ]
                
                if self.current_t is not None and self.current_q is not None:
                    error_t = float(np.linalg.norm(np.array(self.current_t) - np.array(gt_t)))
                    dot_product = abs(np.dot(self.current_q, np.array(gt_q)))
                    dot_product = min(1.0, max(0.0, dot_product))
                    error_r_deg = float(np.degrees(2.0 * np.arccos(dot_product)))
            except Exception as error:
                self.get_logger().warn(f'Ground-truth lookup failed: {error}')
                
        t_x, t_y, t_z = (self.current_t[0], self.current_t[1], self.current_t[2]) if self.current_t is not None else (np.nan, np.nan, np.nan)
        q_x, q_y, q_z, q_w = (self.current_q[0], self.current_q[1], self.current_q[2], self.current_q[3]) if self.current_q is not None else (np.nan, np.nan, np.nan, np.nan)
        
        try:
            with open(self.metrics_csv_path, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    timestamp, num_2d, num_3d, inliers, ratio, rmse, status,
                    t_x, t_y, t_z, q_x, q_y, q_z, q_w,
                    gt_t[0], gt_t[1], gt_t[2], gt_q[0], gt_q[1], gt_q[2], gt_q[3],
                    error_t, error_r_deg, int(status == 'SUCCESS'),
                    cycle_time_sec,
                ])
        except Exception as e:
            self.get_logger().error(f"Failed to write metrics row: {str(e)}")

def main(args=None):
    rclpy.init(args=args)
    node = ExtrinsicSolverNode()
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
