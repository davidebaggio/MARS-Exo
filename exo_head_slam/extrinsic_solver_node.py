import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import TransformStamped, Vector3Stamped
from std_msgs.msg import String
import tf2_ros
import message_filters
from cv_bridge import CvBridge
import numpy as np
import cv2
import csv
import os
from typing import Optional
from scipy.spatial.transform import Rotation as Rotation
from .utils.math_utils import compute_transform_ransac
from .utils.vision_utils import get_3d_point
from .utils.matchers import ORBMatcher, LightGlueMatcher


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
        self.declare_parameter('min_3d_matches', 15)
        self.declare_parameter('matcher_type', 'orb')
        self.declare_parameter('lightglue_device', 'cpu')
        self.declare_parameter('lightglue_max_keypoints', 2048)
        self.declare_parameter('tf_filter_alpha', 0.1)
        self.declare_parameter('min_solver_interval', 0.2)
        self.declare_parameter('max_ransac_rmse', 0.05)
        self.declare_parameter('min_inlier_ratio', 0.3)
        self.declare_parameter('max_trans_jump', 0.3)
        self.declare_parameter('max_rot_jump', 0.5)
        self.declare_parameter('max_trans_jump_moving', 0.8)
        self.declare_parameter('max_rot_jump_moving', 1.0)

        self.declare_parameter('metrics_enabled', False)
        self.declare_parameter('metrics_csv_path', 'extrinsic_metrics.csv')
        self.declare_parameter('gt_parent_frame', '')
        self.declare_parameter('gt_child_frame', '')

        self.declare_parameter('imu_gravity_constraint.enabled', True)
        self.declare_parameter('imu_gravity_constraint.threshold_deg', 15.0)
        self.declare_parameter('head_gravity_topic', '/imu/head/gravity')
        self.declare_parameter('exo_gravity_topic', '/imu/exo/gravity')

        self.declare_parameter('imu_gyro_propagation.enabled', True)
        self.declare_parameter('imu_gyro_propagation.max_duration', 5.0)
        self.declare_parameter('head_gyro_topic', '/imu/head/gyro_filtered')
        self.declare_parameter('exo_gyro_topic', '/imu/exo/gyro_filtered')
        self.declare_parameter('head_motion_topic', '/imu/head/motion_state')
        self.declare_parameter('exo_motion_topic', '/imu/exo/motion_state')

        self.bridge = CvBridge()
        self.matcher = self.create_matcher()
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Re-broadcast last known transform at 10 Hz to prevent TF buffer expiry
        self.create_timer(0.1, self.gyro_broadcast_timer)

        self.tf_filter_alpha = float(self.get_parameter('tf_filter_alpha').value)
        # Initial guess: head ~30cm above exo. Keeps TF tree connected before first solve.
        self.current_t: np.ndarray = np.array([0.0, 0.0, 0.7])
        self.current_q: np.ndarray = np.array([0.0, 0.0, 0.0, 1.0])
        self._has_real_solve = False  # True after first successful visual solve

        self.min_solver_interval = float(self.get_parameter('min_solver_interval').value)
        self.max_ransac_rmse = float(self.get_parameter('max_ransac_rmse').value)
        self.min_inlier_ratio = float(self.get_parameter('min_inlier_ratio').value)
        self.max_trans_jump = float(self.get_parameter('max_trans_jump').value)
        self.max_rot_jump = float(self.get_parameter('max_rot_jump').value)
        self.max_trans_jump_moving = float(self.get_parameter('max_trans_jump_moving').value)
        self.max_rot_jump_moving = float(self.get_parameter('max_rot_jump_moving').value)
        self.last_solver_time: Optional[float] = None

        self.metrics_enabled = bool(self.get_parameter('metrics_enabled').value)
        self.metrics_csv_path = str(self.get_parameter('metrics_csv_path').value)
        self.gt_parent_frame = str(self.get_parameter('gt_parent_frame').value)
        self.gt_child_frame = str(self.get_parameter('gt_child_frame').value)

        if self.metrics_enabled:
            self._init_csv(self.metrics_csv_path, [
                'timestamp', 'num_2d_matches', 'num_3d_matches', 'inliers', 'inlier_ratio', 'rmse', 'status',
                't_x', 't_y', 't_z', 'q_x', 'q_y', 'q_z', 'q_w',
                'gt_t_x', 'gt_t_y', 'gt_t_z', 'gt_q_x', 'gt_q_y', 'gt_q_z', 'gt_q_w',
                'error_t', 'error_r_deg',
                'gravity_error_deg', 'is_stationary', 'imu_constraint_applied',
            ])

        # IMU gravity constraint
        self.imu_gravity_enabled = bool(self.get_parameter('imu_gravity_constraint.enabled').value)
        self.imu_gravity_threshold_deg = float(self.get_parameter('imu_gravity_constraint.threshold_deg').value)
        self.head_gravity: Optional[np.ndarray] = None
        self.exo_gravity: Optional[np.ndarray] = None
        self.head_gravity_frame: Optional[str] = None
        self.exo_gravity_frame: Optional[str] = None

        imu_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=10)

        if self.imu_gravity_enabled:
            self.head_gravity_sub = self.create_subscription(
                Vector3Stamped, self.get_parameter('head_gravity_topic').value,
                self.head_gravity_cb, imu_qos)
            self.exo_gravity_sub = self.create_subscription(
                Vector3Stamped, self.get_parameter('exo_gravity_topic').value,
                self.exo_gravity_cb, imu_qos)

        # IMU gyro propagation
        self.gyro_propagation_enabled = bool(self.get_parameter('imu_gyro_propagation.enabled').value)
        self.gyro_max_duration = float(self.get_parameter('imu_gyro_propagation.max_duration').value)
        self.gyro_T = np.eye(4)
        self.gyro_R = np.eye(3)
        self.gyro_propagation_start = None
        self.in_fallback = False
        self._head_gyro = None
        self._exo_gyro = None
        self._head_gyro_ts = None
        self._exo_gyro_ts = None
        self.head_motion_state = None
        self.exo_motion_state = None

        if self.gyro_propagation_enabled:
            self.head_gyro_sub = self.create_subscription(
                Vector3Stamped, self.get_parameter('head_gyro_topic').value,
                self.head_gyro_cb, imu_qos)
            self.exo_gyro_sub = self.create_subscription(
                Vector3Stamped, self.get_parameter('exo_gyro_topic').value,
                self.exo_gyro_cb, imu_qos)
            self.head_motion_sub = self.create_subscription(
                String, self.get_parameter('head_motion_topic').value,
                lambda msg: setattr(self, 'head_motion_state', msg.data), imu_qos)
            self.exo_motion_sub = self.create_subscription(
                String, self.get_parameter('exo_motion_topic').value,
                lambda msg: setattr(self, 'exo_motion_state', msg.data), imu_qos)

        # Intrinsics
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

        self.head_info_sub = self.create_subscription(CameraInfo, self.head_camera_info_topic, self.head_info_cb, 10)
        self.exo_info_sub = self.create_subscription(CameraInfo, self.exo_camera_info_topic, self.exo_info_cb, 10)
        self.head_rgb_sub = message_filters.Subscriber(self, Image, self.head_rgb_topic)
        self.head_depth_sub = message_filters.Subscriber(self, Image, self.head_depth_topic)
        self.exo_rgb_sub = message_filters.Subscriber(self, Image, self.exo_rgb_topic)
        self.exo_depth_sub = message_filters.Subscriber(self, Image, self.exo_depth_topic)

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.head_rgb_sub, self.head_depth_sub, self.exo_rgb_sub, self.exo_depth_sub],
            queue_size=2, slop=0.05)
        self.ts.registerCallback(self.solve_callback)

        self.get_logger().info(f"Extrinsic Solver Node ({self.matcher.name}) initialized.")
        if self.gyro_propagation_enabled:
            self.get_logger().info("  Gyro propagation enabled")

    def _init_csv(self, path, headers):
        try:
            with open(path, mode='w', newline='') as f:
                csv.writer(f).writerow(headers)
            self.get_logger().info(f"Logging to {os.path.abspath(path)}")
        except Exception as e:
            self.get_logger().error(f"Failed to init CSV {path}: {str(e)}")

    def create_matcher(self):
        matcher_type = str(self.get_parameter('matcher_type').value).strip().lower()
        if matcher_type == 'lightglue':
            lm = LightGlueMatcher(
                device=str(self.get_parameter('lightglue_device').value),
                max_keypoints=int(self.get_parameter('lightglue_max_keypoints').value),
            )
            if lm.available:
                return lm
            self.get_logger().warn(f"LightGlue unavailable, fallback ORB: {lm.error}")
        return ORBMatcher()

    def head_info_cb(self, msg: CameraInfo):
        self.head_k = np.array(msg.k).reshape(3, 3)

    def exo_info_cb(self, msg: CameraInfo):
        self.exo_k = np.array(msg.k).reshape(3, 3)

    def head_gravity_cb(self, msg: Vector3Stamped):
        self.head_gravity = np.array([msg.vector.x, msg.vector.y, msg.vector.z])
        self.head_gravity_frame = msg.header.frame_id

    def exo_gravity_cb(self, msg: Vector3Stamped):
        self.exo_gravity = np.array([msg.vector.x, msg.vector.y, msg.vector.z])
        self.exo_gravity_frame = msg.header.frame_id

    def head_gyro_cb(self, msg: Vector3Stamped):
        try:
            ω = np.array([msg.vector.x, msg.vector.y, msg.vector.z])
            ts = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            if self.in_fallback and self._head_gyro_ts is not None and self._exo_gyro is not None:
                dt = ts - self._head_gyro_ts
                if 0 < dt < 0.15:
                    ω_rel = ω - self.gyro_R @ self._exo_gyro
                    ΔR = Rotation.from_rotvec(ω_rel * dt).as_matrix()
                    self.gyro_T[:3, :3] = ΔR @ self.gyro_T[:3, :3]
                    # Orthogonalize rotation matrix to prevent numerical drift
                    U, _, Vt = np.linalg.svd(self.gyro_T[:3, :3])
                    self.gyro_T[:3, :3] = U @ Vt
                    self.gyro_R = self.gyro_T[:3, :3]
                    self.current_t = self.gyro_T[:3, 3].copy()
                    self.current_q = Rotation.from_matrix(self.gyro_T[:3, :3]).as_quat()
                    self.broadcast_transform(msg.header.stamp)
            self._head_gyro = ω
            self._head_gyro_ts = ts
        except Exception as e:
            self.get_logger().error(f"Error in head gyro callback: {str(e)}")

    def exo_gyro_cb(self, msg: Vector3Stamped):
        try:
            ω = np.array([msg.vector.x, msg.vector.y, msg.vector.z])
            ts = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            if self.in_fallback and self._exo_gyro_ts is not None and self._head_gyro is not None:
                dt = ts - self._exo_gyro_ts
                if 0 < dt < 0.15:
                    ω_rel = self._head_gyro - self.gyro_R @ ω
                    ΔR = Rotation.from_rotvec(ω_rel * dt).as_matrix()
                    self.gyro_T[:3, :3] = ΔR @ self.gyro_T[:3, :3]
                    # Orthogonalize rotation matrix to prevent numerical drift
                    U, _, Vt = np.linalg.svd(self.gyro_T[:3, :3])
                    self.gyro_T[:3, :3] = U @ Vt
                    self.gyro_R = self.gyro_T[:3, :3]
                    self.current_t = self.gyro_T[:3, 3].copy()
                    self.current_q = Rotation.from_matrix(self.gyro_T[:3, :3]).as_quat()
                    self.broadcast_transform(msg.header.stamp)
            self._exo_gyro = ω
            self._exo_gyro_ts = ts
        except Exception as e:
            self.get_logger().error(f"Error in exo gyro callback: {str(e)}")

    def solve_callback(self, h_rgb: Image, h_depth: Image, e_rgb: Image, e_depth: Image):
        if "head" not in h_rgb.header.frame_id or "exo" not in e_rgb.header.frame_id:
            return

        stamp_sec = h_rgb.header.stamp.sec + h_rgb.header.stamp.nanosec * 1e-9

        run_solver = (self.last_solver_time is None or
                      (stamp_sec - self.last_solver_time) >= self.min_solver_interval)

        if not run_solver:
            if self.current_t is not None and not self.in_fallback:
                self.broadcast_transform(e_rgb.header.stamp)
            return

        num_2d = 0
        num_3d = 0
        inliers = 0
        ratio = 0.0
        rmse = 0.0
        status = 'UNKNOWN'
        grav_err = np.nan

        try:
            if self.head_k is None or self.exo_k is None:
                status = 'WAITING_FOR_CAMERA_INFO'
            else:
                head_img = self.bridge.imgmsg_to_cv2(h_rgb, 'bgr8')
                head_dep = self.bridge.imgmsg_to_cv2(h_depth, 'passthrough')
                exo_img = self.bridge.imgmsg_to_cv2(e_rgb, 'bgr8')
                exo_dep = self.bridge.imgmsg_to_cv2(e_depth, 'passthrough')

                pts_head, pts_exo = self.matcher.match(head_img, exo_img)
                num_2d = len(pts_head)

                if len(pts_head) == 0 or len(pts_exo) == 0:
                    status = 'NO_2D_MATCHES'
                else:
                    points_3d_head = []
                    points_3d_exo = []

                    try:
                        t_h_link_opt = self.tf_buffer.lookup_transform(
                            self.head_frame_id, h_rgb.header.frame_id,
                            h_rgb.header.stamp, rclpy.duration.Duration(seconds=0.1))
                        t_e_link_opt = self.tf_buffer.lookup_transform(
                            self.exo_frame_id, e_rgb.header.frame_id,
                            e_rgb.header.stamp, rclpy.duration.Duration(seconds=0.1))

                        def tf_to_matrix(tf):
                            mat = np.eye(4)
                            q = [tf.transform.rotation.x, tf.transform.rotation.y,
                                 tf.transform.rotation.z, tf.transform.rotation.w]
                            mat[:3, :3] = Rotation.from_quat(q).as_matrix()
                            mat[:3, 3] = [tf.transform.translation.x,
                                          tf.transform.translation.y,
                                          tf.transform.translation.z]
                            return mat

                        T_h_link_opt = tf_to_matrix(t_h_link_opt)
                        T_e_link_opt = tf_to_matrix(t_e_link_opt)
                    except Exception as e:
                        self.get_logger().warn(f"Link-to-optical TF: {str(e)}")
                        status = 'WAITING_FOR_LINK_TF'
                        self.log_metrics(stamp_sec, num_2d, 0, 0, 0.0, 0.0, status)
                        return

                    for p_h, p_e in zip(pts_head, pts_exo):
                        p3_h_opt = get_3d_point(int(p_h[0]), int(p_h[1]), head_dep, self.head_k)
                        p3_e_opt = get_3d_point(int(p_e[0]), int(p_e[1]), exo_dep, self.exo_k)
                        if p3_h_opt is not None and p3_e_opt is not None:
                            p3_h_link = (T_h_link_opt[:3, :3] @ p3_h_opt) + T_h_link_opt[:3, 3]
                            p3_e_link = (T_e_link_opt[:3, :3] @ p3_e_opt) + T_e_link_opt[:3, 3]
                            points_3d_head.append(p3_h_link)
                            points_3d_exo.append(p3_e_link)

                    points_3d_head = np.array(points_3d_head)
                    points_3d_exo = np.array(points_3d_exo)
                    num_3d = len(points_3d_head)

                    if num_3d < self.min_3d_matches:
                        status = 'INSUFFICIENT_3D_MATCHES'
                    else:
                        T, inliers, rmse = compute_transform_ransac(points_3d_head, points_3d_exo)

                        if T is None:
                            status = 'RANSAC_FAILED'
                        else:
                            ratio = inliers / num_3d

                            if not (rmse <= self.max_ransac_rmse and
                                    ratio >= self.min_inlier_ratio and
                                    inliers >= self.min_3d_matches):
                                status = 'REJECTED_CONFIDENCE_LIMITS'
                            else:
                                new_t = T[:3, 3]
                                new_q = Rotation.from_matrix(T[:3, :3]).as_quat()
                                dist = np.linalg.norm(new_t)
                                valid = True

                                if not (0.2 <= dist <= 2.2):
                                    status = 'REJECTED_DISTANCE_BOUNDS'
                                    valid = False
                                elif not (0.1 <= new_t[2] <= 1.5):
                                    status = 'REJECTED_HEIGHT_BOUNDS'
                                    valid = False

                                if valid:
                                    moving = (self.head_motion_state == 'moving' or
                                              self.exo_motion_state == 'moving')
                                    tj = self.max_trans_jump_moving if moving else self.max_trans_jump
                                    rj = self.max_rot_jump_moving if moving else self.max_rot_jump

                                    # Skip jump checks before first real solve (initial guess is arbitrary)
                                    if self._has_real_solve:
                                        trans_jump = np.linalg.norm(new_t - self.current_t)
                                        if trans_jump > tj:
                                            status = 'REJECTED_TRANS_JUMP'
                                            valid = False

                                    if valid and self._has_real_solve:
                                        dot = abs(np.dot(self.current_q, new_q))
                                        dot = min(1.0, max(0.0, dot))
                                        rot_jump = 2.0 * np.arccos(dot)
                                        if rot_jump > rj:
                                            status = 'REJECTED_ROT_JUMP'
                                            valid = False

                                if valid and self.imu_gravity_enabled and \
                                   self.head_gravity is not None and self.exo_gravity is not None:
                                    from exo_head_slam.utils.imu_utils import check_gravity_alignment
                                    g_head = self.head_gravity.copy()
                                    g_exo = self.exo_gravity.copy()
                                    try:
                                        T_h_l = self.tf_buffer.lookup_transform(
                                            self.head_frame_id, self.head_gravity_frame,
                                            h_rgb.header.stamp, rclpy.duration.Duration(seconds=0.1))
                                        qh = [T_h_l.transform.rotation.x, T_h_l.transform.rotation.y,
                                              T_h_l.transform.rotation.z, T_h_l.transform.rotation.w]
                                        g_head = Rotation.from_quat(qh).as_matrix() @ g_head
                                    except Exception:
                                        pass
                                    try:
                                        T_e_l = self.tf_buffer.lookup_transform(
                                            self.exo_frame_id, self.exo_gravity_frame,
                                            e_rgb.header.stamp, rclpy.duration.Duration(seconds=0.1))
                                        qe = [T_e_l.transform.rotation.x, T_e_l.transform.rotation.y,
                                              T_e_l.transform.rotation.z, T_e_l.transform.rotation.w]
                                        g_exo = Rotation.from_quat(qe).as_matrix() @ g_exo
                                    except Exception:
                                        pass
                                    aligned, grav_err = check_gravity_alignment(
                                        T[:3, :3], g_head, g_exo, self.imu_gravity_threshold_deg)
                                    if not aligned:
                                        status = 'REJECTED_GRAVITY_MISMATCH'
                                        valid = False

                                if valid:
                                    status = 'SUCCESS'
                                    if not self._has_real_solve:
                                        # First real solve — snap directly, don't blend from initial guess
                                        self.current_t = new_t.copy()
                                        self.current_q = new_q.copy()
                                        self._has_real_solve = True
                                    else:
                                        self.current_t = ((1.0 - self.tf_filter_alpha) * self.current_t +
                                                          self.tf_filter_alpha * new_t)
                                        if np.dot(self.current_q, new_q) < 0:
                                            new_q = -new_q
                                        self.current_q = ((1.0 - self.tf_filter_alpha) * self.current_q +
                                                          self.tf_filter_alpha * new_q)
                                        self.current_q /= np.linalg.norm(self.current_q)

                                    self.last_solver_time = stamp_sec
                                    if self.gyro_propagation_enabled:
                                        self.gyro_T[:3, :3] = T[:3, :3].copy()
                                        self.gyro_T[:3, 3] = T[:3, 3].copy()
                                        self.gyro_R = self.gyro_T[:3, :3].copy()
                                        self.gyro_propagation_start = None
        except Exception as e:
            self.get_logger().error(f"Solver exception: {str(e)}")
            status = f"ERROR_{type(e).__name__}"

        if status == 'SUCCESS':
            self.in_fallback = False
            self.gyro_propagation_start = None
        elif status in ('UNKNOWN', 'WAITING_FOR_CAMERA_INFO', 'WAITING_FOR_LINK_TF'):
            # Benign startup states — keep existing transform, don't enter fallback
            pass
        elif self.current_t is not None and self.gyro_propagation_enabled:
            # Real solver failure with existing transform — try gyro propagation
            if self.gyro_propagation_start is None:
                self.gyro_propagation_start = stamp_sec
            elapsed = stamp_sec - self.gyro_propagation_start
            if elapsed <= self.gyro_max_duration:
                status = 'GYRO_PROPAGATED'
                self.in_fallback = True
            else:
                # Gyro timed out — hold last known transform, stop active propagation
                self.in_fallback = False
                self.gyro_propagation_start = None
                self.get_logger().warn(
                    'Gyro propagation timed out, holding last known transform',
                    throttle_duration_sec=5.0)
        else:
            # No existing transform or gyro disabled — nothing to propagate
            self.in_fallback = False

        imu_constraint_val = float(
            self.imu_gravity_enabled and self.head_gravity is not None and self.exo_gravity is not None)
        stationary = 0.0
        if self.head_motion_state is not None or self.exo_motion_state is not None:
            stationary = 0.0 if (self.head_motion_state == 'moving' or
                                 self.exo_motion_state == 'moving') else 1.0
        self.log_metrics(stamp_sec, num_2d, num_3d, inliers, ratio, rmse, status,
                         gravity_error_deg=grav_err, is_stationary=stationary,
                         imu_constraint_applied=imu_constraint_val)

        if self.current_t is not None and not self.in_fallback:
            self.broadcast_transform(e_rgb.header.stamp)

    def broadcast_transform(self, stamp):
        if self.current_t is None or self.current_q is None:
            return
        msg = TransformStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = self.exo_frame_id
        msg.child_frame_id = self.head_frame_id
        msg.transform.translation.x = float(self.current_t[0])
        msg.transform.translation.y = float(self.current_t[1])
        msg.transform.translation.z = float(self.current_t[2])
        msg.transform.rotation.x = float(self.current_q[0])
        msg.transform.rotation.y = float(self.current_q[1])
        msg.transform.rotation.z = float(self.current_q[2])
        msg.transform.rotation.w = float(self.current_q[3])
        self.tf_broadcaster.sendTransform(msg)

    def gyro_broadcast_timer(self):
        if self.current_t is not None:
            self.broadcast_transform(self.get_clock().now().to_msg())

    def log_metrics(self, timestamp, num_2d, num_3d, inliers, ratio, rmse, status,
                    gravity_error_deg=np.nan, is_stationary=np.nan, imu_constraint_applied=np.nan):
        if not self.metrics_enabled:
            return

        gt_t = [np.nan, np.nan, np.nan]
        gt_q = [np.nan, np.nan, np.nan, np.nan]
        error_t = np.nan
        error_r_deg = np.nan

        if self.gt_parent_frame and self.gt_child_frame:
            try:
                sec = int(timestamp)
                nanosec = int((timestamp - sec) * 1e9)
                stamp_time = rclpy.time.Time(seconds=sec, nanoseconds=nanosec).to_msg()
                gt_tf = self.tf_buffer.lookup_transform(
                    self.gt_parent_frame, self.gt_child_frame,
                    stamp_time, rclpy.duration.Duration(seconds=0.05))
                gt_t = [gt_tf.transform.translation.x, gt_tf.transform.translation.y,
                        gt_tf.transform.translation.z]
                gt_q = [gt_tf.transform.rotation.x, gt_tf.transform.rotation.y,
                        gt_tf.transform.rotation.z, gt_tf.transform.rotation.w]
                if self.current_t is not None and self.current_q is not None:
                    error_t = float(np.linalg.norm(np.array(self.current_t) - np.array(gt_t)))
                    dot = abs(np.dot(self.current_q, np.array(gt_q)))
                    error_r_deg = float(np.degrees(2.0 * np.arccos(min(1.0, max(0.0, dot)))))
            except Exception:
                pass

        t = (self.current_t if self.current_t is not None else [np.nan, np.nan, np.nan])
        q = (self.current_q if self.current_q is not None else [np.nan, np.nan, np.nan, np.nan])

        try:
            with open(self.metrics_csv_path, mode='a', newline='') as f:
                w = csv.writer(f)
                w.writerow([timestamp, num_2d, num_3d, inliers, ratio, rmse, status,
                            t[0], t[1], t[2], q[0], q[1], q[2], q[3],
                            gt_t[0], gt_t[1], gt_t[2], gt_q[0], gt_q[1], gt_q[2], gt_q[3],
                            error_t, error_r_deg,
                            gravity_error_deg, is_stationary, imu_constraint_applied])
        except Exception as e:
            self.get_logger().error(f"CSV write error: {str(e)}")


def main(args=None):
    rclpy.init(args=args)
    node = ExtrinsicSolverNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
