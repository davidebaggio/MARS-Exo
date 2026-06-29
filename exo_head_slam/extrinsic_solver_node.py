import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import TransformStamped, Vector3Stamped
import tf2_ros
import message_filters
from cv_bridge import CvBridge
import numpy as np
import cv2
import csv
import os
from typing import Optional, Tuple
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
        self.declare_parameter('min_3d_matches', 0)
        self.declare_parameter('matcher_type', 'orb')
        self.declare_parameter('lightglue_device', 'cpu')
        self.declare_parameter('lightglue_max_keypoints', 2048)
        self.declare_parameter('tf_filter_alpha', 0.1)
        self.declare_parameter('exo_height_m', -1.0) # Default -1.0 to fallback if not in config
        
        # New parameters for Section 3 fixes
        self.declare_parameter('min_solver_interval', 0.2)
        self.declare_parameter('max_ransac_rmse', 0.05)
        self.declare_parameter('min_inlier_ratio', 0.3)
        self.declare_parameter('max_trans_jump', 0.3)
        self.declare_parameter('max_rot_jump', 0.5)
        
        # Evaluation metrics parameters
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

        self.bridge = CvBridge()
        self.matcher = self.create_matcher()
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # Filtering state
        self.tf_filter_alpha = float(self.get_parameter('tf_filter_alpha').value)
        self.exo_height_m = float(self.get_parameter('exo_height_m').value)
        self.current_t: Optional[np.ndarray] = None
        self.current_q: Optional[np.ndarray] = None
        
        # Section 3 fixes configuration & state
        self.min_solver_interval = float(self.get_parameter('min_solver_interval').value)
        self.max_ransac_rmse = float(self.get_parameter('max_ransac_rmse').value)
        self.min_inlier_ratio = float(self.get_parameter('min_inlier_ratio').value)
        self.max_trans_jump = float(self.get_parameter('max_trans_jump').value)
        self.max_rot_jump = float(self.get_parameter('max_rot_jump').value)
        self.last_solver_time: Optional[float] = None
        
        # Metrics configuration
        self.metrics_enabled = bool(self.get_parameter('metrics_enabled').value)
        self.metrics_csv_path = str(self.get_parameter('metrics_csv_path').value)
        self.imu_csv_path = os.path.join(os.path.dirname(self.metrics_csv_path) if os.path.dirname(self.metrics_csv_path) else '.', 'imu_evaluation.csv')
        self.gt_parent_frame = str(self.get_parameter('gt_parent_frame').value)
        self.gt_child_frame = str(self.get_parameter('gt_child_frame').value)
        
        # ponytail: initialize metrics file overwriting on start to have clean data per run
        if self.metrics_enabled:
            try:
                with open(self.metrics_csv_path, mode='w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        'timestamp', 'num_2d_matches', 'num_3d_matches', 'inliers', 'inlier_ratio', 'rmse', 'status',
                        't_x', 't_y', 't_z', 'q_x', 'q_y', 'q_z', 'q_w',
                        'gt_t_x', 'gt_t_y', 'gt_t_z', 'gt_q_x', 'gt_q_y', 'gt_q_z', 'gt_q_w',
                        'error_t', 'error_r_deg',
                        'gravity_error_deg', 'is_stationary', 'imu_constraint_applied',
                    ])
                self.get_logger().info(f"Logging metrics to {os.path.abspath(self.metrics_csv_path)}")
            except Exception as e:
                self.get_logger().error(f"Failed to initialize metrics CSV: {str(e)}")
                self.metrics_enabled = False
            try:
                with open(self.imu_csv_path, mode='w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        'timestamp', 'num_2d_matches', 'status', 'gravity_error_deg',
                        'is_stationary', 'imu_constraint_applied',
                    ])
                self.get_logger().info(f"Logging IMU metrics to {os.path.abspath(self.imu_csv_path)}")
            except Exception as e:
                self.get_logger().error(f"Failed to initialize IMU metrics CSV: {str(e)}")
                self.metrics_enabled = False
        
        # IMU gravity constraint
        self.imu_gravity_enabled = bool(self.get_parameter('imu_gravity_constraint.enabled').value)
        self.imu_gravity_threshold_deg = float(self.get_parameter('imu_gravity_constraint.threshold_deg').value)
        self.head_gravity: Optional[np.ndarray] = None
        self.exo_gravity: Optional[np.ndarray] = None
        self.head_gravity_frame: Optional[str] = None
        self.exo_gravity_frame: Optional[str] = None

        if self.imu_gravity_enabled:
            self.head_gravity_sub = self.create_subscription(
                Vector3Stamped,
                self.get_parameter('head_gravity_topic').value,
                self.head_gravity_cb,
                10
            )
            self.exo_gravity_sub = self.create_subscription(
                Vector3Stamped,
                self.get_parameter('exo_gravity_topic').value,
                self.exo_gravity_cb,
                10
            )
        
        # IMU gyro propagation state
        self.gyro_propagation_enabled = bool(self.get_parameter('imu_gyro_propagation.enabled').value)
        self.gyro_max_duration = float(self.get_parameter('imu_gyro_propagation.max_duration').value)
        self.gyro_T = np.eye(4)
        self.gyro_R = np.eye(3)
        self.gyro_active = False
        self.gyro_propagation_start: Optional[float] = None
        self._head_gyro: Optional[np.ndarray] = None
        self._exo_gyro: Optional[np.ndarray] = None
        self._head_gyro_ts: Optional[float] = None
        self._exo_gyro_ts: Optional[float] = None

        if self.gyro_propagation_enabled:
            self.head_gyro_sub = self.create_subscription(
                Vector3Stamped,
                self.get_parameter('head_gyro_topic').value,
                self.head_gyro_cb,
                10
            )
            self.exo_gyro_sub = self.create_subscription(
                Vector3Stamped,
                self.get_parameter('exo_gyro_topic').value,
                self.exo_gyro_cb,
                10
            )

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
        self.head_info_sub = self.create_subscription(CameraInfo, self.head_camera_info_topic, self.head_info_cb, 10)
        self.exo_info_sub = self.create_subscription(CameraInfo, self.exo_camera_info_topic, self.exo_info_cb, 10)
        # Subscribers
        self.head_rgb_sub = message_filters.Subscriber(self, Image, self.head_rgb_topic)
        self.head_depth_sub = message_filters.Subscriber(self, Image, self.head_depth_topic)
        self.exo_rgb_sub = message_filters.Subscriber(self, Image, self.exo_rgb_topic)
        self.exo_depth_sub = message_filters.Subscriber(self, Image, self.exo_depth_topic)
        
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.head_rgb_sub, self.head_depth_sub, self.exo_rgb_sub, self.exo_depth_sub],
            queue_size=2,
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

            self.get_logger().warn(
                f"LightGlue richiesto ma non disponibile, fallback a ORB: {lightglue_matcher.error}"
            )

        if matcher_type not in ('orb', 'lightglue'):
            self.get_logger().warn(f"matcher_type sconosciuto '{matcher_type}', uso ORB.")

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
        ω = np.array([msg.vector.x, msg.vector.y, msg.vector.z])
        ts = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self.gyro_active and self._head_gyro_ts is not None and self._exo_gyro is not None:
            dt = ts - self._head_gyro_ts
            if 0 < dt < 0.1:
                ω_rel = ω - self.gyro_R @ self._exo_gyro
                from scipy.spatial.transform import Rotation as R_gyro
                ΔR = R_gyro.from_rotvec(ω_rel * dt).as_matrix()
                self.gyro_T[:3, :3] = ΔR @ self.gyro_T[:3, :3]
                self.gyro_R = self.gyro_T[:3, :3]
        self._head_gyro = ω
        self._head_gyro_ts = ts

    def exo_gyro_cb(self, msg: Vector3Stamped):
        ω = np.array([msg.vector.x, msg.vector.y, msg.vector.z])
        ts = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self.gyro_active and self._exo_gyro_ts is not None and self._head_gyro is not None:
            dt = ts - self._exo_gyro_ts
            if 0 < dt < 0.1:
                ω_rel = self._head_gyro - self.gyro_R @ ω
                from scipy.spatial.transform import Rotation as R_gyro
                ΔR = R_gyro.from_rotvec(ω_rel * dt).as_matrix()
                self.gyro_T[:3, :3] = ΔR @ self.gyro_T[:3, :3]
                self.gyro_R = self.gyro_T[:3, :3]
        self._exo_gyro = ω
        self._exo_gyro_ts = ts

    def solve_callback(self, h_rgb: Image, h_depth: Image, e_rgb: Image, e_depth: Image):
        # Relaxed verification for better stability
        if "head" not in h_rgb.header.frame_id or "exo" not in e_rgb.header.frame_id:
             self.get_logger().error(f"Sync error: head_frame={h_rgb.header.frame_id}, exo_frame={e_rgb.header.frame_id}")
             return

        stamp_sec = h_rgb.header.stamp.sec + h_rgb.header.stamp.nanosec * 1e-9
        
        # ponytail: simple rate throttling to save computation resources (e.g. GPU LightGlue matching)
        run_solver = False
        if self.last_solver_time is None or (stamp_sec - self.last_solver_time) >= self.min_solver_interval:
            run_solver = True

        if run_solver:
            self.get_logger().info("Solver received synced image quad. Processing extrinsic estimation...")
            num_2d = 0
            num_3d = 0
            inliers = 0
            ratio = 0.0
            rmse = 0.0
            status = 'UNKNOWN'
            grav_err = np.nan

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
                            
                            from scipy.spatial.transform import Rotation as R
                            def tf_to_matrix(tf):
                                mat = np.eye(4)
                                q = [tf.transform.rotation.x, tf.transform.rotation.y, tf.transform.rotation.z, tf.transform.rotation.w]
                                mat[:3, :3] = R.from_quat(q).as_matrix()
                                mat[:3, 3] = [tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z]
                                return mat

                            T_h_link_opt = tf_to_matrix(t_h_link_opt)
                            T_e_link_opt = tf_to_matrix(t_e_link_opt)
                        except Exception as e:
                            self.get_logger().warn(f"Waiting for link-to-optical TFs: {str(e)}")
                            status = 'WAITING_FOR_LINK_TF'
                            self.log_metrics(stamp_sec, num_2d, 0, 0, 0.0, 0.0, status,
                                             gravity_error_deg=grav_err)
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
                            T, inliers, rmse = compute_transform_ransac(points_3d_head, points_3d_exo) 
                            
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
                                    new_q = R.from_matrix(T[:3, :3]).as_quat() # [x, y, z, w]
                                    
                                    # Physical bounds check
                                    dist = np.linalg.norm(new_t)
                                    valid = True
                                    if not (0.2 <= dist <= 2.2):
                                        self.get_logger().warn(f"Rejecting transform: distance {dist:.2f} m out of bounds [0.2, 2.2]")
                                        status = 'REJECTED_DISTANCE_BOUNDS'
                                        valid = False
                                    elif not (0.1 <= new_t[2] <= 1.5):
                                        self.get_logger().warn(f"Rejecting transform: relative Z height {new_t[2]:.2f} m out of bounds [0.1, 1.5]")
                                        status = 'REJECTED_HEIGHT_BOUNDS'
                                        valid = False
                                    
                                    # Geometric consistency checks (jump limits)
                                    if valid and self.current_t is not None:
                                        trans_jump = np.linalg.norm(new_t - self.current_t)
                                        if trans_jump > self.max_trans_jump:
                                            self.get_logger().warn(
                                                f"Rejecting transform due to translation jump: {trans_jump:.3f} m > {self.max_trans_jump} m"
                                            )
                                            status = 'REJECTED_TRANS_JUMP'
                                            valid = False
                                    
                                    if valid and self.current_q is not None:
                                        dot_product = abs(np.dot(self.current_q, new_q))
                                        dot_product = min(1.0, max(0.0, dot_product))
                                        rot_jump = 2.0 * np.arccos(dot_product)
                                        if rot_jump > self.max_rot_jump:
                                            self.get_logger().warn(
                                                f"Rejecting transform due to rotation jump: {rot_jump:.3f} rad > {self.max_rot_jump} rad"
                                            )
                                            status = 'REJECTED_ROT_JUMP'
                                            valid = False

                                    if valid and self.imu_gravity_enabled and self.head_gravity is not None and self.exo_gravity is not None:
                                        from exo_head_slam.utils.imu_utils import check_gravity_alignment
                                        g_head = self.head_gravity.copy()
                                        g_exo = self.exo_gravity.copy()
                                        try:
                                            T_h_link = self.tf_buffer.lookup_transform(
                                                self.head_frame_id, self.head_gravity_frame,
                                                h_rgb.header.stamp, rclpy.duration.Duration(seconds=0.1))
                                            q_h = [T_h_link.transform.rotation.x, T_h_link.transform.rotation.y,
                                                   T_h_link.transform.rotation.z, T_h_link.transform.rotation.w]
                                            R_h_link = R.from_quat(q_h).as_matrix()
                                            g_head = R_h_link @ g_head
                                        except Exception:
                                            pass
                                        try:
                                            T_e_link = self.tf_buffer.lookup_transform(
                                                self.exo_frame_id, self.exo_gravity_frame,
                                                e_rgb.header.stamp, rclpy.duration.Duration(seconds=0.1))
                                            q_e = [T_e_link.transform.rotation.x, T_e_link.transform.rotation.y,
                                                   T_e_link.transform.rotation.z, T_e_link.transform.rotation.w]
                                            R_e_link = R.from_quat(q_e).as_matrix()
                                            g_exo = R_e_link @ g_exo
                                        except Exception:
                                            pass
                                        aligned, grav_err = check_gravity_alignment(
                                            T[:3, :3], g_head, g_exo, self.imu_gravity_threshold_deg)
                                        if not aligned:
                                            status = 'REJECTED_GRAVITY_MISMATCH'
                                            valid = False
                                    if valid:
                                        status = 'SUCCESS'
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
                                        
                                        self.last_solver_time = stamp_sec
                                        if self.gyro_propagation_enabled:
                                            self.gyro_T[:3, :3] = T[:3, :3].copy()
                                            self.gyro_T[:3, 3] = T[:3, 3].copy()
                                            self.gyro_R = self.gyro_T[:3, :3].copy()
                                            self.gyro_active = True
                                            self.gyro_propagation_start = None
            except Exception as e:
                self.get_logger().error(f"Solver callback failed: {str(e)}")
                status = f"ERROR_{type(e).__name__}"

            if self.gyro_active and self.gyro_propagation_enabled and status not in ('SUCCESS', 'UNKNOWN', 'WAITING_FOR_CAMERA_INFO', 'WAITING_FOR_LINK_TF'):
                if self.gyro_propagation_start is None:
                    self.gyro_propagation_start = stamp_sec
                elapsed = stamp_sec - self.gyro_propagation_start
                if elapsed <= self.gyro_max_duration:
                    if status in ('NO_2D_MATCHES', 'INSUFFICIENT_3D_MATCHES', 'RANSAC_FAILED',
                                  'REJECTED_CONFIDENCE_LIMITS', 'REJECTED_GRAVITY_MISMATCH'):
                        self.current_t = self.gyro_T[:3, 3].copy()
                        from scipy.spatial.transform import Rotation as R_gyro
                        self.current_q = R_gyro.from_matrix(self.gyro_T[:3, :3]).as_quat()
                        status = 'GYRO_PROPAGATED'
                else:
                    self.gyro_active = False

            imu_constraint_applied_val = float(self.imu_gravity_enabled and self.head_gravity is not None and self.exo_gravity is not None)
            self.log_metrics(stamp_sec, num_2d, num_3d, inliers, ratio, rmse, status,
                             gravity_error_deg=grav_err, imu_constraint_applied=imu_constraint_applied_val)

        # ponytail: simple TF fallback - broadcast the last known good filtered transform 
        # with the current timestamp to keep the TF tree active even when solver fails or is throttled.
        if self.current_t is not None:
            self.broadcast_transform(e_rgb.header.stamp)

    def broadcast_transform(self, stamp):
        if self.current_t is None or self.current_q is None:
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
        
        self.tf_broadcaster.sendTransform(t_msg)

    def log_metrics(self, timestamp: float, num_2d: int, num_3d: int, inliers: int, ratio: float, rmse: float, status: str,
                    gravity_error_deg: float = np.nan, is_stationary: float = np.nan, imu_constraint_applied: float = np.nan):
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
                    self.gt_parent_frame,
                    self.gt_child_frame,
                    stamp_time,
                    timeout=rclpy.duration.Duration(seconds=0.05)
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
            except Exception:
                pass
                
        t_x, t_y, t_z = (self.current_t[0], self.current_t[1], self.current_t[2]) if self.current_t is not None else (np.nan, np.nan, np.nan)
        q_x, q_y, q_z, q_w = (self.current_q[0], self.current_q[1], self.current_q[2], self.current_q[3]) if self.current_q is not None else (np.nan, np.nan, np.nan, np.nan)
        
        try:
            with open(self.metrics_csv_path, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    timestamp, num_2d, num_3d, inliers, ratio, rmse, status,
                    t_x, t_y, t_z, q_x, q_y, q_z, q_w,
                    gt_t[0], gt_t[1], gt_t[2], gt_q[0], gt_q[1], gt_q[2], gt_q[3],
                    error_t, error_r_deg,
                    gravity_error_deg, is_stationary, imu_constraint_applied
                ])
            with open(self.imu_csv_path, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    timestamp, num_2d, status,
                    gravity_error_deg, is_stationary, imu_constraint_applied,
                ])
        except Exception as e:
            self.get_logger().error(f"Failed to write metrics row: {str(e)}")

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
