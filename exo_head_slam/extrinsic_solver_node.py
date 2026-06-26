import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import TransformStamped
import tf2_ros
import message_filters
from cv_bridge import CvBridge
import numpy as np
import cv2
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
            if self.head_k is None or self.exo_k is None:
                self.get_logger().warn("Waiting for CameraInfo...")
                return

            try:
                head_img = self.bridge.imgmsg_to_cv2(h_rgb, 'bgr8')
                head_dep = self.bridge.imgmsg_to_cv2(h_depth, 'passthrough')
                exo_img = self.bridge.imgmsg_to_cv2(e_rgb, 'bgr8')
                exo_dep = self.bridge.imgmsg_to_cv2(e_depth, 'passthrough')

                # Match
                pts_head, pts_exo = self.matcher.match(head_img, exo_img)

                if len(pts_head) == 0 or len(pts_exo) == 0:
                    self.get_logger().warn("Nessun match 2D valido trovato.")
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

                    if len(points_3d_head) < self.min_3d_matches:
                        self.get_logger().warn(f"Insufficient 3D matches ({len(points_3d_head)}/{self.min_3d_matches}). Skipping.")
                    else:
                        # Solve for HeadLink in ExoLink frame (T_exo_head)
                        T, inlier_count, rmse = compute_transform_ransac(points_3d_head, points_3d_exo) 
                        
                        if T is None:
                            self.get_logger().warn(f"RANSAC failed to find a valid transform for {len(points_3d_head)} matches.")
                        else:
                            inlier_ratio = inlier_count / len(points_3d_head)
                            
                            # Confidence checks
                            if rmse <= self.max_ransac_rmse and inlier_ratio >= self.min_inlier_ratio and inlier_count >= self.min_3d_matches:
                                new_t = T[:3, 3]
                                new_q = R.from_matrix(T[:3, :3]).as_quat() # [x, y, z, w]
                                
                                # Physical bounds check
                                dist = np.linalg.norm(new_t)
                                valid = True
                                if not (0.2 <= dist <= 2.2):
                                    self.get_logger().warn(f"Rejecting transform: distance {dist:.2f} m out of bounds [0.2, 2.2]")
                                    valid = False
                                elif not (0.1 <= new_t[2] <= 1.5):
                                    self.get_logger().warn(f"Rejecting transform: relative Z height {new_t[2]:.2f} m out of bounds [0.1, 1.5]")
                                    valid = False
                                
                                # Geometric consistency checks (jump limits)
                                if valid and self.current_t is not None:
                                    trans_jump = np.linalg.norm(new_t - self.current_t)
                                    if trans_jump > self.max_trans_jump:
                                        self.get_logger().warn(
                                            f"Rejecting transform due to translation jump: {trans_jump:.3f} m > {self.max_trans_jump} m"
                                        )
                                        valid = False
                                
                                if valid and self.current_q is not None:
                                    dot_product = abs(np.dot(self.current_q, new_q))
                                    dot_product = min(1.0, max(0.0, dot_product))
                                    rot_jump = 2.0 * np.arccos(dot_product)
                                    if rot_jump > self.max_rot_jump:
                                        self.get_logger().warn(
                                            f"Rejecting transform due to rotation jump: {rot_jump:.3f} rad > {self.max_rot_jump} rad"
                                        )
                                        valid = False

                                if valid:
                                    # Update EMA-filtered state
                                    if self.current_t is None:
                                        self.current_t = new_t
                                        self.current_q = new_q
                                        self.get_logger().info(
                                            f"Initialized extrinsic: t=[{self.current_t[0]:.3f}, {self.current_t[1]:.3f}, {self.current_t[2]:.3f}] "
                                            f"with {inlier_count} inliers (RMSE={rmse:.3f} m)"
                                        )
                                    else:
                                        self.current_t = (1.0 - self.tf_filter_alpha) * self.current_t + self.tf_filter_alpha * new_t
                                        
                                        if np.dot(self.current_q, new_q) < 0:
                                            new_q = -new_q
                                        self.current_q = (1.0 - self.tf_filter_alpha) * self.current_q + self.tf_filter_alpha * new_q
                                        self.current_q /= np.linalg.norm(self.current_q)
                                        self.get_logger().info(
                                            f"Updated extrinsic: t=[{self.current_t[0]:.3f}, {self.current_t[1]:.3f}, {self.current_t[2]:.3f}] "
                                            f"with {inlier_count} inliers (RMSE={rmse:.3f} m)"
                                        )
                                    
                                    self.last_solver_time = stamp_sec
                            else:
                                self.get_logger().warn(
                                    f"Rejecting transform due to confidence checks: "
                                    f"RMSE={rmse:.3f} (max={self.max_ransac_rmse}), "
                                    f"ratio={inlier_ratio:.2f} (min={self.min_inlier_ratio}), "
                                    f"inliers={inlier_count} (min={self.min_3d_matches})"
                                )
                                
            except Exception as e:
                self.get_logger().error(f"Solver callback failed: {str(e)}")

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
