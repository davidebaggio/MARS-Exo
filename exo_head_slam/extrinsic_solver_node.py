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

class ORBMatcher:
    name = 'orb'

    """ORB-based feature matching placeholder."""
    def __init__(self, n_features: int = 1000):
        self.orb = cv2.ORB_create(nfeatures=n_features)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def match(self, img1: np.ndarray, img2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Matches features between two images. Returns (N, 2) arrays of (u, v)."""
        kp1, des1 = self.orb.detectAndCompute(img1, None)
        kp2, des2 = self.orb.detectAndCompute(img2, None)
        
        if des1 is None or des2 is None:
            return np.array([]).reshape(0, 2), np.array([]).reshape(0, 2)
            
        matches = self.bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
        
        return pts1, pts2


class LightGlueMatcher:
    name = 'lightglue'

    def __init__(self, device: str = 'cpu', max_keypoints: int = 2048):
        self.device = device
        self.max_keypoints = max_keypoints
        self.available = False
        self.error: Optional[Exception] = None
        self.torch = None
        self.extractor = None
        self.matcher = None

        try:
            import torch
            from lightglue import LightGlue, SuperPoint

            self.torch = torch
            self.extractor = SuperPoint(max_num_keypoints=max_keypoints).eval().to(device)
            self.matcher = LightGlue(features='superpoint').eval().to(device)
            self.available = True
        except Exception as exc:
            self.error = exc

    def _to_tensor(self, image: np.ndarray):
        gray = image
        if gray.ndim == 3:
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        gray = np.ascontiguousarray(gray.astype(np.float32) / 255.0)
        tensor = self.torch.from_numpy(gray)[None, None, :, :]
        return tensor.to(self.device)

    def _to_numpy(self, value):
        if value is None:
            return None
        if hasattr(value, 'detach'):
            value = value.detach().cpu().numpy()
        return np.asarray(value)

    def _extract_points(self, features):
        keypoints = self._to_numpy(features.get('keypoints'))
        if keypoints is None:
            return np.empty((0, 2), dtype=np.float32)
        return np.squeeze(keypoints, axis=0) if keypoints.ndim == 3 else keypoints

    def match(self, img1: np.ndarray, img2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self.available:
            raise RuntimeError(f'LightGlue unavailable: {self.error}')

        image0 = self._to_tensor(img1)
        image1 = self._to_tensor(img2)

        with self.torch.inference_mode():
            features0 = self.extractor.extract(image0)
            features1 = self.extractor.extract(image1)
            matches = self.matcher({'image0': features0, 'image1': features1})

        keypoints0 = self._extract_points(features0)
        keypoints1 = self._extract_points(features1)

        if 'matches0' in matches:
            matches0 = self._to_numpy(matches['matches0'])
            if matches0 is None:
                return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)
            matches0 = np.squeeze(matches0).astype(np.int64)
            valid = matches0 > -1
            if not np.any(valid):
                return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)
            pts0 = keypoints0[valid]
            pts1 = keypoints1[matches0[valid]]
            return pts0.astype(np.float32), pts1.astype(np.float32)

        if 'matches' in matches:
            paired_matches = self._to_numpy(matches['matches'])
            if paired_matches is None or paired_matches.size == 0:
                return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)
            paired_matches = np.asarray(paired_matches)
            if paired_matches.ndim == 2 and paired_matches.shape[1] == 2:
                pts0 = keypoints0[paired_matches[:, 0].astype(np.int64)]
                pts1 = keypoints1[paired_matches[:, 1].astype(np.int64)]
                return pts0.astype(np.float32), pts1.astype(np.float32)

        return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)

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
            queue_size=30,
            slop=0.1
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

        self.get_logger().info("Solver received synced image quad.")
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
                return
            
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
                return

            # Solve for HeadLink in ExoLink frame (T_exo_head)
            T = compute_transform_ransac(points_3d_head, points_3d_exo) 
            
            if T is not None:
                # Height constraint ground truth
                if self.exo_height_m > 0:
                    try:
                        # Lookup map -> odom or map -> exo transform to constrain exo height
                        # Since exo is now the SLAM source, map -> exo_camera_link is available
                        t_map_e = self.tf_buffer.lookup_transform('map', self.exo_frame_id, e_rgb.header.stamp,
                                                                 timeout=rclpy.duration.Duration(seconds=0.1))
                        
                        # Note: If constraining exo_height, we just use the SLAM output or adjust SLAM.
                        # For now, we keep the solver producing relative TF.
                        pass
                    except Exception as e:
                        self.get_logger().debug(f"Height constraint skipped: {str(e)}")

                self.broadcast_transform(T, e_rgb.header.stamp)
                
        except Exception as e:
            self.get_logger().error(f"Solver callback failed: {str(e)}")

    def broadcast_transform(self, T: np.ndarray, stamp):
        from scipy.spatial.transform import Rotation as R
        new_t = T[:3, 3]
        new_q = R.from_matrix(T[:3, :3]).as_quat() # [x, y, z, w]

        if self.current_t is None:
            self.current_t = new_t
            self.current_q = new_q
        else:
            # EMA for translation
            self.current_t = (1.0 - self.tf_filter_alpha) * self.current_t + self.tf_filter_alpha * new_t
            
            # EMA for quaternion
            if np.dot(self.current_q, new_q) < 0:
                new_q = -new_q
            self.current_q = (1.0 - self.tf_filter_alpha) * self.current_q + self.tf_filter_alpha * new_q
            self.current_q /= np.linalg.norm(self.current_q)

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
