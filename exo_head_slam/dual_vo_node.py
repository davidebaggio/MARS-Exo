import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
import message_filters
from cv_bridge import CvBridge
import cv2
import numpy as np
import tf2_ros
from scipy.spatial.transform import Rotation as R
from typing import Optional

from exo_head_slam.utils.math_utils import compute_transform_ransac
from exo_head_slam.utils.vision_utils import get_3d_point
from exo_head_slam.utils.matchers import ORBMatcher, LightGlueMatcher
from exo_head_slam.utils.imu_utils import gyro_integrate_rotvec
from exo_head_slam.utils.pose_graph import SlidingWindowPoseGraph, PoseGraphKeyframe, PoseGraphEdge


class DualVisualOdometryNode(Node):
    def __init__(self):
        super().__init__('dual_visual_odometry')

        self.declare_parameter('frame_id', 'exo_link')
        self.declare_parameter('odom_frame_id', 'odom')
        self.declare_parameter('map_frame_id', 'map')
        self.declare_parameter('matcher_type', 'lightglue')
        self.declare_parameter('lightglue_device', 'cuda')
        self.declare_parameter('lightglue_max_keypoints', 2048)
        self.declare_parameter('pose_graph_window', 20)
        self.declare_parameter('min_matches_temporal', 10)
        self.declare_parameter('min_matches_cross', 8)
        self.declare_parameter('keyframe_trans_threshold', 0.05)
        self.declare_parameter('keyframe_rot_threshold', 0.05)
        self.declare_parameter('imu_gyro_assist.enabled', True)
        self.declare_parameter('imu_gyro_topic', '/imu/exo/gyro_filtered')
        self.declare_parameter('publish_tf', True)

        self.publish_tf = self.get_parameter('publish_tf').value
        self.frame_id = self.get_parameter('frame_id').value
        self.odom_frame_id = self.get_parameter('odom_frame_id').value
        self.map_frame_id = self.get_parameter('map_frame_id').value
        self.matcher_type = self.get_parameter('matcher_type').value
        self.lightglue_device = self.get_parameter('lightglue_device').value
        self.lightglue_max_keypoints = self.get_parameter('lightglue_max_keypoints').value
        self.pose_graph_window = self.get_parameter('pose_graph_window').value
        self.min_matches_temporal = self.get_parameter('min_matches_temporal').value
        self.min_matches_cross = self.get_parameter('min_matches_cross').value
        self.keyframe_trans_threshold = self.get_parameter('keyframe_trans_threshold').value
        self.keyframe_rot_threshold = self.get_parameter('keyframe_rot_threshold').value

        self.bridge = CvBridge()
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        if self.matcher_type == 'lightglue':
            self.matcher = LightGlueMatcher(
                device=self.lightglue_device,
                max_keypoints=self.lightglue_max_keypoints
            )
            if not self.matcher.available:
                self.get_logger().warn(
                    f"LightGlue requested but not available, falling back to ORB. Error: {self.matcher.error}"
                )
                self.matcher = ORBMatcher()
        else:
            self.matcher = ORBMatcher()

        self.pose_graph = SlidingWindowPoseGraph(window_size=self.pose_graph_window)

        self.head_rgb = None
        self.head_depth = None
        self.head_K = None
        self.head_stamp = None
        self.head_ts = None

        self.exo_rgb = None
        self.exo_depth = None
        self.exo_K = None
        self.exo_stamp = None
        self.exo_ts = None

        self.prev_head_rgb = None
        self.prev_head_depth = None
        self.prev_exo_rgb = None
        self.prev_exo_depth = None

        self.T_odom_exo = np.eye(4)
        self.prev_timestamp = None

        self.imu_gyro_enabled = bool(self.get_parameter('imu_gyro_assist.enabled').value)
        self.latest_gyro: Optional[np.ndarray] = None
        self.gyro_timestamp: Optional[float] = None
        if self.imu_gyro_enabled:
            from geometry_msgs.msg import Vector3Stamped
            self.gyro_sub = self.create_subscription(
                Vector3Stamped,
                self.get_parameter('imu_gyro_topic').value,
                self.gyro_callback,
                10
            )

        head_rgb_sub = message_filters.Subscriber(self, Image, '/head/masked/image_raw')
        head_depth_sub = message_filters.Subscriber(self, Image, '/head/masked/depth_raw')
        head_info_sub = message_filters.Subscriber(self, CameraInfo, '/camera/head/color/camera_info')
        self.head_ts = message_filters.ApproximateTimeSynchronizer(
            [head_rgb_sub, head_depth_sub, head_info_sub],
            queue_size=10,
            slop=0.02
        )
        self.head_ts.registerCallback(self.head_callback)

        exo_rgb_sub = message_filters.Subscriber(self, Image, '/exo/masked/image_raw')
        exo_depth_sub = message_filters.Subscriber(self, Image, '/exo/masked/depth_raw')
        exo_info_sub = message_filters.Subscriber(self, CameraInfo, '/camera/exo/color/camera_info')
        self.exo_ts = message_filters.ApproximateTimeSynchronizer(
            [exo_rgb_sub, exo_depth_sub, exo_info_sub],
            queue_size=10,
            slop=0.02
        )
        self.exo_ts.registerCallback(self.exo_callback)

        self.odom_pub = self.create_publisher(Odometry, '/slam/odom', 10)
        self.create_timer(1.0 / 30.0, self.timer_callback)

        self.get_logger().info(
            f"Dual VO online ({self.matcher.name}). "
            f"Odom: {self.odom_frame_id}, Child: {self.frame_id}"
        )

    def gyro_callback(self, msg):
        self.latest_gyro = np.array([msg.vector.x, msg.vector.y, msg.vector.z])
        self.gyro_timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

    def head_callback(self, rgb_msg, depth_msg, info_msg):
        try:
            self.head_rgb = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='rgb8')
            self.head_depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='32FC1')
            self.head_K = np.array(info_msg.k).reshape(3, 3)
            self.head_stamp = rgb_msg.header.stamp
            self.head_ts = rgb_msg.header.stamp.sec + rgb_msg.header.stamp.nanosec * 1e-9
        except Exception as e:
            self.get_logger().error(f"Error in head callback: {str(e)}")

    def exo_callback(self, rgb_msg, depth_msg, info_msg):
        try:
            self.exo_rgb = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='rgb8')
            self.exo_depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='32FC1')
            self.exo_K = np.array(info_msg.k).reshape(3, 3)
            self.exo_stamp = rgb_msg.header.stamp
            self.exo_ts = rgb_msg.header.stamp.sec + rgb_msg.header.stamp.nanosec * 1e-9
        except Exception as e:
            self.get_logger().error(f"Error in exo callback: {str(e)}")

    def _detect_keypoints(self, rgb):
        if isinstance(self.matcher, ORBMatcher):
            kp = self.matcher.orb.detect(rgb, None)
            return np.float32([k.pt for k in kp])
        elif isinstance(self.matcher, LightGlueMatcher):
            gray = rgb
            if gray.ndim == 3:
                gray = cv2.cvtColor(gray, cv2.COLOR_RGB2GRAY)
            gray = np.ascontiguousarray(gray.astype(np.float32) / 255.0)
            tensor = self.matcher.torch.from_numpy(gray)[None, None, :, :].to(self.matcher.device)
            with self.matcher.torch.inference_mode():
                features = self.matcher.extractor.extract(tensor)
            return self.matcher._extract_points(features)
        return np.empty((0, 2), dtype=np.float32)

    def _backproject_points(self, pts, depth, K):
        pts_3d = []
        for pt in pts:
            u, v = int(round(pt[0])), int(round(pt[1]))
            p3 = get_3d_point(u, v, depth, K)
            if p3 is not None:
                pts_3d.append(p3)
        return np.array(pts_3d) if pts_3d else np.empty((0, 3))

    def timer_callback(self):
        stamp = self.get_clock().now().to_msg()
        if self.exo_stamp is not None:
            stamp = self.exo_stamp
        elif self.head_stamp is not None:
            stamp = self.head_stamp

        if self.head_rgb is None or self.exo_rgb is None:
            self.publish_odometry(stamp)
            return

        try:
            if self.prev_head_rgb is None:
                self.prev_head_rgb = self.head_rgb.copy()
                self.prev_head_depth = self.head_depth.copy()
                self.prev_exo_rgb = self.exo_rgb.copy()
                self.prev_exo_depth = self.exo_depth.copy()
                self.prev_timestamp = self.head_ts
                self.publish_odometry(stamp)
                return

            T_exo_head = None
            try:
                t = self.tf_buffer.lookup_transform(
                    'exo_link', 'head_link', rclpy.time.Time()
                )
                T_exo_head = np.eye(4)
                T_exo_head[:3, 3] = [
                    t.transform.translation.x,
                    t.transform.translation.y,
                    t.transform.translation.z
                ]
                q = t.transform.rotation
                R_exo_head = R.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
                T_exo_head[:3, :3] = R_exo_head
            except Exception:
                pass

            gyro_rotation = None
            if self.imu_gyro_enabled and self.latest_gyro is not None and self.prev_timestamp is not None:
                dt = self.head_ts - self.prev_timestamp if self.head_ts is not None else 0.0
                if dt > 0:
                    rv = gyro_integrate_rotvec(self.latest_gyro, dt)
                    gyro_rotation = R.from_rotvec(rv).as_matrix()

            T_head_prev_curr = None
            head_success = False
            head_inliers = 0
            if self.head_K is not None:
                head_kp_prev, head_kp_curr = self.matcher.match(self.prev_head_rgb, self.head_rgb)
                if len(head_kp_prev) >= self.min_matches_temporal:
                    pts_prev_3d = []
                    pts_curr_3d = []
                    for pt_p, pt_c in zip(head_kp_prev[:100], head_kp_curr[:100]):
                        u_p, v_p = int(round(pt_p[0])), int(round(pt_p[1]))
                        u_c, v_c = int(round(pt_c[0])), int(round(pt_c[1]))
                        p3_p = get_3d_point(u_p, v_p, self.prev_head_depth, self.head_K)
                        p3_c = get_3d_point(u_c, v_c, self.head_depth, self.head_K)
                        if p3_p is not None and p3_c is not None:
                            pts_prev_3d.append(p3_p)
                            pts_curr_3d.append(p3_c)
                    if len(pts_prev_3d) >= self.min_matches_temporal:
                        T_head_prev_curr, head_inliers, _ = compute_transform_ransac(
                            np.array(pts_curr_3d), np.array(pts_prev_3d),
                            threshold=0.05, iterations=100
                        )
                        if T_head_prev_curr is not None and head_inliers >= self.min_matches_temporal:
                            head_success = True

            T_exo_prev_curr = None
            exo_success = False
            exo_inliers = 0
            if self.exo_K is not None:
                exo_kp_prev, exo_kp_curr = self.matcher.match(self.prev_exo_rgb, self.exo_rgb)
                if len(exo_kp_prev) >= self.min_matches_temporal:
                    pts_prev_3d = []
                    pts_curr_3d = []
                    for pt_p, pt_c in zip(exo_kp_prev[:100], exo_kp_curr[:100]):
                        u_p, v_p = int(round(pt_p[0])), int(round(pt_p[1]))
                        u_c, v_c = int(round(pt_c[0])), int(round(pt_c[1]))
                        p3_p = get_3d_point(u_p, v_p, self.prev_exo_depth, self.exo_K)
                        p3_c = get_3d_point(u_c, v_c, self.exo_depth, self.exo_K)
                        if p3_p is not None and p3_c is not None:
                            pts_prev_3d.append(p3_p)
                            pts_curr_3d.append(p3_c)
                    if len(pts_prev_3d) >= self.min_matches_temporal:
                        T_exo_prev_curr, exo_inliers, _ = compute_transform_ransac(
                            np.array(pts_curr_3d), np.array(pts_prev_3d),
                            threshold=0.05, iterations=100
                        )
                        if T_exo_prev_curr is not None and exo_inliers >= self.min_matches_temporal:
                            exo_success = True

            cross_success = False
            if T_exo_head is not None:
                head_kp, exo_kp = self.matcher.match(self.head_rgb, self.exo_rgb)
                if len(head_kp) >= self.min_matches_cross:
                    pts_h_3d = []
                    pts_e_3d = []
                    for pt_h, pt_e in zip(head_kp[:100], exo_kp[:100]):
                        u_h, v_h = int(round(pt_h[0])), int(round(pt_h[1]))
                        u_e, v_e = int(round(pt_e[0])), int(round(pt_e[1]))
                        p3_h = get_3d_point(u_h, v_h, self.head_depth, self.head_K)
                        p3_e = get_3d_point(u_e, v_e, self.exo_depth, self.exo_K)
                        if p3_h is not None and p3_e is not None:
                            p3_e_pred = (T_exo_head[:3, :3] @ p3_h) + T_exo_head[:3, 3]
                            if np.linalg.norm(p3_e_pred - p3_e) < 0.1:
                                pts_h_3d.append(p3_h)
                                pts_e_3d.append(p3_e)
                    if len(pts_h_3d) >= self.min_matches_cross:
                        T_cross, cross_inliers, _ = compute_transform_ransac(
                            np.array(pts_h_3d), np.array(pts_e_3d),
                            threshold=0.05, iterations=100
                        )
                        if T_cross is not None and cross_inliers >= self.min_matches_cross:
                            cross_success = True

            visual_success = False
            T_rel_primary = None
            if exo_success and T_exo_prev_curr is not None:
                self.T_odom_exo = self.T_odom_exo @ T_exo_prev_curr
                T_rel_primary = T_exo_prev_curr.copy()
                visual_success = True
            elif head_success and T_head_prev_curr is not None:
                if T_exo_head is not None:
                    T_exo_from_head = T_exo_head @ T_head_prev_curr @ np.linalg.inv(T_exo_head)
                    self.T_odom_exo = self.T_odom_exo @ T_exo_from_head
                    T_rel_primary = T_exo_from_head.copy()
                    visual_success = True

            if not visual_success:
                if gyro_rotation is not None:
                    T_gyro = np.eye(4)
                    T_gyro[:3, :3] = gyro_rotation
                    self.T_odom_exo = self.T_odom_exo @ T_gyro
                elif self.prev_exo_rgb is not None:
                    self.get_logger().warn("Dual VO: tracking failure, no gyro")

            U, _, Vt = np.linalg.svd(self.T_odom_exo[:3, :3])
            self.T_odom_exo[:3, :3] = U @ Vt

            is_keyframe = False
            latest_kf_pose = self.pose_graph.latest_pose()
            if latest_kf_pose is not None:
                T_rel_kf = np.linalg.inv(latest_kf_pose) @ self.T_odom_exo
                trans_mag = np.linalg.norm(T_rel_kf[:3, 3])
                rot_mag = np.linalg.norm(R.from_matrix(T_rel_kf[:3, :3]).as_rotvec())
                if trans_mag > self.keyframe_trans_threshold or rot_mag > self.keyframe_rot_threshold:
                    is_keyframe = True
            else:
                is_keyframe = True

            if is_keyframe:
                head_kp = self._detect_keypoints(self.head_rgb)
                head_3d = self._backproject_points(head_kp[:100], self.head_depth, self.head_K)
                exo_kp = self._detect_keypoints(self.exo_rgb)
                exo_3d = self._backproject_points(exo_kp[:100], self.exo_depth, self.exo_K)

                prev_id = self.pose_graph.latest_kf_id()

                kf = PoseGraphKeyframe(
                    id=0,
                    pose=self.T_odom_exo.copy(),
                    head_kp=head_kp[:100] if len(head_kp) > 0 else None,
                    head_3d=head_3d if len(head_3d) > 0 else None,
                    exo_kp=exo_kp[:100] if len(exo_kp) > 0 else None,
                    exo_3d=exo_3d if len(exo_3d) > 0 else None,
                    timestamp=self.head_ts if self.head_ts is not None else 0.0
                )
                kf_id = self.pose_graph.add_keyframe(kf)

                if prev_id is not None and T_rel_primary is not None:
                    self.pose_graph.add_edge(PoseGraphEdge(
                        src=prev_id,
                        dst=kf_id,
                        T_rel=T_rel_primary,
                        info=np.eye(6) * 100.0,
                        etype='temporal'
                    ))

                if prev_id is not None and head_success and T_head_prev_curr is not None and T_exo_head is not None:
                    T_rel_head = T_exo_head @ T_head_prev_curr @ np.linalg.inv(T_exo_head)
                    self.pose_graph.add_edge(PoseGraphEdge(
                        src=prev_id,
                        dst=kf_id,
                        T_rel=T_rel_head,
                        info=np.eye(6) * 50.0,
                        etype='temporal'
                    ))

                if prev_id is not None and cross_success:
                    self.pose_graph.add_edge(PoseGraphEdge(
                        src=prev_id,
                        dst=kf_id,
                        T_rel=T_rel_primary.copy(),
                        info=np.eye(6) * 30.0,
                        etype='cross_camera'
                    ))

                self.pose_graph.optimize()

                optimized = self.pose_graph.latest_pose()
                if optimized is not None:
                    self.T_odom_exo = optimized

            self.prev_head_rgb = self.head_rgb.copy()
            self.prev_head_depth = self.head_depth.copy()
            self.prev_exo_rgb = self.exo_rgb.copy()
            self.prev_exo_depth = self.exo_depth.copy()
            if self.head_ts is not None:
                self.prev_timestamp = self.head_ts

            self.publish_odometry(stamp)

        except Exception as e:
            self.get_logger().error(f"Error in dual VO timer: {str(e)}")

    def publish_odometry(self, stamp):
        t = self.T_odom_exo[:3, 3]
        rot_mat = self.T_odom_exo[:3, :3]

        r = R.from_matrix(rot_mat)
        q = r.as_quat()

        now_stamp = self.get_clock().now().to_msg()

        if self.publish_tf:
            t_map_odom = TransformStamped()
            t_map_odom.header.stamp = now_stamp
            t_map_odom.header.frame_id = self.map_frame_id
            t_map_odom.child_frame_id = self.odom_frame_id
            t_map_odom.transform.translation.x = 0.0
            t_map_odom.transform.translation.y = 0.0
            t_map_odom.transform.translation.z = 0.0
            t_map_odom.transform.rotation.x = 0.0
            t_map_odom.transform.rotation.y = 0.0
            t_map_odom.transform.rotation.z = 0.0
            t_map_odom.transform.rotation.w = 1.0
            self.tf_broadcaster.sendTransform(t_map_odom)

            t_odom_child = TransformStamped()
            t_odom_child.header.stamp = now_stamp
            t_odom_child.header.frame_id = self.odom_frame_id
            t_odom_child.child_frame_id = self.frame_id
            t_odom_child.transform.translation.x = float(t[0])
            t_odom_child.transform.translation.y = float(t[1])
            t_odom_child.transform.translation.z = float(t[2])
            t_odom_child.transform.rotation.x = float(q[0])
            t_odom_child.transform.rotation.y = float(q[1])
            t_odom_child.transform.rotation.z = float(q[2])
            t_odom_child.transform.rotation.w = float(q[3])
            self.tf_broadcaster.sendTransform(t_odom_child)

        odom_msg = Odometry()
        odom_msg.header.stamp = stamp
        odom_msg.header.frame_id = self.odom_frame_id
        odom_msg.child_frame_id = self.frame_id

        odom_msg.pose.pose.position.x = float(t[0])
        odom_msg.pose.pose.position.y = float(t[1])
        odom_msg.pose.pose.position.z = float(t[2])

        odom_msg.pose.pose.orientation.x = float(q[0])
        odom_msg.pose.pose.orientation.y = float(q[1])
        odom_msg.pose.pose.orientation.z = float(q[2])
        odom_msg.pose.pose.orientation.w = float(q[3])

        self.odom_pub.publish(odom_msg)


def main(args=None):
    rclpy.init(args=args)
    node = DualVisualOdometryNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
