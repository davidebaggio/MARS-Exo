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

from exo_head_slam.utils.math_utils import compute_transform_ransac
from exo_head_slam.utils.vision_utils import get_3d_point

# ponytail: Fallback visual odometry implemented in python using ORB/LightGlue and existing RANSAC/SVD utils
from exo_head_slam.utils.matchers import ORBMatcher, LightGlueMatcher

class FallbackVisualOdometryNode(Node):
    def __init__(self):
        super().__init__('fallback_rgbd_odometry')
        
        self.declare_parameter('publish_tf', True)
        self.declare_parameter('frame_id', 'exo_link')
        self.declare_parameter('odom_frame_id', 'odom')
        self.declare_parameter('approx_sync', True)
        self.declare_parameter('approx_sync_max_interval', 0.02)
        
        # Matcher Parameters
        self.declare_parameter('matcher_type', 'lightglue')
        self.declare_parameter('lightglue_device', 'cuda')
        self.declare_parameter('lightglue_max_keypoints', 2048)

        
        self.publish_tf = self.get_parameter('publish_tf').value
        self.frame_id = self.get_parameter('frame_id').value
        self.odom_frame_id = self.get_parameter('odom_frame_id').value
        
        self.matcher_type = self.get_parameter('matcher_type').value
        self.lightglue_device = self.get_parameter('lightglue_device').value
        self.lightglue_max_keypoints = self.get_parameter('lightglue_max_keypoints').value
        
        self.bridge = CvBridge()
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        
        # Instantiate Matcher
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
        
        # Tracking State
        self.prev_rgb = None
        self.prev_depth = None
        
        # Accumulated Pose (T_odom_cam)
        self.T_odom_cam = np.eye(4)
        
        # Subscribers (remapped externally)
        self.rgb_sub = message_filters.Subscriber(self, Image, 'rgb/image')
        self.depth_sub = message_filters.Subscriber(self, Image, 'depth/image')
        self.info_sub = message_filters.Subscriber(self, CameraInfo, 'rgb/camera_info')
        
        max_interval = self.get_parameter('approx_sync_max_interval').value
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub, self.info_sub],
            queue_size=10,
            slop=max_interval
        )
        self.ts.registerCallback(self.image_callback)
        
        self.odom_pub = self.create_publisher(Odometry, 'odom', 10)
        self.get_logger().info(f"Fallback VO online ({self.matcher.name}). Parent frame: {self.odom_frame_id}, Child frame: {self.frame_id}")

    def image_callback(self, rgb_msg, depth_msg, info_msg):
        try:
            # Convert images
            rgb_img = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='rgb8')
            depth_img = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='32FC1')
            
            if self.prev_rgb is not None:
                # Match features using configured matcher (ORB or LightGlue)
                pts_prev, pts_curr = self.matcher.match(self.prev_rgb, rgb_img)
                
                if len(pts_prev) >= 10:
                    pts_prev_3d = []
                    pts_curr_3d = []
                    K = np.array(info_msg.k).reshape(3, 3)
                    
                    for pt_p, pt_c in zip(pts_prev[:100], pts_curr[:100]):
                        u_prev, v_prev = int(round(pt_p[0])), int(round(pt_p[1]))
                        u_curr, v_curr = int(round(pt_c[0])), int(round(pt_c[1]))
                        
                        p3_prev = get_3d_point(u_prev, v_prev, self.prev_depth, K)
                        p3_curr = get_3d_point(u_curr, v_curr, depth_img, K)
                        
                        if p3_prev is not None and p3_curr is not None:
                            pts_prev_3d.append(p3_prev)
                            pts_curr_3d.append(p3_curr)
                            
                    pts_prev_3d = np.array(pts_prev_3d)
                    pts_curr_3d = np.array(pts_curr_3d)
                    
                    if len(pts_prev_3d) >= 10:
                        # RANSAC estimation: pts_prev ~ T_prev_curr * pts_curr
                        T_prev_curr, inliers, rmse = compute_transform_ransac(pts_curr_3d, pts_prev_3d, threshold=0.05, iterations=100)
                        
                        if T_prev_curr is not None and inliers >= 8:
                            self.T_odom_cam = self.T_odom_cam @ T_prev_curr
                        else:
                            self.get_logger().warn("Fallback VO: tracking failure, insufficient inliers")
            
            # Save tracking variables
            self.prev_rgb = rgb_img
            self.prev_depth = depth_img
            
            self.publish_current_odom(rgb_msg.header.stamp)
            
        except Exception as e:
            self.get_logger().error(f"Error in fallback VO: {str(e)}")


    def publish_current_odom(self, stamp):
        t = self.T_odom_cam[:3, 3]
        rot_mat = self.T_odom_cam[:3, :3]
        
        r = R.from_matrix(rot_mat)
        q = r.as_quat()
        
        if self.publish_tf:
            t_msg = TransformStamped()
            t_msg.header.stamp = stamp
            t_msg.header.frame_id = self.odom_frame_id
            t_msg.child_frame_id = self.frame_id
            
            t_msg.transform.translation.x = float(t[0])
            t_msg.transform.translation.y = float(t[1])
            t_msg.transform.translation.z = float(t[2])
            
            t_msg.transform.rotation.x = float(q[0])
            t_msg.transform.rotation.y = float(q[1])
            t_msg.transform.rotation.z = float(q[2])
            t_msg.transform.rotation.w = float(q[3])
            
            self.tf_broadcaster.sendTransform(t_msg)
            
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
    node = FallbackVisualOdometryNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
