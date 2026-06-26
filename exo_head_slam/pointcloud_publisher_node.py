import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from cv_bridge import CvBridge
import numpy as np
import message_filters
import tf2_ros
from scipy.spatial.transform import Rotation as R


class PointCloudPublisherNode(Node):
    def __init__(self):
        super().__init__('pointcloud_publisher_node')
        
        self.declare_parameter('input_rgb_topic', 'UNDEFINED')
        self.declare_parameter('input_depth_topic', 'UNDEFINED')
        self.declare_parameter('input_camera_info_topic', 'UNDEFINED')
        self.declare_parameter('output_pcl_topic', 'pcl_output')
        self.declare_parameter('global_frame', 'map')
        self.declare_parameter('downsample_factor', 2) 
        
        self.input_rgb_topic = self.get_parameter('input_rgb_topic').value
        self.input_depth_topic = self.get_parameter('input_depth_topic').value
        self.input_camera_info_topic = self.get_parameter('input_camera_info_topic').value
        self.output_pcl_topic = self.get_parameter('output_pcl_topic').value
        self.global_frame = self.get_parameter('global_frame').value
        self.downsample_factor = self.get_parameter('downsample_factor').value

        if 'UNDEFINED' in [self.input_rgb_topic, self.input_depth_topic, self.input_camera_info_topic]:
            self.get_logger().error('Missing topic parameters!')
            return

        self.bridge = CvBridge()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)


        # Sync subscribers
        self.rgb_sub = message_filters.Subscriber(self, Image, self.input_rgb_topic)
        self.depth_sub = message_filters.Subscriber(self, Image, self.input_depth_topic)
        self.info_sub = message_filters.Subscriber(self, CameraInfo, self.input_camera_info_topic)
        
        # Track raw message arrivals
        self.rgb_sub.registerCallback(lambda _: self._count_msg('rgb'))
        self.depth_sub.registerCallback(lambda _: self._count_msg('depth'))
        self.info_sub.registerCallback(lambda _: self._count_msg('info'))

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub, self.info_sub],
            queue_size=10,
            slop=0.1
        )
        self.ts.registerCallback(self.callback)

        self.pcl_pub = self.create_publisher(PointCloud2, self.output_pcl_topic, 10)
        
        self._msg_counts = {'rgb': 0, 'depth': 0, 'info': 0, 'sync': 0}
        self.create_timer(5.0, self.log_stats)

        self.get_logger().info(f'PointCloud Publisher Node online: {self.output_pcl_topic}')

    def _count_msg(self, key):
        self._msg_counts[key] += 1

    def log_stats(self):
        self.get_logger().info(
            f'[{self.output_pcl_topic}] Stats: RGB={self._msg_counts["rgb"]}, '
            f'Depth={self._msg_counts["depth"]}, Info={self._msg_counts["info"]}, '
            f'Synced={self._msg_counts["sync"]}'
        )

    def callback(self, rgb_msg, depth_msg, info_msg):
        self._msg_counts['sync'] += 1
        try:
            # 1. Convert Images
            rgb_img = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='rgb8')
            depth_img = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='32FC1')

            # 2. Downsample
            if self.downsample_factor > 1:
                rgb_img = rgb_img[::self.downsample_factor, ::self.downsample_factor]
                depth_img = depth_img[::self.downsample_factor, ::self.downsample_factor]
                K = np.array(info_msg.k).reshape(3, 3) / self.downsample_factor
                K[2, 2] = 1.0
            else:
                K = np.array(info_msg.k).reshape(3, 3)

            # 3. Project to 3D
            h, w = depth_img.shape
            u, v = np.meshgrid(np.arange(w), np.arange(h))
            
            valid = (depth_img > 0.1) & (depth_img < 10.0)
            z = depth_img[valid]
            u = u[valid]
            v = v[valid]
            rgb = rgb_img[valid]

            if self._msg_counts['sync'] % 30 == 0:
                self.get_logger().info(f'[{self.output_pcl_topic}] Valid points to project: {len(z)}')

            if len(z) == 0:
                return

            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]

            x = (u - cx) * z / fx
            y = (v - cy) * z / fy
            points_cam = np.vstack((x, y, z)).T 

            # Transform points from camera optical frame to global frame
            output_frame = self.global_frame
            tf = None
            for lookup_time, label in [(depth_msg.header.stamp, 'exact'), (rclpy.time.Time(), 'latest')]:
                try:
                    tf = self.tf_buffer.lookup_transform(
                        self.global_frame,
                        depth_msg.header.frame_id,
                        lookup_time,
                        timeout=rclpy.duration.Duration(seconds=0.2)
                    )
                    break
                except Exception as e:
                    self.get_logger().debug(f"TF lookup ({label}) {depth_msg.header.frame_id} -> {self.global_frame}: {str(e)}")
            
            if tf is not None:
                t = np.array([tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z])
                R_mat = R.from_quat([tf.transform.rotation.x, tf.transform.rotation.y, tf.transform.rotation.z, tf.transform.rotation.w]).as_matrix()
                points_global = (R_mat @ points_cam.T).T + t
            else:
                self.get_logger().warn(f"TF lookup failed for {depth_msg.header.frame_id} -> {self.global_frame}, publishing in camera frame")
                points_global = points_cam
                output_frame = depth_msg.header.frame_id


            # 6. Create PointCloud2 (Packed XYZRGB)
            num_points = len(points_global)
            data = np.zeros(num_points, dtype=[
                ('x', np.float32),
                ('y', np.float32),
                ('z', np.float32),
                ('rgb', np.uint32)
            ])
            data['x'] = points_global[:, 0]
            data['y'] = points_global[:, 1]
            data['z'] = points_global[:, 2]
            
            # Pack RGB into uint32 (0x00RRGGBB)
            rgb_packed = (rgb[:, 0].astype(np.uint32) << 16) | \
                         (rgb[:, 1].astype(np.uint32) << 8) | \
                         (rgb[:, 2].astype(np.uint32))
            data['rgb'] = rgb_packed

            pcl_msg = PointCloud2()
            pcl_msg.header.stamp = rgb_msg.header.stamp
            pcl_msg.header.frame_id = output_frame
            pcl_msg.height = 1
            pcl_msg.width = num_points
            pcl_msg.is_dense = False
            pcl_msg.is_bigendian = False
            pcl_msg.fields = [
                PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
                PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1),
            ]
            pcl_msg.point_step = 16
            pcl_msg.row_step = pcl_msg.point_step * num_points
            pcl_msg.data = data.tobytes()

            self.pcl_pub.publish(pcl_msg)

        except Exception as e:
            self.get_logger().error(f'Error in callback: {str(e)}')

def main(args=None):
    rclpy.init(args=args)
    node = PointCloudPublisherNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
