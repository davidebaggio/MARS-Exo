import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField
from sensor_msgs_py import point_cloud2 as pc2
from std_msgs.msg import Header


class DepthToPointcloudNode(Node):
    """Converts masked depth images to PointCloud2 using camera intrinsics."""

    def __init__(self):
        super().__init__('depth_to_pointcloud')

        self.declare_parameter('depth_topic', '/head/masked/depth_raw')
        self.declare_parameter('camera_info_topic', '/camera/head/color/camera_info')
        self.declare_parameter('pointcloud_topic', '/head/pointcloud')
        self.declare_parameter('pointcloud_frame_id', 'head_camera_link')
        self.declare_parameter('depth_unit_scale', 0.001)
        self.declare_parameter('max_depth', 5.0)
        self.declare_parameter('downsample_factor', 4)

        self.depth_topic = self.get_parameter('depth_topic').value
        self.camera_info_topic = self.get_parameter('camera_info_topic').value
        self.pointcloud_topic = self.get_parameter('pointcloud_topic').value
        self.pointcloud_frame_id = self.get_parameter('pointcloud_frame_id').value
        self.depth_unit_scale = self.get_parameter('depth_unit_scale').value
        self.max_depth = self.get_parameter('max_depth').value
        self.downsample = self.get_parameter('downsample_factor').value

        self.pc_pub = self.create_publisher(PointCloud2, self.pointcloud_topic, 10)
        self.depth_sub = self.create_subscription(Image, self.depth_topic, self._depth_cb, 10)
        self.info_sub = self.create_subscription(CameraInfo, self.camera_info_topic, self._info_cb, 10)

        self.K = None
        self.H = None
        self.W = None
        self.grid_x = None
        self.grid_y = None

        self.get_logger().info(
            f'depth_to_pointcloud initialized: depth={self.depth_topic}, '
            f'info={self.camera_info_topic}, pc={self.pointcloud_topic}'
        )

    def _info_cb(self, msg: CameraInfo):
        if self.K is None:
            self.K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
            self.H = msg.height
            self.W = msg.width
            u, v = np.meshgrid(
                np.arange(0, self.W, self.downsample),
                np.arange(0, self.H, self.downsample),
            )
            self.grid_u = u.ravel()
            self.grid_v = v.ravel()
            self.get_logger().info(
                f'Camera intrinsics received: {self.W}x{self.H}, '
                f'downsample={self.downsample}, points per frame={self.grid_u.shape[0]}'
            )

    def _depth_cb(self, msg: Image):
        if self.K is None:
            return

        if msg.encoding == '16UC1':
            depth = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)
            depth = depth.astype(np.float64) * self.depth_unit_scale
        elif msg.encoding == '32FC1':
            depth = np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width).astype(np.float64)
        else:
            self.get_logger().warn(f'Unsupported depth encoding: {msg.encoding}')
            return

        depth_ds = depth[::self.downsample, ::self.downsample]
        valid = (depth_ds > 0) & (depth_ds < self.max_depth)
        u_valid = self.grid_u[valid.ravel()]
        v_valid = self.grid_v[valid.ravel()]
        z_valid = depth_ds[valid]

        if u_valid.shape[0] == 0:
            return

        x = (u_valid - self.K[0, 2]) * z_valid / self.K[0, 0]
        y = (v_valid - self.K[1, 2]) * z_valid / self.K[1, 1]

        points = np.stack([x, y, z_valid], axis=1)

        header = Header()
        header.stamp = msg.header.stamp
        header.frame_id = self.pointcloud_frame_id

        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]

        pc_msg = pc2.create_cloud(header, fields, points.astype(np.float32))
        self.pc_pub.publish(pc_msg)


def main(args=None):
    rclpy.init(args=args)
    node = DepthToPointcloudNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
