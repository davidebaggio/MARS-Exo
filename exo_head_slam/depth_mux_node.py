import copy

import message_filters
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import CameraInfo, Image


class DepthMuxNode(Node):
    """Republishes synchronized masked depth/color/camera_info from head and exo
    onto a single set of topics so a single NVBlox node can consume them.

    Each camera stream is synchronized independently, and the published output
    bundle uses one shared header stamp across depth, color, and camera_info.
    """

    def __init__(self):
        super().__init__('depth_mux')

        # parameters
        self.declare_parameter('head.depth_topic', '/head/masked/depth_raw')
        self.declare_parameter('head.color_topic', '/head/masked/image_raw')
        self.declare_parameter('head.camera_info_topic', '/camera/head/color/camera_info')

        self.declare_parameter('exo.depth_topic', '/exo/masked/depth_raw')
        self.declare_parameter('exo.color_topic', '/exo/masked/image_raw')
        self.declare_parameter('exo.camera_info_topic', '/camera/exo/color/camera_info')

        self.declare_parameter('merged.depth_topic', '/merged/depth/image_raw')
        self.declare_parameter('merged.color_topic', '/merged/color/image_raw')
        self.declare_parameter('merged.camera_info_topic', '/merged/camera_info')
        self.declare_parameter('sync.queue_size', 30)
        self.declare_parameter('sync.slop', 0.1)

        head_depth = self.get_parameter('head.depth_topic').value
        head_color = self.get_parameter('head.color_topic').value
        head_info = self.get_parameter('head.camera_info_topic').value

        exo_depth = self.get_parameter('exo.depth_topic').value
        exo_color = self.get_parameter('exo.color_topic').value
        exo_info = self.get_parameter('exo.camera_info_topic').value

        self.merged_depth = self.get_parameter('merged.depth_topic').value
        self.merged_color = self.get_parameter('merged.color_topic').value
        self.merged_info = self.get_parameter('merged.camera_info_topic').value
        self.sync_queue_size = int(self.get_parameter('sync.queue_size').value)
        self.sync_slop = float(self.get_parameter('sync.slop').value)

        # publishers
        self.pub_depth = self.create_publisher(Image, self.merged_depth, 10)
        self.pub_color = self.create_publisher(Image, self.merged_color, 10)
        self.pub_info = self.create_publisher(CameraInfo, self.merged_info, 10)

        # Subscribers with synchronization, one bundle per camera.
        self.head_depth_sub = message_filters.Subscriber(self, Image, head_depth)
        self.head_color_sub = message_filters.Subscriber(self, Image, head_color)
        self.head_info_sub = message_filters.Subscriber(self, CameraInfo, head_info)

        self.exo_depth_sub = message_filters.Subscriber(self, Image, exo_depth)
        self.exo_color_sub = message_filters.Subscriber(self, Image, exo_color)
        self.exo_info_sub = message_filters.Subscriber(self, CameraInfo, exo_info)

        self.head_ts = message_filters.ApproximateTimeSynchronizer(
            [self.head_depth_sub, self.head_color_sub, self.head_info_sub],
            queue_size=self.sync_queue_size,
            slop=self.sync_slop,
        )
        self.head_ts.registerCallback(self._head_bundle_cb)

        self.exo_ts = message_filters.ApproximateTimeSynchronizer(
            [self.exo_depth_sub, self.exo_color_sub, self.exo_info_sub],
            queue_size=self.sync_queue_size,
            slop=self.sync_slop,
        )
        self.exo_ts.registerCallback(self._exo_bundle_cb)

        self.get_logger().info(
            f'depth_mux ready, publishing synchronized merged topics: '
            f'{self.merged_depth}, {self.merged_color}, {self.merged_info}'
        )

    def _publish_bundle(self, depth_msg: Image, color_msg: Image, info_msg: CameraInfo) -> None:
        # Publish one synchronized RGB-D + camera_info bundle with aligned headers.
        stamp_header = depth_msg.header

        merged_depth = copy.deepcopy(depth_msg)
        merged_color = copy.deepcopy(color_msg)
        merged_info = copy.deepcopy(info_msg)

        merged_depth.header = stamp_header
        merged_color.header = stamp_header
        merged_info.header = stamp_header

        self.pub_depth.publish(merged_depth)
        self.pub_color.publish(merged_color)
        self.pub_info.publish(merged_info)

    def _head_bundle_cb(self, depth_msg: Image, color_msg: Image, info_msg: CameraInfo) -> None:
        self._publish_bundle(depth_msg, color_msg, info_msg)

    def _exo_bundle_cb(self, depth_msg: Image, color_msg: Image, info_msg: CameraInfo) -> None:
        self._publish_bundle(depth_msg, color_msg, info_msg)


def main(args=None):
    rclpy.init(args=args)
    node = DepthMuxNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
