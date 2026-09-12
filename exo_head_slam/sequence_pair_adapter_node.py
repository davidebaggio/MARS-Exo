import copy

import message_filters
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image


def sliding_pair(previous, current):
    """Return overlapping adjacent frames and retain current for next pair."""
    return current, None if previous is None else (previous, current)


def retag_frame(frame, stamp, frame_id):
    messages = copy.deepcopy(frame)
    for message in messages:
        message.header.stamp = stamp
        message.header.frame_id = frame_id
    return messages


class SequencePairAdapterNode(Node):
    def __init__(self):
        super().__init__('sequence_pair_adapter')
        self.previous = None
        self.pair_count = 0

        self.head_publishers = self._create_publishers('head')
        self.exo_publishers = self._create_publishers('exo')
        subscribers = [
            message_filters.Subscriber(
                self, message_type, topic, qos_profile=qos_profile_sensor_data
            )
            for message_type, topic in (
                (Image, '/camera/rgb/image_color'),
                (Image, '/camera/depth/image'),
                (CameraInfo, '/camera/rgb/camera_info'),
            )
        ]
        self.sync = message_filters.ApproximateTimeSynchronizer(
            subscribers, queue_size=10, slop=0.05
        )
        self.sync.registerCallback(self.callback)
        self.get_logger().info('Sliding sequence-pair adapter online')

    def _create_publishers(self, camera):
        return (
            self.create_publisher(
                Image, f'/camera/{camera}/color/image_raw', qos_profile_sensor_data
            ),
            self.create_publisher(
                Image, f'/camera/{camera}/aligned_depth_to_color/image_raw',
                qos_profile_sensor_data,
            ),
            self.create_publisher(
                CameraInfo, f'/camera/{camera}/color/camera_info',
                qos_profile_sensor_data,
            ),
        )

    def callback(self, rgb, depth, camera_info):
        current = (rgb, depth, camera_info)
        self.previous, pair = sliding_pair(self.previous, current)
        if pair is None:
            return

        stamp = rgb.header.stamp
        head = retag_frame(pair[0], stamp, 'head_camera_color_optical_frame')
        exo = retag_frame(pair[1], stamp, 'front_camera_color_optical_frame')
        for publisher, message in zip(self.head_publishers, head):
            publisher.publish(message)
        for publisher, message in zip(self.exo_publishers, exo):
            publisher.publish(message)

        self.pair_count += 1
        if self.pair_count == 1 or self.pair_count % 25 == 0:
            self.get_logger().info(f'Published {self.pair_count} coupled pairs')


def main(args=None):
    rclpy.init(args=args)
    node = SequencePairAdapterNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if rclpy.ok():
            node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
