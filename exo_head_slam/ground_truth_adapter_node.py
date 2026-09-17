import copy

import rclpy
import tf2_ros
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from rclpy.time import Time
from tf2_msgs.msg import TFMessage


def odometry_transform(message):
    transform = TransformStamped()
    transform.header = message.header
    transform.child_frame_id = message.child_frame_id
    transform.transform.translation.x = message.pose.pose.position.x
    transform.transform.translation.y = message.pose.pose.position.y
    transform.transform.translation.z = message.pose.pose.position.z
    transform.transform.rotation = message.pose.pose.orientation
    return transform


def transform_odometry(transform, child_frame_id='gt_exo_link'):
    message = Odometry()
    message.header = transform.header
    message.child_frame_id = child_frame_id
    message.pose.pose.position.x = transform.transform.translation.x
    message.pose.pose.position.y = transform.transform.translation.y
    message.pose.pose.position.z = transform.transform.translation.z
    message.pose.pose.orientation = transform.transform.rotation
    return message


def renamed_pair(transform, parent='gt_exo_link', child='gt_head_link'):
    result = copy.deepcopy(transform)
    result.header.frame_id = parent
    result.child_frame_id = child
    return result


class GroundTruthAdapterNode(Node):
    """Compose dynamic camera GT without exposing the recorded robot TF tree."""

    def __init__(self):
        super().__init__('ground_truth_adapter')
        self.declare_parameter('exo_camera_frame', 'front_camera_link')
        self.declare_parameter('head_camera_frame', 'head_camera_link')
        self.exo_camera_frame = self.get_parameter('exo_camera_frame').value
        self.head_camera_frame = self.get_parameter('head_camera_frame').value

        self.buffer = tf2_ros.Buffer(cache_time=Duration(seconds=30.0))
        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1000,
        )
        static_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.pair_publisher = self.create_publisher(
            TFMessage, '/ground_truth/pair_tf', reliable_qos
        )
        self.exo_odom_publisher = self.create_publisher(
            Odometry, '/ground_truth/exo_odom', reliable_qos
        )
        self.create_subscription(
            TFMessage, '/ground_truth/tf_static', self.static_callback, static_qos
        )
        self.create_subscription(
            TFMessage, '/ground_truth/tf', self.dynamic_callback,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Odometry, '/exoskeleton/odom', self.odom_callback,
            qos_profile_sensor_data,
        )
        self.get_logger().info(
            'Dynamic camera ground-truth adapter online (evaluation only)'
        )

    def static_callback(self, message):
        for transform in message.transforms:
            self.buffer.set_transform_static(transform, 'ground_truth_bag')

    def dynamic_callback(self, message):
        for transform in message.transforms:
            self.buffer.set_transform(transform, 'ground_truth_bag')
        if message.transforms:
            self.publish_pair(message.transforms[-1].header.stamp)

    def odom_callback(self, message):
        self.buffer.set_transform(odometry_transform(message), 'ground_truth_odom')
        stamp = message.header.stamp
        try:
            world_from_exo = self.buffer.lookup_transform(
                message.header.frame_id, self.exo_camera_frame,
                Time.from_msg(stamp),
            )
        except Exception:
            return
        self.exo_odom_publisher.publish(transform_odometry(world_from_exo))
        self.publish_pair(stamp)

    def publish_pair(self, stamp):
        try:
            exo_from_head = self.buffer.lookup_transform(
                self.exo_camera_frame, self.head_camera_frame,
                Time.from_msg(stamp),
            )
        except Exception:
            return
        exo_from_head.header.stamp = stamp
        self.pair_publisher.publish(TFMessage(
            transforms=[renamed_pair(exo_from_head)]
        ))


def main(args=None):
    rclpy.init(args=args)
    node = GroundTruthAdapterNode()
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
