import rclpy
import tf2_ros
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.time import Time

from exo_head_slam.ground_truth_adapter_node import (
    odometry_transform,
    renamed_pair,
    transform_odometry,
)


def transform(parent, child, x, z=0.0, stamp=None):
    message = TransformStamped()
    message.header.frame_id = parent
    message.child_frame_id = child
    message.header.stamp = (stamp or Time()).to_msg()
    message.transform.translation.x = x
    message.transform.translation.z = z
    message.transform.rotation.w = 1.0
    return message


def test_dynamic_camera_ground_truth_is_composed_from_bag_tf():
    initialized_here = not rclpy.ok()
    if initialized_here:
        rclpy.init()
    try:
        stamp = Time(seconds=1.0)
        buffer = tf2_ros.Buffer()
        buffer.set_transform_static(
            transform('waist_link', 'front_camera_link', 1.0), 'test'
        )
        buffer.set_transform(
            transform('waist_link', 'head_camera_link', 3.0, 0.5, stamp), 'test'
        )

        pair = renamed_pair(buffer.lookup_transform(
            'front_camera_link', 'head_camera_link', stamp
        ))
        assert pair.header.frame_id == 'gt_exo_link'
        assert pair.child_frame_id == 'gt_head_link'
        assert pair.transform.translation.x == 2.0
        assert pair.transform.translation.z == 0.5

        waist_odom = Odometry()
        waist_odom.header.frame_id = 'world'
        waist_odom.child_frame_id = 'waist_link'
        waist_odom.header.stamp = stamp.to_msg()
        waist_odom.pose.pose.position.x = 10.0
        waist_odom.pose.pose.orientation.w = 1.0
        buffer.set_transform(odometry_transform(waist_odom), 'test')
        exo_odom = transform_odometry(buffer.lookup_transform(
            'world', 'front_camera_link', stamp
        ))
        assert exo_odom.child_frame_id == 'gt_exo_link'
        assert exo_odom.pose.pose.position.x == 11.0
    finally:
        if initialized_here:
            rclpy.shutdown()
