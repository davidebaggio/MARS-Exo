import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import cv2


class DepthPreprocessorNode(Node):
    def __init__(self):
        super().__init__('depth_preprocessor_node')

        self.declare_parameter('input_depth_topic', '')
        self.declare_parameter('output_depth_topic', '')
        self.declare_parameter('depth_filter.enabled', True)
        self.declare_parameter('depth_filter.depth_unit_scale', 0.001)
        self.declare_parameter('depth_filter.spatial.kernel_size', 5)
        self.declare_parameter('depth_filter.spatial.sigma_color', 0.08)
        self.declare_parameter('depth_filter.spatial.sigma_space', 3.0)
        self.declare_parameter('depth_filter.temporal.enabled', True)
        self.declare_parameter('depth_filter.temporal.alpha', 0.6)

        self.enabled = self.get_parameter('depth_filter.enabled').value
        self.depth_unit_scale = float(self.get_parameter('depth_filter.depth_unit_scale').value)
        self.kernel_size = int(self.get_parameter('depth_filter.spatial.kernel_size').value)
        self.sigma_color = float(self.get_parameter('depth_filter.spatial.sigma_color').value)
        self.sigma_space = float(self.get_parameter('depth_filter.spatial.sigma_space').value)
        self.temporal_enabled = self.get_parameter('depth_filter.temporal.enabled').value
        self.temporal_alpha = float(self.get_parameter('depth_filter.temporal.alpha').value)
        self.input_depth_topic = self.get_parameter('input_depth_topic').value
        self.output_depth_topic = self.get_parameter('output_depth_topic').value

        if self.kernel_size < 3:
            self.kernel_size = 3
        if self.kernel_size % 2 == 0:
            self.kernel_size += 1

        self.bridge = CvBridge()
        self.previous_filtered: np.ndarray | None = None

        self.depth_sub = self.create_subscription(Image, self.input_depth_topic, self.depth_callback, 10)
        self.filtered_depth_pub = self.create_publisher(Image, self.output_depth_topic, 10)

        self.get_logger().info('Depth preprocessor initialized.')

    def to_meters(self, depth_image: np.ndarray) -> np.ndarray:
        if np.issubdtype(depth_image.dtype, np.integer):
            depth_m = depth_image.astype(np.float32) * self.depth_unit_scale
        else:
            depth_m = depth_image.astype(np.float32)

        return np.nan_to_num(depth_m, nan=0.0, posinf=0.0, neginf=0.0)

    def spatial_filter(self, depth_m: np.ndarray) -> np.ndarray:
        valid_mask = depth_m > 0
        if not np.any(valid_mask):
            return depth_m

        depth_filled = depth_m.copy()
        depth_mm = np.round(depth_m / self.depth_unit_scale).astype(np.uint16)
        median_mm = cv2.medianBlur(depth_mm, self.kernel_size)
        filled_valid = depth_mm > 0
        depth_filled[~filled_valid] = median_mm[~filled_valid].astype(np.float32) * self.depth_unit_scale

        filtered = cv2.bilateralFilter(depth_filled.astype(np.float32), self.kernel_size, self.sigma_color, self.sigma_space)
        filtered[~valid_mask] = depth_filled[~valid_mask]
        return filtered

    def temporal_filter(self, current_depth: np.ndarray) -> np.ndarray:
        if self.previous_filtered is None or not self.temporal_enabled:
            return current_depth

        current_valid = current_depth > 0
        previous_valid = self.previous_filtered > 0
        blended = current_depth.copy()
        overlap = current_valid & previous_valid
        blended[overlap] = (
            self.temporal_alpha * current_depth[overlap]
            + (1.0 - self.temporal_alpha) * self.previous_filtered[overlap]
        )
        blended[~current_valid & previous_valid] = self.previous_filtered[~current_valid & previous_valid]
        return blended

    def depth_callback(self, msg: Image):
        try:
            depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            depth_m = self.to_meters(depth_image)

            if not self.enabled:
                filtered_depth = depth_m
            else:
                filtered_depth = self.spatial_filter(depth_m)
                filtered_depth = self.temporal_filter(filtered_depth)

            self.previous_filtered = filtered_depth.copy()

            filtered_msg = self.bridge.cv2_to_imgmsg(filtered_depth.astype(np.float32), encoding='32FC1')
            filtered_msg.header = msg.header
            self.filtered_depth_pub.publish(filtered_msg)
        except Exception as exc:
            self.get_logger().error(f'Depth preprocessing failed: {exc}')


def main(args=None):
    rclpy.init(args=args)
    node = DepthPreprocessorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()