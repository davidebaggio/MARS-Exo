import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import cv2
import uuid


class DepthPreprocessorNode(Node):
    def __init__(self):
        super().__init__('depth_preprocessor_node')
        self.instance_id = str(uuid.uuid4())[:8]

        # Declare parameters with strict defaults
        self.declare_parameter('input_depth_topic', 'UNDEFINED')
        self.declare_parameter('output_depth_topic', 'UNDEFINED')
        self.declare_parameter('expected_frame_id', 'UNDEFINED')
        self.declare_parameter('depth_filter.enabled', True)
        self.declare_parameter('depth_filter.depth_unit_scale', 0.001)
        self.declare_parameter('depth_filter.spatial.kernel_size', 5)
        self.declare_parameter('depth_filter.spatial.sigma_color', 0.08)
        self.declare_parameter('depth_filter.spatial.sigma_space', 3.0)
        self.declare_parameter('depth_filter.temporal.enabled', True)
        self.declare_parameter('depth_filter.temporal.alpha', 0.6)

        # Get values
        self.input_depth_topic = self.get_parameter('input_depth_topic').value
        self.output_depth_topic = self.get_parameter('output_depth_topic').value
        self.expected_frame_id = self.get_parameter('expected_frame_id').value
        
        if self.input_depth_topic == 'UNDEFINED' or self.output_depth_topic == 'UNDEFINED':
            self.get_logger().error(f'[{self.instance_id}] CRITICAL: input_depth_topic or output_depth_topic not set!')
            return

        self.enabled = self.get_parameter('depth_filter.enabled').value
        self.depth_unit_scale = float(self.get_parameter('depth_filter.depth_unit_scale').value)
        self.kernel_size = int(self.get_parameter('depth_filter.spatial.kernel_size').value)
        self.sigma_color = float(self.get_parameter('depth_filter.spatial.sigma_color').value)
        self.sigma_space = float(self.get_parameter('depth_filter.spatial.sigma_space').value)
        self.temporal_enabled = self.get_parameter('depth_filter.temporal.enabled').value
        self.temporal_alpha = float(self.get_parameter('depth_filter.temporal.alpha').value)

        if self.kernel_size < 3: self.kernel_size = 3
        if self.kernel_size % 2 == 0: self.kernel_size += 1

        self.bridge = CvBridge()
        self.previous_filtered: np.ndarray | None = None
        self._log_count = 0

        # Subscribers and Publishers
        from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=100
        )
        self.depth_sub = self.create_subscription(Image, self.input_depth_topic, self.depth_callback, qos)
        self.filtered_depth_pub = self.create_publisher(Image, self.output_depth_topic, qos)

        self.get_logger().info(f'[{self.instance_id}] Depth preprocessor "{self.get_name()}" online.')
        self.get_logger().info(f'  -> Sub: {self.input_depth_topic}')
        self.get_logger().info(f'  -> Pub: {self.output_depth_topic}')
        self.get_logger().info(f'  -> Expect Frame: {self.expected_frame_id}')

    def to_meters(self, depth_image: np.ndarray) -> np.ndarray:
        # If values are large (>100), assume they are in millimeters and convert to meters
        # This handles cases where float images are published with millimeter values.
        depth_m = depth_image.astype(np.float32).copy()
        if np.max(depth_m) > 100.0:
            depth_m *= self.depth_unit_scale
        
        return np.nan_to_num(depth_m, nan=0.0, posinf=0.0, neginf=0.0)

    def spatial_filter(self, depth_m: np.ndarray) -> np.ndarray:
        valid_mask = depth_m > 0
        if not np.any(valid_mask):
            return depth_m

        depth_mm = (depth_m * 1000).astype(np.uint16)
        median_mm = cv2.medianBlur(depth_mm, self.kernel_size)
        
        depth_filled = depth_m.copy()
        holes = (depth_mm == 0)
        depth_filled[holes] = median_mm[holes].astype(np.float32) * 0.001

        filtered = cv2.bilateralFilter(depth_filled, self.kernel_size, self.sigma_color, self.sigma_space)
        filtered[~valid_mask] = 0.0 
        return filtered

    def temporal_filter(self, current_depth: np.ndarray) -> np.ndarray:
        if self.previous_filtered is None or not self.temporal_enabled:
            return current_depth

        if current_depth.shape != self.previous_filtered.shape:
            return current_depth

        blended = current_depth.copy()
        c_mask = current_depth > 0
        p_mask = self.previous_filtered > 0
        
        overlap = c_mask & p_mask
        blended[overlap] = (self.temporal_alpha * current_depth[overlap] + 
                            (1.0 - self.temporal_alpha) * self.previous_filtered[overlap])
        
        missing = ~c_mask & p_mask
        blended[missing] = self.previous_filtered[missing]
        
        return blended

    def depth_callback(self, msg: Image):
        if self.expected_frame_id != 'UNDEFINED':
            if self.expected_frame_id not in msg.header.frame_id:
                self.get_logger().error(f"[{self.instance_id}] CROSSTALK: Got '{msg.header.frame_id}', expected '{self.expected_frame_id}' on {self.input_depth_topic}")
                return

        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1').copy()
            depth_m = self.to_meters(cv_img)

            if not self.enabled:
                res = depth_m
            else:
                res = self.spatial_filter(depth_m)
                res = self.temporal_filter(res)

            if self._log_count % 30 == 0:
                self.get_logger().info(f"[{self.instance_id}] Stats: min={np.min(res):.2f}, max={np.max(res):.2f}, mean={np.mean(res):.2f}")
            self._log_count += 1

            self.previous_filtered = res.copy()

            out_msg = self.bridge.cv2_to_imgmsg(res.astype(np.float32), encoding='32FC1')
            out_msg.header = msg.header
            self.filtered_depth_pub.publish(out_msg)

        except Exception as e:
            self.get_logger().error(f'[{self.instance_id}] Error: {str(e)}')


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
