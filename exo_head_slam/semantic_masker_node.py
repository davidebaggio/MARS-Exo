import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import message_filters
from cv_bridge import CvBridge
import numpy as np
import cv2
import uuid
import os
from typing import List, Optional
from .utils.vision_utils import apply_semantic_mask


class YOLOSegModel:
    def __init__(self, model_path: str, conf_threshold: float, dynamic_classes: List[int]):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.dynamic_classes = dynamic_classes
        self.available = False
        self.error: Optional[Exception] = None
        self.model = None

        try:
            from ultralytics import YOLO
            from pathlib import Path
            model_p = Path(model_path)
            if not model_p.is_absolute():
                repo_root = Path(__file__).resolve().parents[2]
                candidate = repo_root / model_path
                if candidate.exists():
                    model_path = str(candidate)

            self.model = YOLO(model_path)
            self.available = True
        except Exception as exc:
            self.error = exc

    def predict(self, image: np.ndarray) -> np.ndarray:
        if not self.available:
            return np.zeros(image.shape[:2], dtype=np.uint8)

        results = self.model.predict(source=image, conf=self.conf_threshold, verbose=False)
        if not results:
            return np.zeros(image.shape[:2], dtype=np.uint8)

        result = results[0]
        mask = np.zeros(image.shape[:2], dtype=np.uint8)

        if result.masks is not None and result.boxes is not None:
            masks = result.masks.data
            classes = result.boxes.cls
            for idx in range(len(classes)):
                class_id = int(classes[idx].item())
                if self.dynamic_classes and class_id not in self.dynamic_classes:
                    continue

                instance_mask = masks[idx].detach().cpu().numpy().astype(np.float32)
                instance_mask = cv2.resize(
                    instance_mask,
                    (image.shape[1], image.shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                )
                mask[instance_mask > 0.5] = 1
            return mask

        if result.boxes is not None:
            for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
                if self.dynamic_classes and int(cls.item()) not in self.dynamic_classes:
                    continue
                x1, y1, x2, y2 = box.detach().cpu().numpy().astype(int)
                mask[max(0, y1):y2, max(0, x1):x2] = 1
        return mask

class SemanticMaskerNode(Node):
    def __init__(self):
        super().__init__('semantic_masker_node')
        self.instance_id = str(uuid.uuid4())[:8]
        
        self.declare_parameter('input_rgb_topic', 'UNDEFINED')
        self.declare_parameter('input_depth_topic', 'UNDEFINED')
        self.declare_parameter('output_rgb_topic', 'UNDEFINED')
        self.declare_parameter('output_depth_topic', 'UNDEFINED')
        self.declare_parameter('expected_frame_id', 'UNDEFINED')
        self.declare_parameter('masker.model_path', 'yolov8n-seg.pt')
        self.declare_parameter('masker.conf_threshold', 0.25)
        self.declare_parameter('masker.dynamic_classes', [0])
        
        self.input_rgb_topic = self.get_parameter('input_rgb_topic').value
        self.input_depth_topic = self.get_parameter('input_depth_topic').value
        self.output_rgb_topic = self.get_parameter('output_rgb_topic').value
        self.output_depth_topic = self.get_parameter('output_depth_topic').value
        self.expected_frame_id = self.get_parameter('expected_frame_id').value
        
        model_path = self.get_parameter('masker.model_path').value
        conf_threshold = float(self.get_parameter('masker.conf_threshold').value)
        dynamic_classes = list(self.get_parameter('masker.dynamic_classes').value or [])
        
        self.model = YOLOSegModel(model_path, conf_threshold, list(dynamic_classes))
        self.bridge = CvBridge()
        from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        self.masked_rgb_pub = self.create_publisher(Image, self.output_rgb_topic, qos)
        self.masked_depth_pub = self.create_publisher(Image, self.output_depth_topic, qos)

        self.rgb_sub = message_filters.Subscriber(self, Image, self.input_rgb_topic, qos_profile=qos)
        self.depth_sub = message_filters.Subscriber(self, Image, self.input_depth_topic, qos_profile=qos)
        
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub], queue_size=30, slop=0.05
        )
        self.ts.registerCallback(self.callback)
        
        domain_id = os.environ.get('ROS_DOMAIN_ID', '0')
        self.get_logger().info(f"[{self.instance_id}] Masker online. DOMAIN_ID={domain_id}")
        self.get_logger().info(f"  -> RGB: {self.input_rgb_topic} | Depth: {self.input_depth_topic}")
        self.get_logger().info(f"  -> Expect Frame: {self.expected_frame_id}")

    def callback(self, rgb_msg: Image, depth_msg: Image):
        # Strict Frame ID verification
        if self.expected_frame_id != 'UNDEFINED':
            if self.expected_frame_id not in rgb_msg.header.frame_id:
                self.get_logger().error(f"[{self.instance_id}] CROSSTALK: Got '{rgb_msg.header.frame_id}', expected '{self.expected_frame_id}'")
                return

        try:
            rgb_image = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8').copy()
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough').copy()
            
            mask = self.model.predict(rgb_image)
            masked_rgb = apply_semantic_mask(rgb_image, mask)
            
            masked_depth = depth_image.copy()
            masked_depth[mask > 0] = 0
            
            rgb_out = self.bridge.cv2_to_imgmsg(masked_rgb, encoding='bgr8')
            depth_out = self.bridge.cv2_to_imgmsg(masked_depth, encoding=depth_msg.encoding)
            
            rgb_out.header = rgb_msg.header
            depth_out.header = depth_msg.header
            
            self.masked_rgb_pub.publish(rgb_out)
            self.masked_depth_pub.publish(depth_out)
            
        except Exception as e:
            self.get_logger().error(f"[{self.instance_id}] Error: {str(e)}")

def main(args=None):
    rclpy.init(args=args)
    node = SemanticMaskerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
