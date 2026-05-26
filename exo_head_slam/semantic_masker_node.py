import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import message_filters
from cv_bridge import CvBridge
import numpy as np
import cv2
from typing import List, Optional
from .utils.vision_utils import apply_semantic_mask


class YOLOSegModel:
    """Real segmentation backend with an empty-mask fallback."""

    def __init__(self, model_path: str, conf_threshold: float, dynamic_classes: List[int]):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.dynamic_classes = dynamic_classes
        self.available = False
        self.error: Optional[Exception] = None
        self.model = None

        try:
            from ultralytics import YOLO
            # Resolve relative model paths against the repository root so users can
            # place models in the project root and reference them by name.
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
        """Return a binary mask where 1 marks dynamic pixels."""
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
                class_id = int(cls.item())
                if self.dynamic_classes and class_id not in self.dynamic_classes:
                    continue

                x1, y1, x2, y2 = box.detach().cpu().numpy().astype(int)
                x1 = max(0, min(x1, image.shape[1] - 1))
                y1 = max(0, min(y1, image.shape[0] - 1))
                x2 = max(0, min(x2, image.shape[1]))
                y2 = max(0, min(y2, image.shape[0]))
                mask[y1:y2, x1:x2] = 1

        return mask

class SemanticMaskerNode(Node):
    def __init__(self):
        super().__init__('semantic_masker_node')
        
        self.declare_parameter('input_rgb_topic', '')
        self.declare_parameter('input_depth_topic', '')
        self.declare_parameter('output_rgb_topic', '')
        self.declare_parameter('output_depth_topic', '')
        self.declare_parameter('masker.model_path', '')
        self.declare_parameter('masker.conf_threshold', 0.0)
        # Declare a non-empty integer list as default so rclpy treats this as INTEGER_ARRAY
        self.declare_parameter('masker.dynamic_classes', [0])
        
        self.input_rgb_topic = self.get_parameter('input_rgb_topic').value
        self.input_depth_topic = self.get_parameter('input_depth_topic').value
        self.output_rgb_topic = self.get_parameter('output_rgb_topic').value
        self.output_depth_topic = self.get_parameter('output_depth_topic').value
        model_path = self.get_parameter('masker.model_path').value
        conf_threshold = float(self.get_parameter('masker.conf_threshold').value)
        # Read dynamic classes robustly as a Python list
        dynamic_classes = list(self.get_parameter('masker.dynamic_classes').value or [])
        
        # Model
        self.model = YOLOSegModel(model_path, conf_threshold, list(dynamic_classes))
        self.bridge = CvBridge()
        
        # Publishers
        self.masked_rgb_pub = self.create_publisher(Image, self.output_rgb_topic, 10)
        self.masked_depth_pub = self.create_publisher(Image, self.output_depth_topic, 10)
        
        # Subscribers with synchronization
        self.rgb_sub = message_filters.Subscriber(self, Image, self.input_rgb_topic)
        self.depth_sub = message_filters.Subscriber(self, Image, self.input_depth_topic)
        
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub],
            queue_size=30,
            slop=0.5
        )
        self.ts.registerCallback(self.callback)
        
        if self.model.available:
            self.get_logger().info(f"Semantic Masker initialized with Ultralytics model: {model_path}")
        else:
            self.get_logger().warn(f"Semantic Masker fallback attivo: {self.model.error}")

    def callback(self, rgb_msg: Image, depth_msg: Image):
        # self.get_logger().debug(...) # Removed high-frequency log
        try:
            # Convert ROS messages to OpenCV images
            rgb_image = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
            
            # Inference
            mask = self.model.predict(rgb_image)
            
            # Apply mask to RGB
            masked_rgb = apply_semantic_mask(rgb_image, mask)
            
            # Depth mask: Typically we just pass the original depth if we want to keep it 
            # or mask it out too if it belongs to dynamic objects.
            # Here we mask depth too for SLAM consistency.
            masked_depth = depth_image.copy()
            masked_depth[mask > 0] = 0
            
            # Publish
            masked_rgb_msg = self.bridge.cv2_to_imgmsg(masked_rgb, encoding='bgr8')
            masked_depth_msg = self.bridge.cv2_to_imgmsg(masked_depth, encoding=depth_msg.encoding)
            
            masked_rgb_msg.header = rgb_msg.header
            masked_depth_msg.header = depth_msg.header
            
            self.masked_rgb_pub.publish(masked_rgb_msg)
            self.masked_depth_pub.publish(masked_depth_msg)
            
        except Exception as e:
            self.get_logger().error(f"Error in callback: {str(e)}")

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
