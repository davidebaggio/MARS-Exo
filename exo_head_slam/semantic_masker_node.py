import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import Image
import message_filters
from cv_bridge import CvBridge
import numpy as np
import os
import torch
import torch.nn.functional as F
from typing import List, Optional
from .utils.vision_utils import apply_semantic_mask


class YOLOSegModel:
    def __init__(self, model_path: str):
        self.model_path = model_path
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

    def predict(
        self,
        images: List[np.ndarray],
        conf_thresholds: List[float],
        dynamic_classes: List[List[int]],
    ) -> List[np.ndarray]:
        if not self.available:
            return [np.zeros(image.shape[:2], dtype=np.uint8) for image in images]

        results = self.model.predict(source=images, conf=min(conf_thresholds), verbose=False)
        if len(results) != len(images):
            return [np.zeros(image.shape[:2], dtype=np.uint8) for image in images]
        masks = [
            self._mask_tensor_from_result(result, image.shape[:2], threshold, classes)
            for result, image, threshold, classes in zip(
                results, images, conf_thresholds, dynamic_classes
            )
        ]
        flat_masks = torch.cat([mask.flatten() for mask in masks]).cpu().numpy()
        output = []
        offset = 0
        for mask, image in zip(masks, images):
            size = mask.numel()
            output.append(flat_masks[offset:offset + size].reshape(image.shape[:2]))
            offset += size
        return output

    @staticmethod
    def _mask_tensor_from_result(result, image_shape, conf_threshold, dynamic_classes):
        source = (
            result.boxes.cls if result.boxes is not None
            else result.masks.data if result.masks is not None
            else torch.empty(0)
        )
        device = source.device
        mask = torch.zeros(image_shape, dtype=torch.uint8, device=device)
        if result.boxes is None or result.masks is None:
            return mask

        keep = result.boxes.conf >= conf_threshold
        if dynamic_classes:
            keep &= torch.isin(
                result.boxes.cls.to(torch.int64),
                torch.tensor(dynamic_classes, device=result.boxes.cls.device),
            )
        if not keep.any():
            return mask

        masks = F.interpolate(
            result.masks.data[keep].unsqueeze(1),
            size=image_shape,
            mode="bilinear",
            align_corners=False,
        )
        return (masks.amax(dim=0).squeeze(0) > 0.5).to(torch.uint8)


class SemanticMaskerNode(Node):
    def __init__(self):
        super().__init__('semantic_masker_node')
        
        for camera in ('head', 'exo'):
            self.declare_parameter(f'{camera}.input_rgb_topic', 'UNDEFINED')
            self.declare_parameter(f'{camera}.input_depth_topic', 'UNDEFINED')
            self.declare_parameter(f'{camera}.output_rgb_topic', 'UNDEFINED')
            self.declare_parameter(f'{camera}.output_depth_topic', 'UNDEFINED')
            self.declare_parameter(f'{camera}.conf_threshold', 0.25)
            self.declare_parameter(f'{camera}.dynamic_classes', [0])
        self.declare_parameter('masker.model_path', 'yolov8n-seg.pt')
        self.declare_parameter('sync_slop', 0.035)

        self.topics = {
            camera: {
                kind: self.get_parameter(f'{camera}.{kind}_topic').value
                for kind in ('input_rgb', 'input_depth', 'output_rgb', 'output_depth')
            }
            for camera in ('head', 'exo')
        }
        self.conf_thresholds = [
            float(self.get_parameter(f'{camera}.conf_threshold').value)
            for camera in ('head', 'exo')
        ]
        self.dynamic_classes = [
            list(self.get_parameter(f'{camera}.dynamic_classes').value or [])
            for camera in ('head', 'exo')
        ]

        self.model = YOLOSegModel(self.get_parameter('masker.model_path').value)
        self.bridge = CvBridge()
        from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
        input_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        output_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        self.masked_publishers = {
            camera: (
                self.create_publisher(Image, topics['output_rgb'], output_qos),
                self.create_publisher(Image, topics['output_depth'], output_qos),
            )
            for camera, topics in self.topics.items()
        }
        self.subscribers = [
            message_filters.Subscriber(self, Image, self.topics[camera][kind], qos_profile=input_qos)
            for camera in ('head', 'exo')
            for kind in ('input_rgb', 'input_depth')
        ]
        
        sync_slop = self.get_parameter('sync_slop').value
        self.ts = message_filters.ApproximateTimeSynchronizer(
            # Using queue_size=2 to avoid backlog buildup and lag
            self.subscribers, queue_size=2, slop=sync_slop
        )
        self.ts.registerCallback(self.callback)
        
        domain_id = os.environ.get('ROS_DOMAIN_ID', '0')
        self.get_logger().info(f"Batched masker online. DOMAIN_ID={domain_id}")

    def callback(self, h_rgb: Image, h_depth: Image, e_rgb: Image, e_depth: Image):
        try:
            rgb_images = [
                self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                for msg in (h_rgb, e_rgb)
            ]
            depth_images = [
                self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
                for msg in (h_depth, e_depth)
            ]
            masks = self.model.predict(rgb_images, self.conf_thresholds, self.dynamic_classes)

            for camera, rgb_msg, depth_msg, rgb_image, depth_image, mask in zip(
                ('head', 'exo'),
                (h_rgb, e_rgb),
                (h_depth, e_depth),
                rgb_images,
                depth_images,
                masks,
            ):
                masked_depth = depth_image.copy()
                masked_depth[mask > 0] = 0
                rgb_out = self.bridge.cv2_to_imgmsg(apply_semantic_mask(rgb_image, mask), encoding='bgr8')
                depth_out = self.bridge.cv2_to_imgmsg(masked_depth, encoding=depth_msg.encoding)
                rgb_out.header = rgb_msg.header
                depth_out.header = depth_msg.header
                self.masked_publishers[camera][0].publish(rgb_out)
                self.masked_publishers[camera][1].publish(depth_out)
        except Exception as e:
            self.get_logger().error(f"Error: {str(e)}")

def main(args=None):
    rclpy.init(args=args)
    node = SemanticMaskerNode()
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
