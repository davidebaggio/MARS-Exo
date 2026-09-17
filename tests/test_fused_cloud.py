import numpy as np
from builtin_interfaces.msg import Time

from exo_head_slam.extrinsic_solver_node import ExtrinsicSolverNode


class _Publisher:
    def publish(self, message):
        self.message = message


def test_rejected_pair_fuses_exo_only_and_never_fills_semantic_nan():
    node = ExtrinsicSolverNode.__new__(ExtrinsicSolverNode)
    node.publish_fused_cloud = True
    node.fused_cloud_min_depth = 0.15
    node.fused_cloud_max_depth = 5.0
    node.fused_cloud_voxel_size = 0.02
    node.depth_edge_rtol = 0.03
    node.exo_frame_id = 'exo_link'
    node._quality_metrics = {}
    node.fused_pcl_pub = _Publisher()

    raw = np.ones((4, 4), dtype=np.float32)
    raw[1, 1] = 0.0
    raw[2, 2] = np.nan
    prediction = np.ones_like(raw)
    confidence = np.ones(raw.shape, dtype=bool)
    color = np.zeros((*raw.shape, 3), dtype=np.uint8)
    intrinsic = np.array([
        [10.0, 0.0, 1.5], [0.0, 10.0, 1.5], [0.0, 0.0, 1.0],
    ])

    node.publish_fused_pointcloud(
        raw, raw, color, color, prediction, prediction,
        confidence, confidence, intrinsic, intrinsic,
        np.eye(4), np.eye(4), Time(), include_head=False,
    )

    assert node.fused_pcl_pub.message.header.frame_id == 'exo_link'
    assert node._quality_metrics['fused_head_included'] == 0
    assert node._quality_metrics['fused_sensor_points'] == 14
    assert node._quality_metrics['fused_fill_points'] == 1


def test_fused_cloud_can_be_disabled():
    node = ExtrinsicSolverNode.__new__(ExtrinsicSolverNode)
    node.publish_fused_cloud = False
    node.publish_fused_pointcloud(*([None] * 13), include_head=False)


def test_vggt_input_longest_side_is_512():
    node = ExtrinsicSolverNode.__new__(ExtrinsicSolverNode)
    tensor, *_ = node.preprocess_cv2_image(np.zeros((480, 640, 3), np.uint8))
    assert tensor.shape == (3, 384, 512)
