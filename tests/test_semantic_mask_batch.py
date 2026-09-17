import sys
import types

import torch

sys.modules.setdefault("cv_bridge", types.SimpleNamespace(CvBridge=object))

from exo_head_slam.semantic_masker_node import YOLOSegModel
from exo_head_slam.utils.vision_utils import dilate_mask


def test_result_mask_filters_confidence_and_classes_before_gpu_merge():
    result = types.SimpleNamespace(
        masks=types.SimpleNamespace(data=torch.tensor([
            [[1.0, 0.0], [0.0, 0.0]],
            [[0.0, 1.0], [0.0, 0.0]],
            [[0.0, 0.0], [1.0, 0.0]],
        ])),
        boxes=types.SimpleNamespace(
            conf=torch.tensor([0.30, 0.20, 0.40]),
            cls=torch.tensor([0.0, 0.0, 4.0]),
            xyxy=torch.empty(3, 4),
        ),
    )

    mask = YOLOSegModel._mask_tensor_from_result(result, (2, 2), 0.25, [0])

    assert mask.cpu().tolist() == [[1, 0], [0, 0]]


def test_mask_dilation_covers_uncertain_boundary():
    mask = torch.zeros(7, 7, dtype=torch.uint8).numpy()
    mask[3, 3] = 1
    assert int(dilate_mask(mask, 1).sum()) == 9
