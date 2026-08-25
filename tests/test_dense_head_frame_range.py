import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "vggt-omega"))

from vggt_omega.models.heads.dense_head import DenseHead


def test_dense_head_decodes_only_requested_frames():
    torch.manual_seed(0)
    head = DenseHead(
        dim_in=8,
        patch_size=4,
        features=4,
        out_channels=[4, 4, 4, 4],
        intermediate_layer_idx=[0, 1, 2, 3],
    ).eval()
    images = torch.zeros(1, 4, 3, 8, 8)
    tokens = [torch.randn(1, 4, 5, 8) for _ in range(4)]

    with torch.no_grad():
        full_depth, full_conf = head(tokens, images, patch_token_start=1)
        depth, conf = head(tokens, images, patch_token_start=1, frame_range=(2, 4))

    torch.testing.assert_close(depth, full_depth[:, 2:])
    torch.testing.assert_close(conf, full_conf[:, 2:])
