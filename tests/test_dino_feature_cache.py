import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).parents[1] / "vggt-omega"))

from vggt_omega.models.aggregator import Aggregator


class _PatchEmbed(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, images):
        self.calls += 1
        return images.flatten(2).transpose(1, 2)[..., :2]


class _IdentityBlock(nn.Module):
    def forward(self, tokens, _rope):
        return tokens


class _Rope(nn.Module):
    def forward(self, H, W):
        return torch.zeros(H * W, 1), torch.zeros(H * W, 1)


def test_cached_dino_features_bypass_recomputation():
    aggregator = Aggregator.__new__(Aggregator)
    nn.Module.__init__(aggregator)
    aggregator.patch_embed = _PatchEmbed()
    aggregator.rope_embed = _Rope()
    aggregator.frame_blocks = nn.ModuleList([_IdentityBlock()])
    aggregator.inter_frame_blocks = nn.ModuleList([_IdentityBlock()])
    aggregator.depth = 1
    aggregator.patch_size = 2
    aggregator.cached_layer_indices = {0}
    aggregator.inter_frame_attention_types = ["global"]
    aggregator.patch_token_start = 2
    aggregator.camera_token = nn.Parameter(torch.zeros(1, 2, 1, 2))
    aggregator.register_token = nn.Parameter(torch.zeros(1, 2, 1, 2))
    aggregator.register_buffer("_resnet_mean", torch.zeros(1, 1, 3, 1, 1))
    aggregator.register_buffer("_resnet_std", torch.ones(1, 1, 3, 1, 1))
    images = torch.arange(96, dtype=torch.float32).view(1, 2, 3, 4, 4)

    features = aggregator.extract_dino_features(images)
    expected, _ = aggregator(images)
    actual, _ = aggregator(images, dino_features=features)

    assert aggregator.patch_embed.calls == 2
    assert torch.equal(actual[0], expected[0])
