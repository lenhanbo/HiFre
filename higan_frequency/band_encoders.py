"""Band-specific style encoders used only by frequency conditioning."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn

from networks.utils import init_weights


def _group_norm(channels: int, maximum_groups: int) -> nn.GroupNorm:
    groups = min(int(maximum_groups), int(channels))
    while channels % groups:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class MaskedStyleHead(nn.Module):
    """Pool variable-width features and project them to one style vector."""

    def __init__(self, channels: int, style_dim: int, width_scale: int = 16):
        super().__init__()
        self.width_scale = int(width_scale)
        self.projection = nn.Sequential(
            nn.Linear(channels, channels),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Linear(channels, style_dim),
        )

    def forward(self, feature: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        feature = feature.mean(dim=2)
        valid = torch.div(
            lengths.to(feature.device, dtype=torch.long) + self.width_scale - 1,
            self.width_scale, rounding_mode="floor").clamp_(1, feature.shape[-1])
        positions = torch.arange(feature.shape[-1], device=feature.device)[None, :]
        mask = (positions < valid[:, None]).to(feature.dtype).unsqueeze(1)
        pooled = (feature * mask).sum(dim=-1) / valid.to(feature.dtype).unsqueeze(1)
        return self.projection(pooled)


class GlyphBlock(nn.Module):
    """Multi-receptive-field 2-D block for low-frequency glyph structure."""

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_sizes: Sequence[int], dilation: int, norm_groups: int):
        super().__init__()
        kernels = tuple(int(kernel) for kernel in kernel_sizes)
        if not kernels or any(kernel < 3 or kernel % 2 == 0 for kernel in kernels):
            raise ValueError("low glyph kernels must be non-empty odd values >= 3")
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel, stride=2,
                          padding=kernel // 2),
                _group_norm(out_channels, norm_groups),
                nn.LeakyReLU(0.2, inplace=False),
            )
            for kernel in kernels
        ])
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=2,
                      padding=dilation, dilation=dilation),
            _group_norm(out_channels, norm_groups),
            nn.LeakyReLU(0.2, inplace=False),
        ))
        self.aggregate = nn.Sequential(
            nn.Conv2d(out_channels * len(self.branches), out_channels, 1),
            _group_norm(out_channels, norm_groups),
        )
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1, stride=2)
        self.activation = nn.LeakyReLU(0.2, inplace=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        branches = torch.cat([branch(value) for branch in self.branches], dim=1)
        return self.activation(self.aggregate(branches) + self.shortcut(value))


class LowFrequencyGlyphEncoder(nn.Module):
    """Encode smooth glyph morphology with multi-scale two-dimensional kernels."""

    def __init__(self, style_dim=32, base_channels=32, max_channels=128,
                 kernel_sizes=(3, 5), dilation=2, norm_groups=8, init="N02"):
        super().__init__()
        self.style_dim = int(style_dim)
        base_channels = int(base_channels)
        max_channels = int(max_channels)
        self.stem = nn.Sequential(
            nn.Conv2d(1, base_channels, 5, padding=2),
            _group_norm(base_channels, norm_groups),
            nn.LeakyReLU(0.2, inplace=False),
        )
        channels = [
            base_channels,
            min(base_channels * 2, max_channels),
            min(base_channels * 4, max_channels),
            max_channels,
        ]
        blocks = []
        in_channels = base_channels
        for out_channels in channels:
            blocks.append(GlyphBlock(
                in_channels, out_channels, kernel_sizes, int(dilation), int(norm_groups)))
            in_channels = out_channels
        self.blocks = nn.Sequential(*blocks)
        self.style_head = MaskedStyleHead(channels[-1], self.style_dim)
        if init != "none":
            init_weights(self, init)

    def forward(self, image, lengths, backbone=None, vae_mode=False):
        return self.style_head(self.blocks(self.stem(image.float())), lengths)


class DirectionalBlock(nn.Module):
    """Aggregate horizontal and vertical high-frequency responses."""

    def __init__(self, in_channels: int, out_channels: int,
                 lengths: Sequence[int], norm_groups: int):
        super().__init__()
        directional_lengths = tuple(int(length) for length in lengths)
        if not directional_lengths or any(
                length < 3 or length % 2 == 0 for length in directional_lengths):
            raise ValueError("directional kernel lengths must be non-empty odd values >= 3")
        branch_count = len(directional_lengths) * 2
        branch_channels = max(4, math.ceil(out_channels / branch_count))
        branches = []
        for length in directional_lengths:
            branches.extend([
                nn.Conv2d(in_channels, branch_channels, (1, length), stride=2,
                          padding=(0, length // 2)),
                nn.Conv2d(in_channels, branch_channels, (length, 1), stride=2,
                          padding=(length // 2, 0)),
            ])
        self.branches = nn.ModuleList(branches)
        self.aggregate = nn.Sequential(
            nn.Conv2d(branch_channels * branch_count, out_channels, 1),
            _group_norm(out_channels, norm_groups),
        )
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1, stride=2)
        self.activation = nn.LeakyReLU(0.2, inplace=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        branches = torch.cat([branch(value) for branch in self.branches], dim=1)
        return self.activation(self.aggregate(branches) + self.shortcut(value))


class HighFrequencyDirectionalEncoder(nn.Module):
    """Encode oriented stroke details with 1xL and Lx1 kernels."""

    def __init__(self, style_dim=32, base_channels=32, max_channels=128,
                 directional_lengths=(3, 5, 7), norm_groups=8, init="N02"):
        super().__init__()
        self.style_dim = int(style_dim)
        base_channels = int(base_channels)
        max_channels = int(max_channels)
        self.stem = nn.Sequential(
            nn.Conv2d(1, base_channels, 3, padding=1),
            _group_norm(base_channels, norm_groups),
            nn.LeakyReLU(0.2, inplace=False),
        )
        channels = [
            base_channels,
            min(base_channels * 2, max_channels),
            min(base_channels * 4, max_channels),
            max_channels,
        ]
        blocks = []
        in_channels = base_channels
        for out_channels in channels:
            blocks.append(DirectionalBlock(
                in_channels, out_channels, directional_lengths, int(norm_groups)))
            in_channels = out_channels
        self.blocks = nn.Sequential(*blocks)
        self.style_head = MaskedStyleHead(channels[-1], self.style_dim)
        if init != "none":
            init_weights(self, init)

    def forward(self, image, lengths, backbone=None, vae_mode=False):
        return self.style_head(self.blocks(self.stem(image.float())), lengths)


class BandFeatureFusion(nn.Module):
    """Fuse low, pixel, and high features before hierarchical conditioning."""

    def __init__(self, style_dim=32, num_heads=4, hidden_dim=192, dropout=0.0):
        super().__init__()
        style_dim = int(style_dim)
        if style_dim % int(num_heads):
            raise ValueError("fusion style_dim must be divisible by num_heads")
        self.style_dim = style_dim
        self.output_dim = style_dim * 3
        self.band_embedding = nn.Parameter(torch.zeros(1, 3, style_dim))
        nn.init.normal_(self.band_embedding, std=0.02)
        self.attention = nn.MultiheadAttention(
            style_dim, int(num_heads), dropout=float(dropout), batch_first=True)
        self.norm1 = nn.LayerNorm(style_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(style_dim, style_dim * 4),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(style_dim * 4, style_dim),
        )
        self.norm2 = nn.LayerNorm(style_dim)
        self.fusion = nn.Sequential(
            nn.Linear(self.output_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), self.output_dim),
        )
        self.output_norm = nn.LayerNorm(self.output_dim)

    def forward(self, low: torch.Tensor, pixel: torch.Tensor,
                high: torch.Tensor) -> torch.Tensor:
        tokens = torch.stack((low, pixel, high), dim=1) + self.band_embedding
        attended, _ = self.attention(tokens, tokens, tokens, need_weights=False)
        tokens = self.norm1(tokens + attended)
        tokens = self.norm2(tokens + self.feed_forward(tokens))
        flattened = tokens.flatten(1)
        return self.output_norm(flattened + self.fusion(flattened))


__all__ = [
    "BandFeatureFusion", "GlyphBlock", "HighFrequencyDirectionalEncoder",
    "LowFrequencyGlyphEncoder", "MaskedStyleHead", "DirectionalBlock",
]
