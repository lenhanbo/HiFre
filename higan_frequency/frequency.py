from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from higan_frequency.band_encoders import (BandFeatureFusion,
                                           HighFrequencyDirectionalEncoder,
                                           LowFrequencyGlyphEncoder)
from networks.module import StyleEncoder


FREQUENCY_ENCODER_VARIANT = "band_fusion_v1"


def _pad_right(image: torch.Tensor, width: int, mode: str) -> torch.Tensor:
    amount = width - image.shape[-1]
    if amount <= 0:
        return image
    while amount:
        if image.shape[-1] <= 1:
            return F.pad(image, (0, amount), mode="replicate")
        step = min(amount, image.shape[-1] - 1)
        image = F.pad(image, (0, step), mode=mode)
        amount -= step
    return image


def split_frequency_bands(images: torch.Tensor, image_lens: torch.Tensor,
                          cutoff: float = 0.06,
                          padding_mode: str = "reflect") -> tuple[torch.Tensor, torch.Tensor]:
    """Split each valid word image into differentiable Gaussian low/high bands."""
    if images.ndim != 4:
        raise ValueError("images must have shape [batch, channels, height, width]")
    if len(image_lens) != len(images):
        raise ValueError("image_lens must contain one width per image")
    if cutoff <= 0:
        raise ValueError("cutoff must be positive")
    if padding_mode != "reflect":
        raise ValueError("frequency.padding_mode must be 'reflect'")

    source = images.float()
    canvas_width = source.shape[-1]
    # Copy all lengths to the host in one operation. Calling Tensor.item() for
    # every sample serializes CUDA work once per image and is particularly
    # expensive because this function is used several times per train step.
    widths = [int(width) for width in image_lens.detach().cpu().tolist()]
    groups: dict[int, list[int]] = {}
    for index, width in enumerate(widths):
        if width < 1 or width > canvas_width:
            raise ValueError(f"invalid image width {width} for canvas {canvas_width}")
        groups.setdefault(width, []).append(index)

    # Words with the same valid width can share a batched FFT and one frequency
    # mask. Generated words have widths in multiples of char_width, so this
    # removes many small sequential FFT launches without changing the result.
    lows: list[torch.Tensor | None] = [None] * len(source)
    highs: list[torch.Tensor | None] = [None] * len(source)
    for width, indices in groups.items():
        batch_indices = torch.as_tensor(indices, device=source.device, dtype=torch.long)
        valid = source.index_select(0, batch_indices)[..., :width]
        height = valid.shape[-2]
        fy = torch.fft.fftfreq(height, device=valid.device, dtype=torch.float32)
        fx = torch.fft.fftfreq(width, device=valid.device, dtype=torch.float32)
        distance2 = (fy[:, None] / cutoff).square() + (fx[None, :] / cutoff).square()
        low_mask = torch.exp(-0.5 * distance2)[None, None, ...]
        spectrum = torch.fft.fft2(valid, dim=(-2, -1))
        low = torch.fft.ifft2(spectrum * low_mask, dim=(-2, -1)).real
        high = valid - low
        for group_index, source_index in enumerate(indices):
            lows[source_index] = _pad_right(low[group_index], canvas_width, padding_mode)
            highs[source_index] = _pad_right(high[group_index], canvas_width, padding_mode)
    return torch.stack(lows), torch.stack(highs)  # type: ignore[arg-type]


class FrequencyStyleEncoder(nn.Module):
    """Band-specific low/pixel/high encoders with a HiGAN pixel-style anchor."""

    def __init__(self, cutoff: float = 0.06, padding_mode: str = "reflect",
                 band_config=None, **encoder_kwargs):
        super().__init__()
        if int(encoder_kwargs.get("style_dim", 32)) != 32:
            raise ValueError("FrequencyStyleEncoder requires EncModel.style_dim=32")
        self.cutoff = float(cutoff)
        self.padding_mode = str(padding_mode)
        self.pixel_encoder = StyleEncoder(**encoder_kwargs)
        settings = dict(band_config or {})
        common = {
            "style_dim": 32,
            "base_channels": int(settings.get("base_channels", 32)),
            "max_channels": int(settings.get("max_channels", 128)),
            "norm_groups": int(settings.get("norm_groups", 8)),
            "init": str(settings.get("init", "N02")),
        }
        self.low_encoder = LowFrequencyGlyphEncoder(
            kernel_sizes=tuple(settings.get("low_kernel_sizes", (3, 5))),
            dilation=int(settings.get("low_dilation", 2)), **common)
        self.high_encoder = HighFrequencyDirectionalEncoder(
            directional_lengths=tuple(
                settings.get("high_directional_lengths", (3, 5, 7))), **common)
        self.fusion = BandFeatureFusion(
            style_dim=32,
            num_heads=int(settings.get("fusion_heads", 4)),
            hidden_dim=int(settings.get("fusion_hidden_dim", 192)),
            dropout=float(settings.get("fusion_dropout", 0.0)),
        )

    @property
    def style_dim(self) -> int:
        return self.pixel_encoder.style_dim * 3

    def encode_pixel(self, images: torch.Tensor, image_lens: torch.Tensor,
                     wid_cnn_backbone=None, vae_mode: bool = False):
        """Run the unchanged HiGAN StyleEncoder on the original pixel image."""
        return self.pixel_encoder(
            images.float(), image_lens, wid_cnn_backbone, vae_mode=vae_mode)

    def forward(self, images: torch.Tensor, image_lens: torch.Tensor,
                wid_cnn_backbone=None, vae_mode: bool = False,
                frequency_bands: tuple[torch.Tensor, torch.Tensor] | None = None):
        if frequency_bands is None:
            low, high = split_frequency_bands(
                images, image_lens, cutoff=self.cutoff, padding_mode=self.padding_mode)
        else:
            low, high = frequency_bands
            if low.shape != images.shape or high.shape != images.shape:
                raise ValueError("precomputed frequency bands must match images")
        low_style = self.low_encoder(low, image_lens)
        high_style = self.high_encoder(high, image_lens)
        if not vae_mode:
            pixel_style = self.encode_pixel(
                images, image_lens, wid_cnn_backbone, vae_mode=False)
            return self.fusion(low_style, pixel_style, high_style)
        pixel_z, pixel_mu, pixel_logvar = self.encode_pixel(
            images, image_lens, wid_cnn_backbone, vae_mode=True)
        fused_style = self.fusion(low_style, pixel_z, high_style)
        return fused_style, pixel_mu, pixel_logvar
