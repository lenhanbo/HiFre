from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from networks.module import StyleEncoder


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
    lows, highs = [], []
    for image, raw_width in zip(source, image_lens):
        width = int(raw_width.item())
        if width < 1 or width > canvas_width:
            raise ValueError(f"invalid image width {width} for canvas {canvas_width}")
        valid = image[..., :width]
        height = valid.shape[-2]
        fy = torch.fft.fftfreq(height, device=valid.device, dtype=torch.float32)
        fx = torch.fft.fftfreq(width, device=valid.device, dtype=torch.float32)
        distance2 = (fy[:, None] / cutoff).square() + (fx[None, :] / cutoff).square()
        low_mask = torch.exp(-0.5 * distance2)[None, ...]
        spectrum = torch.fft.fft2(valid, dim=(-2, -1))
        low = torch.fft.ifft2(spectrum * low_mask, dim=(-2, -1)).real
        high = valid - low
        lows.append(_pad_right(low, canvas_width, padding_mode))
        highs.append(_pad_right(high, canvas_width, padding_mode))
    return torch.stack(lows), torch.stack(highs)


class FrequencyStyleEncoder(nn.Module):
    """One shared 32-D HiGAN StyleEncoder applied low/original/high in order."""

    def __init__(self, cutoff: float = 0.06, padding_mode: str = "reflect", **encoder_kwargs):
        super().__init__()
        if int(encoder_kwargs.get("style_dim", 32)) != 32:
            raise ValueError("FrequencyStyleEncoder requires EncModel.style_dim=32")
        self.cutoff = float(cutoff)
        self.padding_mode = str(padding_mode)
        self.encoder = StyleEncoder(**encoder_kwargs)

    @property
    def style_dim(self) -> int:
        return self.encoder.style_dim * 3

    def forward(self, images: torch.Tensor, image_lens: torch.Tensor,
                wid_cnn_backbone=None, vae_mode: bool = False):
        low, high = split_frequency_bands(
            images, image_lens, cutoff=self.cutoff, padding_mode=self.padding_mode)
        ordered = (low, images.float(), high)
        if not vae_mode:
            return torch.cat([
                self.encoder(band, image_lens, wid_cnn_backbone, vae_mode=False)
                for band in ordered
            ], dim=1)
        encoded = [
            self.encoder(band, image_lens, wid_cnn_backbone, vae_mode=True)
            for band in ordered
        ]
        return tuple(torch.cat([item[index] for item in encoded], dim=1) for index in range(3))
