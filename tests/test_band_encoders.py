import torch
from torch import nn

from higan_frequency.band_encoders import (BandFeatureFusion,
                                           HighFrequencyDirectionalEncoder,
                                           LowFrequencyGlyphEncoder)
from higan_frequency.frequency import FrequencyStyleEncoder
from lib.utils import yaml2config
from networks.BigGAN_networks import Generator
from networks.module import StyleEncoder
from pathlib import Path


ROOT = Path(__file__).parents[1]


def small_kwargs():
    return {
        "style_dim": 32,
        "base_channels": 4,
        "max_channels": 8,
        "norm_groups": 2,
        "init": "none",
    }


def test_low_encoder_contains_multiscale_glyph_kernels_and_dilation():
    encoder = LowFrequencyGlyphEncoder(
        kernel_sizes=(3, 5), dilation=2, **small_kwargs())
    convolutions = [module for module in encoder.modules() if isinstance(module, nn.Conv2d)]
    assert any(module.kernel_size == (5, 5) for module in convolutions)
    assert any(module.kernel_size == (3, 3) and module.dilation == (2, 2)
               for module in convolutions)


def test_high_encoder_contains_horizontal_and_vertical_directional_kernels():
    encoder = HighFrequencyDirectionalEncoder(
        directional_lengths=(3, 5, 7), **small_kwargs())
    kernels = {module.kernel_size for module in encoder.modules()
               if isinstance(module, nn.Conv2d)}
    for length in (3, 5, 7):
        assert (1, length) in kernels
        assert (length, 1) in kernels


def test_band_encoders_preserve_batch_shape_and_backpropagate():
    low = LowFrequencyGlyphEncoder(
        kernel_sizes=(3, 5), dilation=2, **small_kwargs())
    high = HighFrequencyDirectionalEncoder(
        directional_lengths=(3, 5), **small_kwargs())
    images = torch.randn(2, 1, 32, 48, requires_grad=True)
    lengths = torch.tensor([39, 48])
    low_style = low(images, lengths)
    high_style = high(images, lengths)
    assert low_style.shape == high_style.shape == (2, 32)
    (low_style.mean() + high_style.mean()).backward()
    assert images.grad is not None
    assert all(any(parameter.grad is not None for parameter in encoder.parameters())
               for encoder in (low, high))


def test_pixel_path_is_the_upstream_higan_style_encoder():
    module = FrequencyStyleEncoder(
        style_dim=32, share_wid=False, init="none",
        band_config={
            "base_channels": 4, "max_channels": 8, "norm_groups": 2,
            "init": "none", "low_kernel_sizes": [3, 5],
            "high_directional_lengths": [3, 5],
        })
    assert type(module.pixel_encoder) is StyleEncoder
    assert isinstance(module.low_encoder, LowFrequencyGlyphEncoder)
    assert isinstance(module.high_encoder, HighFrequencyDirectionalEncoder)
    assert isinstance(module.fusion, BandFeatureFusion)


def test_fusion_outputs_one_96d_style_and_couples_all_bands():
    fusion = BandFeatureFusion(style_dim=32, num_heads=4, hidden_dim=64)
    low = torch.randn(2, 32, requires_grad=True)
    pixel = torch.randn(2, 32, requires_grad=True)
    high = torch.randn(2, 32, requires_grad=True)
    output = fusion(low, pixel, high)
    assert output.shape == (2, 96)
    output[:, :32].square().mean().backward()
    for feature in (low, pixel, high):
        assert feature.grad is not None
        assert torch.count_nonzero(feature.grad) > 0


def test_all_three_conditioning_paths_receive_generator_gradients():
    encoder = FrequencyStyleEncoder(
        style_dim=32, resolution=4, max_dim=16, in_channel=1,
        share_wid=False, norm="none", init="none",
        band_config={
            "base_channels": 4, "max_channels": 8, "norm_groups": 2,
            "init": "none", "low_kernel_sizes": [3, 5],
            "low_dilation": 2, "high_directional_lengths": [3, 5],
        })
    cfg = yaml2config(ROOT / "configs" / "config.yaml")
    cfg.GenModel.G_ch = 4
    cfg.GenModel.G_attn = "64"
    cfg.GenModel.init = "none"
    generator = Generator(**cfg.GenModel)
    images = torch.randn(2, 1, 32, 32)
    style, _, _ = encoder(images, torch.tensor([32, 32]), vae_mode=True)
    latent = torch.cat((torch.randn(2, 32), style), dim=1)
    generated = generator(
        latent, torch.tensor([[1, 2], [3, 4]]), torch.tensor([2, 2]))
    generated.mean().backward()
    assert any(parameter.grad is not None for parameter in encoder.low_encoder.parameters())
    assert encoder.pixel_encoder.mu.weight.grad is not None
    assert any(parameter.grad is not None for parameter in encoder.high_encoder.parameters())
