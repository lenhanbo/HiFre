from pathlib import Path

import pytest
import torch

from lib.utils import yaml2config
from higan_frequency.frequency import FrequencyStyleEncoder
from networks.BigGAN_networks import Generator
from networks.generator import build_generator, validate_checkpoint_architecture
from networks.stylegan2_generator import (ModulatedConv2d, StyleGAN2GBlock,
                                          StyleGAN2Generator)

ROOT = Path(__file__).parents[1]


def config():
    cfg = yaml2config(ROOT / "configs" / "config.yaml")
    cfg.architecture.generator_variant = "stylegan2"
    cfg.GenModel.init = "none"
    cfg.GenModel.G_ch = 4
    cfg.GenModel.G_attn = "64"
    return cfg


def test_factory_builds_both_variants_and_rejects_unknown_variant():
    cfg = config()
    assert isinstance(build_generator(cfg), StyleGAN2Generator)
    cfg.architecture.generator_variant = "baseline"
    assert type(build_generator(cfg.GenModel, cfg.architecture)) is Generator
    cfg.architecture.generator_variant = "unknown"
    with pytest.raises(ValueError, match="unknown generator variant"):
        build_generator(cfg.GenModel, cfg.architecture)


def test_modulated_conv_has_per_input_channel_modulation_and_demodulation():
    layer = ModulatedConv2d(4, 6, 32, kernel_size=3, padding=1)
    value = torch.randn(2, 4, 5, 7, requires_grad=True)
    style = torch.randn(2, 32, requires_grad=True)
    output = layer(value, style)
    assert output.shape == (2, 6, 5, 7)
    assert layer.modulation.weight.shape == (4, 32)
    assert layer.demodulate is True
    output.mean().backward()
    assert style.grad is not None
    assert layer.weight.grad is not None
    assert layer.modulation.weight.grad is not None


def test_stylegan2_block_has_no_batch_norm_and_keeps_residual_shape():
    block = StyleGAN2GBlock(
        8, 4, 32,
        which_shortcut_conv=lambda left, right, **kwargs: torch.nn.Conv2d(
            left, right, kwargs.get("kernel_size", 3), padding=kwargs.get("padding", 1)),
        activation=torch.nn.ReLU(),
        upsample=lambda value: torch.nn.functional.interpolate(value, scale_factor=2),
    )
    assert not any(isinstance(module, torch.nn.BatchNorm2d) for module in block.modules())
    assert isinstance(block.conv1, ModulatedConv2d)
    assert isinstance(block.conv2, ModulatedConv2d)
    output = block(torch.randn(2, 8, 4, 7), torch.randn(2, 32))
    assert output.shape == (2, 4, 8, 14)


def test_generator_accumulates_three_non_demodulated_skip_to_image_paths():
    generator = build_generator(config()).eval()
    assert len(generator.to_images) == 3
    assert all(layer.demodulate is False for layer in generator.to_images)
    latent = torch.randn(2, 128)
    output = generator(
        latent, torch.tensor([[1, 2], [3, 4]]), torch.tensor([2, 2]))
    assert output.shape == (2, 1, 32, 32)


def test_style_chunks_reach_scales_8_16_32_and_modulations_receive_gradients():
    cfg = config()
    generator = build_generator(cfg).train()
    captured = []
    hooks = [blocklist[0].register_forward_pre_hook(
        lambda module, inputs: captured.append(inputs[1].detach().clone()))
        for blocklist in generator.blocks]
    latent = torch.randn(2, 128, requires_grad=True)
    labels = torch.tensor([[1, 2], [3, 4]])
    lengths = torch.tensor([2, 2])
    generator(latent, labels, lengths).mean().backward()
    for hook in hooks:
        hook.remove()
    assert len(captured) == 3
    for index, condition in enumerate(captured, start=1):
        torch.testing.assert_close(condition, latent.detach()[:, index * 32:(index + 1) * 32])
    for blocklist in generator.blocks:
        block = blocklist[0]
        assert block.conv1.modulation.weight.grad is not None
        assert block.conv2.modulation.weight.grad is not None
    assert latent.grad is not None


def test_frequency_encoder_and_all_modulations_receive_end_to_end_gradients():
    class TrainableBandEncoder(torch.nn.Module):
        style_dim = 32

        def __init__(self):
            super().__init__()
            self.gain = torch.nn.Parameter(torch.ones(32))

        def forward(self, image, lengths, backbone=None, vae_mode=False):
            value = image.mean((1, 2, 3))[:, None] * self.gain[None, :]
            return (value, value, torch.zeros_like(value)) if vae_mode else value

    cfg = config()
    generator = build_generator(cfg).train()
    encoder = FrequencyStyleEncoder(style_dim=32, share_wid=False, init="none")
    encoder.encoder = TrainableBandEncoder()
    images = torch.randn(2, 1, 32, 32, requires_grad=True)
    style = encoder(images, torch.tensor([32, 32]))
    latent = torch.cat((torch.randn(2, 32), style), dim=1)
    labels = torch.tensor([[1, 2], [3, 4]])
    generator(latent, labels, torch.tensor([2, 2])).mean().backward()
    assert encoder.encoder.gain.grad is not None
    for blocklist in generator.blocks:
        assert blocklist[0].conv1.modulation.weight.grad is not None
        assert blocklist[0].conv2.modulation.weight.grad is not None


def test_each_style_chunk_changes_only_its_corresponding_modulation():
    cfg = config()
    generator = build_generator(cfg).eval()
    base = torch.zeros(2, 128)
    for target in range(3):
        changed = base.clone()
        changed[:, (target + 1) * 32:(target + 2) * 32] = 1
        for index, blocklist in enumerate(generator.blocks):
            modulation = blocklist[0].conv1.modulation
            base_value = modulation(base[:, (index + 1) * 32:(index + 2) * 32])
            changed_value = modulation(changed[:, (index + 1) * 32:(index + 2) * 32])
            assert torch.equal(base_value, changed_value) is (index != target)


def test_checkpoint_architecture_metadata_rules():
    cfg = config()
    stylegan2 = {"architecture": {
        "generator_variant": "stylegan2",
        "stylegan2": {
            "demodulate": True,
            "spatial_noise": False,
            "mapping_network": False,
            "skip_to_image": True,
        },
    }}
    validate_checkpoint_architecture(stylegan2, cfg.architecture)
    cfg.architecture.generator_variant = "baseline"
    with pytest.raises(ValueError, match="variant mismatch"):
        validate_checkpoint_architecture(stylegan2, cfg.architecture)
    validate_checkpoint_architecture({}, cfg.architecture)
    cfg.architecture.generator_variant = "stylegan2"
    with pytest.raises(ValueError, match="legacy checkpoint"):
        validate_checkpoint_architecture({}, cfg.architecture)
