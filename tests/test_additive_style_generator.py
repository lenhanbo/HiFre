from pathlib import Path

import pytest
import torch
from torch import nn

from lib.utils import yaml2config
from networks.BigGAN_networks import Generator
from networks.generator import (AdditiveStyleGBlock, AdditiveStyleGenerator,
                                build_generator, validate_checkpoint_architecture)


ROOT = Path(__file__).parents[1]


def config():
    cfg = yaml2config(ROOT / "configs" / "config.yaml")
    cfg.GenModel.init = "none"
    cfg.GenModel.G_ch = 4
    cfg.GenModel.G_attn = "64"
    return cfg


def test_factory_builds_both_variants_and_rejects_unknown_variant():
    cfg = config()
    additive = build_generator(cfg)
    assert isinstance(additive, AdditiveStyleGenerator)
    cfg.architecture.generator_variant = "baseline"
    assert type(build_generator(cfg.GenModel, cfg.architecture)) is Generator
    cfg.architecture.generator_variant = "unknown"
    with pytest.raises(ValueError, match="unknown generator variant"):
        build_generator(cfg.GenModel, cfg.architecture)


def test_additive_block_has_plain_batch_norm_two_affines_and_residual_shape():
    block = AdditiveStyleGBlock(
        8, 4, 32,
        which_conv1=lambda left, right, **kwargs: nn.Conv2d(
            left, right, kwargs.get("kernel_size", 3), padding=kwargs.get("padding", 1)),
        which_conv2=lambda left, right, **kwargs: nn.Conv2d(left, right, 3, padding=1),
        which_linear=nn.Linear, activation=nn.ReLU(),
        upsample=lambda value: nn.functional.interpolate(value, scale_factor=2),
    )
    assert type(block.bn1) is nn.BatchNorm2d
    assert type(block.bn2) is nn.BatchNorm2d
    assert isinstance(block.affine1, nn.Linear)
    assert isinstance(block.affine2, nn.Linear)
    output = block(torch.randn(2, 8, 4, 7), torch.randn(2, 32))
    assert output.shape == (2, 4, 8, 14)


def test_style_chunks_reach_scales_8_16_32_and_affines_receive_gradients():
    cfg = config()
    generator = build_generator(cfg.GenModel, cfg.architecture).train()
    captured = []
    hooks = [blocklist[0].register_forward_pre_hook(
        lambda module, inputs: captured.append(inputs[1].detach().clone()))
        for blocklist in generator.blocks]
    latent = torch.randn(2, 128, requires_grad=True)
    labels = torch.tensor([[1, 2], [3, 4]])
    lengths = torch.tensor([2, 2])
    output = generator(latent, labels, lengths)
    output.mean().backward()
    for hook in hooks:
        hook.remove()
    assert len(captured) == 3
    for index, condition in enumerate(captured, start=1):
        torch.testing.assert_close(condition, latent.detach()[:, index * 32:(index + 1) * 32])
    for blocklist in generator.blocks:
        block = blocklist[0]
        assert block.affine1.weight.grad is not None
        assert block.affine2.weight.grad is not None
    assert latent.grad is not None


def test_each_style_chunk_changes_only_its_corresponding_block_projection():
    cfg = config()
    generator = build_generator(cfg.GenModel, cfg.architecture).eval()
    base = torch.zeros(2, 128)
    for target in range(3):
        changed = base.clone()
        changed[:, (target + 1) * 32:(target + 2) * 32] = 1
        for index, blocklist in enumerate(generator.blocks):
            block = blocklist[0]
            base_projection = block.affine1(base[:, (index + 1) * 32:(index + 2) * 32])
            changed_projection = block.affine1(
                changed[:, (index + 1) * 32:(index + 2) * 32])
            assert torch.equal(base_projection, changed_projection) is (index != target)


def test_checkpoint_architecture_metadata_rules():
    cfg = config()
    additive = {"architecture": {
        "generator_variant": "additive_style",
        "additive_style": {"injection_strength": 1.0},
    }}
    validate_checkpoint_architecture(additive, cfg.architecture)
    cfg.architecture.generator_variant = "baseline"
    with pytest.raises(ValueError, match="variant mismatch"):
        validate_checkpoint_architecture(additive, cfg.architecture)
    validate_checkpoint_architecture({}, cfg.architecture)
    cfg.architecture.generator_variant = "additive_style"
    with pytest.raises(ValueError, match="legacy checkpoint"):
        validate_checkpoint_architecture({}, cfg.architecture)
