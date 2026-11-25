"""Generator variants and checkpoint compatibility for HiGAN-Frequency."""

from __future__ import annotations

import functools
from collections.abc import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from networks import BigGAN_layers as layers
from networks.BigGAN_networks import Generator as BaselineGenerator
from networks.utils import init_weights


BASELINE_VARIANT = "baseline"
ADDITIVE_STYLE_VARIANT = "additive_style"
SUPPORTED_GENERATOR_VARIANTS = (BASELINE_VARIANT, ADDITIVE_STYLE_VARIANT)


def _get(value, name, default=None):
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def generator_variant(architecture) -> str:
    variant = str(_get(architecture, "generator_variant", ADDITIVE_STYLE_VARIANT))
    if variant not in SUPPORTED_GENERATOR_VARIANTS:
        choices = ", ".join(SUPPORTED_GENERATOR_VARIANTS)
        raise ValueError(f"unknown generator variant {variant!r}; expected one of: {choices}")
    return variant


def architecture_metadata(architecture) -> dict:
    variant = generator_variant(architecture)
    metadata = {"generator_variant": variant}
    if variant == ADDITIVE_STYLE_VARIANT:
        additive = _get(architecture, "additive_style", {})
        strength = float(_get(additive, "injection_strength", 1.0))
        if strength != 1.0:
            raise ValueError("additive_style.injection_strength must be 1.0")
        metadata["additive_style"] = {"injection_strength": strength}
    return metadata


def validate_checkpoint_architecture(checkpoint: Mapping, architecture) -> None:
    """Reject generator state dicts that were built for another architecture."""
    expected = architecture_metadata(architecture)
    actual = checkpoint.get("architecture")
    if actual is None:
        if expected["generator_variant"] == BASELINE_VARIANT:
            return
        raise ValueError(
            "legacy checkpoint has no architecture metadata and can only be loaded "
            "with architecture.generator_variant=baseline; additive_style requires retraining"
        )
    actual_variant = actual.get("generator_variant") if isinstance(actual, Mapping) else None
    if actual_variant != expected["generator_variant"]:
        raise ValueError(
            "checkpoint generator variant mismatch: "
            f"checkpoint={actual_variant!r}, configured={expected['generator_variant']!r}"
        )
    if actual_variant == ADDITIVE_STYLE_VARIANT:
        actual_additive = actual.get("additive_style", {})
        actual_strength = actual_additive.get("injection_strength")
        expected_strength = expected["additive_style"]["injection_strength"]
        if actual_strength is None or float(actual_strength) != expected_strength:
            raise ValueError(
                "checkpoint additive-style metadata mismatch: "
                f"injection_strength={actual_strength!r}, expected={expected_strength!r}"
            )


class AdditiveStyleGBlock(nn.Module):
    """HiGAN residual GBlock with channel-wise additive style projections."""

    def __init__(self, in_channels, out_channels, style_dim, which_conv1,
                 which_conv2, which_linear, activation, upsample=None,
                 injection_strength=1.0, bn_eps=1e-5):
        super().__init__()
        if float(injection_strength) != 1.0:
            raise ValueError("additive style injection strength is fixed at 1.0")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation = activation
        self.upsample = upsample
        self.injection_strength = 1.0
        self.conv1 = which_conv1(in_channels, out_channels)
        self.conv2 = which_conv2(out_channels, out_channels)
        self.bn1 = nn.BatchNorm2d(in_channels, eps=bn_eps)
        self.bn2 = nn.BatchNorm2d(out_channels, eps=bn_eps)
        self.affine1 = which_linear(style_dim, out_channels)
        self.affine2 = which_linear(style_dim, out_channels)
        self.learnable_sc = in_channels != out_channels or upsample is not None
        if self.learnable_sc:
            self.conv_sc = which_conv1(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x, style):
        h = self.activation(self.bn1(x))
        if self.upsample:
            h = self.upsample(h)
            x = self.upsample(x)
        h = self.conv1(h)
        h = h + self.affine1(style)[:, :, None, None]
        h = self.activation(self.bn2(h))
        h = self.conv2(h)
        h = h + self.affine2(style)[:, :, None, None]
        if self.learnable_sc:
            x = self.conv_sc(x)
        return h + x


class AdditiveStyleGenerator(BaselineGenerator):
    """HiGAN Generator whose hierarchical style chunks are additive biases."""

    def __init__(self, injection_strength=1.0, **kwargs):
        # The upstream constructor owns text initialization, latent partitioning,
        # convolutions, attention, output masking, and the forward path. Replacing
        # only its block stack keeps those behaviors identical without modifying it.
        super().__init__(**kwargs)
        if not self.hier or self.z_chunk_size <= 0:
            raise ValueError("additive_style requires hierarchical generator latents")
        if float(injection_strength) != 1.0:
            raise ValueError("additive_style.injection_strength must be 1.0")
        num_svs = int(kwargs.get("num_G_SVs", 1))
        num_itrs = int(kwargs.get("num_G_SV_itrs", 1))
        blocks = []
        for index, out_channels in enumerate(self.arch["out_channels"]):
            conv1 = self.which_conv
            conv2 = self.which_conv
            if "kernel1" in self.arch:
                conv1 = functools.partial(
                    layers.SNConv2d,
                    kernel_size=self.arch["kernel1"][index],
                    padding=1 if self.arch["kernel1"][index] > 1 else 0,
                    num_svs=num_svs, num_itrs=num_itrs, eps=self.SN_eps)
                conv2 = functools.partial(
                    layers.SNConv2d,
                    kernel_size=self.arch["kernel2"][index],
                    padding=1 if self.arch["kernel2"][index] > 1 else 0,
                    num_svs=num_svs, num_itrs=num_itrs, eps=self.SN_eps)
            block = [AdditiveStyleGBlock(
                in_channels=self.arch["in_channels"][index],
                out_channels=out_channels,
                style_dim=self.z_chunk_size,
                which_conv1=conv1,
                which_conv2=conv2,
                which_linear=self.which_linear,
                activation=self.activation,
                upsample=functools.partial(
                    F.interpolate, scale_factor=self.arch["upsample"][index]),
                injection_strength=injection_strength,
                bn_eps=self.BN_eps,
            )]
            resolution = self.arch["resolution"][index]
            if self.arch["attention"][resolution]:
                block.append(layers.Attention(out_channels, self.which_conv))
            blocks.append(block)
        self.blocks = nn.ModuleList([nn.ModuleList(block) for block in blocks])
        if self.init != "none":
            init_weights(self.blocks, self.init)


def build_generator(gen_config, architecture=None):
    """Build the configured Generator variant from the shared train/eval path."""
    if architecture is None:
        root_config = gen_config
        gen_config = _get(root_config, "GenModel")
        architecture = _get(root_config, "architecture")
        if gen_config is None or architecture is None:
            raise ValueError("build_generator requires GenModel and architecture configuration")
    variant = generator_variant(architecture)
    kwargs = dict(gen_config)
    if variant == BASELINE_VARIANT:
        generator = BaselineGenerator(**kwargs)
    else:
        metadata = architecture_metadata(architecture)
        generator = AdditiveStyleGenerator(
            injection_strength=metadata["additive_style"]["injection_strength"],
            **kwargs,
        )
    generator.generator_variant = variant
    return generator


__all__ = [
    "ADDITIVE_STYLE_VARIANT", "BASELINE_VARIANT", "AdditiveStyleGBlock",
    "AdditiveStyleGenerator", "architecture_metadata", "build_generator",
    "generator_variant", "validate_checkpoint_architecture",
]
