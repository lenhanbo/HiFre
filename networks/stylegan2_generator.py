"""StyleGAN2-modulated synthesis blocks for the HiGAN text generator."""

from __future__ import annotations

import functools
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from networks import BigGAN_layers as layers
from networks.BigGAN_networks import Generator as BaselineGenerator
from networks.utils import _len2mask, init_weights, make_one_hot


class EqualizedLinear(nn.Module):
    def __init__(self, in_features, out_features, bias_init=0.0, lr_mul=1.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) / lr_mul)
        self.bias = nn.Parameter(torch.full((out_features,), float(bias_init)))
        self.scale = (1.0 / math.sqrt(in_features)) * lr_mul
        self.lr_mul = float(lr_mul)

    def forward(self, value):
        return F.linear(value, self.weight * self.scale, self.bias * self.lr_mul)


class ModulatedConv2d(nn.Module):
    """Per-sample weight modulation and output-channel demodulation."""

    def __init__(self, in_channels, out_channels, style_dim, kernel_size=3,
                 padding=1, demodulate=True, eps=1e-8):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = int(kernel_size)
        self.padding = int(padding)
        self.demodulate = bool(demodulate)
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.randn(
            1, self.out_channels, self.in_channels, self.kernel_size, self.kernel_size))
        self.weight_scale = 1.0 / math.sqrt(
            self.in_channels * self.kernel_size * self.kernel_size)
        self.modulation = EqualizedLinear(style_dim, self.in_channels, bias_init=1.0)
        self.bias = nn.Parameter(torch.zeros(self.out_channels))

    def forward(self, value, style):
        batch, _, height, width = value.shape
        coefficients = self.modulation(style).view(batch, 1, self.in_channels, 1, 1)
        weight = self.weight * self.weight_scale * coefficients
        if self.demodulate:
            demodulation = torch.rsqrt(weight.square().sum((2, 3, 4)) + self.eps)
            weight = weight * demodulation.view(batch, self.out_channels, 1, 1, 1)
        value = value.reshape(1, batch * self.in_channels, height, width)
        weight = weight.reshape(
            batch * self.out_channels, self.in_channels,
            self.kernel_size, self.kernel_size)
        output = F.conv2d(value, weight, padding=self.padding, groups=batch)
        output = output.view(batch, self.out_channels, output.shape[-2], output.shape[-1])
        return output + self.bias.view(1, -1, 1, 1)


class StyleGAN2GBlock(nn.Module):
    """HiGAN residual/upsampling block with two StyleGAN2 modulated convs."""

    def __init__(self, in_channels, out_channels, style_dim, which_shortcut_conv,
                 activation, upsample=None, demodulate=True, eps=1e-8):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.activation = activation
        self.upsample = upsample
        self.conv1 = ModulatedConv2d(
            in_channels, out_channels, style_dim, demodulate=demodulate, eps=eps)
        self.conv2 = ModulatedConv2d(
            out_channels, out_channels, style_dim, demodulate=demodulate, eps=eps)
        self.learnable_sc = in_channels != out_channels or upsample is not None
        if self.learnable_sc:
            self.conv_sc = which_shortcut_conv(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x, style):
        # Preserve HiGAN's pre-activation residual topology; only the two
        # convolutions themselves use StyleGAN2 modulation/demodulation.
        h = self.activation(x)
        if self.upsample:
            h = self.upsample(h)
            x = self.upsample(x)
        h = self.conv1(h, style)
        h = self.activation(h)
        h = self.conv2(h, style)
        if self.learnable_sc:
            x = self.conv_sc(x)
        return h + x


class StyleGAN2Generator(BaselineGenerator):
    """HiGAN text synthesis with per-band StyleGAN2 weight modulation."""

    def __init__(self, demodulate=True, **kwargs):
        super().__init__(**kwargs)
        if not self.hier or self.z_chunk_size <= 0:
            raise ValueError("stylegan2 requires hierarchical generator latents")
        if not demodulate:
            raise ValueError("StyleGAN2Generator requires weight demodulation")
        self.activation = nn.LeakyReLU(0.2, inplace=False)
        blocks = []
        to_images = []
        for index, out_channels in enumerate(self.arch["out_channels"]):
            block = [StyleGAN2GBlock(
                in_channels=self.arch["in_channels"][index],
                out_channels=out_channels,
                style_dim=self.z_chunk_size,
                which_shortcut_conv=self.which_conv,
                activation=self.activation,
                upsample=functools.partial(
                    F.interpolate, scale_factor=self.arch["upsample"][index]),
                demodulate=True,
                eps=self.SN_eps,
            )]
            resolution = self.arch["resolution"][index]
            if self.arch["attention"][resolution]:
                block.append(layers.Attention(out_channels, self.which_conv))
            blocks.append(block)
            to_images.append(ModulatedConv2d(
                out_channels, int(kwargs.get("input_nc", 3)), self.z_chunk_size,
                kernel_size=1, padding=0, demodulate=False, eps=self.SN_eps))
        self.blocks = nn.ModuleList([nn.ModuleList(block) for block in blocks])
        self.to_images = nn.ModuleList(to_images)
        # The StyleGAN2 skip-to-image path replaces HiGAN's single final
        # BN/ReLU/Conv output head in this experimental generator only.
        self.output_layer = nn.Identity()
        if self.init != "none":
            init_weights(self.blocks, self.init)
            for module in self.blocks.modules():
                if isinstance(module, ModulatedConv2d):
                    nn.init.normal_(module.weight)
                    nn.init.zeros_(module.bias)
                    nn.init.normal_(module.modulation.weight)
                    nn.init.constant_(module.modulation.bias, 1.0)

    def forward(self, z, y, y_lens):
        """Run HiGAN text initialization followed by StyleGAN2 skip synthesis."""
        output_device = z.device
        if self.hier:
            zs = torch.split(z, self.z_chunk_size, 1)
            z = zs[0]
            if len(y.shape) < 2:
                y = y.unsqueeze(1)
            if self.first_layer:
                ys = zs[1:]
            else:
                ys = [torch.cat([y.type(torch.float32), item], 1) for item in zs[1:]]
        else:
            ys = [y] * len(self.blocks)

        if self.first_layer:
            if self.one_hot:
                y = make_one_hot(y, y_lens, self.n_classes).float().to(y.device)
            if self.one_hot_k == 1:
                z = (z.unsqueeze(1).repeat(1, y.shape[1], y.shape[2])
                     * torch.repeat_interleave(y, z.shape[1], 2))
            elif self.one_hot_k > 1:
                y = torch.repeat_interleave(y, self.one_hot_k, 2)
                z = torch.cat((z.unsqueeze(1).repeat(1, y.shape[1], 1), y), 2)
            elif self.one_hot_k == 0:
                z = y
            else:
                z = torch.cat((z.unsqueeze(1).repeat(1, y.shape[1], 1), y), 2)

        h = self.linear(z)
        if self.first_layer:
            h = h.view(h.size(0), h.shape[1] * self.bottom_width, self.bottom_height, -1)
            h = h.permute(0, 3, 2, 1)
        else:
            h = h.view(h.size(0), -1, self.bottom_width, self.bottom_height)

        image = None
        for index, blocklist in enumerate(self.blocks):
            for block in blocklist:
                h = block(h, ys[index])
            current = self.to_images[index](h, ys[index])
            if image is None:
                image = current
            else:
                image = F.interpolate(image, size=current.shape[-2:], mode="nearest") + current
        output = torch.tanh(image)

        if not self.training:
            out_lens = y_lens * output.size(-2) // 2
            mask = _len2mask(out_lens.int(), output.size(-1), torch.float32)
            mask = mask.to(output_device).detach().unsqueeze(1).unsqueeze(1)
            output = output * mask + (mask - 1)
        return output


__all__ = [
    "EqualizedLinear", "ModulatedConv2d", "StyleGAN2GBlock", "StyleGAN2Generator",
]
