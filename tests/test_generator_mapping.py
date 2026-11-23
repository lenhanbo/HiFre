from pathlib import Path

import torch

from lib.utils import yaml2config
from networks.BigGAN_networks import Generator


def test_generator_receives_low_original_high_in_three_gblocks():
    root = Path(__file__).parents[1]
    cfg = yaml2config(root / "configs" / "config.yaml")
    cfg.GenModel.init = "none"
    generator = Generator(**cfg.GenModel).eval()
    captured = []
    hooks = [blocklist[0].register_forward_pre_hook(
        lambda module, inputs: captured.append(inputs[1].detach().clone()))
        for blocklist in generator.blocks]
    latent = torch.randn(2, 128)
    labels = torch.tensor([[1, 2], [3, 4]])
    lengths = torch.tensor([2, 2])
    with torch.no_grad():
        generator(latent, labels, lengths)
    for hook in hooks:
        hook.remove()
    assert len(captured) == 3
    for index, condition in enumerate(captured, start=1):
        torch.testing.assert_close(condition, latent[:, index * 32:(index + 1) * 32])


def test_upstream_networks_are_not_frequency_forks():
    root = Path(__file__).parents[1]
    frequency_text = (root / "higan_frequency" / "frequency.py").read_text(encoding="utf-8")
    assert "Generator" not in frequency_text
    assert "Discriminator" not in frequency_text
