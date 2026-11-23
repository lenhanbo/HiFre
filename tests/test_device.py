import torch

from networks.utils import make_one_hot


def test_generator_one_hot_uses_label_device():
    labels = torch.tensor([[1, 2, 0], [3, 4, 5]])
    lengths = torch.tensor([2, 3])
    encoded = make_one_hot(labels, lengths, 81)
    assert encoded.device == labels.device
    assert encoded.shape == (2, 3, 81)
    assert encoded.sum().item() == 5
