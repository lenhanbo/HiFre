import torch

from higan_frequency.frequency import FrequencyStyleEncoder, split_frequency_bands


def test_fft_reconstructs_square_and_rectangular_inputs_and_backpropagates():
    images = torch.randn(2, 1, 32, 64, requires_grad=True)
    lengths = torch.tensor([32, 57])
    low, high = split_frequency_bands(images, lengths, cutoff=0.06)
    for index, width in enumerate(lengths.tolist()):
        torch.testing.assert_close(low[index, ..., :width] + high[index, ..., :width],
                                   images[index, ..., :width], atol=2e-6, rtol=2e-6)
    (low.square().mean() + high.square().mean()).backward()
    assert images.grad is not None
    assert torch.isfinite(images.grad).all()


def test_fft_uses_height_and_width_and_ignores_external_padding():
    valid = torch.randn(1, 1, 32, 41)
    left = torch.cat((valid, torch.full((1, 1, 32, 23), -1.0)), dim=-1)
    right = torch.cat((valid, torch.full((1, 1, 32, 23), 37.0)), dim=-1)
    low_left, high_left = split_frequency_bands(left, torch.tensor([41]))
    low_right, high_right = split_frequency_bands(right, torch.tensor([41]))
    torch.testing.assert_close(low_left[..., :41], low_right[..., :41])
    torch.testing.assert_close(high_left[..., :41], high_right[..., :41])

    impulse = torch.zeros(1, 1, 16, 64)
    impulse[..., 8, 32] = 1
    low, _ = split_frequency_bands(impulse, torch.tensor([64]))
    assert low[0, 0, 8, 31] != low[0, 0, 7, 32]


def test_grouped_fft_matches_individual_width_processing():
    images = torch.randn(4, 1, 32, 64)
    lengths = torch.tensor([32, 57, 32, 57])
    grouped_low, grouped_high = split_frequency_bands(images, lengths)
    for index in range(len(images)):
        single_low, single_high = split_frequency_bands(
            images[index:index + 1], lengths[index:index + 1])
        torch.testing.assert_close(grouped_low[index:index + 1], single_low)
        torch.testing.assert_close(grouped_high[index:index + 1], single_high)


class RecordingEncoder(torch.nn.Module):
    style_dim = 32

    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, image, lengths, backbone=None, vae_mode=False):
        self.calls.append(image)
        value = image.mean((1, 2, 3), keepdim=False)[:, None].repeat(1, 32)
        return (value, value + 10, value + 20) if vae_mode else value


def test_frequency_encoder_shares_one_encoder_and_concatenates_low_original_high():
    module = FrequencyStyleEncoder(style_dim=32, share_wid=False, init="none")
    recorder = RecordingEncoder()
    module.encoder = recorder
    images = torch.randn(2, 1, 32, 48)
    lengths = torch.tensor([39, 48])
    z, mu, logvar = module(images, lengths, vae_mode=True)
    assert len(recorder.calls) == 3
    assert recorder.calls[1] is images
    assert z.shape == mu.shape == logvar.shape == (2, 96)
    for output, offset in ((z, 0), (mu, 10), (logvar, 20)):
        for index, call in enumerate(recorder.calls):
            expected = call.mean((1, 2, 3)) + offset
            torch.testing.assert_close(output[:, index * 32], expected)
