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


class RecordingFusion(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, low, pixel, high):
        self.calls.append((low, pixel, high))
        return torch.cat((low, pixel, high), dim=1)


def small_band_config():
    return {
        "base_channels": 4, "max_channels": 8, "norm_groups": 2,
        "init": "none", "low_kernel_sizes": [3, 5], "low_dilation": 2,
        "high_directional_lengths": [3, 5],
    }


def test_frequency_encoder_uses_band_specific_encoders_and_pixel_vae_anchor():
    module = FrequencyStyleEncoder(
        style_dim=32, share_wid=False, init="none", band_config=small_band_config())
    low_recorder = RecordingEncoder()
    pixel_recorder = RecordingEncoder()
    high_recorder = RecordingEncoder()
    module.low_encoder = low_recorder
    module.pixel_encoder = pixel_recorder
    module.high_encoder = high_recorder
    fusion = RecordingFusion()
    module.fusion = fusion
    images = torch.randn(2, 1, 32, 48)
    lengths = torch.tensor([39, 48])
    z, mu, logvar = module(images, lengths, vae_mode=True)
    assert len(low_recorder.calls) == len(pixel_recorder.calls) == len(high_recorder.calls) == 1
    assert len(fusion.calls) == 1
    assert pixel_recorder.calls[0] is images
    assert z.shape == (2, 96)
    assert mu.shape == logvar.shape == (2, 32)
    expected_calls = (low_recorder.calls[0], images, high_recorder.calls[0])
    for index, call in enumerate(expected_calls):
        expected = call.mean((1, 2, 3))
        torch.testing.assert_close(z[:, index * 32], expected)
    expected_pixel = images.mean((1, 2, 3))
    torch.testing.assert_close(mu[:, 0], expected_pixel + 10)
    torch.testing.assert_close(logvar[:, 0], expected_pixel + 20)


def test_frequency_encoder_accepts_precomputed_bands_in_the_same_order():
    module = FrequencyStyleEncoder(style_dim=32, share_wid=False, init="none")
    low_recorder = RecordingEncoder()
    pixel_recorder = RecordingEncoder()
    high_recorder = RecordingEncoder()
    module.low_encoder = low_recorder
    module.pixel_encoder = pixel_recorder
    module.high_encoder = high_recorder
    module.fusion = RecordingFusion()
    images = torch.randn(2, 1, 32, 48)
    low = torch.full_like(images, 2.0)
    high = torch.full_like(images, 3.0)
    output = module(
        images, torch.tensor([48, 48]), frequency_bands=(low, high))
    assert low_recorder.calls[0] is low
    assert pixel_recorder.calls[0] is images
    assert high_recorder.calls[0] is high
    assert output.shape == (2, 96)
