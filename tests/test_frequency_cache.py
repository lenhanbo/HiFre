import h5py
import numpy as np
import torch

from higan_frequency.data import (FrequencyCachedDataset, SpiSGANHDF5Dataset,
                                  collate_frequency_words)
from higan_frequency.frequency import split_frequency_bands
from higan_frequency.frequency_cache import prepare_train_frequency_cache


def _write_train_split(path):
    widths = np.array([19, 33, 19, 33], dtype=np.int32)
    texts = ["one", "two", "red", "blue"]
    encoded = np.array([ord(ch) for text in texts for ch in text], dtype=np.int32)
    with h5py.File(path, "w") as handle:
        handle["imgs"] = np.random.default_rng(7).integers(
            0, 256, (32, int(widths.sum())), dtype=np.uint8)
        handle["lbs"] = encoded
        handle["img_seek_idxs"] = np.concatenate(([0], np.cumsum(widths)[:-1]))
        text_lengths = np.array([len(text) for text in texts], dtype=np.int32)
        handle["lb_seek_idxs"] = np.concatenate(([0], np.cumsum(text_lengths)[:-1]))
        handle["img_lens"] = widths
        handle["lb_lens"] = text_lengths
        handle["wids"] = np.array([1, 2, 1, 2], dtype=np.int32)


def test_precomputed_train_cache_matches_direct_fft_and_is_reused(tmp_path):
    _write_train_split(tmp_path / "train_32.hdf5")
    dataset = SpiSGANHDF5Dataset(tmp_path, "train", verify_counts=False)
    messages = []
    cache_path = prepare_train_frequency_cache(
        dataset, tmp_path / "output", torch.device("cpu"), batch_size=2,
        cutoff=0.06, padding_mode="reflect", log=messages.append)
    wrapped = FrequencyCachedDataset(dataset, cache_path)
    images, lengths, *_, cached_low, cached_high = collate_frequency_words(
        [wrapped[index] for index in range(len(wrapped))])
    direct_low, direct_high = split_frequency_bands(images, lengths, cutoff=0.06)
    torch.testing.assert_close(cached_low, direct_low, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(cached_high, direct_high, atol=2e-6, rtol=2e-6)

    reused = prepare_train_frequency_cache(
        dataset, tmp_path / "output", torch.device("cpu"), batch_size=2,
        cutoff=0.06, padding_mode="reflect", log=messages.append)
    assert reused == cache_path
    assert any(message.startswith("Using frequency cache:") for message in messages)
