from __future__ import annotations

import json
import math
import os
from pathlib import Path
import time
from typing import Callable

import h5py
import numpy as np
import torch

from higan_frequency.data import SpiSGANHDF5Dataset, collate_words
from higan_frequency.frequency import split_frequency_bands


CACHE_VERSION = 1


def _source_metadata(dataset: SpiSGANHDF5Dataset, cutoff: float,
                     padding_mode: str, shape: tuple[int, int]) -> dict:
    stat = dataset.path.stat()
    return {
        "version": CACHE_VERSION,
        "source": str(dataset.path),
        "source_size": int(stat.st_size),
        "source_mtime_ns": int(stat.st_mtime_ns),
        "samples": len(dataset),
        "shape": list(shape),
        "dtype": "float32",
        "cutoff": float(cutoff),
        "padding_mode": str(padding_mode),
    }


def _valid_cache(cache_path: Path, metadata_path: Path, expected: dict) -> bool:
    if not cache_path.is_file() or not metadata_path.is_file():
        return False
    try:
        actual = json.loads(metadata_path.read_text(encoding="utf-8"))
        if actual != expected:
            return False
        cached = np.load(cache_path, mmap_mode="r")
        return list(cached.shape) == expected["shape"] and cached.dtype == np.float32
    except (OSError, ValueError, json.JSONDecodeError):
        return False


@torch.inference_mode()
def prepare_train_frequency_cache(
        dataset: SpiSGANHDF5Dataset, output_root: str | Path, device: torch.device,
        batch_size: int, cutoff: float, padding_mode: str,
        log: Callable[[str], None] = print) -> Path:
    """Compute every train low-frequency image once and store an mmap cache."""
    if dataset.split != "train":
        raise ValueError("only train_32.hdf5 may be frequency-precomputed")
    if batch_size < 1:
        raise ValueError("frequency precompute batch_size must be positive")
    with h5py.File(dataset.path, "r") as handle:
        shape = tuple(int(value) for value in handle["imgs"].shape)
    if len(shape) != 2 or shape[0] != 32:
        raise ValueError(f"unexpected train image storage shape: {shape}")

    cache_root = Path(output_root).resolve() / "frequency_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_path = cache_root / "train_32_low_float32.npy"
    metadata_path = cache_root / "train_32_low_float32.json"
    expected = _source_metadata(dataset, cutoff, padding_mode, shape)
    if _valid_cache(cache_path, metadata_path, expected):
        size_gib = cache_path.stat().st_size / (1024 ** 3)
        log(f"Using frequency cache: {cache_path} ({size_gib:.2f} GiB)")
        return cache_path

    temporary_cache = cache_root / "train_32_low_float32.tmp.npy"
    temporary_metadata = cache_root / "train_32_low_float32.tmp.json"
    temporary_cache.unlink(missing_ok=True)
    temporary_metadata.unlink(missing_ok=True)
    cached = np.lib.format.open_memmap(
        temporary_cache, mode="w+", dtype=np.float32, shape=shape)
    total_batches = math.ceil(len(dataset) / batch_size)
    started_at = time.monotonic()
    log(f"Precomputing train frequency cache: {len(dataset)} images, "
        f"{total_batches} batches, cutoff={cutoff}")
    for batch_number, first in enumerate(range(0, len(dataset), batch_size), 1):
        indices = range(first, min(first + batch_size, len(dataset)))
        samples = [dataset[index] for index in indices]
        images, lengths, *_ = collate_words(samples)
        low, _ = split_frequency_bands(
            images.to(device), lengths.to(device), cutoff=cutoff,
            padding_mode=padding_mode)
        low = low.cpu().numpy()
        for batch_index, dataset_index in enumerate(indices):
            start = int(dataset.img_seek_idxs[dataset_index])
            width = int(dataset.img_lens[dataset_index])
            cached[:, start:start + width] = low[batch_index, 0, :, :width]
        if batch_number == 1 or batch_number == total_batches or batch_number % 100 == 0:
            log(f"[frequency-cache] {batch_number}/{total_batches} batches")
    cached.flush()
    del cached
    dataset.close()
    os.replace(temporary_cache, cache_path)
    temporary_metadata.write_text(
        json.dumps(expected, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary_metadata, metadata_path)
    size_gib = cache_path.stat().st_size / (1024 ** 3)
    elapsed_minutes = (time.monotonic() - started_at) / 60.0
    log(f"Frequency cache ready: {cache_path} ({size_gib:.2f} GiB, "
        f"{elapsed_minutes:.1f} minutes)")
    return cache_path
