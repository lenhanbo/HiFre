from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from lib.alphabet import Alphabets, strLabelConverter


FILES = {"train": "train_32.hdf5", "test": "test_32.hdf5"}
KEYS = ("imgs", "lbs", "img_seek_idxs", "lb_seek_idxs", "img_lens", "lb_lens", "wids")
EXPECTED = {"train": {"images": 66603, "writers": 339},
            "test": {"images": 25766, "writers": 161}}


class SpiSGANHDF5Dataset(Dataset):
    """Lazy reader for one released SpiS-GAN HDF5 split."""

    def __init__(self, root: str | Path, split: str, verify_counts: bool = True):
        if split not in FILES:
            raise ValueError(f"unknown split: {split}")
        self.split = split
        self.path = Path(root).expanduser().resolve() / FILES[split]
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        self.converter = strLabelConverter("spisgan_iam")
        self._handle = None
        with h5py.File(self.path, "r") as handle:
            missing = [key for key in KEYS if key not in handle]
            if missing:
                raise ValueError(f"{self.path.name} is missing keys: {missing}")
            for key in KEYS[2:]:
                setattr(self, key, np.asarray(handle[key]))
            self.lbs = np.asarray(handle["lbs"])
        writers = sorted({int(value) for value in self.wids.tolist()})
        self.writer_to_index = {value: index for index, value in enumerate(writers)}
        if verify_counts:
            actual = {"images": len(self.img_lens), "writers": len(writers)}
            if actual != EXPECTED[split]:
                raise ValueError(f"unexpected {split} split counts: {actual}; expected {EXPECTED[split]}")

    def _file(self):
        if self._handle is None:
            self._handle = h5py.File(self.path, "r")
        return self._handle

    def __len__(self):
        return len(self.img_lens)

    def __getitem__(self, index: int):
        start, width = int(self.img_seek_idxs[index]), int(self.img_lens[index])
        pixels = np.asarray(self._file()["imgs"][:, start:start + width], dtype=np.float32).copy()
        image = torch.from_numpy(pixels / 127.5 - 1.0).unsqueeze(0)
        label_start, label_len = int(self.lb_seek_idxs[index]), int(self.lb_lens[index])
        text = "".join(chr(int(ch)) for ch in self.lbs[label_start:label_start + label_len])
        unknown = sorted(set(text) - set(Alphabets["spisgan_iam"]))
        if unknown:
            raise ValueError(f"characters outside spisgan_iam alphabet: {unknown}")
        label = torch.tensor(self.converter.encode(text), dtype=torch.long)
        writer = self.writer_to_index[int(self.wids[index])]
        return image, label, writer, text

    def close(self):
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_handle"] = None
        return state

    def __del__(self):
        self.close()


def collate_words(batch):
    images, labels, writers, texts = zip(*batch)
    valid_lens = torch.tensor([image.shape[-1] for image in images], dtype=torch.int32)
    canvas_width = int((((valid_lens.max() + 15) // 16) * 16).item())
    padded = torch.full((len(images), 1, 32, canvas_width), -1.0)
    for index, image in enumerate(images):
        padded[index, :, :, :image.shape[-1]] = image
    label_lens = torch.tensor([len(label) for label in labels], dtype=torch.int32)
    padded_labels = torch.zeros((len(labels), int(label_lens.max())), dtype=torch.long)
    for index, label in enumerate(labels):
        padded_labels[index, :len(label)] = label
    return (padded, valid_lens, padded_labels, label_lens,
            torch.tensor(writers, dtype=torch.long), list(texts))
