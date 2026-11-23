import h5py
import numpy as np

from higan_frequency.data import SpiSGANHDF5Dataset, collate_words


def test_hdf5_dataset_and_collation_keep_valid_widths(tmp_path):
    path = tmp_path / "train_32.hdf5"
    widths = np.array([19, 33], dtype=np.int32)
    texts = ["Hi", "GAN"]
    encoded = np.array([ord(ch) for text in texts for ch in text], dtype=np.int32)
    with h5py.File(path, "w") as handle:
        handle["imgs"] = np.random.randint(0, 256, (32, int(widths.sum())), dtype=np.uint8)
        handle["lbs"] = encoded
        handle["img_seek_idxs"] = np.array([0, widths[0]])
        handle["lb_seek_idxs"] = np.array([0, len(texts[0])])
        handle["img_lens"] = widths
        handle["lb_lens"] = np.array([len(text) for text in texts])
        handle["wids"] = np.array([7, 9])
    dataset = SpiSGANHDF5Dataset(tmp_path, "train", verify_counts=False)
    images, valid_lens, labels, label_lens, writers, decoded = collate_words(
        [dataset[0], dataset[1]])
    assert images.shape == (2, 1, 32, 48)
    assert valid_lens.tolist() == widths.tolist()
    assert label_lens.tolist() == [2, 3]
    assert writers.tolist() == [0, 1]
    assert decoded == texts
