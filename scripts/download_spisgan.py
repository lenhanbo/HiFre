from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

import h5py
import numpy as np
from huggingface_hub import hf_hub_download


FILES = {
    "train_32.hdf5": "d31090dc037554619fdb70d53aa90597f3a69d0cb8eb1dfd833a44e99395b0d0",
    "test_32.hdf5": "5e832ec551a57a27d6824be57f1a01ce5cbf10172e4c803f8e3aa5a299b63790",
}
EXPECTED = {"train_32.hdf5": (66603, 339), "test_32.hdf5": (25766, 161)}
KEYS = ("imgs", "lbs", "img_seek_idxs", "lb_seek_idxs", "img_lens", "lb_lens", "wids")
ALPHABET = set('` ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789\'-"/,.+_!#&():;?*')


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repo-id", default="DAIR-Group/SpiS-GAN")
    parser.add_argument("--revision", default="main")
    args = parser.parse_args()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    for filename, expected_hash in FILES.items():
        cached = hf_hub_download(args.repo_id, f"data/{filename}",
                                 repo_type="model", revision=args.revision)
        target = output / filename
        shutil.copy2(cached, target)
        actual_hash = sha256(target)
        if actual_hash != expected_hash:
            raise ValueError(f"SHA-256 mismatch for {filename}: {actual_hash}")
        with h5py.File(target, "r") as handle:
            missing = [key for key in KEYS if key not in handle]
            if missing:
                raise ValueError(f"{filename} missing keys: {missing}")
            actual = (len(handle["img_lens"]), len(np.unique(handle["wids"][:])))
            characters = {chr(int(value)) for value in np.unique(handle["lbs"][:])}
            if not characters <= ALPHABET:
                raise ValueError(f"{filename} contains unknown characters: {characters - ALPHABET}")
        if actual != EXPECTED[filename]:
            raise ValueError(f"{filename} counts {actual} != {EXPECTED[filename]}")
        print(f"verified {filename}: sha256={actual_hash}, images={actual[0]}, writers={actual[1]}")


if __name__ == "__main__":
    main()
