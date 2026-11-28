from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import linalg
from torch.utils.data import DataLoader

from fid_kid.inception import InceptionV3
from higan_frequency.data import SpiSGANHDF5Dataset, collate_words
from higan_frequency.frequency import FREQUENCY_ENCODER_VARIANT, FrequencyStyleEncoder
from higan_frequency.preview import save_preview
from lib.utils import yaml2config
from networks.BigGAN_networks import Generator
from networks.module import WriterIdentifier


def inception_features(model, images, lengths, device):
    """Released SpiS-GAN 32x128 white-pad/crop preprocessing."""
    images = images.to(device).clone()
    for index, length in enumerate(lengths.tolist()):
        length = min(int(length), images.shape[-1])
        if length < images.shape[-1]:
            images[index, :, :, length:] = 1.0
    images = ((images + 1) / 2).repeat(1, 3, 1, 1)
    if images.shape[-1] < 128:
        images = F.pad(images, (0, 128 - images.shape[-1]), value=1.0)
    else:
        images = images[..., :128]
    features = model(images)[0]
    if features.shape[-2:] != (1, 1):
        features = F.adaptive_avg_pool2d(features, 1)
    return features.cpu().reshape(len(images), -1).numpy()


def fid_kid(real, fake):
    real_mean, fake_mean = real.mean(0), fake.mean(0)
    real_cov, fake_cov = np.cov(real, rowvar=False), np.cov(fake, rowvar=False)
    result = linalg.sqrtm(real_cov @ fake_cov)
    covariance = result[0] if isinstance(result, tuple) else result
    if not np.isfinite(covariance).all():
        offset = np.eye(real_cov.shape[0]) * 1e-6
        result = linalg.sqrtm((real_cov + offset) @ (fake_cov + offset))
        covariance = result[0] if isinstance(result, tuple) else result
    covariance = covariance.real if np.iscomplexobj(covariance) else covariance
    difference = real_mean - fake_mean
    fid = float(difference @ difference + np.trace(real_cov + fake_cov - 2 * covariance))

    subset_size = min(1000, len(real), len(fake))
    rng = np.random.RandomState(42)
    scores = []
    for _ in range(50):
        left = real[rng.choice(len(real), subset_size, replace=False)]
        right = fake[rng.choice(len(fake), subset_size, replace=False)]
        dimension = real.shape[1]
        k_ll = (left @ left.T / dimension + 1) ** 3
        k_rr = (right @ right.T / dimension + 1) ** 3
        k_lr = (left @ right.T / dimension + 1) ** 3
        denominator = subset_size * (subset_size - 1)
        scores.append((k_ll.sum() - np.trace(k_ll)) / denominator
                      + (k_rr.sum() - np.trace(k_rr)) / denominator - 2 * k_lr.mean())
    return fid, float(np.mean(scores) * 100.0)


@torch.inference_mode()
def evaluate(config, dataset_root, checkpoint_path, output, max_samples=0):
    cfg = yaml2config(config)
    if str(cfg.frequency.encoder_variant) != FREQUENCY_ENCODER_VARIANT:
        raise ValueError(
            f"frequency.encoder_variant must be {FREQUENCY_ENCODER_VARIANT!r}")
    device = torch.device(cfg.device)
    checkpoint_path = Path(checkpoint_path).resolve()
    if checkpoint_path.name != "last.pth":
        raise ValueError("final evaluation must load last.pth")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    variant = checkpoint.get("frequency", {}).get("encoder_variant")
    if variant != FREQUENCY_ENCODER_VARIANT:
        raise ValueError(
            "checkpoint frequency encoder mismatch: "
            f"checkpoint={variant!r}, configured={FREQUENCY_ENCODER_VARIANT!r}")
    G = Generator(**cfg.GenModel).to(device)
    E = FrequencyStyleEncoder(cutoff=cfg.frequency.cutoff,
                              padding_mode=cfg.frequency.padding_mode,
                              band_config=cfg.frequency.band_encoders,
                              **cfg.EncModel).to(device)
    W = WriterIdentifier(**cfg.WidModel).to(device)
    for name, model in (("G", G), ("E", E), ("W", W)):
        model.load_state_dict(checkpoint["models"][name])
        model.eval()

    # This is the only production path that opens test_32.hdf5.
    dataset = SpiSGANHDF5Dataset(dataset_root, "test")
    count = len(dataset) if max_samples <= 0 else min(len(dataset), int(max_samples))
    if count < 2:
        raise ValueError("FID/KID require at least two test samples")
    use_pinned_memory = device.type == "cuda"
    loader = DataLoader(dataset, batch_size=cfg.evaluation.batch_size, shuffle=False,
                        num_workers=0, pin_memory=use_pinned_memory,
                        collate_fn=collate_words)
    block = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    inception = InceptionV3([block]).eval().to(device)
    rng = torch.Generator(device=device).manual_seed(int(cfg.evaluation.seed))
    real_features, fake_features = [], []
    processed = 0
    preview_path = Path(output).resolve().parent / "preview.png"
    for batch_index, batch in enumerate(loader):
        images, lengths, labels, label_lens, writers, texts = batch
        remaining = count - processed
        if remaining <= 0:
            break
        images = images[:remaining].to(device, non_blocking=use_pinned_memory)
        lengths = lengths[:remaining]
        labels = labels[:remaining].to(device, non_blocking=use_pinned_memory)
        label_lens = label_lens[:remaining].to(device, non_blocking=use_pinned_memory)
        device_lengths = lengths.to(device, non_blocking=use_pinned_memory)
        style = E(images, device_lengths, W.cnn_backbone)
        noise = torch.randn((len(images), 32), generator=rng, device=device)
        generated = G(torch.cat((noise, style), dim=1), labels, label_lens)
        generated_lens = label_lens * int(cfg.char_width)
        real_features.append(inception_features(inception, images, lengths, device))
        fake_features.append(inception_features(inception, generated, generated_lens.cpu(), device))
        if batch_index == 0:
            preview_count = min(int(cfg.training.preview_items), len(images))
            preview_generator = torch.Generator(device=device).manual_seed(
                int(cfg.evaluation.seed) + 1_000_003)
            random_z_single = torch.randn(
                (1, 128), generator=preview_generator, device=device)
            random_images = G(
                random_z_single.expand(preview_count, -1),
                labels[:preview_count], label_lens[:preview_count])
            save_preview(
                images[:preview_count], device_lengths[:preview_count], random_images,
                generated[:preview_count], generated_lens[:preview_count],
                list(texts[:preview_count]), list(texts[:preview_count]), preview_path,
                max_items=preview_count, writer_ids=writers[:preview_count].tolist(),
                header=(f"FINAL EVALUATION | checkpoint=last.pth | "
                        f"seed={int(cfg.evaluation.seed)}"),
                metadata={
                    "phase": "evaluation",
                    "checkpoint": str(checkpoint_path),
                    "seed": int(cfg.evaluation.seed),
                    "latent_order": ["noise32", "fused_style96"],
                    "frequency_encoder_variant": FREQUENCY_ENCODER_VARIANT,
                    "random_preview_z": random_z_single[0].float().cpu().tolist(),
                    "fused_style_vectors": style[:preview_count].float().cpu().tolist(),
                })
        processed += len(images)
        if batch_index == 0 or processed == count or batch_index % int(cfg.evaluation.log_every) == 0:
            print(f"[evaluation] {processed}/{count}", flush=True)
    print("[evaluation] computing global FID/KID", flush=True)
    fid, kid = fid_kid(np.concatenate(real_features), np.concatenate(fake_features))
    metrics = {
        "checkpoint": str(checkpoint_path), "samples": processed,
        "frequency_encoder_variant": FREQUENCY_ENCODER_VARIANT,
        "fid": fid, "kid": kid, "scope": "global SpiS-GAN test replication",
        "preprocessing": "fixed 32x128 white-pad/crop", "preview": str(preview_path),
    }
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="One-shot final SpiS-GAN test evaluation")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-samples", type=int, default=0)
    args = parser.parse_args()
    evaluate(args.config, args.dataset_root, args.checkpoint, args.output, args.max_samples)


if __name__ == "__main__":
    main()
