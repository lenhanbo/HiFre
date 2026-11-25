# HiGAN-Frequency

HiGAN-Frequency keeps the original HiGAN generator, discriminator, recognizer,
writer identifier, objectives, and adaptive gradient balancing. It replaces the
single reference style with one shared 32-D style encoder applied in this order:

```text
[noise32 | low-frequency32 | original32 | high-frequency32]
              GBlock@8          @16             @32
```

The Gaussian FFT split is computed independently over every valid word width,
uses `float32`, reflection padding, and does not clamp or normalize either band.

Before production training starts, the full train split is scanned once and its
low-frequency images are written to
`OUTPUT/frequency_cache/train_32_low_float32.npy`. The mmap cache stores only
`low`; each batch reconstructs `high = original - low`. Its JSON metadata binds
the cache to the source path, size, modification time, cutoff, padding mode,
shape, and dtype, so a matching cache is reused and a stale cache is rebuilt.
Generated images still use differentiable FFT at runtime for style
reconstruction because they change after every optimizer step.

## Data protocol

Training uses all 66,603 images and 339 writers in `train_32.hdf5`. There is no
validation split, validation loader, best-checkpoint selection, or per-epoch
FID/KID. `test_32.hdf5` is not opened during training.

After the final epoch, `train.py` loads `checkpoints/last.pth` and evaluates it
exactly once on all 25,766 test images. Every image is its own style reference
and its transcript is regenerated. Global FID/KID use the released SpiS-GAN
fixed 32×128 white-pad/crop preprocessing. These metrics are reporting-only.

The official artifacts are hosted in the
[DAIR-Group/SpiS-GAN Hugging Face repository](https://huggingface.co/DAIR-Group/SpiS-GAN).
The downloader pins their published SHA-256 values and validates the HDF5 schema,
counts, writers, and 81-class alphabet.

## Run

```bash
python -m pip install -r requirements.txt
python scripts/download_spisgan.py --output-dir data
python -m pytest -q
python train.py --config configs/config.yaml --dataset-root data --output outputs/run
```

For a one-batch backward-pass check that never opens the test file:

```bash
python train.py --config configs/config.yaml --dataset-root data \
  --output outputs/smoke --smoke-test
```

Configuration lives in [configs/config.yaml](configs/config.yaml). The relevant
switches are:

```yaml
architecture:
  generator_variant: additive_style  # use baseline for the untouched HiGAN Generator
  additive_style:
    injection_strength: 1.0
frequency:
  cutoff: 0.06
  padding_mode: reflect
  precompute_train: true
  precompute_batch_size: 64
training:
  max_hours: 4.0
  style_reconstruction: true
  save_every_epochs: 5
  preview_every_steps: 2000
  preview_items: 16
```

The additive experiment lives in a separate generator module. It keeps the
upstream HiGAN `Generator`, `Discriminator`, `Recognizer`, and
`WriterIdentifier` unchanged. Checkpoints record the selected generator variant;
legacy checkpoints without this metadata are accepted only in `baseline` mode.

Training writes an annotated panel and companion JSON metadata file under
`OUTPUT/previews/step_NNNNNNNN.{png,json}` every `preview_every_steps` optimizer
iterations. Each row records its writer index, reference text, generated query,
valid widths, deterministic preview seed, and compares the real reference,
one shared random style, and `[noise, low, original, high]` reference style.
Setting `preview_every_steps: 0` disables training previews. The smoke test always
writes one preview, and final evaluation writes an annotated `preview.png` plus
`preview.json` without opening the test split during training.

`training.max_hours` is a soft wall-clock limit checked after every completed
batch. At the limit, training writes a final annotated preview and `last.pth`,
then the normal one-shot final evaluation runs from that checkpoint. The current
configuration allows 4 training hours; set it to `0` to disable the limit. A partial epoch never receives an
`epoch_NNN.pth` name.

When style reconstruction is disabled, `info_loss`, `grad_fake_info`, `gp_info`,
and the associated generator-loss term are skipped; `info_loss` and `gp_info`
are logged as zero.

## Kaggle

Open [kaggle/higan_frequency.ipynb](kaggle/higan_frequency.ipynb), edit its single
configuration cell, enable GPU and Internet, then run all cells. The notebook
clones/pulls the chosen branch, downloads and verifies both HDF5 files, runs
compile/unit/smoke checks, trains, performs the final-only evaluation, and shows
the checkpoints, preview, and FID/KID report.

This repository intentionally does not include resume/optimizer-state machinery
or the SupCon/CTC-guided trainer from HiGAN-Contrastive.
"# HiFre" 
"# HiFre" 
