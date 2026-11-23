# HiGAN-Frequency project conventions

- Keep `networks.BigGAN_networks.Generator`, `Discriminator`,
  `networks.module.Recognizer`, and `WriterIdentifier` identical to upstream HiGAN.
- Frequency conditioning is `[noise, low, original, high]`, with the three style
  chunks feeding the hierarchical GBlocks at scales 8, 16, and 32.
- Training may open only `train_32.hdf5`. Do not add validation or best-checkpoint
  selection. Save `last.pth` each epoch and periodic `epoch_NNN.pth` files.
- Full `test_32.hdf5` evaluation runs once, after training, from `last.pth`.
- Keep one config (`configs/config.yaml`) and one training entry point (`train.py`).
- Run compile checks and the complete pytest suite after behavior changes.
