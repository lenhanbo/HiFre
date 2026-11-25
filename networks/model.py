from __future__ import annotations

from itertools import chain
from pathlib import Path
import random
import time

import torch
import torch.nn.functional as F
from munch import Munch
from torch.nn import CTCLoss, CrossEntropyLoss
from torch.utils.data import DataLoader

from higan_frequency.data import (FrequencyCachedDataset, SpiSGANHDF5Dataset,
                                  collate_frequency_words, collate_words)
from higan_frequency.frequency import FrequencyStyleEncoder
from higan_frequency.frequency_cache import prepare_train_frequency_cache
from higan_frequency.preview import save_preview
from lib.alphabet import Alphabets, get_lexicon, strLabelConverter, word_capitalize
from lib.utils import AverageMeterManager, get_logger, option_to_string
from networks.BigGAN_networks import Discriminator
from networks.generator import (architecture_metadata, build_generator,
                                validate_checkpoint_architecture)
from networks.module import Recognizer, WriterIdentifier
from networks.utils import get_scheduler, idx_to_words, rand_clip, set_requires_grad

try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:
    from tensorboardX import SummaryWriter


def kl_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return torch.mean(-0.5 * torch.sum(1 + logvar - mu.square() - logvar.exp(), dim=1))


def training_time_limit_reached(started_at: float, max_hours: float,
                                now: float | None = None) -> bool:
    if max_hours <= 0:
        return False
    current = time.monotonic() if now is None else float(now)
    return current - started_at >= max_hours * 3600.0


class BaseModel:
    def __init__(self, opt, log_root: str, dataset_root: str):
        self.opt = opt
        self.device = torch.device(opt.device)
        self.log_root = str(Path(log_root).resolve())
        self.dataset_root = str(Path(dataset_root).resolve())
        self.models = Munch()
        self.logger = None
        self.writer = None
        self.label_converter = strLabelConverter("spisgan_iam")

    def create_logger(self):
        Path(self.log_root).mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.log_root)
        Path(self.log_root, "config.txt").write_text(option_to_string(self.opt), encoding="utf-8")
        self.logger = get_logger(self.log_root)

    def print(self, value):
        (self.logger.info if self.logger else print)(value)

    def set_mode(self, mode: str):
        for model in self.models.values():
            model.train(mode == "train")

    def checkpoint_path(self, tag: str) -> Path:
        return Path(self.log_root, self.opt.training.ckpt_dir, f"{tag}.pth")

    def save(self, tag: str, epoch: int):
        path = self.checkpoint_path(tag)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "models": {name: model.state_dict() for name, model in self.models.items()},
            "epoch": int(epoch),
            "architecture": architecture_metadata(self.opt.architecture),
            "frequency": {"cutoff": float(self.opt.frequency.cutoff),
                          "padding_mode": str(self.opt.frequency.padding_mode)},
            "alphabet": "spisgan_iam",
        }, path)
        return path

    def load(self, path: str | Path):
        checkpoint = torch.load(path, map_location=self.device)
        validate_checkpoint_architecture(checkpoint, self.opt.architecture)
        for name, model in self.models.items():
            model.load_state_dict(checkpoint["models"][name])
        return checkpoint


class AdversarialModel(BaseModel):
    """Original HiGAN objectives with frequency-conditioned reference styles."""

    def __init__(self, opt, log_root="./", dataset_root="data"):
        super().__init__(opt, log_root, dataset_root)
        expected_classes = len(Alphabets["spisgan_iam"])
        configured = [int(opt.GenModel.n_class), int(opt.DiscModel.n_class), int(opt.OcrModel.n_class)]
        if configured != [expected_classes] * 3:
            raise ValueError(f"SpiS-GAN requires {expected_classes} G/D/R classes; got {configured}")
        if int(opt.WidModel.n_writer) != 339:
            raise ValueError("SpiS-GAN train split requires WidModel.n_writer=339")
        if int(opt.GenModel.style_dim) != 128 or int(opt.EncModel.style_dim) != 32:
            raise ValueError("frequency conditioning requires GenModel.style_dim=128 and EncModel.style_dim=32")

        self.lexicon = get_lexicon(opt.training.lexicon, Alphabets["spisgan_iam"],
                                   max_length=opt.training.max_word_len)
        self.models = Munch(
            G=build_generator(opt.GenModel, opt.architecture).to(self.device),
            D=Discriminator(**opt.DiscModel).to(self.device),
            R=Recognizer(**opt.OcrModel).to(self.device),
            E=FrequencyStyleEncoder(cutoff=opt.frequency.cutoff,
                                    padding_mode=opt.frequency.padding_mode,
                                    **opt.EncModel).to(self.device),
            W=WriterIdentifier(**opt.WidModel).to(self.device),
        )
        self.ctc_loss = CTCLoss(zero_infinity=True, reduction="mean")
        self.classify_loss = CrossEntropyLoss()
        self.style_reconstruction = bool(opt.training.style_reconstruction)
        self.preview_every_steps = int(opt.training.preview_every_steps)
        self.preview_items = int(opt.training.preview_items)
        self.max_hours = float(opt.training.max_hours)
        if self.preview_every_steps < 0:
            raise ValueError("training.preview_every_steps must be non-negative")
        if self.preview_items < 1:
            raise ValueError("training.preview_items must be positive")
        if self.max_hours < 0:
            raise ValueError("training.max_hours must be non-negative")

    def _random_labels(self, batch_size: int):
        indices = torch.randint(len(self.lexicon), (batch_size,), device=self.device)
        words = idx_to_words(indices.cpu().tolist(), self.lexicon, self.opt.training.capitalize_ratio)
        labels, lengths = self.label_converter.encode(words)
        return labels.to(self.device).long(), lengths.to(self.device).long()

    def _reference_latent(self, images, image_lens, vae_mode=True, frequency_bands=None):
        encoded = self.models.E(
            images, image_lens, self.models.W.cnn_backbone, vae_mode=vae_mode,
            frequency_bands=frequency_bands)
        style = encoded[0] if vae_mode else encoded
        noise = torch.randn((len(images), 32), device=self.device)
        return torch.cat((noise, style), dim=1), encoded

    def _generate(self, latent, labels, label_lens):
        return self.models.G(latent, labels, label_lens)

    @torch.inference_mode()
    def save_training_preview(self, batch, epoch: int, global_step: int) -> Path:
        modes = {name: model.training for name, model in self.models.items()}
        for model in self.models.values():
            model.eval()
        try:
            images, image_lens, _, _, writers, reference_texts = batch[:6]
            count = min(self.preview_items, len(images))
            images = images[:count].to(self.device)
            image_lens = image_lens[:count].to(self.device)
            writers = writers[:count].tolist()
            reference_texts = list(reference_texts[:count])

            preview_seed = int(self.opt.seed) + int(global_step)
            word_rng = random.Random(preview_seed)
            query_texts = []
            for reference_text in reference_texts:
                candidates = self.lexicon
                query = word_rng.choice(candidates)
                if len(candidates) > 1:
                    while query.lower() == reference_text.lower():
                        query = word_rng.choice(candidates)
                if word_rng.random() < float(self.opt.training.capitalize_ratio):
                    query = word_capitalize(query)
                query_texts.append(query)
            labels, label_lens = self.label_converter.encode(query_texts)
            labels = labels.to(self.device).long()
            label_lens = label_lens.to(self.device).long()
            generated_lens = label_lens * int(self.opt.char_width)

            generator = torch.Generator(device=self.device)
            generator.manual_seed(preview_seed)
            random_z_single = torch.randn(
                (1, 128), generator=generator, device=self.device)
            random_z = random_z_single.expand(count, -1)
            frequency_style = self.models.E(
                images, image_lens, self.models.W.cnn_backbone, vae_mode=False)
            reference_noise = torch.randn(
                (count, 32), generator=generator, device=self.device)
            random_images = self._generate(random_z, labels, label_lens)
            reference_images = self._generate(
                torch.cat((reference_noise, frequency_style), dim=1), labels, label_lens)

            metadata = {
                "phase": "training",
                "generator_variant": self.models.G.generator_variant,
                "epoch": int(epoch),
                "global_step": int(global_step),
                "seed": preview_seed,
                "latent_order": ["noise32", "low32", "original32", "high32"],
                "frequency": {
                    "cutoff": float(self.opt.frequency.cutoff),
                    "padding_mode": str(self.opt.frequency.padding_mode),
                },
                "random_style": {
                    "definition": "One shared 128-D N(0,I) latent for every row.",
                    "z": random_z_single[0].float().cpu().tolist(),
                },
                "reference_style": {
                    "definition": "Per-row [noise32, low32, original32, high32].",
                    "noise": reference_noise.float().cpu().tolist(),
                    "frequency_vectors": frequency_style.float().cpu().tolist(),
                },
            }
            path = Path(self.log_root, "previews", f"step_{global_step:08d}.png")
            return save_preview(
                images, image_lens, random_images, reference_images, generated_lens,
                reference_texts, query_texts, path, max_items=count,
                writer_ids=writers,
                header=f"TRAIN | epoch={epoch:03d} | global_step={global_step:08d} | seed={preview_seed}",
                metadata=metadata)
        finally:
            for name, model in self.models.items():
                model.train(modes[name])

    def train(self, max_batches: int = 0):
        self.create_logger()
        opt = self.opt
        # Deliberately instantiate only train_32.hdf5. test_32.hdf5 is first
        # touched by evaluate.py after the final epoch.
        train_dataset = SpiSGANHDF5Dataset(self.dataset_root, "train")
        collate_fn = collate_words
        if bool(opt.frequency.precompute_train) and not max_batches:
            cache_path = prepare_train_frequency_cache(
                train_dataset, self.log_root, self.device,
                int(opt.frequency.precompute_batch_size), float(opt.frequency.cutoff),
                str(opt.frequency.padding_mode), log=self.print)
            train_dataset = FrequencyCachedDataset(train_dataset, cache_path)
            collate_fn = collate_frequency_words
        elif bool(opt.frequency.precompute_train) and max_batches:
            self.print("Smoke test skips the full train frequency precompute")
        use_pinned_memory = self.device.type == "cuda"
        num_workers = int(opt.training.num_workers)
        loader = DataLoader(train_dataset, batch_size=opt.training.batch_size, shuffle=True,
                            drop_last=True, num_workers=num_workers,
                            pin_memory=use_pinned_memory,
                            persistent_workers=num_workers > 0,
                            collate_fn=collate_fn)
        optimizers = Munch(
            G=torch.optim.Adam(chain(self.models.G.parameters(), self.models.E.parameters()),
                               lr=opt.training.lr,
                               betas=(opt.training.adam_b1, opt.training.adam_b2)),
            D=torch.optim.Adam(chain(self.models.D.parameters(), self.models.R.parameters(),
                                     self.models.W.parameters()), lr=opt.training.lr,
                               betas=(opt.training.adam_b1, opt.training.adam_b2)),
        )
        schedulers = [get_scheduler(optimizer, opt.training) for optimizer in optimizers.values()]
        meters = AverageMeterManager([
            "adv_loss", "fake_disc_loss", "real_disc_loss", "info_loss",
            "fake_ctc_loss", "real_ctc_loss", "fake_wid_loss", "real_wid_loss",
            "kl_loss", "gp_ctc", "gp_info", "gp_wid",
        ])
        iteration = 0
        started_at = time.monotonic()
        stopped_by_time_limit = False
        if self.max_hours > 0:
            self.print(f"Training time limit: {self.max_hours:.2f} hours")
        for epoch in range(1, int(opt.training.epochs) + 1):
            for batch_index, batch in enumerate(loader):
                images, image_lens, labels, label_lens, writers, _ = batch[:6]
                real = images.to(self.device, non_blocking=use_pinned_memory)
                real_lens = image_lens.to(self.device, non_blocking=use_pinned_memory)
                real_labels = labels.to(
                    self.device, dtype=torch.long, non_blocking=use_pinned_memory)
                real_label_lens = label_lens.to(
                    self.device, dtype=torch.long, non_blocking=use_pinned_memory)
                writers = writers.to(self.device, non_blocking=use_pinned_memory)
                frequency_bands = None
                if len(batch) == 8:
                    frequency_bands = (
                        batch[6].to(self.device, non_blocking=use_pinned_memory),
                        batch[7].to(self.device, non_blocking=use_pinned_memory),
                    )

                self.set_mode("train")
                optimizers.D.zero_grad(set_to_none=True)
                set_requires_grad([self.models.G, self.models.E], False)
                set_requires_grad([self.models.D, self.models.R, self.models.W], True)
                real_ctc = self.models.R(real)
                real_ctc_loss = self.ctc_loss(real_ctc, real_labels, real_lens // 8, real_label_lens)
                clipped, clipped_lens = rand_clip(real, real_lens)
                real_wid_loss = self.classify_loss(self.models.W(clipped, clipped_lens), writers)
                with torch.no_grad():
                    fake_labels, fake_label_lens = self._random_labels(len(real))
                    random_z = torch.randn((len(real), 128), device=self.device)
                    random_images = self._generate(random_z, fake_labels, fake_label_lens)
                    reference_z, _ = self._reference_latent(
                        real, real_lens, frequency_bands=frequency_bands)
                    reference_images = self._generate(reference_z, fake_labels, fake_label_lens)
                    fake_cat = torch.cat((random_images, reference_images), dim=0)
                    fake_lens_cat = (fake_label_lens * opt.char_width).repeat(2)
                    fake_label_lens_cat = fake_label_lens.repeat(2)
                fake_disc_loss = F.relu(1 + self.models.D(
                    fake_cat, fake_lens_cat, fake_label_lens_cat)).mean()
                real_disc_loss = F.relu(1 - self.models.D(real, real_lens, real_label_lens)).mean()
                (real_ctc_loss + real_wid_loss + fake_disc_loss + real_disc_loss).backward()
                optimizers.D.step()

                meters.update("real_ctc_loss", real_ctc_loss.item())
                meters.update("real_wid_loss", real_wid_loss.item())
                meters.update("fake_disc_loss", fake_disc_loss.item())
                meters.update("real_disc_loss", real_disc_loss.item())

                if iteration % int(opt.training.num_critic_train) == 0:
                    optimizers.G.zero_grad(set_to_none=True)
                    set_requires_grad([self.models.D, self.models.R, self.models.W], False)
                    set_requires_grad([self.models.G, self.models.E], True)
                    fake_labels, fake_label_lens = self._random_labels(len(real))
                    fake_lens = fake_label_lens * opt.char_width
                    random_z = torch.randn((len(real), 128), device=self.device)
                    random_images = self._generate(random_z, fake_labels, fake_label_lens)
                    reference_z, (_, mu, logvar) = self._reference_latent(
                        real, real_lens, frequency_bands=frequency_bands)
                    reference_images = self._generate(reference_z, fake_labels, fake_label_lens)
                    fake_cat = torch.cat((random_images, reference_images), dim=0)
                    labels_cat = fake_labels.repeat(2, 1)
                    label_lens_cat = fake_label_lens.repeat(2)
                    image_lens_cat = fake_lens.repeat(2)

                    adv_loss = -self.models.D(fake_cat, image_lens_cat, label_lens_cat).mean()
                    fake_ctc = self.models.R(fake_cat)
                    fake_ctc_loss = self.ctc_loss(
                        fake_ctc, labels_cat, image_lens_cat // 8, label_lens_cat)
                    fake_wid_logits = self.models.W(reference_images, fake_lens)
                    fake_wid_loss = self.classify_loss(fake_wid_logits, writers)
                    kld = kl_loss(mu, logvar)

                    grad_adv = torch.autograd.grad(
                        adv_loss, fake_cat, create_graph=False, retain_graph=True)[0]
                    grad_ctc = torch.autograd.grad(
                        fake_ctc_loss, fake_ctc, create_graph=False, retain_graph=True)[0]
                    grad_wid = torch.autograd.grad(
                        fake_wid_loss, fake_wid_logits, create_graph=False, retain_graph=True)[0]
                    std_adv = grad_adv.std()
                    gp_ctc = (std_adv / (grad_ctc.std() + 1e-8)).detach().add(1).clamp_max(100)
                    gp_wid = (std_adv / (grad_wid.std() + 1e-8)).detach().add(1).clamp_max(10)

                    if self.style_reconstruction:
                        reconstructed = self.models.E(random_images, fake_lens,
                                                      self.models.W.cnn_backbone)
                        info_loss = (reconstructed - random_z[:, 32:].detach()).abs().mean()
                        grad_info = torch.autograd.grad(
                            info_loss, random_images, create_graph=False, retain_graph=True)[0]
                        gp_info = (std_adv / (grad_info.std() + 1e-8)).detach().add(1)
                    else:
                        info_loss = adv_loss.new_zeros(())
                        gp_info = adv_loss.new_zeros(())

                    generator_loss = (2 * adv_loss + gp_ctc * fake_ctc_loss
                                      + gp_wid * fake_wid_loss + opt.training.lambda_kl * kld)
                    if self.style_reconstruction:
                        generator_loss = generator_loss + gp_info * info_loss
                    generator_loss.backward()
                    optimizers.G.step()
                    for key, value in (("adv_loss", adv_loss), ("fake_ctc_loss", fake_ctc_loss),
                                       ("fake_wid_loss", fake_wid_loss), ("kl_loss", kld),
                                       ("info_loss", info_loss), ("gp_ctc", gp_ctc),
                                       ("gp_wid", gp_wid), ("gp_info", gp_info)):
                        meters.update(key, float(value.detach()))

                if iteration % int(opt.training.print_iter_val) == 0:
                    values = meters.eval_all()
                    self.print(f"[{epoch:03d}/{int(opt.training.epochs):03d}] "
                               f"[{batch_index:04d}/{len(loader):04d}] "
                               + " ".join(f"{key}={value:.4f}" for key, value in values.items()))
                    for key, value in values.items():
                        self.writer.add_scalar(f"loss/{key}", value, iteration)
                    meters.reset_all()
                global_step = iteration + 1
                stopped_by_time_limit = training_time_limit_reached(
                    started_at, self.max_hours)
                should_preview = (self.preview_every_steps > 0
                                  and global_step % self.preview_every_steps == 0)
                if stopped_by_time_limit or (max_batches and batch_index + 1 >= max_batches):
                    should_preview = True
                if should_preview:
                    preview_path = self.save_training_preview(batch, epoch, global_step)
                    self.print(f"Saved preview: {preview_path}")
                iteration += 1
                if stopped_by_time_limit:
                    elapsed_hours = (time.monotonic() - started_at) / 3600.0
                    self.print(
                        f"Training time limit reached after {elapsed_hours:.2f} hours "
                        f"at epoch={epoch:03d}, batch={batch_index + 1:04d}, "
                        f"global_step={global_step:08d}; saving last.pth")
                    break
                if max_batches and batch_index + 1 >= max_batches:
                    break

            self.save("last", epoch)
            cadence = int(opt.training.save_every_epochs)
            epoch_completed = batch_index + 1 >= len(loader)
            if cadence > 0 and epoch % cadence == 0 and epoch_completed:
                self.save(f"epoch_{epoch:03d}", epoch)
            if epoch_completed:
                for scheduler in schedulers:
                    scheduler.step(epoch)
            if max_batches or stopped_by_time_limit:
                break
        self.writer.close()
        return self.checkpoint_path("last")
