from __future__ import annotations

from itertools import chain
from pathlib import Path

import torch
import torch.nn.functional as F
from munch import Munch
from torch.nn import CTCLoss, CrossEntropyLoss
from torch.utils.data import DataLoader

from higan_frequency.data import SpiSGANHDF5Dataset, collate_words
from higan_frequency.frequency import FrequencyStyleEncoder
from lib.alphabet import Alphabets, get_lexicon, strLabelConverter
from lib.utils import AverageMeterManager, get_logger, option_to_string
from networks.BigGAN_networks import Discriminator, Generator
from networks.module import Recognizer, WriterIdentifier
from networks.utils import get_scheduler, idx_to_words, rand_clip, set_requires_grad

try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:
    from tensorboardX import SummaryWriter


def kl_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return torch.mean(-0.5 * torch.sum(1 + logvar - mu.square() - logvar.exp(), dim=1))


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
            "frequency": {"cutoff": float(self.opt.frequency.cutoff),
                          "padding_mode": str(self.opt.frequency.padding_mode)},
            "alphabet": "spisgan_iam",
        }, path)
        return path

    def load(self, path: str | Path):
        checkpoint = torch.load(path, map_location=self.device)
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
            G=Generator(**opt.GenModel).to(self.device),
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

    def _random_labels(self, batch_size: int):
        indices = torch.randint(len(self.lexicon), (batch_size,), device=self.device)
        words = idx_to_words(indices.cpu().tolist(), self.lexicon, self.opt.training.capitalize_ratio)
        labels, lengths = self.label_converter.encode(words)
        return labels.to(self.device).long(), lengths.to(self.device).long()

    def _reference_latent(self, images, image_lens, vae_mode=True):
        encoded = self.models.E(images, image_lens, self.models.W.cnn_backbone, vae_mode=vae_mode)
        style = encoded[0] if vae_mode else encoded
        noise = torch.randn((len(images), 32), device=self.device)
        return torch.cat((noise, style), dim=1), encoded

    def _generate(self, latent, labels, label_lens):
        return self.models.G(latent, labels, label_lens)

    def train(self, max_batches: int = 0):
        self.create_logger()
        opt = self.opt
        # Deliberately instantiate only train_32.hdf5. test_32.hdf5 is first
        # touched by evaluate.py after the final epoch.
        train_dataset = SpiSGANHDF5Dataset(self.dataset_root, "train")
        loader = DataLoader(train_dataset, batch_size=opt.training.batch_size, shuffle=True,
                            drop_last=True, num_workers=opt.training.num_workers,
                            collate_fn=collate_words)
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
        for epoch in range(1, int(opt.training.epochs) + 1):
            for batch_index, batch in enumerate(loader):
                images, image_lens, labels, label_lens, writers, _ = batch
                real = images.to(self.device)
                real_lens = image_lens.to(self.device)
                real_labels = labels.to(self.device).long()
                real_label_lens = label_lens.to(self.device).long()
                writers = writers.to(self.device)

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
                    reference_z, _ = self._reference_latent(real, real_lens)
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
                    reference_z, (_, mu, logvar) = self._reference_latent(real, real_lens)
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
                        adv_loss, fake_cat, create_graph=True, retain_graph=True)[0]
                    grad_ctc = torch.autograd.grad(
                        fake_ctc_loss, fake_ctc, create_graph=True, retain_graph=True)[0]
                    grad_wid = torch.autograd.grad(
                        fake_wid_loss, fake_wid_logits, create_graph=True, retain_graph=True)[0]
                    std_adv = grad_adv.std()
                    gp_ctc = (std_adv / (grad_ctc.std() + 1e-8)).detach().add(1).clamp_max(100)
                    gp_wid = (std_adv / (grad_wid.std() + 1e-8)).detach().add(1).clamp_max(10)

                    if self.style_reconstruction:
                        reconstructed = self.models.E(random_images, fake_lens,
                                                      self.models.W.cnn_backbone)
                        info_loss = (reconstructed - random_z[:, 32:].detach()).abs().mean()
                        grad_info = torch.autograd.grad(
                            info_loss, random_images, create_graph=True, retain_graph=True)[0]
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
                iteration += 1
                if max_batches and batch_index + 1 >= max_batches:
                    break

            self.save("last", epoch)
            cadence = int(opt.training.save_every_epochs)
            if cadence > 0 and epoch % cadence == 0:
                self.save(f"epoch_{epoch:03d}", epoch)
            for scheduler in schedulers:
                scheduler.step(epoch)
            if max_batches:
                break
        self.writer.close()
        return self.checkpoint_path("last")
