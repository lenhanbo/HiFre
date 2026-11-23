from pathlib import Path

from lib.alphabet import Alphabets
from lib.utils import yaml2config


ROOT = Path(__file__).parents[1]


def test_spisgan_configuration_dimensions():
    cfg = yaml2config(ROOT / "configs" / "config.yaml")
    assert len(Alphabets["spisgan_iam"]) == 81
    assert cfg.GenModel.n_class == cfg.DiscModel.n_class == cfg.OcrModel.n_class == 81
    assert cfg.WidModel.n_writer == 339
    assert cfg.EncModel.style_dim == 32
    assert cfg.GenModel.style_dim == 128


def test_training_has_no_validation_best_or_test_dataset_path():
    source = (ROOT / "networks" / "model.py").read_text(encoding="utf-8")
    assert 'SpiSGANHDF5Dataset(self.dataset_root, "train")' in source
    assert 'SpiSGANHDF5Dataset(self.dataset_root, "test")' not in source
    assert 'save("best"' not in source
    assert "validate(" not in source
    assert "FID" not in source and "KID" not in source


def test_final_test_is_last_checkpoint_only_and_after_train():
    training = (ROOT / "train.py").read_text(encoding="utf-8")
    evaluation = (ROOT / "evaluate.py").read_text(encoding="utf-8")
    assert training.index("model.train") < training.index("from evaluate import evaluate")
    assert 'checkpoint_path.name != "last.pth"' in evaluation
    assert 'SpiSGANHDF5Dataset(dataset_root, "test")' in evaluation


def test_style_reconstruction_disabled_removes_info_term_and_logs_zero():
    source = (ROOT / "networks" / "model.py").read_text(encoding="utf-8")
    assert "if self.style_reconstruction:" in source
    assert "info_loss = adv_loss.new_zeros(())" in source
    assert "gp_info = adv_loss.new_zeros(())" in source
    assert "generator_loss = generator_loss + gp_info * info_loss" in source
