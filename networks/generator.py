"""Generator factory and checkpoint compatibility for HiGAN-Frequency."""

from __future__ import annotations

from collections.abc import Mapping

from networks.BigGAN_networks import Generator as BaselineGenerator
from networks.stylegan2_generator import StyleGAN2Generator

BASELINE_VARIANT = "baseline"
STYLEGAN2_VARIANT = "stylegan2"
SUPPORTED_GENERATOR_VARIANTS = (BASELINE_VARIANT, STYLEGAN2_VARIANT)


def _get(value, name, default=None):
    return value.get(name, default) if isinstance(value, Mapping) else getattr(value, name, default)


def generator_variant(architecture) -> str:
    variant = str(_get(architecture, "generator_variant", STYLEGAN2_VARIANT))
    if variant not in SUPPORTED_GENERATOR_VARIANTS:
        choices = ", ".join(SUPPORTED_GENERATOR_VARIANTS)
        raise ValueError(f"unknown generator variant {variant!r}; expected one of: {choices}")
    return variant


def architecture_metadata(architecture) -> dict:
    variant = generator_variant(architecture)
    metadata = {"generator_variant": variant}
    if variant == STYLEGAN2_VARIANT:
        settings = _get(architecture, "stylegan2", {})
        demodulate = bool(_get(settings, "demodulate", True))
        spatial_noise = bool(_get(settings, "spatial_noise", False))
        mapping_network = bool(_get(settings, "mapping_network", False))
        skip_to_image = bool(_get(settings, "skip_to_image", True))
        if not demodulate:
            raise ValueError("stylegan2.demodulate must be true")
        if spatial_noise:
            raise ValueError("stylegan2.spatial_noise is not supported by this HiGAN hybrid")
        if mapping_network:
            raise ValueError("stylegan2.mapping_network is not supported by this HiGAN hybrid")
        if not skip_to_image:
            raise ValueError("stylegan2.skip_to_image must be true")
        metadata["stylegan2"] = {
            "demodulate": True,
            "spatial_noise": False,
            "mapping_network": False,
            "skip_to_image": True,
        }
    return metadata


def validate_checkpoint_architecture(checkpoint: Mapping, architecture) -> None:
    expected = architecture_metadata(architecture)
    actual = checkpoint.get("architecture")
    if actual is None:
        if expected["generator_variant"] == BASELINE_VARIANT:
            return
        raise ValueError(
            "legacy checkpoint has no architecture metadata and can only be loaded "
            "with architecture.generator_variant=baseline; stylegan2 requires retraining")
    actual_variant = actual.get("generator_variant") if isinstance(actual, Mapping) else None
    if actual_variant != expected["generator_variant"]:
        raise ValueError(
            "checkpoint generator variant mismatch: "
            f"checkpoint={actual_variant!r}, configured={expected['generator_variant']!r}")
    if actual_variant == STYLEGAN2_VARIANT and actual.get("stylegan2") != expected["stylegan2"]:
        raise ValueError(
            "checkpoint StyleGAN2 metadata mismatch: "
            f"checkpoint={actual.get('stylegan2')!r}, configured={expected['stylegan2']!r}")


def build_generator(gen_config, architecture=None):
    if architecture is None:
        root_config = gen_config
        gen_config = _get(root_config, "GenModel")
        architecture = _get(root_config, "architecture")
        if gen_config is None or architecture is None:
            raise ValueError("build_generator requires GenModel and architecture configuration")
    variant = generator_variant(architecture)
    kwargs = dict(gen_config)
    if variant == BASELINE_VARIANT:
        generator = BaselineGenerator(**kwargs)
    else:
        settings = architecture_metadata(architecture)["stylegan2"]
        generator = StyleGAN2Generator(demodulate=settings["demodulate"], **kwargs)
    generator.generator_variant = variant
    return generator


__all__ = [
    "BASELINE_VARIANT", "STYLEGAN2_VARIANT", "architecture_metadata",
    "build_generator", "generator_variant", "validate_checkpoint_architecture",
]
