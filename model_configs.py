from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PID_MODEL_TYPE = "PID_MODEL"
PID_CATEGORY = "latent/PiD"
PID_CHECKPOINT_PLACEHOLDER = "<put model_ema_bf16.pth under ComfyUI/models/pid>"

LATENT_COMPRESSION = 8
SUPPORTED_BATCH_SIZE = 1

PRECISION_MAP = {
    "bf16": "bfloat16",
    "fp16": "float16",
    "fp32": "float32",
}


@dataclass(frozen=True)
class BackboneConfig:
    key: str
    display_name: str
    official_backbone: str
    latent_channels: int = 16
    scale: int = 4
    max_batch_size: int = SUPPORTED_BATCH_SIZE


@dataclass(frozen=True)
class CheckpointVariant:
    key: str
    experiment_name: str
    expected_lowres_resolution: int
    display_target: str
    target_output_resolution: int


@dataclass(frozen=True)
class ResolvedCheckpointSpec:
    variant: CheckpointVariant
    checkpoint_path: str


BACKBONES = {
    "flux1": BackboneConfig(
        key="flux1",
        display_name="Flux 1 dev",
        official_backbone="flux",
    ),
    "zimage": BackboneConfig(
        key="zimage",
        display_name="Z-Image",
        official_backbone="zimage",
    ),
}

CHECKPOINT_VARIANTS = {
    "2k": CheckpointVariant(
        key="2k",
        experiment_name="PiD_res2k_sr4x_official_flux_distill_4step",
        expected_lowres_resolution=512,
        display_target="512 -> 2048",
        target_output_resolution=2048,
    ),
    "2kto4k": CheckpointVariant(
        key="2kto4k",
        experiment_name="PiD_res2kto4k_sr4x_official_flux_distill_4step",
        expected_lowres_resolution=1024,
        display_target="1024 -> 4K",
        target_output_resolution=3840,
    ),
}

UNSUPPORTED_CHECKPOINT_MARKERS = ("flux2", "sd3", "dinov2", "siglip")


def get_backbone_config(backbone_key: str) -> BackboneConfig:
    try:
        return BACKBONES[backbone_key]
    except KeyError as exc:
        valid = ", ".join(sorted(BACKBONES))
        raise ValueError(f"Unsupported PiD backbone '{backbone_key}'. Valid values: {valid}.") from exc


def validate_precision(precision_key: str) -> str:
    try:
        return PRECISION_MAP[precision_key]
    except KeyError as exc:
        valid = ", ".join(sorted(PRECISION_MAP))
        raise ValueError(f"Unsupported PiD precision '{precision_key}'. Valid values: {valid}.") from exc


def _detect_checkpoint_variant(checkpoint_path: str) -> CheckpointVariant:
    lowered = checkpoint_path.replace("\\", "/").lower()
    if "2kto4k" in lowered:
        return CHECKPOINT_VARIANTS["2kto4k"]
    return CHECKPOINT_VARIANTS["2k"]


def validate_checkpoint_path(backbone: BackboneConfig, checkpoint_path: str) -> ResolvedCheckpointSpec:
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"PiD checkpoint not found: {checkpoint_path}")
    if path.suffix.lower() != ".pth":
        raise ValueError(
            "PiD checkpoint must be a consolidated '.pth' file. "
            f"Received: {checkpoint_path}"
        )

    lowered = checkpoint_path.replace("\\", "/").lower()
    for marker in UNSUPPORTED_CHECKPOINT_MARKERS:
        if marker in lowered:
            raise ValueError(
                "This node currently only supports the official Flux1/Z-Image PiD checkpoints. "
                f"Unsupported checkpoint marker '{marker}' found in: {checkpoint_path}"
            )

    if backbone.key == "flux1" and "zimage" in lowered:
        raise ValueError(
            "Selected a Z-Image-labelled checkpoint while the loader is set to Flux1. "
            "Use the Flux-labelled official PiD checkpoint for Flux1."
        )

    return ResolvedCheckpointSpec(
        variant=_detect_checkpoint_variant(checkpoint_path),
        checkpoint_path=str(path.resolve()),
    )
