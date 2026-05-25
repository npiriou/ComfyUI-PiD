from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

import folder_paths
import torch

from .model_configs import (
    BACKBONES,
    PID_CHECKPOINT_PLACEHOLDER,
    PID_CATEGORY,
    PID_MODEL_TYPE,
    BackboneConfig,
    ResolvedCheckpointSpec,
    get_backbone_config,
    validate_checkpoint_path,
    validate_precision,
)
from .vendor import ensure_pid_on_path

_CACHE_LOCK = threading.Lock()
_MODEL_CACHE: dict[tuple[str, str, str, str, str], "PidModelHandle"] = {}


@dataclass
class PidModelHandle:
    model: object | None
    backbone: BackboneConfig
    checkpoint_path: str
    checkpoint_spec: ResolvedCheckpointSpec
    precision_key: str
    device: str


def _available_pid_checkpoints() -> list[str]:
    try:
        filenames = folder_paths.get_filename_list("pid")
    except Exception:
        filenames = []

    if not filenames:
        pid_root = get_pid_models_root()
        if pid_root.exists():
            filenames = sorted(
                str(path.relative_to(pid_root)).replace("\\", "/")
                for path in pid_root.rglob("*.pth")
                if path.is_file()
            )

    checkpoint_files = [name for name in filenames if str(name).lower().endswith(".pth")]
    return checkpoint_files or [PID_CHECKPOINT_PLACEHOLDER]


def get_pid_models_root() -> Path:
    return Path(folder_paths.models_dir) / "pid"


def get_pid_support_checkpoint_path(relative_path: str) -> Path:
    return get_pid_models_root() / relative_path


def _resolve_checkpoint_path(pid_checkpoint: str) -> str:
    if pid_checkpoint == PID_CHECKPOINT_PLACEHOLDER:
        pid_dir = get_pid_models_root()
        raise FileNotFoundError(
            "No PiD checkpoint was found. Put an official Flux1/Z-Image PiD checkpoint under "
            f"'{pid_dir}\\checkpoints\\...\\model_ema_bf16.pth'."
        )

    full_path = folder_paths.get_full_path("pid", pid_checkpoint)
    if full_path:
        return full_path

    if Path(pid_checkpoint).is_file():
        return str(Path(pid_checkpoint).resolve())

    raise FileNotFoundError(f"Unable to resolve PiD checkpoint path for selection: {pid_checkpoint}")


def _resolve_device(device_choice: str) -> torch.device:
    if device_choice not in ("auto", "cuda"):
        raise ValueError(f"Unsupported device selection '{device_choice}'. Use 'auto' or 'cuda'.")

    if not torch.cuda.is_available():
        raise RuntimeError("PiD currently requires CUDA. No CUDA device is available.")

    return torch.device("cuda")


def _load_official_pid_model(
    checkpoint_path: str,
    experiment_name: str,
    precision_key: str,
) -> object:
    ensure_pid_on_path()

    from pid._ext.imaginaire.lazy_config import instantiate
    from pid._ext.imaginaire.utils import misc
    from pid._ext.imaginaire.utils.config_helper import override
    from pid._ext.imaginaire.utils.easy_io import easy_io
    from pid._src.configs.pid import config as pid_config

    precision_name = validate_precision(precision_key)
    flux_vae_path = get_pid_support_checkpoint_path("checkpoints/ae.safetensors")
    if not flux_vae_path.is_file():
        raise FileNotFoundError(
            "Missing PiD support VAE file. Download the official Flux VAE support asset to "
            f"'{flux_vae_path}'."
        )

    config = pid_config.make_config()
    config = override(
        config,
        [
            "--",
            f"experiment={experiment_name}",
            f"model.config.precision={precision_name}",
            f"+model.config.tokenizer.vae_pth={flux_vae_path.as_posix()}",
        ],
    )
    config.validate()
    config.freeze()  # type: ignore[attr-defined]

    misc.set_random_seed(seed=0, by_rank=True)
    torch.backends.cudnn.deterministic = config.trainer.cudnn.deterministic
    torch.backends.cudnn.benchmark = config.trainer.cudnn.benchmark
    torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = True

    if hasattr(config.model.config, "fsdp_shard_size"):
        config.model.config.fsdp_shard_size = 1

    model = instantiate(config.model).cuda()
    model.on_train_start()
    model.load_state_dict(easy_io.load(checkpoint_path), strict=False)
    model = model.to(dtype=model.precision)
    model.eval()
    torch.cuda.empty_cache()
    return model


def get_or_load_pid_model(
    checkpoint_path: str,
    backbone: BackboneConfig,
    precision_key: str,
    device_choice: str,
) -> PidModelHandle:
    device = _resolve_device(device_choice)
    checkpoint_spec = validate_checkpoint_path(backbone, checkpoint_path)

    cache_key = (
        checkpoint_spec.checkpoint_path,
        backbone.key,
        checkpoint_spec.variant.key,
        precision_key,
        str(device),
    )
    with _CACHE_LOCK:
        cached = _MODEL_CACHE.get(cache_key)
        if cached is not None:
            return cached

        handle = PidModelHandle(
            model=None,
            backbone=backbone,
            checkpoint_path=checkpoint_spec.checkpoint_path,
            checkpoint_spec=checkpoint_spec,
            precision_key=precision_key,
            device=str(device),
        )
        _MODEL_CACHE[cache_key] = handle
        return handle


def materialize_pid_model(handle: PidModelHandle) -> PidModelHandle:
    if handle.model is not None:
        return handle

    with _CACHE_LOCK:
        if handle.model is None:
            handle.model = _load_official_pid_model(
                checkpoint_path=handle.checkpoint_path,
                experiment_name=handle.checkpoint_spec.variant.experiment_name,
                precision_key=handle.precision_key,
            )
    return handle


class PiDModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pid_checkpoint": (_available_pid_checkpoints(),),
                "backbone_type": (list(BACKBONES.keys()), {"default": "flux1"}),
                "precision": (["bf16", "fp16", "fp32"], {"default": "bf16"}),
                "device": (["auto", "cuda"], {"default": "auto"}),
            }
        }

    RETURN_TYPES = (PID_MODEL_TYPE,)
    RETURN_NAMES = ("pid_model",)
    FUNCTION = "load_model"
    CATEGORY = PID_CATEGORY
    DESCRIPTION = "Loads an official NVIDIA PiD checkpoint for Flux1/Z-Image decoding."

    def load_model(self, pid_checkpoint: str, backbone_type: str, precision: str, device: str):
        backbone = get_backbone_config(backbone_type)
        checkpoint_path = _resolve_checkpoint_path(pid_checkpoint)
        handle = get_or_load_pid_model(checkpoint_path, backbone, precision, device)
        return (handle,)
