from __future__ import annotations

import torch

from .model_configs import BackboneConfig, LATENT_COMPRESSION


def extract_comfy_latent(latent_input: dict) -> torch.Tensor:
    if not isinstance(latent_input, dict) or "samples" not in latent_input:
        raise ValueError("PiD Decode expected a Comfy LATENT dictionary with a 'samples' tensor.")

    latent = latent_input["samples"]
    if not isinstance(latent, torch.Tensor):
        raise TypeError(f"LATENT['samples'] must be a torch.Tensor, received {type(latent)!r}.")

    if latent.is_nested:
        latent = latent.unbind()[0]

    if latent.ndim != 4:
        raise ValueError(
            "PiD Flux1 expects a final clean 4-D latent tensor shaped [B, C, H, W]. "
            f"Received shape: {tuple(latent.shape)}"
        )
    return latent.contiguous()


def infer_lowres_size_from_latent(latent: torch.Tensor) -> tuple[int, int]:
    return latent.shape[-2] * LATENT_COMPRESSION, latent.shape[-1] * LATENT_COMPRESSION


def convert_comfy_flux_latent_to_pid_latent(latent: torch.Tensor, backbone: BackboneConfig) -> torch.Tensor:
    batch, channels, _, _ = latent.shape
    if batch != backbone.max_batch_size:
        raise ValueError(
            f"PiD currently supports batch size {backbone.max_batch_size}. Received batch size: {batch}"
        )

    if channels != backbone.latent_channels:
        raise ValueError(
            f"PiD {backbone.display_name} expects {backbone.latent_channels} latent channels. "
            f"Received latent shape: {tuple(latent.shape)}"
        )

    return latent.to(dtype=torch.float32).contiguous()
