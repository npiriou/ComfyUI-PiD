from __future__ import annotations

import torch


def decode_lowres_condition_image(vae, latent: torch.Tensor) -> torch.Tensor:
    if vae is None:
        raise ValueError("PiD Decode requires a VAE input. The native VAE is still needed internally.")

    image = vae.decode(latent)
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"VAE.decode() must return a torch.Tensor, received {type(image)!r}.")
    if image.ndim != 4:
        raise ValueError(
            "PiD Decode expected VAE.decode() to return a 4-D IMAGE tensor shaped [B, H, W, C]. "
            f"Received shape: {tuple(image.shape)}"
        )
    return image.clamp(0.0, 1.0).contiguous()


def comfy_image_to_pid_image(image: torch.Tensor) -> torch.Tensor:
    if image.shape[-1] not in (3, 4):
        raise ValueError(
            "PiD Decode expected a 3-channel Comfy IMAGE tensor from the VAE decode. "
            f"Received shape: {tuple(image.shape)}"
        )
    image = image[..., :3]
    return image.movedim(-1, 1).mul(2.0).sub(1.0).to(dtype=torch.float32).contiguous()


def pid_output_to_comfy_image(samples: torch.Tensor) -> torch.Tensor:
    if samples.ndim == 5:
        samples = samples.squeeze(2)
    if samples.ndim != 4:
        raise ValueError(
            "PiD output must be shaped [B, C, H, W] or [B, C, 1, H, W]. "
            f"Received shape: {tuple(samples.shape)}"
        )
    return samples.float().clamp(-1.0, 1.0).add(1.0).mul(0.5).movedim(1, -1).cpu().contiguous()
