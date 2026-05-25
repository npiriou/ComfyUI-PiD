from __future__ import annotations

import comfy.model_management
import torch

from .latent_utils import (
    convert_comfy_flux_latent_to_pid_latent,
    extract_comfy_latent,
    infer_lowres_size_from_latent,
)
from .model_configs import PID_CATEGORY
from .pid_loader import PidModelHandle, materialize_pid_model
from .vae_utils import (
    comfy_image_to_pid_image,
    decode_lowres_condition_image,
    pid_output_to_comfy_image,
)


def _maybe_default_shift(shift: float) -> float | None:
    return None if shift < 0 else float(shift)


class PiDDecode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "vae": ("VAE",),
                "pid_model": ("PID_MODEL",),
                "prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Optional caption forwarded to PiD's text encoder.",
                    },
                ),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "pid_steps": ("INT", {"default": 4, "min": 1, "max": 16}),
                "cfg_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 20.0, "step": 0.05}),
                "scale": ("INT", {"default": 4, "min": 1, "max": 8}),
                "shift": (
                    "FLOAT",
                    {
                        "default": -1.0,
                        "min": -1.0,
                        "max": 20.0,
                        "step": 0.05,
                        "tooltip": "Use -1.0 to keep the official PiD checkpoint default.",
                    },
                ),
                "degrade_sigma": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "decode"
    CATEGORY = PID_CATEGORY
    DESCRIPTION = "Decodes a final clean Flux/Z-Image latent with NVIDIA PiD."

    def decode(
        self,
        latent,
        vae,
        pid_model: PidModelHandle,
        prompt: str,
        seed: int,
        pid_steps: int,
        cfg_scale: float,
        scale: int,
        shift: float,
        degrade_sigma: float,
    ):
        if not isinstance(pid_model, PidModelHandle):
            raise TypeError("PiD Decode expected a PID_MODEL produced by PiD Model Loader.")

        if scale != pid_model.backbone.scale:
            raise ValueError(
                f"PiD currently supports scale={pid_model.backbone.scale} for {pid_model.backbone.display_name}. "
                f"Received scale={scale}."
            )

        if not torch.cuda.is_available():
            raise RuntimeError("PiD currently requires CUDA for decoding.")

        pid_model = materialize_pid_model(pid_model)
        latent_tensor = extract_comfy_latent(latent)
        pid_latent = convert_comfy_flux_latent_to_pid_latent(latent_tensor, pid_model.backbone)
        lowres_h, lowres_w = infer_lowres_size_from_latent(pid_latent)
        expected_lowres = pid_model.checkpoint_spec.variant.expected_lowres_resolution

        if (lowres_h, lowres_w) != (expected_lowres, expected_lowres):
            raise ValueError(
                "PiD Flux1/Z-Image expected a final clean latent compatible with the selected checkpoint "
                f"variant '{pid_model.checkpoint_spec.variant.key}' "
                f"({pid_model.checkpoint_spec.variant.display_target}, base {expected_lowres}px). "
                f"Received latent shape {tuple(pid_latent.shape)}, which maps to {lowres_h}x{lowres_w}."
            )

        try:
            with torch.inference_mode():
                comfy.model_management.unload_all_models()
                comfy.model_management.soft_empty_cache(force=True)
                model_input_dtype = getattr(pid_model.model, "precision", torch.bfloat16)
                lowres_image = decode_lowres_condition_image(vae, pid_latent)
                pid_lq_image = comfy_image_to_pid_image(lowres_image).to(device="cuda", dtype=model_input_dtype)

                data_batch = {
                    pid_model.model.config.input_caption_key: [prompt or ""],
                    "LQ_video_or_image": pid_lq_image,
                    "LQ_latent": pid_latent.to(device="cuda", dtype=model_input_dtype),
                    "degrade_sigma": torch.tensor([float(degrade_sigma)], device="cuda", dtype=torch.float32),
                }

                target_output_resolution = pid_model.checkpoint_spec.variant.target_output_resolution
                infer_image_size = (target_output_resolution, target_output_resolution)
                samples = pid_model.model.generate_samples_from_batch(
                    data_batch,
                    cfg_scale=float(cfg_scale),
                    num_steps=int(pid_steps),
                    seed=int(seed),
                    shift=_maybe_default_shift(shift),
                    image_size=infer_image_size,
                )
                image = pid_output_to_comfy_image(samples)
                return (image,)
        except torch.OutOfMemoryError as exc:
            comfy.model_management.soft_empty_cache(force=True)
            raise RuntimeError(
                "PiD decode ran out of GPU memory. PiD is substantially heavier than native VAE decode."
            ) from exc
        except RuntimeError as exc:
            message = str(exc).lower()
            if "out of memory" in message:
                comfy.model_management.soft_empty_cache(force=True)
                raise RuntimeError(
                    "PiD decode ran out of GPU memory. PiD is substantially heavier than native VAE decode."
                ) from exc
            raise
        finally:
            comfy.model_management.soft_empty_cache()
