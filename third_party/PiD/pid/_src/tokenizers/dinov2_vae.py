# DINOv2-B-with-registers Representation AutoEncoder (RAE) tokenizer.
#
# Pairs the frozen DINOv2-with-registers-base encoder with the RAE paper's
# trained ViT-XL decoder (https://arxiv.org/abs/2510.11690), giving a real
# reconstruction path instead of a zeros stub. Used by the pixel-diffusion
# decoder training pipeline (see configs/pixel_diffusion/
# experiement_pixeldit_final/pixeldit_sr_rae.py).
#
# Shape contract:
#   Input  : (B, 3, 1, H, W)  pixel range [-1, 1]   (H = W = 512 for LQ=512)
#   Latent : (B, 768, 1, 32, 32)                    (virtual 16x compression)
#   Output : (B, 3, 1, 512, 512) pixel range [-1, 1] — only when decoder weights
#            are supplied; otherwise decode() returns zeros.
#
# Internal pipeline per encode():
#   1. [-1,1] -> [0,1]
#   2. bicubic resize H,W -> 448,448 (32 * patch_size=14 — divisible)
#   3. ImageNet normalization
#   4. DINOv2WithRegisters forward -> (B, 1024, 768) last-hidden tokens
#   5. strip 5 special tokens (1 CLS + 4 registers)
#   6. reshape (B, 1024, 768) -> (B, 768, 32, 32)
#   7. (optional) per-(C,H,W) latent normalization using RAE's ImageNet stats
#
# Internal pipeline per decode():
#   1. (optional) inverse latent normalization
#   2. (B, 768, 32, 32) -> (B, 1024, 768)
#   3. ViT-XL decoder -> (B, 1024, 16*16*3) patch logits (ImageNet-normalized)
#   4. unpatchify -> (B, 3, 512, 512), de-normalize back to [0, 1]
#   5. [0, 1] -> [-1, 1]
#
# Architectural knobs (mirroring configs/stage1/pretrained/DINOv2-B_512.yaml):
#   pretrained_path              = "facebook/dinov2-with-registers-base"
#   resize_target                = 448
#   normalize_layernorm_affine   = True
#   pretrained_decoder_path      = checkpoints/rae/.../ViTXL_n08_i512/model.pt
#   normalization_stat_path      = checkpoints/rae/.../imagenet1k_512/stat.pt

import math
import os
from contextlib import nullcontext
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from pid._ext.imaginaire.lazy_config import LazyCall as L
from pid._ext.imaginaire.lazy_config import LazyDict
from pid._ext.imaginaire.utils import log
from pid._ext.imaginaire.utils.distributed import get_rank, rank0_first, sync_model_states
from pid._src.tokenizers.interface import VideoTokenizerInterface
from pid._src.tokenizers.rae_decoder import GeneralDecoder

__all__ = [
    "DINOv2Encoder",
    "DINOv2VAE",
    "DINOv2VAEInterface",
    "DINOv2VAEConfig",
]


# ViT-XL decoder hyperparameters (configs/decoder/ViTXL/config.json in the RAE
# repo), baked in here so callers only specify the weight path.
_VITXL_DECODER_KWARGS = dict(
    hidden_size=768,
    decoder_hidden_size=1152,
    decoder_intermediate_size=4096,
    decoder_num_attention_heads=16,
    decoder_num_hidden_layers=28,
    intermediate_size=3072,
    num_attention_heads=12,
    num_hidden_layers=12,
    num_channels=3,
    patch_size=16,
    layer_norm_eps=1e-12,
    hidden_dropout_prob=0.0,
    attention_probs_dropout_prob=0.0,
    qkv_bias=True,
    hidden_act="gelu",
    initializer_range=0.02,
)


# ===========================================================================
# Layer 1 — DINOv2 encoder + (optional) RAE ViT-XL decoder
# ===========================================================================


class DINOv2Encoder(nn.Module):
    """Frozen DINOv2-with-registers encoder (+ optional frozen RAE decoder).

    Args:
        pretrained_path: HF repo id or local directory
            (default ``facebook/dinov2-with-registers-base``).
        normalize_layernorm_affine: when True, removes the final LayerNorm's
            affine (weight/bias) so features are approximately zero-mean/
            unit-variance per token — matches the RAE "Dinov2withNorm" recipe.
        pretrained_decoder_path: path to RAE's ViT-XL decoder weights
            (``checkpoints/rae/.../ViTXL_n08_i512/model.pt``). If None, the
            decoder is not instantiated and ``self.decoder`` stays None.
        decoder_num_patches: patch count the decoder was trained with — fixes
            the decoder's positional grid. 32*32 = 1024 for the i512 variant.
        decoder_image_size: reconstructed image height/width (pixels) for the
            configured decoder (512 for i512).
    """

    def __init__(
        self,
        pretrained_path: str = "facebook/dinov2-with-registers-base",
        normalize_layernorm_affine: bool = True,
        pretrained_decoder_path: Optional[str] = None,
        decoder_num_patches: int = 1024,
        decoder_image_size: int = 512,
    ):
        super().__init__()
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        from transformers import AutoImageProcessor, AutoModel
        from transformers.models.vit_mae.configuration_vit_mae import ViTMAEConfig

        try:
            encoder = AutoModel.from_pretrained(pretrained_path, local_files_only=True)
        except (OSError, ValueError, AttributeError):
            encoder = AutoModel.from_pretrained(pretrained_path)
        encoder.requires_grad_(False)
        encoder.eval()

        if normalize_layernorm_affine and hasattr(encoder, "layernorm"):
            encoder.layernorm.elementwise_affine = False
            encoder.layernorm.weight = None
            encoder.layernorm.bias = None

        self.encoder = encoder
        # DINOv2-B: patch_size=14, hidden_size=768. with-registers adds 4 reg tokens.
        self.patch_size = int(encoder.config.patch_size)
        self.hidden_size = int(encoder.config.hidden_size)
        self.num_register_tokens = int(getattr(encoder.config, "num_register_tokens", 0))
        # First N special tokens to strip from last_hidden_state: 1 CLS + R registers.
        self.num_special_tokens = 1 + self.num_register_tokens

        # ImageNet mean/std (from the HF processor). Non-persistent — not saved to ckpt.
        try:
            proc = AutoImageProcessor.from_pretrained(pretrained_path, local_files_only=True)
        except (OSError, ValueError, AttributeError):
            proc = AutoImageProcessor.from_pretrained(pretrained_path)
        mean = torch.tensor(proc.image_mean, dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor(proc.image_std, dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("imagenet_mean", mean, persistent=False)
        self.register_buffer("imagenet_std", std, persistent=False)

        # Optional RAE decoder.
        self.decoder: Optional[GeneralDecoder] = None
        if pretrained_decoder_path is not None:
            kwargs = dict(_VITXL_DECODER_KWARGS)
            kwargs["image_size"] = decoder_image_size
            cfg = ViTMAEConfig(**kwargs)
            decoder = GeneralDecoder(cfg, num_patches=decoder_num_patches)
            log.info(f"Loading RAE decoder weights from {pretrained_decoder_path}")
            state_dict = torch.load(pretrained_decoder_path, map_location="cpu")
            missing, unexpected = decoder.load_state_dict(state_dict, strict=False)
            if missing:
                log.warning(f"RAE decoder missing keys: {missing}")
            if unexpected:
                log.warning(f"RAE decoder unexpected keys: {unexpected}")
            decoder.requires_grad_(False)
            decoder.eval()
            self.decoder = decoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, H, W) already ImageNet-normalized.

        Returns patch tokens (B, N_patches, hidden_size).
        """
        out = self.encoder(x)
        return out.last_hidden_state[:, self.num_special_tokens :]


# ===========================================================================
# Layer 2 — DINOv2VAE wrapper (dtype/AMP + resize + preprocess + decode)
# ===========================================================================


_DECODE_WARNED = {"v": False}


@rank0_first
def _build_dinov2_encoder(
    pretrained_path: str,
    normalize_layernorm_affine: bool,
    pretrained_decoder_path: Optional[str],
    decoder_num_patches: int,
    decoder_image_size: int,
) -> DINOv2Encoder:
    """Build encoder (+ optional decoder) — rank 0 populates HF cache first."""
    log.info(f"Loading DINOv2 encoder from {pretrained_path} (decoder={pretrained_decoder_path}) on rank {get_rank()}")
    return DINOv2Encoder(
        pretrained_path=pretrained_path,
        normalize_layernorm_affine=normalize_layernorm_affine,
        pretrained_decoder_path=pretrained_decoder_path,
        decoder_num_patches=decoder_num_patches,
        decoder_image_size=decoder_image_size,
    )


class DINOv2VAE:
    """Dtype/AMP wrapper around DINOv2Encoder + RAE decoder.

    All tensors are 4D (B, C, H, W).

    encode(images): (B,3,H,W) in [-1,1]  →  (B, hidden, H/spatial_compression, W/spatial_compression)
    decode(zs)    : (B, hidden, h, w)    →  (B,3,H_out,W_out) in [-1,1] when a decoder is loaded,
                    otherwise a zeros tensor of the same pixel shape.

    When ``normalization_stat_path`` is provided, encode() additionally
    applies the RAE per-(C,H,W) latent normalization and decode() applies
    the inverse — required to keep the decoder in its trained distribution.
    """

    def __init__(
        self,
        pretrained_path: str = "facebook/dinov2-with-registers-base",
        resize_target: int = 448,
        normalize_layernorm_affine: bool = True,
        spatial_compression_factor: int = 16,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
        is_amp: bool = False,
        pretrained_decoder_path: Optional[str] = None,
        normalization_stat_path: Optional[str] = None,
        decoder_num_patches: int = 1024,
        decoder_image_size: int = 512,
        decoder_out_pixel_size: int = 512,
        latent_norm_eps: float = 1e-5,
    ):
        self.dtype = dtype
        self.device = device
        self.resize_target = resize_target
        self.spatial_compression_factor = spatial_compression_factor
        self.decoder_out_pixel_size = decoder_out_pixel_size

        self.model = _build_dinov2_encoder(
            pretrained_path=pretrained_path,
            normalize_layernorm_affine=normalize_layernorm_affine,
            pretrained_decoder_path=pretrained_decoder_path,
            decoder_num_patches=decoder_num_patches,
            decoder_image_size=decoder_image_size,
        )
        self.model = self.model.to(device).eval().requires_grad_(False)
        sync_model_states(self.model)
        self.has_decoder = self.model.decoder is not None

        # Latent (C,H,W) normalization stats. Stored as fp32 tensors on-device.
        self.latent_norm_eps = latent_norm_eps
        self.has_latent_norm = False
        self._latent_mean: Optional[torch.Tensor] = None
        self._latent_var: Optional[torch.Tensor] = None
        if normalization_stat_path is not None:
            stats = torch.load(normalization_stat_path, map_location="cpu")
            mean = stats.get("mean", None)
            var = stats.get("var", None)
            if mean is None or var is None:
                raise ValueError(f"Normalization stats at {normalization_stat_path} must contain 'mean' and 'var'.")
            self._latent_mean = mean.to(device=device, dtype=torch.float32).unsqueeze(0)  # (1,C,H,W)
            self._latent_var = var.to(device=device, dtype=torch.float32).unsqueeze(0)
            self.has_latent_norm = True
            log.info(
                f"Loaded RAE latent norm stats from {normalization_stat_path} (mean/var shape={tuple(mean.shape)})"
            )

        self.is_amp = is_amp
        if not is_amp:
            self.model = self.model.to(dtype=dtype)
            self.context = nullcontext()
        else:
            self.context = torch.amp.autocast("cuda", dtype=dtype)

    def count_param(self):
        return sum(p.numel() for p in self.model.parameters())

    def _normalize_latent(self, z: torch.Tensor) -> torch.Tensor:
        if not self.has_latent_norm:
            return z
        m = self._latent_mean.to(z.device)
        v = self._latent_var.to(z.device)
        return (z.float() - m) / torch.sqrt(v + self.latent_norm_eps)

    def _denormalize_latent(self, z: torch.Tensor) -> torch.Tensor:
        if not self.has_latent_norm:
            return z
        m = self._latent_mean.to(z.device)
        v = self._latent_var.to(z.device)
        return z.float() * torch.sqrt(v + self.latent_norm_eps) + m

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """images: (B, 3, H, W) in [-1, 1].

        Returns (B, hidden_size, H/spatial_compression, W/spatial_compression).
        """
        assert images.ndim == 4, f"DINOv2VAE.encode expects 4D input, got {images.shape}"
        in_dtype = images.dtype
        B, _, H, W = images.shape

        x = (images + 1.0) / 2.0  # [-1,1] → [0,1]

        if H != self.resize_target or W != self.resize_target:
            x = F.interpolate(
                x.to(torch.float32),
                size=(self.resize_target, self.resize_target),
                mode="bicubic",
                align_corners=False,
            )

        mean = self.model.imagenet_mean.to(x.device, x.dtype)
        std = self.model.imagenet_std.to(x.device, x.dtype)
        x = (x - mean) / std

        with self.context:
            if not self.is_amp:
                x = x.to(self.dtype)
            tokens = self.model(x)  # (B, N_patches, hidden_size)

        N, C = tokens.shape[1], tokens.shape[2]
        h = int(math.isqrt(N))
        assert h * h == N, f"DINOv2 patch count {N} is not a perfect square"
        z = tokens.transpose(1, 2).contiguous().view(B, C, h, h)
        z = self._normalize_latent(z)
        return z.to(in_dtype)

    @torch.no_grad()
    def decode(self, zs: torch.Tensor) -> torch.Tensor:
        """zs: (B, hidden, h, w). Returns (B, 3, H_out, W_out) in [-1, 1].

        If the decoder was not loaded, returns zeros of the expected shape
        (compat with existing RAE-encoder-only pixel-diffusion pipelines).
        """
        if not self.has_decoder:
            if not _DECODE_WARNED["v"]:
                log.warning("DINOv2VAE.decode is a no-op stub; no decoder weights loaded.")
                _DECODE_WARNED["v"] = True
            B, _, h, w = zs.shape
            H = h * self.spatial_compression_factor
            W = w * self.spatial_compression_factor
            return torch.zeros((B, 3, H, W), device=zs.device, dtype=zs.dtype)

        in_dtype = zs.dtype
        B, C, h, w = zs.shape
        # Inverse latent-norm is cheapest in fp32, then cast to model dtype.
        z = self._denormalize_latent(zs)

        # (B, C, h, w) → (B, h*w, C)
        tokens = z.reshape(B, C, h * w).transpose(1, 2)

        with self.context:
            if not self.is_amp:
                tokens = tokens.to(self.dtype)
            logits = self.model.decoder(tokens, drop_cls_token=False)  # (B, N, p*p*3)
            pixels = self.model.decoder.unpatchify(
                logits,
                original_image_size=(self.decoder_out_pixel_size, self.decoder_out_pixel_size),
            )  # (B, 3, H_out, W_out) in ImageNet-normalized space

        # De-ImageNet-normalize to [0,1], then to [-1,1].
        pixels = pixels.to(torch.float32)
        mean = self.model.imagenet_mean.to(pixels.device, pixels.dtype)
        std = self.model.imagenet_std.to(pixels.device, pixels.dtype)
        pixels = pixels * std + mean
        pixels = pixels.clamp(0.0, 1.0) * 2.0 - 1.0
        return pixels.to(in_dtype)


# ===========================================================================
# Layer 3 — DINOv2VAEInterface(VideoTokenizerInterface)
# ===========================================================================


class DINOv2VAEInterface(VideoTokenizerInterface):
    """Pipeline-compatible RAE using DINOv2-B-with-registers encoder and the
    RAE ViT-XL decoder.

    Image-only (T=1). Latent shape: (B, 768, 1, H/16, W/16) when called with
    a 5D (B, 3, 1, H, W) input and H=W=512 (gives a 32x32 token grid).

    Note on compression factor: DINOv2 internally resizes to 448 before
    patchifying at stride 14, producing a 32x32 grid. So the *virtual* ratio
    between the caller-provided spatial size (512) and the output grid (32)
    is 16 — matching the pixel DiT's expected ``latent_spatial_down_factor=16``.
    """

    def __init__(
        self,
        chunk_duration: int = 1,
        pretrained_path: str = "facebook/dinov2-with-registers-base",
        resize_target: int = 448,
        normalize_layernorm_affine: bool = True,
        native_spatial_resolution: int = 512,
        pretrained_decoder_path: Optional[str] = None,
        normalization_stat_path: Optional[str] = None,
        decoder_num_patches: int = 1024,
        decoder_image_size: int = 512,
        decoder_out_pixel_size: int = 512,
        **kwargs,
    ):
        del kwargs  # absorb LazyDict metadata (e.g. "name")
        self.model = DINOv2VAE(
            pretrained_path=pretrained_path,
            resize_target=resize_target,
            normalize_layernorm_affine=normalize_layernorm_affine,
            spatial_compression_factor=16,
            dtype=torch.bfloat16,
            is_amp=False,
            pretrained_decoder_path=pretrained_decoder_path,
            normalization_stat_path=normalization_stat_path,
            decoder_num_patches=decoder_num_patches,
            decoder_image_size=decoder_image_size,
            decoder_out_pixel_size=decoder_out_pixel_size,
        )
        self.chunk_duration = chunk_duration
        self._spatial_resolution = native_spatial_resolution

    @property
    def dtype(self):
        return self.model.dtype

    def reset_dtype(self):
        pass

    def encode(self, state: torch.Tensor) -> torch.Tensor:
        """Accept 5D (B,C,T,H,W) for pipeline compat. T must be 1. Returns 5D."""
        if state.ndim == 5:
            assert state.shape[2] == 1, f"Image-only VAE requires T=1, got T={state.shape[2]}"
            x = state.squeeze(2)
        else:
            x = state
        latent = self.model.encode(x)  # (B, 768, h, w)
        return latent.unsqueeze(2)  # (B, 768, 1, h, w)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Accept 5D (B,C,T,H,W) for pipeline compat. T must be 1. Returns 5D."""
        if latent.ndim == 5:
            assert latent.shape[2] == 1, f"Image-only VAE requires T=1, got T={latent.shape[2]}"
            z = latent.squeeze(2)
        else:
            z = latent
        recon = self.model.decode(z)  # (B, 3, H, W)
        return recon.unsqueeze(2)  # (B, 3, 1, H, W)

    def get_latent_num_frames(self, num_pixel_frames: int) -> int:
        return num_pixel_frames

    def get_pixel_num_frames(self, num_latent_frames: int) -> int:
        return num_latent_frames

    @property
    def spatial_compression_factor(self):
        return 16

    @property
    def temporal_compression_factor(self):
        return 1

    @property
    def pixel_chunk_duration(self):
        return self.chunk_duration

    @property
    def latent_chunk_duration(self):
        return self.chunk_duration

    @property
    def latent_ch(self):
        return 768

    @property
    def spatial_resolution(self):
        return self._spatial_resolution

    @property
    def name(self):
        return "dinov2_vae_tokenizer"


# ===========================================================================
# Layer 4 — LazyDict config
# ===========================================================================

# Encoder-only configuration (legacy; used before RAE decoder integration).
DINOv2VAEConfig: LazyDict = L(DINOv2VAEInterface)(
    name="dinov2_vae_tokenizer",
    pretrained_path="facebook/dinov2-with-registers-base",
    resize_target=448,
    normalize_layernorm_affine=True,
    native_spatial_resolution=512,
)

# Full RAE at 512×512 (DINOv2-B encoder + ViT-XL decoder, ImageNet-1k stats).
# Weights are under checkpoints/rae/ — see README / CLAUDE.md for the
# download command (`hf download nyu-visionx/RAE-collections …`).
DINOv2RAEConfig: LazyDict = L(DINOv2VAEInterface)(
    name="dinov2_rae_tokenizer",
    pretrained_path="facebook/dinov2-with-registers-base",
    resize_target=448,
    normalize_layernorm_affine=True,
    native_spatial_resolution=512,
    pretrained_decoder_path="checkpoints/rae/decoders/dinov2/wReg_base/ViTXL_n08_i512/model.pt",
    normalization_stat_path="checkpoints/rae/stats/dinov2/wReg_base/imagenet1k_512/stat.pt",
    decoder_num_patches=1024,
    decoder_image_size=512,
    decoder_out_pixel_size=512,
)
