# Scale-RAE Representation AutoEncoder tokenizer.
#
# Pairs the frozen SigLIP-2 So400M (`google/siglip2-so400m-patch14-224`) encoder
# with the Scale-RAE paper's ViT-XL decoder
# (https://huggingface.co/nyu-visionx/Scale-RAE-Qwen1.5B_DiT2.4B). This mirrors
# the existing DINOv2-based RAE tokenizer (`dinov2_vae.py`) but for the SigLIP-2
# feature space the Scale-RAE T2I diffusion model targets.
#
# Shape contract (16-multiple I/O matching dinov2_vae conventions; the latent
# grid is Scale-RAE's native 16×16 / 256-token / patch-14 layout):
#   Input  : (B, 3, 1, H, W)  pixel range [-1, 1]   (H = W = 256)
#   Latent : (B, 1152, 1, 16, 16)                   (virtual 16x compression)
#   Output : (B, 3, 1, 256, 256) pixel range [-1, 1] — only when decoder weights
#            are supplied; otherwise decode() returns zeros.
#
# Scale-RAE was natively trained at 224 input → 16×16 token grid (patch 14).
# We bridge the 14-multiple internal grid with the pipeline's 16-multiple
# interface using two bicubic interpolations at the pixel boundary, so both
# encoder and decoder run IN DISTRIBUTION on the 16×16 / 224-pixel grid:
#
#   encode: 256 (16×16) → bicubic ↓ → 224 (14×16)  → SigLIP-2 patch=14 → 16×16 grid
#   decode: 16×16 grid  → SigLIP-2 decoder patch=14 → 224 (14×16) → bicubic ↑ → 256
#
# Internal pipeline per encode():
#   1. [-1,1] -> [0,1]
#   2. bicubic ↓ resize H,W -> 224,224 (with antialias=True)
#   3. SigLIP normalize using the encoder's image_mean/std (typically 0.5/0.5)
#   4. SigLIP-2 vision_tower forward -> (B, 256, 1152) last-hidden tokens
#   5. F.layer_norm(.., (1152,), eps=1e-6) — affine-free layer norm matches the
#      Scale-RAE feature space the decoder + diffusion DiT were trained on
#   6. reshape (B, 256, 1152) -> (B, 1152, 16, 16)
#
# Internal pipeline per decode():
#   1. (B, 1152, 16, 16) -> (B, 256, 1152)
#   2. prepend zero CLS -> (B, 257, 1152)
#   3. GeneralDecoder forward (drop_cls_token=True) -> (B, 256, 14*14*3)
#   4. unpatchify with 16×16 grid -> (B, 3, 224, 224) in SigLIP-normalized space
#   5. denormalize via image_std/mean -> [0, 1]
#   6. bicubic ↑ resize 224 -> 256 (the "14→16 multiple bridge")
#   7. map to [-1, 1]
#
# Architectural knobs (mirroring inference.py:siglip2-so400m-web73m):
#   pretrained_path              = "google/siglip2-so400m-patch14-224"
#   resize_target                = 224  (encoder internal, 14-multiple)
#   native_spatial_resolution    = 256  (interface, 16-multiple)
#   decoder_out_pixel_size       = 224  (decoder raw output, 14-multiple)
#   final_pixel_size             = 256  (after 14→16 multiple bridge)
#   pretrained_decoder_path      = checkpoints/scale_rae/decoder/siglip2_sop14_i224_web73M_ganw3_decXL.pt
#   decoder_config_path          = checkpoints/scale_rae/decoder/XL_decoder_config.json

import json
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
from pid._src.tokenizers.scale_rae_decoder import GeneralDecoder

__all__ = [
    "ScaleRAEEncoder",
    "ScaleRAEVAE",
    "ScaleRAEVAEInterface",
    "ScaleRAEConfig",
]


# ===========================================================================
# Layer 1 — SigLIP-2 encoder + (optional) Scale-RAE ViT-XL decoder
# ===========================================================================


class ScaleRAEEncoder(nn.Module):
    """Frozen SigLIP-2 So400M encoder (+ optional frozen Scale-RAE decoder).

    Args:
        pretrained_path: HF repo id or local directory
            (default ``google/siglip2-so400m-patch14-224``).
        pretrained_decoder_path: path to the Scale-RAE ViT-XL decoder weights
            (e.g. ``siglip2_sop14_i224_web73M_ganw3_decXL.pt``). If None the
            decoder stays None and decode() is a no-op.
        decoder_config_path: path to the decoder JSON config (e.g.
            ``XL_decoder_config.json``). Required when ``pretrained_decoder_path``
            is given.
        decoder_num_patches: token grid size the decoder was trained with
            (16*16 = 256 for SigLIP-2 So400M @ 224).
    """

    def __init__(
        self,
        pretrained_path: str = "google/siglip2-so400m-patch14-224",
        pretrained_decoder_path: Optional[str] = None,
        decoder_config_path: Optional[str] = None,
        decoder_num_patches: int = 256,
    ):
        super().__init__()
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        from transformers import AutoImageProcessor, AutoModel
        from transformers.models.vit_mae.configuration_vit_mae import ViTMAEConfig

        try:
            full_model = AutoModel.from_pretrained(pretrained_path, local_files_only=True)
        except (OSError, ValueError, AttributeError):
            full_model = AutoModel.from_pretrained(pretrained_path)
        # SigLIP wraps a vision_model + text_model; we only need the vision tower.
        vision_tower = getattr(full_model, "vision_model", full_model)
        vision_tower.requires_grad_(False)
        vision_tower.eval()

        # Bypass the SigLIP-2 post_layernorm: the Scale-RAE decoder was trained
        # against the *pre*-final-LN hidden state (with an affine-free LN
        # applied externally — see Scale-RAE's `SigLIPEncoderForDebugging`,
        # multimodal_decoder/__init__.py:80-93). On transformers 4.57+
        # `Siglip2VisionModel.forward` no longer surfaces hidden_states even
        # when `output_hidden_states=True` is passed, so the only way to reach
        # the pre-LN tensor is to neutralize `post_layernorm` itself.
        if hasattr(vision_tower, "post_layernorm"):
            vision_tower.post_layernorm = nn.Identity()

        self.encoder = vision_tower
        # patch14 + 224 input -> 16x16 = 256 tokens; hidden_size = 1152.
        vis_cfg = getattr(full_model.config, "vision_config", full_model.config)
        self.patch_size = int(vis_cfg.patch_size)
        self.hidden_size = int(vis_cfg.hidden_size)

        # SigLIP image mean/std (0.5/0.5 for siglip2-so400m). Read from the
        # processor to avoid hard-coding. Non-persistent — not saved to ckpt.
        try:
            proc = AutoImageProcessor.from_pretrained(pretrained_path, local_files_only=True)
        except (OSError, ValueError, AttributeError):
            proc = AutoImageProcessor.from_pretrained(pretrained_path)
        mean = torch.tensor(proc.image_mean, dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor(proc.image_std, dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("siglip_mean", mean, persistent=False)
        self.register_buffer("siglip_std", std, persistent=False)

        # Optional Scale-RAE decoder.
        self.decoder: Optional[GeneralDecoder] = None
        if pretrained_decoder_path is not None:
            if decoder_config_path is None:
                raise ValueError("decoder_config_path is required when pretrained_decoder_path is set")
            with open(decoder_config_path) as f:
                cfg_dict = json.load(f)
            # Match Scale-RAE/multimodal_decoder/__init__.py: hidden_size is the
            # encoder's vision hidden dim (1152 for SigLIP-2 So400M).
            cfg_dict["hidden_size"] = self.hidden_size
            cfg = ViTMAEConfig(**cfg_dict)
            decoder = GeneralDecoder(cfg, num_patches=decoder_num_patches)
            log.info(f"Loading Scale-RAE decoder weights from {pretrained_decoder_path}")
            state_dict = torch.load(pretrained_decoder_path, map_location="cpu")
            missing, unexpected = decoder.load_state_dict(state_dict, strict=False)
            if missing:
                log.warning(f"Scale-RAE decoder missing keys: {missing}")
            if unexpected:
                log.warning(f"Scale-RAE decoder unexpected keys: {unexpected}")
            decoder.requires_grad_(False)
            decoder.eval()
            self.decoder = decoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, H, W) already SigLIP-normalized.

        Returns the pre-final-LN patch tokens (B, N_patches, hidden_size).
        We neutralize `vision_model.post_layernorm` at __init__ time, so
        ``out.last_hidden_state`` IS the pre-LN tensor that Scale-RAE's
        decoder was trained against. The ScaleRAEVAE wrapper then applies
        an affine-free `F.layer_norm` to match `SigLIPEncoderForDebugging`.
        """
        out = self.encoder(x)
        return out.last_hidden_state


# ===========================================================================
# Layer 2 — ScaleRAEVAE wrapper (dtype/AMP + resize + preprocess + decode)
# ===========================================================================


_DECODE_WARNED = {"v": False}


@rank0_first
def _build_scale_rae_encoder(
    pretrained_path: str,
    pretrained_decoder_path: Optional[str],
    decoder_config_path: Optional[str],
    decoder_num_patches: int,
) -> ScaleRAEEncoder:
    log.info(
        f"Loading Scale-RAE encoder from {pretrained_path} (decoder={pretrained_decoder_path}) on rank {get_rank()}"
    )
    return ScaleRAEEncoder(
        pretrained_path=pretrained_path,
        pretrained_decoder_path=pretrained_decoder_path,
        decoder_config_path=decoder_config_path,
        decoder_num_patches=decoder_num_patches,
    )


class ScaleRAEVAE:
    """Dtype/AMP wrapper around ScaleRAEEncoder + Scale-RAE decoder.

    All tensors are 4D (B, C, H, W).

    encode(images): (B,3,H,W) in [-1,1]  →  (B, 1152, H/spatial_compression, W/spatial_compression)
    decode(zs)    : (B, 1152, h, w)       →  (B,3,H_out,W_out) in [-1,1] when a
                    decoder is loaded, otherwise a zeros tensor.
    """

    def __init__(
        self,
        pretrained_path: str = "google/siglip2-so400m-patch14-224",
        resize_target: int = 224,
        spatial_compression_factor: int = 16,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
        is_amp: bool = False,
        pretrained_decoder_path: Optional[str] = None,
        decoder_config_path: Optional[str] = None,
        decoder_num_patches: int = 256,
        decoder_out_pixel_size: int = 224,
        final_pixel_size: int = 256,
    ):
        self.dtype = dtype
        self.device = device
        self.resize_target = resize_target
        self.spatial_compression_factor = spatial_compression_factor
        self.decoder_out_pixel_size = decoder_out_pixel_size
        self.final_pixel_size = final_pixel_size

        self.model = _build_scale_rae_encoder(
            pretrained_path=pretrained_path,
            pretrained_decoder_path=pretrained_decoder_path,
            decoder_config_path=decoder_config_path,
            decoder_num_patches=decoder_num_patches,
        )
        self.model = self.model.to(device).eval().requires_grad_(False)
        sync_model_states(self.model)
        self.has_decoder = self.model.decoder is not None

        self.is_amp = is_amp
        if not is_amp:
            self.model = self.model.to(dtype=dtype)
            self.context = nullcontext()
        else:
            self.context = torch.amp.autocast("cuda", dtype=dtype)

    def count_param(self):
        return sum(p.numel() for p in self.model.parameters())

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """images: (B, 3, H, W) in [-1, 1].

        Returns (B, 1152, H/spatial_compression, W/spatial_compression).
        """
        assert images.ndim == 4, f"ScaleRAEVAE.encode expects 4D input, got {images.shape}"
        in_dtype = images.dtype
        B, _, H, W = images.shape

        x = (images + 1.0) / 2.0  # [-1,1] → [0,1]

        if H != self.resize_target or W != self.resize_target:
            # Downsample (e.g. 256 → 224 = 16 × patch_size 14). antialias=True
            # adds the recommended low-pass filter for bicubic downsampling
            # (PyTorch ≥ 1.11) — avoids moire when shrinking to a non-integer
            # ratio. The 16→14 multiple bridge on the encoder side.
            x = F.interpolate(
                x.to(torch.float32),
                size=(self.resize_target, self.resize_target),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            )

        mean = self.model.siglip_mean.to(x.device, x.dtype)
        std = self.model.siglip_std.to(x.device, x.dtype)
        x = (x - mean) / std

        with self.context:
            if not self.is_amp:
                x = x.to(self.dtype)
            tokens = self.model(x)  # (B, N_patches, hidden_size)

        # Affine-free layer norm — matches Scale-RAE feature space.
        tokens = F.layer_norm(tokens.float(), (self.model.hidden_size,), weight=None, bias=None, eps=1e-6)

        N, C = tokens.shape[1], tokens.shape[2]
        h = int(math.isqrt(N))
        assert h * h == N, f"SigLIP-2 patch count {N} is not a perfect square"
        z = tokens.transpose(1, 2).contiguous().view(B, C, h, h)
        return z.to(in_dtype)

    @torch.no_grad()
    def decode(self, zs: torch.Tensor) -> torch.Tensor:
        """zs: (B, 1152, h, w). Returns (B, 3, final_pixel_size, final_pixel_size) in [-1, 1].

        If the decoder was not loaded, returns zeros of the expected shape.
        """
        if not self.has_decoder:
            if not _DECODE_WARNED["v"]:
                log.warning("ScaleRAEVAE.decode is a no-op stub; no decoder weights loaded.")
                _DECODE_WARNED["v"] = True
            B, _, h, w = zs.shape
            H = h * self.spatial_compression_factor
            W = w * self.spatial_compression_factor
            return torch.zeros((B, 3, H, W), device=zs.device, dtype=zs.dtype)

        in_dtype = zs.dtype
        B, C, h, w = zs.shape
        intermediate = self.decoder_out_pixel_size

        # (B, C, h, w) → (B, h*w, C); prepend zero CLS to (B, h*w+1, C).
        tokens = zs.reshape(B, C, h * w).transpose(1, 2)
        cls = torch.zeros((B, 1, C), device=zs.device, dtype=zs.dtype)
        tokens = torch.cat([cls, tokens], dim=1)

        with self.context:
            if not self.is_amp:
                tokens = tokens.to(self.dtype)
            decoder_out = self.model.decoder(tokens, drop_cls_token=True)
            logits = decoder_out.logits  # (B, 256, p*p*3)
            pixels = self.model.decoder.unpatchify(
                logits,
                original_image_size=(intermediate, intermediate),
            )  # (B, 3, intermediate, intermediate) in SigLIP-normalized space

        # De-SigLIP-normalize to [0,1].
        pixels = pixels.to(torch.float32)
        mean = self.model.siglip_mean.to(pixels.device, pixels.dtype)
        std = self.model.siglip_std.to(pixels.device, pixels.dtype)
        pixels = (pixels * std + mean).clamp(0.0, 1.0)

        # 14-multiple → 16-multiple bridge: bicubic upsample (e.g. 224 → 256).
        # antialias is irrelevant for upsampling (no aliasing introduced).
        if pixels.shape[-1] != self.final_pixel_size:
            pixels = F.interpolate(
                pixels,
                size=(self.final_pixel_size, self.final_pixel_size),
                mode="bicubic",
                align_corners=False,
            ).clamp(0.0, 1.0)

        # [0,1] → [-1,1].
        pixels = pixels * 2.0 - 1.0
        return pixels.to(in_dtype)


# ===========================================================================
# Layer 3 — ScaleRAEVAEInterface(VideoTokenizerInterface)
# ===========================================================================


class ScaleRAEVAEInterface(VideoTokenizerInterface):
    """Pipeline-compatible Scale-RAE: SigLIP-2 So400M encoder + ViT-XL decoder.

    Image-only (T=1). Default I/O at 256×256 (16-multiple, virtual 16x
    compression to a 16×16 token grid) — unified with dinov2_vae and the
    pixel-diffusion pipeline. Internally bridges Scale-RAE's native
    14-multiple resolution (224 = 16×14) with two bicubic interpolations:
    encode-side downsample (256→224, antialias) and decode-side upsample
    (224→256). Both encoder and decoder run IN DISTRIBUTION on the pretrained
    16×16 token grid.
    """

    def __init__(
        self,
        chunk_duration: int = 1,
        pretrained_path: str = "google/siglip2-so400m-patch14-224",
        resize_target: int = 224,
        native_spatial_resolution: int = 256,
        pretrained_decoder_path: Optional[str] = None,
        decoder_config_path: Optional[str] = None,
        decoder_num_patches: int = 256,
        decoder_out_pixel_size: int = 224,
        final_pixel_size: int = 256,
        **kwargs,
    ):
        del kwargs  # absorb LazyDict metadata (e.g. "name")
        self.model = ScaleRAEVAE(
            pretrained_path=pretrained_path,
            resize_target=resize_target,
            spatial_compression_factor=16,
            dtype=torch.bfloat16,
            is_amp=False,
            pretrained_decoder_path=pretrained_decoder_path,
            decoder_config_path=decoder_config_path,
            decoder_num_patches=decoder_num_patches,
            decoder_out_pixel_size=decoder_out_pixel_size,
            final_pixel_size=final_pixel_size,
        )
        self.chunk_duration = chunk_duration
        self._spatial_resolution = native_spatial_resolution

    @property
    def dtype(self):
        return self.model.dtype

    def reset_dtype(self):
        pass

    def encode(self, state: torch.Tensor) -> torch.Tensor:
        if state.ndim == 5:
            assert state.shape[2] == 1, f"Image-only VAE requires T=1, got T={state.shape[2]}"
            x = state.squeeze(2)
        else:
            x = state
        latent = self.model.encode(x)  # (B, 1152, h, w)
        return latent.unsqueeze(2)  # (B, 1152, 1, h, w)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
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
        return 1152

    @property
    def spatial_resolution(self):
        return self._spatial_resolution

    @property
    def name(self):
        return "scale_rae_tokenizer"


# ===========================================================================
# Layer 4 — LazyDict config
# ===========================================================================

# Encoder-only configuration (no decoder, e.g. for diffusion training that only
# needs the latent target and decodes at evaluation time). 256 input → 16×16
# latent grid at 1152-D (16-multiple I/O, virtual 16x spatial compression).
ScaleRAEEncoderOnlyConfig: LazyDict = L(ScaleRAEVAEInterface)(
    name="scale_rae_encoder_only_tokenizer",
    pretrained_path="google/siglip2-so400m-patch14-224",
    resize_target=224,
    native_spatial_resolution=256,
)

# Full Scale-RAE at 256×256 I/O (SigLIP-2 So400M encoder + ViT-XL decoder,
# web73M). Both encoder and decoder run IN DISTRIBUTION on Scale-RAE's
# pretrained 16×16 / 224-pixel grid; bicubic interpolations bridge to the
# pipeline's 16-multiple 256-pixel boundary at I/O.
ScaleRAEConfig: LazyDict = L(ScaleRAEVAEInterface)(
    name="scale_rae_tokenizer",
    pretrained_path="google/siglip2-so400m-patch14-224",
    resize_target=224,
    native_spatial_resolution=256,
    pretrained_decoder_path="checkpoints/scale_rae/decoder/siglip2_sop14_i224_web73M_ganw3_decXL.pt",
    decoder_config_path="checkpoints/scale_rae/decoder/XL_decoder_config.json",
    decoder_num_patches=256,
    decoder_out_pixel_size=224,
    final_pixel_size=256,
)
