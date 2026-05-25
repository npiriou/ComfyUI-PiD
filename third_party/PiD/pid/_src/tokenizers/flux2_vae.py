# Flux 2 VAE tokenizer — self-contained implementation.
#
# Architecture: standard 2D image VAE with 32 latent channels, 8x spatial compression
# from the encoder, plus 2x2 patchification → effective 128 latent channels at 16x
# spatial compression.
#
# Key differences from Flux 1 VAE:
#   - z_channels = 32 (vs 16)
#   - Normalization uses BatchNorm2d with running stats (vs scale_factor/shift_factor)
#   - 2x2 spatial patchification: channels * 4 = 128 effective latent channels
#   - Encoder has quant_conv (Conv2d 2*z_ch → 2*z_ch)
#   - Decoder has post_quant_conv (Conv2d z_ch → z_ch)
#   - Effective spatial compression: 16x (8x encoder + 2x patchify)
#
# The raw AutoEncoder code below is adapted from the official Flux 2 repository
# with one change: Upsample.forward casts to float32 before interpolate (bfloat16 safety).
#
# Follows the same 5-layer pattern as flux_vae.py:
#   Layer 1: Raw AutoEncoder modules (from Flux 2)
#   Layer 2: Factory function _flux2_vae()
#   Layer 3: Flux2VAE wrapper (dtype/AMP handling)
#   Layer 4: Flux2VAEInterface(VideoTokenizerInterface)
#   Layer 5: Flux2VAEConfig LazyDict

import math
from contextlib import nullcontext
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from pid._ext.imaginaire.lazy_config import LazyCall as L
from pid._ext.imaginaire.lazy_config import LazyDict
from pid._ext.imaginaire.utils import log
from pid._ext.imaginaire.utils.distributed import get_rank, sync_model_states
from pid._src.models.utils import load_state_dict
from pid._src.tokenizers.interface import VideoTokenizerInterface

__all__ = [
    "AutoEncoder",
    "Flux2VAE",
    "Flux2VAEInterface",
    "Flux2VAEConfig",
]


# ===========================================================================
# Layer 1 — Raw Flux 2 AutoEncoder (copied inline from official Flux 2 repo)
# ===========================================================================


@dataclass
class AutoEncoderParams:
    resolution: int = 256
    in_channels: int = 3
    ch: int = 128
    out_ch: int = 3
    ch_mult: list = field(default_factory=lambda: [1, 2, 4, 4])
    num_res_blocks: int = 2
    z_channels: int = 32


FLUX2_VAE_PARAMS = AutoEncoderParams()


def swish(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


class AttnBlock(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def attention(self, h: torch.Tensor) -> torch.Tensor:
        B, C, H, W = h.shape
        q = self.q(h).reshape(B, 1, C, H * W).transpose(2, 3)
        k = self.k(h).reshape(B, 1, C, H * W).transpose(2, 3)
        v = self.v(h).reshape(B, 1, C, H * W).transpose(2, 3)
        h = F.scaled_dot_product_attention(q, k, v)
        return h.transpose(2, 3).reshape(B, C, H, W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.proj_out(self.attention(self.norm(x)))


class ResnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = None):
        super().__init__()
        self.in_channels = in_channels
        out_channels = out_channels or in_channels
        self.out_channels = out_channels

        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_channels, eps=1e-6, affine=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if in_channels != out_channels:
            self.nin_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.nin_shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(swish(self.norm1(x)))
        h = self.conv2(swish(self.norm2(h)))
        return self.nin_shortcut(x) + h


class Downsample(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (0, 1, 0, 1), mode="constant", value=0)
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Cast to float32 before interpolate for bfloat16 safety (matching Wan VAE convention)
        x = F.interpolate(x.float(), scale_factor=2.0, mode="nearest").type_as(x)
        return self.conv(x)


class Encoder(nn.Module):
    def __init__(
        self,
        resolution: int,
        in_channels: int,
        ch: int,
        ch_mult: list,
        num_res_blocks: int,
        z_channels: int,
    ):
        super().__init__()
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks

        # Flux2: quant_conv after encoder output
        self.quant_conv = nn.Conv2d(2 * z_channels, 2 * z_channels, 1)

        # downsampling
        self.conv_in = nn.Conv2d(in_channels, ch, kernel_size=3, stride=1, padding=1)

        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()
        block_in = ch
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for _ in range(num_res_blocks):
                block.append(ResnetBlock(block_in, block_out))
                block_in = block_out
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in)
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(block_in, block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(block_in, block_in)

        # end
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(block_in, 2 * z_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv_in(x)
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](h)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
            if i_level != self.num_resolutions - 1:
                h = self.down[i_level].downsample(h)
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)
        h = swish(self.norm_out(h))
        h = self.conv_out(h)
        h = self.quant_conv(h)
        return h


class Decoder(nn.Module):
    def __init__(
        self,
        ch: int,
        out_ch: int,
        ch_mult: list,
        num_res_blocks: int,
        z_channels: int,
    ):
        super().__init__()
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks

        # Flux2: post_quant_conv before decoder
        self.post_quant_conv = nn.Conv2d(z_channels, z_channels, 1)

        block_in = ch * ch_mult[-1]

        # z to block_in
        self.conv_in = nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(block_in, block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(block_in, block_in)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for _ in range(num_res_blocks + 1):
                block.append(ResnetBlock(block_in, block_out))
                block_in = block_out
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in)
            self.up.insert(0, up)

        # end
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = self.post_quant_conv(z)

        # get dtype for proper tracing
        upscale_dtype = next(self.up.parameters()).dtype

        h = self.conv_in(z)
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # cast to proper dtype
        h = h.to(upscale_dtype)
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)
        h = swish(self.norm_out(h))
        h = self.conv_out(h)
        return h


class AutoEncoder(nn.Module):
    """Flux 2 AutoEncoder with BatchNorm2d normalization and 2x2 patchification.

    encode: image → encoder → take mean → patchify 2x2 → BN normalize
            (B, 3, H, W) → (B, 128, H/16, W/16)
    decode: BN inv_normalize → unpatchify 2x2 → decoder
            (B, 128, H/16, W/16) → (B, 3, H, W)
    """

    def __init__(self, params: AutoEncoderParams = None):
        super().__init__()
        if params is None:
            params = FLUX2_VAE_PARAMS
        self.encoder = Encoder(
            resolution=params.resolution,
            in_channels=params.in_channels,
            ch=params.ch,
            ch_mult=params.ch_mult,
            num_res_blocks=params.num_res_blocks,
            z_channels=params.z_channels,
        )
        self.decoder = Decoder(
            ch=params.ch,
            out_ch=params.out_ch,
            ch_mult=params.ch_mult,
            num_res_blocks=params.num_res_blocks,
            z_channels=params.z_channels,
        )

        self.bn_eps = 1e-4
        self.bn_momentum = 0.1
        self.ps = [2, 2]
        self.bn = nn.BatchNorm2d(
            math.prod(self.ps) * params.z_channels,
            eps=self.bn_eps,
            momentum=self.bn_momentum,
            affine=False,
            track_running_stats=True,
        )

    def normalize(self, z: torch.Tensor) -> torch.Tensor:
        self.bn.eval()
        return self.bn(z)

    def inv_normalize(self, z: torch.Tensor) -> torch.Tensor:
        self.bn.eval()
        s = torch.sqrt(self.bn.running_var.view(1, -1, 1, 1) + self.bn_eps)
        m = self.bn.running_mean.view(1, -1, 1, 1)
        return z * s + m

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # Support 5D (B, C, T, H, W) for pipeline compatibility
        if x.ndim == 5:
            assert x.shape[2] == 1, f"Image-only VAE requires T=1, got T={x.shape[2]}"
            x = x.squeeze(2)
            video_format_input = True
        else:
            video_format_input = False

        moments = self.encoder(x)
        mean = torch.chunk(moments, 2, dim=1)[0]

        # Patchify: (B, z_ch, H/8, W/8) → (B, z_ch*4, H/16, W/16)
        z = rearrange(
            mean,
            "... c (i pi) (j pj) -> ... (c pi pj) i j",
            pi=self.ps[0],
            pj=self.ps[1],
        )
        z = self.normalize(z)

        if video_format_input:
            z = z.unsqueeze(2)  # (B, 128, 1, h, w)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        # Support 5D (B, C, T, H, W) for pipeline compatibility
        if z.ndim == 5:
            assert z.shape[2] == 1, f"Image-only VAE requires T=1, got T={z.shape[2]}"
            z = z.squeeze(2)
            video_format_input = True
        else:
            video_format_input = False

        z = self.inv_normalize(z)
        # Unpatchify: (B, z_ch*4, H/16, W/16) → (B, z_ch, H/8, W/8)
        z = rearrange(
            z,
            "... (c pi pj) i j -> ... c (i pi) (j pj)",
            pi=self.ps[0],
            pj=self.ps[1],
        )
        dec = self.decoder(z)

        if video_format_input:
            dec = dec.unsqueeze(2)  # (B, 3, 1, H, W)
        return dec

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


# ===========================================================================
# Layer 2 — Factory function
# ===========================================================================


def _flux2_vae(
    pretrained_path: str = None,
    device: str = "cpu",
    s3_credential_path: str = "credentials/s3_training.secret",
) -> AutoEncoder:
    """Build Flux 2 AutoEncoder with optional checkpoint loading.

    Uses meta-device init + rank-0 loading + sync_model_states, same pattern as Flux 1 VAE.
    Supports .safetensors and .pth via load_state_dict() from pid._src.models.utils.
    Also supports HuggingFace download from black-forest-labs/FLUX.2-dev.
    """
    params = FLUX2_VAE_PARAMS

    with torch.device("meta"):
        model = AutoEncoder(params)

    if pretrained_path is None:
        model.to_empty(device=device)
    else:
        if get_rank() == 0:
            # Try HuggingFace download if path looks like a repo ID
            if pretrained_path.startswith("hf://"):
                import huggingface_hub

                repo_id = pretrained_path[len("hf://") :]
                actual_path = huggingface_hub.hf_hub_download(
                    repo_id=repo_id,
                    filename="ae.safetensors",
                    repo_type="model",
                )
                ckpt = load_state_dict(actual_path)
            else:
                ckpt = load_state_dict(
                    pretrained_path,
                    s3_credential_path=s3_credential_path if pretrained_path.startswith("s3://") else None,
                )
            log.info(f"Loading Flux 2 VAE from {pretrained_path}")
            model.load_state_dict(ckpt, assign=True)
            model.to(device)
        else:
            model.to_empty(device=device)
    sync_model_states(model)

    return model


# ===========================================================================
# Layer 3 — Flux2VAE wrapper (dtype / AMP handling)
# ===========================================================================


class Flux2VAE:
    """Wrapper with dtype/AMP handling. All tensors are 4D (B, C, H, W).

    Unlike Flux 1, Flux 2 uses BatchNorm2d running stats for normalization
    and 2x2 patchification. Effective latent: 128 channels at 16x spatial.
    """

    def __init__(
        self,
        vae_pth: str = "./checkpoints/flux2_ae.safetensors",
        s3_credential_path: str = "credentials/s3_training.secret",
        dtype: torch.dtype = torch.float,
        device: str = "cuda",
        is_amp: bool = True,
    ):
        self.dtype = dtype
        self.device = device
        self.scale = None

        self.model = _flux2_vae(
            pretrained_path=vae_pth,
            device=device,
            s3_credential_path=s3_credential_path,
        )
        self.model = self.model.eval().requires_grad_(False)
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
        """images: (B, 3, H, W) in [-1, 1]. Returns (B, 128, H/16, W/16)."""
        in_dtype = images.dtype
        with self.context:
            if not self.is_amp:
                images = images.to(self.dtype)
            latent = self.model.encode(images)
        return latent.to(in_dtype)

    @torch.no_grad()
    def decode(self, zs: torch.Tensor) -> torch.Tensor:
        """zs: (B, 128, h, w). Returns (B, 3, H, W)."""
        in_dtype = zs.dtype
        with self.context:
            if not self.is_amp:
                zs = zs.to(self.dtype)
            recon = self.model.decode(zs)
        return recon.to(in_dtype)


# ===========================================================================
# Layer 4 — Flux2VAEInterface(VideoTokenizerInterface)
# ===========================================================================


class Flux2VAEInterface(VideoTokenizerInterface):
    """Pipeline-compatible interface for Flux 2 VAE. Image-only (temporal_compression_factor=1).

    Latent shape: (B, 128, H/16, W/16) — 32 z_channels * 4 from 2x2 patchification.
    """

    def __init__(self, chunk_duration: int = 1, **kwargs):
        self.model = Flux2VAE(
            dtype=torch.bfloat16,
            is_amp=False,
            vae_pth=kwargs.get("vae_pth", "./checkpoints/flux2_ae.safetensors"),
            s3_credential_path=kwargs.get("s3_credential_path", "credentials/s3_training.secret"),
        )
        self.chunk_duration = chunk_duration

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

        latent = self.model.encode(x)  # (B, 128, h, w)
        return latent.unsqueeze(2)  # (B, 128, 1, h, w)

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
        return 128

    @property
    def spatial_resolution(self):
        return 512

    @property
    def name(self):
        return "flux2_vae_tokenizer"


# ===========================================================================
# Layer 5 — LazyDict config
# ===========================================================================

Flux2VAEConfig: LazyDict = L(Flux2VAEInterface)(
    name="flux2_vae_tokenizer",
)
