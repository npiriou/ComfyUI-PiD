# Scale-RAE ViT-MAE decoder (ported from
# https://github.com/nyu-visionx/Scale-RAE — scale_rae/model/multimodal_decoder/decoder.py:805-1015).
#
# Mirrors the upstream `GeneralDecoder` 1:1 so the published Scale-RAE
# checkpoints (e.g. siglip2_sop14_i224_web73M_ganw3_decXL.pt) load cleanly with
# strict=True. Differences vs the sibling `rae_decoder.py` (DINOv2-RAE):
#   * `forward` returns `ViTMAEDecoderOutput` (has `.logits` attribute).
#   * `interpolate_pos_encoding` is provided for variable token grids.
#   * `set_trainable_cls_token(tensor=None)` setter for parity with upstream.
#   * `forward(.., drop_cls_token=False)` raises NotImplementedError, matching
#     the upstream `else` branch — Scale-RAE's published decoders are always
#     called with `drop_cls_token=True`.
#   * `forward(.., skip_interpolate_latent=True)` lets a caller feed an
#     arbitrary token grid (e.g. 32×32 = 1024 tokens) instead of the
#     pretrained 16×16; without this flag, `interpolate_latent` always
#     down/up-samples back to `num_patches`. Pair with
#     `interpolate_pos_encoding=True` so the sin-cos pos embed is bicubic
#     resized to match the input grid.

from copy import deepcopy
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from transformers.modeling_outputs import ModelOutput
from transformers.models.vit_mae.configuration_vit_mae import ViTMAEConfig
from transformers.models.vit_mae.modeling_vit_mae import ViTMAELayer


def _get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


def _get_2d_sincos_pos_embed_from_grid(embed_dim: int, grid: np.ndarray) -> np.ndarray:
    assert embed_dim % 2 == 0
    emb_h = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)


def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int, add_cls_token: bool = False) -> np.ndarray:
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # w goes first
    grid = np.stack(grid, axis=0).reshape([2, 1, grid_size, grid_size])
    pos_embed = _get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if add_cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


@dataclass
class ViTMAEDecoderOutput(ModelOutput):
    logits: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


class GeneralDecoder(nn.Module):
    """Scale-RAE ViT-MAE decoder.

    Input convention: `(B, num_patches+1, hidden_size)` — first token is a
    placeholder CLS that is stripped (`drop_cls_token=True`) and replaced by
    a learned `trainable_cls_token`. Patch tokens are bilinearly rescaled to
    the decoder's training grid (`interpolate_latent`) when needed.

    Output: `ViTMAEDecoderOutput(logits=(B, num_patches, patch_size**2 * num_channels))`.
    """

    def __init__(self, config: ViTMAEConfig, num_patches: int):
        super().__init__()
        self.decoder_embed = nn.Linear(config.hidden_size, config.decoder_hidden_size, bias=True)
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, config.decoder_hidden_size), requires_grad=False
        )

        decoder_config = deepcopy(config)
        decoder_config.hidden_size = config.decoder_hidden_size
        decoder_config.num_hidden_layers = config.decoder_num_hidden_layers
        decoder_config.num_attention_heads = config.decoder_num_attention_heads
        decoder_config.intermediate_size = config.decoder_intermediate_size
        decoder_config._attn_implementation = getattr(config, "_attn_implementation", "sdpa") or "sdpa"

        self.decoder_layers = nn.ModuleList(
            [ViTMAELayer(decoder_config) for _ in range(config.decoder_num_hidden_layers)]
        )

        self.decoder_norm = nn.LayerNorm(config.decoder_hidden_size, eps=config.layer_norm_eps)
        self.decoder_pred = nn.Linear(config.decoder_hidden_size, config.patch_size**2 * config.num_channels, bias=True)
        self.gradient_checkpointing = False
        self.config = config
        self.num_patches = num_patches
        self.decoder_config = decoder_config

        # Init fixed sin-cos decoder pos embed (CLS slot stays zero).
        pos = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(num_patches**0.5), add_cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(pos).float().unsqueeze(0))

        self.set_trainable_cls_token()

    def set_trainable_cls_token(self, tensor: Optional[torch.Tensor] = None):
        tensor = torch.zeros(1, 1, self.decoder_config.hidden_size) if tensor is None else tensor
        self.trainable_cls_token = nn.Parameter(tensor)

    def interpolate_pos_encoding(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Bicubic-resize the patch portion of decoder_pos_embed to the input grid size."""
        embeddings_positions = embeddings.shape[1] - 1
        num_positions = self.decoder_pos_embed.shape[1] - 1

        class_pos_embed = self.decoder_pos_embed[:, 0, :]
        patch_pos_embed = self.decoder_pos_embed[:, 1:, :]
        dim = self.decoder_pos_embed.shape[-1]

        patch_pos_embed = patch_pos_embed.reshape(1, 1, -1, dim).permute(0, 3, 1, 2)
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed,
            scale_factor=(1, embeddings_positions / num_positions),
            mode="bicubic",
            align_corners=False,
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def interpolate_latent(self, x: torch.Tensor) -> torch.Tensor:
        """(B, L, C) → (B, num_patches, C), bilinear on the 2D grid."""
        b, l, c = x.shape
        if l == self.num_patches:
            return x
        h = w = int(l**0.5)
        assert h * w == l, f"cannot reshape length {l} to a square grid"
        x = x.reshape(b, h, w, c).permute(0, 3, 1, 2)
        target = int(self.num_patches**0.5)
        x = nn.functional.interpolate(x, size=(target, target), mode="bilinear", align_corners=False)
        return x.permute(0, 2, 3, 1).contiguous().view(b, self.num_patches, c)

    def unpatchify(
        self,
        patchified_pixel_values: torch.Tensor,
        original_image_size: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        patch_size, num_channels = self.config.patch_size, self.config.num_channels
        H, W = (
            original_image_size if original_image_size is not None else (self.config.image_size, self.config.image_size)
        )
        nph, npw = H // patch_size, W // patch_size
        if nph * npw != patchified_pixel_values.shape[1]:
            raise ValueError(f"patch count {patchified_pixel_values.shape[1]} does not match grid {nph}*{npw}")
        B = patchified_pixel_values.shape[0]
        patches = patchified_pixel_values.reshape(B, nph, npw, patch_size, patch_size, num_channels)
        patches = torch.einsum("nhwpqc->nchpwq", patches)
        return patches.reshape(B, num_channels, nph * patch_size, npw * patch_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        interpolate_pos_encoding: bool = False,
        drop_cls_token: bool = False,
        skip_interpolate_latent: bool = False,
    ):
        x = self.decoder_embed(hidden_states)
        x_ = x[:, 1:, :]  # input first token is a placeholder CLS, stripped here
        if drop_cls_token:
            cls_token = self.trainable_cls_token.expand(x_.shape[0], -1, -1)
            if not skip_interpolate_latent:
                x_ = self.interpolate_latent(x_)
            x = torch.cat([cls_token, x_], dim=1)
        else:
            raise NotImplementedError("drop_cls_token=False is not implemented for Scale-RAE decoder")

        if interpolate_pos_encoding:
            decoder_pos_embed = self.interpolate_pos_encoding(x)
        else:
            decoder_pos_embed = self.decoder_pos_embed
        hidden_states = x + decoder_pos_embed

        # Note: in transformers 4.57+ `ViTMAELayer.forward` no longer accepts
        # `head_mask` / `output_attentions` and returns the hidden tensor
        # directly (not a tuple). We therefore drop the optional outputs and
        # call each layer with just the hidden states — matching the sibling
        # `rae_decoder.py` convention.
        del output_attentions, output_hidden_states  # unused on this transformers version
        all_hidden_states = None
        all_self_attentions = None
        for layer_module in self.decoder_layers:
            out = layer_module(hidden_states)
            hidden_states = out[0] if isinstance(out, tuple) else out

        hidden_states = self.decoder_norm(hidden_states)
        logits = self.decoder_pred(hidden_states)
        logits = logits[:, 1:, :]  # strip CLS slot

        if not return_dict:
            return tuple(v for v in [logits, all_hidden_states, all_self_attentions] if v is not None)
        return ViTMAEDecoderOutput(
            logits=logits,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )
