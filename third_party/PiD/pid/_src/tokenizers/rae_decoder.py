# Minimal RAE ViT-MAE decoder (ported from
# https://github.com/bytetriper/RAE — src/stage1/decoders/decoder.py).
#
# Only the decoder-side pieces are needed here: a trainable CLS token is
# prepended to the patch tokens, fixed 2D sin-cos positional embeddings are
# added, a stack of ViTMAELayer blocks processes them, and a linear head
# predicts patch_size**2 * 3 values per patch which are unpatchified to
# pixels. We reuse HuggingFace's `ViTMAELayer` / `ViTMAEConfig` so the
# published RAE state_dict keys (`decoder_layers.N.attention.attention.*`
# etc.) load cleanly with strict=True.

from copy import deepcopy
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
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


class GeneralDecoder(nn.Module):
    """ViT-MAE decoder used by RAE.

    Input:  (B, N, hidden_size) patch tokens — N usually matches num_patches.
    Output: pixel reconstruction (B, 3, image_size, image_size), via
            `unpatchify(decoder_pred(decoder_layers(..)))`.

    Differences from vanilla HF `ViTMAEDecoder`:
      * A learnable ``trainable_cls_token`` is prepended (no mask tokens).
      * Input length may differ from ``num_patches``; `interpolate_latent`
        bilinearly rescales it to match the decoder's positional grid.
      * `drop_cls_token=True` lets callers pass tokens that already contain
        a CLS at index 0 (stripped and replaced). RAE's pretrained weights
        always call with `drop_cls_token=False`, matching the pure-patch
        latent produced by our DINOv2 encoder.
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
        # Newer transformers routes attention through ALL_ATTENTION_FUNCTIONS
        # keyed by `_attn_implementation`; default ("sdpa") keeps math identical
        # but dispatches via scaled_dot_product_attention.
        decoder_config._attn_implementation = getattr(config, "_attn_implementation", "sdpa") or "sdpa"
        self.decoder_layers = nn.ModuleList(
            [ViTMAELayer(decoder_config) for _ in range(config.decoder_num_hidden_layers)]
        )

        self.decoder_norm = nn.LayerNorm(config.decoder_hidden_size, eps=config.layer_norm_eps)
        self.decoder_pred = nn.Linear(config.decoder_hidden_size, config.patch_size**2 * config.num_channels, bias=True)

        self.config = config
        self.num_patches = num_patches
        self.trainable_cls_token = nn.Parameter(torch.zeros(1, 1, config.decoder_hidden_size))

        # Init fixed sin-cos decoder pos embed (CLS slot stays zero).
        pos = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(num_patches**0.5), add_cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(pos).float().unsqueeze(0))

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
        patches: torch.Tensor,
        original_image_size: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        patch_size, num_channels = self.config.patch_size, self.config.num_channels
        H, W = (
            original_image_size if original_image_size is not None else (self.config.image_size, self.config.image_size)
        )
        nph, npw = H // patch_size, W // patch_size
        assert nph * npw == patches.shape[1], f"patch count {patches.shape[1]} does not match grid {nph}*{npw}"
        B = patches.shape[0]
        patches = patches.reshape(B, nph, npw, patch_size, patch_size, num_channels)
        patches = torch.einsum("nhwpqc->nchpwq", patches)
        return patches.reshape(B, num_channels, nph * patch_size, npw * patch_size)

    def forward(self, hidden_states: torch.Tensor, drop_cls_token: bool = False) -> torch.Tensor:
        """Returns patch logits (B, num_patches, patch_size**2 * 3)."""
        x = self.decoder_embed(hidden_states)
        if drop_cls_token:
            x = self.interpolate_latent(x[:, 1:, :])
        else:
            x = self.interpolate_latent(x)
        cls_token = self.trainable_cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_token, x], dim=1) + self.decoder_pos_embed
        for layer in self.decoder_layers:
            x = layer(x)
        x = self.decoder_norm(x)
        logits = self.decoder_pred(x)
        return logits[:, 1:, :]  # strip CLS slot
