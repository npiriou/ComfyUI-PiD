# Shared run_demo_from_clean() entrypoint for the from-clean decoder demo.
#
# Input modes (mutually exclusive, one required):
#   --input_path <PATH>       single image + a single --prompt (or fixed_positive_prompt)
#   --manifest   <JSONL>      one {"image": <path>, "prompt": <str>} per line; the
#                             "prompt" key is optional (falls back to --prompt → fixed
#                             prompt). Samples are round-robin sharded across ranks
#                             when launched with torchrun --nproc_per_node>1.
#
# Pipeline (no latent diffusion model is run):
#   1. Load a single user-supplied image (--input_path or one entry from --manifest).
#   2. Center-crop + bicubic-resize to a square --input_resolution (default 512).
#      Should match the tokenizer's native pixel resolution:
#        Flux / SD3 / Flux2 VAE         : 512
#        DINOv2-RAE (dinov2_vae.py)     : 512
#        Scale-RAE  (scale_rae_vae.py)  : 256
#   3. VAE-encode it via the loaded pixel-decoder model's own VAE (model.vae_encoder).
#      Whichever tokenizer the model was trained with (Flux / SD3 / Flux2 standard VAEs
#      OR the SigLIP/DINOv2 representation autoencoders in tokenizers/scale_rae_vae.py
#      and tokenizers/dinov2_vae.py) is the one we use here — latent format is
#      guaranteed consistent and no separate HF pipeline load is needed.
#   4. For each σ in --degrade_sigmas (default [0.0]):
#        - Optional noise:  x_t = (1-σ)·x_0 + σ·ε
#        - VAE decode    -> baseline image at the tokenizer's NATIVE output pixel
#                           size (e.g. 256 for Scale-RAE, 512 for DINOv2-RAE/Flux —
#                           may differ from --input_resolution when a RAE-style
#                           tokenizer has a fixed decode resolution).
#        - Pixel decoder -> ours image (LQ_video_or_image=baseline, LQ_latent=x_t, sigma=σ)
#                           at vae_native * --scale resolution.
#   5. Save side-by-side and (optionally) async-upload to S3 under the same flat
#      one-level <experiment_name> layout as _demo_common.py (expected by
#      scripts/comparsion_display_presigned.py), collapsing tag/sigma into a
#      single segment so the same checkpoint across VAEs / sigmas lands in
#      distinct experiment folders:
#        s3://pid/streamlit_assets/<group_name>/<tag>_<sigma_label>/<filename>
#        s3://pid/streamlit_assets/<group_name>/<backbone_tag>_vae_decode_<sigma_label>/<filename>
#        s3://pid/streamlit_assets/<group_name>/<backbone_tag>_input/<filename>
#
# Per-VAE wrapper files (flux_vae.py, sd3_vae.py, flux2_vae.py, rae_vae.py,
# scale_rae_vae.py) call run_demo_from_clean() with a backbone_tag — that tag
# is purely cosmetic (prepended to the run tag); the actual VAE always comes
# from the model checkpoint.

import argparse
import json
import logging
import os
from typing import List, Optional, Tuple

import torch
from PIL import Image

from pid._src.inference._demo_common import AsyncUploader, save_image
from pid._src.inference.checkpoint_registry import VALID_CKPT_TYPES, get_pid_checkpoint
from pid._src.inference.inference_utils import (
    generate_tag_from_checkpoint,
    get_rank_and_world_size,
    maybe_upload_video,
)
from pid._src.utils.model_loader import load_model_from_checkpoint

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

torch.enable_grad(False)


# =============================================================================
# Image I/O
# =============================================================================


def _load_samples(args) -> List[Tuple[str, Optional[str]]]:
    """Resolve the (image_path, per_sample_prompt_or_None) list from argparse.

    --input_path : returns a single-element list, prompt always None (defers to --prompt /
                   fixed_positive_prompt downstream).
    --manifest   : reads a JSONL file; each object must have an "image" key and may
                   optionally carry a "prompt" key. Per-line prompts override --prompt.
    """
    if args.manifest is not None:
        samples: List[Tuple[str, Optional[str]]] = []
        with open(args.manifest, "r") as f:
            for ln, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"--manifest {args.manifest} line {ln} is not valid JSON: {e}")
                if "image" not in obj:
                    raise ValueError(f'--manifest {args.manifest} line {ln} missing "image" key: {obj!r}')
                samples.append((str(obj["image"]), obj.get("prompt")))
        if not samples:
            raise ValueError(f"--manifest {args.manifest} is empty after stripping.")
        return samples
    return [(args.input_path, None)]


def _load_input_image(
    path: str,
    resolution: int,
    keep_input_size: bool = False,
    pad_to_multiple: int = 16,
) -> torch.Tensor:
    """Load image and return [1, 3, H, W] float32 in [-1, 1] on CPU.

    Default: center-crop to square, bicubic-resize to (resolution, resolution).
    keep_input_size=True: preserve native H, W (only center-crop a few pixels so each
    side is a multiple of pad_to_multiple, which keeps the VAE latent grid integer).
    """
    img = Image.open(path).convert("RGB")
    if keep_input_size:
        w, h = img.size
        new_w = (w // pad_to_multiple) * pad_to_multiple
        new_h = (h // pad_to_multiple) * pad_to_multiple
        if new_w == 0 or new_h == 0:
            raise ValueError(f"Image {path} size {w}x{h} is smaller than pad_to_multiple={pad_to_multiple}.")
        if (new_w, new_h) != (w, h):
            left = (w - new_w) // 2
            top = (h - new_h) // 2
            img = img.crop((left, top, left + new_w, top + new_h))
    else:
        w, h = img.size
        side = min(w, h)
        left = (w - side) // 2
        top = (h - side) // 2
        img = img.crop((left, top, left + side, top + side))
        img = img.resize((resolution, resolution), Image.BICUBIC)

    import numpy as np

    arr = np.asarray(img, np.uint8).astype("float32")
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
    return t


# =============================================================================
# Argument parsing
# =============================================================================


def _build_parser(backbone_tag: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=f"From-clean demo: image -> {backbone_tag} VAE encode -> optional noise -> ours pixel decoder"
    )

    # Our pixel decoder.
    # --experiment / --checkpoint_path default to whatever checkpoint_registry.py registers
    # for this backbone_tag — pass them explicitly only to override.
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="Our pixel decoder experiment config name (default: checkpoint_registry[backbone_tag].experiment)",
    )
    parser.add_argument(
        "--config_file",
        type=str,
        default="pid/_src/configs/pid/config.py",
        help="Hydra config file for our decoder",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Path to our pixel decoder checkpoint (default: checkpoint_registry[backbone_tag].checkpoint_path)",
    )
    parser.add_argument(
        "--pid_ckpt_type",
        type=str,
        choices=list(VALID_CKPT_TYPES),
        default="2k",
        help="Which PiD checkpoint variant to load from the registry when "
        "--experiment / --checkpoint_path are omitted. Default: '2k' (the "
        "original 2048px-trained decoders). '2kto4k' picks the multi-res-"
        "trained decoders (1024 LDM → 4K output) for flux/flux2/sd3/zimage.",
    )
    parser.add_argument("--load_ema_to_reg", action="store_true", help="Load EMA weights into the regular model")

    # Input source (mutually exclusive, required): single --input_path OR a JSONL --manifest.
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input_path", type=str, default=None, help="Path to a single input image (PNG/JPG/...).")
    input_group.add_argument(
        "--manifest",
        type=str,
        default=None,
        help='JSONL file with one {"image": <path>, "prompt": <str>} object per line. '
        'The "prompt" key is optional and falls back to --prompt → model.config.fixed_positive_prompt. '
        "Image paths are interpreted as absolute or CWD-relative. Samples are round-robin sharded across ranks.",
    )

    # Text prompt — used as a global default (or only prompt under --input_path mode).
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Text prompt describing the input image — fed into the pixel decoder's data batch as the "
        "caption condition. Under --input_path, this is THE prompt. Under --manifest, this is the "
        "fallback used for entries without a per-line 'prompt' key. If omitted entirely, falls back "
        "to model.config.fixed_positive_prompt when model.config.use_fixed_prompt is True; otherwise "
        "the script raises for any sample with no resolvable caption.",
    )
    parser.add_argument(
        "--input_resolution",
        type=int,
        default=512,
        help="Square resolution to center-crop + bicubic-resize the input image to before VAE encode. "
        "Should match the tokenizer's native input resolution: 512 for Flux / SD3 / Flux2 / DINOv2-RAE; "
        "256 for Scale-RAE (SigLIP-2 So400M @ patch14, native 16×16 token grid). Default: 512. "
        "Ignored when --keep_input_size is set.",
    )
    parser.add_argument(
        "--keep_input_size",
        action="store_true",
        help="Skip center-crop + square resize; feed the image at its native H, W into the VAE "
        "(only a small center-crop to make each side a multiple of 16 so the latent grid is integer). "
        "Useful when inputs are already bucket-resized to model-friendly resolutions.",
    )

    # Noise sweep
    parser.add_argument(
        "--degrade_sigmas",
        type=float,
        nargs="+",
        default=[0.0],
        help="List of sigma values (each in [0, 1]) to inject into the clean latent before decoding. "
        "x_t = (1 - sigma) * x_0 + sigma * eps. 0.0 = clean round-trip. One decode + save per sigma.",
    )

    # Pixel decoder inference params (match the existing demo's flag names)
    parser.add_argument("--seed", type=int, default=5, help="Base random seed (resets the noise generator per sigma)")
    parser.add_argument("--cfg_scale", type=float, default=1.0, help="Our pixel decoder CFG scale")
    parser.add_argument(
        "--pid_inference_steps",
        type=int,
        default=None,
        help="Pixel-diffusion decoder denoising steps (default from model config)",
    )
    parser.add_argument("--shift", type=float, default=None, help="Our pixel decoder flow shift")
    parser.add_argument(
        "--scale", type=int, default=4, help="Our decoder upscale factor (output = input_resolution * scale)"
    )

    # Output / S3
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help=f"Output directory. Default: ./results/official_demo_from_clean/{backbone_tag}",
    )
    parser.add_argument(
        "--save_format",
        type=str,
        choices=["png", "jpg"],
        default="jpg",
        help="Image format for saved outputs (jpg uses quality=95)",
    )
    parser.add_argument("--upload", action="store_true", help="Upload results to S3 (async)")
    parser.add_argument("--group_name", type=str, default="official_demo_from_clean", help="S3 group name")
    parser.add_argument("--note", type=str, default="", help="Note appended to tag")

    return parser


def parse_args(backbone_tag: str) -> argparse.Namespace:
    parser = _build_parser(backbone_tag)
    args, unknown = parser.parse_known_args()
    args.extra_experiment_opts = unknown
    args.backbone_tag = backbone_tag

    # Fill in --experiment / --checkpoint_path from the official registry when omitted.
    if args.experiment is None or args.checkpoint_path is None:
        default_ckpt = get_pid_checkpoint(backbone_tag, args.pid_ckpt_type)
        if args.experiment is None:
            args.experiment = default_ckpt.experiment
        if args.checkpoint_path is None:
            args.checkpoint_path = default_ckpt.checkpoint_path

    for s in args.degrade_sigmas:
        if not (0.0 <= s <= 1.0):
            parser.error(f"--degrade_sigmas value {s} out of range [0.0, 1.0]")
    return args


# =============================================================================
# VAE helpers
# =============================================================================


def _vae_decode(model, latent_4d: torch.Tensor) -> torch.Tensor:
    """Wrap model.vae_encoder.decode to handle the 5D <-> 4D shape contract.

    Input  latent_4d: [B, C, zH, zW]
    Output recon:     [B, 3, H, W] in [-1, 1]
    """
    z5 = latent_4d.unsqueeze(2)  # [B, C, 1, zH, zW]
    recon5 = model.vae_encoder.decode(z5)  # [B, 3, 1, H, W]
    if recon5.ndim == 5:
        recon5 = recon5[:, :, 0]  # [B, 3, H, W]
    return recon5


def _add_noise(clean_latent: torch.Tensor, sigma: float, generator: torch.Generator) -> torch.Tensor:
    """x_t = (1 - sigma) * x_0 + sigma * eps."""
    if sigma <= 0.0:
        return clean_latent
    noise = torch.randn(
        clean_latent.shape,
        generator=generator,
        device=clean_latent.device,
        dtype=clean_latent.dtype,
    )
    return (1.0 - sigma) * clean_latent + sigma * noise


# =============================================================================
# Main demo
# =============================================================================


def run_demo_from_clean(backbone_tag: str):
    args = parse_args(backbone_tag)
    rank, world_size = get_rank_and_world_size()
    if world_size > 1:
        torch.cuda.set_device(rank)
    is_rank0 = rank == 0

    # ---- Tag (mirror the existing demo) ----
    extra_params = {"cfg": args.cfg_scale}
    if args.pid_inference_steps is not None:
        extra_params["steps"] = args.pid_inference_steps
    if args.shift is not None:
        extra_params["shift"] = args.shift
    if args.note:
        extra_params["note_"] = args.note
    base_tag = generate_tag_from_checkpoint(args.checkpoint_path, extra_params, load_ema=args.load_ema_to_reg)
    tag = f"{backbone_tag}_{base_tag}"

    if is_rank0:
        logger.info(
            f"Backbone(VAE): {backbone_tag}  input_resolution: {args.input_resolution}  "
            f"sigmas: {sorted(args.degrade_sigmas)}  scale: {args.scale}  pid_steps: {args.pid_inference_steps}"
        )
        logger.info(f"Tag: {tag}")

    experiment_opts = list(args.extra_experiment_opts) if args.extra_experiment_opts else []
    if is_rank0 and experiment_opts:
        logger.info(f"Extra experiment options: {experiment_opts}")

    # ---- Load model (also loads its VAE) ----
    if is_rank0:
        logger.info(f"Loading our pixel decoder from {args.checkpoint_path} ...")
    model, _config = load_model_from_checkpoint(
        experiment_name=args.experiment,
        checkpoint_path=args.checkpoint_path,
        config_file=args.config_file,
        enable_fsdp=False,
        experiment_opts=experiment_opts,
        strict=False,
        load_ema_to_reg=args.load_ema_to_reg,
    )
    model.eval()

    # ---- Resolve default text prompt.
    # PiD requires a caption in the data batch. Order of fallback applied per sample:
    #   1) per-sample "prompt" from --manifest line  (manifest mode only)
    #   2) --prompt CLI flag                          (global default)
    #   3) model.config.fixed_positive_prompt         (when use_fixed_prompt=True)
    #   4) ValueError                                 (no caption resolvable)
    fixed_prompt = model.config.fixed_positive_prompt if getattr(model.config, "use_fixed_prompt", False) else None
    if is_rank0 and fixed_prompt is not None and args.prompt is None:
        logger.info(f"Default caption falls back to model's fixed prompt: {fixed_prompt[:80]}...")

    # ---- Output dirs / uploader ----
    output_dir = args.output_dir or f"./results/official_demo_from_clean/{backbone_tag}"
    os.makedirs(output_dir, exist_ok=True)
    if is_rank0:
        logger.info(f"Outputs -> {output_dir}")
    uploader = AsyncUploader(max_workers=8) if args.upload else None

    # ---- Resolve sample list (single image OR JSONL manifest) and shard across ranks ----
    samples_all = _load_samples(args)
    indexed_samples = list(enumerate(samples_all))
    if world_size > 1:
        my_samples = indexed_samples[rank::world_size]
        logger.info(f"[Rank {rank}/{world_size}] Processing {len(my_samples)} of {len(samples_all)} samples")
    else:
        my_samples = indexed_samples
        if is_rank0:
            logger.info(f"Processing {len(my_samples)} sample(s)")

    # Track whether we've already logged the VAE-native resolution mismatch banner
    # — the latent grid is fixed across samples (same VAE + same --input_resolution),
    # so the message would otherwise repeat per sample.
    _vae_native_logged = False

    for idx, (image_path, per_sample_prompt) in my_samples:
        # ---- Resolve caption for this sample ----
        caption = per_sample_prompt or args.prompt or fixed_prompt
        if caption is None:
            raise ValueError(
                f"Sample idx={idx} image={image_path!r} has no resolvable caption — "
                f"provide a per-line 'prompt' in the manifest, --prompt, or enable "
                f"use_fixed_prompt in the model config."
            )

        # ---- Filename layout: under --manifest disambiguate with idx prefix; under
        # --input_path keep the bare basename (preserves the prior single-image UX).
        bn = os.path.splitext(os.path.basename(image_path))[0]
        sample_id = f"{idx:08d}_{bn}" if args.manifest is not None else bn

        # ---- Load + encode ----
        input_tensor = _load_input_image(image_path, args.input_resolution, keep_input_size=args.keep_input_size).to(
            dtype=torch.bfloat16, device="cuda"
        )
        clean_latent = model.encode_lq_latent(input_tensor)  # [1, C, zH, zW]

        # ---- Derive VAE-native pixel size from the latent grid times the tokenizer's
        # spatial_compression_factor. This is the resolution the VAE decoder will
        # produce — for standard VAEs (Flux/SD3/Flux2) it equals --input_resolution;
        # for RAE-style tokenizers with a fixed decode resolution (Scale-RAE → 256,
        # DINOv2-RAE → 512) it is independent of --input_resolution. We anchor
        # target_hw to this VAE-native size so the LQ image fed to the pixel decoder
        # and the SR output stay consistent with the model's training-time scale
        # = SR_out / vae_native, regardless of how the user pre-resized the input.
        vae_compression = int(model.vae_encoder.spatial_compression_factor)
        vae_h = int(clean_latent.shape[-2]) * vae_compression
        vae_w = int(clean_latent.shape[-1]) * vae_compression
        target_hw = (vae_h * args.scale, vae_w * args.scale)
        if not _vae_native_logged:
            logger.info(
                f"[idx={idx}] Clean latent shape={tuple(clean_latent.shape)}  "
                f"vae_native=({vae_h}x{vae_w})  target_hw={target_hw}  caption={caption[:60]!r}"
            )
            if (vae_h, vae_w) != (args.input_resolution, args.input_resolution):
                logger.info(
                    f"VAE-native output size = ({vae_h}x{vae_w}) differs from --input_resolution "
                    f"{args.input_resolution} (typical for RAE-style fixed-resolution decoders). "
                    f"Using vae_native * --scale = {target_hw} as the SR target / display resolution."
                )
            _vae_native_logged = True

        # ---- Save the input itself at its NATIVE resolution (no upsample). The VAE
        # baseline + ours outputs are bicubic-upsampled to target_hw inside the loop
        # for fair side-by-side display, but the input copy is what the user fed in. ----
        input_save = input_tensor.float().cpu().squeeze(0).clamp(-1, 1)
        input_path_out = os.path.join(output_dir, "input", f"{sample_id}.{args.save_format}")
        save_image(input_save, input_path_out)
        if args.upload:
            input_upload_tag = f"{backbone_tag}_input"
            if uploader is not None:
                uploader.submit(maybe_upload_video, input_path_out, input_upload_tag, True, args.group_name)
            else:
                maybe_upload_video(input_path_out, input_upload_tag, True, args.group_name)

        # ---- σ sweep ----
        for sigma in sorted(args.degrade_sigmas):
            sigma_label = f"sigma_{sigma:.3f}"

            # Per-σ deterministic noise generator (re-seeded so the same σ always gives the same noise)
            gen = torch.Generator(device="cuda").manual_seed(args.seed + idx)
            latent = _add_noise(clean_latent.float(), float(sigma), gen).to(dtype=torch.bfloat16)

            # VAE decode (baseline)
            with torch.no_grad():
                vae_img = _vae_decode(model, latent)  # [1, 3, R, R] in [-1, 1]

            # Pixel decoder (ours).
            # LQ_video_or_image is a zeros placeholder — the model conditions on
            # LQ_latent + degrade_sigma + caption only; the pixel-domain LQ branch
            # is unused for the from-clean demo. We still keep vae_img above for
            # saving the VAE baseline image to disk.
            lq_placeholder = torch.zeros_like(vae_img, dtype=torch.bfloat16, device="cuda")
            data_batch = {
                model.config.input_caption_key: [caption],
                "LQ_video_or_image": lq_placeholder,
                "LQ_latent": latent.to(dtype=torch.bfloat16, device="cuda"),
                "degrade_sigma": torch.tensor([float(sigma)], device="cuda", dtype=torch.float32),
            }
            samples_out = model.generate_samples_from_batch(
                data_batch,
                cfg_scale=args.cfg_scale,
                num_steps=args.pid_inference_steps,
                seed=args.seed + idx,
                shift=args.shift,
                image_size=target_hw,
            )
            ours_img = samples_out[0].float().cpu().clamp(-1, 1)  # [C, 1, H_out, W_out]

            # Save ours (native SR resolution)
            ours_path = os.path.join(output_dir, tag, sigma_label, f"{sample_id}.{args.save_format}")
            save_image(ours_img, ours_path)

            # Save VAE baseline at its native resolution (no bicubic upsampling).
            vae_path = os.path.join(output_dir, "vae_decode", sigma_label, f"{sample_id}.{args.save_format}")
            save_image(vae_img.float().cpu().squeeze(0).clamp(-1, 1), vae_path)

            logger.info(f"[idx={idx}] sigma={sigma:.3f} -> ours={ours_path}  vae={vae_path}")

            # S3 upload — flat one-level experiment_name expected by
            #   scripts/comparsion_display_presigned.py:
            #     s3://<bucket>/streamlit_assets/<group>/<experiment_name>/<filename>
            # so we collapse "<tag>/<sigma_label>" into "<tag>_<sigma_label>". The VAE
            # side uses "<backbone_tag>_vae_decode_<sigma_label>" to avoid cross-VAE
            # collisions when sharing --group_name.
            if args.upload:
                ours_upload_tag = f"{tag}_{sigma_label}"
                vae_upload_tag = f"{backbone_tag}_vae_decode_{sigma_label}"
                if uploader is not None:
                    uploader.submit(maybe_upload_video, ours_path, ours_upload_tag, True, args.group_name)
                    uploader.submit(maybe_upload_video, vae_path, vae_upload_tag, True, args.group_name)
                else:
                    maybe_upload_video(ours_path, ours_upload_tag, True, args.group_name)
                    maybe_upload_video(vae_path, vae_upload_tag, True, args.group_name)

    if uploader is not None:
        logger.info(f"[Rank {rank}] Waiting for background uploads to complete ...")
        uploader.wait()

    if is_rank0:
        logger.info(f"Done! Results saved under {output_dir}")
