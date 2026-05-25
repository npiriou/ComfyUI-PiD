# Shared run_demo() entrypoint for the official decoder-vs-VAE demo.
#
# Pipeline:
#   1. Run a HuggingFace latent diffusion backbone (flux / flux2 / sd3 / zimage)
#      on a text prompt — OR the upstream RAE (class-conditional 512px) / Scale-RAE
#      (text-conditional 256px) latent diffusion model, neither of which lives in
#      diffusers and so goes through its own load+sample path.
#   2. Capture intermediate noisy latents xt at user-specified denoising steps via
#      XtCaptureCallback (callback_on_step_end). The final clean latent (x0) is
#      always taken from the pipeline's return value.
#   3. For each captured latent (xt at step K and the final x0):
#        VAE decode  -> baseline image
#        ours decode -> feed (VAE image as LQ_video_or_image, latent as LQ_latent,
#                            captured per-step sigma as degrade_sigma) to our
#                            pixel-diffusion decoder model
#   4. Save both per-step PNGs locally and (optionally) async-upload them to S3
#      under a flat one-level <experiment_name> layout (expected by
#      scripts/comparsion_display_presigned.py), collapsing tag/step into a
#      single segment so the same checkpoint across backbones / steps lands in
#      distinct experiment folders:
#        s3://pid/streamlit_assets/<group_name>/<tag>_step_<label>/<filename>
#        s3://pid/streamlit_assets/<group_name>/<backbone>_decode_step_<label>/<filename>
#
# Per-backbone wrapper files (from_ldm_*.py) just call run_demo(backbone="<name>");
# diffusers-backed backbones read all defaults (resolution, steps, guidance,
# extra_generate_kwargs) from pid/_src/inference/pipeline_registry.py.

import argparse
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from types import SimpleNamespace
from typing import List, Optional

import numpy as np
import torch
from PIL import Image

from pid._src.inference.checkpoint_registry import VALID_CKPT_TYPES, get_pid_checkpoint
from pid._src.inference.create_dataset import XtCaptureCallback
from pid._src.inference.inference_utils import (
    generate_tag_from_checkpoint,
    get_rank_and_world_size,
    maybe_upload_video,
)
from pid._src.inference.pipeline_registry import (
    PIPELINE_REGISTRY,
    decode_with_pipeline_vae,
    extract_latent,
    load_pipeline,
)
from pid._src.utils.model_loader import load_model_from_checkpoint

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

torch.enable_grad(False)


# =============================================================================
# Async S3 uploader
# =============================================================================


class AsyncUploader:
    """Fire-and-forget S3 uploader backed by a thread pool.

    Usage:
        uploader = AsyncUploader(max_workers=8)
        uploader.submit(maybe_upload_video, path, tag, upload, group)
        ...
        uploader.wait()   # block until all queued uploads finish
    """

    def __init__(self, max_workers: int = 8):
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._futures = []

    def submit(self, fn, *args, **kwargs):
        future = self._executor.submit(fn, *args, **kwargs)
        self._futures.append(future)

    def wait(self):
        failed = 0
        for fut in as_completed(self._futures):
            try:
                ok = fut.result()
                if ok is False:
                    failed += 1
            except Exception as e:
                logger.error(f"Async upload error: {e}")
                failed += 1
        self._futures.clear()
        self._executor.shutdown(wait=False)
        if failed:
            logger.warning(f"{failed} upload(s) failed")


# =============================================================================
# Image I/O
# =============================================================================


def _tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert [C, H, W] in [-1, 1] to PIL Image."""
    tensor = (tensor.float().clamp(-1, 1) + 1) * 127.5
    arr = tensor.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    return Image.fromarray(arr)


def save_image(sample: torch.Tensor, save_path: str, quality: int = 95) -> str:
    """Save [C, H, W] or [C, 1, H, W] tensor in [-1, 1]. Format inferred from extension."""
    if sample.dim() == 4:
        sample = sample.squeeze(1)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    img = _tensor_to_pil(sample)
    if save_path.lower().endswith((".jpg", ".jpeg")):
        img.save(save_path, quality=quality)
    else:
        img.save(save_path)
    return save_path


def _load_prompts(args) -> List[str]:
    if args.prompt is not None:
        return [args.prompt]
    with open(args.prompt_file, "r") as f:
        prompts = [line.strip() for line in f if line.strip()]
    if not prompts:
        raise ValueError(f"--prompt_file {args.prompt_file} is empty after stripping.")
    return prompts


# =============================================================================
# Argument parsing
# =============================================================================


def _build_parser(backbone: str) -> argparse.ArgumentParser:
    # Backbone-flavored parser. Diffusers-only args (--ldm_inference_steps, --guidance_scale,
    # --cpu_offload, --backbone_model_id) are skipped for "rae" and "scale_rae" because those
    # backbones bypass the diffusers pipeline. RAE is class-conditional, so --prompt /
    # --prompt_file are also omitted for it; scale_rae remains text-conditional.
    is_rae = backbone == "rae"
    is_scale_rae = backbone == "scale_rae"
    is_diffusers = not (is_rae or is_scale_rae)
    cfg = PIPELINE_REGISTRY[backbone] if is_diffusers else None

    parser = argparse.ArgumentParser(
        description=f"Official demo: {backbone} latent diffusion vs ours pixel-diffusion decoder"
    )

    # Our pixel decoder (common to all backbones).
    # --experiment / --checkpoint_path default to whatever checkpoint_registry.py registers
    # for this backbone — pass them explicitly only to override.
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="Our pixel decoder experiment config name (default: checkpoint_registry[backbone].experiment)",
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
        help="Path to our pixel decoder checkpoint (default: checkpoint_registry[backbone].checkpoint_path)",
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

    parser.add_argument(
        "--dtype",
        type=str,
        choices=["bf16", "fp32"],
        default="bf16",
        help="Backbone dtype",
    )

    # Step capture (intermediate xt) — common to all backbones; range validated per-flow.
    parser.add_argument(
        "--save_xt_steps",
        type=int,
        nargs="+",
        default=None,
        help="Capture noisy latent AFTER K forward passes for each K (1-indexed). "
        "Final clean latent (x0) is always saved. K must be in [1, num_inference_steps].",
    )

    # Our decoder inference params (common)
    parser.add_argument("--seed", type=int, default=0, help="Base random seed (incremented per prompt/class)")
    parser.add_argument("--cfg_scale", type=float, default=1.0, help="Our pixel decoder CFG scale")
    parser.add_argument(
        "--pid_inference_steps",
        type=int,
        default=None,
        help="Pixel-diffusion decoder denoising steps (default from model config)",
    )
    parser.add_argument("--shift", type=float, default=None, help="Our pixel decoder flow shift")
    parser.add_argument("--scale", type=int, default=4, help="Our decoder upscale factor (output = baseline * scale)")

    # Output / S3 (common)
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help=f"Output directory. Default: ./results/official_demo/{backbone}",
    )
    parser.add_argument(
        "--save_format",
        type=str,
        choices=["png", "jpg"],
        default="jpg",
        help="Image format for saved outputs (jpg uses quality=95)",
    )
    parser.add_argument("--upload", action="store_true", help="Upload results to S3 (async)")
    parser.add_argument("--group_name", type=str, default="official_demo", help="S3 group name")
    parser.add_argument("--note", type=str, default="", help="Note appended to tag")

    # ===== Backbone-specific =====
    if is_diffusers:
        parser.add_argument(
            "--backbone_model_id",
            type=str,
            default=None,
            help=f"Override HuggingFace model ID (default: {cfg.default_model_id})",
        )
        parser.add_argument(
            "--resolution",
            type=int,
            default=512,
            help="LDM generation resolution (square). Default: 512",
        )
        parser.add_argument(
            "--ldm_inference_steps",
            type=int,
            default=None,
            help=f"Latent diffusion backbone denoising steps. Default: {cfg.default_num_inference_steps}",
        )
        parser.add_argument(
            "--guidance_scale",
            type=float,
            default=None,
            help=f"Backbone CFG scale. Default: {cfg.default_guidance_scale}",
        )
        parser.add_argument(
            "--cpu_offload",
            action="store_true",
            help="Use enable_model_cpu_offload (needed for large models like Flux2 on small GPUs)",
        )
        prompt_group = parser.add_mutually_exclusive_group(required=True)
        prompt_group.add_argument("--prompt", type=str, default=None, help="Single inline prompt string")
        prompt_group.add_argument("--prompt_file", type=str, default=None, help="Text file with one prompt per line")
    elif is_rae:
        # Class-conditional ImageNet-512. RAE-specific flags come from
        # rae_generation.add_rae_args.
        from pid._src.inference.rae_generation import add_rae_args

        add_rae_args(parser)
        parser.add_argument(
            "--num_inference_steps",
            type=int,
            default=50,
            help="RAE ODE step count. Drives Sampler.sample_ode(num_steps=num_inference_steps+1).",
        )
        parser.add_argument(
            "--resolution",
            type=int,
            default=512,
            help="RAE backbone only supports 512.",
        )
    else:  # scale_rae
        from pid._src.inference.scale_rae_generation import add_scale_rae_args

        add_scale_rae_args(parser)
        parser.add_argument(
            "--resolution",
            type=int,
            default=256,
            help="Scale-RAE backbone only supports 256 (decoder is 14-multiple 224, bicubic to 256).",
        )
        prompt_group = parser.add_mutually_exclusive_group(required=True)
        prompt_group.add_argument("--prompt", type=str, default=None, help="Single inline prompt string")
        prompt_group.add_argument("--prompt_file", type=str, default=None, help="Text file with one prompt per line")

    return parser


def parse_args(backbone: str) -> argparse.Namespace:
    parser = _build_parser(backbone)
    args, unknown = parser.parse_known_args()
    args.extra_experiment_opts = unknown
    args.backbone = backbone

    # Fill in --experiment / --checkpoint_path from the official registry when omitted.
    if args.experiment is None or args.checkpoint_path is None:
        default_ckpt = get_pid_checkpoint(backbone, args.pid_ckpt_type)
        if args.experiment is None:
            args.experiment = default_ckpt.experiment
        if args.checkpoint_path is None:
            args.checkpoint_path = default_ckpt.checkpoint_path

    return args


# =============================================================================
# Main demo loop
# =============================================================================


def _maybe_init_distributed(world_size: int, rank: int):
    if world_size > 1:
        import torch.distributed as dist

        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        torch.cuda.set_device(rank)


def _capture_steps(
    pipeline,
    pipe_cfg,
    xt_callback: Optional[XtCaptureCallback],
    final_latent: torch.Tensor,
    H: int,
    W: int,
    dtype: torch.dtype,
    ldm_inference_steps: int,
):
    """Yield (step_label, latent_unpacked_on_cuda, sigma) for each captured step.

    - Intermediate xt at user K (sorted ascending): label = f"{K:02d}xt", sigma from callback.
    - Final clean x0: label = "x0", sigma ≈ 0 (sigmas[-1] from scheduler).

    Uses extract_latent to unpack any backbone-specific packed format (Flux/Flux2/QwenImage).
    """
    if xt_callback is not None:
        for K in sorted(xt_callback.captured.keys()):
            xt_packed_cpu, sigma = xt_callback.captured[K]
            xt_packed = xt_packed_cpu.to(device="cuda", dtype=dtype)
            xt_latent = extract_latent(pipeline, SimpleNamespace(images=xt_packed), pipe_cfg, H, W)
            yield f"{K:02d}xt", xt_latent, sigma

    final_sigma = float(pipeline.scheduler.sigmas[-1].item())
    yield "x0", final_latent, final_sigma


def run_demo(backbone: str):
    args = parse_args(backbone)

    # Non-diffusers backbones use their own load + capture + decode paths.
    if backbone == "rae":
        return _run_demo_rae(args)
    if backbone == "scale_rae":
        return _run_demo_scale_rae(args)

    rank, world_size = get_rank_and_world_size()
    _maybe_init_distributed(world_size, rank)
    is_rank0 = rank == 0

    # ---- Resolve backbone defaults ----
    pipe_cfg_default = PIPELINE_REGISTRY[backbone]
    H = W = args.resolution or pipe_cfg_default.default_resolution[0]
    ldm_inference_steps = args.ldm_inference_steps or pipe_cfg_default.default_num_inference_steps
    guidance_scale = args.guidance_scale if args.guidance_scale is not None else pipe_cfg_default.default_guidance_scale
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32

    # Validate xt step indices against the resolved ldm_inference_steps.
    save_xt_set = set(args.save_xt_steps) if args.save_xt_steps else set()
    for k in save_xt_set:
        if k < 1 or k > ldm_inference_steps:
            raise ValueError(f"--save_xt_steps value {k} out of range [1, {ldm_inference_steps}]")

    prompts = _load_prompts(args)

    # ---- Tag: backbone prefix added so demos for the same checkpoint across
    # backbones land in distinct S3 folders. ----
    tag = _build_tag(args, backbone)

    if is_rank0:
        logger.info(
            f"Backbone: {backbone}  resolution: {H}x{W}  ldm_steps: {ldm_inference_steps}  "
            f"guidance: {guidance_scale}  pid_steps: {args.pid_inference_steps}"
        )
        logger.info(f"Tag: {tag}")
        logger.info(f"#Prompts: {len(prompts)}  save_xt_steps: {sorted(save_xt_set)}  scale: {args.scale}")

    experiment_opts = list(args.extra_experiment_opts) if args.extra_experiment_opts else []
    if is_rank0 and experiment_opts:
        logger.info(f"Extra experiment options: {experiment_opts}")

    # ---- Load HF pipeline (warm-cache pattern across ranks) ----
    if world_size > 1:
        import torch.distributed as dist

        for r in range(world_size):
            if rank == r:
                msg = "from disk" if r == 0 else "from OS cache"
                logger.info(f"[Rank {rank}] Loading {backbone} pipeline ({msg}) ...")
                pipeline, pipe_cfg = load_pipeline(
                    backbone, args.backbone_model_id, dtype=dtype, cpu_offload=args.cpu_offload
                )
            dist.barrier()
    else:
        logger.info(f"Loading {backbone} pipeline ...")
        pipeline, pipe_cfg = load_pipeline(backbone, args.backbone_model_id, dtype=dtype, cpu_offload=args.cpu_offload)

    # ---- Load our pixel decoder ----
    model = _load_our_decoder(args, experiment_opts, is_rank0)

    output_dir = args.output_dir or f"./results/official_demo/{backbone}"
    os.makedirs(output_dir, exist_ok=True)
    if is_rank0:
        logger.info(f"Outputs -> {output_dir}")

    uploader = AsyncUploader(max_workers=8) if args.upload else None

    # ---- Shard prompts across ranks (round-robin keeps load balanced when len%world!=0) ----
    indexed_prompts = list(enumerate(prompts))
    if world_size > 1:
        indexed_prompts = indexed_prompts[rank::world_size]
        logger.info(f"[Rank {rank}/{world_size}] Processing {len(indexed_prompts)} prompts")

    for prompt_idx, prompt in indexed_prompts:
        seed = args.seed + prompt_idx
        sample_id = f"{prompt_idx:08d}"
        generator = torch.Generator(device="cuda").manual_seed(seed)

        xt_cb = XtCaptureCallback(save_xt_set) if save_xt_set else None

        gen_kwargs = dict(
            prompt=prompt,
            height=H,
            width=W,
            num_inference_steps=ldm_inference_steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=1,
            output_type="latent",
            generator=generator,
        )
        gen_kwargs.update(pipe_cfg.extra_generate_kwargs)
        if xt_cb is not None:
            gen_kwargs["callback_on_step_end"] = xt_cb
            gen_kwargs["callback_on_step_end_tensor_inputs"] = ["latents"]

        logger.info(f"[{prompt_idx}] Running {backbone} pipeline (seed={seed}): {prompt[:80]!r}")
        raw_output = pipeline(**gen_kwargs)
        final_latent = extract_latent(pipeline, raw_output, pipe_cfg, H, W)

        # ---- Decode each step (intermediate xt + final x0) ----
        for step_label, latent, sigma in _capture_steps(
            pipeline, pipe_cfg, xt_cb, final_latent, H, W, dtype, ldm_inference_steps
        ):
            # 1) VAE decode (baseline) — returns (1, 3, H, W) in [0, 1]
            with torch.no_grad():
                vae_img_01 = decode_with_pipeline_vae(pipeline, latent, pipe_cfg)

            _run_ours_and_save_step(
                model=model,
                args=args,
                tag=tag,
                sample_id=sample_id,
                prompt_idx=prompt_idx,
                step_label=step_label,
                latent=latent,
                baseline_01=vae_img_01,
                sigma=sigma,
                caption=prompt,
                output_dir=output_dir,
                uploader=uploader,
                baseline_subdir="vae_decode",
                baseline_upload_tag_prefix=f"{backbone}_vae_decode",
            )

    if uploader is not None:
        if is_rank0:
            logger.info("Waiting for background uploads to complete ...")
        uploader.wait()

    if is_rank0:
        logger.info(f"Done! Results saved under {output_dir}")


# =============================================================================
# Shared helpers (used by both diffusers + rae / scale_rae paths)
# =============================================================================


def _build_tag(args, backbone: str) -> str:
    extra_params = {"cfg": args.cfg_scale}
    if args.pid_inference_steps is not None:
        extra_params["steps"] = args.pid_inference_steps
    if args.shift is not None:
        extra_params["shift"] = args.shift
    if args.note:
        extra_params["note_"] = args.note
    base_tag = generate_tag_from_checkpoint(args.checkpoint_path, extra_params, load_ema=args.load_ema_to_reg)
    return f"{backbone}_{base_tag}"


def _load_our_decoder(args, experiment_opts: list, is_rank0: bool):
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
    return model


def _run_ours_and_save_step(
    *,
    model,
    args,
    tag: str,
    sample_id: str,
    prompt_idx: int,
    step_label: str,
    latent: torch.Tensor,
    baseline_01: torch.Tensor,  # (1, 3, H, W) in [0, 1]
    sigma: float,
    caption: str,
    output_dir: str,
    uploader: Optional[AsyncUploader],
    baseline_subdir: str,
    baseline_upload_tag_prefix: str,
):
    """Run our pixel decoder once on the captured latent + baseline image, save both,
    optionally upload to S3. Mirrors the per-step body of the diffusers run_demo loop.

    `baseline_subdir` is the local-filesystem subdir name for the native baseline
    (e.g. "vae_decode" for diffusers, "rae_decode" for RAE, "scale_rae_decode" for
    Scale-RAE). `baseline_upload_tag_prefix` becomes "<prefix>_step_<label>" on S3.
    """
    baseline_neg1_1 = baseline_01 * 2.0 - 1.0  # [-1, 1] for our decoder's LQ_video_or_image

    data_batch = {
        model.config.input_caption_key: [caption],
        "LQ_video_or_image": baseline_neg1_1.to(dtype=torch.bfloat16, device="cuda"),
        "LQ_latent": latent.to(dtype=torch.bfloat16, device="cuda"),
        "degrade_sigma": torch.tensor([sigma], device="cuda", dtype=torch.float32),
    }

    lq_h, lq_w = baseline_01.shape[-2], baseline_01.shape[-1]
    infer_image_size = (lq_h * args.scale, lq_w * args.scale)

    samples = model.generate_samples_from_batch(
        data_batch,
        cfg_scale=args.cfg_scale,
        num_steps=args.pid_inference_steps,
        seed=args.seed,
        shift=args.shift,
        image_size=infer_image_size,
    )
    ours_img = samples[0].float().cpu().clamp(-1, 1)

    ours_path = os.path.join(output_dir, tag, f"step_{step_label}", f"{sample_id}.{args.save_format}")
    save_image(ours_img, ours_path)

    baseline_path = os.path.join(output_dir, baseline_subdir, f"step_{step_label}", f"{sample_id}.{args.save_format}")
    save_image(baseline_neg1_1.float().cpu().squeeze(0).clamp(-1, 1), baseline_path)

    logger.info(f"[{prompt_idx}] step={step_label} sigma={sigma:.4f} -> ours={ours_path}  baseline={baseline_path}")

    if args.upload:
        ours_upload_tag = f"{tag}_step_{step_label}"
        baseline_upload_tag = f"{baseline_upload_tag_prefix}_step_{step_label}"
        if uploader is not None:
            uploader.submit(maybe_upload_video, ours_path, ours_upload_tag, True, args.group_name)
            uploader.submit(maybe_upload_video, baseline_path, baseline_upload_tag, True, args.group_name)
        else:
            maybe_upload_video(ours_path, ours_upload_tag, True, args.group_name)
            maybe_upload_video(baseline_path, baseline_upload_tag, True, args.group_name)


# =============================================================================
# RAE / scale_rae flows
# =============================================================================
#
# Both backbones bypass the diffusers pipeline:
#   - RAE       (rae_generation.py):       custom ODE sampler, class-conditional 512px,
#                                          full-trajectory tensor capture (slice traj[K]).
#   - scale_rae (scale_rae_generation.py): Qwen LM + DiT diffusion head, text-conditional
#                                          256px, monkey-patched p_sample_loop capture.
#
# After the captured latent + native-decoder image are in hand, the rest of the per-step
# work — running our pixel decoder, saving images, S3 upload — is identical to the
# diffusers flow, so it lives in the shared `_run_ours_and_save_step` helper above.


def _run_demo_rae(args):
    from pid._src.inference.rae_generation import (
        decode_rae_latent,
        load_class_names,
        load_rae_stack,
        resolve_rae_class_ids,
        resolve_rae_dit_ckpts,
        sample_rae_trajectory,
    )

    rank, world_size = get_rank_and_world_size()
    _maybe_init_distributed(world_size, rank)
    is_rank0 = rank == 0

    if args.resolution != 512:
        raise ValueError(f"RAE backbone only supports --resolution 512, got {args.resolution}")
    num_inference_steps = args.num_inference_steps
    save_xt_set = sorted(set(args.save_xt_steps)) if args.save_xt_steps else []
    for k in save_xt_set:
        if k < 1 or k > num_inference_steps:
            raise ValueError(f"--save_xt_steps value {k} out of range [1, {num_inference_steps}]")

    class_ids = resolve_rae_class_ids(args)
    rae_dit_main_ckpt, rae_dit_guid_ckpt = resolve_rae_dit_ckpts(args)
    class_names_path = os.path.join(os.path.dirname(__file__), "prompts", "imagenet_classes.txt")
    class_names = load_class_names(class_names_path)

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32

    tag = _build_tag(args, "rae")
    if is_rank0:
        logger.info(
            f"Backbone: rae  resolution: 512  num_inference_steps: {num_inference_steps}  "
            f"rae_cfg_scale: {args.rae_cfg_scale}  pid_steps: {args.pid_inference_steps}"
        )
        logger.info(f"Tag: {tag}")
        logger.info(f"#Classes: {len(class_ids)}  save_xt_steps: {save_xt_set}  scale: {args.scale}")

    experiment_opts = list(args.extra_experiment_opts) if args.extra_experiment_opts else []
    if is_rank0 and experiment_opts:
        logger.info(f"Extra experiment options: {experiment_opts}")

    # ---- Load RAE stack (warm-cache pattern across ranks) ----
    if world_size > 1:
        import torch.distributed as dist

        rae = dit_main = dit_guid = sample_fn = t_schedule = None
        for r in range(world_size):
            if rank == r:
                msg = "from disk" if r == 0 else "from OS cache"
                logger.info(f"[Rank {rank}] Loading RAE stack ({msg}) ...")
                rae, dit_main, dit_guid, sample_fn, t_schedule = load_rae_stack(
                    repo_path=args.rae_repo_path,
                    decoder_ckpt=args.rae_decoder_ckpt,
                    stats_path=args.rae_stats_path,
                    dit_main_ckpt=rae_dit_main_ckpt,
                    dit_guid_ckpt=rae_dit_guid_ckpt,
                    num_inference_steps=num_inference_steps,
                    device="cuda",
                    dtype=dtype,
                )
            dist.barrier()
    else:
        logger.info("Loading RAE stack ...")
        rae, dit_main, dit_guid, sample_fn, t_schedule = load_rae_stack(
            repo_path=args.rae_repo_path,
            decoder_ckpt=args.rae_decoder_ckpt,
            stats_path=args.rae_stats_path,
            dit_main_ckpt=rae_dit_main_ckpt,
            dit_guid_ckpt=rae_dit_guid_ckpt,
            num_inference_steps=num_inference_steps,
            device="cuda",
            dtype=dtype,
        )

    if is_rank0:
        logger.info(f"t_schedule range: {t_schedule[0].item():.4f} (noise) -> {t_schedule[-1].item():.4f} (clean)")

    model = _load_our_decoder(args, experiment_opts, is_rank0)

    output_dir = args.output_dir or "./results/official_demo/rae"
    os.makedirs(output_dir, exist_ok=True)
    if is_rank0:
        logger.info(f"Outputs -> {output_dir}")

    uploader = AsyncUploader(max_workers=8) if args.upload else None

    indexed_classes = list(enumerate(class_ids))
    if world_size > 1:
        indexed_classes = indexed_classes[rank::world_size]
        logger.info(f"[Rank {rank}/{world_size}] Processing {len(indexed_classes)} classes")

    for prompt_idx, cid in indexed_classes:
        seed = args.seed + prompt_idx
        sample_id = f"{prompt_idx:08d}"
        gen = torch.Generator(device="cuda").manual_seed(seed)
        caption = class_names[cid]

        logger.info(f"[{prompt_idx}] Sampling RAE trajectory (seed={seed}, class={cid}: {caption!r})")
        traj = sample_rae_trajectory(
            class_id=cid,
            dit_main=dit_main,
            dit_guid=dit_guid,
            sample_fn=sample_fn,
            device="cuda",
            dtype=dtype,
            cfg_scale=args.rae_cfg_scale,
            cfg_interval=args.rae_cfg_interval,
            generator=gen,
        )  # (num_inference_steps+1, 768, 32, 32)

        # Yield (label, latent_[1,768,32,32], sigma) for each xt step + final x0.
        steps: list[tuple[str, torch.Tensor, float]] = []
        for K in save_xt_set:
            steps.append((f"{K:02d}xt", traj[K : K + 1], float(t_schedule[K].item())))
        steps.append(("x0", traj[-1:], 0.0))

        for step_label, latent, sigma in steps:
            with torch.no_grad():
                baseline_01 = decode_rae_latent(rae, latent)  # (1, 3, 512, 512) in [0, 1]

            _run_ours_and_save_step(
                model=model,
                args=args,
                tag=tag,
                sample_id=sample_id,
                prompt_idx=prompt_idx,
                step_label=step_label,
                latent=latent,
                baseline_01=baseline_01,
                sigma=sigma,
                caption=caption,
                output_dir=output_dir,
                uploader=uploader,
                baseline_subdir="rae_decode",
                baseline_upload_tag_prefix="rae_decode",
            )

    if uploader is not None:
        if is_rank0:
            logger.info("Waiting for background uploads to complete ...")
        uploader.wait()

    if is_rank0:
        logger.info(f"Done! Results saved under {output_dir}")


def _run_demo_scale_rae(args):
    from pid._src.inference.scale_rae_generation import (
        _install_xt_capture,
        _postprocess_captured_xt,
        _resolve_decoder_paths,
        _validate_diff_head_is_full_sequence,
        decode_xt_to_image,
        generate_scale_rae_image,
        load_scale_rae_stack,
    )

    rank, world_size = get_rank_and_world_size()
    _maybe_init_distributed(world_size, rank)
    is_rank0 = rank == 0

    if args.resolution != 256:
        raise ValueError(f"Scale-RAE backbone only supports --resolution 256, got {args.resolution}")
    save_xt_set = sorted(set(args.save_xt_steps)) if args.save_xt_steps else []

    prompts = _load_prompts(args)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32

    tag = _build_tag(args, "scale_rae")
    if is_rank0:
        logger.info(
            f"Backbone: scale_rae  resolution: 256  guidance: {args.scale_rae_guidance_level}  "
            f"pid_steps: {args.pid_inference_steps}"
        )
        logger.info(f"Tag: {tag}")
        logger.info(f"#Prompts: {len(prompts)}  save_xt_steps: {save_xt_set}  scale: {args.scale}")

    experiment_opts = list(args.extra_experiment_opts) if args.extra_experiment_opts else []
    if is_rank0 and experiment_opts:
        logger.info(f"Extra experiment options: {experiment_opts}")

    decoder_config_path, decoder_ckpt_path = _resolve_decoder_paths(
        args.scale_rae_decoder_config,
        args.scale_rae_decoder_ckpt,
        args.scale_rae_repo_path,
    )

    # ---- Load Scale-RAE stack (warm-cache pattern across ranks) ----
    if world_size > 1:
        import torch.distributed as dist

        tokenizer = sr_model = decoder = None
        for r in range(world_size):
            if rank == r:
                msg = "from disk" if r == 0 else "from OS cache"
                logger.info(f"[Rank {rank}] Loading Scale-RAE stack ({msg}) ...")
                tokenizer, sr_model, decoder = load_scale_rae_stack(
                    repo_path=args.scale_rae_repo_path,
                    model_path=args.scale_rae_model_path,
                    decoder_config_path=decoder_config_path,
                    decoder_ckpt=decoder_ckpt_path,
                    pretrained_encoder_path=args.scale_rae_pretrained_encoder,
                    device="cuda",
                    dtype=dtype,
                )
            dist.barrier()
    else:
        logger.info("Loading Scale-RAE stack ...")
        tokenizer, sr_model, decoder = load_scale_rae_stack(
            repo_path=args.scale_rae_repo_path,
            model_path=args.scale_rae_model_path,
            decoder_config_path=decoder_config_path,
            decoder_ckpt=decoder_ckpt_path,
            pretrained_encoder_path=args.scale_rae_pretrained_encoder,
            device="cuda",
            dtype=dtype,
        )

    capture_state = None
    if save_xt_set:
        _validate_diff_head_is_full_sequence(sr_model)
        capture_state = _install_xt_capture(sr_model, save_xt_set)

    model = _load_our_decoder(args, experiment_opts, is_rank0)

    output_dir = args.output_dir or "./results/official_demo/scale_rae"
    os.makedirs(output_dir, exist_ok=True)
    if is_rank0:
        logger.info(f"Outputs -> {output_dir}")

    uploader = AsyncUploader(max_workers=8) if args.upload else None

    prompt_prefix = args.scale_rae_prompt_prefix or ""
    if is_rank0 and prompt_prefix:
        logger.info(f"Prepending {prompt_prefix!r} to every prompt before LM tokenization")

    indexed_prompts = list(enumerate(prompts))
    if world_size > 1:
        indexed_prompts = indexed_prompts[rank::world_size]
        logger.info(f"[Rank {rank}/{world_size}] Processing {len(indexed_prompts)} prompts")

    for prompt_idx, prompt in indexed_prompts:
        seed = args.seed + prompt_idx
        sample_id = f"{prompt_idx:08d}"
        # Scale-RAE's autoregressive LM uses torch.manual_seed for sampling determinism.
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        if capture_state is not None:
            capture_state["captured"] = {}

        lm_prompt = prompt_prefix + prompt if prompt_prefix else prompt
        logger.info(f"[{prompt_idx}] Generating scale_rae (seed={seed}): {prompt[:80]!r}")

        latent_2d, image_01 = generate_scale_rae_image(
            prompt=lm_prompt,
            tokenizer=tokenizer,
            model=sr_model,
            decoder=decoder,
            guidance_level=args.scale_rae_guidance_level,
            max_new_tokens=args.scale_rae_max_new_tokens,
            final_pixel_size=256,
        )
        # latent_2d: (1152, 16, 16) bf16 cpu; image_01: (3, 256, 256) f32 cpu

        # Build (label, latent_[1,1152,16,16], baseline_01_[1,3,256,256], sigma) per step.
        steps: list[tuple[str, torch.Tensor, torch.Tensor, float]] = []
        if capture_state is not None:
            for K in save_xt_set:
                if K not in capture_state["captured"]:
                    raise RuntimeError(
                        f"xt capture for step {K} did not fire — check that "
                        f"model.diff_head.inference_flow.p_sample_loop is the entry point"
                    )
                xt_bchw, t_K = capture_state["captured"][K]
                xt_bld = _postprocess_captured_xt(sr_model, xt_bchw)  # (1, 256, 1152)
                grid = int(xt_bld.shape[1] ** 0.5)
                xt_latent = (
                    xt_bld[0].reshape(grid, grid, xt_bld.shape[2]).permute(2, 0, 1).contiguous().unsqueeze(0)
                )  # (1, 1152, 16, 16)
                xt_baseline = decode_xt_to_image(sr_model, decoder, xt_bld, final_pixel_size=256).unsqueeze(0)
                steps.append((f"{K:02d}xt", xt_latent, xt_baseline, float(t_K)))

        steps.append(("x0", latent_2d.unsqueeze(0), image_01.unsqueeze(0), 0.0))

        for step_label, latent, baseline_01, sigma in steps:
            _run_ours_and_save_step(
                model=model,
                args=args,
                tag=tag,
                sample_id=sample_id,
                prompt_idx=prompt_idx,
                step_label=step_label,
                latent=latent,
                baseline_01=baseline_01,
                sigma=sigma,
                caption=prompt,  # original caption — not the LM-prefixed one
                output_dir=output_dir,
                uploader=uploader,
                baseline_subdir="scale_rae_decode",
                baseline_upload_tag_prefix="scale_rae_decode",
            )

    if uploader is not None:
        if is_rank0:
            logger.info("Waiting for background uploads to complete ...")
        uploader.wait()

    if is_rank0:
        logger.info(f"Done! Results saved under {output_dir}")
