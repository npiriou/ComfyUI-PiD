"""From-clean demo: input image -> Scale-RAE encode -> optional noise -> ours pixel decoder.

No latent diffusion model is run. The image is center-cropped + bicubic-resized to a square
(--input_resolution; pass 256 to match Scale-RAE's native 256×256 / 16×16 token grid),
encoded via the SigLIP-2 So400M (patch14) encoder + Scale-RAE ViT-XL decoder bundled with the
loaded pixel-decoder model (pid/_src/tokenizers/scale_rae_vae.py:ScaleRAEConfig),
optionally noised by sigma in --degrade_sigmas, then decoded twice (Scale-RAE baseline + ours)
at the VAE-native 256 * --scale resolution. Use --scale 8 (the trained ratio: 256 → 2048).

Scale-RAE specifics: 14-multiple internal grid (224 = 16×14) is bridged to the pipeline's
16-multiple (256) via two bicubic interpolations at the pixel boundary, so both encoder and
decoder run IN DISTRIBUTION. Feature space is affine-free LayerNorm'd; the diffusion DiT was
trained against this normalized SigLIP-2 representation. As with DINOv2-RAE, normalized
features tolerate less added noise than VAE latents — keep --degrade_sigmas in [0, 0.5].

--experiment / --checkpoint_path default to the registry entry for backbone_tag="scale_rae"
(see checkpoint_registry.py). Pass either flag on the CLI to override.

>>> Single GPU, sigma sweep:
PYTHONPATH=. python -m pid._src.inference.from_clean_siglip \
    --load_ema_to_reg \
    --manifest assets/clean_image_manifest.jsonl \
    --input_resolution 256 \
    --degrade_sigmas 0.0 0.1 0.2 \
    --output_dir ./results/official_demo_from_clean/scale_rae \
    --cfg_scale 1 --pid_inference_steps 4 --scale 8

>>> Multi-GPU, S3 upload:
PYTHONPATH=. /usr/local/bin/torchrun --nproc_per_node=4 \
    -m pid._src.inference.from_clean_siglip \
    --load_ema_to_reg \
    --manifest assets/clean_image_manifest.jsonl \
    --input_resolution 256 \
    --degrade_sigmas 0.0 0.1 \
    --output_dir ./results/official_demo_from_clean/scale_rae \
    --cfg_scale 1 --pid_inference_steps 4 --scale 8 \
    --upload --group_name official_demo_from_clean_scale_rae
"""

from pid._src.inference._demo_from_clean_common import run_demo_from_clean

if __name__ == "__main__":
    run_demo_from_clean(backbone_tag="scale_rae")
