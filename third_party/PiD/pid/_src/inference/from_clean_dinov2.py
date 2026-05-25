"""From-clean demo: input image -> DINOv2-RAE encode -> optional noise -> ours pixel decoder.

No latent diffusion model is run. The image is center-cropped + bicubic-resized to a square
(--input_resolution; pass 512 to match DINOv2-RAE's native 512×512 / 32×32 token grid),
encoded via the DINOv2-with-registers + RAE ViT-XL decoder bundled with the loaded
pixel-decoder model (pid/_src/tokenizers/dinov2_vae.py:DINOv2RAEConfig), optionally
noised by sigma in --degrade_sigmas, then decoded twice (RAE baseline + ours) at the
VAE-native 512 * --scale resolution.

DINOv2-RAE specifics: feature space is per-(C,H,W)-normalized via ImageNet stats stored
under checkpoints/rae/stats/dinov2/wReg_base/imagenet1k_512/stat.pt. RAE's normalized
DINOv2 features tolerate less added noise than VAE latents — keep --degrade_sigmas in
[0, 0.5] to mirror the training distribution.

--experiment / --checkpoint_path default to the registry entry for backbone_tag="rae"
(see checkpoint_registry.py). Pass either flag on the CLI to override.

>>> Single GPU, sigma sweep:
PYTHONPATH=. python -m pid._src.inference.from_clean_dinov2 \
    --load_ema_to_reg \
    --manifest assets/clean_image_manifest.jsonl \
    --input_resolution 512 \
    --degrade_sigmas 0.0 0.1 0.2 \
    --output_dir ./results/official_demo_from_clean/rae \
    --cfg_scale 1 --pid_inference_steps 4 --scale 4

>>> Multi-GPU, S3 upload:
PYTHONPATH=. /usr/local/bin/torchrun --nproc_per_node=4 \
    -m pid._src.inference.from_clean_dinov2 \
    --load_ema_to_reg \
    --manifest assets/clean_image_manifest.jsonl \
    --input_resolution 512 \
    --degrade_sigmas 0.0 0.1 \
    --output_dir ./results/official_demo_from_clean/rae \
    --cfg_scale 1 --pid_inference_steps 4 --scale 4 \
    --upload --group_name official_demo_from_clean_rae
"""

from pid._src.inference._demo_from_clean_common import run_demo_from_clean

if __name__ == "__main__":
    run_demo_from_clean(backbone_tag="rae")
