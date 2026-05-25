"""From-clean demo: input image -> SD3 VAE encode -> optional noise -> ours pixel decoder.

No latent diffusion model is run. The image is center-cropped + bicubic-resized to a square
(--input_resolution, default 512), VAE-encoded via the SD3 VAE that ships with the loaded
pixel-decoder model, optionally noised by sigma in --degrade_sigmas, then decoded twice
(VAE baseline + ours) at --scale * input_resolution.

PYTHONPATH=. python -m pid._src.inference.from_clean_sd3 \
    --manifest assets/clean_image_manifest.jsonl \
    --input_resolution 512 \
    --degrade_sigmas 0.0 \
    --output_dir ./results/official_demo_from_clean/sd3 \
    --cfg_scale 1 --pid_inference_steps 4 --scale 4
"""

from pid._src.inference._demo_from_clean_common import run_demo_from_clean

if __name__ == "__main__":
    run_demo_from_clean(backbone_tag="sd3")
