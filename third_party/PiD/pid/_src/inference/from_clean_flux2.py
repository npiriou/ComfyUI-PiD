"""From-clean demo: input image -> Flux2 VAE encode -> optional noise -> ours pixel decoder.

No latent diffusion model is run. The image is center-cropped + bicubic-resized to a square
(--input_resolution, default 512), VAE-encoded via the Flux2 VAE (32 channels +
2x2 patchification + BN normalization) that ships with the loaded pixel-decoder model,
optionally noised by sigma in --degrade_sigmas, then decoded twice (VAE baseline + ours)
at --scale * input_resolution.

>>> Single image, sigma sweep:
PYTHONPATH=. python -m pid._src.inference.from_clean_flux2 \
    --manifest assets/clean_image_manifest.jsonl \
    --input_resolution 512 \
    --degrade_sigmas 0.0 \
    --output_dir ./results/official_demo_from_clean/flux2 \
    --cfg_scale 2.75 --pid_inference_steps 25 --scale 4
"""

from pid._src.inference._demo_from_clean_common import run_demo_from_clean

if __name__ == "__main__":
    run_demo_from_clean(backbone_tag="flux2")
