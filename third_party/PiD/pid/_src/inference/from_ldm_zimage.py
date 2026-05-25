"""Official demo: Z-Image latent diffusion vs ours pixel-diffusion decoder.

Runs ZImagePipeline (Tongyi-MAI/Z-Image) on a text prompt, captures intermediate
xt at user-specified denoising steps and the final clean x0, then decodes each
captured latent twice — once with the Z-Image VAE (baseline, affine
denormalization with scale/shift read from pipeline.vae.config at runtime) and
once with our pixel-diffusion decoder (ours, scale=4 SR by default). Outputs are
saved side-by-side and (optionally) async-uploaded to S3.

>>> Single GPU:
PYTHONPATH=. python -m pid._src.inference.from_ldm_zimage \
    --prompt "A futuristic cityscape at sunset, ultra-detailed" \
    --ldm_inference_steps 50 --save_xt_steps 38 40 42 44 46 48 \
    --output_dir ./results/official_demo/zimage \
    --cfg_scale 1 --pid_inference_steps 4 --scale 4

>>> Multi-GPU, prompt file, S3 upload:
PYTHONPATH=. /usr/local/bin/torchrun --nproc_per_node=4 \
    -m pid._src.inference.from_ldm_zimage \
    --prompt_file pid/_src/inference/prompts/prompt_ai.txt \
    --ldm_inference_steps 50 --save_xt_steps 38 40 42 44 46 48 \
    --cfg_scale 1 --pid_inference_steps 4 --scale 4
"""

from pid._src.inference._demo_common import run_demo

if __name__ == "__main__":
    run_demo(backbone="zimage")
