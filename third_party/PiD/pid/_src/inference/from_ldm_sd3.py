"""Official demo: Stable Diffusion 3 latent diffusion vs ours pixel-diffusion decoder.

Runs StableDiffusion3Pipeline (stabilityai/stable-diffusion-3-medium-diffusers) on a
text prompt, captures intermediate xt at user-specified denoising steps and the
final clean x0, then decodes each captured latent twice — once with the SD3 VAE
(baseline, affine scale=1.5305 / shift=0.0609) and once with our pixel-diffusion
decoder (ours, scale=4 SR by default). Outputs are saved side-by-side and
(optionally) async-uploaded to S3.

>>> Single GPU, prompt file:
PYTHONPATH=. python -m pid._src.inference.from_ldm_sd3 \
    --prompt_file pid/_src/inference/prompts/prompt_creative.txt \
    --ldm_inference_steps 28 --save_xt_steps 16 18 20 22 24 26 \
    --output_dir ./results/official_demo/sd3 \
    --cfg_scale 1 --pid_inference_steps 4 --scale 4

>>> Multi-GPU, prompt file, S3 upload:
PYTHONPATH=. /usr/local/bin/torchrun --nproc_per_node=4 \
    -m pid._src.inference.from_ldm_sd3 \
    --prompt_file pid/_src/inference/prompts/prompt_creative.txt \
    --ldm_inference_steps 28 --save_xt_steps 16 18 20 22 24 26 \
    --cfg_scale 1 --pid_inference_steps 4 --scale 4
"""

from pid._src.inference._demo_common import run_demo

if __name__ == "__main__":
    run_demo(backbone="sd3")
