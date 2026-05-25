"""Official demo: Flux.1-dev latent diffusion vs ours pixel-diffusion decoder.

Runs FluxPipeline (black-forest-labs/FLUX.1-dev) on a text prompt, captures
intermediate xt at user-specified denoising steps and the final clean x0, then
decodes each captured latent twice — once with the Flux VAE (baseline) and once
with our pixel-diffusion decoder (ours, scale=4 SR by default). Outputs are saved
side-by-side and (optionally) async-uploaded to S3.

>>> Single GPU, single prompt:
PYTHONPATH=. python -m pid._src.inference.from_ldm_flux \
    --prompt "Cinematic, High Contrast, highly detailed cat" \
    --ldm_inference_steps 28 --save_xt_steps 16 18 20 22 24 26 \
    --output_dir ./results/official_demo/flux \
    --cfg_scale 1 --pid_inference_steps 4 --scale 4

>>> Multi-GPU, prompt file, S3 upload:
PYTHONPATH=. /usr/local/bin/torchrun --nproc_per_node=4 \
    -m pid._src.inference.from_ldm_flux \
    --prompt_file pid/_src/inference/prompts/prompt_creative.txt \
    --ldm_inference_steps 28 --save_xt_steps 16 18 20 22 24 26 \
    --cfg_scale 1 --pid_inference_steps 4 --scale 4
"""

from pid._src.inference._demo_common import run_demo

if __name__ == "__main__":
    run_demo(backbone="flux")
