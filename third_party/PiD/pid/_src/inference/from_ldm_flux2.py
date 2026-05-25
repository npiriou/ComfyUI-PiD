"""Official demo: Flux2 latent diffusion vs ours pixel-diffusion decoder.

Runs Flux2Pipeline (black-forest-labs/FLUX.2-dev) on a text prompt, captures
intermediate xt at user-specified denoising steps and the final clean x0, then
decodes each captured latent twice — once with the Flux2 VAE (baseline, BatchNorm-
based denormalization + 2x2 unpatchify) and once with our pixel-diffusion decoder
(ours, scale=4 SR by default). Outputs are saved side-by-side and (optionally)
async-uploaded to S3.

>>> Single GPU with cpu_offload:
PYTHONPATH=. python -m pid._src.inference.from_ldm_flux2 \
    --prompt "A cinematic still of a fox in autumn leaves" \
    --ldm_inference_steps 50 --save_xt_steps 38 40 42 44 46 48 \
    --output_dir ./results/official_demo/flux2 \
    --cfg_scale 1 --pid_inference_steps 4 --scale 4

>>> Multi-GPU, prompt file, S3 upload:
PYTHONPATH=. /usr/local/bin/torchrun --nproc_per_node=4 \
    -m pid._src.inference.from_ldm_flux2 \
    --prompt_file pid/_src/inference/prompts/prompt_ai.txt \
    --ldm_inference_steps 50 --save_xt_steps 38 40 42 44 46 48  \
    --cfg_scale 1 --pid_inference_steps 4 --scale 4
"""

from pid._src.inference._demo_common import run_demo

if __name__ == "__main__":
    run_demo(backbone="flux2")
