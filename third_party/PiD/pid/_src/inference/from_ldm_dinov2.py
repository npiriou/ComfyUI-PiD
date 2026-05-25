"""Official demo: RAE class-conditional latent diffusion vs ours pixel-diffusion decoder.

Class-conditional ImageNet-512 (DINOv2-B encoder + ViT-XL decoder + DiT^DH-XL/S
autoguidance). Uses --rae_class_ids / --rae_class_range instead of --prompt /
--prompt_file (no text conditioning). Native generation resolution is 512×512.

Per-step semantics mirror the diffusers demos:
  - --save_xt_steps K captures traj[K] (state AFTER K Euler ODE steps).
  - The final clean latent (x0 = traj[-1]) is always saved.
  - Each captured latent is decoded twice — once with the RAE ViT-XL decoder
    (baseline) and once with our pixel-diffusion decoder.

Requires the upstream RAE GitHub repo (https://github.com/bytetriper/RAE) cloned
on disk; ``RAE_REPO_PATH`` env var (or ``--rae_repo_path``) points at it. See
README for installation instructions.

--experiment / --checkpoint_path default to the registry entry for backbone="rae"
(see checkpoint_registry.py). Pass either flag on the CLI to override.

>>> Single GPU, three classes:
export RAE_REPO_PATH=$(realpath ../RAE)
PYTHONPATH=. python -m pid._src.inference.from_ldm_dinov2 \
    --load_ema_to_reg \
    --rae_class_ids 207 281 387 --num_inference_steps 50 \
    --save_xt_steps 44 46 48 \
    --cfg_scale 1 --pid_inference_steps 4 --scale 4 \
    --output_dir ./results/official_demo/rae

>>> Multi-GPU, class range, S3 upload:
PYTHONPATH=. /usr/local/bin/torchrun --nproc_per_node=4 \
    -m pid._src.inference.from_ldm_dinov2 \
    --load_ema_to_reg \
    --rae_class_range 200 300 --num_inference_steps 50 \
    --save_xt_steps 44 46 48 \
    --cfg_scale 1 --pid_inference_steps 4 --scale 4 \
    --upload --group_name official_demo_rae
"""

from pid._src.inference._demo_common import run_demo

if __name__ == "__main__":
    run_demo(backbone="rae")
