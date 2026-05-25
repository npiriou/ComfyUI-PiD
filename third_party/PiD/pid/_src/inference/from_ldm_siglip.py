"""Official demo: Scale-RAE T2I (Qwen 1.5B LM + 2.4B DiT) vs ours pixel-diffusion decoder.

Text-conditional. Native generation resolution is 256×256 (decoder is 14-multiple
224, bicubic-upsampled to 256 to match the rest of the pixel-diffusion pipeline).

Per-step semantics: --save_xt_steps K captures the diffusion trajectory state
after K rectified-flow steps via a monkey-patch of
`model.diff_head.inference_flow.p_sample_loop`. The final clean image embeddings
are always saved as "x0".

The model was trained on request-style prompts; --scale_rae_prompt_prefix
("Could you generate an image of " by default) is prepended before tokenization.
The ORIGINAL prompt is what is used as the caption fed into our pixel decoder.

Requires the upstream Scale-RAE GitHub repo
(https://github.com/ZitengWangNYU/Scale-RAE) cloned on disk and installed
(``pip install -e .``). The ``SCALE_RAE_REPO_PATH`` env var (or
``--scale_rae_repo_path``) points at it. See README for installation instructions.

--experiment / --checkpoint_path default to the registry entry for
backbone="scale_rae" (see checkpoint_registry.py). Pass either flag on the CLI
to override.

>>> Single GPU, single prompt:
export SCALE_RAE_REPO_PATH=$(realpath ../Scale-RAE)
PYTHONPATH=. python -m pid._src.inference.from_ldm_siglip \
    --load_ema_to_reg \
    --prompt "A cat sitting on a windowsill at sunset" \
    --save_xt_steps 44 46 48 \
    --cfg_scale 1 --pid_inference_steps 4 --scale 8 \
    --output_dir ./results/official_demo/scale_rae

>>> Multi-GPU, prompt file, S3 upload:
PYTHONPATH=. /usr/local/bin/torchrun --nproc_per_node=4 \
    -m pid._src.inference.from_ldm_siglip \
    --load_ema_to_reg \
    --prompt_file pid/_src/inference/prompts/prompt_creative.txt \
    --save_xt_steps 44 46 48 \
    --cfg_scale 1 --pid_inference_steps 4 --scale 8 \
    --upload --group_name official_demo_scale_rae
"""

from pid._src.inference._demo_common import run_demo

if __name__ == "__main__":
    run_demo(backbone="scale_rae")
