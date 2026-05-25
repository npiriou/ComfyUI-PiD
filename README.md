# ComfyUI PiD

ComfyUI custom nodes for [NVIDIA PiD](https://huggingface.co/nvidia/PiD).

PiD is not a VAE file. It is a separate diffusion decoder used after sampling. The native VAE is still required internally to produce the low-resolution conditioning image.

## Supported

- Flux 1 dev `512 -> 2048`
- Flux 1 dev `1024 -> 3840`
- Z-Image with Flux-compatible 16-channel latents
- Final clean latents only
- Batch size `1`
- CUDA only

## Install

1. Put this repo in `ComfyUI/custom_nodes/`.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Put the PiD files here:

```text
ComfyUI/models/pid/checkpoints/ae.safetensors
ComfyUI/models/pid/checkpoints/PiD_res2k_sr4x_official_flux_distill_4step/model_ema_bf16.pth
ComfyUI/models/pid/checkpoints/PiD_res2kto4k_sr4x_official_flux_distill_4step/model_ema_bf16.pth
```

4. Restart ComfyUI.

## Workflow

```text
Flux/Z-Image sampler
-> latent
-> PiD Model Loader
-> PiD Decode
-> image
```

Use:

- `PiD_res2k_sr4x_official_flux_distill_4step` for `512 -> 2048`
- `PiD_res2kto4k_sr4x_official_flux_distill_4step` for `1024 -> 3840`

If the latent resolution does not match the selected checkpoint family, the node raises an explicit error.

## Example Graphs

- `examples/api_flux1_pid_2k.json`
- `examples/api_flux1_pid_2kto4k.json`

## Notes

- PiD is much heavier than normal VAE decode.
- The first PiD load may download upstream text-encoder assets through `transformers`.

## Upstream

The repo vendors NVIDIA's `pid` package under `third_party/PiD`. See `third_party/PiD/LICENSE` for the upstream license.
