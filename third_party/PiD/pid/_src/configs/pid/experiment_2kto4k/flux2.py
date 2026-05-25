from hydra.core.config_store import ConfigStore

from pid._ext.imaginaire.lazy_config import LazyDict
from pid._src.configs.pid.experiment_2kto4k.shared_config import (
    _common_model_overrides_2kto4k,
)


def _flux2_2kto4k_distill_experiment(name: str) -> LazyDict:
    """Flux2 needs explicit net overrides: the default `pid_sr4x` net is sized
    for 16-ch / 8× compression VAEs, but the Flux2 VAE is 128-ch (32 raw ×
    2×2 patchify) / 16× compression."""
    cfg = _common_model_overrides_2kto4k(state_ch=128)
    cfg["net"] = {
        **cfg["net"],
        "lq_latent_channels": 128,
        "latent_spatial_down_factor": 16,
    }
    return LazyDict(
        dict(
            defaults=[
                {"override /model": "ddp_distill_pid"},
                {"override /net": "pid_sr4x"},
                {"override /conditioner": "pid_caption_lq"},
                {"override /ckpt_type": "dcp"},
                {"override /ema": None},
                {"override /checkpoint": "local"},
                {"override /tokenizer": "flux2_vae_tokenizer"},
                "_self_",
            ],
            job=dict(group="pid_official", name=name),
            model=dict(config=cfg),
        ),
    )


PID_RES2KTO4K_SR4X_OFFICIAL_FLUX2_DISTILL_4STEP = _flux2_2kto4k_distill_experiment(
    "PiD_res2kto4k_sr4x_official_flux2_distill_4step"
)


cs = ConfigStore.instance()
cs.store(
    group="experiment",
    package="_global_",
    name=PID_RES2KTO4K_SR4X_OFFICIAL_FLUX2_DISTILL_4STEP["job"]["name"],
    node=PID_RES2KTO4K_SR4X_OFFICIAL_FLUX2_DISTILL_4STEP,
)
