from hydra.core.config_store import ConfigStore

from pid._ext.imaginaire.lazy_config import LazyDict
from pid._src.configs.pid.experiment.shared_config import (
    _common_model_overrides,
)


def _siglip_distill_experiment(name: str) -> LazyDict:
    """SigLIP-2 So400M + Scale-RAE ViT-XL tokenizer (state_ch=1152, virtual 16x
    compression). 8x SR (LQ=256 → HQ=2048): the default PID_SR4X net has
    ``sr_scale=4`` so we override it to 8 here."""
    cfg = _common_model_overrides(state_ch=1152)
    cfg["net"] = {
        **cfg["net"],
        "lq_latent_channels": 1152,
        "latent_spatial_down_factor": 16,
        "sr_scale": 8,
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
                {"override /tokenizer": "scale_rae_tokenizer"},
                "_self_",
            ],
            job=dict(group="pid_official", name=name),
            model=dict(config=cfg),
        ),
    )


PID_RES2K_SR8X_OFFICIAL_SIGLIP_DISTILL_4STEP = _siglip_distill_experiment(
    "PiD_res2k_sr8x_official_siglip_distill_4step"
)


cs = ConfigStore.instance()
cs.store(
    group="experiment",
    package="_global_",
    name=PID_RES2K_SR8X_OFFICIAL_SIGLIP_DISTILL_4STEP["job"]["name"],
    node=PID_RES2K_SR8X_OFFICIAL_SIGLIP_DISTILL_4STEP,
)
