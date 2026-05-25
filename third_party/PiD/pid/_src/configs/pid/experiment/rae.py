from hydra.core.config_store import ConfigStore

from pid._ext.imaginaire.lazy_config import LazyDict
from pid._src.configs.pid.experiment.shared_config import (
    _common_model_overrides,
)


def _dinov2_distill_experiment(name: str) -> LazyDict:
    """DINOv2-B + RAE ViT-XL tokenizer (state_ch=768, virtual 16x compression)."""
    cfg = _common_model_overrides(state_ch=768)
    cfg["net"] = {
        **cfg["net"],
        "lq_latent_channels": 768,
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
                {"override /tokenizer": "dinov2_rae_tokenizer"},
                "_self_",
            ],
            job=dict(group="pid_official", name=name),
            model=dict(config=cfg),
        ),
    )


PID_RES2K_SR4X_OFFICIAL_DINOV2_DISTILL_4STEP = _dinov2_distill_experiment(
    "PiD_res2k_sr4x_official_dinov2_distill_4step"
)


cs = ConfigStore.instance()
cs.store(
    group="experiment",
    package="_global_",
    name=PID_RES2K_SR4X_OFFICIAL_DINOV2_DISTILL_4STEP["job"]["name"],
    node=PID_RES2K_SR4X_OFFICIAL_DINOV2_DISTILL_4STEP,
)
