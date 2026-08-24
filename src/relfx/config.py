"""Configuration for the ISMIR 2026 RelFx research release.

Paths are provided through environment variables so the release contains no
machine- or organization-specific locations.
"""

from __future__ import annotations

import os
from pathlib import Path


REPOSITORY_ROOT = Path(
    os.environ.get("RELFX_REPOSITORY_ROOT", Path.cwd())
).resolve()


def _path_list(name: str) -> list[str]:
    value = os.environ.get(name, "")
    return [item for item in value.split(os.pathsep) if item]


# Data
AUDIO_DIRS = _path_list("RELFX_AUDIO_DIRS")
AUDIO_DIR = AUDIO_DIRS[0] if AUDIO_DIRS else ""

# Optional JSON mapping track IDs to structural segments. See
# docs/data-format.md for the public schema.
STRUCTURE_SEGMENT_JSON = os.environ.get("RELFX_STRUCTURE_SEGMENTS")
STRUCTURED_AUDIO_DIRS = set(_path_list("RELFX_STRUCTURED_AUDIO_DIRS"))
DENSITY_FILTER_AUDIO_DIRS = set(_path_list("RELFX_DENSITY_FILTER_AUDIO_DIRS"))
VALID_STRUCTURE_LABELS = {"verse", "chorus"}

# Outputs
CHECKPOINT_DIR = os.environ.get(
    "RELFX_CHECKPOINT_DIR", str(REPOSITORY_ROOT / "outputs" / "checkpoints")
)
LOG_DIR = os.environ.get(
    "RELFX_LOG_DIR", str(REPOSITORY_ROOT / "outputs" / "logs")
)

# Audio
SAMPLE_RATE = 44_100
SEGMENT_DURATION = 10.0
SEGMENT_SAMPLES = int(SAMPLE_RATE * SEGMENT_DURATION)

# Training effect chain: 8 processors and 72 continuous parameters.
FX_CHAIN_ORDER = [
    "eq",
    "distortion",
    "multiband_comp",
    "gain",
    "imager",
    "limiter",
    "delay",
    "reverb",
]
FX_PROB = {
    "eq": 0.6,
    "distortion": 0.3,
    "multiband_comp": 0.8,
    "gain": 0.6,
    "imager": 0.6,
    "limiter": 0.6,
    "delay": 0.6,
    "reverb": 0.6,
}

DYNAMIC_FX_PROB_CONFIG = {
    "enabled": True,
    "update_every_epochs": 1,
    "min_prob": 0.3,
    "max_prob": 0.8,
    "smoothing_alpha": 0.3,
    "eval_method": "per_fx_retrieval",
    "eval_samples_per_fx": 64,
    "warmup_epochs": 10,
}

# The optional in-training Ld probe used private, precomputed triplets and is
# disabled in the public release. The standalone ITO evaluation is public.
LD_EVAL_CONFIG = {
    "enabled": False,
    "every_epochs": 10,
    "triplets_dir": "",
    "dataset": "musdb18",
}

CROSS_SEGMENT = True

# The public checkpoint is the Base variant. The separately trained paper
# variant uses these defaults when --model-variant bidirectional is selected.
BIDIRECTIONAL_CONFIG = {
    "enabled": False,
    "flip_ratio": 0.5,
}

MODEL_CONFIG = {
    "model_variant": "base",
    "fusion_type": "diff_gate",
    "cross_attn_stages": [3, 5],
    "cross_attn_heads": 4,
    "cross_attn_pool": 4,
    "cross_attn_scale": 0.1,
}

LOSS_SWITCHES = {
    "param_regression": True,
    "soft_contrastive": False,
    "distance_margin": False,
    "hard_negative": False,
}

PARAM_REGRESSION_CONFIG = {
    "weight": 0.5,
    "hidden_dim": 512,
    "predict_activate": True,
    "activate_weight": 0.3,
    "per_fx_head": False,
}
SOFT_CONTRASTIVE_CONFIG = {
    "weight": 1.0,
    "sigma": 0.2,
    "distance_type": "hybrid",
    "replace_infonce": False,
    "extra_weight": 0.5,
}
DISTANCE_MARGIN_CONFIG = {
    "weight": 0.5,
    "base_margin": 0.1,
    "margin_scale": 0.5,
    "max_margin": 1.0,
    "distance_type": "hybrid",
}
HARD_NEGATIVE_CONFIG = {
    "strategy": "same_activate",
    "perturb_std": 0.1,
    "hard_ratio": 0.5,
    "curriculum": True,
    "curriculum_start_epoch": 0,
    "curriculum_end_epoch": 50,
}

TRAIN_CONFIG = {
    "batch_size": 48,
    "num_workers": 8,
    "epochs": 200,
    "lr": 3e-4,
    "weight_decay": 1e-5,
    "temperature": 0.15,
    "embed_dim": 2048,
    "proj_dim": 128,
    "grad_accum_steps": 4,
    "warmup_epochs": 10,
    "use_triplet": True,
    "triplet_margin": 0.3,
    "triplet_weight": 0.5,
    "shuffle_fx_order": True,
    "fx_sample_mode": "uniform",
    "conservative_ratio": 0.4,
    "conservative_range": (0.3, 0.7),
    "save_every": 20,
    "val_every": 10,
    "device": "cuda",
    "precision": "fp32",
}

EVAL_CONFIG = {
    "ito_lr": 0.01,
    "ito_iters": 200,
    "ito_restarts": 7,
    "loss_type": "embedding",
}


def ensure_dirs() -> None:
    Path(CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)
    Path(LOG_DIR).mkdir(parents=True, exist_ok=True)


def get_active_methods() -> str:
    active = [name for name, enabled in LOSS_SWITCHES.items() if enabled]
    if DYNAMIC_FX_PROB_CONFIG.get("enabled", False):
        active.append("dynamic_fx_prob")
    return " + ".join(active) if active else "baseline"
