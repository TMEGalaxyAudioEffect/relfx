"""Public configuration for the RelFx ITO parameter-matching evaluation."""

from __future__ import annotations

import os
from pathlib import Path


REPOSITORY_ROOT = Path(
    os.environ.get("RELFX_REPOSITORY_ROOT", Path.cwd())
).resolve()
MUSDB18_ROOT = os.environ.get("MUSDB18_ROOT", "")
MEDLEYDB_ROOT = os.environ.get("MEDLEYDB_ROOT", "")
FXNORM_MUSDB_DIR = os.environ.get("FXNORM_MUSDB_DIR", "")
FXNORM_MEDLEYDB_DIR = os.environ.get("FXNORM_MEDLEYDB_DIR", "")
FXNORM_MUSDB_NPY_DIR = os.environ.get("FXNORM_MUSDB_NPY_DIR", "")
FXNORM_STEMS_DIR = FXNORM_MUSDB_DIR

DATA_DIR = os.environ.get(
    "RELFX_ITO_DATA_DIR", str(REPOSITORY_ROOT / "outputs" / "ito-data")
)
TRIPLETS_DIR = os.path.join(DATA_DIR, "triplets")
RESULTS_DIR = os.environ.get(
    "RELFX_ITO_RESULTS_DIR", str(REPOSITORY_ROOT / "outputs" / "ito-results")
)

SAMPLE_RATE = 44_100
SEGMENT_DURATION = 10.0
SEGMENT_SAMPLES = int(SAMPLE_RATE * SEGMENT_DURATION)
LOUDNESS_RANGE = (-18.0, -14.0)

# The paper evaluation follows the seven-processor FxEncoder++ protocol and
# excludes reverb. This chain has 47 continuous parameters.
FX_CHAIN_ORDER = [
    "eq",
    "multiband_comp",
    "imager",
    "gain",
    "distortion",
    "delay",
    "limiter",
]
FXENCPP_TOTAL_PARAMS = 47
FX_PROB = {
    "eq": 0.6,
    "distortion": 0.3,
    "multiband_comp": 0.8,
    "gain": 0.6,
    "imager": 0.6,
    "limiter": 0.6,
    "delay": 0.6,
}
FX_CHAIN_ORDER_WITH_REVERB = [
    "eq",
    "distortion",
    "multiband_comp",
    "gain",
    "imager",
    "limiter",
    "delay",
    "reverb",
]
FXENCPP_TOTAL_PARAMS_WITH_REVERB = 72

NUM_SAMPLES_PER_INSTRUMENT = 100
MUSDB_INSTRUMENTS = ["drums", "bass", "vocals", "other"]
MEDLEYDB_INSTRUMENTS = ["mandolin", "alto_saxophone", "horn", "trumpet"]
DATASETS = {
    "musdb18": {
        "instruments": MUSDB_INSTRUMENTS,
        "fxnorm_dir": FXNORM_MUSDB_DIR,
        "raw_dir": MUSDB18_ROOT,
    },
    "medleydb": {
        "instruments": MEDLEYDB_INSTRUMENTS,
        "fxnorm_dir": FXNORM_MEDLEYDB_DIR,
        "raw_dir": MEDLEYDB_ROOT,
    },
}

ITO_CONFIG = {
    # This matches the manuscript. The exact command used for the archived
    # table run must still be verified against the original experiment host.
    "lr": 0.01,
    "n_iters": 200,
    "optimizer": "adam",
    "scheduler": "cosine",
    "loss_type": "embedding",
    "num_restarts": 7,
    "sample_batch_size": 1,
    "es_patience": 50,
}
ITO_CONFIG_FAST = {
    **ITO_CONFIG,
    "n_iters": 30,
    "num_restarts": 1,
    "es_patience": 0,
}

MODELS = {
    "relfx": {
        "ckpt_path": os.environ.get("RELFX_CHECKPOINT", ""),
        "model_sample_rate": SAMPLE_RATE,
        "embed_dim": 2048,
        "proj_dim": 128,
        "description": "RelFx ISMIR 2026 paper checkpoint",
        "fusion_type": "diff_gate",
        "cross_attn_stages": [3, 5],
        "cross_attn_heads": 4,
        "cross_attn_pool": 4,
        "cross_attn_scale": 0.1,
    }
}


def ensure_dirs() -> None:
    for directory in (DATA_DIR, TRIPLETS_DIR, RESULTS_DIR):
        Path(directory).mkdir(parents=True, exist_ok=True)
