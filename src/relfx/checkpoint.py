"""Checkpoint loading helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import load_file

from .model import create_model


RELFX_METADATA_KEY = "relfx_metadata"


def load_safetensors_artifact(
    checkpoint_path: str | Path,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Load tensors and JSON metadata from a RelFx safetensors artifact."""
    path = Path(checkpoint_path)
    if path.suffix.lower() != ".safetensors":
        raise ValueError(
            "Public RelFx checkpoints must use the .safetensors format"
        )

    state_dict = load_file(str(path), device="cpu")
    if not state_dict:
        raise ValueError(f"Checkpoint contains no tensors: {path}")

    with safe_open(str(path), framework="pt", device="cpu") as handle:
        encoded_metadata = (handle.metadata() or {}).get(RELFX_METADATA_KEY)
    if encoded_metadata is None:
        raise ValueError(f"Checkpoint is missing {RELFX_METADATA_KEY!r}: {path}")

    try:
        metadata = json.loads(encoded_metadata)
    except json.JSONDecodeError as error:
        raise ValueError(f"Checkpoint metadata is not valid JSON: {path}") from error
    if not isinstance(metadata, dict):
        raise ValueError(f"Checkpoint metadata must be a JSON object: {path}")

    return state_dict, metadata


def load_model(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Load a public RelFx V6 artifact and return the model and metadata."""
    state_dict, metadata = load_safetensors_artifact(checkpoint_path)
    model_config = metadata.get("model_config", {})
    model = create_model(
        fusion_type=model_config.get("fusion_type", "diff_gate"),
        cross_attn_stages=model_config.get("cross_attn_stages", [3, 5]),
        cross_attn_heads=model_config.get("cross_attn_heads", 4),
        cross_attn_pool=model_config.get("cross_attn_pool", 4),
        cross_attn_scale=model_config.get("cross_attn_scale", 0.1),
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, metadata
