"""Checkpoint loading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .model import create_model


def load_model(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Load a RelFx V6 checkpoint and return the model and metadata."""
    checkpoint = torch.load(
        Path(checkpoint_path), map_location=device, weights_only=False
    )
    model_config = checkpoint.get("model_config", {})
    model = create_model(
        fusion_type=model_config.get("fusion_type", "diff_gate"),
        cross_attn_stages=model_config.get("cross_attn_stages", [3, 5]),
        cross_attn_heads=model_config.get("cross_attn_heads", 4),
        cross_attn_pool=model_config.get("cross_attn_pool", 4),
        cross_attn_scale=model_config.get("cross_attn_scale", 0.1),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    metadata = {
        key: value
        for key, value in checkpoint.items()
        if not key.endswith("_state_dict")
    }
    return model, metadata
