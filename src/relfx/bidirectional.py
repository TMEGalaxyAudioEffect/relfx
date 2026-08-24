"""Utilities for training ordered pairs in both relation directions."""

from __future__ import annotations

import torch


def swap_ordered_pairs(
    first: torch.Tensor,
    second: torch.Tensor,
    reverse_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Swap selected pairs while preserving all non-selected examples."""
    mask = reverse_mask[:, None, None]
    return torch.where(mask, second, first), torch.where(mask, first, second)


def orient_fusion_for_regression(
    fusion: torch.Tensor,
    reverse_mask: torch.Tensor,
) -> torch.Tensor:
    """Map reverse-direction antisymmetric fusion back to forward labels."""
    if not reverse_mask.any():
        return fusion
    oriented = fusion.clone()
    oriented[reverse_mask] = -fusion[reverse_mask]
    return oriented
