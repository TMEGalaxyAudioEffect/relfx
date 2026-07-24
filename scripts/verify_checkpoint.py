#!/usr/bin/env python3
"""Verify that a candidate artifact is the ISMIR 2026 paper checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


EXPECTED_MODEL_CONFIG = {
    "fusion_type": "diff_gate",
    "cross_attn_stages": [3, 5],
    "cross_attn_heads": 4,
    "cross_attn_pool": 4,
    "cross_attn_scale": 0.1,
}
EXPECTED_SWITCHES = {
    "param_regression": True,
    "soft_contrastive": False,
    "distance_margin": False,
    "hard_negative": False,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--expected-epoch", type=int, default=199)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = Path(args.checkpoint)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    checks = {
        "epoch": checkpoint.get("epoch") == args.expected_epoch,
        "version": checkpoint.get("version") == "v6",
        "cross_segment": checkpoint.get("cross_segment") is True,
        "model_config": checkpoint.get("model_config") == EXPECTED_MODEL_CONFIG,
        "switches": checkpoint.get("switches") == EXPECTED_SWITCHES,
        "model_state_dict": "model_state_dict" in checkpoint,
    }
    report = {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "epoch": checkpoint.get("epoch"),
        "version": checkpoint.get("version"),
        "metrics": checkpoint.get("metrics"),
        "checks": checks,
        "verified": all(checks.values()),
    }
    print(json.dumps(report, indent=2))
    if not report["verified"]:
        raise SystemExit("Checkpoint does not match the expected paper metadata")


if __name__ == "__main__":
    main()
