#!/usr/bin/env python3
"""Verify that a candidate artifact is the ISMIR 2026 paper checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from relfx.checkpoint import load_safetensors_artifact


EXPECTED_SHA256 = (
    "211717c01fd7c56c97cd93e1be226905e9a84f6147597bfa5b3f90fea67af078"
)

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
    parser.add_argument("--expected-epoch", type=int, default=179)
    parser.add_argument("--expected-training-data", default="moisesdb-only")
    parser.add_argument("--expected-sha256", default=EXPECTED_SHA256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = Path(args.checkpoint)
    state_dict, metadata = load_safetensors_artifact(path)
    artifact_sha256 = sha256(path)

    checks = {
        "sha256": artifact_sha256 == args.expected_sha256,
        "artifact_format": metadata.get("artifact_format")
        == "relfx-safetensors-v1",
        "epoch": metadata.get("epoch") == args.expected_epoch,
        "training_data": metadata.get("training_data")
        == args.expected_training_data,
        "version": metadata.get("version") == "v6",
        "cross_segment": metadata.get("cross_segment") is True,
        "model_config": metadata.get("model_config") == EXPECTED_MODEL_CONFIG,
        "switches": metadata.get("switches") == EXPECTED_SWITCHES,
        "state_dict": bool(state_dict),
    }
    report = {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": artifact_sha256,
        "tensor_count": len(state_dict),
        "epoch": metadata.get("epoch"),
        "version": metadata.get("version"),
        "metrics": metadata.get("metrics"),
        "checks": checks,
        "verified": all(checks.values()),
    }
    print(json.dumps(report, indent=2))
    if not report["verified"]:
        raise SystemExit("Artifact does not match the expected paper metadata")


if __name__ == "__main__":
    main()
