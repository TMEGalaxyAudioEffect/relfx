#!/usr/bin/env python3
"""Verify a release artifact and its cross-segment sampling provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from relfx.checkpoint import load_safetensors_artifact


EXPECTED_RECIPE_VERSION = "same-section-adjacent-v1"
EXPECTED_SOURCE_RECIPE_VERSION = "ismir2026-camera-ready-v1"
EXPECTED_CROSS_SEGMENT_POLICY = "same_section_adjacent"
EXPECTED_CHECKPOINT_VERSION = "v6-fix"
EXPECTED_SOURCE_CHECKPOINT_VERSION = "v6"
EXPECTED_EPOCH = 339
EXPECTED_ARTIFACT_SHA256 = (
    "ea186f4b5a029ae3ba23a17e9d48dafb7c488e9f6636d0a407054ab5a986abc0"
)
EXPECTED_TRAINING_SAMPLER_SHA256 = (
    "71f678694695c8c4390f5c487e4249e1db11d8a0a0b8e6fc9eb7ad927deef739"
)
EXPECTED_SOURCE_SCOPE_COUNTS = {
    "sq_presegmented_section": 4406,
    "moises_adjacent_within_stem": 2585,
    "quku_structural_section": 2000,
}
EXPECTED_SOURCE_SECTION_LABELS = [
    "chorus",
    "presegmented_section",
    "unsegmented_stem",
    "verse",
]
EXPECTED_MODEL_CONFIG = {
    "model_variant": "base",
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


def is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.lower())
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--expected-epoch", type=int)
    parser.add_argument("--expected-training-data", default="moisesdb-only")
    parser.add_argument(
        "--expected-sha256",
        default=EXPECTED_ARTIFACT_SHA256,
        help="Expected artifact hash (defaults to the current public release).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = Path(args.checkpoint)
    state_dict, metadata = load_safetensors_artifact(path)
    artifact_sha256 = sha256(path)
    sampling_config = metadata.get("sampling_config")
    expected_epoch = (
        args.expected_epoch
        if args.expected_epoch is not None
        else EXPECTED_EPOCH
    )

    checks = {
        "artifact_format": metadata.get("artifact_format")
        == "relfx-safetensors-v1",
        "epoch": metadata.get("epoch") == expected_epoch,
        "training_data": metadata.get("training_data")
        == args.expected_training_data,
        "version": metadata.get("version") == EXPECTED_CHECKPOINT_VERSION,
        "source_checkpoint_version": metadata.get("source_checkpoint_version")
        == EXPECTED_SOURCE_CHECKPOINT_VERSION,
        "cross_segment": metadata.get("cross_segment") is True,
        "cross_segment_recipe_version": metadata.get(
            "cross_segment_recipe_version"
        )
        == EXPECTED_RECIPE_VERSION,
        "source_cross_segment_recipe_version": metadata.get(
            "source_cross_segment_recipe_version"
        )
        == EXPECTED_SOURCE_RECIPE_VERSION,
        "cross_segment_policy": metadata.get("cross_segment_policy")
        == EXPECTED_CROSS_SEGMENT_POLICY,
        "sampling_config": (
            isinstance(sampling_config, dict)
            and sampling_config.get("cross_segment") is True
            and sampling_config.get("policy")
            == EXPECTED_CROSS_SEGMENT_POLICY
            and sampling_config.get("sample_rate") == 44_100
            and sampling_config.get("segment_samples") == 441_000
            and is_sha256(sampling_config.get("structural_manifest_sha256"))
            and sampling_config.get("sampler_source_sha256")
            == EXPECTED_TRAINING_SAMPLER_SHA256
            and sampling_config.get("dataset") == "MoisesDB"
            and sampling_config.get("section_labels") == []
            and sampling_config.get("source_section_labels")
            == EXPECTED_SOURCE_SECTION_LABELS
            and sampling_config.get("pairing_scope_counts")
            == {"full_audio": 2585}
            and sampling_config.get("source_manifest_pairing_scope_counts")
            == EXPECTED_SOURCE_SCOPE_COUNTS
            and sampling_config.get("eligible_files_before_split") == 2585
            and sampling_config.get("selected_files") == 2231
        ),
        "model_config": metadata.get("model_config") == EXPECTED_MODEL_CONFIG,
        "switches": metadata.get("switches") == EXPECTED_SWITCHES,
        "state_dict": bool(state_dict),
    }
    checks["sha256"] = artifact_sha256 == args.expected_sha256.lower()
    report = {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": artifact_sha256,
        "tensor_count": len(state_dict),
        "epoch": metadata.get("epoch"),
        "version": metadata.get("version"),
        "cross_segment_recipe_version": metadata.get(
            "cross_segment_recipe_version"
        ),
        "cross_segment_policy": metadata.get("cross_segment_policy"),
        "metrics": metadata.get("metrics"),
        "checks": checks,
        "verified": all(checks.values()),
    }
    print(json.dumps(report, indent=2))
    if not report["verified"]:
        raise SystemExit("Artifact does not match the expected release metadata")


if __name__ == "__main__":
    main()
