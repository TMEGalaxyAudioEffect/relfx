#!/usr/bin/env python3
"""Convert a trusted RelFx PyTorch checkpoint into a safe inference artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file


METADATA_KEY = "relfx_metadata"
ARTIFACT_FORMAT = "relfx-safetensors-v1"
SAME_SECTION_RELEASE_RECIPE = "same-section-adjacent-v1"
EXPECTED_SOURCE_RECIPE = "ismir2026-camera-ready-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a trusted local RelFx .pt checkpoint to safetensors. "
            "Never run this converter on an untrusted pickle checkpoint."
        )
    )
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--training-data",
        choices=("full-data", "moisesdb-only"),
        help="Record the training-data scope in the artifact metadata.",
    )
    parser.add_argument(
        "--release-recipe",
        choices=(SAME_SECTION_RELEASE_RECIPE,),
        help="Validate and normalize metadata for a named release recipe.",
    )
    return parser.parse_args()


def apply_release_recipe(
    metadata: dict, *, recipe: str | None, training_data: str | None
) -> dict:
    """Validate source provenance and add release-facing metadata."""
    if recipe is None:
        return metadata
    if recipe != SAME_SECTION_RELEASE_RECIPE:
        raise ValueError(f"Unsupported release recipe: {recipe}")
    if training_data != "moisesdb-only":
        raise ValueError(
            f"{recipe} currently supports only the MoisesDB-only release"
        )
    if metadata.get("version") != "v6":
        raise ValueError("Expected a source checkpoint with version='v6'")
    if metadata.get("cross_segment") is not True:
        raise ValueError("Source checkpoint did not enable cross-segment pairs")
    if metadata.get("cross_segment_policy") != "same_section_adjacent":
        raise ValueError(
            "Source checkpoint does not record same-section adjacent sampling"
        )

    source_recipe = metadata.get("cross_segment_recipe_version")
    if source_recipe != EXPECTED_SOURCE_RECIPE:
        raise ValueError(
            "Unexpected source recipe version: "
            f"{source_recipe!r} (expected {EXPECTED_SOURCE_RECIPE!r})"
        )

    sampling = metadata.get("sampling_config")
    if not isinstance(sampling, dict):
        raise ValueError("Source checkpoint has no sampling_config")
    source_scope_counts = sampling.get("pairing_scope_counts")
    if not isinstance(source_scope_counts, dict):
        raise ValueError("Source checkpoint has no pairing scope counts")
    moises_count = source_scope_counts.get("moises_adjacent_within_stem")
    if moises_count != sampling.get("eligible_files_before_split"):
        raise ValueError(
            "MoisesDB pairing count does not match eligible file count"
        )

    normalized = dict(metadata)
    normalized["source_checkpoint_version"] = metadata["version"]
    normalized["version"] = "v6-fix"
    normalized["source_cross_segment_recipe_version"] = source_recipe
    normalized["cross_segment_recipe_version"] = recipe

    model_config = dict(metadata.get("model_config", {}))
    model_variant = model_config.get("model_variant")
    if model_variant not in (None, "base"):
        raise ValueError(f"Expected Base model, found {model_variant!r}")
    model_config["model_variant"] = "base"
    normalized["model_config"] = model_config

    normalized_sampling = dict(sampling)
    normalized_sampling["source_manifest_pairing_scope_counts"] = dict(
        source_scope_counts
    )
    normalized_sampling["source_section_labels"] = list(
        sampling.get("section_labels", [])
    )
    normalized_sampling["pairing_scope_counts"] = {"full_audio": moises_count}
    normalized_sampling["section_labels"] = []
    normalized_sampling["dataset"] = "MoisesDB"
    normalized["sampling_config"] = normalized_sampling
    return normalized


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    if source.suffix.lower() not in {".pt", ".pth"}:
        raise SystemExit("Source must be a trusted .pt or .pth checkpoint")
    if output.suffix.lower() != ".safetensors":
        raise SystemExit("Output must use the .safetensors suffix")
    if output.exists():
        raise SystemExit(f"Refusing to overwrite existing output: {output}")

    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise SystemExit("Expected a checkpoint dictionary")
    raw_state_dict = checkpoint.get("model_state_dict")
    if not isinstance(raw_state_dict, dict) or not raw_state_dict:
        raise SystemExit("Checkpoint has no model_state_dict")

    state_dict: dict[str, torch.Tensor] = {}
    for name, tensor in raw_state_dict.items():
        if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
            raise SystemExit("model_state_dict must map strings to tensors")
        state_dict[name] = tensor.detach().cpu().contiguous()

    metadata = {
        key: value
        for key, value in checkpoint.items()
        if not key.endswith("_state_dict")
    }
    metadata["source_artifact_format"] = metadata.get("artifact_format")
    metadata["artifact_format"] = ARTIFACT_FORMAT
    metadata["source_artifact_sha256"] = sha256(source)
    if args.training_data is not None:
        metadata["training_data"] = args.training_data
    try:
        metadata = apply_release_recipe(
            metadata,
            recipe=args.release_recipe,
            training_data=args.training_data,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    encoded_metadata = json.dumps(
        metadata, allow_nan=False, separators=(",", ":"), sort_keys=True
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        state_dict,
        str(output),
        metadata={
            "format": "pt",
            "artifact_format": ARTIFACT_FORMAT,
            METADATA_KEY: encoded_metadata,
        },
    )

    reloaded = load_file(str(output), device="cpu")
    if reloaded.keys() != state_dict.keys():
        raise SystemExit("Tensor names changed during conversion")
    for name, tensor in state_dict.items():
        if not torch.equal(tensor, reloaded[name]):
            raise SystemExit(f"Tensor changed during conversion: {name}")
    with safe_open(str(output), framework="pt", device="cpu") as handle:
        reloaded_metadata = json.loads((handle.metadata() or {})[METADATA_KEY])
    if reloaded_metadata != metadata:
        raise SystemExit("Metadata changed during conversion")

    report = {
        "source": str(source),
        "source_sha256": metadata["source_artifact_sha256"],
        "output": str(output),
        "output_bytes": output.stat().st_size,
        "output_sha256": sha256(output),
        "tensor_count": len(state_dict),
        "state_element_count": sum(
            tensor.numel() for tensor in state_dict.values()
        ),
        "verified_exact": True,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
