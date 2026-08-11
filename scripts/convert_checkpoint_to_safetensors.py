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
    return parser.parse_args()


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
