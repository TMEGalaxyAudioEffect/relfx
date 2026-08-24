#!/usr/bin/env python3
"""Extract a relative-effects embedding from a pair of audio files."""

from __future__ import annotations

import argparse

import numpy as np
import torch

from relfx.audio import load_audio
from relfx.checkpoint import load_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--processed", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--representation",
        choices=("projected", "fusion"),
        default="projected",
        help=(
            "Representation to save: 'projected' is the normalized 128-d "
            "output used for retrieval and ITO; 'fusion' is the raw 2048-d "
            "intermediate used by the auxiliary parameter-regression head"
        ),
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def select_representation(
    output: dict[str, torch.Tensor], representation: str
) -> torch.Tensor:
    if representation == "projected":
        return output["embedding"]
    if representation == "fusion":
        return output["fusion"]
    raise ValueError(f"Unknown representation: {representation}")


def main() -> None:
    args = parse_args()
    model, metadata = load_model(args.checkpoint, device=args.device)
    reference = load_audio(args.reference).unsqueeze(0).to(args.device)
    processed = load_audio(args.processed).unsqueeze(0).to(args.device)
    with torch.inference_mode():
        output = model(reference, processed)
        representation = select_representation(output, args.representation)
    np.save(args.output, representation.squeeze(0).cpu().numpy())
    print(
        f"Saved {args.representation} representation "
        f"{tuple(representation.shape)} from "
        f"checkpoint epoch {metadata.get('epoch', 'unknown')} to {args.output}"
    )


if __name__ == "__main__":
    main()
