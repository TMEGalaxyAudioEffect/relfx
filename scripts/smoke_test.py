#!/usr/bin/env python3
"""Small CPU/GPU forward-pass check for the release model."""

from __future__ import annotations

import argparse

import torch

from relfx.model import create_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--seconds", type=float, default=1.0)
    args = parser.parse_args()

    model = create_model().to(args.device).eval()
    samples = int(44_100 * args.seconds)
    reference = torch.randn(1, 2, samples, device=args.device) * 0.01
    processed = torch.randn(1, 2, samples, device=args.device) * 0.01
    with torch.inference_mode():
        output = model(reference, processed)
    assert output["embedding"].shape == (1, 128)
    assert output["z"].shape == (1, 128)
    assert output["fusion"].shape == (1, 2048)
    assert torch.equal(output["embedding"], output["z"])
    print(
        f"Smoke test passed on {args.device}: "
        f"fusion={tuple(output['fusion'].shape)}, "
        f"embedding={tuple(output['embedding'].shape)}"
    )


if __name__ == "__main__":
    main()
