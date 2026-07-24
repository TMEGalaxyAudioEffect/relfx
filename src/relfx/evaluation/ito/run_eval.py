#!/usr/bin/env python3
"""Run RelFx inference-time parameter matching on prepared triplets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from relfx.checkpoint import load_model

from .config import ITO_CONFIG, ITO_CONFIG_FAST, RESULTS_DIR
from .fx_chain_wrapper import DifferentiableFxChain
from .parameter_matcher import ParameterMatcher


def _read_audio(path: Path) -> np.ndarray:
    audio, _ = sf.read(path, dtype="float32", always_2d=True)
    return audio.T


def _sample_dirs(root: Path, instrument: str | None) -> list[Path]:
    roots = [root / instrument] if instrument else [
        path for path in root.iterdir() if path.is_dir()
    ]
    samples = []
    for instrument_root in roots:
        if not instrument_root.is_dir():
            continue
        samples.extend(
            path
            for path in sorted(instrument_root.iterdir())
            if path.is_dir() and (path / "clean.wav").is_file()
        )
    return samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--triplets-dir", required=True)
    parser.add_argument("--output", default=str(Path(RESULTS_DIR) / "relfx.json"))
    parser.add_argument("--instrument")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--n-iters", type=int)
    parser.add_argument("--num-restarts", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--es-patience", type=int)
    parser.add_argument(
        "--embed-mode",
        choices=["wet_wet", "dry_wet", "cross_dry_wet"],
        default="dry_wet",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model, checkpoint_metadata = load_model(
        args.checkpoint, device=args.device
    )
    model._model_type = "cascaded_v4"

    config = dict(ITO_CONFIG_FAST if args.fast else ITO_CONFIG)
    overrides = {
        "n_iters": args.n_iters,
        "num_restarts": args.num_restarts,
        "lr": args.lr,
        "es_patience": args.es_patience,
    }
    for key, value in overrides.items():
        if value is not None:
            config[key] = value
    if config["n_iters"] < 1 or config["num_restarts"] < 1:
        raise ValueError("n_iters and num_restarts must be at least 1")
    if config["lr"] <= 0 or config["es_patience"] < 0:
        raise ValueError("lr must be positive and es_patience non-negative")
    fx_chain = DifferentiableFxChain(device=args.device)
    matcher = ParameterMatcher(
        fx_chain,
        loss_type=config["loss_type"],
        lr=config["lr"],
        n_iters=config["n_iters"],
        num_restarts=config["num_restarts"],
        sample_batch_size=config["sample_batch_size"],
        es_patience=config["es_patience"],
        device=args.device,
        embed_mode=args.embed_mode,
    )

    sample_dirs = _sample_dirs(Path(args.triplets_dir), args.instrument)
    if args.limit is not None:
        sample_dirs = sample_dirs[: args.limit]
    if not sample_dirs:
        raise RuntimeError("No prepared triplets were found")

    records = []
    for index, sample_dir in enumerate(sample_dirs, start=1):
        result = matcher.match_single(
            clean=_read_audio(sample_dir / "clean.wav"),
            target=_read_audio(sample_dir / "target.wav"),
            reference=_read_audio(sample_dir / "reference.wav"),
            clean_ref=_read_audio(sample_dir / "clean_ref.wav"),
            model=model,
            sample_idx=index,
            verbose=True,
            log_prefix=f"[{sample_dir.parent.name}]",
        )
        records.append(
            {
                "sample": str(sample_dir.relative_to(args.triplets_dir)),
                "instrument": sample_dir.parent.name,
                "ld": result["ld"],
                "params": result["params"],
            }
        )

    by_instrument = {}
    for instrument in sorted({record["instrument"] for record in records}):
        values = [
            record["ld"]
            for record in records
            if record["instrument"] == instrument
        ]
        by_instrument[instrument] = {
            "count": len(values),
            "median_ld": float(np.median(values)),
            "mean_ld": float(np.mean(values)),
        }

    output = {
        "checkpoint_epoch": checkpoint_metadata.get("epoch"),
        "embed_mode": args.embed_mode,
        "ito_config": config,
        "summary": by_instrument,
        "samples": records,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()
