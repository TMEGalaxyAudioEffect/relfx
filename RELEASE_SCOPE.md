# Release Scope

## Public source repository

- RelFx V6 dual-branch architecture.
- Historical training loss and auxiliary parameter-regression head.
- Generic audio-folder dataset, song-level split, cross-segment sampler, and
  optional public structural-segment manifest.
- Dynamic effect-probability scheduler and DDP training entry point.
- Differentiable training effect chain with third-party attribution.
- Embedding extraction example.
- Seven-effect ITO data preparation and evaluation.
- Paper hyperparameter configuration, tests, model card, and documentation.

## Gated model repository

- Exact epoch-199 checkpoint used for the paper tables.
- Checkpoint SHA-256 and immutable revision identifier.
- Model card, training-data summary, intended use, and limitations.
- Manual access request for non-commercial research.
- Owner-approved model-weight license.

## Not released

- Licensed training audio or metadata that identifies internal assets.
- Company SDK effect implementations, presets, impulse responses, or binaries.
- Models trained with the later SDK effect library.
- Production inference code, services, deployment configuration, or telemetry.
- Internal experiment logs, server paths, review files, patent materials, and
  development checkpoints.
- Third-party baseline weights.

## Reproducibility boundary

The source release allows researchers to inspect and rerun the method with
their own authorized audio. The gated paper checkpoint and public MUSDB18-based
evaluation path support the reported main-model evaluation. The licensed
internal training collection is not redistributed, so bit-for-bit retraining
of the mixed-data checkpoint is not promised.
