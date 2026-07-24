# Demo Inputs

No audio is distributed in this repository.

Use two audio files that you are authorized to process:

- `reference.wav`: the reference or less-processed signal;
- `processed.wav`: the corresponding processed signal whose relative effects
  transformation should be embedded.

The files do not need to have identical musical content. RelFx resamples them
to 44.1 kHz, converts them to stereo, and pads or crops the first ten seconds.

Run the launcher from the repository root:

```bash
export RELFX_CHECKPOINT_PATH=/path/to/relfx-checkpoint.pt
bash scripts/run_demo.sh \
  /path/to/reference.wav \
  /path/to/processed.wav \
  outputs/demo/embedding.npy
```

The verified epoch-199 paper checkpoint will be hosted separately in a gated
model repository after approval. No checkpoint is part of this public source
repository.
