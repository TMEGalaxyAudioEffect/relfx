# Training RelFx

## Environment

Install the training and ITO dependencies from the package metadata:

```bash
pip install -e ".[train]"
```

The validated GPU environment uses Python 3.10, PyTorch 2.7.1+cu118,
Torchaudio 2.7.1+cu118, and an NVIDIA Tesla T4.

## Audio inputs

Pass one or more audio roots with repeated `--audio-dir` arguments:

```bash
torchrun --nproc_per_node=4 -m relfx.train \
  --audio-dir /path/to/source-a \
  --audio-dir /path/to/source-b
```

Alternatively, set `RELFX_AUDIO_DIRS` to an OS-path-separator-delimited list.
The loader performs a song-disjoint training/validation split. See
[data-format.md](data-format.md) for filename conventions and optional
structural metadata.

## Paper configuration

The public paper configuration uses:

- 44.1 kHz stereo, 10-second clips;
- an eight-processor, 72-parameter differentiable training chain;
- dual shared CNN branches with cross-attention at stages 3 and 5;
- a 2048-dimensional effects embedding and 128-dimensional projection;
- dynamic per-effect sampling probabilities;
- AdamW with an initial learning rate of `3e-4`.

The complete sanitized configuration is in
[`configs/paper.yaml`](../configs/paper.yaml).

## Single-batch validation

Run one complete training and validation step without writing checkpoints:

```bash
python -m relfx.train \
  --audio-dir /path/to/smoke-test-audio \
  --epochs 1 \
  --batch-size 1 \
  --num-workers 0 \
  --grad-accum-steps 1 \
  --amp \
  --max-train-batches 1 \
  --max-val-batches 1 \
  --no-save
```

## Historical loss compatibility

The released loss module preserves the contrastive implementation used to
train the paper checkpoint. Its positive logit appears twice in the
denominator. This historical behavior is covered by a regression test and is
not silently changed in the release.
