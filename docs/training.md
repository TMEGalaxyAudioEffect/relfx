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
  --audio-dir /path/to/source-b \
  --structure-segments /path/to/segments.json \
  --density-filter-audio-dir /path/to/moisesdb \
  --stem-audio-dir /path/to/moisesdb
```

Alternatively, use `RELFX_AUDIO_DIRS`, `RELFX_STRUCTURE_SEGMENTS`,
`RELFX_DENSITY_FILTER_AUDIO_DIRS`, and `RELFX_STEM_AUDIO_DIRS`. Variables
containing multiple paths use the operating system's path separator. The
loader performs a song-disjoint training/validation split. The default paper
recipe requires a structural-section manifest:

```bash
export RELFX_STRUCTURE_SEGMENTS=/path/to/segments.json
```

See [data-format.md](data-format.md) for the manifest schema and filename/song
ID conventions.

## Paper configuration

The public paper configuration uses:

- 44.1 kHz stereo, 10-second clips;
- adjacent, non-overlapping clip pairs sampled within one verse or chorus;
- an eight-processor, 72-parameter differentiable training chain;
- dual shared CNN branches with cross-attention at stages 3 and 5;
- a 2048-dimensional fusion representation and 128-dimensional projected
  representation;
- dynamic per-effect sampling probabilities;
- AdamW with an initial learning rate of `5e-4`.

The complete sanitized configuration is in
[`configs/paper.yaml`](../configs/paper.yaml).

## Bidirectional variant

The main release checkpoint uses the Base model. To train the separate
bidirectional-compatible paper architecture, select it explicitly:

```bash
torchrun --nproc_per_node=4 -m relfx.train \
  --audio-dir /path/to/source-a \
  --audio-dir /path/to/source-b \
  --model-variant bidirectional \
  --bidirectional-flip-ratio 0.5
```

This selects the architecture used for the paper's bidirectional experiment:
a gate conditioned on the sum of the branch embeddings, gated difference-only
fusion, and a Tanh projection. One mask swaps the input order for both positive
examples and the negative example in each triplet. For the auxiliary
parameter-regression objective, the strictly reversed fusion is negated back
to the forward direction before the regression head.

Only the fusion operation is mathematically guaranteed to be antisymmetric.
The projection Linear layers use learned biases, so antisymmetry of the final
L2-normalized representation is an empirical property. No checkpoint for this
variant is included in the current release.

The untrained architecture can be checked independently of any checkpoint:

```bash
python scripts/smoke_test.py --model-variant bidirectional --device cpu
```

## Single-batch validation

Run one complete training and validation step without writing checkpoints:

```bash
python -m relfx.train \
  --audio-dir /path/to/smoke-test-audio \
  --no-cross-segment \
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
