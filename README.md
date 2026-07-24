# RelFx: Relative Audio Effects Representations

This directory is the sanitized research-release candidate for the ISMIR 2026
paper **"Beyond Dry References: Learning Relative Audio Effects
Representations via Contrastive Distance Learning."**

It is intentionally separate from the development workspace. It contains the
paper model, training pipeline, generic data adapters, embedding example, and
inference-time optimization (ITO) evaluation. It does not contain training
audio, company SDKs, internal effect assets, later production models, or
development checkpoints.

> Release status: source and model licensing still require owner approval. The
> epoch-199 paper checkpoint candidate has passed metadata, history-log, GPU
> inference, and ITO smoke checks, but remains outside this source repository.
> Do not publish this candidate until every blocking item in
> [RELEASE_CHECKLIST.md](RELEASE_CHECKLIST.md) is complete.
> Current technical checks are recorded in
> [VALIDATION_REPORT.md](VALIDATION_REPORT.md).

## Repository layout

```text
.
├── src/relfx/                 # model, training, data, effects, and ITO code
├── scripts/embed.py           # primary inference entry point
├── scripts/run_demo.sh        # concise user-audio launcher
├── demo/                      # instructions for user-supplied audio
├── configs/paper.yaml         # sanitized paper configuration
├── requirements.txt           # inference environment
├── requirements-train.txt     # inference plus training/evaluation packages
├── MODEL_CARD.md              # gated-weight model-card template
└── THIRD_PARTY_NOTICES.md
```

## Quick start

### 1. Create an environment

Install the PyTorch build appropriate for the target CUDA runtime first, then
install the inference environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Use `requirements-train.txt` instead when training or running ITO evaluation.
The effect-chain dependencies and FxEncoder++ attribution are documented in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

### 2. Obtain model assets

Request access to the gated RelFx model repository and accept its model-card
terms. Keep the checkpoint outside this Git checkout and expose it through:

```bash
export RELFX_CHECKPOINT_PATH=/absolute/path/to/relfx-checkpoint.pt
```

The final Hugging Face URL, approved filename, and immutable checkpoint
revision are pending owner approval. No weight file is stored in this source
repository.

### 3. Extract an embedding

RelFx represents the transformation between a reference signal and a processed
signal. Use audio that you are authorized to process:

```bash
bash scripts/run_demo.sh \
  /path/to/reference.wav \
  /path/to/processed.wav \
  outputs/demo/embedding.npy
```

For direct control, call `scripts/embed.py` with `--checkpoint`, `--reference`,
`--processed`, and `--output`.

## Train

Install the training environment:

```bash
pip install -r requirements-train.txt
```

Pass one or more audio roots explicitly:

```bash
torchrun --nproc_per_node=4 -m relfx.train \
  --audio-dir /path/to/source-a \
  --audio-dir /path/to/source-b
```

Alternatively, set `RELFX_AUDIO_DIRS` to an OS-path-separator-delimited list.
Optional structural metadata and density filtering are described in
[docs/data-format.md](docs/data-format.md).

To validate one complete training and validation step without writing
checkpoints:

```bash
python -m relfx.train \
  --audio-dir /path/to/smoke-test-audio \
  --epochs 1 --batch-size 1 --num-workers 0 --grad-accum-steps 1 --amp \
  --max-train-batches 1 --max-val-batches 1 --no-save
```

The public configuration uses:

- 44.1 kHz stereo, 10-second clips;
- an eight-processor, 72-parameter differentiable training chain;
- dual shared CNN branches with cross-attention at stages 3 and 5;
- a 2048-dimensional effects embedding and 128-dimensional projection;
- dynamic per-effect sampling probabilities;
- AdamW with an initial learning rate of `3e-4`.

The loss module preserves the historical contrastive implementation used by
the checkpoint. Its positive logit appears twice in the denominator; this is
documented in code and is not silently changed in the release.

## ITO evaluation

Prepare licensed MUSDB18 inputs:

```bash
python -m relfx.evaluation.ito.prepare_data \
  --dataset musdb18 \
  --output-dir outputs/ito-data
```

Then run one of the paper's embedding protocols:

```bash
python -m relfx.evaluation.ito.run_eval \
  --checkpoint /path/to/relfx-ismir2026-epoch199.pt \
  --triplets-dir outputs/ito-data/musdb18 \
  --embed-mode dry_wet \
  --output outputs/ito-results/standard.json
```

`wet_wet`, `dry_wet`, and `cross_dry_wet` correspond to the Self-ref,
Standard, and Oracle reference constructions in the paper.
For a one-sample installation check, add
`--limit 1 --n-iters 1 --num-restarts 1 --es-patience 0`.

## Scope and limitations

- The model estimates a global transformation between two clips. It does not
  disentangle independent source-wise effects inside a mixture.
- The training effect order is shuffled, but the paper does not include a
  held-out order-generalization benchmark.
- Evaluation uses a controlled differentiable DSP chain; commercial plug-ins,
  non-differentiable processors, and a systematic real-world preset benchmark
  are outside the reported scope.
- Training audio is not distributed. Users must provide legally authorized
  data.

## Responsible use

- Process only audio and reference material that you have the right to use.
- Respect the gated checkpoint terms and every third-party dependency license.
- Do not present relative embeddings as unique or physically exact processor
  parameter estimates.

## License and acknowledgements

The first-party source license and model-weight terms must be approved by the
project owner before public distribution. The current root `LICENSE` is a
release-candidate hold notice, not the final public license. FxEncoder++-derived
components remain subject to CC BY-NC 4.0; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Citation

Citation metadata will be finalized after the proceedings entry is available.
See [CITATION.cff](CITATION.cff).
