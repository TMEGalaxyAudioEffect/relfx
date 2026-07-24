# RelFx: Learning Relative Audio Effects Representations

RelFx encodes the audio-effects transformation between two audio clips. Given
a reference clip and a processed clip, it produces a normalized
2048-dimensional embedding that describes how their effects differ, even when
the clips contain different musical content.

The embedding can be used to compare or retrieve processing styles and to guide
differentiable audio-effect parameter matching. This repository accompanies
the ISMIR 2026 paper **"Beyond Dry References: Learning Relative Audio Effects
Representations via Contrastive Distance Learning."**

> This is a private review candidate. Source and model licensing still require
> owner approval. The verified epoch-199 checkpoint is distributed separately
> and is never stored in this source repository.

## Quick start

### 1. Install

The release was tested with Python 3.10, PyTorch 2.7.1+cu118, Torchaudio
2.7.1+cu118, and an NVIDIA Tesla T4.

```bash
git clone https://github.com/meteroad/relfx-ismir2026-release.git
cd relfx-ismir2026-release

python -m venv .venv
source .venv/bin/activate

pip install torch==2.7.1 torchaudio==2.7.1 \
  --index-url https://download.pytorch.org/whl/cu118
pip install -e .
```

Confirm that PyTorch can see the intended device:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

CPU inference is supported but is substantially slower.

### 2. Obtain the checkpoint

Request access to the gated RelFx model repository and keep the downloaded
checkpoint outside this Git checkout:

```bash
export RELFX_CHECKPOINT_PATH=/absolute/path/to/relfx-ismir2026-epoch199.pt
python scripts/verify_checkpoint.py "$RELFX_CHECKPOINT_PATH"
```

The verifier checks the epoch, V6 architecture, model configuration, training
switches, state dictionary, and SHA-256. The final Hugging Face URL and
immutable model revision will be added after owner approval.

### 3. Extract an embedding

Use two audio files that you are authorized to process:

```bash
bash scripts/run_demo.sh \
  /path/to/reference.wav \
  /path/to/processed.wav \
  outputs/demo/embedding.npy
```

RelFx resamples both files to 44.1 kHz, converts them to stereo, and uses the
first ten seconds. The files do not need to contain the same performance. The
saved NumPy array is a normalized 2048-dimensional effects embedding.

For direct control, run `scripts/embed.py --help`.

## ITO parameter matching

The inference-time optimization (ITO) evaluation uses the embedding to guide a
47-parameter differentiable effect chain toward a reference transformation.

Install the evaluation dependencies:

```bash
pip install -e ".[train]"
```

Prepare authorized MUSDB18 inputs and run the Standard protocol:

```bash
python -m relfx.evaluation.ito.prepare_data \
  --dataset musdb18 \
  --output-dir outputs/ito-data

python -m relfx.evaluation.ito.run_eval \
  --checkpoint "$RELFX_CHECKPOINT_PATH" \
  --triplets-dir outputs/ito-data/musdb18 \
  --embed-mode dry_wet \
  --output outputs/ito-results/standard.json
```

The available protocols are `wet_wet` (Self-ref), `dry_wet` (Standard), and
`cross_dry_wet` (Oracle). Add
`--limit 1 --n-iters 1 --num-restarts 1 --es-patience 0` for a quick
installation check.

## Training

RelFx can be trained on one or more folders of legally authorized audio:

```bash
pip install -e ".[train]"

torchrun --nproc_per_node=4 -m relfx.train \
  --audio-dir /path/to/source-a \
  --audio-dir /path/to/source-b
```

See [docs/training.md](docs/training.md) for the tested configuration, a
single-batch validation command, and implementation compatibility notes. See
[docs/data-format.md](docs/data-format.md) for supported audio layouts and
optional structural metadata.

## Repository structure

```text
.
├── src/relfx/                 # model, training, data, effects, and ITO code
├── scripts/                   # embedding, validation, and release utilities
├── configs/paper.yaml         # sanitized paper configuration
├── docs/                      # training and data-format documentation
├── tests/                     # focused release tests
├── pyproject.toml             # package metadata and dependency groups
└── THIRD_PARTY_NOTICES.md     # third-party attribution and license scope
```

## Limitations and license

- RelFx estimates one global transformation between two clips; it does not
  disentangle independent source-wise effects inside a mixture.
- The reported evaluation uses a controlled differentiable DSP chain and does
  not cover commercial plug-ins or every real-world preset.
- Relative embeddings are not unique physical processor parameters.
- Training audio is not distributed. Use only audio and reference material
  that you have the right to process.

The first-party source license and model-weight terms require owner approval
before public distribution. FxEncoder++-derived components remain subject to
CC BY-NC 4.0; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Citation

If you use RelFx in your research, please cite:

```bibtex
@inproceedings{liu2026beyond,
  title     = {Beyond Dry References: Learning Relative Audio Effects
               Representations via Contrastive Distance Learning},
  author    = {Liu, Xinlu and Lin, Huibin and Wei, Weixing and Yan, Zhenhai},
  booktitle = {Proceedings of the 27th International Society for Music
               Information Retrieval Conference (ISMIR)},
  year      = {2026}
}
```

The DOI and page range will be added after the proceedings entry is available.
