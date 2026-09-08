# RelFx: Learning Relative Audio Effects Representations

**[Project page](https://relative-fx.github.io/)** ·
**[Paper (arXiv)](https://arxiv.org/abs/2608.10573)** ·
**[Model checkpoint](https://huggingface.co/TMEGalaxyAudioEffect/relfx-ismir2026)**

RelFx encodes the audio-effects transformation between two audio clips. Given
a reference clip and a processed clip, it produces a normalized
128-dimensional projected representation that describes how their effects
differ, even when the clips contain different musical content. The underlying
2048-dimensional fusion representation is also available for analysis.

The embedding can be used to compare or retrieve processing styles and to guide
differentiable audio-effect parameter matching. This repository accompanies
the ISMIR 2026 paper **"Beyond Dry References: Learning Relative Audio Effects
Representations via Contrastive Distance Learning."**

The source includes both paper architectures: the default Base Diff-Gate model
and the separately trained bidirectional-compatible variant. Released
checkpoints use the Base model only.

> First-party RelFx source code is released under the MIT License; third-party
> components remain under their respective licenses. MoisesDB-only checkpoints
> are hosted separately in the
> [RelFx Hugging Face model repository](https://huggingface.co/TMEGalaxyAudioEffect/relfx-ismir2026)
> under CC BY-NC-SA 4.0 and is never stored in this source repository.

## Quick start

### 1. Install

The release was tested with Python 3.10, PyTorch 2.7.1+cu118, Torchaudio
2.7.1+cu118, and an NVIDIA Tesla T4.

```bash
git clone https://github.com/TMEGalaxyAudioEffect/relfx.git
cd relfx

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

> A replacement checkpoint using same-section adjacent sampling is being
> prepared. The verifier in this source revision rejects earlier artifacts
> that do not record the `same_section_adjacent` sampling policy.

After the replacement is published, download it with the Hugging Face CLI:

```bash
pip install -U huggingface_hub

export RELFX_WEIGHTS_DIR="${RELFX_WEIGHTS_DIR:-$HOME/.cache/relfx}"
hf download TMEGalaxyAudioEffect/relfx-ismir2026 \
  model.safetensors \
  --local-dir "$RELFX_WEIGHTS_DIR"
```

Keep the checkpoint outside this Git checkout and point RelFx to the downloaded
file:

```bash
export RELFX_CHECKPOINT_PATH="$RELFX_WEIGHTS_DIR/model.safetensors"
python scripts/verify_checkpoint.py "$RELFX_CHECKPOINT_PATH"
```

The verifier checks the V6-fix checkpoint version, Base model configuration,
same-section adjacent sampling metadata, and state dictionary. It always
reports the SHA-256 and compares it when `--expected-sha256` is provided.

Legacy V6 artifacts remain loadable for inference, but only newly trained
V6-fix artifacts pass the release verifier.

The public MoisesDB-only artifact uses Safetensors and does not require pickle
deserialization. Training-resume checkpoints are not distributed.

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
saved NumPy array is the normalized 128-dimensional projected representation
used for contrastive training, retrieval, and ITO in the paper.

`scripts/embed.py` exposes both representation levels:

- `--representation projected` (default) saves the normalized 128-dimensional
  projection `z` used by the paper's retrieval and ITO evaluations.
- `--representation fusion` saves the raw 2048-dimensional fusion
  representation `e_fx` used as the input to the auxiliary parameter-regression
  head.

The Python API follows the same convention: `model.get_embedding(...)`,
`model(...)["embedding"]`, and `model(...)["z"]` all return the final
128-dimensional representation. The raw unprojected intermediate is available
as `model(...)["fusion"]` or `model.get_fusion_representation(...)`; pass
`normalized=True` to the latter when an L2-normalized copy is needed. Run
`scripts/embed.py --help` for direct CLI control.

## Model variants

- `base` (default) uses the Diff-Gate in Eq. (4) and a ReLU projection head.
  Swapping its two inputs is not guaranteed to reverse the representation. The
  public MoisesDB-only checkpoint is this variant.
- `bidirectional` implements Eqs. (6)-(7): its gate uses the branch sum, its
  fusion output contains only the gated difference, and its projection uses
  Tanh. The fusion output is strictly antisymmetric. Because both projection
  Linear layers retain learned biases, the final normalized 128-dimensional
  representation is only empirically approximately antisymmetric.

The bidirectional architecture and training path are public, but no
bidirectional checkpoint is distributed. Negating a Base embedding is not an
implementation of the bidirectional model. See
[docs/training.md](docs/training.md) for the explicit training command.

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
  --audio-dir /path/to/source-b \
  --structure-segments /path/to/segments.json \
  --density-filter-audio-dir /path/to/moisesdb
```

The paper recipe uses adjacent, non-overlapping 10-second clips. Structured
full mixes use verse/chorus boundaries, pre-segmented files use their complete
section, and unannotated stems use their complete audio range without a
synthetic structure label. Pass the sampling manifest directly or set
`RELFX_STRUCTURE_SEGMENTS`.

See [docs/training.md](docs/training.md) for the tested configuration, a
single-batch validation command, and implementation compatibility notes. See
[docs/data-format.md](docs/data-format.md) for supported audio layouts and
the required sampling manifest.

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

First-party RelFx source code is released under the MIT License. The model
checkpoint is released separately under CC BY-NC-SA 4.0. FxEncoder++-derived
components remain subject to CC BY-NC 4.0, and PANNs-derived portions retain
their MIT notice; see [LICENSE](LICENSE) and
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
