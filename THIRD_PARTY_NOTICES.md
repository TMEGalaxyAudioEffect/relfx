# Third-Party Notices

## FxEncoder++

RelFx builds on the FxEncoder++ research implementation:

- Project: Fx-Encoder++
- Repository: https://github.com/SonyResearch/Fx-Encoder_PlusPlus
- Audited revision: `7b1e1f7efaa8a7c212ae4d62c01a5bac5f61c47b`
- Upstream license: Creative Commons Attribution-NonCommercial 4.0

The following release files are copied from or adapted from FxEncoder++:

- `src/relfx/fx_chain/ddsp_cores.py`
- `src/relfx/fx_chain/dsp_signal.py`
- `src/relfx/fx_chain/fx_processors.py`
- `src/relfx/fx_chain/fx_aug.py`
- `src/relfx/fx_chain/constants.py`
- portions of the Cnn14-style frontend and convolutional backbone in
  `src/relfx/model.py`

The upstream license text is retained at
`licenses/CC-BY-NC-4.0-FXENCODERPP.txt`. Modifications include package-relative
imports, the RelFx effect-chain configuration, and the dual-branch
cross-attention architecture.

Do not remove this notice or represent the listed components as exclusively
first-party work.

## PANNs / audioset_tagging_cnn

The Cnn14-style initialization helpers, convolutional blocks, log-mel
frontend, and pooling flow in `src/relfx/model.py` trace to PANNs:

- Project: PANNs / audioset_tagging_cnn
- Repository: https://github.com/qiuqiangkong/audioset_tagging_cnn
- Audited revision: `542c8c8bbb4287de5b2b622bc502423b27474b33`
- Upstream license: MIT
- Copyright: Copyright (c) 2018-2020 Qiuqiang Kong

This material reached RelFx through the FxEncoder++ implementation and was
substantially adapted into the shared dual-branch encoder. The original MIT
license text is retained at `licenses/MIT-PANNS.txt`.

## Safetensors

The public model artifact is stored and loaded using Safetensors 0.7.0:

- Project: Safetensors
- Repository: https://github.com/huggingface/safetensors
- Upstream license: Apache License 2.0

The upstream license text is retained at
`licenses/APACHE-2.0-SAFETENSORS.txt`.

## Python dependencies

The repository depends on PyTorch, torchaudio, torchlibrosa, dasp-pytorch,
torchcomp, auraloss, NumPy, SciPy, SoundFile, Numba, pyloudnorm, tqdm, and
PyYAML. The validated SoundFile 0.13.1 wheel bundles libsndfile 1.2.2 under
LGPL-2.1-or-later. These licenses are not replaced by any repository-level
license. When redistributing dependency binaries, retain the license and notice
files bundled with those distributions.
