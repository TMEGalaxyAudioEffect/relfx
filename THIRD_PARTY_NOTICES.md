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

## Python dependencies

The repository depends on PyTorch, torchaudio, torchlibrosa, dasp-pytorch,
torchcomp, auraloss, NumPy, SciPy, SoundFile, Numba, pyloudnorm, and tqdm.
Their licenses are not replaced by any repository-level license. A final
software-composition scan and notice bundle are required before publication.
