---
license: other
license_name: relfx-noncommercial-research
gated: true
library_name: relfx
tags:
  - audio
  - music
  - audio-effects
  - representation-learning
---

# RelFx ISMIR 2026

## Artifact status

This card is a release-candidate template. The epoch-199 paper checkpoint
candidate and its SHA-256 have been technically verified. Upload it only after
ownership, model-license terms, and the final gated repository are approved.

## Artifact identity

- Training epoch: `199`
- Checkpoint size: `1,209,846,955` bytes
- SHA-256:
  `28a28395fc26b2eed510cd1c2e43ce6c485ef65869289a7f7c502d83a11d75bb`
- Architecture: RelFx V6, `diff_gate`, cross-attention stages `[3, 5]`

Assign the public filename and immutable gated-repository revision during the
approved upload. Do not infer identity from a generic filename such as
`best.pt`; verify the SHA-256.

## Model description

RelFx is a dual-branch audio encoder that represents the relative effects
transformation between two audio clips. Both branches share a Cnn14-style
backbone and exchange intermediate information through cross-attention. The
released embedding dimension is 2048.

## Intended use

The checkpoint is intended for non-commercial research on audio-effects
representation, retrieval, analysis, and differentiable parameter matching.

## Out-of-scope use

- Commercial products or services.
- Claims that the model recovers unique physical processor parameters.
- Source-wise effect disentanglement from a mixed recording.
- Processing audio without the necessary rights or consent.

## Training data

The paper model was trained on:

- a licensed internal collection summarized in the paper as 6,447 mixed
  tracks from 2,681 songs (185.9 hours); and
- 2,585 stems from 240 MoisesDB songs (156.4 hours).

Training audio and internal metadata are not distributed with the model.

## Training effects

The training chain contains EQ, distortion, multiband compression, gain,
stereo imaging, limiting, delay, and reverberation. It has 72 continuous
parameters plus activation switches.

## Evaluation

The paper's parameter-matching evaluation uses a separate seven-effect,
47-parameter chain without reverb. Reported results must be tied to the exact
checkpoint revision and evaluation protocol.

## Limitations

See the source repository README for the global-transform, effect-order,
processor-domain, interpretability, and real-world evaluation limitations.

## License

The final owner-approved model license must replace
`MODEL_LICENSE_DRAFT.md`. Gated access alone does not define usage rights.
