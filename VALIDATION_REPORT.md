# Release-Candidate Validation

Validation date: 2026-07-24

## Paper checkpoint candidate

The release code path was validated with the recovered epoch-199 checkpoint
used by the archived cascaded-v6 parameter-matching runs. This checkpoint is
not copied into the source repository and is intended for the separately
gated model repository after owner approval.

- Version: `v6`
- Epoch: `199`
- Size: `1,209,846,955` bytes
- SHA-256:
  `28a28395fc26b2eed510cd1c2e43ce6c485ef65869289a7f7c502d83a11d75bb`
- Validation loss: `1.3672196207`
- R@1 / R@5 / R@10: `0.726293 / 0.894397 / 0.936782`
- Cross-segment flag: enabled
- Fusion: `diff_gate`
- Cross-attention stages: `[3, 5]`

The verifier returned `verified: true` for epoch, V6 architecture, method
switches, cross-segment flag, model configuration, and model state dictionary.
The epoch and all stored validation metrics exactly match the historical
`cascaded-v6` ITO logs for bass, drums, vocals, and other.

## Packaged runtime

The final source state produced the following wheel:

- File: `relfx-0.1.0rc1-py3-none-any.whl`
- Size: `81,668` bytes
- SHA-256:
  `a4b5151f66a68aae02a560bf98f13ef0a07b2f39fcd5e218460cd925ff2c1435`

The wheel was installed into a temporary target and imported outside the
source tree. The declared runtime was Python 3.10, PyTorch 2.7.1+cu118, and
Torchaudio 2.7.1+cu118; `pip check` reported no broken requirements in that
runtime. GPU checks used an NVIDIA Tesla T4 with a 515.65.01 driver.

## GPU inference

A rights-controlled dry/wet pair already used by the internal demonstration
was passed through the public `scripts/run_demo.sh` entry point and the epoch
199 checkpoint.

- Output shape: `(2048,)`
- Data type: `float32`
- All values finite: yes
- L2 norm: `0.99999994`
- Value range: `[-0.05694941, 0.23550077]`

This verifies audio loading, checkpoint reconstruction, dual-branch inference,
normalization, and NumPy output from the packaged code.

## Training smoke test

The packaged `relfx.train` entry point ran one complete AMP training batch and
one validation batch on the Tesla T4, including:

- scanning 400 licensed 44.1 kHz stereo clips;
- song-disjoint splitting into 340 training and 60 validation clips;
- the 72-parameter differentiable training effect chain;
- model and combined-loss forward passes;
- backward propagation and an AdamW optimizer step;
- validation metrics and the no-output `--no-save` path.

The smoke run used batch size 1 and gradient accumulation 1. It completed with
training loss `0.9998` and validation loss `0.9841`. These values only confirm
execution and are not research results.

## ITO smoke test

The packaged ITO entry point ran on the real `bass/0000` triplet with the
Standard (`dry_wet`) construction, one restart, and one optimization
iteration. The 47-parameter differentiable chain, embedding loss, backward
pass, parameter update, Ld evaluation, and JSON output all completed.

- Baseline Ld: `2.4550`
- Smoke-test Ld: `2.7781`
- Serialized parameter count: `47`

This one-iteration result verifies the evaluation path only; it does not
reproduce or replace the full paper evaluation, and its random one-step Ld is
not a model-quality result.

## Other checks

- Three standard-library unit tests pass from the installed wheel.
- Historical contrastive-loss behavior is covered by a regression test.
- The release scan reports no internal paths, known identifiers, secrets,
  audio files, model weights, or oversized files.
- Generated build metadata and bytecode caches were removed from the source
  directory after packaging.

## Remaining validation

- Assign the approved public filename, gated model revision, ownership terms,
  and model license without copying the checkpoint into the source repository.
- Reproduce the archived full ITO medians after the exact evaluation learning
  rate and command are confirmed against the original experiment record.
