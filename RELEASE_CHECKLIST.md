# Release Checklist

## Blocking

- [ ] Obtain written company approval for source publication.
- [ ] Confirm patent/public-disclosure status.
- [ ] Confirm the first-party copyright owner and replace the staging LICENSE.
- [ ] Complete legal review of FxEncoder++ derivative-code obligations.
- [ ] Complete dependency license and vulnerability scanning.
- [x] Recover the epoch-199 checkpoint used by the archived paper evaluations.
- [x] Run `scripts/verify_checkpoint.py` and record the checkpoint SHA-256.
- [ ] Confirm whether the archived ITO run used learning rate `0.01` or `0.05`.
- [ ] Locate and archive the V8 checkpoint and structural-evaluation output if
      V8 artifacts are promised.
- [ ] Approve the final model-weight license and Hugging Face access terms.

## Technical

- [x] Run release unit tests in the current environment.
- [x] Run CPU embedding extraction with an approved internal validation pair.
- [x] Verify the epoch-199 checkpoint against V6 metadata and archived logs.
- [x] Build the Python wheel.
- [x] Confirm the ITO and training chains expose 47 and 72 parameters.
- [ ] Repeat unit tests in a new clean environment.
- [x] Run one-GPU embedding extraction with an approved internal validation
      pair.
- [ ] Run one-GPU embedding extraction with a rights-cleared public audio pair.
- [x] Run one GPU training batch and one validation batch without saving.
- [x] Run a small ITO smoke evaluation.
- [ ] Reproduce the paper-table medians from the archived checkpoint.
- [ ] Pin a tested CUDA/PyTorch environment.
- [x] Run `scripts/release_scan.py` with no findings.
- [ ] Review all audio examples for redistribution rights.
- [ ] Replace placeholder citation metadata with the proceedings record.

## Publication

- [ ] Create a fresh public Git history; do not copy development history.
- [ ] Tag the source snapshot, for example `ismir2026-v1.0`.
- [ ] Upload weights only to the gated model repository.
- [ ] Link the immutable source tag and gated model revision from the paper
      website.
- [ ] Archive access approvals and the exact terms accepted by each recipient.
