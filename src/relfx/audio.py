"""Audio loading utilities shared by release examples."""

from __future__ import annotations

from pathlib import Path

import soundfile as sf
import torch
import torchaudio


def load_audio(
    path: str | Path,
    *,
    sample_rate: int = 44_100,
    duration: float = 10.0,
) -> torch.Tensor:
    """Load, resample, peak-normalize, and pad/crop audio to stereo."""
    audio, source_rate = sf.read(path, dtype="float32", always_2d=True)
    waveform = torch.from_numpy(audio.T)
    if waveform.shape[0] == 1:
        waveform = waveform.repeat(2, 1)
    elif waveform.shape[0] > 2:
        waveform = waveform[:2]
    if source_rate != sample_rate:
        waveform = torchaudio.functional.resample(
            waveform, source_rate, sample_rate
        )

    target_samples = int(sample_rate * duration)
    if waveform.shape[-1] < target_samples:
        waveform = torch.nn.functional.pad(
            waveform, (0, target_samples - waveform.shape[-1])
        )
    waveform = waveform[..., :target_samples]
    peak = waveform.abs().max()
    if peak > 1e-6:
        waveform = waveform / peak * 0.5
    return waveform
