"""Differentiable effect chain adapted from FxEncoder++.

Source: https://github.com/SonyResearch/Fx-Encoder_PlusPlus
License: CC BY-NC 4.0
See THIRD_PARTY_NOTICES.md in the release repository.
"""

from .constants import (
    ALL_PROCESSORS,
    ALL_PROCESSORS_WITHOUT_GAIN,
    FX_TO_LABEL,
    DEFAULT_FX_PROB,
    EPS,
)

from .fx_aug import (
    select_processors,
    generate_all_processors,
    Random_FX_Chain,
    Random_Single_FX_Chain,
)

from .fx_processors import (
    Distortion,
    Limiter,
    Multiband_Compressor,
    Delay,
    Imager,
    Reverb,
)

from .ddsp_cores import (
    gain,
    distortion,
    parametric_eq,
    compressor,
    noise_shaped_reverberation,
    stereo_widener,
    panning,
    multiband_compressor,
    limiter,
    delay,
)

from .dsp_signal import (
    fft_freqz,
    fft_sosfreqz,
    freqdomain_fir,
    octave_band_filterbank,
    lfilter_via_fsm,
    sosfilt_via_fsm,
    one_pole_butter_lowpass,
    one_pole_filter,
    biquad,
)

__all__ = [
    # constants
    'ALL_PROCESSORS',
    'ALL_PROCESSORS_WITHOUT_GAIN',
    'FX_TO_LABEL',
    'DEFAULT_FX_PROB',
    'EPS',
    # fx_aug
    'select_processors',
    'generate_all_processors',
    'Random_FX_Chain',
    'Random_Single_FX_Chain',
    # fx_processors
    'Distortion',
    'Limiter',
    'Multiband_Compressor',
    'Delay',
    'Imager',
    'Reverb',
    # ddsp_cores
    'gain',
    'distortion',
    'parametric_eq',
    'compressor',
    'noise_shaped_reverberation',
    'stereo_widener',
    'panning',
    'multiband_compressor',
    'limiter',
    'delay',
    # dsp_signal
    'fft_freqz',
    'fft_sosfreqz',
    'freqdomain_fir',
    'octave_band_filterbank',
    'lfilter_via_fsm',
    'sosfilt_via_fsm',
    'one_pole_butter_lowpass',
    'one_pole_filter',
    'biquad',
]
