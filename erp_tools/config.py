"""
config.py
=========

Dataclass definitions for carrying analysis parameters.

Why dataclasses:
- When there are more than 10 parameters, passing them as function arguments becomes unwieldy
- In a notebook you can update a single value with `cfg.tmin = -0.1`
- Centralising defaults here makes change history easy to track

Usage (notebook side):
    from erp_tools import FilterConfig, EpochConfig

    filter_default = FilterConfig(l_freq=0.5, h_freq=40.0)
    epoch_default = EpochConfig(tmin=-0.2, tmax=0.5)
"""

from dataclasses import dataclass, field
from typing import Optional, Union


@dataclass
class FilterConfig:
    """Parameters for filtering and resampling.

    Attributes
    ----------
    l_freq : float or None
        High-pass filter cutoff frequency (Hz).
        Typical values for ERP: 0.1–1.0 Hz.
        Set to None to skip high-pass filtering.
    h_freq : float or None
        Low-pass filter cutoff frequency (Hz).
        Typical values for ERP: 30–40 Hz.
        Set to None to skip low-pass filtering.
    notch_freqs : tuple of float, 'auto', or None
        Frequency/frequencies (Hz) for notch filtering.
        - 'auto': reads PowerLineFrequency from the BIDS *_eeg.json and applies
          a notch at that fundamental frequency (recommended; selects 50/60 Hz
          automatically based on recording site)
        - (50.0, 100.0): manual specification; harmonics can be included freely
        - None: skip notch filtering (use when you have confirmed it is unnecessary)
    resample_sfreq : float or None
        Target sampling frequency after downsampling (Hz).
        Example: downsampling from 1000 to 250 Hz speeds up subsequent processing
        and reduces memory usage.
        None means no resampling.
        Note: always resample after filtering to prevent aliasing.
    """

    l_freq: Optional[float] = 0.1
    h_freq: Optional[float] = 40.0
    notch_freqs: Union[tuple, str, None] = "auto"
    resample_sfreq: Optional[float] = None


@dataclass
class EpochConfig:
    """Parameters for epoching and artifact rejection.

    Attributes
    ----------
    tmin : float
        Epoch start time (seconds). Negative values indicate time before stimulus onset (t=0).
        Example: -0.2 means start 200 ms before the stimulus.
    tmax : float
        Epoch end time (seconds).
        Example: 0.5 means end 500 ms after the stimulus.
    baseline : tuple of (float or None, float or None)
        Baseline correction window (seconds).
        (None, 0) means "from epoch start to stimulus onset".
        Pass None to skip baseline correction.
    reject : dict or None
        Rejection threshold based on peak-to-peak amplitude.
        Example: dict(eeg=150e-6) rejects any epoch where any channel exceeds 150 µV.
        None disables threshold rejection (use None when using autoreject).
    decim : int
        Decimation factor applied during epoching.
        Example: recording at 1000 Hz with decim=4 yields epochs at 250 Hz.
        Ensure the remaining sampling frequency is well above h_freq
        (Nyquist criterion: at least 2× h_freq).
    """

    tmin: float = -0.2
    tmax: float = 0.5
    baseline: Optional[tuple] = (None, 0)
    reject: Optional[dict] = field(default_factory=lambda: dict(eeg=150e-6))
    decim: int = 1
