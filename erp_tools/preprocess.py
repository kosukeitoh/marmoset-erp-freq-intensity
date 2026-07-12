"""
preprocess.py
=============

Handles raw-level preprocessing: filtering, re-referencing, and resampling.

Recommended processing order:
    1. Notch (power-line noise removal)
    2. High-pass + low-pass (band-pass)
    3. Re-reference
    4. Resample (if needed)

Rationale: filtering must come before resampling to prevent aliasing.
Re-referencing is conventionally applied after filtering but before epoching
(common practice in BIDS-pipeline workflows).
"""

from typing import Union, List, Optional

import numpy as np
import mne

from .config import FilterConfig


def apply_filter(
    raw: mne.io.BaseRaw,
    cfg: FilterConfig,
    picks: Optional[Union[str, List[str]]] = "eeg",
) -> mne.io.BaseRaw:
    """Apply notch, band-pass filter, and resampling to the full raw signal.

    Parameters
    ----------
    raw : mne.io.Raw
        Input raw object.
    cfg : FilterConfig
        Filter settings. Uses l_freq, h_freq, notch_freqs, resample_sfreq.
    picks : str or list of str
        Channels to filter. Default "eeg" (all EEG channels).
        Restricting to EEG prevents accidentally filtering trigger or EOG channels.

    Returns
    -------
    raw : mne.io.Raw
        Filtered raw object (modified in-place).

    Notes
    -----
    - MNE's default FIR filter is zero-phase, so it does not distort ERP phase.
    - A high-pass cutoff above ~1 Hz attenuates slow components such as P300.
      Choose the cutoff based on the components of interest.
    - When notch_freqs='auto': reads raw.info['line_freq'] (from BIDS
      PowerLineFrequency) and applies a notch at that fundamental frequency only.
      If line_freq is not set, a warning is issued and the notch is skipped.
    """
    # 1. Notch filter (remove 50/60 Hz power-line noise)
    notch_to_apply = _resolve_notch_freqs(cfg.notch_freqs, raw)
    if notch_to_apply is not None:
        raw.notch_filter(freqs=list(notch_to_apply), picks=picks)

    # 2. Band-pass (high-pass + low-pass)
    #    If either l_freq or h_freq is None, only the other side is applied
    if cfg.l_freq is not None or cfg.h_freq is not None:
        raw.filter(l_freq=cfg.l_freq, h_freq=cfg.h_freq, picks=picks)

    # 3. Resample (if requested)
    #    Must come after filtering to prevent aliasing
    if cfg.resample_sfreq is not None:
        raw.resample(sfreq=cfg.resample_sfreq)

    return raw


def _resolve_notch_freqs(notch_freqs, raw: mne.io.BaseRaw):
    """Resolve FilterConfig.notch_freqs to an actual tuple of frequencies.

    Possible inputs:
    - None: skip notch → return None
    - 'auto': read raw.info['line_freq'] and use that fundamental frequency
    - tuple/list: return as-is (manual specification)

    Returns
    -------
    freqs : tuple of float or None
        Frequencies to notch. None means skip.
    """
    import warnings

    if notch_freqs is None:
        return None

    if isinstance(notch_freqs, str):
        if notch_freqs.lower() == "auto":
            line_freq = raw.info.get("line_freq")
            if line_freq is None:
                warnings.warn(
                    "notch_freqs='auto' was specified but raw.info['line_freq'] is not set "
                    "(BIDS PowerLineFrequency may be missing). "
                    "Skipping notch filter. Set line_freq manually or provide an explicit tuple."
                )
                return None
            return (float(line_freq),)
        else:
            raise ValueError(
                f"Unknown string value for notch_freqs: {notch_freqs!r}. "
                f"Use 'auto', None, or a tuple of frequencies."
            )

    # tuple/list → return as tuple
    return tuple(notch_freqs)


def apply_reference(
    raw: mne.io.BaseRaw,
    ref_channels: Union[str, List[str]],
) -> mne.io.BaseRaw:
    """Apply re-referencing to raw data.

    Parameters
    ----------
    raw : mne.io.Raw
        Input raw object.
    ref_channels : str or list of str
        Reference electrode(s).
        - Single-electrode reference: pass a string, e.g. 'Cz'
        - Linked-ears (average of two mastoids): pass a list, e.g. ['A1', 'A2']
        - Average reference: pass 'average'
        - REST reference: pass 'REST' (requires a forward model)

    Returns
    -------
    raw : mne.io.Raw
        Re-referenced raw object (modified in-place).

    Notes
    -----
    - For linked-ears referencing, the reference electrodes (e.g. A1, A2) must
      be present in the BIDS channels.tsv as EEG channels. If the recording
      reference is missing from the file, add a zero-value channel with
      mne.add_reference_channels() before re-referencing.
    - Re-referencing is conventionally applied after filtering and before epoching.
    - Common median reference ('median'): for each time sample, the median across
      all EEG channels is subtracted. More robust to outlier channels than the
      average reference. Not a built-in MNE option; implemented here directly.
    """
    if ref_channels == 'median':
        raw.load_data()
        picks = mne.pick_types(raw.info, eeg=True, meg=False, ref_meg=False)
        data = raw._data[picks]
        raw._data[picks] = data - np.median(data, axis=0, keepdims=True)
        return raw
    raw.set_eeg_reference(ref_channels=ref_channels, projection=False)
    return raw
