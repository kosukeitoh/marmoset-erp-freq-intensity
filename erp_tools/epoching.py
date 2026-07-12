"""
epoching.py
===========

Handles epoch extraction and rejection of epochs containing artifacts.

Two rejection functions are provided:
    1. reject_by_threshold: fixed peak-to-peak amplitude threshold (standard MNE approach)
    2. reject_by_autoreject: data-driven rejection via the autoreject package

For a new dataset, it is recommended to first run option 1 on one subject
and compare the results with option 2.
"""

from typing import Dict, Tuple, Optional, List

import mne

from .config import EpochConfig


# Labels excluded by default during automatic event ID extraction.
# These are boundary markers and system-level annotations, not analysis targets.
DEFAULT_EXCLUDE_LABELS = (
    "BAD_boundary",
    "EDGE_boundary",
    "boundary",
    "New Segment",
    "DC Correction",
)


def get_event_id_from_raw(
    raw: mne.io.BaseRaw,
    include: Optional[List[str]] = None,
    exclude: Optional[List[str]] = None,
) -> Dict[str, int]:
    """Extract an event_id dictionary from raw.annotations.

    For BIDS-formatted data, the trial_type labels from events.tsv become
    the dictionary keys directly.
    Labels not relevant to analysis (e.g. boundary) are excluded by default.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw object with annotations (loaded via read_raw_bids).
    include : list of str, optional
        Explicit list of labels to include. When provided, all other labels
        are ignored. None uses all labels after applying exclude.
    exclude : list of str, optional
        Labels to exclude. None uses DEFAULT_EXCLUDE_LABELS.
        Pass an empty list [] to disable all exclusion.

    Returns
    -------
    event_id : dict
        {label: numeric_id} dictionary, ready to pass to make_epochs.

    Examples
    --------
    >>> # Automatically retrieve all conditions (boundary etc. are excluded)
    >>> event_id = get_event_id_from_raw(raw)
    >>>
    >>> # Only specific conditions
    >>> event_id = get_event_id_from_raw(raw, include=['tone_high', 'tone_low'])
    >>>
    >>> # Add custom exclusions on top of the defaults
    >>> event_id = get_event_id_from_raw(raw, exclude=list(DEFAULT_EXCLUDE_LABELS) + ['practice'])
    """
    _, event_id_auto = mne.events_from_annotations(raw, verbose=False)

    if include is not None:
        # Explicit mode: keep only labels listed in include
        result = {label: eid for label, eid in event_id_auto.items() if label in include}
        missing = set(include) - set(result)
        if missing:
            raise ValueError(
                f"The following labels were not found in annotations: {missing}\n"
                f"Available labels: {list(event_id_auto.keys())}"
            )
        return result

    # Automatic mode: return everything after applying exclude
    if exclude is None:
        exclude = list(DEFAULT_EXCLUDE_LABELS)
    return {label: eid for label, eid in event_id_auto.items() if label not in exclude}


def make_epochs(
    raw: mne.io.BaseRaw,
    event_id: Optional[Dict[str, int]] = None,
    cfg: EpochConfig = None,
    preload: bool = True,
) -> mne.Epochs:
    """Extract epochs from raw data.

    Event information is obtained automatically from raw.annotations
    (assumes read_raw_bids has already attached events.tsv as annotations).

    Parameters
    ----------
    raw : mne.io.Raw
        Filtered and re-referenced raw data.
    event_id : dict, optional
        Mapping of condition name (trial_type label) to numeric ID.
        Example: {'tone_high': 1, 'tone_low': 2}
        Events whose labels are not in this dict are ignored.
        None triggers automatic retrieval via get_event_id_from_raw(),
        which excludes boundary markers and similar labels.
    cfg : EpochConfig
        Epoch settings. Uses tmin, tmax, baseline, reject, and decim.
    preload : bool
        Load data into memory. Required for downstream autoreject or
        averaging; True by default.

    Returns
    -------
    epochs : mne.Epochs
        Epoched data.

    Notes
    -----
    - Setting cfg.reject to None disables threshold rejection. Use this
      when applying autoreject downstream (double rejection reduces trial count too much).
    - When event_id=None, system labels such as boundary are automatically
      excluded via DEFAULT_EXCLUDE_LABELS.
    """
    if cfg is None:
        raise ValueError("cfg (EpochConfig) is required.")

    # Generate events array from annotations
    events, event_id_auto = mne.events_from_annotations(raw, verbose=False)

    if event_id is None:
        # Automatic mode: delegate to helper (applies default exclusions)
        used_event_id = get_event_id_from_raw(raw)
    else:
        # Explicit mode: extract only the specified labels.
        # Handles the case where the user wants label 'tone_high' even if its
        # numeric ID in annotations differs from the value in event_id.
        # Only checks that the user-specified labels exist in annotations;
        # the actual ID mapping follows event_id_auto (user-side IDs are ignored).
        used_event_id = {
            label: event_id_auto[label]
            for label in event_id
            if label in event_id_auto
        }
        missing = set(event_id) - set(used_event_id)
        if missing:
            raise ValueError(
                f"The following labels were not found in annotations: {missing}\n"
                f"Available labels: {list(event_id_auto.keys())}"
            )

    epochs = mne.Epochs(
        raw,
        events=events,
        event_id=used_event_id,
        tmin=cfg.tmin,
        tmax=cfg.tmax,
        baseline=cfg.baseline,
        reject=cfg.reject,
        decim=cfg.decim,
        preload=preload,
        verbose=False,
    )
    return epochs


def reject_by_threshold(
    epochs: mne.Epochs,
    reject: Optional[Dict[str, float]] = None,
    flat: Optional[Dict[str, float]] = None,
) -> mne.Epochs:
    """Reject epochs by peak-to-peak amplitude threshold (standard MNE approach).

    If cfg.reject was already set during make_epochs(), rejection has already
    been applied. Use this function when you want to re-evaluate with a
    different threshold after epoching.

    Parameters
    ----------
    epochs : mne.Epochs
        Input epochs.
    reject : dict, optional
        Maximum peak-to-peak value per channel type (in volts).
        Example: dict(eeg=150e-6) rejects epochs exceeding 150 µV on any channel.
        None disables this criterion.
    flat : dict, optional
        Epochs with peak-to-peak below this value are rejected as "too flat".
        Example: dict(eeg=1e-6) rejects epochs below 1 µV (detects broken channels).
        None disables this criterion.

    Returns
    -------
    epochs : mne.Epochs
        Epochs after rejection (returns a copy; the original is not modified).

    Notes
    -----
    - Units are volts (V), so 150 µV must be written as 150e-6. This is a
      common source of mistakes.
    - Use epochs_clean.plot_drop_log() to see which epochs were dropped and why.
    """
    epochs_clean = epochs.copy()
    epochs_clean.drop_bad(reject=reject, flat=flat)
    return epochs_clean


def reject_by_autoreject(
    epochs: mne.Epochs,
    n_interpolate: Optional[list] = None,
    consensus: Optional[list] = None,
    random_state: int = 42,
) -> Tuple[mne.Epochs, object]:
    """Reject and interpolate epochs in a data-driven manner using autoreject.

    autoreject implements the algorithm from Jas et al. (2017).
    It estimates per-epoch, per-channel thresholds via cross-validation,
    interpolates channels that can be salvaged, and rejects the rest.

    Parameters
    ----------
    epochs : mne.Epochs
        Input epochs (recommended: created with cfg.reject=None).
    n_interpolate : list of int, optional
        Candidate values for the maximum number of bad channels to interpolate.
        None uses the autoreject default [1, 4, 32].
        For animal EEG with few channels, use smaller values such as [1, 2, 4].
    consensus : list of float, optional
        Candidate fractions (0–1) of bad channels required to reject an epoch.
        None uses the autoreject default.
    random_state : int
        Random seed for cross-validation. Fixed for reproducibility.

    Returns
    -------
    epochs_clean : mne.Epochs
        Epochs after interpolation and rejection.
    reject_log : autoreject.RejectLog
        Log of how each epoch/channel was processed.
        Visualise with reject_log.plot('horizontal').

    Notes
    -----
    - Requires installation: `pip install autoreject`
    - Computation can be slow (minutes to tens of minutes). Start with a
      reduced trial count to test.
    - For animal EEG, compare with manual rejection on the first dataset to
      validate the approach.
    - Interpolation becomes less effective when the channel count is below ~8.
    """
    try:
        from autoreject import AutoReject
    except ImportError as e:
        raise ImportError(
            "autoreject is not installed.\n"
            "Install it with:  pip install autoreject"
        ) from e

    ar_kwargs = {"random_state": random_state, "verbose": False}
    if n_interpolate is not None:
        ar_kwargs["n_interpolate"] = n_interpolate
    if consensus is not None:
        ar_kwargs["consensus"] = consensus

    ar = AutoReject(**ar_kwargs)
    epochs_clean, reject_log = ar.fit_transform(epochs, return_log=True)
    return epochs_clean, reject_log


def compute_peak_to_peak(
    epochs: mne.Epochs,
    picks: str = "eeg",
):
    """Compute peak-to-peak amplitude for each epoch and channel.

    Parameters
    ----------
    epochs : mne.Epochs
        Input epochs.
    picks : str
        Channel type to include. Default 'eeg'.

    Returns
    -------
    ptp : ndarray, shape (n_epochs, n_channels)
        Peak-to-peak amplitude in volts (V).
    ch_names : list of str
        Corresponding channel names.
    """
    import numpy as np

    pick_idx = mne.pick_types(epochs.info, eeg=(picks == "eeg"))
    data = epochs.get_data(picks=pick_idx)  # (n_epochs, n_channels, n_times)
    ptp = data.max(axis=-1) - data.min(axis=-1)  # (n_epochs, n_channels)
    ch_names = [epochs.ch_names[i] for i in pick_idx]
    return ptp, ch_names


def reject_by_sd(
    epochs: mne.Epochs,
    n_sd: float = 3.0,
    method: str = "mad",
    picks: str = "eeg",
    return_info: bool = False,
):
    """Reject epochs using a data-driven, per-channel threshold.

    For each channel, a threshold is computed from the distribution of
    peak-to-peak amplitudes across all epochs. Any epoch that exceeds the
    threshold on at least one channel is excluded across all channels.

    Parameters
    ----------
    epochs : mne.Epochs
        Input epochs (create with reject=None).
    n_sd : float
        Threshold = center + n_sd × spread.
        Typical range: 1.0–3.0. Default 3.0 (lenient).
    method : str
        How to compute center and spread:
        - 'sd': center=mean, spread=SD (straightforward but sensitive to outliers)
        - 'mad': center=median, spread=MAD×1.4826 (robust; recommended)
          MAD = median(|x - median(x)|); scaled by 1.4826 to match SD for a
          normal distribution.
    picks : str
        Channel type to include. Default 'eeg'.
    return_info : bool
        If True, return (epochs_clean, info_dict).
        info contains per-channel thresholds, rejection count, etc.

    Returns
    -------
    epochs_clean : mne.Epochs
        Epochs after rejection.
    info : dict (only when return_info=True)
        - 'thresholds': dict[ch_name, threshold_in_volts]
        - 'n_rejected': int
        - 'n_total': int
        - 'reject_mask': ndarray of bool, shape (n_epochs,) — True means rejected

    Notes
    -----
    - 'mad' (default) is robust to outliers and performs stably on animal EEG
      data that occasionally contains very large-amplitude epochs.
    - If any channel exceeds its threshold, the entire epoch is excluded
      (all-or-nothing; no interpolation). Use autoreject if interpolation is needed.
    - Thresholds are computed over all epochs pooled (not per condition).
    """
    import numpy as np

    ptp, ch_names = compute_peak_to_peak(epochs, picks=picks)
    n_epochs, n_channels = ptp.shape

    # Compute per-channel threshold
    if method == "sd":
        center = ptp.mean(axis=0)
        spread = ptp.std(axis=0)
    elif method == "mad":
        center = np.median(ptp, axis=0)
        # Scale MAD by 1.4826 to make it equivalent to SD for a normal distribution
        spread = 1.4826 * np.median(np.abs(ptp - center), axis=0)
    else:
        raise ValueError(f"method must be 'sd' or 'mad'. Got: {method!r}")

    thresholds = center + n_sd * spread  # shape (n_channels,)

    # Check which epochs exceed the threshold on any channel
    over = ptp > thresholds[np.newaxis, :]  # (n_epochs, n_channels)
    reject_mask = over.any(axis=1)  # (n_epochs,) True means rejected

    # Drop rejected epochs
    epochs_clean = epochs.copy()
    drop_indices = np.where(reject_mask)[0]
    if len(drop_indices) > 0:
        epochs_clean.drop(drop_indices, reason=f"reject_by_sd ({method}, n_sd={n_sd})")

    if return_info:
        info = {
            "thresholds": dict(zip(ch_names, thresholds)),
            "n_rejected": int(reject_mask.sum()),
            "n_total": int(n_epochs),
            "reject_mask": reject_mask,
            "method": method,
            "n_sd": n_sd,
        }
        return epochs_clean, info
    return epochs_clean
