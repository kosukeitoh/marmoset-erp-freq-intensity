"""
pipeline.py
===========

Encapsulates the main ERP analysis steps as functions, shared by notebooks
02 and 03 to avoid duplicating the same processing logic.

Functions provided:
- load_and_preprocess: BIDS loading + preprocessing for a single run
- load_preprocess_epoch_multi_run: multi-run support (individual preprocessing
  and epoching per run, then concatenation)
- reject_epochs: epoch rejection only (does not include make_epochs)

Design rationale for multi-run support:
- For each run: load raw → preprocess (filter, re-reference) → epoch individually
- Preprocessing settings (filter_default, ref_channels) are shared across runs
- Epochs are concatenated after epoching with concatenate_epochs
- Rejection is applied once to the concatenated epochs, using the pooled
  distribution from all runs to determine thresholds
"""

from dataclasses import dataclass
from typing import Optional, Dict, Union, List, Any, Tuple

import mne
import pandas as pd

from .config import FilterConfig, EpochConfig
from .io import load_subject_raw, ensure_eeg_channel_type, set_montage_safely
from .preprocess import apply_filter, apply_reference
from .epoching import (
    make_epochs,
    reject_by_threshold,
    reject_by_autoreject,
    reject_by_sd,
)


@dataclass
class RejectionResult:
    """Container for the return value of reject_epochs.

    Attributes
    ----------
    epochs_clean : mne.Epochs
        Epochs after rejection.
    method : str
        Rejection method used ('threshold' / 'sd' / 'autoreject' / 'none').
    reject_log : object or None
        reject_log from autoreject, if that method was used; None otherwise.
    sd_info : dict or None
        Detailed info from SD-based rejection (per-channel thresholds, count, etc.);
        None otherwise.
    """

    epochs_clean: mne.Epochs
    method: str
    reject_log: Optional[Any] = None
    sd_info: Optional[Dict] = None


def load_and_preprocess(
    bids_root,
    subject: str,
    task: str,
    session: Optional[str] = None,
    run: Optional[str] = None,
    filter_default: Optional[FilterConfig] = None,
    ref_channels: Union[str, List[str], None] = None,
    montage=None,
    extension: str = ".vhdr",
) -> mne.io.BaseRaw:
    """Load and preprocess (filter + re-reference) a single run from BIDS.

    Processing order:
        1. read_raw_bids — load from BIDS (bad channels applied from channels.tsv)
        2. ensure_eeg_channel_type — convert misc → eeg (for animal EEG)
        3. set_montage_safely — set montage
        4. apply_filter — filter and resample
        5. apply_reference — re-reference (skipped if ref_channels is None)

    Parameters
    ----------
    bids_root : str or Path
    subject, task : str
    session, run : str, optional
    filter_default : FilterConfig
        Filter and resample settings. None skips filtering.
    ref_channels : str, list, or None
        Electrode(s) to use for re-referencing. None skips re-referencing.
    montage : str, DigMontage, or None
        Montage specification. None skips montage setup.
    extension : str
        BIDS file extension. '.vhdr' for BrainVision.

    Returns
    -------
    raw : mne.io.Raw
        Preprocessed raw data.

    Notes
    -----
    - Bad channels are automatically applied from the 'status' column
      in the BIDS channels.tsv file. To override, set raw.info['bads']
      manually after calling this function.
    """
    raw = load_subject_raw(
        bids_root=bids_root,
        subject=subject,
        task=task,
        session=session,
        run=run,
        extension=extension,
    )
    raw = ensure_eeg_channel_type(raw)
    raw = set_montage_safely(raw, montage, on_missing="warn")

    if filter_default is not None:
        raw = apply_filter(raw, filter_default)

    if ref_channels is not None:
        raw = apply_reference(raw, ref_channels=ref_channels)

    return raw


def load_preprocess_epoch_multi_run(
    bids_root,
    subject: str,
    task: str,
    runs: Optional[List[str]] = None,
    session: Optional[str] = None,
    filter_default: Optional[FilterConfig] = None,
    ref_channels: Union[str, List[str], None] = None,
    montage=None,
    epoch_default: Optional[EpochConfig] = None,
    event_id: Optional[Dict[str, int]] = None,
    extension: str = ".vhdr",
) -> Tuple[List[mne.io.BaseRaw], mne.Epochs]:
    """Load, preprocess, epoch, and concatenate one or more runs.

    Calls load_and_preprocess + make_epochs for each run, then concatenates
    the resulting epochs with mne.concatenate_epochs.
    Preprocessing settings (filter_default, ref_channels) are shared across
    runs, ensuring consistent frequency characteristics.

    No rejection is applied at epoching time (reject=None). Call reject_epochs
    on the returned epochs to apply rejection.

    Parameters
    ----------
    bids_root, subject, task : required.
    runs : list of str, optional
        List of run IDs to combine. Example: ['01', '02', '03']
        None treats data as a single run (no run entity in BIDS).
    session : str, optional
    filter_default, ref_channels, montage : passed to load_and_preprocess.
    epoch_default : EpochConfig
        Epoch settings. tmin/tmax/baseline/decim are used; reject is ignored.
    event_id : dict, optional
        Condition-name → numeric-ID mapping. None for automatic extraction.
    extension : str
        BIDS file extension.

    Returns
    -------
    raws : list of mne.io.Raw
        Preprocessed raw objects, one per run. The last element can be used
        as a reference raw for inspection.
        A single-element list when runs=None.
    epochs : mne.Epochs
        Concatenated epochs from all runs (no rejection applied).

    Notes
    -----
    - Filtering is applied per run (technically unavoidable), but the
      FilterConfig is shared, so frequency characteristics are consistent.
    - Re-referencing is likewise applied per run with a shared config.
    - Concatenation uses mne.concatenate_epochs. Epochs that straddle a run
      boundary are not created (each run is cut at its own boundaries).
    - A 'run' column is added to epochs.metadata so run information can be
      traced downstream.
    """
    if epoch_default is None:
        raise ValueError("epoch_default (EpochConfig) is required.")

    # Treat runs=None or runs=[None] as a single run
    if runs is None or runs == [None]:
        run_list = [None]
    else:
        run_list = list(runs)

    # Copy EpochConfig with reject=None for epoching (rejection is applied later)
    e_cfg_no_reject = EpochConfig(
        tmin=epoch_default.tmin,
        tmax=epoch_default.tmax,
        baseline=epoch_default.baseline,
        reject=None,
        decim=epoch_default.decim,
    )

    raws = []
    epochs_per_run = []
    for r in run_list:
        raw = load_and_preprocess(
            bids_root=bids_root,
            subject=subject,
            task=task,
            session=session,
            run=r,
            filter_default=filter_default,
            ref_channels=ref_channels,
            montage=montage,
            extension=extension,
        )
        raws.append(raw)
        ep = make_epochs(raw, event_id=event_id, cfg=e_cfg_no_reject)
        # Add run information to metadata for downstream traceability
        run_label = r if r is not None else "single"
        ep.metadata = pd.DataFrame({"run": [run_label] * len(ep)})
        epochs_per_run.append(ep)

    # No concatenation needed for a single run
    if len(epochs_per_run) == 1:
        epochs = epochs_per_run[0]
    else:
        epochs = mne.concatenate_epochs(epochs_per_run, add_offset=True)

    return raws, epochs


def reject_epochs(
    epochs: mne.Epochs,
    method: str = "none",
    sd_n: float = 3.0,
    sd_method: str = "mad",
    threshold_reject: Optional[Dict[str, float]] = None,
    autoreject_n_interpolate: Optional[List[int]] = None,
) -> RejectionResult:
    """Apply artifact rejection to epochs (epoching is not included).

    Parameters
    ----------
    epochs : mne.Epochs
        Epochs to reject (before any rejection).
    method : str
        - 'threshold': fixed threshold rejection using threshold_reject.
        - 'sd'       : per-channel SD/MAD-based rejection
        - 'autoreject': autoreject package
        - 'none'     : no rejection (returns a copy as-is)
    sd_n : float
        Threshold multiplier when method='sd'.
    sd_method : str
        Statistic when method='sd' ('sd' or 'mad').
    threshold_reject : dict, optional
        Threshold dictionary when method='threshold' (e.g. {'eeg': 150e-6}).
        Note: None effectively disables rejection.
    autoreject_n_interpolate : list of int, optional
        Interpolation candidate list when method='autoreject'.

    Returns
    -------
    result : RejectionResult
        Contains epochs_clean, method, reject_log, and sd_info.
    """
    reject_log = None
    sd_info = None

    if method == "threshold":
        if threshold_reject is None:
            raise ValueError("threshold_reject is required when method='threshold'.")
        epochs_clean = reject_by_threshold(epochs, reject=threshold_reject)

    elif method == "sd":
        epochs_clean, sd_info = reject_by_sd(
            epochs, n_sd=sd_n, method=sd_method, return_info=True,
        )

    elif method == "autoreject":
        kwargs = {}
        if autoreject_n_interpolate is not None:
            kwargs["n_interpolate"] = autoreject_n_interpolate
        epochs_clean, reject_log = reject_by_autoreject(epochs, **kwargs)

    elif method == "none":
        epochs_clean = epochs.copy()

    else:
        raise ValueError(
            f"Unknown method: {method!r}. "
            f"Must be one of 'threshold', 'sd', 'autoreject', or 'none'."
        )

    return RejectionResult(
        epochs_clean=epochs_clean,
        method=method,
        reject_log=reject_log,
        sd_info=sd_info,
    )
