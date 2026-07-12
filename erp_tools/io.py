"""
io.py
=====

Handles loading of BIDS-formatted EEG data.

Wraps mne-bids `read_raw_bids` and adds adjustments specific to animal EEG
(e.g. fixing channel types).
"""

from pathlib import Path
from typing import Optional, Union

import mne
from mne_bids import BIDSPath, read_raw_bids


def load_subject_raw(
    bids_root: Union[str, Path],
    subject: str,
    task: str,
    session: Optional[str] = None,
    run: Optional[str] = None,
    datatype: str = "eeg",
    suffix: str = "eeg",
    extension: str = ".vhdr",
    verbose: bool = False,
) -> mne.io.BaseRaw:
    """Load raw data for one subject and one run from a BIDS dataset.

    Parameters
    ----------
    bids_root : str or Path
        Root directory of the BIDS dataset.
        Example: "/path/to/my_bids_dataset"
    subject : str
        Subject ID — the part after 'sub-' in BIDS. Example: "001", "rat01"
    task : str
        Task name — the part after 'task-' in BIDS. Example: "auditory", "rest"
    session : str, optional
        Session ID. Specify if 'ses-' is used in the dataset.
    run : str, optional
        Run ID. Specify when multiple runs were recorded. Example: "01"
    datatype : str
        BIDS data type. For EEG, always "eeg".
    suffix : str
        Filename suffix. For EEG, always "eeg".
    extension : str
        File extension. ".vhdr" for BrainVision, ".edf" for EDF.
    verbose : bool
        Whether to print MNE-BIDS detailed log. Default False (silent).

    Returns
    -------
    raw : mne.io.Raw
        Raw object with events, channel types, and line frequency set.

    Notes
    -----
    - read_raw_bids automatically reads events.tsv and attaches it as annotations.
      During epoching, convert to an events array with mne.events_from_annotations().
    - In animal EEG, channel types may be set to "misc". It is recommended to
      call ensure_eeg_channel_type() after loading.
    """
    bids_path = BIDSPath(
        subject=subject,
        session=session,
        task=task,
        run=run,
        datatype=datatype,
        suffix=suffix,
        extension=extension,
        root=str(bids_root),
    )
    raw = read_raw_bids(bids_path, verbose=verbose)
    # Must preload here; downstream filtering raises an exception otherwise
    raw.load_data()
    return raw


def ensure_eeg_channel_type(
    raw: mne.io.BaseRaw,
    eeg_channels: Optional[list] = None,
) -> mne.io.BaseRaw:
    """Convert channels marked as 'misc' to type 'eeg' in animal EEG data.

    Animal electrodes do not match standard montages, so MNE-BIDS sometimes
    reads them as 'misc' to be safe. In this state, set_eeg_reference() and
    plot_topomap() will not work, so an explicit conversion to 'eeg' is needed.

    Parameters
    ----------
    raw : mne.io.Raw
        Input raw object.
    eeg_channels : list of str, optional
        Names of channels to treat as EEG.
        If None, all channels currently typed as 'misc' are converted to 'eeg'.
        Pass an explicit list if EOG, ECG, or trigger channels are mixed in.

    Returns
    -------
    raw : mne.io.Raw
        Raw object with corrected channel types (modified in-place; same object returned).

    Examples
    --------
    >>> # Convert all misc channels to eeg (when no EOG etc. are present)
    >>> raw = ensure_eeg_channel_type(raw)
    >>>
    >>> # Convert only specific channels to EEG, leave others unchanged
    >>> raw = ensure_eeg_channel_type(raw, eeg_channels=['Fz', 'Cz', 'Pz'])
    """
    if eeg_channels is None:
        # Convert all channels currently typed as misc to eeg
        ch_types = raw.get_channel_types()
        eeg_channels = [
            ch for ch, t in zip(raw.ch_names, ch_types) if t == "misc"
        ]
        if not eeg_channels:
            # Nothing to convert; return as-is
            return raw

    mapping = {ch: "eeg" for ch in eeg_channels}
    raw.set_channel_types(mapping)
    return raw


def set_montage_safely(
    raw: mne.io.BaseRaw,
    montage,
    on_missing: str = "warn",
    match_case: bool = False,
) -> mne.io.BaseRaw:
    """Set a montage (electrode position information) on raw data safely.

    A montage is required for topographic maps and some source estimation methods.
    For human EEG, pass an MNE built-in montage name (string). For animal EEG,
    pass a custom mne.channels.DigMontage.

    Parameters
    ----------
    raw : mne.io.Raw
        Input raw object.
    montage : str or mne.channels.DigMontage or None
        - str: MNE built-in montage name (e.g. "standard_1020", "biosemi32")
        - DigMontage: custom montage
        - None: no-op (do nothing)
    on_missing : str
        Behaviour when a channel in raw is not found in the montage.
        'raise' / 'warn' / 'ignore'. Default 'warn'.
        For animal EEG where only some channels have montage positions, 'warn'
        or 'ignore' is appropriate.
    match_case : bool
        Whether to match channel names case-sensitively. Default False.

    Returns
    -------
    raw : mne.io.Raw
        Raw object with montage set (modified in-place).

    Examples
    --------
    >>> # Human 32-channel (standard 10-20)
    >>> raw = set_montage_safely(raw, 'standard_1020')
    >>>
    >>> # Marmoset (custom montage)
    >>> import numpy as np
    >>> from mne.channels import make_dig_montage
    >>> ch_pos = {
    ...     'Fr_L': np.array([-0.005, 0.010, 0.005]),
    ...     'Fr_R': np.array([ 0.005, 0.010, 0.005]),
    ...     # ... coordinates in metres, bregma-referenced
    ... }
    >>> montage = make_dig_montage(ch_pos=ch_pos, coord_frame='head')
    >>> raw = set_montage_safely(raw, montage)
    >>>
    >>> # Do nothing
    >>> raw = set_montage_safely(raw, None)
    """
    if montage is None:
        return raw

    if isinstance(montage, str):
        montage = mne.channels.make_standard_montage(montage)

    raw.set_montage(montage, on_missing=on_missing, match_case=match_case)
    return raw
