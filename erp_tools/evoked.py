"""
evoked.py
=========

Computes condition-level evoked responses and grand averages across subjects.

evoked = epochs averaged per condition — the basic unit of an ERP.
grand average = evoked responses averaged further across subjects — a group-level result.
"""

from typing import Dict, List

import mne


def compute_condition_evokeds(
    epochs: mne.Epochs,
) -> Dict[str, mne.Evoked]:
    """Average epochs per condition to produce evoked responses.

    Parameters
    ----------
    epochs : mne.Epochs
        Epochs with condition labels (created by make_epochs).

    Returns
    -------
    evokeds : dict
        {condition_name: Evoked} dictionary.

    Examples
    --------
    >>> evokeds = compute_condition_evokeds(epochs_clean)
    >>> evokeds['tone_high'].plot()
    >>> evokeds['tone_low'].plot()
    """
    # Average across epochs for each condition key in epochs.event_id
    evokeds = {
        condition: epochs[condition].average()
        for condition in epochs.event_id.keys()
    }
    return evokeds


def grand_average(
    subject_evokeds: List[mne.Evoked],
    weight_by_nave: bool = True,
) -> mne.Evoked:
    """Compute a grand average from evoked responses across subjects.

    Parameters
    ----------
    subject_evokeds : list of mne.Evoked
        Evoked responses for the same condition from each subject.
        Example: [sub01 'tone_high', sub02 'tone_high', ...]
    weight_by_nave : bool
        If True, weight each evoked by its trial count (nave) before averaging.
        Recommended when trial counts vary across subjects.
        If False, use a simple unweighted average (each subject contributes equally).

    Returns
    -------
    grand_avg : mne.Evoked
        Grand-averaged evoked response.

    Notes
    -----
    - For statistical testing, pass per-subject evoked arrays directly to
      functions such as mne.stats.permutation_cluster_test rather than using
      the grand average. This function produces a representative waveform for
      visualisation purposes.
    - All subjects must have the same channel configuration. If channel sets
      differ across subjects, align them to a common subset beforehand.
    """
    weights = "nave" if weight_by_nave else "equal"
    grand_avg = mne.combine_evoked(subject_evokeds, weights=weights)
    return grand_avg
