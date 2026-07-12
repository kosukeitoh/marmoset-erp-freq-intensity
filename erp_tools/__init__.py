"""
erp_tools: EEG ERP analysis utilities for both animal and human data
====================================================================

A collection of functions for loading BIDS-formatted EEG data, preprocessing,
epoching, condition-level averaging (evoked), and grand averaging.

Design principles:
- Numbers, parameters, and ordering live in the notebook; algorithms live in this package
- Parameters are passed as dataclass Config objects
- Species/electrode default values are provided as preset functions in profiles.py
- Functions process a single subject; subject loops are written in the notebook
- Final outputs are saved in BIDS-derivatives format (save.py); exploratory outputs go to outputs/exploration/
"""

from .config import FilterConfig, EpochConfig
from .io import load_subject_raw, ensure_eeg_channel_type, set_montage_safely
from .preprocess import apply_filter, apply_reference
from .epoching import (
    make_epochs,
    reject_by_threshold,
    reject_by_autoreject,
    reject_by_sd,
    compute_peak_to_peak,
    get_event_id_from_raw,
    DEFAULT_EXCLUDE_LABELS,
)
from .evoked import compute_condition_evokeds, grand_average
from .pipeline import (
    load_and_preprocess,
    load_preprocess_epoch_multi_run,
    reject_epochs,
    RejectionResult,
)
from .viz import (
    plot_evoked_comparison,
    plot_topomap_timecourse,
    plot_peak_to_peak_distribution,
    plot_evoked_topo_layout,
)
from .save import (
    get_derivatives_root,
    ensure_dataset_description,
    save_epochs,
    save_condition_evokeds,
    save_grand_average,
    get_subject_figures_dir,
    get_group_figures_dir,
    save_subject_figure,
    save_group_figure,
    save_trial_counts,
    save_subject_trial_counts,
)

__all__ = [
    "FilterConfig",
    "EpochConfig",
    "load_subject_raw",
    "ensure_eeg_channel_type",
    "set_montage_safely",
    "apply_filter",
    "apply_reference",
    "make_epochs",
    "reject_by_threshold",
    "reject_by_autoreject",
    "reject_by_sd",
    "compute_peak_to_peak",
    "get_event_id_from_raw",
    "DEFAULT_EXCLUDE_LABELS",
    "compute_condition_evokeds",
    "grand_average",
    "load_and_preprocess",
    "load_preprocess_epoch_multi_run",
    "reject_epochs",
    "RejectionResult",
    "plot_evoked_comparison",
    "plot_topomap_timecourse",
    "plot_peak_to_peak_distribution",
    "plot_evoked_topo_layout",
    "get_derivatives_root",
    "ensure_dataset_description",
    "save_epochs",
    "save_condition_evokeds",
    "save_grand_average",
    "get_subject_figures_dir",
    "get_group_figures_dir",
    "save_subject_figure",
    "save_group_figure",
    "save_trial_counts",
    "save_subject_trial_counts",
]
