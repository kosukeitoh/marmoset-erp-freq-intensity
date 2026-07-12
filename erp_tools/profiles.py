"""
profiles.py
===========

Default parameter sets (profiles) for different species and electrode configurations.

Each profile function returns a (FilterConfig, EpochConfig, dict) tuple.
The dict collects information that does not fit in the core config objects
(reference electrodes, recommended montage, species metadata, etc.).

Usage:
    from erp_tools.profiles import marmoset_surface_eeg

    filter_default, epoch_default, meta = marmoset_surface_eeg()
    # Use as-is, or override individual values:
    filter_default.l_freq = 0.5

To add a new profile, append a function following the same pattern at the bottom.
"""

from typing import Tuple, Dict, Any

from .config import FilterConfig, EpochConfig


def marmoset_surface_eeg() -> Tuple[FilterConfig, EpochConfig, Dict[str, Any]]:
    """Default settings for marmoset surface EEG.

    Assumptions:
    - 4–16 screw electrodes positioned relative to bregma
    - Recording under restraint or mild sedation
    - Primary interest is early sensory responses (20–200 ms)

    Rationale for values:
    - l_freq=0.5: animal data often shows pronounced skin-potential drift,
      so a slightly higher high-pass cutoff is used
    - h_freq=100: slightly wider bandwidth than rodents, to preserve the
      sharpness of early components
    - notch_freqs='auto': automatic selection based on BIDS PowerLineFrequency (50/60 Hz)
    - tmin/tmax=(-0.1, 0.4): shorter window than humans, as early components dominate
    - reject=200µV: animal data tends to show larger amplitudes, so the threshold is relaxed
    """
    filter_default = FilterConfig(
        l_freq=0.5,
        h_freq=100.0,
        notch_freqs="auto",  # auto-detected from BIDS PowerLineFrequency; override with e.g. (50.0, 100.0)
        resample_sfreq=None,
    )
    epoch_default = EpochConfig(
        tmin=-0.1,
        tmax=0.4,
        baseline=(None, 0),
        reject=dict(eeg=200e-6),
        decim=1,
    )
    meta = {
        "species": "Callithrix jacchus",
        "ncbi_taxon": "NCBITaxon:9483",
        "ref_channels_suggestion": None,
        # Many marmoset studies keep the recording reference (a frontal or cerebellar screw).
        # Specify in the notebook according to the project.
        "montage": "standard_1020",
        # No standard montage exists; pass a custom DigMontage to set_montage_safely()
        "autoreject_n_interpolate": [1, 2, 4],
        # Keep interpolation candidates small for animal EEG with few electrodes
        "notes": (
            "Electrode layout often follows the 10-20 system, but inter-electrode distances "
            "differ from humans. Pass a custom montage to set_montage_safely() for topographic plots. "
            "autoreject has limited validation on animal EEG; compare with threshold rejection on the first dataset."
        ),
    }
    return filter_default, epoch_default, meta


def human_eeg_32ch() -> Tuple[FilterConfig, EpochConfig, Dict[str, Any]]:
    """Default settings for human 32-channel EEG.

    Assumptions:
    - 32-channel montage based on the international 10-20 system
      (e.g. BioSemi 32, Brain Products 32)
    - Task performed during relaxed wakefulness
    - General ERP analysis including N100/P200/N200/P300 etc.

    Rationale for values:
    - l_freq=0.1: low cutoff to preserve slow components such as P300
    - h_freq=40: conventional low-pass for ERP
    - notch_freqs='auto': automatic selection based on BIDS PowerLineFrequency (50/60 Hz)
    - tmin/tmax=(-0.2, 0.8): window long enough to capture P300
    - reject=150µV: typical threshold for human EEG
    """
    filter_default = FilterConfig(
        l_freq=0.1,
        h_freq=40.0,
        notch_freqs="auto",  # auto-detected from BIDS PowerLineFrequency
        resample_sfreq=None,
    )
    epoch_default = EpochConfig(
        tmin=-0.2,
        tmax=0.8,
        baseline=(None, 0),
        reject=dict(eeg=150e-6),
        decim=1,
    )
    meta = {
        "species": "Homo sapiens",
        "ncbi_taxon": "NCBITaxon:9606",
        "ref_channels_suggestion": "['A1', 'A2']",
        # With 32+ channels, average reference is common. For fewer channels,
        # linked mastoids (['TP9', 'TP10']) or similar may be preferred.
        "montage": "standard_1020",
        # MNE built-in montage name; pass directly to set_montage_safely().
        "autoreject_n_interpolate": [1, 4, 8],
        "notes": (
            "If electrode names follow 10-20 conventions, "
            "set_montage_safely(raw, 'standard_1020') enables topographic plots. "
            "Consider adding ICA if blink artifacts are prominent."
        ),
    }
    return filter_default, epoch_default, meta


# Template for future profiles:
#
# def rodent_surface_eeg() -> Tuple[FilterConfig, EpochConfig, Dict[str, Any]]:
#     """Default settings for rodent surface EEG."""
#     ...
#
# def marmoset_anesthetized() -> Tuple[FilterConfig, EpochConfig, Dict[str, Any]]:
#     """Settings for anaesthetised marmoset (higher high-pass, longer window)."""
#     ...
