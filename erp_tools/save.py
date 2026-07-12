"""
save.py
=======

Helpers for saving ERP analysis intermediates and final outputs in
BIDS-derivatives format.

File naming convention (follows BIDS-derivatives practice):
    sub-<ID>_task-<TASK>[_run-<R>][_desc-<DESC>]_<SUFFIX>.<EXT>

Examples:
    sub-001_task-auditory_desc-clean_epo.fif       # epochs after rejection
    sub-001_task-auditory_cond-toneHigh_ave.fif    # condition-level evoked
    group_task-auditory_cond-toneHigh_ave.fif      # grand average

Derivatives root example:
    data/<dataset>/derivatives/erp-pipeline/
        ├── dataset_description.json
        ├── sub-001/eeg/...
        ├── sub-002/eeg/...
        └── group/...
"""

from pathlib import Path
from typing import Dict, Optional, Union
import json

import mne


DEFAULT_PIPELINE_NAME = "erp-pipeline"


def get_derivatives_root(
    bids_root: Union[str, Path],
    pipeline_name: str = DEFAULT_PIPELINE_NAME,
) -> Path:
    """Return the path to the pipeline directory under derivatives.

    Parameters
    ----------
    bids_root : str or Path
        Root of the BIDS dataset.
    pipeline_name : str
        Pipeline name under derivatives. Default 'erp-pipeline'.

    Returns
    -------
    deriv_root : Path
        Example: /workspace/data/marmoset_auditory/derivatives/erp-pipeline
    """
    return Path(bids_root) / "derivatives" / pipeline_name


def ensure_dataset_description(
    deriv_root: Union[str, Path],
    pipeline_name: str = DEFAULT_PIPELINE_NAME,
    pipeline_version: str = "0.1.0",
    source_datasets_name: Optional[str] = None,
) -> Path:
    """Create dataset_description.json in the derivatives root if absent.

    BIDS-derivatives requires a dataset_description.json in each derivatives
    directory. Pipeline information is recorded in the GeneratedBy field.

    Parameters
    ----------
    deriv_root : str or Path
        Pipeline directory under derivatives.
    pipeline_name : str
        Pipeline name. Written to GeneratedBy.Name.
    pipeline_version : str
        Pipeline version.
    source_datasets_name : str, optional
        Name of the source dataset (from its dataset_description.json Name field).

    Returns
    -------
    path : Path
        Path to the created (or pre-existing) dataset_description.json.

    Notes
    -----
    Does not overwrite if the file already exists, to preserve any manual edits.
    """
    deriv_root = Path(deriv_root)
    deriv_root.mkdir(parents=True, exist_ok=True)
    desc_path = deriv_root / "dataset_description.json"

    if desc_path.exists():
        return desc_path

    desc = {
        "Name": pipeline_name,
        "BIDSVersion": "1.9.0",
        "DatasetType": "derivative",
        "GeneratedBy": [
            {
                "Name": pipeline_name,
                "Version": pipeline_version,
                "Description": "ERP analysis pipeline (erp_tools)",
            }
        ],
    }
    if source_datasets_name is not None:
        desc["SourceDatasets"] = [{"Name": source_datasets_name}]

    with open(desc_path, "w") as f:
        json.dump(desc, f, indent=4)
    return desc_path


def _build_filename(
    subject: str,
    task: str,
    suffix: str,
    extension: str = ".fif",
    run: Optional[str] = None,
    desc: Optional[str] = None,
    cond: Optional[str] = None,
) -> str:
    """Assemble a BIDS-derivatives-style filename.

    Entity order: sub → task → run → cond → desc → suffix.extension
    (follows the BIDS specification ordering)
    """
    parts = [f"sub-{subject}", f"task-{task}"]
    if run is not None:
        parts.append(f"run-{run}")
    if cond is not None:
        # 'condition' is not an official BIDS entity, but is conventionally placed here
        parts.append(f"cond-{cond}")
    if desc is not None:
        parts.append(f"desc-{desc}")
    base = "_".join(parts)
    return f"{base}_{suffix}{extension}"


def save_epochs(
    epochs: mne.Epochs,
    bids_root: Union[str, Path],
    subject: str,
    task: str,
    desc: str = "clean",
    run: Optional[str] = None,
    pipeline_name: str = DEFAULT_PIPELINE_NAME,
    overwrite: bool = True,
) -> Path:
    """Save epochs to the derivatives directory.

    Example output path:
        derivatives/erp-pipeline/sub-001/eeg/sub-001_task-auditory_desc-clean_epo.fif

    Parameters
    ----------
    epochs : mne.Epochs
        Epochs to save.
    bids_root : str or Path
        Root of the source BIDS dataset.
    subject, task : str
        BIDS entities.
    desc : str
        The desc entity in the filename. E.g. 'clean' or 'filtered'.
    run : str, optional
        Run entity. Use None when aggregating across runs.
    pipeline_name : str
        Pipeline name under derivatives.
    overwrite : bool
        Whether to overwrite an existing file.

    Returns
    -------
    path : Path
        Path to the saved file.
    """
    deriv_root = get_derivatives_root(bids_root, pipeline_name)
    ensure_dataset_description(deriv_root, pipeline_name)

    subject_dir = deriv_root / f"sub-{subject}" / "eeg"
    subject_dir.mkdir(parents=True, exist_ok=True)

    fname = _build_filename(
        subject=subject, task=task, suffix="epo",
        extension=".fif", run=run, desc=desc,
    )
    path = subject_dir / fname
    epochs.save(path, overwrite=overwrite)
    return path


def save_condition_evokeds(
    evokeds: Dict[str, mne.Evoked],
    bids_root: Union[str, Path],
    subject: str,
    task: str,
    run: Optional[str] = None,
    pipeline_name: str = DEFAULT_PIPELINE_NAME,
    overwrite: bool = True,
) -> Dict[str, Path]:
    """Save condition-level evoked responses to the derivatives directory.

    Each condition is written to a separate file. Examples:
        sub-001_task-auditory_cond-toneHigh_ave.fif
        sub-001_task-auditory_cond-toneLow_ave.fif

    Parameters
    ----------
    evokeds : dict
        {condition_name: Evoked} dictionary.
    bids_root, subject, task, run, pipeline_name, overwrite
        Same as save_epochs.

    Returns
    -------
    paths : dict
        {condition_name: Path} for each saved file.
    """
    deriv_root = get_derivatives_root(bids_root, pipeline_name)
    ensure_dataset_description(deriv_root, pipeline_name)

    subject_dir = deriv_root / f"sub-{subject}" / "eeg"
    subject_dir.mkdir(parents=True, exist_ok=True)

    paths = {}
    for cond_name, evoked in evokeds.items():
        # Strip characters that are invalid in filenames
        safe_cond = cond_name.replace("_", "").replace("-", "")
        fname = _build_filename(
            subject=subject, task=task, suffix="ave",
            extension=".fif", run=run, cond=safe_cond,
        )
        path = subject_dir / fname
        evoked.save(path, overwrite=overwrite)
        paths[cond_name] = path
    return paths


def save_grand_average(
    evoked: mne.Evoked,
    bids_root: Union[str, Path],
    task: str,
    cond: str,
    pipeline_name: str = DEFAULT_PIPELINE_NAME,
    overwrite: bool = True,
) -> Path:
    """Save a grand-average evoked response to derivatives/group/.

    Example output path:
        derivatives/erp-pipeline/group/group_task-auditory_cond-toneHigh_ave.fif

    Parameters
    ----------
    evoked : mne.Evoked
        Grand-average evoked response.
    bids_root, task, pipeline_name, overwrite
        Same as above.
    cond : str
        Condition name. Included in the filename.

    Returns
    -------
    path : Path
        Path to the saved file.
    """
    deriv_root = get_derivatives_root(bids_root, pipeline_name)
    ensure_dataset_description(deriv_root, pipeline_name)

    group_dir = deriv_root / "group"
    group_dir.mkdir(parents=True, exist_ok=True)

    safe_cond = cond.replace("_", "").replace("-", "")
    fname = f"group_task-{task}_cond-{safe_cond}_ave.fif"
    path = group_dir / fname
    evoked.save(path, overwrite=overwrite)
    return path


def get_subject_figures_dir(
    bids_root,
    subject: str,
    pipeline_name: str = DEFAULT_PIPELINE_NAME,
) -> Path:
    """Return the figures directory path for a single subject.

    derivatives/erp-pipeline/sub-XXX/eeg/figures/
    """
    deriv_root = get_derivatives_root(bids_root, pipeline_name)
    subject_dir = deriv_root / f"sub-{subject}" / "eeg" / "figures"
    subject_dir.mkdir(parents=True, exist_ok=True)
    return subject_dir


def get_group_figures_dir(
    bids_root,
    pipeline_name: str = DEFAULT_PIPELINE_NAME,
) -> Path:
    """Return the figures directory path for group-level analysis.

    derivatives/erp-pipeline/group/figures/
    """
    deriv_root = get_derivatives_root(bids_root, pipeline_name)
    group_dir = deriv_root / "group" / "figures"
    group_dir.mkdir(parents=True, exist_ok=True)
    return group_dir


def save_subject_figure(
    fig,
    bids_root,
    subject: str,
    task: str,
    desc: str,
    run: Optional[str] = None,
    cond: Optional[str] = None,
    extension: str = ".png",
    dpi: int = 150,
    pipeline_name: str = DEFAULT_PIPELINE_NAME,
) -> Path:
    """Save a per-subject figure to the derivatives directory.

    Example output path:
        derivatives/erp-pipeline/sub-001/eeg/figures/
            sub-001_task-auditory_desc-erp-compare.png

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save.
    bids_root, subject, task : required.
    desc : str
        Describes the figure content. E.g. 'erp-compare', 'topo-layout', 'ptp-distribution'.
    run, cond : str, optional
        Specify if needed. Follow BIDS naming conventions.
    extension : str
        '.png' or '.pdf' etc.
    dpi : int
        Resolution for raster formats.
    pipeline_name : str
        Pipeline name.
    """
    fig_dir = get_subject_figures_dir(bids_root, subject, pipeline_name)
    fname = _build_filename(
        subject=subject, task=task, suffix="figure", extension="",
        run=run, cond=cond, desc=desc,
    )
    # _build_filename appends _figure; remove it for figure files
    fname = fname.replace("_figure", "")
    path = fig_dir / f"{fname}{extension}"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path


def save_group_figure(
    fig,
    bids_root,
    task: str,
    desc: str,
    cond: Optional[str] = None,
    extension: str = ".png",
    dpi: int = 150,
    pipeline_name: str = DEFAULT_PIPELINE_NAME,
) -> Path:
    """Save a group-level figure to the derivatives directory.

    Example output path:
        derivatives/erp-pipeline/group/figures/
            group_task-auditory_desc-erp-compare.png

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    bids_root, task : required.
    desc : str
        Describes the figure content.
    cond : str, optional
        Specify for condition-specific figures.
    extension : str
        '.png' or '.pdf' etc.
    dpi : int
        Resolution for raster formats.
    """
    fig_dir = get_group_figures_dir(bids_root, pipeline_name)
    parts = ["group", f"task-{task}"]
    if cond is not None:
        safe_cond = cond.replace("_", "").replace("-", "")
        parts.append(f"cond-{safe_cond}")
    parts.append(f"desc-{desc}")
    fname = "_".join(parts) + extension
    path = fig_dir / fname
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path


def save_subject_trial_counts(
    counts_series,
    bids_root,
    subject: str,
    task: str,
    pipeline_name: str = DEFAULT_PIPELINE_NAME,
) -> Path:
    """Save per-condition trial counts for one subject as a TSV file under sub-XXX/eeg/.

    Parameters
    ----------
    counts_series : pandas.Series
        index=condition name, values=trial count.
    bids_root, subject, task : required.

    Returns
    -------
    path : Path
        Output path (sub-XXX/eeg/sub-XXX_task-YYY_desc-trial-counts.tsv).
    """
    deriv_root = get_derivatives_root(bids_root, pipeline_name)
    subject_dir = deriv_root / f"sub-{subject}" / "eeg"
    subject_dir.mkdir(parents=True, exist_ok=True)
    fname = f"sub-{subject}_task-{task}_desc-trial-counts.tsv"
    path = subject_dir / fname
    counts_series.to_csv(path, sep="\t", header=["n_trials"])
    return path


def save_trial_counts(
    counts_df,
    bids_root,
    task: str,
    pipeline_name: str = DEFAULT_PIPELINE_NAME,
) -> Path:
    """Save a subject × condition trial count table as a TSV file under group/.

    Used for group-level analysis steps such as grand averaging.
    For the per-subject loop (notebook 03), use save_subject_trial_counts() instead.

    Parameters
    ----------
    counts_df : pandas.DataFrame
        index=subject ID, columns=condition name, values=trial count.
    bids_root, task : required.

    Returns
    -------
    path : Path
        Output path (group/group_task-XXX_desc-trial-counts.tsv).
    """
    fig_dir = get_group_figures_dir(bids_root, pipeline_name)
    group_root = fig_dir.parent
    fname = f"group_task-{task}_desc-trial-counts.tsv"
    path = group_root / fname
    counts_df.to_csv(path, sep="\t")
    return path
