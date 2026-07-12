# Marmoset Auditory ERP — Frequency × Intensity Dataset and Analysis Pipeline

EEG dataset and analysis code for auditory evoked potentials (AEPs) recorded from common marmosets (*Callithrix jacchus*) during passive listening to pure tones varying in frequency and intensity.

## Overview

Two adult male marmosets (sub-Cj399, sub-Cj459) were recorded with 8-channel scalp EEG while pure tones of 8 frequencies × 3 intensity levels were presented passively. The dataset is released in [BIDS](https://bids.neuroimaging.io/) format together with a reproducible Python analysis pipeline.

## Dataset

### Subjects

| Subject | Age (years) | Sex | Weight (kg) | Species |
|---------|------------|-----|-------------|---------|
| sub-Cj399 | 13 | M | 0.312 | *Callithrix jacchus* |
| sub-Cj459 | 11 | M | 0.290 | *Callithrix jacchus* |

### Stimuli

Pure tones presented in a passive oddball paradigm (no task required).

- **Frequencies**: 125, 250, 500, 1000, 2000, 4000, 8000, 16000 Hz (8 levels)
- **Intensities**: 45, 60, 75 dB SPL (3 levels)
- **Design**: 3 intensity × 8 frequency = 24 conditions, presented in randomised order
- Each intensity level is a separate BIDS task (`Passive45dB`, `Passive60dB`, `Passive75dB`), with 4 runs per task per subject
- **Trials**: 400 per condition per subject (100 per run × 4 runs); 9,600 per subject; 19,200 total

### EEG acquisition

| Parameter | Value |
|-----------|-------|
| System | Brain Products |
| Channels | 8 EEG (Fz, Cz, Pz, Oz, C3, C4, A1, A2) |
| Placement | International 10-20 system |
| Sampling rate | 5000 Hz |
| Recording reference | Oz |
| Power line | 60 Hz |

### BIDS structure

```
data/marmoset_FreqIntensity/
├── dataset_description.json
├── participants.tsv
├── sub-Cj399/eeg/
│   ├── sub-Cj399_task-Passive45dB_run-{1..4}_eeg.vhdr   # BrainVision format
│   ├── sub-Cj399_task-Passive45dB_run-{1..4}_events.tsv
│   ├── sub-Cj399_task-Passive45dB_run-{1..4}_channels.tsv
│   └── ... (same for Passive60dB, Passive75dB)
├── sub-Cj459/eeg/
│   └── ...
└── derivatives/
    ├── erp-pipeline-ref-linkedEars/   # epochs + evokeds (linked-ears reference)
    ├── erp-pipeline-ref-CAR/          # epochs + evokeds (common average reference)
    └── erp-pipeline-ref-CMR/          # epochs + evokeds (common median reference)
```

Raw data files (`.eeg`, `.vhdr`, `.vmrk`) and derivatives (`.fif`) are excluded from the git repository due to size. The full dataset is available on OSF: **https://osf.io/4xctw/**

Place the BIDS dataset under `data/` before running the notebooks.

## Analysis pipeline

### Requirements

The analysis environment is defined in [.devcontainer/](.devcontainer/) and runs in Docker via the [VS Code Dev Containers](https://code.visualstudio.com/docs/devcontainers/containers) extension. The container uses Python 3.12 with a conda environment (`mne_env`) pre-installed with:

- `mne`, `mne-bids`, `numpy`, `scipy`, `matplotlib`, `scikit-learn` (conda-forge)
- `autoreject`, `ipympl`, `pingouin` (pip)

To start: open the repository in VS Code and select **Reopen in Container** when prompted.

### Repository layout

```
erp_tools/        Python package — reusable ERP processing utilities
notebooks/        Analysis notebooks (run in order)
outputs/          Pipeline outputs: figures, stats tables, component_peaks.json
data/             BIDS dataset root (files excluded from git; place data here)
```

### Notebooks

Run the notebooks in order:

| Notebook | Purpose |
|----------|---------|
| `01_raw-to-aep.ipynb` | Raw BIDS → epochs → evoked responses. Filters, epochs, rejects artefacts (SD method), and saves per-condition evoked `.fif` files to `derivatives/`. Processes three references: linked-ears, CAR, and CMR. Outputs a trial-count summary (non-rejected epochs per subject × condition × reference). |
| `02_aep_viewer.ipynb` | Detects ERP component peaks (P1, N1, P2, N2, LN) from the 75 dB grand-average GFP; saves peak latencies to `outputs/component_peaks.json`. Interactive viewer for GFP and per-channel AEP waveforms with component latency overlays and optional log time axis. |
| `03_gfp_stats.ipynb` | Group-level statistics on GFP amplitude and peak latency. Runs repeated-measures ANOVA and pairwise comparisons across frequency and intensity conditions using the windows defined in `component_peaks.json`. |

### erp_tools package

| Module | Contents |
|--------|----------|
| `config.py` | `FilterConfig`, `EpochConfig` dataclasses |
| `io.py` | BIDS loading, channel type correction, montage setup |
| `preprocess.py` | Notch filter, bandpass, re-reference, resample |
| `epoching.py` | Epoch extraction, threshold / SD / autoreject artefact rejection |
| `evoked.py` | Condition-level averages, grand average |
| `pipeline.py` | High-level wrappers combining the above steps |
| `viz.py` | ERP comparison plots, topographic layouts, peak-to-peak histograms |
| `save.py` | BIDS-derivatives file and figure saving |
| `profiles.py` | Default parameter sets for marmoset and human EEG |
| `erp_viewer.py` | Interactive waveform viewer with drag-to-measure and ANOVA |

### Reproducing results

1. Download the dataset from [OSF (https://osf.io/4xctw/)](https://osf.io/4xctw/) and place it at `data/marmoset_FreqIntensity/`
2. Open and run `notebooks/01_raw-to-aep.ipynb` (processes all subjects × tasks × 3 references)
3. Run `notebooks/02_aep_viewer.ipynb` sections 0–4 to generate waveform plots and `outputs/component_peaks.json`
4. Run `notebooks/03_gfp_stats.ipynb` for statistics

`BIDS_ROOT` at the top of each notebook should point to your local dataset path.

## ERP components

Components detected from the 75 dB grand-average GFP (linked-ears reference):

| Component | Polarity | Approximate latency |
|-----------|----------|-------------------|
| P1 | Positive | ~17–19 ms |
| N1 | Negative | ~39 ms |
| P2 | Positive | ~80–82 ms |
| N2 | Negative | ~125–170 ms |
| LN (late negativity) | Negative | ~290–316 ms |

## Data availability

The BIDS-formatted dataset (raw EEG + derivatives) is openly available on OSF:
**https://osf.io/4xctw/**

## License

This dataset and code are licensed under the [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/) licence. You are free to copy, modify, and redistribute the work for any purpose, provided you give appropriate credit to the original author.

## Citation

If you use this dataset or code, please cite:

> Itoh, K. (2026). *Marmoset auditory ERP: frequency × intensity dataset and analysis pipeline* (v1.0.0). GitHub. https://github.com/kosukeitoh/marmoset-erp-freq-intensity

## Author

Kosuke Itoh
