"""
viz.py
======

ERP visualisation functions.

Design principles:
    - Functions only return the figure; saving is left to the caller (notebook).
    - This allows the notebook to make adjustments (title, layout) before saving.
"""

from typing import Dict, Optional, List

import matplotlib.pyplot as plt
import mne


def plot_evoked_comparison(
    evokeds: Dict[str, mne.Evoked],
    picks: Optional[List[str]] = None,
    title: Optional[str] = None,
) -> plt.Figure:
    """Plot evoked responses from multiple conditions overlaid.

    Parameters
    ----------
    evokeds : dict
        {condition_name: Evoked} dictionary — the return value of
        compute_condition_evokeds.
    picks : list of str, optional
        Channel names to plot. None plots the mean across all EEG channels.
        For ERP, it is common to focus on a single channel of interest, e.g. ['Cz'].
    title : str, optional
        Figure title.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The rendered figure.
    """
    fig = mne.viz.plot_compare_evokeds(
        evokeds,
        picks=picks,
        title=title,
        show=False,
    )
    # plot_compare_evokeds can return a list; unpack uniformly
    if isinstance(fig, list):
        fig = fig[0]
    return fig


def plot_topomap_timecourse(
    evoked: mne.Evoked,
    times: Optional[List[float]] = None,
    title: Optional[str] = None,
) -> plt.Figure:
    """Plot topographic maps at specified time points arranged in a row.

    Parameters
    ----------
    evoked : mne.Evoked
        Input evoked response.
    times : list of float, optional
        Time points (seconds) at which to plot topomaps.
        Example: [0.05, 0.1, 0.15, 0.2, 0.3]
        None lets MNE select time points automatically (e.g. at peaks).
    title : str, optional
        Figure title.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The rendered figure.

    Notes
    -----
    - Plotting topomaps requires electrode position information (montage).
      For animal EEG, standard montages often do not exist; set a custom
      montage with raw.set_montage() beforehand.
      Without a montage, this will raise an error or produce an empty topomap.
    """
    fig = evoked.plot_topomap(times=times, show=False)
    if title is not None:
        fig.suptitle(title)
    return fig


def plot_peak_to_peak_distribution(
    epochs,
    threshold_uv=None,
    method=None,
    n_sd=3.0,
    max_cols=4,
):
    """Plot per-channel peak-to-peak amplitude histograms.

    Diagnostic plot for assessing threshold rejection. Threshold lines make
    it visually clear what fraction of epochs would be rejected at a given value.

    Parameters
    ----------
    epochs : mne.Epochs
        Input epochs (use before applying rejection for a meaningful diagnostic).
    threshold_uv : float, optional
        Fixed threshold in µV. When specified, a vertical line at this value
        is drawn on all channel plots. Example: 150 draws a line at 150 µV.
    method : str, optional
        'sd' or 'mad'. When specified, a per-channel data-driven threshold line
        is drawn. Can be combined with threshold_uv (both lines are shown).
    n_sd : float
        Threshold multiplier when method is given: threshold = center + n_sd × spread.
    max_cols : int
        Number of columns in the histogram grid.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Histogram figure.

    Notes
    -----
    - X-axis is in µV (1 V = 1e6 µV).
    - Bins exceeding the threshold are shown in a red tone to give an
      intuitive sense of what fraction would be rejected.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from .epoching import compute_peak_to_peak

    ptp, ch_names = compute_peak_to_peak(epochs, picks="eeg")
    ptp_uv = ptp * 1e6  # V → µV
    n_channels = ptp_uv.shape[1]

    # Per-channel thresholds (when method is specified)
    ch_thresholds_uv = None
    if method is not None:
        if method == "sd":
            center = ptp_uv.mean(axis=0)
            spread = ptp_uv.std(axis=0)
        elif method == "mad":
            center = np.median(ptp_uv, axis=0)
            spread = 1.4826 * np.median(np.abs(ptp_uv - center), axis=0)
        else:
            raise ValueError(f"method must be 'sd' or 'mad'. Got: {method!r}")
        ch_thresholds_uv = center + n_sd * spread

    # Subplot layout
    n_cols = min(max_cols, n_channels)
    n_rows = (n_channels + n_cols - 1) // n_cols
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(3.5 * n_cols, 2.5 * n_rows),
        squeeze=False,
    )

    for i, ch in enumerate(ch_names):
        ax = axes[i // n_cols][i % n_cols]
        values = ptp_uv[:, i]

        # Histogram
        ax.hist(values, bins=30, color="steelblue", alpha=0.7)

        # Fixed threshold line
        if threshold_uv is not None:
            ax.axvline(threshold_uv, color="red", linestyle="--", linewidth=1.5,
                       label=f"fixed: {threshold_uv:.0f} µV")
            n_over = (values > threshold_uv).sum()
            ax.text(0.98, 0.95, f"fixed exceeds: {n_over}/{len(values)}",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=8, color="red")

        # Per-channel threshold line
        if ch_thresholds_uv is not None:
            th = ch_thresholds_uv[i]
            ax.axvline(th, color="darkorange", linestyle=":", linewidth=1.5,
                       label=f"{method}: {th:.0f} µV")
            n_over = (values > th).sum()
            ax.text(0.98, 0.80, f"{method} exceeds: {n_over}/{len(values)}",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=8, color="darkorange")

        ax.set_title(ch, fontsize=10)
        ax.set_xlabel("peak-to-peak (µV)", fontsize=8)
        ax.set_ylabel("count", fontsize=8)
        ax.tick_params(labelsize=8)
        if threshold_uv is not None or ch_thresholds_uv is not None:
            ax.legend(fontsize=7, loc="upper right")

    # Hide unused axes
    for j in range(n_channels, n_rows * n_cols):
        axes[j // n_cols][j % n_cols].set_visible(False)

    fig.tight_layout()
    return fig


def plot_evoked_topo_layout(
    epochs_by_condition: Dict[str, mne.Epochs],
    picks: Optional[List[str]] = None,
    exclude: Optional[List[str]] = None,
    ylim: tuple = (-6.0, 6.0),
    title: Optional[str] = None,
    figsize: tuple = (10.0, 8.0),
    ci: float = 0.95,
) -> plt.Figure:
    """Plot condition-level evoked responses overlaid in a topographic layout.

    A small axes is placed at each electrode's head position, showing the
    waveforms for all conditions overlaid with confidence interval bands.

    Parameters
    ----------
    epochs_by_condition : dict
        {condition_name: mne.Epochs} dictionary.
        Epochs (not evokeds) are accepted so that CI bands can be computed.
        Example: {'tone_high': epochs_clean['tone_high'],
                  'tone_low': epochs_clean['tone_low']}
        or split from epochs_clean using keys from epochs_clean.event_id.
    picks : list of str, optional
        Channel names to plot. None plots all EEG channels.
    exclude : list of str, optional
        Channel names to exclude from the topographic layout.
        E.g. ['A1', 'A2', 'VEOG', 'HEOG'] to omit reference or EOG channels.
        None means no exclusions.
    ylim : tuple of (float, float)
        Y-axis range in µV. Default (-6, 6).
    title : str, optional
        Figure title.
    figsize : tuple of (float, float)
        Figure size in inches. Default (10, 8).
    ci : float
        Confidence interval level (0–1). Default 0.95 for 95% CI.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The rendered figure.

    Notes
    -----
    - Plotting a topographic layout requires electrode position information (montage).
      Set it beforehand with raw.set_montage() (set_montage_safely is recommended).
    - The legend includes trial counts (n=...) for each condition.
    - The topo layout internally calls plot_compare_evokeds with axes='topo'.
      The legend axes is placed as the last element.
    """
    import matplotlib as mpl
    import numpy as np

    if not epochs_by_condition:
        raise ValueError("epochs_by_condition is empty.")

    # Channel selection: combine picks and exclude
    first_epochs = next(iter(epochs_by_condition.values()))
    if picks is None:
        ch_names = list(first_epochs.info["ch_names"])
    else:
        ch_names = list(picks)
    if exclude is not None:
        ch_names = [ch for ch in ch_names if ch not in exclude]

    # Include trial count in legend labels.
    # plot_compare_evokeds accepts a dict of lists-of-Evoked.
    # Epochs cannot be passed directly, so convert via iter_evoked().
    labelled = {
        f"{name} (n={len(ep)})": list(ep.iter_evoked())
        for name, ep in epochs_by_condition.items()
    }

    # Compact rcParams for the legend
    compact = {
        "legend.fontsize": "x-small",
        "legend.handlelength": 1.2,
        "legend.handletextpad": 0.3,
        "legend.borderaxespad": 0.2,
    }
    with mpl.rc_context(compact):
        fig = mne.viz.plot_compare_evokeds(
            labelled,
            axes="topo",
            picks=ch_names,
            title=title,
            invert_y=True,
            ylim=dict(eeg=ylim),
            ci=ci,
            show=False,
        )

    # MNE may return a list depending on version
    if isinstance(fig, list):
        fig = fig[0]
    fig.set_size_inches(*figsize)

    # Shrink the legend axes so it does not crowd the waveform panels
    if len(fig.axes) > 0:
        legend_ax = fig.axes[-1]
        scale = 0.60
        x0, y0, w, h = legend_ax.get_position().bounds
        legend_ax.set_position([x0, y0, w * scale, h * scale])
        leg = legend_ax.get_legend()
        if leg is not None:
            for txt in leg.get_texts():
                txt.set_fontsize("xx-small")

    return fig
