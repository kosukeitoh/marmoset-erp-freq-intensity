"""
erp_viewer.py — ERPLAB-style waveform viewer for MNE-Python.

Input: {condition_name: mne.Epochs}  (single subject, one Epochs per condition)

Each condition is drawn as:
  - ERP line  = mean across trials
  - CI band   = ±t₀.₉₇₅ × SEM across trials  (parametric 95% CI by default)

Features
--------
- Layout modes: 'topo', 'grid', 'custom'
- All channel axes share linked x/y sliders
- Drag a time window on any channel to measure mean amp, peak amp, peak latency
  → results written to viewer.measurements (pandas DataFrame)

Usage
-----
    %matplotlib widget
    from erp_tools.erp_viewer import ERPViewer

    epochs = {
        '45dB': mne.read_epochs('sub-XX_task-Passive45dB_desc-clean_epo.fif'),
        '60dB': mne.read_epochs('sub-XX_task-Passive60dB_desc-clean_epo.fif'),
        '75dB': mne.read_epochs('sub-XX_task-Passive75dB_desc-clean_epo.fif'),
    }
    v = ERPViewer(epochs, layout='topo')
    v.show()
    v.measurements.to_csv('measures.csv', index=False)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector
from scipy.stats import t as _t_dist, f_oneway as _f_oneway

import mne
import ipywidgets as _W


def _erp_and_ci(epochs: mne.Epochs, ch_idx: int, scale: float, ci: float):
    """Return (times_ms, mean, lo, hi) for one channel."""
    data = epochs.get_data(picks=[ch_idx])[:, 0, :] * scale  # (n_trials, n_times)
    n = data.shape[0]
    mean = data.mean(axis=0)
    if n < 2:
        return mean, mean, mean
    sem = data.std(axis=0, ddof=1) / np.sqrt(n)
    k = _t_dist.ppf(0.5 + ci / 2, df=n - 1)
    return mean, mean - k * sem, mean + k * sem


class ERPViewer:
    def __init__(self, epochs, layout="topo", ch_type="eeg",
                 nrows=None, ncols=None, custom_pos=None,
                 grid=None,
                 ci=0.95, colors=None, scale=1e6, unit="µV"):
        """
        Parameters
        ----------
        epochs : dict  {condition_name: mne.Epochs}
            One Epochs object per condition, all from the same subject.
        layout : 'topo' | 'grid' | 'custom'
        ch_type : str
            Channel type to display ('eeg', 'mag', …).
        grid : list of list of str, optional
            Explicit 2-D grid for layout='grid'.  Each inner list is one row;
            use '' for an empty cell.  Example::

                grid=[['Fz', '',   'Cz'],
                      ['',  'Pz',  ''  ],
                      ['',  'Oz',  ''  ]]

            When provided, nrows/ncols are ignored.
        nrows, ncols : int, optional
            Grid shape for layout='grid' (ignored when grid= is given).
        custom_pos : dict {ch_name: (row, col)}, optional
            For layout='custom'.
        ci : float
            Confidence interval level, e.g. 0.95 for 95% CI.
        colors : dict {condition: color}, optional
        scale : float
            Multiplier for display (1e6 → V to µV).
        unit : str
            Y-axis label.
        """
        self.epochs     = epochs
        self.conditions = list(epochs.keys())
        self.layout     = layout
        self.ch_type    = ch_type
        self.grid       = grid
        self.nrows      = nrows
        self.ncols      = ncols
        self.custom_pos = custom_pos
        self.ci         = ci
        self.scale      = scale
        self.unit       = unit

        # reference info / channel list from the first condition
        ref_epo = epochs[self.conditions[0]].copy().pick(ch_type)
        self._ref_info  = ref_epo.info
        self.ch_names   = ref_epo.ch_names
        self.times_ms   = ref_epo.times * 1000.0

        if colors is None:
            cyc = plt.rcParams["axes.prop_cycle"].by_key()["color"]
            colors = {c: cyc[i % len(cyc)] for i, c in enumerate(self.conditions)}
        self.colors = colors

        # pre-pick epochs to the right channel type once; ensure data is in memory.
        # load_data() is called per condition lazily — avoid holding all conditions
        # in RAM simultaneously by only picking (no full copy into memory here).
        self._picked = {}
        for c, ep in epochs.items():
            picked = ep.copy().pick(ch_type)
            if not picked.preload:
                picked.load_data()
            self._picked[c] = picked

        self.measurements = pd.DataFrame(
            columns=["condition", "channel", "n_trials",
                     "tmin_ms", "tmax_ms",
                     "mean_amp", "peak_amp", "peak_lat_ms"])

        self.fig       = None
        self.axes_map  = {}
        self._span     = None
        self._log      = _W.Output()

    # ── layout ────────────────────────────────────────────────────────────────

    def _positions(self):
        """Return {ch_name: (x, y)} in 0–1 figure fractions."""
        if self.layout == "topo":
            lay = mne.channels.find_layout(self._ref_info, ch_type=self.ch_type)
            pos = lay.pos[:, :2].astype(float).copy()
            pos -= pos.min(0)
            span = pos.max(0)
            span[span == 0] = 1.0
            pos /= span
            return {n: tuple(p) for n, p in zip(lay.names, pos)
                    if n in self.ch_names}

        if self.layout == "custom":
            if not self.custom_pos:
                raise ValueError("layout='custom' requires custom_pos.")
            rows = [r for r, _ in self.custom_pos.values()]
            cols = [c for _, c in self.custom_pos.values()]
            R, C = max(rows) + 1, max(cols) + 1
            return {n: (c / max(C - 1, 1), 1 - r / max(R - 1, 1))
                    for n, (r, c) in self.custom_pos.items()
                    if n in self.ch_names}

        # 'grid'
        if self.grid is not None:
            # explicit 2-D layout: grid[row][col], '' = empty cell
            # accept a flat list of strings as a single-row grid
            grid = self.grid
            if grid and isinstance(grid[0], str):
                grid = [grid]
            nrows = len(grid)
            ncols = max(len(row) for row in grid)
            out     = {}
            missing = []
            for r, row in enumerate(grid):
                for c, name in enumerate(row):
                    if not name:
                        continue
                    if name not in self.ch_names:
                        missing.append(name)
                    else:
                        out[name] = (c / max(ncols - 1, 1),
                                     1 - r / max(nrows - 1, 1))
            if missing:
                print(f"⚠ grid: channel(s) not found in data and skipped: "
                      f"{missing}\n  available: {self.ch_names}")
            return out

        n     = len(self.ch_names)
        ncols = self.ncols or int(np.ceil(np.sqrt(n)))
        nrows = self.nrows or int(np.ceil(n / ncols))
        out   = {}
        for i, name in enumerate(self.ch_names):
            r, c = divmod(i, ncols)
            out[name] = (c / max(ncols - 1, 1), 1 - r / max(nrows - 1, 1))
        return out

    # ── drawing ───────────────────────────────────────────────────────────────

    def _draw_channel(self, ax, ch):
        i = self.ch_names.index(ch)
        for cond in self.conditions:
            col  = self.colors[cond]
            mean, lo, hi = self._erp_cache[cond][i]
            ax.plot(self.times_ms, mean, lw=0.9, color=col, label=cond)
            ax.fill_between(self.times_ms, lo, hi,
                            color=col, alpha=0.2, lw=0)
        ax.axhline(0, color="0.6", lw=0.4)
        ax.axvline(0, color="0.6", lw=0.4)
        ax.invert_yaxis()   # ERP convention: negative up
        ax.set_title(ch, fontsize=6, pad=1)
        ax.tick_params(labelsize=5)

    def build(self, figsize=None, ax_w=None, ax_h=None, ax_ratio=None):
        # close any previous figure from this viewer to free memory
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None

        # pre-compute all ERPs once; reused by _draw_channel and _autoscale_y
        # _erp_cache[cond][ch_idx] = (mean, lo, hi)
        self._erp_cache = {}
        for cond in self.conditions:
            self._erp_cache[cond] = {}
            ep = self._picked[cond]
            for i in range(len(self.ch_names)):
                self._erp_cache[cond][i] = _erp_and_ci(ep, i, self.scale, self.ci)

        positions = self._positions()
        n = len(positions)

        # infer grid dimensions from normalised positions
        xs = [x for x, y in positions.values()]
        ys = [y for x, y in positions.values()]
        unique_cols = len(set(round(x, 6) for x in xs)) or 1
        unique_rows = len(set(round(y, 6) for y in ys)) or 1

        # usable area: leave left/bottom margin and right margin for legend
        # top margin (1 - h_span - bottom) reserved for suptitle
        left, bottom = 0.06, 0.06
        w_span = 0.76   # leave ~0.18 on right for legend
        h_span = 0.83

        spacing = 0.72

        # base ax sizes derived from grid
        base_ax_w = (w_span / max(unique_cols, 1)) * spacing
        base_ax_h = (h_span / max(unique_rows, 1)) * spacing

        # ax_ratio stretches width relative to height (width = ratio × height in
        # figure-fraction units scaled by the figure aspect).  When ax_ratio is
        # given, expand figsize horizontally so that y-axis tick labels of one
        # column's axes do not reach the waveform area of the adjacent column.
        if ax_ratio is not None:
            # target ax_w in figure-fraction so that ax_w / ax_h ≈ ax_ratio
            # (figure is square-ish by default, so fraction ratio ≈ data ratio)
            target_ax_w = base_ax_h * ax_ratio
            if target_ax_w > base_ax_w:
                # scale figsize width so axes don't actually overlap neighbours;
                # expand figure width by the same factor as ax_w grows
                scale_w = target_ax_w / base_ax_w
                base_ax_w = target_ax_w
                # also widen the figure so the extra width is real pixels
                if figsize is None:
                    side = max(5, min(14, 2.5 * np.sqrt(n)))
                    figsize = (side * scale_w, side)

        if ax_w is None:
            ax_w = base_ax_w
        if ax_h is None:
            ax_h = base_ax_h

        # auto figure size fallback
        if figsize is None:
            side = max(5, min(14, 2.5 * np.sqrt(n)))
            figsize = (side, side)

        self.fig = plt.figure(figsize=figsize)
        self.axes_map.clear()

        for name, (x, y) in positions.items():
            ax = self.fig.add_axes([left + w_span * x * (1 - ax_w),
                                    bottom + h_span * y * (1 - ax_h),
                                    ax_w, ax_h])
            self.axes_map[name] = ax
            self._draw_channel(ax, name)

        h, l = next(iter(self.axes_map.values())).get_legend_handles_labels()
        self.fig.legend(h, l, loc="upper right", fontsize=8,
                        bbox_to_anchor=(1.0, 1.0))
        self.set_xlim(self.times_ms[0], self.times_ms[-1])
        self._autoscale_y()
        return self.fig

    # ── linked ranges ─────────────────────────────────────────────────────────

    def set_xlim(self, lo, hi):
        for ax in self.axes_map.values():
            ax.set_xlim(lo, hi)
        if self.fig:
            self.fig.canvas.draw_idle()

    def set_ylim(self, lo, hi):
        # inverted axis: set (hi, lo) so negative values appear at top
        for ax in self.axes_map.values():
            ax.set_ylim(hi, lo)
        if self.fig:
            self.fig.canvas.draw_idle()

    def _autoscale_y(self):
        m = 0.0
        for i in range(len(self.ch_names)):
            for cond in self.conditions:
                mean, _, _ = self._erp_cache[cond][i]
                m = max(m, np.abs(mean).max())
        self.set_ylim(-m * 1.1, m * 1.1)  # set_ylim inverts to (m, -m) internally

    # ── measurement ───────────────────────────────────────────────────────────

    def _measure(self, tmin_ms, tmax_ms):
        tmask = (self.times_ms >= tmin_ms) & (self.times_ms <= tmax_ms)
        rows       = []
        # trial-level data per channel per condition for ANOVA
        # {ch: {cond: array(n_trials,)}}
        trial_amps = {ch: {} for ch in self.ch_names}

        for cond in self.conditions:
            ep = self._picked[cond]
            n  = len(ep)
            data = ep.get_data() * self.scale   # (n_trials, n_ch, n_times)
            for i, ch in enumerate(self.ch_names):
                mean, _, _ = _erp_and_ci(ep, i, self.scale, self.ci)
                seg = mean[tmask]
                if seg.size == 0:
                    continue
                pk_idx = np.argmax(np.abs(seg))
                rows.append(dict(
                    condition=cond, channel=ch, n_trials=n,
                    tmin_ms=round(tmin_ms, 1), tmax_ms=round(tmax_ms, 1),
                    mean_amp=float(seg.mean()),
                    peak_amp=float(seg[pk_idx]),
                    peak_lat_ms=float(self.times_ms[tmask][pk_idx]),
                ))
                # per-trial window mean for this channel
                trial_amps[ch][cond] = data[:, i, :][:, tmask].mean(axis=1)

        new_df = pd.DataFrame(rows)
        new_df["anova_p"] = np.nan
        for ch in self.ch_names:
            groups = [v for v in trial_amps[ch].values() if len(v) >= 2]
            if len(groups) >= 2:
                _, p = _f_oneway(*groups)
                new_df.loc[new_df["channel"] == ch, "anova_p"] = p

        self.measurements = pd.concat([self.measurements, new_df], ignore_index=True)
        return new_df

    def _log_print(self, msg):
        with self._log:
            print(msg)

    def _on_span(self, lo, hi):
        df = self._measure(lo, hi)
        for ax in self.axes_map.values():
            ax.axvspan(lo, hi, color="0.85", alpha=0.4, zorder=0)
        self.fig.canvas.draw_idle()

        # ANOVA significance label per channel
        def _sig(p):
            if pd.isna(p):   return ""
            if p < 0.001:    return " ***"
            if p < 0.01:     return " **"
            if p < 0.05:     return " *"
            return ""

        p_per_ch = (df.drop_duplicates("channel")
                      .set_index("channel")["anova_p"])

        # pivot: rows=channel, columns=condition, values=mean_amp
        tbl = (df.pivot(index="channel", columns="condition", values="mean_amp")
                 .reindex(self.ch_names)
                 .round(2))
        tbl.columns.name = None
        tbl.index = [f"{ch}{_sig(p_per_ch.get(ch, np.nan))}"
                     for ch in tbl.index]

        from IPython.display import display, HTML
        with self._log:
            display(HTML(
                f"<b>📏 {lo:.0f}–{hi:.0f} ms &nbsp; mean amp (µV)</b>"
                f"&nbsp;&nbsp;<small>* p&lt;.05 &nbsp;** p&lt;.01 &nbsp;*** p&lt;.001 &nbsp;(one-way ANOVA across conditions)</small>"
            ))
            display(tbl)

    def enable_span(self):
        ax0 = next(iter(self.axes_map.values()))
        self._span = SpanSelector(
            ax0, self._on_span, "horizontal", useblit=False,
            props=dict(alpha=0.3, facecolor="tab:blue"))
        self._log_print("ℹ️  Drag over the waveform to measure a time window.")

    # ── Jupyter front-end ─────────────────────────────────────────────────────

    def show(self, xlim=None, ylim=None, ax_ratio=None):
        """Build the figure.

        Parameters
        ----------
        xlim : (float, float), optional
            Time axis range in ms, e.g. (-100, 400).
        ylim : (float, float), optional
            Amplitude axis range in µV, e.g. (-5, 5).
        ax_ratio : float, optional
            Width-to-height ratio for each subplot panel.  Values > 1 stretch
            the time axis horizontally (e.g. ax_ratio=2.5).  Figure width is
            expanded proportionally so neighbouring y-axis labels do not
            overlap the adjacent panel's waveform area.
        """
        self.build(ax_ratio=ax_ratio)
        self.enable_span()
        if xlim is not None:
            self.set_xlim(*xlim)
        if ylim is not None:
            self.set_ylim(*ylim)
        from IPython.display import display
        display(self._log)
        return self.fig
