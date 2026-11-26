"""Turn a table of spectral lines into the settings of a CSI spectral fit.

The interactive fit tools show one table row per line (metabolite): a name, a
start frequency with a search width, a start T2* with bounds and an on/off
switch. :func:`peak_fit_config` converts these rows into the dictionary that
:func:`hypermri.analysis.fitting.spectroscopy.def_fit_params` accepts, so the
notebook plugin and the browser viewer build fits the same way.
:func:`peak_settings_from_detection` and :func:`manual_peak_setting` give the
rows a table starts with.

All functions are pure: they read their arguments, never modify them and do
not touch scan data.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import xarray as xr

from hypermri.analysis.spectroscopy.pipeline import (
    _choose_auto_fit_resolution,
    _get_frequency_axis_hz,
)

__all__ = [
    "FIT_RESULT_SEEDING_KEYS",
    "MIN_PEAK_SEPARATION_HZ",
    "PeakSetting",
    "frequency_axis_hz",
    "frequency_step_hz",
    "manual_peak_setting",
    "peak_fit_config",
    "peak_settings_from_detection",
    "unique_peak_names",
]

# Entries of a peak fit configuration that steer seeding start values from
# earlier fit results. They are not fit options and must be removed before the
# configuration is passed to def_fit_params.
FIT_RESULT_SEEDING_KEYS = (
    "source_metab_names",
    "use_fit_results_init",
    "fit_init_t2_windows_s",
)

#: Default smallest distance in Hz between fitted line frequencies of the
#: interactive fit tools (``min_peak_separation_hz`` of the fit).
MIN_PEAK_SEPARATION_HZ = 5.0

# Smallest T2* in s a row may give; it keeps zero or negative input positive.
_MIN_T2_S = 1e-6


@dataclass(frozen=True)
class PeakSetting:
    """Fit settings of one spectral line, as shown in one table row.

    Parameters
    ----------
    name : str
        Metabolite name used as ``metabolite`` coordinate of the fit. Empty
        and repeated names are made unique by :func:`peak_fit_config`.
    freq_hz : float
        Start frequency in Hz on the frequency axis of the spectra.
    freq_range_hz : float
        Full width in Hz of the frequency search window around
        ``freq_hz``. The sign is ignored; ``0`` fixes the frequency.
    initial_t2_ms : float
        Start T2* in ms. It is moved into ``[min_t2_ms, max_t2_ms]``.
    min_t2_ms, max_t2_ms : float
        Lower and upper T2* bound in ms. Swapped bounds are put in order.
    fixed_freq : bool, default False
        Keep the frequency at ``freq_hz`` (search width 0) while
        ``freq_range_hz`` stays available for later use.
    fixed_t2 : bool, default False
        Keep T2* at its start value (T2* search width 0).
    fit : bool, default True
        Fit this line. ``False`` keeps the line in the model with its start
        values (``skip_fitting``).
    seed_from_fit : bool, default False
        Take per-voxel start values from earlier fit results. The seeding
        itself happens when the fit parameters are built.
    t2_window_ms : float or None, default None
        Full width in ms of a T2* window around each voxel's seeded start
        value. Used only with ``seed_from_fit`` and without ``fixed_t2``;
        ``None`` keeps the bounds.
    """

    name: str
    freq_hz: float
    freq_range_hz: float
    initial_t2_ms: float
    min_t2_ms: float
    max_t2_ms: float
    fixed_freq: bool = False
    fixed_t2: bool = False
    fit: bool = True
    seed_from_fit: bool = False
    t2_window_ms: float | None = None


def unique_peak_names(names: Sequence[str]) -> tuple[list[str], bool]:
    """Return unique, non-empty line names in the given order.

    Leading and trailing spaces are removed, an empty name at position ``i``
    becomes ``metab_{i + 1}`` and a repeated name gets the suffix ``_2``,
    ``_3`` and so on.

    Parameters
    ----------
    names : sequence of str
        Names as typed, one per line.

    Returns
    -------
    names : list of str
        The unique names, as many as given.
    changed : bool
        Whether any name differs from the given one.

    Examples
    --------
    >>> unique_peak_names(["pyr", "", "pyr"])
    (['pyr', 'metab_2', 'pyr_2'], True)
    """
    unique: list[str] = []
    used: set[str] = set()
    changed = False
    for index, raw_name in enumerate(names):
        base = str(raw_name).strip()
        if not base:
            base = f"metab_{index + 1}"
        candidate = base
        suffix = 2
        while candidate in used:
            candidate = f"{base}_{suffix}"
            suffix += 1
        changed = changed or candidate != raw_name
        used.add(candidate)
        unique.append(candidate)
    return unique, changed


def frequency_axis_hz(data: xr.DataArray) -> np.ndarray:
    """Return the spectral frequency axis of CSI data in Hz.

    The axis is the one peak detection uses: the ``freq_axis_hz`` coordinate
    or attribute, or the axis following from ``attrs["dwell_time_s"]`` (s).
    Entry ``k`` belongs to spectral sample ``k`` of the shifted spectrum,
    as the viewers show it.

    Parameters
    ----------
    data : xarray.DataArray
        Samples with the spectral dimension ``acquisition_idx``. Not modified.

    Returns
    -------
    numpy.ndarray
        Float64 frequencies in Hz, one per sample of ``acquisition_idx``.

    Raises
    ------
    ValueError
        If ``data`` has no ``acquisition_idx`` of at least two samples or no
        frequency axis can be derived.
    """
    n_points = data.sizes.get("acquisition_idx", 0)
    if n_points < 2:
        raise ValueError(
            "frequency_axis_hz needs data with at least two samples along "
            f"acquisition_idx; got dimensions {dict(data.sizes)}."
        )
    return _get_frequency_axis_hz(data, n_points)


def frequency_step_hz(data: xr.DataArray) -> float:
    """Return the spacing of the spectral frequency axis of CSI data in Hz.

    Parameters
    ----------
    data : xarray.DataArray
        Samples with the spectral dimension ``acquisition_idx``. Not modified.

    Returns
    -------
    float
        Median absolute spacing of neighbouring frequencies of
        :func:`frequency_axis_hz` in Hz.

    Raises
    ------
    ValueError
        As :func:`frequency_axis_hz`.
    """
    return float(np.median(np.abs(np.diff(frequency_axis_hz(data)))))


def peak_settings_from_detection(
    detected: Mapping[str, Any],
    *,
    min_t2_ms: float,
    max_t2_ms: float,
) -> list[PeakSetting]:
    """Return one table row per line found by automatic peak detection.

    Parameters
    ----------
    detected : mapping
        Result of
        :func:`hypermri.analysis.spectroscopy.pipeline.detect_csi_peaks`. Its
        ``cluster_summary`` gives name and centre frequency (Hz) and the
        parallel lists ``freq_ranges_hz`` and ``initial_t2s_s`` give the search
        width (Hz) and start T2* (s) of each line.
    min_t2_ms, max_t2_ms : float
        T2* bounds in ms given to every row, normally those of the detection
        options.

    Returns
    -------
    list of PeakSetting
        Rows in detection order; all lines are fitted.
    """
    return [
        PeakSetting(
            name=str(summary["name"]),
            freq_hz=float(summary["center_hz"]),
            freq_range_hz=float(freq_range_hz),
            initial_t2_ms=1000.0 * float(initial_t2_s),
            min_t2_ms=float(min_t2_ms),
            max_t2_ms=float(max_t2_ms),
        )
        for summary, freq_range_hz, initial_t2_s in zip(
            detected["cluster_summary"],
            detected["freq_ranges_hz"],
            detected["initial_t2s_s"],
            strict=True,
        )
    ]


def manual_peak_setting(
    freq_hz: float,
    *,
    name: str,
    freq_range_hz: float,
    min_t2_ms: float,
    max_t2_ms: float,
) -> PeakSetting:
    """Return the table row of a line added by hand.

    The start T2* lies in the middle of the bounds.

    Parameters
    ----------
    freq_hz : float
        Start frequency in Hz, usually the spectral cursor.
    name : str
        Name of the new line.
    freq_range_hz : float
        Full frequency search width in Hz.
    min_t2_ms, max_t2_ms : float
        T2* bounds in ms; swapped bounds are put in order.

    Returns
    -------
    PeakSetting
        The new row; the line is fitted.
    """
    low, high = sorted((float(min_t2_ms), float(max_t2_ms)))
    return PeakSetting(
        name=name,
        freq_hz=float(freq_hz),
        freq_range_hz=float(freq_range_hz),
        initial_t2_ms=0.5 * (low + high),
        min_t2_ms=low,
        max_t2_ms=high,
    )


def peak_fit_config(
    peaks: Sequence[PeakSetting],
    *,
    freq_step_hz: float,
    snr_threshold: float,
    min_peak_separation_hz: float,
    nsamplepoints: int,
    rep_fitting: int | None = None,
    npoints: int | None = None,
) -> tuple[dict[str, Any], str]:
    """Return the fit configuration of a table of spectral lines.

    Per line, T2* values are converted to s and kept at least 1 µs, swapped
    bounds are put in order and the start value is moved into the bounds. The
    T2* search width is 0 for a fixed T2*, the seeding window when one is
    used, and otherwise the full bound range. Without explicit values the
    fit resolution (``rep_fitting``, ``npoints``) follows from the widest
    frequency search window in frequency steps.

    Parameters
    ----------
    peaks : sequence of PeakSetting
        Table rows, one per line, in fit order. Must not be empty.
    freq_step_hz : float
        Spacing of the spectral frequency axis in Hz.
    snr_threshold : float
        Amplitude threshold handed to the fit (``snr_threshold``).
    min_peak_separation_hz : float
        Smallest allowed distance in Hz between fitted line frequencies;
        negative values count as 0.
    nsamplepoints : int
        Number of time samples (``acquisition_idx``) the fit uses.
    rep_fitting, npoints : int or None, default None
        Fit repetitions and grid points of the frequency search. ``None``
        takes the automatic choice. Values are raised to at least 1 and 3.

    Returns
    -------
    config : dict
        Fit options for ``def_fit_params`` (``metabs``,
        ``metabs_freqs_hz``, ``metabs_freq_ranges``, ``initial_t2s_s``,
        ``metabs_t2_ranges``, ``min_t2_s``, ``max_t2_s``, ``skip_fitting``,
        ``rep_fitting``, ``npoints``, ``nsamplepoints``,
        ``min_peak_separation_hz``, ``snr_threshold``, ``show_tqdm``) plus
        the seeding entries ``use_fit_results_init`` and
        ``fit_init_t2_windows_s`` (T2* windows in s or ``None``). Remove the
        keys in :data:`FIT_RESULT_SEEDING_KEYS` before calling
        ``def_fit_params``.
    note : str
        Empty, or a remark for the user: names were made unique, or start
        frequencies lie closer than ``min_peak_separation_hz``.

    Raises
    ------
    ValueError
        If ``peaks`` is empty.

    Examples
    --------
    >>> peaks = [
    ...     PeakSetting("pyr", 0.0, 20.0, 30.0, 5.0, 100.0),
    ...     PeakSetting("lac", 390.0, 20.0, 30.0, 5.0, 100.0),
    ... ]
    >>> config, note = peak_fit_config(
    ...     peaks,
    ...     freq_step_hz=10.0,
    ...     snr_threshold=0.0,
    ...     min_peak_separation_hz=5.0,
    ...     nsamplepoints=64,
    ... )
    >>> config["metabs"], config["metabs_t2_ranges"]
    (['pyr', 'lac'], [0.095, 0.095])
    """
    if not peaks:
        raise ValueError(
            "peak_fit_config needs at least one spectral line; detect peaks "
            "or add one by hand."
        )
    names, names_changed = unique_peak_names([peak.name for peak in peaks])
    freqs_hz = [float(peak.freq_hz) for peak in peaks]
    freq_ranges_hz = [
        0.0 if peak.fixed_freq else abs(float(peak.freq_range_hz)) for peak in peaks
    ]
    initial_t2s_s: list[float] = []
    min_t2s_s: list[float] = []
    max_t2s_s: list[float] = []
    t2_ranges_s: list[float] = []
    t2_windows_s: list[float | None] = []
    for peak in peaks:
        min_t2_s = max(float(peak.min_t2_ms) / 1000.0, _MIN_T2_S)
        max_t2_s = max(float(peak.max_t2_ms) / 1000.0, _MIN_T2_S)
        min_t2_s, max_t2_s = min(min_t2_s, max_t2_s), max(min_t2_s, max_t2_s)
        initial_t2_s = max(float(peak.initial_t2_ms) / 1000.0, _MIN_T2_S)
        initial_t2s_s.append(min(max(initial_t2_s, min_t2_s), max_t2_s))
        min_t2s_s.append(min_t2_s)
        max_t2s_s.append(max_t2_s)
        window_s = None
        if peak.seed_from_fit and not peak.fixed_t2 and peak.t2_window_ms is not None:
            window_s = float(peak.t2_window_ms) / 1000.0
        t2_windows_s.append(window_s)
        if peak.fixed_t2:
            t2_ranges_s.append(0.0)
        elif window_s is not None:
            t2_ranges_s.append(window_s)
        else:
            t2_ranges_s.append(max(max_t2_s - min_t2_s, _MIN_T2_S))

    # The resolution heuristic needs a positive width; fixed lines (0 Hz) do
    # not search, so one frequency step stands in for them.
    auto_rep_fitting, auto_npoints = _choose_auto_fit_resolution(
        max_freq_range_hz=max([*freq_ranges_hz, max(float(freq_step_hz), 1e-6)]),
        freq_step_hz=max(1e-9, float(freq_step_hz)),
    )
    rep_fitting = max(1, int(auto_rep_fitting if rep_fitting is None else rep_fitting))
    npoints = max(3, int(auto_npoints if npoints is None else npoints))

    note = ""
    if names_changed:
        note = "Metabolite names were normalized to unique non-empty values. "
    min_peak_separation_hz = max(0.0, float(min_peak_separation_hz))
    if min_peak_separation_hz > 0.0 and len(freqs_hz) > 1:
        min_spacing_hz = float(np.min(np.diff(np.sort(np.asarray(freqs_hz)))))
        if min_spacing_hz < min_peak_separation_hz:
            note += (
                "Selected initial frequencies are closer than the requested "
                f"{min_peak_separation_hz:.3g} Hz fit separation; the fitter "
                "will reject closer candidate frequencies. "
            )

    config = {
        "metabs": names,
        "metabs_freqs_hz": freqs_hz,
        "metabs_freq_ranges": freq_ranges_hz,
        "initial_t2s_s": initial_t2s_s,
        "metabs_t2_ranges": t2_ranges_s,
        "min_t2_s": min_t2s_s,
        "max_t2_s": max_t2s_s,
        "use_fit_results_init": [bool(peak.seed_from_fit) for peak in peaks],
        "fit_init_t2_windows_s": t2_windows_s,
        "skip_fitting": [not peak.fit for peak in peaks],
        "rep_fitting": rep_fitting,
        "npoints": npoints,
        "nsamplepoints": int(nsamplepoints),
        "min_peak_separation_hz": min_peak_separation_hz,
        "snr_threshold": float(snr_threshold),
        "show_tqdm": False,
    }
    return config, note
