"""State and steps of the "fit" tool of the browser view.

The browser fit tool rebuilds the notebook peak selector: it finds spectral
lines automatically or takes lines added at the spectral cursor, lets the
user edit them in a table, and fits all voxels with the chosen lineshape.
Advanced settings refine detection and the fit, couple the T2* of one line
to another and tie lines to one common phase law (the phase band). After a
fit, each line may take its start frequency and, separately, its start T2*
per voxel from an earlier fit (:class:`LineSeeding`). After a
fit, :func:`align_spectra` shifts every voxel spectrum so that its lines lie
at the common weighted mean frequency, as "Align Spectra" of the notebook
fit workflow does.
:class:`FitToolState` holds the table and the settings;
the functions below change it or run the fit. They create no widgets and do
not draw: :class:`~hypermri.visualization.scan_view.ScanView` calls them for
browser messages and shows the result as overlay maps and fitted spectra.

The scientific steps are those of the analysis package:
:func:`~hypermri.analysis.spectroscopy.pipeline.detect_csi_peaks` finds the
lines, :func:`~hypermri.analysis.spectroscopy.peak_fit_setup.peak_fit_config`
turns the table into fit options, and
:func:`~hypermri.analysis.spectroscopy.fitting.fit_spectrum` fits.

Frequencies are in Hz and T2* values in ms, as the table shows them.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, replace
from typing import Any

import numpy as np
import xarray as xr

from hypermri.analysis.anatomical.roi import mask_statistics
from hypermri.analysis.fitting.models.spectroscopy.fid_model import (
    GAUSS_MAX_FWHM_HZ_DEFAULT,
)
from hypermri.analysis.fitting.spectroscopy import (
    def_fit_params,
    mask_fit_dataset_where_input_nonfinite,
)
from hypermri.analysis.spectroscopy.alignment import align_spectra_by_fit_frequency
from hypermri.analysis.spectroscopy.fit_diagnostics import order_fit_problem_voxels
from hypermri.analysis.spectroscopy.fit_seeding import (
    apply_t2_windows,
    seed_start_values,
)
from hypermri.analysis.spectroscopy.fitting import fit_spectrum
from hypermri.analysis.spectroscopy.frequency import add_fit_freqs_ppm
from hypermri.analysis.spectroscopy.peak_fit_setup import (
    FIT_RESULT_SEEDING_KEYS,
    MIN_PEAK_SEPARATION_HZ,
    PeakSetting,
    frequency_step_hz,
    manual_peak_setting,
    peak_fit_config,
    peak_settings_from_detection,
    unique_peak_names,
)
from hypermri.analysis.spectroscopy.phase_timing import phase_band_tau_s_from_scan
from hypermri.analysis.spectroscopy.pipeline import (
    CSIPeakDetectionOptions,
    detect_csi_peaks,
)
from hypermri.analysis.spectroscopy.refitting import (
    SpectroscopyRefitEdit,
    refit_spectroscopy_voxel,
)
from hypermri.analysis.spectroscopy.relevance import add_line_from_residual

__all__ = [
    "FitToolState",
    "LineCoupling",
    "LineSeeding",
    "add_peak",
    "add_residual_line",
    "align_spectra",
    "detect_peaks",
    "edit_peak",
    "fit_tool_settings",
    "lines_from_fit",
    "mask_fit_table",
    "next_problem_voxel",
    "refit_voxel",
    "remove_peak",
    "run_fit",
    "set_fit_option",
    "store_fit_on_scan",
    "stored_fit",
    "voxel_fit_table",
]

# Fit names on a scan, as (results, parameters), in the order they are read:
# the browser and notebook fit workflow first, then the review snapshot, then
# the scan's own fit.
STORED_FIT_NAMES = (
    ("interactive_fit_results", "interactive_fit_params"),
    ("fit_results_review", "fit_params_review"),
    ("fit_results", "fit_params"),
)

_DETECTION_DEFAULTS = CSIPeakDetectionOptions()

#: Lineshapes the tool offers.
LINESHAPES = ("lorentz", "voigt")

#: Start phase tolerance in degrees of a line joining the phase band, as in
#: the notebook peak selector.
PHASE_BAND_TOLERANCE_START_DEG = 10.0

# Table columns the browser may edit, with the type of their values.
_PEAK_FIELDS = {
    "name": str,
    "freq_hz": float,
    "freq_range_hz": float,
    "fixed_freq": bool,
    "initial_t2_ms": float,
    "min_t2_ms": float,
    "max_t2_ms": float,
    "fixed_t2": bool,
    "fit": bool,
}

# Coupling columns of the table, with the type of their values.
_COUPLING_FIELDS = {
    "t2_leader": str,
    "t2_ratio_low": float,
    "t2_ratio_high": float,
    "phase_band": bool,
    "phase_tol_deg": float,
}

# Start-value columns of the table, with the type of their values.
_SEEDING_FIELDS = {
    "freq_from_fit": bool,
    "t2_from_fit": bool,
    "t2_window_ms": float,
}

# Settings the browser may change, with the type of their values.
_OPTIONS = {
    "peak_height_snr": float,
    "prominence_snr": float,
    "b0_shift_tolerance_hz": float,
    "cluster_min_separation_hz": float,
    "max_peaks": int,
    "max_peaks_per_spectrum": int,
    "min_peak_distance_hz": float,
    "snr_threshold_factor": float,
    "freq_search_range_hz": float,
    "min_t2_ms": float,
    "max_t2_ms": float,
    "lineshape": str,
    "gauss_max_fwhm_hz": float,
    "min_peak_separation_hz": float,
    "rep_fitting": int,
    "npoints": int,
    "nsamplepoints": int,
    "phase_band_tau_ms": float,
    "t2_coupling_clip_start": bool,
    "n_jobs": int,
}

# Result variables of one voxel that a refit replaces; others stay.
_REFIT_VARIABLES = (
    "fit_amps",
    "fit_freqs",
    "fit_t2s",
    "fit_gauss_fwhm_hz",
    "fit_gauss_fwhm_status",
    "fit_lorentz_fwhm_hz",
    "fit_fwhm_hz",
    "fit_magnitude_fwhm_hz",
    "fit_freq_status",
    "fit_t2_status",
    "fit_rss",
    "fit_objective",
    "fit_n_free_parameters",
    "fit_residual_degrees_of_freedom",
    "fit_reduced_chi_squared",
)

# Settings that must not be negative; 0 means "off" or "automatic" for some.
_NON_NEGATIVE = (
    "peak_height_snr",
    "prominence_snr",
    "b0_shift_tolerance_hz",
    "cluster_min_separation_hz",
    "max_peaks",
    "min_peak_distance_hz",
    "snr_threshold_factor",
    "freq_search_range_hz",
    "gauss_max_fwhm_hz",
    "min_peak_separation_hz",
    "rep_fitting",
    "npoints",
    "nsamplepoints",
    "phase_band_tau_ms",
)


@dataclass(frozen=True)
class LineCoupling:
    """Constraints that tie one table line to other lines.

    Attributes
    ----------
    t2_leader : str
        Name of the line whose T2* this line follows; empty for a free T2*.
        For a Voigt fit the Lorentz part of T2* is coupled.
    t2_ratio_low, t2_ratio_high : float
        Bounds of the ratio ``T2*(this line) / T2*(leader)``. Equal values
        fix the ratio; different values allow a ratio band. Used only with a
        leader.
    phase_band : bool
        The line joins the common phase law of all lines with this flag: the
        phase of each line follows ``phi0 + 2 pi f tau`` within its
        tolerance, with the common delay ``tau`` of
        :attr:`FitToolState.phase_band_tau_ms`.
    phase_tol_deg : float
        Allowed phase deviation in degrees of this line from the common
        phase law.
    """

    t2_leader: str = ""
    t2_ratio_low: float = 1.0
    t2_ratio_high: float = 1.0
    phase_band: bool = False
    phase_tol_deg: float = PHASE_BAND_TOLERANCE_START_DEG


@dataclass(frozen=True)
class LineSeeding:
    """Start values of one table line taken from an earlier fit.

    With a flag set, :func:`run_fit` replaces the typed start value of that
    line in every voxel by the value the earlier fit found there (see
    :func:`~hypermri.analysis.spectroscopy.fit_seeding.seed_start_values`).
    Voxels without a finite earlier value keep the typed start.

    Attributes
    ----------
    freq_from_fit : bool
        Take the start frequency from the earlier fit. The search range
        stays the typed full width around the new start.
    t2_from_fit : bool
        Take the start T2* from the earlier fit. Without a window, a value
        outside the line's T2* bounds is moved to the nearer bound.
    t2_window_ms : float
        Full width in ms of a T2* window around each voxel's start value
        that replaces the min/max bounds; 0 keeps the bounds. Used only with
        ``t2_from_fit`` and without a fixed T2*.
    """

    freq_from_fit: bool = False
    t2_from_fit: bool = False
    t2_window_ms: float = 0.0


@dataclass
class FitToolState:
    """Peak table and settings of the browser fit tool.

    The object is changed in place by the functions of this module.

    The detection settings have the meaning and defaults of
    :class:`~hypermri.analysis.spectroscopy.pipeline.CSIPeakDetectionOptions`;
    there, 0 stands for "off" where the options use None.

    Attributes
    ----------
    peaks : list of PeakSetting
        Table rows, one per spectral line, in fit order.
    couplings : list of LineCoupling
        T2* coupling and phase band of each row, in the order of ``peaks``.
    seedings : list of LineSeeding
        Start values from an earlier fit of each row, in the order of
        ``peaks``.
    peak_height_snr, prominence_snr : float
        Smallest peak height and prominence for automatic detection, in
        multiples of the spectral-edge noise.
    b0_shift_tolerance_hz : float
        Largest distance in Hz of a peak found in one voxel from the line
        centre it joins, so that B0 shifts between voxels keep one line.
    cluster_min_separation_hz : float
        Weaker lines closer than this distance in Hz to a stronger one are
        dropped; 0 keeps them.
    max_peaks : int
        Largest number of lines detection keeps (the strongest); 0 keeps all.
    max_peaks_per_spectrum : int
        Largest number of peaks taken from one voxel spectrum.
    min_peak_distance_hz : float
        Smallest distance in Hz of two peaks in one voxel spectrum.
    snr_threshold_factor : float
        Amplitude threshold of the fit in multiples of the noise of the FID
        tail, set by detection; 0 gives no threshold.
    freq_search_range_hz : float
        Smallest frequency search width in Hz of a detected line, and the
        width of a line added by hand.
    min_t2_ms, max_t2_ms : float
        T2* bounds in ms of detected lines and of lines added by hand.
    lineshape : {"lorentz", "voigt"}
        Lineshape of the fit.
    gauss_max_fwhm_hz : float
        Largest Gaussian full width at half maximum in Hz of a Voigt fit.
    min_peak_separation_hz : float
        Smallest allowed distance in Hz between fitted line frequencies.
    rep_fitting, npoints : int
        Repetitions and grid points per parameter of the start-value search;
        0 chooses both from the widest frequency search window.
    nsamplepoints : int
        Number of leading FID samples fitted; 0 fits all samples.
    phase_band_tau_ms : float
        Common delay ``tau`` in ms of the phase law of the phase band; 0
        derives it from the timing of the scan when the fit runs.
    t2_coupling_clip_start : bool
        Move T2* start values of coupled lines into their allowed ratio band
        instead of failing when they lie outside.
    n_jobs : int
        Worker processes of the fit; -1 uses all CPU cores.
    snr_threshold : float
        Amplitude threshold handed to the fit, in signal units. Detection
        sets it; it is 0 (no threshold) otherwise.
    n_manual : int
        Number of lines added by hand so far; names the next one.
    message : str
        Last note or error shown below the tool.
    """

    peaks: list[PeakSetting] = field(default_factory=list)
    couplings: list[LineCoupling] = field(default_factory=list)
    seedings: list[LineSeeding] = field(default_factory=list)
    peak_height_snr: float = _DETECTION_DEFAULTS.peak_height_snr
    prominence_snr: float = _DETECTION_DEFAULTS.prominence_snr
    b0_shift_tolerance_hz: float = _DETECTION_DEFAULTS.b0_shift_tolerance_hz
    cluster_min_separation_hz: float = 0.0
    max_peaks: int = 0
    max_peaks_per_spectrum: int = _DETECTION_DEFAULTS.max_peaks_per_spectrum
    min_peak_distance_hz: float = _DETECTION_DEFAULTS.min_peak_distance_hz
    snr_threshold_factor: float = 0.0
    freq_search_range_hz: float = _DETECTION_DEFAULTS.freq_search_range_hz
    min_t2_ms: float = 1000.0 * _DETECTION_DEFAULTS.min_t2_s
    max_t2_ms: float = 1000.0 * _DETECTION_DEFAULTS.max_t2_s
    lineshape: str = "lorentz"
    gauss_max_fwhm_hz: float = GAUSS_MAX_FWHM_HZ_DEFAULT
    min_peak_separation_hz: float = MIN_PEAK_SEPARATION_HZ
    rep_fitting: int = 0
    npoints: int = 0
    nsamplepoints: int = 0
    phase_band_tau_ms: float = 0.0
    t2_coupling_clip_start: bool = False
    n_jobs: int = -1
    snr_threshold: float = 0.0
    n_manual: int = 0
    message: str = ""


def detect_peaks(tool: FitToolState, data: xr.DataArray) -> None:
    """Fill the table with the lines found by automatic peak detection.

    The detection settings not offered by the tool (noise fractions, line
    support) keep the defaults of
    :class:`~hypermri.analysis.spectroscopy.pipeline.CSIPeakDetectionOptions`.
    The table is replaced, also rows added by hand, and all couplings and
    start values from an earlier fit are cleared.

    Parameters
    ----------
    tool : FitToolState
        Tool state; ``peaks``, ``couplings``, ``seedings``,
        ``snr_threshold`` and ``message`` are replaced.
    data : xarray.DataArray
        CSI samples with ``acquisition_idx`` and a frequency axis. Not
        modified.

    Raises
    ------
    ValueError
        If detection fails or finds no line. The table is then unchanged.
    """
    options = CSIPeakDetectionOptions(
        b0_shift_tolerance_hz=float(tool.b0_shift_tolerance_hz),
        peak_height_snr=float(tool.peak_height_snr),
        prominence_snr=float(tool.prominence_snr),
        max_peaks_per_spectrum=max(1, int(tool.max_peaks_per_spectrum)),
        max_peaks=tool.max_peaks if tool.max_peaks > 0 else None,
        cluster_min_separation_hz=float(tool.cluster_min_separation_hz) or None,
        min_peak_distance_hz=float(tool.min_peak_distance_hz),
        freq_search_range_hz=float(tool.freq_search_range_hz),
        min_t2_s=tool.min_t2_ms / 1000.0,
        max_t2_s=tool.max_t2_ms / 1000.0,
        snr_threshold_factor=float(tool.snr_threshold_factor) or None,
    )
    detected = detect_csi_peaks(data, options)
    peaks = peak_settings_from_detection(
        detected, min_t2_ms=tool.min_t2_ms, max_t2_ms=tool.max_t2_ms
    )
    if not peaks:
        raise ValueError(
            "Detect peaks found no line; lower the height SNR or add lines "
            "at the spectral cursor."
        )
    tool.peaks = peaks
    tool.couplings = [LineCoupling() for _ in peaks]
    tool.seedings = [LineSeeding() for _ in peaks]
    tool.snr_threshold = float(detected["snr_threshold"] or 0.0)
    tool.message = f"Found {len(peaks)} line(s)."


def add_peak(tool: FitToolState, freq_hz: float) -> None:
    """Add a line at a frequency, named ``manual_1``, ``manual_2``, ...

    Parameters
    ----------
    tool : FitToolState
        Tool state; the row is appended to ``peaks`` with an empty
        :class:`LineCoupling` and :class:`LineSeeding`.
    freq_hz : float
        Start frequency in Hz, usually the spectral cursor.
    """
    tool.n_manual += 1
    peak = manual_peak_setting(
        freq_hz,
        name=f"manual_{tool.n_manual}",
        freq_range_hz=tool.freq_search_range_hz,
        min_t2_ms=tool.min_t2_ms,
        max_t2_ms=tool.max_t2_ms,
    )
    tool.peaks.append(peak)
    tool.couplings.append(LineCoupling())
    tool.seedings.append(LineSeeding())
    _rename_unique(tool, [peak.name for peak in tool.peaks])
    tool.message = f"Added {tool.peaks[-1].name} at {freq_hz:.1f} Hz."


def edit_peak(tool: FitToolState, index: int, column: str, value: Any) -> None:
    """Change one cell of the table.

    Names are made unique at once (a repeated name gets ``_2``, an empty
    one ``metab_<row>``); lines that follow a renamed line keep following it.

    Parameters
    ----------
    tool : FitToolState
        Tool state; the row is replaced.
    index : int
        Row number, starting at 0.
    column : str
        One of ``name``, ``freq_hz``, ``freq_range_hz``, ``fixed_freq``,
        ``initial_t2_ms``, ``min_t2_ms``, ``max_t2_ms``, ``fixed_t2`` and
        ``fit`` (Hz and ms as in the table), or a :class:`LineCoupling`
        column: ``t2_leader`` (name of another line or empty),
        ``t2_ratio_low``, ``t2_ratio_high``, ``phase_band`` and
        ``phase_tol_deg``, or a :class:`LineSeeding` column:
        ``freq_from_fit``, ``t2_from_fit`` and ``t2_window_ms``.
    value : Any
        New value, converted to the type of the column.

    Raises
    ------
    ValueError
        If the row or column does not exist, the value is not a finite
        number where one is needed, a ratio or tolerance is not positive, or
        the T2* leader is the line itself or no other line of the table, or
        the T2* window is negative.
    """
    _check_row(tool, index)
    if column in _PEAK_FIELDS:
        converted = _converted(_PEAK_FIELDS[column], value, column)
        previous_names = [peak.name for peak in tool.peaks]
        tool.peaks[index] = replace(tool.peaks[index], **{column: converted})
        _rename_unique(tool, previous_names)
    elif column in _COUPLING_FIELDS:
        converted = _converted(_COUPLING_FIELDS[column], value, column)
        _check_coupling(tool, index, column, converted)
        tool.couplings[index] = replace(tool.couplings[index], **{column: converted})
    elif column in _SEEDING_FIELDS:
        converted = _converted(_SEEDING_FIELDS[column], value, column)
        if column == "t2_window_ms" and converted < 0:
            raise ValueError(
                f"Fit tool: t2_window_ms must not be negative; got {converted}."
            )
        tool.seedings[index] = replace(tool.seedings[index], **{column: converted})
    else:
        raise ValueError(
            f"Fit tool: cannot edit column {column!r}; choose one of "
            f"{list(_PEAK_FIELDS) + list(_COUPLING_FIELDS) + list(_SEEDING_FIELDS)}."
        )
    tool.message = ""


def remove_peak(tool: FitToolState, index: int) -> None:
    """Remove one row of the table.

    Lines that followed the T2* of the removed line become free.

    Parameters
    ----------
    tool : FitToolState
        Tool state; the row is removed from ``peaks``, ``couplings`` and
        ``seedings``.
    index : int
        Row number, starting at 0.

    Raises
    ------
    ValueError
        If the row does not exist.
    """
    _check_row(tool, index)
    removed = tool.peaks.pop(index)
    tool.couplings.pop(index)
    tool.seedings.pop(index)
    tool.couplings = [
        replace(coupling, t2_leader="")
        if coupling.t2_leader == removed.name
        else coupling
        for coupling in tool.couplings
    ]
    tool.message = f"Removed {removed.name}."


def set_fit_option(tool: FitToolState, name: str, value: Any) -> None:
    """Change one detection or fit setting.

    Parameters
    ----------
    tool : FitToolState
        Tool state; the attribute ``name`` is replaced.
    name : str
        A setting of :class:`FitToolState` other than ``peaks``,
        ``couplings``, ``seedings``, ``snr_threshold``, ``n_manual`` and ``message``.
    value : Any
        New value, converted to the type of the setting.

    Raises
    ------
    ValueError
        If the setting does not exist, the lineshape is unknown, a number is
        not finite, a count, distance, width or threshold is negative,
        ``max_peaks_per_spectrum`` is below 1 or ``n_jobs`` is 0.
    """
    if name not in _OPTIONS:
        raise ValueError(
            f"Fit tool: unknown setting {name!r}; choose one of {list(_OPTIONS)}."
        )
    converted = _converted(_OPTIONS[name], value, name)
    if name == "lineshape" and converted not in LINESHAPES:
        raise ValueError(
            f"Fit tool: lineshape must be one of {LINESHAPES}; got {converted!r}."
        )
    if name in _NON_NEGATIVE and converted < 0:
        raise ValueError(f"Fit tool: {name} must not be negative; got {converted}.")
    if name == "max_peaks_per_spectrum" and converted < 1:
        raise ValueError("Fit tool: max_peaks_per_spectrum must be at least 1.")
    if name == "n_jobs" and converted == 0:
        raise ValueError("Fit tool: n_jobs must be positive or -1 (all cores).")
    setattr(tool, name, converted)


def run_fit(
    tool: FitToolState,
    data: xr.DataArray,
    scan: Any | None = None,
    earlier_fit: xr.Dataset | None = None,
) -> tuple[xr.Dataset, xr.Dataset]:
    """Fit every voxel with the lines of the table.

    The table becomes fit options with
    :func:`~hypermri.analysis.spectroscopy.peak_fit_setup.peak_fit_config`
    (fit resolution, fitted samples and smallest line distance of the tool),
    the lineshape, the T2* couplings and the phase band;
    :func:`~hypermri.analysis.spectroscopy.fitting.fit_spectrum` then fits
    all spectra. Rows switched off are not fitted. Rows with a
    :class:`LineSeeding` flag first take their start frequency or start T2*
    per voxel from ``earlier_fit``, and a T2* window then sets their T2*
    bounds around these starts. Voxels whose input samples are not all
    finite get NaN in both returned Datasets.

    Parameters
    ----------
    tool : FitToolState
        Tool state; ``message`` is replaced, and ``phase_band_tau_ms`` when
        it is 0 and derived from the scan.
    data : xarray.DataArray
        Complex FIDs with ``acquisition_idx`` and a frequency axis. Not
        modified.
    scan : object, optional
        Scan whose timing gives the phase-band delay when the phase band is
        used and ``phase_band_tau_ms`` is 0, see
        :func:`~hypermri.analysis.spectroscopy.phase_timing.phase_band_tau_s_from_scan`.
    earlier_fit : xarray.Dataset, optional
        Fit results of the same voxels (``fit_freqs`` in Hz, ``fit_t2s`` in
        s) whose values seed the rows with a :class:`LineSeeding` flag. A
        line takes the earlier line of the same name. Without it, or for a
        Voigt fit seeding a Lorentz fit, the typed starts are used and the
        message says so. Not modified.

    Returns
    -------
    fit_params : xarray.Dataset
        Fit options of :func:`~hypermri.analysis.fitting.spectroscopy.def_fit_params`.
    fit_results : xarray.Dataset
        Fit results with ``fit_amps``, ``fit_freqs``, ``fit_t2s`` and, for
        calibrated data, ``fit_freqs_ppm``, over the non-spectral dimensions
        of ``data`` and ``metabolite``.

    Raises
    ------
    ValueError
        If the table is empty, a T2* leader is no line of the table, the
        phase-band delay is needed and cannot be derived, or the fit cannot
        be set up.
    """
    n_samples = int(data.sizes["acquisition_idx"])
    config, note = peak_fit_config(
        tool.peaks,
        freq_step_hz=frequency_step_hz(data),
        snr_threshold=tool.snr_threshold,
        min_peak_separation_hz=float(tool.min_peak_separation_hz),
        nsamplepoints=min(tool.nsamplepoints, n_samples) or n_samples,
        rep_fitting=tool.rep_fitting or None,
        npoints=tool.npoints or None,
    )
    if tool.lineshape == "voigt":
        config.update(
            lineshape="voigt", gauss_max_fwhm_hz=float(tool.gauss_max_fwhm_hz)
        )
    config.update(_coupling_options(tool, config["metabs"], scan))
    for key in FIT_RESULT_SEEDING_KEYS:
        config.pop(key, None)
    fit_params = def_fit_params(data=data, fit_params=config)
    fit_params, seed_note = _seeded_params(tool, fit_params, earlier_fit)
    fit_results = fit_spectrum(data, fit_params, n_jobs=tool.n_jobs)
    tool.message = " ".join(
        part.strip() for part in ("Fit done.", seed_note, note) if part
    )
    return (
        mask_fit_dataset_where_input_nonfinite(fit_params, data),
        mask_fit_dataset_where_input_nonfinite(fit_results, data),
    )


def align_spectra(
    tool: FitToolState,
    data: xr.DataArray,
    fit_params: xr.Dataset,
    fit_results: xr.Dataset,
    scan: Any | None = None,
) -> tuple[xr.DataArray, xr.Dataset, xr.Dataset]:
    """Shift every voxel spectrum to the common weighted mean frequency.

    Each voxel is shifted by the difference between its amplitude-weighted
    fitted frequency and the mean of these frequencies over all voxels, with
    :func:`~hypermri.analysis.spectroscopy.alignment.align_spectra_by_fit_frequency`.
    B0 differences between voxels then no longer move the lines, and a new
    fit can use narrower frequency windows. The fit is shifted with the data;
    fitting the aligned spectra again gives the fit of the aligned data.

    Parameters
    ----------
    tool : FitToolState
        Tool state; ``message`` is replaced.
    data : xarray.DataArray
        Complex FIDs (or spectra with ``attrs["signal_domain"] ==
        "spectral"``) with ``acquisition_idx`` that were fitted. Not
        modified.
    fit_params, fit_results : xarray.Dataset
        Fit options and results of :func:`run_fit` for ``data``. Not
        modified.
    scan : object, optional
        Scan whose ``sampling_time_axis_s`` or ``dwell_time_s`` (s) is used
        when ``data`` has no such attribute.

    Returns
    -------
    aligned_data : xarray.DataArray
        Shifted copy of ``data`` with the same dimensions, coordinates and
        attributes, and the coordinate ``frequency_alignment_offset_hz``
        (Hz, voxel dimensions) of the total shift applied so far.
    aligned_params : xarray.Dataset
        ``fit_params`` with start frequencies and bounds shifted per voxel.
    aligned_results : xarray.Dataset
        ``fit_results`` with ``fit_freqs`` shifted per voxel.

    Raises
    ------
    ValueError
        If no voxel has a valid fitted frequency and amplitude or the timing
        of the samples is unknown.
    """

    def scan_timing(name: str) -> Any:
        """Timing of the scan, used only when the data do not carry it."""
        if data.attrs.get(name) is not None or scan is None:
            return None
        return getattr(scan, name, None)

    aligned_data, aligned_results, aligned_params, info = (
        align_spectra_by_fit_frequency(
            data,
            fit_results,
            fit_params,
            sampling_time_axis_s=scan_timing("sampling_time_axis_s"),
            dwell_time_s=scan_timing("dwell_time_s"),
        )
    )
    tool.message = (
        "Spectra aligned to "
        f"{float(info['global_mean_frequency_hz']):.1f} Hz "
        f"({int(info['n_valid_voxels'])} voxels). Run the fit again on the "
        "aligned spectra."
    )
    return aligned_data, aligned_params, aligned_results


def store_fit_on_scan(
    scan: Any, fit_params: xr.Dataset, fit_results: xr.Dataset, data: xr.DataArray
) -> None:
    """Store a fit on the scan under the names of the notebook fit workflow.

    The scan then holds ``fit_params_review`` and ``interactive_fit_params``
    (fit options), ``fit_results_review`` and ``interactive_fit_results``
    (results) and ``data_review`` (a copy of the fitted samples).

    Parameters
    ----------
    scan : object
        Scan object; its attributes are set.
    fit_params, fit_results : xarray.Dataset
        Fit options and results of :func:`run_fit`. Stored, not copied.
    data : xarray.DataArray
        Fitted samples. Not modified; a deep copy is stored.
    """
    scan.fit_params_review = fit_params
    scan.interactive_fit_params = fit_params
    scan.fit_results_review = fit_results
    scan.interactive_fit_results = fit_results
    scan.data_review = data.copy(deep=True)


def stored_fit(scan: Any | None) -> tuple[str, xr.Dataset, xr.Dataset | None] | None:
    """Return the spectral fit stored on a scan, if there is one.

    The names of :data:`STORED_FIT_NAMES` are read in order; the first Dataset with ``fit_freqs``, ``fit_t2s`` and a ``metabolite``
    coordinate is returned. Fits of other kinds (for example the dictionary
    of map fits of an anatomy scan) are skipped.

    Parameters
    ----------
    scan : object or None
        Scan object. Not modified; nothing is loaded or fitted.

    Returns
    -------
    tuple or None
        ``(name, fit_results, fit_params)`` with the attribute name of the
        results and the parameters stored next to them (None when there
        are none), or None without a spectral fit.
    """
    if scan is None:
        return None
    for results_name, params_name in STORED_FIT_NAMES:
        results = getattr(scan, results_name, None)
        if (
            isinstance(results, xr.Dataset)
            and "fit_freqs" in results
            and "fit_t2s" in results
            and "metabolite" in results.coords
        ):
            params = getattr(scan, params_name, None)
            return (
                results_name,
                results,
                params if isinstance(params, xr.Dataset) else None,
            )
    return None


def lines_from_fit(tool: FitToolState, fit_results: xr.Dataset) -> None:
    """Fill the table with the lines of an earlier fit.

    Each fitted metabolite becomes one row with its name, the median over
    all voxels of the fitted frequency and T2* as typed start values, the
    tool's frequency search width and T2* bounds, and both start values
    taken per voxel from the fit (:class:`LineSeeding`). The table is
    replaced and all couplings are cleared.

    Parameters
    ----------
    tool : FitToolState
        Tool state; ``peaks``, ``couplings``, ``seedings``, ``message`` and
        ``snr_threshold`` (set to 0, so that no voxel is skipped) are
        replaced.
    fit_results : xarray.Dataset
        Earlier fit with ``fit_freqs`` (Hz) and ``fit_t2s`` (s) over
        ``metabolite`` and the voxel dimensions. Not modified.

    Raises
    ------
    ValueError
        If the fit has no ``fit_freqs``, ``fit_t2s`` or ``metabolite``, or
        no line has a finite fitted frequency. The table is then unchanged.
    """
    if (
        "fit_freqs" not in fit_results
        or "fit_t2s" not in fit_results
        or "metabolite" not in fit_results.coords
    ):
        raise ValueError(
            "Lines from fit: the fit has no fit_freqs/fit_t2s per metabolite."
        )
    low, high = sorted((float(tool.min_t2_ms), float(tool.max_t2_ms)))
    peaks = []
    for name in fit_results.coords["metabolite"].values:
        freqs = np.asarray(fit_results["fit_freqs"].sel(metabolite=name), dtype=float)
        t2s = np.asarray(fit_results["fit_t2s"].sel(metabolite=name), dtype=float)
        freqs = freqs[np.isfinite(freqs)]
        t2s = t2s[np.isfinite(t2s) & (t2s > 0)]
        if not freqs.size:
            continue
        t2_ms = 1000.0 * float(np.median(t2s)) if t2s.size else 0.5 * (low + high)
        peaks.append(
            PeakSetting(
                name=str(name),
                freq_hz=float(np.median(freqs)),
                freq_range_hz=float(tool.freq_search_range_hz),
                initial_t2_ms=min(max(t2_ms, low), high),
                min_t2_ms=low,
                max_t2_ms=high,
            )
        )
    if not peaks:
        raise ValueError(
            "Lines from fit: no line of the fit has a finite fitted frequency."
        )
    tool.peaks = peaks
    tool.couplings = [LineCoupling() for _ in peaks]
    tool.seedings = [LineSeeding(freq_from_fit=True, t2_from_fit=True) for _ in peaks]
    tool.snr_threshold = 0.0
    tool.message = (
        f"Took {len(peaks)} line(s) from the earlier fit; frequency and T2* "
        "start per voxel from it."
    )


def refit_voxel(
    tool: FitToolState,
    data: xr.DataArray,
    fit_params: xr.Dataset,
    fit_results: xr.Dataset,
    indices: dict[str, int],
    scan: Any | None = None,
) -> xr.Dataset:
    """Fit one voxel again with the start values and bounds of the table.

    Each table row whose name is a fitted line becomes a
    :class:`~hypermri.analysis.spectroscopy.refitting.SpectroscopyRefitEdit`
    (start frequency and search width in Hz, start T2* and bounds in ms
    converted to s; a row switched off keeps its earlier values);
    :func:`~hypermri.analysis.spectroscopy.refitting.refit_spectroscopy_voxel`
    then fits the voxel. Fitted lines without a row are fitted from their
    earlier values. Only the voxel's values change; ``fit_freqs_ppm`` is
    recomputed when present.

    Parameters
    ----------
    tool : FitToolState
        Tool state; ``message`` is replaced.
    data : xarray.DataArray
        Complex FIDs that were fitted, with ``acquisition_idx``. Not modified.
    fit_params, fit_results : xarray.Dataset
        Parameters and results of the fit of ``data``. Not modified.
    indices : dict of str to int
        Positional index of the voxel along each non-spectral dimension of
        ``data`` (the view's slice indices); dimensions of size one may be
        missing.
    scan : object, optional
        Scan whose timing gives the sampling times when ``data`` has none.

    Returns
    -------
    xarray.Dataset
        A copy of ``fit_results`` with the voxel refitted.

    Raises
    ------
    ValueError
        If a table row that is fitted names no line of the fit (fit all
        voxels after adding or renaming lines), no line is switched on, or
        the refit cannot be set up.
    """
    fitted = [str(name) for name in fit_results.coords["metabolite"].values]
    unknown = [peak.name for peak in tool.peaks if peak.fit and peak.name not in fitted]
    if unknown:
        raise ValueError(
            f"Refit voxel: line(s) {unknown} are not in the fit; fit all voxels "
            "after adding or renaming lines."
        )
    edits = {
        peak.name: SpectroscopyRefitEdit(
            initial_freq_hz=float(peak.freq_hz),
            freq_search_range_hz=float(peak.freq_range_hz),
            initial_t2_s=peak.initial_t2_ms / 1000.0,
            min_t2_s=peak.min_t2_ms / 1000.0,
            max_t2_s=peak.max_t2_ms / 1000.0,
            enabled=bool(peak.fit),
        )
        for peak in tool.peaks
        if peak.name in fitted
    }
    selected = _selected_voxel(data, indices)
    voxel = _fit_voxel(fit_results, selected)
    refitted = refit_spectroscopy_voxel(
        data,
        fit_params,
        fit_results,
        voxel_indices=voxel,
        data_indices=selected,
        metabolites=fitted,
        edits=edits,
        scan=scan,
    )
    if refitted["status"] == "no_enabled_metabolites":
        raise ValueError("Refit voxel: switch on at least one line of the table.")
    updated = fit_results.copy(deep=True)
    _write_voxel(updated, refitted["fit_results"], voxel)
    if "fit_freqs_ppm" in updated:
        updated = add_fit_freqs_ppm(updated)
    chi2 = _voxel_value(updated, "fit_reduced_chi_squared", voxel)
    tool.message = f"Refitted voxel {voxel}" + (
        f": chi2/N {chi2:.2f}." if math.isfinite(chi2) else "."
    )
    return updated


def next_problem_voxel(
    tool: FitToolState, fit_results: xr.Dataset, indices: dict[str, int]
) -> dict[str, int]:
    """Return the voxel after the selected one in the order of fit problems.

    The order is that of
    :func:`~hypermri.analysis.spectroscopy.fit_diagnostics.order_fit_problem_voxels`:
    voxels with a line at a bound first, then the highest reduced
    chi-squared. From a voxel not in the list, the first entry is returned;
    after the last, the list starts again.

    Parameters
    ----------
    tool : FitToolState
        Tool state; ``message`` names the voxel's place in the list, its
        reduced chi-squared and bound flags.
    fit_results : xarray.Dataset
        Fit results with ``fit_rss`` and ``fit_freqs``. Not modified.
    indices : dict of str to int
        Positional index of the selected voxel per dimension.

    Returns
    -------
    dict of str to int
        Positional index of the next voxel along each spectrum dimension of
        the fit.

    Raises
    ------
    ValueError
        If no voxel is fitted.
    """
    order = order_fit_problem_voxels(fit_results)
    count = order.sizes["problem_voxel"]
    if count == 0:
        raise ValueError("Next problem voxel: no voxel is fitted.")
    dims = order.attrs["spectrum_dims"]
    positions = [{d: int(order[d][k]) for d in dims} for k in range(count)]
    current = {d: int(indices.get(d, 0)) for d in dims}
    following = (positions.index(current) + 1 if current in positions else 0) % count
    flags = str(order["fit_bound_flags"][following].values)
    chi2 = float(order["fit_reduced_chi_squared"][following])
    tool.message = (
        f"Problem voxel {following + 1}/{count} {positions[following]}: "
        f"chi2/N {chi2:.2f}" + (f", {flags}" if flags else "") + "."
    )
    return positions[following]


def add_residual_line(
    tool: FitToolState,
    data: xr.DataArray,
    fit_params: xr.Dataset,
    indices: dict[str, int],
    search_range_hz: tuple[float, float],
) -> bool:
    """Search the selected voxel's fit residual for one more line.

    :func:`~hypermri.analysis.spectroscopy.relevance.add_line_from_residual`
    proposes a line at the largest residual peak in ``search_range_hz`` and
    keeps it only when the information criterion says the data need it. A
    kept line becomes a new table row at its fitted frequency and T2*, with
    the search's frequency window and the tool's T2* bounds; the next fit
    fits it. The noise is the acquisition-tail noise ``data`` carries.

    Parameters
    ----------
    tool : FitToolState
        Tool state; a row may be appended, ``message`` is replaced.
    data : xarray.DataArray
        Complex FIDs that were fitted, with ``acquisition_idx``. Not modified.
    fit_params : xarray.Dataset
        Parameters of the fit of ``data``. Not modified.
    indices : dict of str to int
        Positional index of the selected voxel per dimension.
    search_range_hz : tuple of float
        ``(low, high)`` in Hz where the new line may start.

    Returns
    -------
    bool
        Whether a line was added.

    Raises
    ------
    ValueError
        If the range is not two finite numbers, or the search cannot run (for
        example without noise information).
    """
    low, high = sorted(float(value) for value in search_range_hz)
    if not (math.isfinite(low) and math.isfinite(high)) or low == high:
        raise ValueError(
            "Line from residual: give the search range as two different finite "
            "frequencies in Hz."
        )
    selected = _selected_voxel(data, indices)
    spectrum = data.isel(selected)
    params = fit_params.isel(
        {dim: index for dim, index in selected.items() if dim in fit_params.dims}
    )
    name = f"residual_{tool.n_manual + 1}"
    result = add_line_from_residual(
        spectrum, params, search_range_hz=(low, high), new_metab_name=name
    )
    delta = float(result.line_search_delta_chi2)
    threshold = float(result.line_search_threshold)
    if not bool(result.line_search_tested):
        tool.message = (
            "No line tested: the voxel is not fitted, the residual has no peak in "
            "the range, or the new line hit its amplitude floor."
        )
        return False
    if not bool(result.line_search_accepted):
        tool.message = (
            f"No line added: delta chi2 {delta:.1f} <= threshold {threshold:.1f} "
            f"(residual peak at {float(result.line_search_start_freq_hz):.1f} Hz)."
        )
        return False
    tool.n_manual += 1
    freq_hz = float(result.fit_freqs.sel(metabolite=name))
    t2_ms = 1000.0 * float(result.fit_t2s.sel(metabolite=name))
    peak = manual_peak_setting(
        freq_hz,
        name=name,
        freq_range_hz=float(result.attrs["line_search_window_hz"]),
        min_t2_ms=tool.min_t2_ms,
        max_t2_ms=tool.max_t2_ms,
    )
    low_t2, high_t2 = sorted((peak.min_t2_ms, peak.max_t2_ms))
    tool.peaks.append(replace(peak, initial_t2_ms=min(max(t2_ms, low_t2), high_t2)))
    tool.couplings.append(LineCoupling())
    tool.seedings.append(LineSeeding())
    _rename_unique(tool, [row.name for row in tool.peaks])
    tool.message = (
        f"Added {tool.peaks[-1].name} at {freq_hz:.1f} Hz: delta chi2 {delta:.1f} "
        f"> threshold {threshold:.1f}. Fit again to apply it."
    )
    return True


def fit_tool_settings(tool: FitToolState) -> dict[str, Any]:
    """Return the tool state as the browser shows it.

    Parameters
    ----------
    tool : FitToolState
        Tool state. Not modified.

    Returns
    -------
    dict
        The settings of :class:`FitToolState` (without ``n_manual``,
        ``couplings`` and ``seedings``) with ``peaks`` as a list of dicts
        with the table columns, the :class:`LineCoupling` and
        :class:`LineSeeding` columns included.
    """
    settings = asdict(tool)
    settings.pop("n_manual")
    settings.pop("couplings")
    settings.pop("seedings")
    settings["peaks"] = [
        {column: getattr(peak, column) for column in _PEAK_FIELDS}
        | asdict(coupling)
        | asdict(seeding)
        for peak, coupling, seeding in zip(
            tool.peaks, tool.couplings, tool.seedings, strict=True
        )
    ]
    return settings


def _selected_voxel(data: xr.DataArray, indices: dict[str, int]) -> dict[str, int]:
    """Return the view's slice indices clamped to the non-spectral dims of ``data``.

    The view keeps indices of dimensions it no longer shows, which may lie
    outside ``data``; the spectrum shown is the one at the clamped index.
    """
    return {
        dim: min(max(int(indices.get(dim, 0)), 0), size - 1)
        for dim, size in data.sizes.items()
        if dim != "acquisition_idx"
    }


def _fit_voxel(fit_results: xr.Dataset, indices: dict[str, int]) -> dict[str, int]:
    """Return the positional index of a voxel along each voxel dimension of a fit."""
    return {
        dim: int(indices.get(dim, 0))
        for dim in fit_results["fit_freqs"].dims
        if dim != "metabolite"
    }


def voxel_fit_table(fit_results: xr.Dataset, indices: dict[str, int]) -> dict[str, Any]:
    """Return the fitted values of one voxel as rows of a small table.

    The browser fit tool shows this table for the selected voxel. Each
    fitted line gives one row; amplitude and phase come from the complex
    ``fit_amps``, frequency and T2* from ``fit_freqs`` and ``fit_t2s``
    together with their bound status. A Voigt fit adds its Gaussian width,
    which all lines of a spectrum share.

    Parameters
    ----------
    fit_results : xarray.Dataset
        Spectral fit results with a ``metabolite`` dimension: ``fit_amps``
        (complex, a.u.), ``fit_freqs`` (Hz) and ``fit_t2s`` (s); optional
        ``fit_freqs_ppm`` (ppm), ``fit_freq_status``, ``fit_t2_status``,
        ``fit_gauss_fwhm_hz`` (Hz) and ``fit_gauss_fwhm_status``. Not
        modified.
    indices : dict of str to int
        Zero-based position of the voxel along the voxel dimensions of
        ``fit_freqs`` (all dimensions except ``metabolite``). Missing
        dimensions count as 0; indices outside a dimension are clamped to
        it, so a fit with size 1 along a dimension answers for every index.

    Returns
    -------
    dict
        ``{"rows": [...], "gauss_fwhm_hz": float or None,
        "gauss_status": str}``. Each row holds ``line`` (name),
        ``amplitude`` (``|fit_amps|``, a.u.), ``phase_deg`` (angle of
        ``fit_amps`` in degrees, -180 to 180), ``freq_hz``, ``freq_ppm``
        (None without ``fit_freqs_ppm``), ``t2_ms`` and the bound status
        ``freq_status`` and ``t2_status``. A status is one of
        ``"estimated"``, ``"at_lower_bound"``, ``"at_upper_bound"``,
        ``"fixed"``, ``"not_fitted"`` or ``""`` when the fit has none.
        ``gauss_fwhm_hz`` is None for a Lorentz fit. Missing values are
        None, so the result can be sent as JSON.

    Examples
    --------
    >>> table = voxel_fit_table(csi.fit_results, {"x": 7, "y": 9})  # doctest: +SKIP
    >>> [row["line"] for row in table["rows"]]  # doctest: +SKIP
    ['pyruvate', 'lactate']
    """
    freqs = fit_results["fit_freqs"]
    voxel = {
        dim: min(max(int(indices.get(dim, 0)), 0), freqs.sizes[dim] - 1)
        for dim in freqs.dims
        if dim != "metabolite"
    }

    def line_values(name: str) -> list[Any] | None:
        if name not in fit_results:
            return None
        values = fit_results[name].isel(
            {d: i for d, i in voxel.items() if d in fit_results[name].dims}
        )
        if "metabolite" not in values.dims:
            return None
        return list(values.transpose("metabolite", ...).values.ravel())

    def number(value: Any, scale: float = 1.0) -> float | None:
        value = float(value) * scale
        return value if math.isfinite(value) else None

    lines = [str(name) for name in freqs.coords["metabolite"].values]
    amps = line_values("fit_amps") or [math.nan] * len(lines)
    freq_hz = line_values("fit_freqs")
    freq_ppm = line_values("fit_freqs_ppm")
    t2_s = line_values("fit_t2s") or [math.nan] * len(lines)
    freq_status = line_values("fit_freq_status") or [""] * len(lines)
    t2_status = line_values("fit_t2_status") or [""] * len(lines)
    rows = []
    for index, line in enumerate(lines):
        amp = complex(amps[index])
        finite = math.isfinite(amp.real) and math.isfinite(amp.imag)
        rows.append(
            {
                "line": line,
                "amplitude": abs(amp) if finite else None,
                "phase_deg": math.degrees(math.atan2(amp.imag, amp.real))
                if finite
                else None,
                "freq_hz": number(freq_hz[index]),
                "freq_ppm": None if freq_ppm is None else number(freq_ppm[index]),
                "freq_status": str(freq_status[index]),
                "t2_ms": number(t2_s[index], 1000.0),
                "t2_status": str(t2_status[index]),
            }
        )
    gauss = _voxel_value(fit_results, "fit_gauss_fwhm_hz", voxel)
    gauss_status = ""
    if "fit_gauss_fwhm_status" in fit_results:
        status = fit_results["fit_gauss_fwhm_status"].isel(
            {
                d: i
                for d, i in voxel.items()
                if d in fit_results["fit_gauss_fwhm_status"].dims
            }
        )
        gauss_status = str(status.values.ravel()[0]) if status.size == 1 else ""
    return {
        "rows": rows,
        "gauss_fwhm_hz": number(gauss),
        "gauss_status": gauss_status,
    }


# Fit variables of the mask statistics table: (variable, table key, scale).
_MASK_TABLE_VARIABLES = (
    ("fit_amps", "amplitude", 1.0),
    ("fit_freqs", "freq_hz", 1.0),
    ("fit_freqs_ppm", "freq_ppm", 1.0),
    ("fit_t2s", "t2_ms", 1000.0),
    ("fit_gauss_fwhm_hz", "gauss_fwhm_hz", 1.0),
)


def mask_fit_table(
    fit_results: xr.Dataset,
    masks: Mapping[str, xr.DataArray],
    indices: dict[str, int],
) -> dict[str, Any]:
    """Return mean and spread of the fitted values inside each mask.

    The masks tool of the browser view shows this table. For every mask
    and fitted line it gives the coverage-weighted mean and standard
    deviation of amplitude, frequency and T2* over all voxels of the mask,
    computed by :func:`~hypermri.analysis.anatomical.roi.mask_statistics`.
    Voxels with a failed fit (NaN) are left out.

    Parameters
    ----------
    fit_results : xarray.Dataset
        Spectral fit results as for :func:`voxel_fit_table`, on the voxel
        grid of the masks. Not modified.
    masks : mapping of str to xarray.DataArray
        Coverage in [0, 1] by mask name, on the voxel dimensions of the
        fit (for example ``read_idx``, ``phase_idx``, ``slice_idx``).
    indices : dict of str to int
        Zero-based position along the further dimensions of the fit that
        the masks do not cover, such as ``repetitions``; the statistics
        are taken at this position. Missing dimensions count as 0 and
        indices outside a dimension are clamped to it.

    Returns
    -------
    dict
        ``{"rows": [...], "message": str}``. Each row holds ``mask``,
        ``line``, ``n_voxels`` (voxels with a finite frequency and coverage
        above 0) and, as ``[mean, std]``, ``amplitude`` (``|fit_amps|``,
        a.u.), ``freq_hz``, ``freq_ppm`` (None without ``fit_freqs_ppm``),
        ``t2_ms`` and ``gauss_fwhm_hz`` (Hz, None for a Lorentz fit).
        Missing values are None, so the result can be sent as JSON.
        Without masks the rows are empty. ``message`` explains why there
        are no rows when the masks do not lie on the grid of the fit.

    Examples
    --------
    >>> table = mask_fit_table(csi.fit_results, csi.masks, {})  # doctest: +SKIP
    >>> table["rows"][0]["t2_ms"]  # doctest: +SKIP
    [21.4, 3.2]
    """
    masks = dict(masks)
    if not masks:
        return {"rows": [], "message": ""}
    mask_dims = {dim for mask in masks.values() for dim in mask.dims}
    freqs = fit_results["fit_freqs"]
    position = {
        dim: min(max(int(indices.get(dim, 0)), 0), freqs.sizes[dim] - 1)
        for dim in freqs.dims
        if dim not in mask_dims and dim != "metabolite"
    }
    names = [name for name, _, _ in _MASK_TABLE_VARIABLES if name in fit_results]
    maps = fit_results[names].isel(
        {dim: i for dim, i in position.items() if dim in fit_results.dims}
    )
    try:
        table = mask_statistics(maps, masks, variables=names)
    except ValueError as error:
        return {"rows": [], "message": str(error)}

    def pair(statistics: Any, scale: float) -> list[float | None]:
        return [
            float(value) * scale if math.isfinite(value) else None
            for value in (statistics["mean"], statistics["std"])
        ]

    lines = [str(name) for name in freqs.coords["metabolite"].values]
    rows = []
    for mask in masks:
        of_mask = table[table["mask"] == str(mask)]
        for line in lines:
            row: dict[str, Any] = {"mask": str(mask), "line": line}
            for name, key, scale in _MASK_TABLE_VARIABLES:
                found = of_mask[of_mask["variable"] == name]
                if name in maps and "metabolite" in maps[name].dims:
                    found = found[found["metabolite"].astype(str) == line]
                row[key] = pair(found.iloc[0], scale) if len(found) else None
                if name == "fit_freqs":
                    row["n_voxels"] = int(found["n_voxels"].iloc[0])
            rows.append(row)
    return {"rows": rows, "message": ""}


def _write_voxel(
    fit_results: xr.Dataset, voxel_results: xr.Dataset, voxel: dict[str, int]
) -> None:
    """Copy the refitted values of one voxel into ``fit_results`` in place.

    Variables of :data:`_REFIT_VARIABLES` missing from either Dataset are
    skipped; variables with a metabolite dimension are matched by name.
    """
    for name in _REFIT_VARIABLES:
        if name not in fit_results or name not in voxel_results:
            continue
        target = fit_results[name]
        source = voxel_results[name]
        if "metabolite" in target.dims:
            source = source.sel(metabolite=target.metabolite.values)
        values = np.asarray(
            source.transpose(*[d for d in target.dims if d not in voxel]).values
        )
        if target.dtype.kind == "U" and values.dtype.itemsize > target.dtype.itemsize:
            target = fit_results[name] = target.astype(values.dtype)
        index = tuple(voxel.get(d, slice(None)) for d in target.dims)
        target.values[index] = values


def _voxel_value(fit_results: xr.Dataset, name: str, voxel: dict[str, int]) -> float:
    """Return a scalar result of one voxel, or NaN when it is absent."""
    if name not in fit_results:
        return math.nan
    selected = fit_results[name].isel(
        {d: i for d, i in voxel.items() if d in fit_results[name].dims}
    )
    return float(selected) if selected.size == 1 else math.nan


def _seeded_params(
    tool: FitToolState, fit_params: xr.Dataset, earlier_fit: xr.Dataset | None
) -> tuple[xr.Dataset, str]:
    """Apply the start values from an earlier fit and the T2* windows.

    ``fit_params`` hold the lines of the table in row order. Returns the new
    parameters and a note for the tool message.
    """
    # Rows switched off are not in fit_params, so match rows by line name.
    rows = {
        peak.name: (peak, seeding)
        for peak, seeding in zip(tool.peaks, tool.seedings, strict=True)
    }
    names = [
        str(name)
        for name in fit_params.coords["metabolite"].values
        if str(name) in rows
    ]
    seedings = [rows[name][1] for name in names]
    windows = [_window_s(*rows[name]) for name in names]
    seeded, note = seed_start_values(
        fit_params,
        earlier_fit,
        metabs=names,
        seed_freq=[seeding.freq_from_fit for seeding in seedings],
        seed_t2=[seeding.t2_from_fit for seeding in seedings],
        t2_windows_s=windows,
    )
    if seeded is fit_params:
        # Without seeded starts, a window would lie around the typed start.
        return fit_params, note
    return apply_t2_windows(seeded, metabs=names, windows_s=windows), note


def _window_s(peak: PeakSetting, seeding: LineSeeding) -> float | None:
    """Return the T2* window in s of a seeded line, or None."""
    if seeding.t2_from_fit and not peak.fixed_t2 and seeding.t2_window_ms > 0:
        return seeding.t2_window_ms / 1000.0
    return None


def _check_row(tool: FitToolState, index: int) -> None:
    """Fail when ``index`` is not a row of the table."""
    if not 0 <= index < len(tool.peaks):
        raise ValueError(
            f"Fit tool: no table row {index}; the table has {len(tool.peaks)} rows."
        )


def _converted(kind: type, value: Any, name: str) -> Any:
    """Convert a browser value to ``kind``; numbers must be finite."""
    if kind is bool:
        return bool(value)
    if kind is str:
        return str(value)
    try:
        number = kind(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Fit tool: {name} needs a number; got {value!r}.") from error
    if number != number or number in (float("inf"), float("-inf")):
        raise ValueError(f"Fit tool: {name} needs a finite number; got {value!r}.")
    return number


def _rename_unique(tool: FitToolState, previous_names: list[str]) -> None:
    """Give the rows unique, non-empty names; followers keep their leader.

    ``previous_names`` are the row names before the change, in row order;
    a line that followed a renamed row follows it under its new name.
    """
    names, _ = unique_peak_names([peak.name for peak in tool.peaks])
    tool.peaks = [
        replace(peak, name=name) for peak, name in zip(tool.peaks, names, strict=True)
    ]
    renamed = dict(zip(previous_names, names, strict=True))
    tool.couplings = [
        replace(coupling, t2_leader=renamed.get(coupling.t2_leader, ""))
        if coupling.t2_leader
        else coupling
        for coupling in tool.couplings
    ]


def _check_coupling(tool: FitToolState, index: int, column: str, value: Any) -> None:
    """Fail for a leader that is no other line or a non-positive number."""
    if column == "t2_leader" and value:
        others = [peak.name for row, peak in enumerate(tool.peaks) if row != index]
        if value not in others:
            raise ValueError(
                f"Fit tool: T2* leader of {tool.peaks[index].name} must be "
                f"another line of the table ({others}) or empty; got {value!r}."
            )
    if column in ("t2_ratio_low", "t2_ratio_high", "phase_tol_deg") and value <= 0:
        raise ValueError(f"Fit tool: {column} must be positive; got {value}.")


def _coupling_options(
    tool: FitToolState, names: list[str], scan: Any | None
) -> dict[str, Any]:
    """Return the T2* coupling and phase-band options of the fit.

    ``names`` are the line names of the fit configuration, in table order.
    The keys and value forms are those of
    :func:`~hypermri.analysis.fitting.spectroscopy.def_fit_params`, which
    validates them. A missing phase-band delay is derived from the scan once
    and kept in the tool.
    """
    table_names = [peak.name for peak in tool.peaks]
    t2_coupling: dict[str, tuple[str, Any]] = {}
    phase_band: dict[str, float] = {}
    for name, coupling in zip(names, tool.couplings, strict=True):
        if coupling.t2_leader:
            if coupling.t2_leader not in table_names:
                raise ValueError(
                    f"Fit tool: T2* leader {coupling.t2_leader!r} of {name} is "
                    "no line of the table; choose another leader."
                )
            leader = names[table_names.index(coupling.t2_leader)]
            low, high = coupling.t2_ratio_low, coupling.t2_ratio_high
            t2_coupling[name] = (leader, low if low == high else (low, high))
        if coupling.phase_band:
            phase_band[name] = float(coupling.phase_tol_deg)
    options: dict[str, Any] = {}
    if t2_coupling:
        options["t2_coupling"] = t2_coupling
        options["t2_coupling_clip_start"] = bool(tool.t2_coupling_clip_start)
    if phase_band:
        if not tool.phase_band_tau_ms:
            tool.phase_band_tau_ms = 1000.0 * _phase_band_tau_s(scan)
        options["phase_band_tolerance_deg"] = phase_band
        options["phase_band_tau_s"] = tool.phase_band_tau_ms / 1000.0
    return options


def _phase_band_tau_s(scan: Any | None) -> float:
    """Derive the phase-band delay in s from the scan timing."""
    if scan is None:
        raise ValueError(
            "Fit tool: the phase band needs the delay tau; no scan is attached "
            "to derive it. Enter tau in ms."
        )
    try:
        tau_s = float(phase_band_tau_s_from_scan(scan))
    except (AttributeError, KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "Fit tool: the phase band needs the delay tau; deriving it from "
            f"the scan failed ({error}). Enter tau in ms."
        ) from error
    if not math.isfinite(tau_s):
        raise ValueError(
            f"Fit tool: the scan gives no finite phase-band delay ({tau_s}); "
            "enter tau in ms."
        )
    return tau_s
