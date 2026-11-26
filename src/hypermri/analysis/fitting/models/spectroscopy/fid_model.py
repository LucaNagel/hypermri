"""Numerical FID basis, preparation and iterative pseudo-inverse estimation.

Frequencies are Hz, sampling and Lorentz decay times seconds, amplitudes
complex line areas in input signal units and optional Gaussian widths Hz.
This module owns existing scientific defaults and scalar numerical operations;
labelled execution, diagnostics and uncertainty delegate to these operations
from the spectroscopy workflows. Numerical functions never import a workflow
or call the generic fitting facade. Input mutation, precision and result axes
are documented per operation; no handler cache is introduced.
"""

from __future__ import annotations

import difflib
import warnings
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import xarray as xr
from scipy.optimize import least_squares, minimize_scalar, nnls
from scipy.special import wofz

from hypermri.analysis.spectroscopy.frequency import freq_hz_to_ppm, freq_ppm_to_hz
from hypermri.analysis.spectroscopy.metabolites import get_metab_cs_ppm
from hypermri.compatibility import resolve_frequency_axis

from .diagnostics import FIDNumericalError

MODEL_ID = "fid_pseudo_inv"
RESULT_MODEL_NAME = "FID"
DEFAULT_MIN_PEAK_SEPARATION_HZ = 5.0
NPOINTS_DEFAULT = 31
NITER_DEFAULT = 1
REP_FITTING_DEFAULT = 11
MIN_T2_S_DEFAULT = 0.005
MAX_T2_S_DEFAULT = 0.07
METABS_FREQ_RANGE_DEFAULT = 70.0
METABS_T2_RANGE_DEFAULT = 0.04
INITIAL_T2_S_DEFAULT = 0.025
PSEUDO_INV_DEFAULT = True
SNR_THRESHOLD_DEFAULT = None
USE_SSE_MAG_DEFAULT = False
ZOOMFACTOR_DEFAULT = 1.5
SHOW_TQDM_DEFAULT = False
UNCERTAINTY_BOOTSTRAP_SAMPLES_DEFAULT = 999
UNCERTAINTY_CONFIDENCE_LEVEL_DEFAULT = 0.95
UNCERTAINTY_SEED_DEFAULT = 0
LINESHAPES = ("lorentz", "voigt")
LINESHAPE_DEFAULT = "lorentz"
SEARCH_ORDERS = ("metabolite", "amplitude")
SEARCH_ORDER_DEFAULT = "amplitude"
JOINT_REFINEMENT_DEFAULT = True
GAUSS_INITIAL_FWHM_HZ_DEFAULT = 0.0
GAUSS_MAX_FWHM_HZ_DEFAULT = 40.0
GAUSS_SEARCH_WIDTH_HZ_DEFAULT = 40.0
# In Voigt mode fit_t2s is the Lorentz part only, which is longer than a
# Lorentz-only T2*; 1 s keeps the lowest Lorentz width at about 0.3 Hz.
VOIGT_MAX_T2_S_DEFAULT = 1.0
_GAUSS_OPTION_KEYS = frozenset(
    {"gauss_initial_fwhm_hz", "gauss_max_fwhm_hz", "gauss_search_width_hz"}
)
# Absolute tolerance for reporting a fitted Gaussian FWHM at a search bound.
_GAUSS_BOUND_TOLERANCE_HZ = 1e-9
# Olivero & Longbothum (1977) approximation of the Voigt FWHM, accurate to
# about 0.02 % over the full Lorentz/Gauss range.
_OLIVERO_LINEAR = 0.5346
_OLIVERO_QUADRATIC = 0.2166
# Phase-band tolerances above 90 degrees no longer give a convex set of
# allowed amplitudes, so the amplitude problem stops being a cone projection.
_PHASE_BAND_MAX_TOLERANCE_DEG = 90.0
# Coarse phi0 grid of the phase-band solver; its best point is refined by a
# bounded Brent search over one grid step to either side.
_PHASE_BAND_PHASE_GRID_POINTS = 24
_PHASE_BAND_PHASE_XATOL_RAD = 1e-4
# Rounding margin when testing whether free amplitudes lie inside their
# phase band; far below the phase-search tolerance above.
_PHASE_BAND_EDGE_MARGIN_RAD = 1e-12


def get_fid_pseudo_inv_params_info() -> dict[str, object]:
    """Describe the supported FID scientific options and result fields.

    Returns
    -------
    dict[str, object]
        Fresh descriptions of required/optional options, fitted variables and
        the numerical model. Frequency and linewidth entries use Hz, decay and
        sampling use seconds, and complex amplitudes use input signal units.
        Descriptions do not prepare parameters or fit data. Calls are eager,
        deterministic and do not mutate scientific defaults or retain a cache."""
    return {
        "required_fit_params": {
            "metabs": "List of metabolite names to fit (e.g., ['Pyruvate', 'Lactate', 'Alanine'])",
        },
        "optional_fit_params": {
            "b0": "Magnetic field strength in Tesla (e.g., 7.0 for 7T)",
            "bw_hz": "Bandwidth in Hz. Default: calculated from dwell time",
            "cut_off": "Frequency cutoff for filtering. Default: None",
            "dwell_time_s": "Dwell time in seconds. Default: calculated from input_data",
            "freq_axis_hz": "Frequency axis in Hz. Default: auto-calculated",
            "freq_axis_ppm": "Frequency axis in ppm. Default: auto-calculated",
            "metabs_freqs_hz": "Initial frequencies in Hz, in metabolite order; no ppm axis required. The legacy spelling metabs_freqs_Hz is also accepted.",
            "metabs_freqs_ppm": "Initial frequencies in ppm, in metabolite order. Requires a ppm axis and takes precedence over Hz values. Missing entries use explicit Hz values or known chemical shifts.",
            "initial_t2s_s": "Initial T2* values in seconds (per metabolite). Default: 0.025",
            "max_t2_s": "Maximum T2* value in seconds. Default: 0.07",
            "metabs_freq_ranges": "Full frequency search width in Hz (per metabolite); the search spans initial_freq +/- width/2. Default: 70.0",
            "metabs_t2_ranges": "Full T2* search width in seconds (per metabolite); non-negative, with zero freezing T2*. Default: 0.04",
            "metabs_t2s": "Legacy alias for initial_t2s_s; the canonical key takes precedence. Default: 0.025",
            "min_t2_s": "Minimum T2* value in seconds. Default: 0.005",
            "min_peak_separation_hz": "Minimum allowed separation between fitted metabolite frequencies (Hz). Candidates violating this are rejected. fit_data_pseudo_inv and add_metab_to_fit_params raise before fitting when two start frequencies (initial_freqs_hz) of lines that are not skipped lie closer in any spectrum. Default: 5.0",
            "t2_coupling": "Optional mapping {dependent: (leader, t2_ratio)} or {dependent: leader} (t2_ratio 1.0) that ties T2* values: T2*(dependent) = t2_ratio * T2*(leader), with t2_ratio > 0 (the Lorentz part in Voigt mode). The leader's T2* search moves both lines; candidates leaving either line's [min_t2_s, max_t2_s] are rejected. The dependent's own initial_t2s_s and metabs_t2_ranges are not used and a frozen leader freezes it. If the leader is skipped or amplitude-filtered, the first remaining dependent (metabolite order) takes over as leader, searching with the old leader's metabs_t2_ranges times its t2_ratio; the other dependents follow it with their t2_ratio divided by its t2_ratio, and a lone remaining dependent is fitted independently. A ratio band {dependent: (leader, (low, high))} with 0 < low <= high couples more loosely: the dependent keeps its own initial_t2s_s and metabs_t2_ranges and is searched itself, but every accepted state keeps low <= T2*(dependent) / T2*(leader) <= high; the leader's search moves it with its current ratio, and in Voigt mode it keeps its own total width. A band leader is then searched once more on its own: its banded dependents keep their T2* and the bands limit its candidates, so a weak leader can leave a strong dependent. The start ratio must lie in the band; low == high is the fixed ratio low. A banded dependent never takes over a missing leader; it follows a fixed new leader with its band divided by that leader's t2_ratio, and is fitted independently if no fixed dependent remains. Leaders cannot be dependents. Names, T2* values, coupling roles and phase_band_tolerance_deg entries belong to frequency positions: the k-th lowest line of a coupling group (a leader and its dependents) carries the group's name with the k-th lowest initial frequency (mean over all spectra; equal means by metabolite name). The frequency search scores every candidate in this assignment, so a line that passes another line of its group takes over that line's name and role. Each line keeps its own frequency window (metabs_freq_ranges and its bounds) and is searched only inside it; the window moves with the line, so windows of any widths and positions are allowed, and a frozen line (metabs_freq_ranges 0) keeps its frequency even when another line passes it and its name changes. For coupled lines fit_component_index equals the metabolite index and does not record which line started where. Not supported with detection_target or local covariance. Default: None (independent T2*)",
            "t2_coupling_clip_start": "If True, start T2* values that t2_coupling would reject are moved to the nearest admissible value before fitting instead of raising an error: a banded dependent's initial_t2s_s is clipped to [low, high] times its leader's start (and its own [min_t2_s, max_t2_s]); a leader's initial_t2s_s is clipped so that every fixed dependent starts inside its [min_t2_s, max_t2_s]. Admissible start values stay unchanged. Spectra without any admissible start still raise. Default: False",
            "phase_band_tolerance_deg": "Optional mapping {metabolite: tolerance in degrees} that constrains complex amplitudes to one phase law: a_k = A_k * exp(1j * (phi0 + 2*pi*f_k*phase_band_tau_s + delta_k)) with A_k >= 0 and abs(delta_k) <= tolerance. phi0 is solved per spectrum and shared by all listed lines; unlisted lines keep free complex amplitudes. Tolerances lie in [0, 90]; 0 fixes the phase. Requires phase_band_tau_s and amp_solver='pinv'. Not supported with detection_target or local covariance. Default: None (free amplitudes)",
            "phase_band_tau_s": "Time in seconds from the common signal origin, where all lines share phi0, to t = 0 of sampling_time_axis_s (the first acquired sample); positive when the origin lies before that sample. Finite; required with, and only used by, phase_band_tolerance_deg. For Bruker FID scans, hypermri.analysis.spectroscopy.phase_band_tau_s_from_scan derives it from TE and the dwell time (validated for <User:lucaCSI4>, a warning otherwise)",
            "amp_solver": "Amplitude solver ('pinv', 'normal_eq', or 'ridge'); explicit values override pseudo_inv. Aliases: pseudo_inv/pseudoinv, normal, tikhonov. Default: 'pinv' (or legacy 'normal_eq' when pseudo_inv=False)",
            "amp_ridge_lambda": "Finite, non-negative regularization penalty, used only by amp_solver='ridge'. Default: 0.0",
            "niter": "Number of iterative zoom refinements. Default: 1",
            "nucleus": "Nucleus being measured (e.g., '1H', '13C', '31P')",
            "npoints": "Number of grid points for parameter search. Default: 31",
            "nsamplepoints": "Number of sampling points. Default: from input_data",
            "pseudo_inv": "Select pinv (True) or legacy normal_eq (False) when amp_solver is omitted. Default: True",
            "rep_fitting": "Number of alternating frequency/T2* refinement passes. Default: 11",
            "sampling_time_axis_s": "Time axis in seconds. Default: auto-calculated",
            "signal_domain": "Domain of the signal ('temporal' or 'spectral'). Default: 'temporal'",
            "skip_fitting": "Boolean scalar or one value per metabolite; True excludes that component from fitting. Edit the prepared map for voxel-specific exclusions. Skipped amplitudes are zero and diagnostics are NaN. Default: False everywhere",
            "snr_threshold": "Amplitude threshold for metabolite exclusion. Metabolites with |amplitude| below threshold are refitted without. Default: None",
            "spectral_dim": "Name of the spectral dimension. Default: 'acquisition_idx'",
            "show_tqdm": "Show voxel-wise fitting progress when supported. Default: False",
            "search_order": "Order in which every frequency and T2* grid step visits the lines. 'amplitude' (default) solves the amplitudes once at the start of each step and visits the line with the largest fitted signal (norm of its component FID) first, so results do not depend on how the metabolite list is ordered; equal signals keep metabolite order. Costs one extra amplitude solve per step with two or more lines. 'metabolite' follows the metabolite list and reproduces fits made before the default changed in November 2026. Default: 'amplitude'",
            "joint_refinement": "If True (default), after the rep_fitting grid passes a local least-squares step (scipy.optimize.least_squares, trust-region reflective) refines all free frequencies and T2* values together, inside their absolute bounds and t2_coupling ratios and bands, with amplitudes solved linearly as in the grid search; the Voigt Gaussian FWHM stays fixed. It is accepted only if it lowers the grid objective and keeps min_peak_separation and the frequency order of coupled lines. This recovers lines that one-at-a-time grid moves leave between grid points or along a narrow valley. False reproduces fits made before the default changed in November 2026. Default: True",
            "keep_fit_order": "If False (default), fit_data_pseudo_inv relabels the fitted components per spectrum so that their fitted frequencies follow the order of the initial frequencies (mean over all spectra; equal means are ordered by metabolite name). This removes label swaps between lines with overlapping search ranges. Only components with the same constraint role are exchanged: non-skipped lines that are not in t2_coupling and have the same phase_band_tolerance_deg entry. All per-metabolite outputs except the optimization history move together, fit_component_index records the solver component behind each label, and the fitted signal is unchanged. Without effect in statistical mode (detection_target). True keeps the solver order",
            "store_optimization_history": "Store complete candidate and accepted-state records, including complex amplitudes and RSS. Default: False; memory grows with spectra, metabolites and candidate count",
            "uncertainty_bootstrap_samples": "Number B of complete parametric-bootstrap refits per selected spectrum. Default when uncertainty is requested: 999; setting this option alone does not run a bootstrap",
            "uncertainty_confidence_level": "Central percentile interval level for parameter bootstrap. Default when uncertainty is requested: 0.95",
            "uncertainty_seed": "Non-negative base seed for deterministic per-spectrum parameter-bootstrap streams. Default when uncertainty is requested: 0",
            "detection_target": "Optional single metabolite label (initial scope: bicarbonate) for mandatory full-refit M0/M1 detection; None disables statistical comparison",
            "detection_sigma": "Required in statistical mode: supplied white-complex Gaussian noise SD per real/imaginary quadrature of the fitted temporal FID; E[abs(noise)**2]=2*sigma**2",
            "detection_null_samples": "Explicit positive integer: fixed number of full M0/M1 null refits per spectrum; no adaptive stopping",
            "detection_seed": "Explicit non-negative integer seed for calibration; use independent seeds for external validation and alternatives",
            "detection_alpha": "Uncorrected per-spectrum significance level, default 0.05; requires independent calibration validation",
            "detection_weak_snr": "Provisional conditional amplitude SNR diagnostic threshold, default 3.0; does not select tests or restrict the model",
            "detection_store_null": "Retain raw null statistics, failure reason codes and recovery flags for later analysis. Default: False; does not change fitting or decisions",
            "use_sse_mag": "Score candidates using spectral-magnitude SSE instead of temporal complex SSE. Default: False",
            "zoomfactor": "Factor by which to zoom search grid per iteration. Default: 1.5",
            "lineshape": "'lorentz' (default) or 'voigt'. 'voigt' multiplies every component FID by exp(-(pi*G*t)**2/(4*ln 2)) with one shared Gaussian FWHM G in Hz per spectrum; fit_t2s is then the Lorentz part only. In Voigt mode metabs_t2_ranges only enables (>0) or freezes (0) the Lorentz-width search, whose width in Hz starts at the range allowed by min_t2_s/max_t2_s, and max_t2_s defaults to 1.0 s. Not supported together with detection_target or store_optimization_history",
            "gauss_initial_fwhm_hz": "Voigt only: starting Gaussian FWHM G in Hz, shared by all metabolites of a spectrum. Default: 0.0 (start from the Lorentz shape); edit the prepared initial_gauss_fwhm_hz map for voxel-specific starts",
            "gauss_max_fwhm_hz": "Voigt only: upper bound for G in Hz; the lower bound is 0. Default: 40.0",
            "gauss_search_width_hz": "Voigt only: full search width for G in Hz, shrunk by zoomfactor after each repetition; candidates keep every line's total Voigt FWHM. 0 keeps G fixed at its start value. Default: 40.0",
        },
        "output_parameters": {
            "fit_amps": "Fitted amplitudes for each metabolite (line areas at t=0)",
            "fit_freqs": "Fitted frequencies in Hz for each metabolite",
            "fit_t2s": "Fitted T2* relaxation times in seconds for each metabolite; the Lorentz part only in Voigt mode",
            "fit_fwhm_hz": "Absorption-spectrum full-width at half-maximum in Hz: 1/(pi*T2*) for Lorentz; Olivero-Longbothum Voigt FWHM from fit_lorentz_fwhm_hz and fit_gauss_fwhm_hz in Voigt mode. Line broadening applied before the fit is contained in T2*; FID truncation is not included",
            "fit_magnitude_fwhm_hz": "Magnitude-spectrum full-width at half-maximum in Hz: sqrt(3)/(pi*T2*) for Lorentz; the exact half-maximum width of the magnitude Voigt line in Voigt mode. FID truncation is not included",
            "fit_lorentz_fwhm_hz": "Voigt only: Lorentz FWHM 1/(pi*fit_t2s) in Hz for each metabolite",
            "fit_gauss_fwhm_hz": "Voigt only: fitted Gaussian FWHM in Hz shared by all metabolites of a spectrum",
            "fit_gauss_fwhm_status": "Voigt only: estimated, at_lower_bound, at_upper_bound, fixed or not_fitted",
            "fit_rss": "Temporal complex residual sum of squares reconstructed from the returned joint parameter state",
            "fit_objective": "Final selected objective; see fit_objective_name metadata",
            "fit_status": "Numerical status per spectrum: success, zero_signal, skipped, invalid_input or fit_failed; not a detection decision",
            "fit_failure_reason": "Stable numerical failure reason, or none",
            "detection_status": "Statistical mode only: detected, not_detected, inconclusive, fit_failed or not_tested; single configured target",
            "test_statistic_raw": "Statistical mode: (rss_m0-rss_m1)/sigma**2, without clipping; not Student t",
            "detection_p_value": "Approximate full-refit M0 plug-in Monte Carlo p-value using (n_extreme+1)/(n_null+1); not a signal/absence probability",
            "weak_candidate": "Provisional conditional SNR, boundary and conditioning flags; never modify the model or select tests",
            "fit_amps_accepted": "Separate NaN-masked view of M1 amplitudes; only the tested and detected target is accepted",
            "calibration_id": "SHA256 identifier including fitted-M0 state, input, supplied noise, settings, stream/seed, runtime and current package sources; no automatic reuse",
            "fit_search_diagnostics_freq": "Candidate Hz, temporal RSS and spectral magnitude RSS, with labelled search_metric axis",
            "fit_search_diagnostics_t2": "Candidate seconds, temporal RSS and spectral magnitude RSS, with labelled search_metric axis",
            "fit_r2_freq": "Frequency-search diagnostics: candidate Hz, complex-FID SSE and magnitude-spectrum SSE (legacy name, not R-squared)",
            "fit_r2_t2": "T2*-search diagnostics: candidate seconds, complex-FID SSE and magnitude-spectrum SSE (legacy name, not R-squared)",
            "fit_low_snr_refitted": "Boolean flag indicating metabolites excluded due to low amplitude",
            "fit_component_index": "Index, in metabolite order, of the solver component reported under each label; differs from the label's own index where keep_fit_order=False exchanged components",
        },
        "model_equation": "Iterative pseudo-inverse fitting of Free Induction Decay (FID) signals",
    }


# The parameter info above is the single list of accepted options.  The
# notebook spelling of the Hz alias and the model markers written by
# def_fit_params are accepted as well, so prepared attributes can be prepared
# again.
_PARAMS_INFO = get_fid_pseudo_inv_params_info()
_ACCEPTED_FIT_PARAM_KEYS = frozenset(
    {
        *_PARAMS_INFO["required_fit_params"],
        *_PARAMS_INFO["optional_fit_params"],
        "metabs_freqs_Hz",
        "model_name",
        "fit_type",
    }
)
del _PARAMS_INFO


def _reject_unknown_fit_params(fit_params: dict) -> None:
    """Raise for option names that the spectroscopic fit does not read.

    Names starting with ``gauss_`` or ``detection_`` and the prepared map name
    ``initial_gauss_fwhm_hz`` are left to the lineshape and detection checks
    later in ``def_fit_params``, which give option-specific messages.
    """
    unknown = sorted(
        str(key)
        for key in fit_params
        if key not in _ACCEPTED_FIT_PARAM_KEYS
        and not str(key).startswith(("gauss_", "detection_"))
        and key != "initial_gauss_fwhm_hz"
    )
    if not unknown:
        return
    hints = []
    for key in unknown:
        close = difflib.get_close_matches(key, _ACCEPTED_FIT_PARAM_KEYS, n=1)
        if close:
            hints.append(f"{key!r} (did you mean {close[0]!r}?)")
        else:
            hints.append(repr(key))
    raise ValueError(
        f"def_fit_params: unknown fit_params key(s) {', '.join(hints)}. Valid "
        f"keys are {sorted(_ACCEPTED_FIT_PARAM_KEYS)}; see "
        "get_fid_pseudo_inv_params_info() for their meaning."
    )


def _def_fit_params_calc_sampling(data: xr.DataArray, fit_params: dict) -> dict:
    """Calculate sampling parameters."""
    # get the spectral dimension name
    spectral_dim = fit_params.get("spectral_dim", "acquisition_idx")
    # Number of points to be cut off from the beginning of the time axis
    cut_off = fit_params.get("cut_off", 0)
    # B0 fueld strength in Tesla
    b0 = fit_params.get("b0", 7)
    # nucleus type, default to 13C if not specified
    nucleus = fit_params.get("nucleus", "13c")
    # number of spectral points in the input data
    spectral_size = int(data.sizes[spectral_dim])
    # Determine the sampling time axis in seconds
    sampling_axis_coord = data.coords.get("sampling_time_axis_s", None)
    if sampling_axis_coord is not None:
        full_time_coords = np.asarray(sampling_axis_coord.values)
    else:
        sampling_axis_attr = fit_params.get("sampling_time_axis_s")
        if sampling_axis_attr is None:
            sampling_axis_attr = data.attrs.get("sampling_time_axis_s")

        if sampling_axis_attr is not None:
            full_time_coords = np.asarray(sampling_axis_attr)
        else:
            full_time_coords = np.asarray(data.coords[spectral_dim].values)

    nsamplepoints_total = len(full_time_coords)
    if nsamplepoints_total > spectral_size and "cut_off" not in fit_params:
        cut_off = nsamplepoints_total - spectral_size

    cut_off = max(0, min(cut_off, nsamplepoints_total - 1))
    sampling_time_axis_s = (full_time_coords - full_time_coords[0])[cut_off:]
    if len(sampling_time_axis_s) > spectral_size:
        sampling_time_axis_s = sampling_time_axis_s[:spectral_size]

    requested_nsamplepoints = fit_params.get("nsamplepoints", None)
    if requested_nsamplepoints is not None:
        requested_nsamplepoints = max(1, int(requested_nsamplepoints))
        if requested_nsamplepoints < len(sampling_time_axis_s):
            sampling_time_axis_s = sampling_time_axis_s[:requested_nsamplepoints]

    nsamplepoints = len(sampling_time_axis_s)

    if nsamplepoints > 1:
        dwell_time_s = np.abs(sampling_time_axis_s[1] - sampling_time_axis_s[0])
    else:
        dwell_time_s = (
            np.abs(full_time_coords[1] - full_time_coords[0])
            if nsamplepoints_total > 1
            else 1.0
        )

    signal_domain = fit_params.get(
        "signal_domain", data.attrs.get("signal_domain", "temporal")
    )

    from hypermri.analysis.spectroscopy.frequency import get_bw_from_freq_axis

    original_freq_axis_hz = resolve_frequency_axis(data, "Hz")
    original_freq_axis_ppm = resolve_frequency_axis(data, "ppm")

    freq_axis_hz = (
        None if original_freq_axis_hz is None else np.asarray(original_freq_axis_hz)
    )
    if freq_axis_hz is None or freq_axis_hz.size != nsamplepoints:
        freq_axis_hz = np.fft.fftshift(np.fft.fftfreq(nsamplepoints, d=dwell_time_s))

    if nsamplepoints > 1:
        bw_hz = get_bw_from_freq_axis(freq_axis=freq_axis_hz)
    else:
        bw_hz = 1.0 / dwell_time_s if dwell_time_s else 0.0

    freq_axis_ppm = (
        None if original_freq_axis_ppm is None else np.asarray(original_freq_axis_ppm)
    )
    if freq_axis_ppm is not None and freq_axis_ppm.size != nsamplepoints:
        original_hz = (
            None if original_freq_axis_hz is None else np.asarray(original_freq_axis_hz)
        )
        if (
            original_hz is not None
            and original_hz.size == freq_axis_ppm.size
            and original_hz.size > 1
        ):
            freq_axis_ppm = freq_hz_to_ppm(
                freq_axis_hz,
                hz_axis=original_hz,
                ppm_axis=freq_axis_ppm,
            )
        else:
            freq_axis_ppm = None

    fit_params.update(
        {
            "nsamplepoints": nsamplepoints,
            "dwell_time_s": dwell_time_s,
            "sampling_time_axis_s": sampling_time_axis_s,
            "freq_axis_hz": freq_axis_hz,
            "freq_axis_ppm": freq_axis_ppm,
            "bw_hz": bw_hz,
            "cut_off": cut_off,
            "b0": b0,
            "nucleus": nucleus,
            "signal_domain": signal_domain,
            "spectral_dim": spectral_dim,
        }
    )
    return fit_params


def _coerce_metabolite_frequencies(
    values: object, num_metabs: int, parameter_name: str
) -> list[float | None]:
    """Normalize scalar or one-dimensional frequencies in metabolite order.

    Preserve ``None`` entries and pad short lists with ``None`` so callers can
    supply another unit or a default for those metabolites. Reject extra
    entries, nonnumeric values and nonfinite numbers with the parameter name.
    """
    if values is None:
        return [None] * num_metabs
    array = np.atleast_1d(np.asarray(values, dtype=object))
    if array.ndim != 1 or len(array) > num_metabs:
        raise ValueError(
            f"'{parameter_name}' must be a scalar or a one-dimensional list "
            f"with at most {num_metabs} entries, in metabolite order."
        )

    frequencies = []
    for value in array:
        if value is not None:
            try:
                value = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"'{parameter_name}' must contain finite numbers or None."
                ) from exc
            if not np.isfinite(value):
                raise ValueError(
                    f"'{parameter_name}' must contain finite numbers or None."
                )
        frequencies.append(value)
    return frequencies + [None] * (num_metabs - len(frequencies))


def _def_fit_params_get_metabs(fit_params: dict) -> dict:
    """Resolve frequencies and store them under canonical parameter names.

    Explicit ppm values retain precedence over Hz values and require a ppm
    axis for conversion. Explicit Hz values need no ppm axis; their derived
    ppm metadata is ``None`` when that calibration is absent. Missing values
    fall back to the existing chemical-shift defaults when an axis is present.

    Accept ``metabs_freqs_Hz`` as an alias for ``metabs_freqs_hz``. If both are
    supplied, require matching values. Invalid frequencies or missing ppm
    calibration raise ``ValueError`` before constructing the fit maps.
    """
    metabs = fit_params.get("metabs", ["pyruvate", "lactate", "alanine"])
    if isinstance(metabs, str):
        metabs = [metabs]
    num_metabs = len(metabs)

    hz_axis = fit_params["freq_axis_hz"]
    ppm_axis = fit_params["freq_axis_ppm"]

    user_freqs_ppm = _coerce_metabolite_frequencies(
        fit_params.get("metabs_freqs_ppm"), num_metabs, "metabs_freqs_ppm"
    )
    user_freqs_hz = _coerce_metabolite_frequencies(
        fit_params.get("metabs_freqs_hz"), num_metabs, "metabs_freqs_hz"
    )
    legacy_hz = fit_params.pop("metabs_freqs_Hz", None)
    if legacy_hz is not None:
        legacy_hz = _coerce_metabolite_frequencies(
            legacy_hz, num_metabs, "metabs_freqs_Hz"
        )
        if fit_params.get("metabs_freqs_hz") is not None and user_freqs_hz != legacy_hz:
            raise ValueError(
                "'metabs_freqs_hz' and 'metabs_freqs_Hz' contain conflicting "
                "values. Use only 'metabs_freqs_hz' or supply matching values."
            )
        user_freqs_hz = legacy_hz

    final_freqs_ppm = []
    final_freqs_hz = []

    for i, metab_name in enumerate(metabs):
        ppm_val = user_freqs_ppm[i]
        hz_val = user_freqs_hz[i]

        if ppm_val is not None or hz_val is None:
            if ppm_axis is None:
                raise ValueError(
                    f"Cannot resolve the frequency of '{metab_name}' without "
                    "'freq_axis_ppm'. Attach a ppm coordinate, or provide "
                    "'metabs_freqs_hz' with no explicit ppm value for this metabolite."
                )
            if ppm_val is None:
                ppm_val = get_metab_cs_ppm(metab_name) or 0.0
            hz_val = freq_ppm_to_hz(ppm_val, hz_axis, ppm_axis)
        elif ppm_axis is not None:
            ppm_val = freq_hz_to_ppm(hz_val, hz_axis, ppm_axis)

        if not np.isfinite(hz_val):
            raise ValueError(
                f"Resolved frequency for '{metab_name}' is not finite. "
                "Check the frequency values and calibration axes."
            )
        final_freqs_hz.append(float(hz_val))
        final_freqs_ppm.append(None if ppm_val is None else float(ppm_val))

    fit_params.update(
        {
            "metabs": metabs,
            "metabs_freqs_ppm": final_freqs_ppm,
            "metabs_freqs_hz": final_freqs_hz,
        }
    )
    return fit_params


def def_fit_params(
    data: xr.DataArray = None,
    fit_params: dict = None,
    return_params_info: bool = False,
) -> xr.Dataset | None:
    """Prepare spectroscopy fitting parameters.

    Parameters
    ----------
    data : xr.DataArray
        Spectroscopy data with a sampling-time coordinate. A calibrated
        ``freq_axis_ppm`` coordinate is needed only when resolving ppm values
        or default chemical shifts, not for explicit Hz frequencies.
    fit_params : dict, optional
        Model and execution options. The accepted keys are those listed by
        ``get_fid_pseudo_inv_params_info``, the alias ``metabs_freqs_Hz`` and
        the markers ``model_name`` and ``fit_type`` written by this function,
        so the attributes of a prepared Dataset can be passed again.
        ``metabs_freqs_hz`` supplies initial frequencies in Hz. Explicit
        ``metabs_freqs_ppm`` values take precedence and require a ppm axis.
        Frequencies may be scalars or one-dimensional sequences in metabolite
        order. Missing entries use the other unit or a known chemical shift.
        The mapping is copied and not mutated.
    return_params_info : bool, optional
        If True, return None without preparing parameters. Use
        ``get_fid_pseudo_inv_params_info`` for the parameter descriptions.

    Returns
    -------
    xr.Dataset or None
        Metabolite parameter maps and global options. Hz-only input without
        ppm calibration retains ``None`` for unavailable ppm metadata. With
        ``lineshape='voigt'`` the dataset also holds ``initial_gauss_fwhm_hz``
        (Gaussian start FWHM in Hz, loop dimensions only, no ``metabolite``
        dimension), and ``max_t2_s`` defaults to 1 s because ``T2*`` is then
        the Lorentz part only.

    Raises
    ------
    ValueError
        If ``fit_params`` contains an unknown key (the message names it,
        suggests the closest valid key and lists all valid keys), frequency
        aliases conflict, values are invalid, or a required ppm-to-Hz
        calibration is missing. T2* values must be positive and lie
        within ordered positive bounds. Search widths must be non-negative;
        zero freezes the corresponding parameter. Numeric metabolite settings
        accept a scalar for all metabolites or one value per metabolite.
        Invalid lineshape settings raise as described in the parameter info.

    Examples
    --------
    Fit Voigt lines with one shared Gaussian FWHM per spectrum:

    >>> params = def_fit_params(
    ...     data, {"metabs": ["pyruvate", "lactate"], "lineshape": "voigt"}
    ... )  # doctest: +SKIP
    >>> params["initial_gauss_fwhm_hz"]  # doctest: +SKIP

    Notes
    -----
    Parameter maps use ``metabolite`` followed by the non-spectral input
    dimensions in input order, with their coordinate labels preserved. Real
    settings use float64 and the exclusion map uses bool. Preparation reads
    sampling and metadata, fits no signal, preserves inputs and retains no
    cache. It is deterministic; broadcasting preserves the input backend.
    """
    if return_params_info:
        return None

    fit_params = dict(fit_params or {})
    _reject_unknown_fit_params(fit_params)
    fit_params = _def_fit_params_calc_sampling(data, fit_params)
    fit_params = _def_fit_params_get_metabs(fit_params)

    fit_params = _prepare_global_fit_options(fit_params)

    attrs = {
        **fit_params,
        "model_name": RESULT_MODEL_NAME,
        "fit_type": "fid_pseudo_inv",
    }

    spectral_dim = attrs.get("spectral_dim", "acquisition_idx")
    metabolites = attrs["metabs"]
    n_metabs = len(metabolites)
    template = data.isel({spectral_dim: 0}, drop=True)
    voigt = _uses_voigt_lineshape(fit_params)

    defaults = {
        "initial_t2s_s": INITIAL_T2_S_DEFAULT,
        "metabs_freq_ranges": METABS_FREQ_RANGE_DEFAULT,
        "metabs_t2_ranges": METABS_T2_RANGE_DEFAULT,
        "min_t2_s": MIN_T2_S_DEFAULT,
        "max_t2_s": VOIGT_MAX_T2_S_DEFAULT if voigt else MAX_T2_S_DEFAULT,
        "skip_fitting": False,
    }
    values = {"initial_freqs_hz": np.asarray(attrs["metabs_freqs_hz"], dtype=float)}
    for key, default in defaults.items():
        raw = fit_params.get(key)
        # Keep the established alias precedence: the canonical T2* key wins.
        if key == "initial_t2s_s" and raw is None:
            raw = fit_params.get("metabs_t2s")
        values[key] = _coerce_metabolite_values(
            default if raw is None else raw, n_metabs, parameter_name=key
        )

    for key in ("metabs_freq_ranges", "metabs_t2_ranges"):
        if np.any(values[key] < 0):
            raise ValueError(f"{key} must be non-negative; use zero to freeze it.")
    if np.any(values["min_t2_s"] <= 0):
        raise ValueError("min_t2_s must be positive.")
    if np.any(values["max_t2_s"] < values["min_t2_s"]):
        raise ValueError("max_t2_s must be greater than or equal to min_t2_s.")
    if np.any(values["initial_t2s_s"] < values["min_t2_s"]) or np.any(
        values["initial_t2s_s"] > values["max_t2_s"]
    ):
        raise ValueError("initial_t2s_s must lie within [min_t2_s, max_t2_s].")
    values["skip_fitting"] = _coerce_skip_fitting_mask(values["skip_fitting"], n_metabs)

    data_vars = {}
    for key, per_metabolite in values.items():
        parameter = xr.DataArray(
            per_metabolite, dims=["metabolite"], coords={"metabolite": metabolites}
        )
        data_vars[key], _ = xr.broadcast(parameter, template)
    if voigt:
        data_vars["initial_gauss_fwhm_hz"], _ = xr.broadcast(
            xr.DataArray(float(attrs["gauss_initial_fwhm_hz"])), template
        )
    return _update_frequency_bounds(xr.Dataset(data_vars, attrs=attrs))


def _coerce_finite_scalar(value: object, *, parameter_name: str) -> float:
    """Return a finite real scalar or raise a parameter-specific error."""
    try:
        if np.iscomplexobj(value):
            raise ValueError("Complex values are not scalar fit settings.")
        value_array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{parameter_name} must be a finite scalar value.") from exc
    if value_array.size != 1 or not np.all(np.isfinite(value_array)):
        raise ValueError(f"{parameter_name} must be a finite scalar value.")
    return float(value_array.reshape(-1)[0])


def _uses_voigt_lineshape(options: dict) -> bool:
    """Return whether prepared options or result attributes select Voigt lines.

    A missing ``lineshape`` entry means the Lorentz default, so datasets
    produced before the option existed are read as Lorentz fits.
    """
    return options.get("lineshape", LINESHAPE_DEFAULT) == "voigt"


def _describes_voigt_fit(source) -> bool:
    """Return whether options, fit parameters or fit results are Voigt.

    Datasets also count as Voigt when they carry a Gaussian FWHM map, so a
    Voigt dataset whose attributes were dropped is not read as Lorentz.
    """
    if isinstance(source, xr.Dataset) and (
        "fit_gauss_fwhm_hz" in source or "initial_gauss_fwhm_hz" in source
    ):
        return True
    return _uses_voigt_lineshape(getattr(source, "attrs", source))


def _require_lorentz_lineshape(operation: str, *sources) -> None:
    """Reject Voigt inputs for an operation whose model is Lorentz only.

    Parameters
    ----------
    operation : str
        Public operation that rejects the input; it starts the error message.
    *sources : xr.Dataset, Mapping or None
        Fit parameters, fit results or option mappings. ``None`` is skipped.

    Raises
    ------
    ValueError
        If any source describes a ``lineshape='voigt'`` fit.
    """
    if any(source is not None and _describes_voigt_fit(source) for source in sources):
        raise ValueError(
            f"{operation} supports only lineshape='lorentz': its signal model "
            "has no Gaussian broadening, so a lineshape='voigt' fit would be "
            "evaluated with the wrong basis. Use a Lorentz fit (the default "
            "lineshape) for this step."
        )


def _require_same_lineshape(
    operation: str, fit_params: xr.Dataset, fit_results: xr.Dataset
) -> bool:
    """Require fit parameters and results of one lineshape and return it.

    Lorentz ``fit_t2s`` describe the full linewidth, Voigt ``fit_t2s`` only
    its Lorentz part, so results of one lineshape cannot be compared with or
    seed a fit of the other.

    Parameters
    ----------
    operation : str
        Public operation that checks the pair; it starts the error message.
    fit_params, fit_results : xr.Dataset
        Prepared fit parameters and fit results.

    Returns
    -------
    bool
        ``True`` for a Voigt pair, ``False`` for a Lorentz pair.

    Raises
    ------
    ValueError
        If exactly one of the two datasets describes a Voigt fit.
    """
    voigt = _describes_voigt_fit(fit_params)
    if voigt != _describes_voigt_fit(fit_results):
        lineshapes = [
            "voigt" if _describes_voigt_fit(ds) else "lorentz"
            for ds in (fit_params, fit_results)
        ]
        raise ValueError(
            f"{operation} requires fit_params and fit_results with the same "
            f"lineshape, got {lineshapes[0]!r} and {lineshapes[1]!r}. Refit "
            "with one lineshape before combining the two."
        )
    return voigt


def _prepare_lineshape_options(options: dict) -> dict:
    """Validate the lineshape choice and its Gaussian settings in place.

    The Lorentz default is not written back, so options without
    ``lineshape`` stay unchanged. Voigt mode writes its effective settings
    ``gauss_initial_fwhm_hz``, ``gauss_max_fwhm_hz`` and
    ``gauss_search_width_hz`` (all Gaussian FWHM values in Hz) back as floats.

    Parameters
    ----------
    options : dict
        Global fit options after the search and history defaults were applied.
        Modified in place.

    Returns
    -------
    dict
        The same ``options`` object.

    Raises
    ------
    ValueError
        If ``lineshape`` is not ``'lorentz'`` or ``'voigt'``, a ``gauss_*``
        key is unknown or given without ``lineshape='voigt'``, the Gaussian
        settings violate ``0 <= gauss_initial_fwhm_hz <= gauss_max_fwhm_hz``
        or ``gauss_search_width_hz >= 0``, or Voigt mode is combined with
        statistical detection or optimization history, which assume the
        Lorentz model.
    """
    if "initial_gauss_fwhm_hz" in options:
        raise ValueError(
            "initial_gauss_fwhm_hz is the prepared per-spectrum start map; pass "
            "the global start value as gauss_initial_fwhm_hz instead."
        )
    gauss_keys = sorted(key for key in options if str(key).startswith("gauss_"))
    unknown = [key for key in gauss_keys if key not in _GAUSS_OPTION_KEYS]
    if unknown:
        raise ValueError(
            f"Unknown Gaussian lineshape option(s) {unknown}; supported options "
            f"are {sorted(_GAUSS_OPTION_KEYS)}."
        )
    lineshape = options.get("lineshape", LINESHAPE_DEFAULT)
    if not isinstance(lineshape, str) or lineshape.strip().lower() not in LINESHAPES:
        raise ValueError(
            f"lineshape must be one of {list(LINESHAPES)}; got {lineshape!r}."
        )
    lineshape = lineshape.strip().lower()
    if "lineshape" in options:
        options["lineshape"] = lineshape
    if lineshape == "lorentz":
        if gauss_keys:
            raise ValueError(
                f"Option(s) {gauss_keys} require lineshape='voigt'; remove them "
                "or set lineshape='voigt'."
            )
        return options

    if options.get("detection_target") is not None:
        raise ValueError(
            "lineshape='voigt' does not support detection_target yet; the "
            "detection null model assumes Lorentz lines. Use lineshape='lorentz' "
            "or detection_target=None."
        )
    if options["store_optimization_history"]:
        raise ValueError(
            "lineshape='voigt' does not support store_optimization_history yet; "
            "set store_optimization_history=False."
        )
    for key, default in (
        ("gauss_initial_fwhm_hz", GAUSS_INITIAL_FWHM_HZ_DEFAULT),
        ("gauss_max_fwhm_hz", GAUSS_MAX_FWHM_HZ_DEFAULT),
        ("gauss_search_width_hz", GAUSS_SEARCH_WIDTH_HZ_DEFAULT),
    ):
        options[key] = _coerce_finite_scalar(
            options.get(key, default), parameter_name=key
        )
    if not 0 <= options["gauss_initial_fwhm_hz"] <= options["gauss_max_fwhm_hz"]:
        raise ValueError(
            "Voigt settings require 0 <= gauss_initial_fwhm_hz <= "
            "gauss_max_fwhm_hz (Gaussian FWHM in Hz)."
        )
    if options["gauss_search_width_hz"] < 0:
        raise ValueError(
            "gauss_search_width_hz must be non-negative; use zero to keep the "
            "Gaussian FWHM fixed."
        )
    return options


def _gaussian_decay(time_axis_s, gauss_fwhm_hz: float):
    """Return the Gaussian envelope ``exp(-(pi * G * t)**2 / (4 * ln 2))``.

    Its Fourier transform is a Gaussian line whose FWHM in Hz is exactly
    ``gauss_fwhm_hz``. The envelope is 1 at ``t = 0``, so complex amplitudes
    keep their meaning as line areas at model time zero. ``G = 0`` returns
    exactly 1.0 everywhere.

    Parameters
    ----------
    time_axis_s : numpy.ndarray or xarray.DataArray
        Sampling times in seconds relative to the model time origin.
    gauss_fwhm_hz : float or xarray.DataArray
        Non-negative Gaussian FWHM in Hz.

    Returns
    -------
    numpy.ndarray or xarray.DataArray
        Real envelope broadcast over the inputs.
    """
    return np.exp(-((np.pi * gauss_fwhm_hz * time_axis_s) ** 2) / (4.0 * np.log(2.0)))


def _voigt_fwhm_hz(lorentz_fwhm_hz, gauss_fwhm_hz):
    """Return the Olivero-Longbothum Voigt FWHM in Hz.

    ``f_V = 0.5346 f_L + sqrt(0.2166 f_L**2 + f_G**2)``, accurate to about
    0.02 %. Works element-wise on floats, NumPy arrays and xarray objects.

    Parameters
    ----------
    lorentz_fwhm_hz : float, numpy.ndarray or xarray.DataArray
        Non-negative Lorentz FWHM ``1 / (pi * T2)`` in Hz.
    gauss_fwhm_hz : float, numpy.ndarray or xarray.DataArray
        Non-negative Gaussian FWHM in Hz.

    Returns
    -------
    float, numpy.ndarray or xarray.DataArray
        Total Voigt FWHM in Hz.
    """
    return _OLIVERO_LINEAR * lorentz_fwhm_hz + np.sqrt(
        _OLIVERO_QUADRATIC * lorentz_fwhm_hz**2 + gauss_fwhm_hz**2
    )


MAGNITUDE_LORENTZ_FWHM_FACTOR = float(np.sqrt(3.0))
"""Ratio of magnitude-spectrum to absorption-spectrum FWHM of a Lorentz line.

``|1 / (gamma + i f)|`` falls to half its peak at ``f = sqrt(3) gamma``, so a
line with T2* has the magnitude FWHM ``sqrt(3) / (pi * T2*)``.
"""

_MAGNITUDE_FWHM_BISECTION_STEPS = 64


def _magnitude_fwhm_core(lorentz_fwhm_hz, gauss_fwhm_hz) -> np.ndarray:
    """Return magnitude-spectrum FWHM values for NumPy inputs.

    Parameters
    ----------
    lorentz_fwhm_hz : float or numpy.ndarray
        Non-negative Lorentz FWHM in Hz; NaN marks a missing value.
    gauss_fwhm_hz : float or numpy.ndarray
        Non-negative Gaussian FWHM in Hz, broadcastable against
        ``lorentz_fwhm_hz``; NaN marks a missing value.

    Returns
    -------
    numpy.ndarray
        Magnitude FWHM in Hz with the broadcast shape; NaN where an input is
        NaN.
    """
    lorentz, gauss = np.broadcast_arrays(
        np.asarray(lorentz_fwhm_hz, dtype=float),
        np.asarray(gauss_fwhm_hz, dtype=float),
    )
    width = MAGNITUDE_LORENTZ_FWHM_FACTOR * lorentz
    width = np.where(np.isnan(gauss), np.nan, width)
    voigt = np.isfinite(lorentz) & np.isfinite(gauss) & (gauss > 0.0)
    if not voigt.any():
        return width
    # The complex Voigt line is proportional to w((f + i gamma) / s) with
    # gamma = f_L / 2 and s = sigma * sqrt(2) = f_G / (2 sqrt(ln 2)). Its
    # magnitude falls monotonically with |f|, so bisection on x = f / s
    # finds the unique half-maximum point.
    scale = gauss[voigt] / (2.0 * np.sqrt(np.log(2.0)))
    y = 0.5 * lorentz[voigt] / scale
    half = 0.5 * np.abs(wofz(1j * y))
    low = np.zeros_like(y)
    high = MAGNITUDE_LORENTZ_FWHM_FACTOR * y + 2.0
    above = np.abs(wofz(high + 1j * y)) > half
    while above.any():
        high = np.where(above, 2.0 * high, high)
        above = np.abs(wofz(high + 1j * y)) > half
    for _ in range(_MAGNITUDE_FWHM_BISECTION_STEPS):
        middle = 0.5 * (low + high)
        above = np.abs(wofz(middle + 1j * y)) > half
        low = np.where(above, middle, low)
        high = np.where(above, high, middle)
    width[voigt] = 2.0 * scale * 0.5 * (low + high)
    return width


def _magnitude_fwhm_hz(lorentz_fwhm_hz, gauss_fwhm_hz):
    """Return the FWHM in Hz of the magnitude spectrum of a model line.

    The absorption widths reported as ``fit_fwhm_hz`` describe the real part
    of a phased line. Many spectra are read in magnitude mode, where the same
    line is wider. For a pure Lorentz line (``gauss_fwhm_hz == 0``) the
    magnitude FWHM is exactly ``sqrt(3) * lorentz_fwhm_hz``. For a Voigt line
    it is the half-maximum width of ``|w((f + i f_L / 2) / s)|`` with the
    Faddeeva function ``w`` and ``s = f_G / (2 sqrt(ln 2))``, found by a
    64-step vectorised bisection (relative accuracy far below 1e-12). Both
    widths describe an infinitely long line: truncation of the acquired FID
    is not included, and line broadening applied before the fit is already
    contained in the fitted widths.

    Parameters
    ----------
    lorentz_fwhm_hz : float, numpy.ndarray or xarray.DataArray
        Non-negative Lorentz FWHM ``1 / (pi * T2*)`` in Hz.
    gauss_fwhm_hz : float, numpy.ndarray or xarray.DataArray
        Non-negative Gaussian FWHM in Hz of the FID factor
        ``exp(-(pi * G * t)**2 / (4 ln 2))``; 0 for a Lorentz line.
        Broadcast against ``lorentz_fwhm_hz`` (by dimension name for xarray).

    Returns
    -------
    float, numpy.ndarray or xarray.DataArray
        Magnitude FWHM in Hz, NaN where an input is NaN. A float for scalar
        inputs; xarray inputs keep their labelled dimensions and coordinates
        and are evaluated lazily when dask-backed.
    """
    if isinstance(lorentz_fwhm_hz, xr.DataArray) or isinstance(
        gauss_fwhm_hz, xr.DataArray
    ):
        return xr.apply_ufunc(
            _magnitude_fwhm_core,
            lorentz_fwhm_hz,
            gauss_fwhm_hz,
            dask="parallelized",
            output_dtypes=[float],
        )
    width = _magnitude_fwhm_core(lorentz_fwhm_hz, gauss_fwhm_hz)
    if width.ndim == 0:
        return float(width)
    return width


def _lorentz_fwhm_from_voigt_hz(voigt_fwhm_hz, gauss_fwhm_hz):
    """Return the Lorentz FWHM that gives ``voigt_fwhm_hz`` with ``gauss_fwhm_hz``.

    Exact inverse of :func:`_voigt_fwhm_hz` for ``0 <= f_G < f_V``, written in
    the cancellation-free form
    ``f_L = 2 (f_V**2 - f_G**2) / (1.0692 f_V + sqrt(1.0692**2 f_V**2 -
    4 A (f_V**2 - f_G**2)))`` with ``A = 0.5346**2 - 0.2166``.

    Parameters
    ----------
    voigt_fwhm_hz : float, numpy.ndarray or xarray.DataArray
        Positive total FWHM in Hz.
    gauss_fwhm_hz : float, numpy.ndarray or xarray.DataArray
        Gaussian FWHM in Hz, ``0 <= gauss_fwhm_hz < voigt_fwhm_hz``. Callers
        must enforce this range; values outside it have no Lorentz solution.

    Returns
    -------
    float, numpy.ndarray or xarray.DataArray
        Positive Lorentz FWHM in Hz.
    """
    linear = 2.0 * _OLIVERO_LINEAR
    quadratic = _OLIVERO_LINEAR**2 - _OLIVERO_QUADRATIC
    excess = voigt_fwhm_hz**2 - gauss_fwhm_hz**2
    return (
        2.0
        * excess
        / (
            linear * voigt_fwhm_hz
            + np.sqrt((linear * voigt_fwhm_hz) ** 2 - 4.0 * quadratic * excess)
        )
    )


def _prepare_global_fit_options(fit_params: dict) -> dict:
    """Return global defaults and validated search settings without mutating input.

    Search counts are positive integers, zoom is positive, amplitude/ridge
    thresholds and peak separation are finite and non-negative, and
    ``search_order`` is one of ``SEARCH_ORDERS``. Unknown option
    names are rejected earlier by ``def_fit_params``; the established
    amplitude-solver aliases are preserved for downstream callers.
    """
    options = {
        "niter": NITER_DEFAULT,
        "npoints": NPOINTS_DEFAULT,
        "pseudo_inv": PSEUDO_INV_DEFAULT,
        "rep_fitting": REP_FITTING_DEFAULT,
        "show_tqdm": SHOW_TQDM_DEFAULT,
        "snr_threshold": SNR_THRESHOLD_DEFAULT,
        "use_sse_mag": USE_SSE_MAG_DEFAULT,
        "zoomfactor": ZOOMFACTOR_DEFAULT,
        "store_optimization_history": False,
        "t2_coupling_clip_start": False,
        "keep_fit_order": False,
        "search_order": SEARCH_ORDER_DEFAULT,
        "joint_refinement": JOINT_REFINEMENT_DEFAULT,
        **fit_params,
    }
    options["search_order"] = _normalize_search_order(options["search_order"])
    for key in (
        "store_optimization_history",
        "t2_coupling_clip_start",
        "keep_fit_order",
        "joint_refinement",
    ):
        if not isinstance(options[key], (bool, np.bool_)):
            raise ValueError(f"{key} must be boolean.")
    for key in ("niter", "npoints", "rep_fitting"):
        value = _coerce_finite_scalar(options[key], parameter_name=key)
        if (
            value < 1
            or not value.is_integer()
            or np.asarray(options[key]).dtype == bool
        ):
            raise ValueError(f"{key} must be a positive integer.")
        options[key] = int(value)

    if "uncertainty_bootstrap_samples" in options:
        value = _coerce_finite_scalar(
            options["uncertainty_bootstrap_samples"],
            parameter_name="uncertainty_bootstrap_samples",
        )
        if (
            value < 2
            or not value.is_integer()
            or np.asarray(options["uncertainty_bootstrap_samples"]).dtype == bool
        ):
            raise ValueError("uncertainty_bootstrap_samples must be an integer >= 2.")
        options["uncertainty_bootstrap_samples"] = int(value)

    if "uncertainty_seed" in options:
        value = _coerce_finite_scalar(
            options["uncertainty_seed"], parameter_name="uncertainty_seed"
        )
        if (
            value < 0
            or not value.is_integer()
            or np.asarray(options["uncertainty_seed"]).dtype == bool
        ):
            raise ValueError("uncertainty_seed must be a non-negative integer.")
        options["uncertainty_seed"] = int(value)

    if "uncertainty_confidence_level" in options:
        value = _coerce_finite_scalar(
            options["uncertainty_confidence_level"],
            parameter_name="uncertainty_confidence_level",
        )
        if not 0 < value < 1:
            raise ValueError(
                "uncertainty_confidence_level must lie strictly between zero and one."
            )
        options["uncertainty_confidence_level"] = value

    options["zoomfactor"] = _coerce_finite_scalar(
        options["zoomfactor"], parameter_name="zoomfactor"
    )
    if options["zoomfactor"] <= 0:
        raise ValueError("zoomfactor must be positive.")

    for key, default in (
        ("snr_threshold", SNR_THRESHOLD_DEFAULT),
        ("min_peak_separation_hz", DEFAULT_MIN_PEAK_SEPARATION_HZ),
        ("amp_ridge_lambda", 0.0),
    ):
        raw = options.get(key, default)
        if key == "snr_threshold" and raw is None:
            continue
        value = _coerce_finite_scalar(raw, parameter_name=key)
        if value < 0:
            raise ValueError(f"{key} must be non-negative.")
        options[key] = value
    options["amp_solver"] = _normalize_amplitude_solver(
        options.get("amp_solver"), options["pseudo_inv"]
    )
    _prepare_lineshape_options(options)
    if options.get("t2_coupling") is not None:
        if options.get("detection_target") is not None:
            raise ValueError(
                "t2_coupling does not support detection_target yet; the detection "
                "null model refits independent T2* values. Remove one of them."
            )
        options["t2_coupling"] = _normalize_t2_coupling(
            options["t2_coupling"], options.get("metabs")
        )
    _normalize_phase_band_options(options)
    from .model_comparison import prepare_detection_options

    return prepare_detection_options(options)


def _normalize_t2_coupling(
    coupling: object, metabolites: Sequence[str] | None
) -> dict[str, tuple[str, float | tuple[float, float]]] | None:
    """Return validated T2* coupling as ``{dependent: (leader, t2_ratio)}``.

    ``t2_ratio`` is a positive float (fixed ratio) or a ``(low, high)`` tuple
    with ``0 < low < high`` (ratio band); a band with ``low == high`` becomes
    that fixed ratio. A plain leader name means ``t2_ratio = 1.0``. An empty
    mapping returns None. Every name must be a fitted metabolite, a line cannot
    follow itself and a leader cannot itself be a dependent, so each coupling
    is applied in one step without chains or cycles.
    """
    if coupling is None:
        return None
    if not isinstance(coupling, dict):
        raise ValueError(
            "t2_coupling must be a dict {dependent: (leader, t2_ratio)} or "
            "{dependent: leader}, using metabolite names."
        )
    if not coupling:
        return None
    names = [str(name) for name in (metabolites or [])]
    normalized = {}
    for dependent, value in coupling.items():
        if isinstance(value, str):
            leader, ratio = value, 1.0
        else:
            try:
                leader, ratio = value
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"t2_coupling[{dependent!r}] must be a leader name or a "
                    "(leader, t2_ratio) pair."
                ) from exc
        dependent, leader = str(dependent), str(leader)
        if isinstance(ratio, (list, tuple, np.ndarray)):
            ratio = _normalize_t2_band(ratio, dependent)
        else:
            ratio = _coerce_finite_scalar(
                ratio, parameter_name=f"t2_coupling[{dependent!r}] t2_ratio"
            )
            if ratio <= 0:
                raise ValueError(
                    f"t2_coupling[{dependent!r}] t2_ratio must be positive."
                )
        unknown = [name for name in (dependent, leader) if name not in names]
        if unknown:
            raise ValueError(
                f"t2_coupling names {unknown} are not fitted metabolites {names}; "
                "add the metabolite first (add_metab_to_fit_params) or correct "
                "the name."
            )
        if dependent == leader:
            raise ValueError(f"t2_coupling: {dependent!r} cannot follow itself.")
        normalized[dependent] = (leader, ratio)
    chained = sorted(set(normalized) & {leader for leader, _ in normalized.values()})
    if chained:
        raise ValueError(
            f"t2_coupling: {chained} are both leader and dependent; couple every "
            "dependent directly to one leader."
        )
    return normalized


def _normalize_t2_band(band: object, dependent: str) -> float | tuple[float, float]:
    """Return a validated T2* ratio band ``(low, high)``, or ``low`` if both agree."""
    try:
        low, high = band
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"t2_coupling[{dependent!r}] t2_ratio band must be a (low, high) pair."
        ) from exc
    low, high = (
        _coerce_finite_scalar(
            value, parameter_name=f"t2_coupling[{dependent!r}] t2_ratio band"
        )
        for value in (low, high)
    )
    if not 0 < low <= high:
        raise ValueError(
            f"t2_coupling[{dependent!r}] t2_ratio band ({low}, {high}) must "
            "satisfy 0 < low <= high."
        )
    return low if low == high else (low, high)


def _normalize_phase_band_options(options: dict) -> None:
    """Validate the phase-band options in place, as ``{name: degrees}`` and seconds.

    Missing, None or empty tolerances remove both options. Otherwise every
    tolerance name must be a fitted metabolite with a finite value in
    ``[0, 90]`` degrees, ``phase_band_tau_s`` must be finite,
    the amplitude solver must be pinv and no detection target may be set.
    """
    tolerances = options.get("phase_band_tolerance_deg")
    if tolerances is None or (isinstance(tolerances, dict) and not tolerances):
        if options.get("phase_band_tau_s") is not None:
            raise ValueError(
                "phase_band_tau_s is only used with phase_band_tolerance_deg; "
                "add tolerances for the lines that follow the phase law or "
                "remove phase_band_tau_s."
            )
        options.pop("phase_band_tolerance_deg", None)
        options.pop("phase_band_tau_s", None)
        return
    if not isinstance(tolerances, dict):
        raise ValueError(
            "phase_band_tolerance_deg must be a dict {metabolite: tolerance in "
            "degrees}; lines that are not listed keep free amplitudes."
        )
    names = [str(name) for name in (options.get("metabs") or [])]
    normalized = {}
    for name, value in tolerances.items():
        name = str(name)
        if name not in names:
            raise ValueError(
                f"phase_band_tolerance_deg name {name!r} is not a fitted "
                f"metabolite {names}; add the metabolite first or correct the name."
            )
        tolerance = _coerce_finite_scalar(
            value, parameter_name=f"phase_band_tolerance_deg[{name!r}]"
        )
        if not 0 <= tolerance <= _PHASE_BAND_MAX_TOLERANCE_DEG:
            raise ValueError(
                f"phase_band_tolerance_deg[{name!r}]={tolerance} must lie in "
                f"[0, {_PHASE_BAND_MAX_TOLERANCE_DEG}] degrees; leave the line "
                "out of the mapping to keep its amplitude free."
            )
        normalized[name] = tolerance
    if options.get("phase_band_tau_s") is None:
        raise ValueError(
            "phase_band_tolerance_deg requires phase_band_tau_s, the time in "
            "seconds from the common signal origin to t = 0 of "
            "sampling_time_axis_s."
        )
    tau_s = _coerce_finite_scalar(
        options["phase_band_tau_s"], parameter_name="phase_band_tau_s"
    )
    if options["amp_solver"] != "pinv":
        raise ValueError(
            "phase_band_tolerance_deg requires amp_solver='pinv'; the phase-band "
            f"solver replaces the free pinv amplitudes, not {options['amp_solver']!r}."
        )
    if options.get("detection_target") is not None:
        raise ValueError(
            "phase_band_tolerance_deg does not support detection_target yet; the "
            "detection null model assumes free amplitudes. Remove one of them."
        )
    options["phase_band_tolerance_deg"] = normalized
    options["phase_band_tau_s"] = tau_s


def _t2_coupling_pairs(fit_params: dict) -> list[tuple[int, int | None, float]]:
    """Return fixed ``(dependent, leader, t2_ratio)`` indices into ``metabs``.

    Ratio bands are excluded (see ``_t2_band_pairs``). Only dependents present
    in ``metabs`` are returned. The leader index is None when the leader is
    absent from ``metabs``; such a dependent keeps its current T2*. Fit subsets
    never reach this case because ``_promote_t2_coupling`` hands an absent
    leader's group to a present line.
    """
    coupling = fit_params.get("t2_coupling") or {}
    names = [str(name) for name in (fit_params.get("metabs") or [])]
    return [
        (
            names.index(dependent),
            names.index(leader) if leader in names else None,
            float(ratio),
        )
        for dependent, (leader, ratio) in coupling.items()
        if dependent in names and not isinstance(ratio, tuple)
    ]


def _t2_band_pairs(fit_params: dict) -> list[tuple[int, int, float, float]]:
    """Return ``(dependent, leader, low, high)`` indices of T2* ratio bands.

    A banded dependent is searched like an independent line, but every T2*
    state must keep ``low <= T2*(dependent) / T2*(leader) <= high``. Pairs
    whose dependent or leader is absent from ``metabs`` are not returned.
    """
    coupling = fit_params.get("t2_coupling") or {}
    names = [str(name) for name in (fit_params.get("metabs") or [])]
    return [
        (names.index(dependent), names.index(leader), *ratio)
        for dependent, (leader, ratio) in coupling.items()
        if isinstance(ratio, tuple) and dependent in names and leader in names
    ]


def _inside_t2_bands(t2s: np.ndarray, bands: list) -> bool:
    """Return whether every T2* ratio of ``bands`` lies in its band.

    A relative tolerance of 1e-12 accepts states whose ratio sits on a band
    edge up to floating-point rounding of the multiplication that set it.
    """
    return all(
        low * t2s[leader] * (1 - 1e-12)
        <= t2s[dependent]
        <= high * t2s[leader] * (1 + 1e-12)
        for dependent, leader, low, high in bands
    )


def _apply_t2_coupling(t2s: np.ndarray, fit_params: dict) -> np.ndarray:
    """Return a copy of T2* values in seconds with every dependent set from its leader."""
    t2s = np.array(t2s, dtype=float, copy=True)
    for dependent, leader, ratio in _t2_coupling_pairs(fit_params):
        if leader is not None:
            t2s[dependent] = ratio * t2s[leader]
    return t2s


def _t2_coupling_groups(fit_params: dict) -> list[np.ndarray]:
    """Return the solver indices of each ``t2_coupling`` group in ascending order.

    A group is one leader with all its dependents (fixed ratios and ratio
    bands) that are present in ``metabs``. Dependents of an absent leader form
    no group.
    """
    groups: dict[int, set[int]] = {}
    for dependent, leader, _ in _t2_coupling_pairs(fit_params):
        if leader is not None:
            groups.setdefault(leader, {leader}).add(dependent)
    for dependent, leader, _, _ in _t2_band_pairs(fit_params):
        groups.setdefault(leader, {leader}).add(dependent)
    return [np.array(sorted(members)) for _, members in sorted(groups.items())]


# Per-line frequency settings that move with a line when coupled lines
# exchange their names (see ``_coupled_line_order``).
_FREQUENCY_WINDOW_KEYS = (
    "metabs_freq_ranges",
    "freq_range_lower_bound_hz",
    "freq_range_upper_bound_hz",
)


def _frequency_position_rank(fit_params: dict) -> np.ndarray:
    """Return the rank of every metabolite in the frequency position order.

    The position order is ascending start frequency, equal starts ordered by
    metabolite name. ``fit_data_pseudo_inv`` supplies the order of the mean
    start frequencies over all spectra in the private key
    ``"_frequency_position_rank"``, the same order its relabelling of uncoupled
    lines uses; without that key the order follows this call's
    ``metabs_freqs_hz``.

    Parameters
    ----------
    fit_params : dict
        Single-spectrum options with ``metabs`` and ``metabs_freqs_hz`` (Hz),
        both in solver order.

    Returns
    -------
    numpy.ndarray
        Integer rank per metabolite, shape ``(n_lines,)`` in solver order; a
        lower rank is a lower frequency position. Only the relative order of
        the values is meaningful.
    """
    rank = fit_params.get("_frequency_position_rank")
    if rank is not None:
        return np.asarray(rank)
    names = [str(name) for name in fit_params["metabs"]]
    starts_hz = np.asarray(np.real(fit_params["metabs_freqs_hz"]), dtype=float)
    order = sorted(range(len(names)), key=lambda i: (starts_hz[i], names[i]))
    return np.argsort(order)


def _coupled_frequency_positions(fit_params: dict) -> list[np.ndarray]:
    """Return each ``t2_coupling`` group's solver indices in position order.

    Every coupling role (leader, fixed or banded dependent), together with the
    name, T2* value, T2* bounds and phase-band tolerance of that index, belongs
    to a frequency position: the k-th entry of a group always holds the line
    with the k-th lowest frequency of the group (``_coupled_line_order``).

    Parameters
    ----------
    fit_params : dict
        Single-spectrum options with ``metabs``, ``metabs_freqs_hz`` and the
        optional ``t2_coupling`` mapping.

    Returns
    -------
    list of numpy.ndarray
        One integer array per coupling group, ordered by
        ``_frequency_position_rank``; empty without ``t2_coupling``.
    """
    groups = _t2_coupling_groups(fit_params)
    if not groups:
        return []
    rank = _frequency_position_rank(fit_params)
    return [group[np.argsort(rank[group], kind="stable")] for group in groups]


def _coupled_line_order(
    frequencies_hz: np.ndarray, positions: list[np.ndarray]
) -> np.ndarray:
    """Return the line permutation that puts coupled lines on their positions.

    Within each group the line with the k-th lowest frequency moves to the
    k-th position of ``positions``; equal frequencies keep their slots. Lines
    outside the groups stay in place. ``frequencies_hz[order]`` is then the
    state in which every name holds the line at its position; the same
    permutation moves the per-line frequency windows
    (``_move_frequency_windows``), so a line keeps its own search width and
    bounds while it changes its name.

    Parameters
    ----------
    frequencies_hz : numpy.ndarray
        Line frequencies in Hz, shape ``(n_lines,)``, one per slot.
    positions : list of numpy.ndarray
        Coupling groups in position order from ``_coupled_frequency_positions``.

    Returns
    -------
    numpy.ndarray
        Integer permutation of shape ``(n_lines,)``: slot ``k`` receives the
        line from slot ``order[k]``. The identity if every group is already in
        position order.
    """
    order = np.arange(len(frequencies_hz))
    for group in positions:
        order[group] = group[np.argsort(frequencies_hz[group], kind="stable")]
    return order


def _move_frequency_windows(options: dict, line_order: np.ndarray) -> None:
    """Move every line's frequency search width and bounds with the line.

    Replaces the arrays under ``_FREQUENCY_WINDOW_KEYS`` in ``options`` (a
    working copy owned by the caller) by their permuted copies; the arrays
    themselves are not modified.

    Parameters
    ----------
    options : dict
        Working single-spectrum options of one alternating search.
    line_order : numpy.ndarray
        Permutation from ``_coupled_line_order``.
    """
    for key in _FREQUENCY_WINDOW_KEYS:
        options[key] = np.asarray(np.real(options[key]), dtype=float)[line_order]


def _normalize_search_order(search_order: object) -> str:
    """Return ``search_order`` as one of ``SEARCH_ORDERS``.

    Surrounding spaces and letter case are ignored.

    Raises
    ------
    ValueError
        If the value is not a string naming one of ``SEARCH_ORDERS``.
    """
    if (
        not isinstance(search_order, str)
        or search_order.strip().lower() not in SEARCH_ORDERS
    ):
        raise ValueError(
            f"search_order must be one of {list(SEARCH_ORDERS)}; got {search_order!r}."
        )
    return search_order.strip().lower()


def _search_visit_order(
    temporal: np.ndarray,
    frequencies_hz: np.ndarray,
    t2s_s: np.ndarray,
    amplitude_options: dict,
    fit_params: dict,
) -> np.ndarray | None:
    """Return the line order of one grid step, or None for metabolite order.

    ``search_order='metabolite'`` and single-line fits return None without
    solving anything; the step then visits the lines in slot order.
    ``'amplitude'`` (the default) solves the complex amplitudes once for the
    entry state and returns the slots sorted by decreasing norm of their
    fitted component FID, the line's share of the model signal. The norm,
    not the amplitude alone, also ranks a broad line of large area below a
    narrow line that carries more signal. Equal norms keep slot order, so
    the result is deterministic. The order is fixed for the whole step.

    Parameters
    ----------
    temporal : numpy.ndarray
        Measured FID of the step, shape ``(n_samples,)``.
    frequencies_hz, t2s_s : numpy.ndarray
        Entry state of the step, one value per slot, in Hz and seconds.
    amplitude_options : dict
        Keyword arguments for ``solve_for_complex_amplitudes`` from
        ``_prepare_grid_search``.
    fit_params : dict
        Single-spectrum options; only ``search_order`` is read.

    Returns
    -------
    numpy.ndarray or None
        Integer slot indices of shape ``(n_lines,)`` in visiting order, or
        None for metabolite order.
    """
    search_order = _normalize_search_order(
        fit_params.get("search_order", SEARCH_ORDER_DEFAULT)
    )
    if search_order == "metabolite" or len(frequencies_hz) < 2:
        return None
    components, _ = solve_for_complex_amplitudes(
        values=temporal,
        freq_hz=frequencies_hz,
        t2_star_s=t2s_s,
        **amplitude_options,
    )
    return np.argsort(-np.linalg.norm(components, axis=0), kind="stable")


def _t2_search_moves(
    n_lines: int,
    coupling: list[tuple],
    bands: list[tuple],
    visit_order: Sequence[int] | None = None,
) -> list[tuple[int, bool]]:
    """Return the coordinate moves of one grid iteration in search order.

    Lines are visited in ``visit_order`` (from ``_search_visit_order``), or in
    slot order when it is None. Each line except fixed dependents is searched
    once with all its dependents following (``True``). A leader with ratio
    bands is then searched once more on its own (``False``): its fixed
    dependents follow, its banded dependents keep their T2* and the bands
    limit its candidates. Without this second move a band leader could change
    its T2* only together with its banded dependents. ``coupling`` and
    ``bands`` come from ``_t2_coupling_pairs`` and ``_t2_band_pairs``; both
    are empty in the frequency search, which then visits every line once.
    """
    fixed = {dependent for dependent, _, _ in coupling}
    band_leaders = {leader for _, leader, _, _ in bands}
    moves = []
    for line in range(n_lines) if visit_order is None else visit_order:
        line = int(line)
        if line in fixed:
            continue
        moves.append((line, True))
        if line in band_leaders:
            moves.append((line, False))
    return moves


def _t2_move_constraints(
    line: int,
    bands_follow: bool,
    t2s: np.ndarray,
    coupling: list[tuple],
    bands: list[tuple],
) -> tuple[list[tuple[int, float]], list[tuple]]:
    """Return the followers and checked ratio bands of one T2* move.

    ``followers`` lists ``(dependent, ratio)`` pairs whose T2* is set to
    ``ratio`` times each candidate of ``line``: fixed dependents always, banded
    dependents with their current ratio in ``t2s`` only if ``bands_follow``.
    The checked bands are those in which ``line`` is the dependent and, for a
    move without following bands, those it leads.
    """
    followers = [
        (dependent, ratio) for dependent, leader, ratio in coupling if leader == line
    ]
    checked = [band for band in bands if band[0] == line]
    if bands_follow:
        followers += [
            (dependent, t2s[dependent] / t2s[line])
            for dependent, leader, _, _ in bands
            if leader == line
        ]
    else:
        checked += [band for band in bands if band[1] == line]
    return followers, checked


def _promote_t2_coupling(
    fit_params: dict, active: np.ndarray, t2_search_range_s: np.ndarray
) -> tuple[dict, np.ndarray]:
    """Return options and T2* search widths for fitting only the ``active`` lines.

    A leader that is not active (skipped or amplitude-filtered) hands its group
    to the first active dependent with a fixed t2_ratio, in
    ``fit_params['metabs']`` order. This new leader searches with the old
    leader's T2* search width times its own t2_ratio, so a frozen leader keeps
    the group frozen. Every other dependent of the group follows the new leader
    with its t2_ratio or ratio band divided by the new leader's t2_ratio, which
    keeps all T2* ratios within the group. Without an active fixed dependent,
    the active dependents become independent. Groups whose leader is active,
    or that have no active member, are unchanged.

    Applying the returned coupling to full-length T2* arrays sets every
    dependent, fitted or not, from the line that was actually fitted; the
    inactive old leader keeps its own value. Neither input is modified.
    """
    coupling = fit_params.get("t2_coupling") or {}
    names = [str(name) for name in fit_params["metabs"]]
    active_names = {name for name, keep in zip(names, active, strict=True) if keep}
    t2_search_range_s = np.array(t2_search_range_s, dtype=float, copy=True)
    promoted = dict(coupling)
    for leader in dict.fromkeys(leader for leader, _ in coupling.values()):
        group = [name for name in names if coupling.get(name, (None,))[0] == leader]
        present = [name for name in group if name in active_names]
        if leader in active_names or leader not in names or not present:
            continue
        fixed = [name for name in present if not isinstance(coupling[name][1], tuple)]
        if not fixed:
            for dependent in present:
                del promoted[dependent]
            continue
        new_leader, new_ratio = fixed[0], coupling[fixed[0]][1]
        del promoted[new_leader]
        for dependent in group:
            if dependent != new_leader:
                ratio = coupling[dependent][1]
                promoted[dependent] = (
                    new_leader,
                    tuple(value / new_ratio for value in ratio)
                    if isinstance(ratio, tuple)
                    else ratio / new_ratio,
                )
        t2_search_range_s[names.index(new_leader)] = (
            new_ratio * t2_search_range_s[names.index(leader)]
        )
    return {**fit_params, "t2_coupling": promoted or None}, t2_search_range_s


def _coerce_metabolite_values(
    values: object, num_metabs: int, *, parameter_name: str
) -> np.ndarray:
    """Return finite real values from a scalar or one entry per metabolite.

    Scalars apply to every metabolite. Sequences must have exactly one entry
    per metabolite; unlike frequency input, missing entries are not supported.
    """
    try:
        if np.iscomplexobj(values):
            raise ValueError("Complex values are not real fit parameters.")
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{parameter_name} must contain finite real values.") from exc
    if array.ndim == 0:
        array = np.full(num_metabs, array.item())
    if array.ndim != 1 or array.size != num_metabs:
        raise ValueError(
            f"{parameter_name} must be a scalar or a one-dimensional sequence "
            f"of length {num_metabs}, in metabolite order."
        )
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{parameter_name} must contain finite real values.")
    return array


def _update_frequency_bounds(fit_params: xr.Dataset) -> xr.Dataset:
    """Return parameter maps with frequency bounds at center +/- half full width.

    Use the same calculation when preparing, extending or refining a fit.
    Assigning the derived variables preserves existing metadata and coordinates
    without changing the input dataset.
    """
    half_width = fit_params["metabs_freq_ranges"] / 2
    return fit_params.assign(
        freq_range_lower_bound_hz=fit_params["initial_freqs_hz"] - half_width,
        freq_range_upper_bound_hz=fit_params["initial_freqs_hz"] + half_width,
    )


# Stored bounds closer than this to start +/- half range are the same bounds;
# shifting start and bounds separately by one offset differs by ~1e-13 Hz.
_FREQUENCY_BOUND_TOLERANCE_HZ = 1e-6


def _derive_frequency_bounds(fit_params: xr.Dataset, *, operation: str) -> xr.Dataset:
    """Return parameters whose frequency bounds follow start and range.

    ``initial_freqs_hz`` and the full search width ``metabs_freq_ranges`` own
    the frequency bounds. Stored ``freq_range_lower_bound_hz`` and
    ``freq_range_upper_bound_hz`` that differ from them, for example after the
    start was edited in place, are replaced and a ``UserWarning`` names
    ``operation`` and the lines. Parameters without start or range maps are
    returned unchanged, so callers report the missing maps themselves. The
    input dataset is not modified.
    """
    if "initial_freqs_hz" not in fit_params or "metabs_freq_ranges" not in fit_params:
        return fit_params
    derived = _update_frequency_bounds(fit_params)
    changed: set[str] = set()
    for name in ("freq_range_lower_bound_hz", "freq_range_upper_bound_hz"):
        if name not in fit_params:
            continue
        stored = fit_params[name]
        expected = derived[name]
        differs = (np.abs(stored - expected) > _FREQUENCY_BOUND_TOLERANCE_HZ) | (
            stored.isnull() != expected.isnull()
        )
        # The start map has the metabolite dimension, so the comparison does.
        other_dims = [dim for dim in differs.dims if dim != "metabolite"]
        per_line = differs.any(dim=other_dims) if other_dims else differs
        changed.update(str(m) for m in per_line.metabolite.values[per_line.values])
    if changed:
        warnings.warn(
            f"{operation}: stored frequency bounds of {sorted(changed)} differ "
            "from initial_freqs_hz -/+ metabs_freq_ranges / 2 and are replaced "
            "by them. Change the start or the range, not the bounds; the bounds "
            "are derived from both.",
            UserWarning,
            stacklevel=3,
        )
    return derived


def _coerce_skip_fitting_mask(values: object, num_metabs: int) -> np.ndarray:
    """Return one boolean exclusion value per metabolite.

    ``None`` means no exclusions. A scalar applies to every metabolite; a
    sequence must have exactly one value per metabolite. Accept booleans or 0/1,
    rejecting other numbers and non-finite values before they can remove peaks.
    Preparation and direct single-spectrum fitting use the same rules.
    """
    mask = _coerce_metabolite_values(
        False if values is None else values, num_metabs, parameter_name="skip_fitting"
    )
    if not np.all(np.isin(mask, [0, 1])):
        raise ValueError("skip_fitting must contain booleans or 0/1 values.")
    return mask.astype(bool)


def fid_basefunc(
    freq_hz: float | list[float],
    t2_star_s: float | list[float],
    time_axis_s: np.ndarray,
    *,
    gauss_fwhm_hz: float | None = None,
) -> np.ndarray:
    """Calculate basis FID signals.

    Defined as ``s(t) = exp(1j * 2 * pi * f * t) * exp(-t / T2*)`` for each
    metabolite. With ``gauss_fwhm_hz`` every signal is additionally multiplied
    by the shared Gaussian envelope ``exp(-(pi * G * t)**2 / (4 * ln 2))``,
    giving Voigt lines whose Gaussian FWHM is ``G`` Hz.

    Parameters
    ----------
    freq_hz : float or list of float
        Frequencies in Hz for each metabolite.
    t2_star_s : float or list of float
        T2* relaxation times in seconds for each metabolite; the Lorentz part
        when ``gauss_fwhm_hz`` is given. Non-positive values give zero signals.
    time_axis_s : numpy.ndarray
        Time axis in seconds for the FID signal.
    gauss_fwhm_hz : float or None, optional
        Non-negative Gaussian FWHM in Hz shared by all signals. ``None`` (the
        default) evaluates the Lorentz basis without any extra operation.

    Returns
    -------
    numpy.ndarray
        Complex FID signals for each metabolite, shape (len(freq_hz), len(time_axis_s)).

    Raises
    ------
    ValueError
        If a supplied Gaussian width is negative or non-finite.

    Notes
    -----
    The basis has positive complex frequency rotation and the supplied time
    orientation. Dtype follows NumPy input promotion; no storage-precision cast
    is applied. Matching frequency/decay vectors are expected; the historical
    pairing stops at the shorter vector. Evaluation is eager and deterministic,
    with no input mutation or cache.

    Examples
    --------
    >>> fid_basefunc([40.0], [0.025], np.array([0.0, 0.001])).shape
    (1, 2)"""
    if not isinstance(freq_hz, (list, np.ndarray)):
        freq_hz = [freq_hz]
    if not isinstance(t2_star_s, (list, np.ndarray)):
        t2_star_s = [t2_star_s]

    sig = []
    for f, T2 in zip(freq_hz, t2_star_s, strict=False):
        if T2 <= 0:
            sig.append(np.zeros_like(time_axis_s))
        else:
            sig.append(
                np.exp(1j * 2.0 * np.pi * f * time_axis_s) * np.exp(-time_axis_s / T2)
            )

    if gauss_fwhm_hz is None:
        return np.array(sig)
    return np.array(sig) * _gaussian_decay(np.asarray(time_axis_s), gauss_fwhm_hz)


def _normalize_amplitude_solver(
    amp_solver: str | None, pseudo_inv: bool = PSEUDO_INV_DEFAULT
) -> str:
    """Return the canonical solver name, preserving aliases and legacy defaults.

    An explicit solver overrides ``pseudo_inv``. Without one, use ``pinv``
    for true and ``normal_eq`` for false. Names ignore surrounding whitespace
    and case. Preparation and direct amplitude solving share this validation.
    """
    if amp_solver is None:
        return "pinv" if pseudo_inv else "normal_eq"
    aliases = {
        "pinv": "pinv",
        "pseudo_inv": "pinv",
        "pseudoinv": "pinv",
        "normal_eq": "normal_eq",
        "normal": "normal_eq",
        "ridge": "ridge",
        "tikhonov": "ridge",
    }
    normalized = str(amp_solver).strip().lower()
    if normalized not in aliases:
        raise ValueError(
            "Unknown amp_solver "
            f"'{amp_solver}'. Expected one of "
            "{'pinv', 'normal_eq', 'ridge'}."
        )
    return aliases[normalized]


def _prepare_amplitude_solver_kwargs(fit_params: dict) -> dict:
    """Return shared solver arguments for scoring an already temporal signal.

    Use the sampling axis and global options from the single-spectrum fit.
    Retain the numerical entry points' legacy scalar fallback and clamping for
    ridge lambda; public parameter preparation validates it strictly beforehand.
    Neither this function nor solver-name normalization changes ``fit_params``.
    In Voigt mode the single-spectrum fit stores the current shared Gaussian
    FWHM under the private key ``"_gauss_fwhm_hz"``; it is passed on as
    ``gauss_fwhm_hz`` and omitted entirely for Lorentz fits.

    With ``phase_band_tolerance_deg``, the tolerances are passed on as one
    value per entry of ``fit_params['metabs']`` (``inf`` for free lines),
    together with ``phase_band_tau_s``. Building the array by name keeps fit
    subsets of skipped or amplitude-filtered lines consistent.
    """
    pseudo_inv = fit_params.get("pseudo_inv", PSEUDO_INV_DEFAULT)
    solver_kwargs = {
        "time_axis_s": fit_params["sampling_time_axis_s"],
        "signal_domain": "temporal",
        "pseudo_inv": pseudo_inv,
        "amp_solver": _normalize_amplitude_solver(
            fit_params.get("amp_solver"), pseudo_inv
        ),
        "amp_ridge_lambda": _get_scalar_fit_param(
            fit_params, "amp_ridge_lambda", 0.0, min_value=0.0
        ),
    }
    if "_gauss_fwhm_hz" in fit_params:
        solver_kwargs["gauss_fwhm_hz"] = fit_params["_gauss_fwhm_hz"]
    return {**solver_kwargs, **_phase_band_solver_kwargs(fit_params)}


def _phase_band_solver_kwargs(fit_params: dict) -> dict:
    """Return the phase-band arguments of ``solve_for_complex_amplitudes``.

    The ``{name: degrees}`` mapping becomes one tolerance per entry of
    ``fit_params['metabs']``, ``inf`` for lines that are not listed, so any
    subset of metabolites receives its own tolerances. Without tolerances the
    result is empty. ``fit_params`` is not modified.
    """
    tolerances = fit_params.get("phase_band_tolerance_deg")
    if not tolerances:
        return {}
    return {
        "phase_band_tolerance_deg": np.array(
            [tolerances.get(str(name), np.inf) for name in fit_params["metabs"]],
            dtype=float,
        ),
        "phase_band_tau_s": fit_params.get("phase_band_tau_s"),
    }


def solve_for_complex_amplitudes(
    values: np.ndarray,
    freq_hz: np.ndarray,
    t2_star_s: np.ndarray,
    time_axis_s: np.ndarray,
    signal_domain: str = "spectral",
    pseudo_inv: bool = PSEUDO_INV_DEFAULT,
    amp_solver: str | None = None,
    amp_ridge_lambda: float = 0.0,
    *,
    gauss_fwhm_hz: float | None = None,
    phase_band_tolerance_deg: np.ndarray | None = None,
    phase_band_tau_s: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit complex amplitudes for fixed metabolite frequencies and T2* values.

    Parameters
    ----------
    values : numpy.ndarray
        One-dimensional FID or centered spectrum, with one value per sample.
    freq_hz, t2_star_s : numpy.ndarray
        Frequency in Hz and positive T2* in seconds, one per metabolite.
    time_axis_s : numpy.ndarray
        Sampling times in seconds, matching the input length.
    signal_domain : str, default "spectral"
        ``"spectral"`` input is converted using ``ifft(ifftshift(values))``;
        ``"temporal"`` input is already an FID. Fitting always uses the FID.
    pseudo_inv : bool, default True
        Select ``pinv`` when true or ``normal_eq`` when false, unless an explicit
        ``amp_solver`` is supplied.
    amp_solver : str or None, default None
        ``"pinv"``, ``"normal_eq"`` or ``"ridge"``. Aliases ``"pseudo_inv"``
        and ``"pseudoinv"`` select pinv; ``"normal"`` selects normal_eq;
        ``"tikhonov"`` selects ridge. Names ignore case and outer whitespace.
    amp_ridge_lambda : float, default 0.0
        Ridge penalty; ignored by other solvers. Direct calls retain historical
        clamping of negative values to zero. Parameter preparation requires a
        finite, non-negative value.
    gauss_fwhm_hz : float or None, optional
        Shared Gaussian FWHM in Hz of a Voigt basis; ``t2_star_s`` is then the
        Lorentz part. ``None`` (the default) uses the Lorentz basis.
    phase_band_tolerance_deg : numpy.ndarray or None, optional
        Phase tolerance in degrees, one per metabolite: finite values in
        ``[0, 90]`` degrees put the line under the phase law
        described in the Notes, ``inf`` keeps its amplitude free. ``None``
        (the default) solves free amplitudes with the selected solver.
    phase_band_tau_s : float or None, optional
        Time in seconds from the common signal origin to ``t = 0`` of
        ``time_axis_s``; positive when the origin lies before it. Required
        with, and only used by, ``phase_band_tolerance_deg``.

    Returns
    -------
    fitted_components : numpy.ndarray
        Complex temporal component signals, shape ``(samples, metabolites)``,
        including fitted amplitudes. Sum along axis 1 for the total fitted FID,
        also when the input was a spectrum.
    complex_amplitudes : numpy.ndarray
        Complex coefficients in metabolite order, shape ``(metabolites,)``.

    Raises
    ------
    ValueError
        If the solver name, Gaussian width or phase-band options are invalid,
        the phase band is combined with a solver other than pinv, or
        basis/data shapes do not agree.
    numpy.linalg.LinAlgError
        If the numerical decomposition cannot produce a finite solution.

    Notes
    -----
    With basis matrix ``X``, pinv solves ``pinv(X) @ y``. Both ``normal_eq``
    and ridge use the Hermitian transpose required for complex least squares.
    This corrects the historical plain-transpose ``normal_eq`` result.
    Normal equations square the condition number; pinv remains the default.
    Ridge uses the conjugate
    transpose in ``(X.conj().T @ X + lambda * I) @ beta = X.conj().T @ y`` and
    falls back to a pseudoinverse of that matrix if solving raises LinAlgError.

    With finite phase-band tolerances the amplitudes of those lines follow
    ``a_k = A_k * exp(1j * (phi0 + 2*pi*f_k*phase_band_tau_s + delta_k))``
    with ``A_k >= 0`` and ``abs(delta_k) <= tolerance_k``; ``phi0`` is shared
    and chosen per call. The solver returns the least-squares amplitudes under
    this constraint: free lines are projected out, the banded amplitudes come
    from non-negative least squares on rays at ``-tolerance, 0, +tolerance``
    (one ray at zero tolerance), ``phi0`` is refined from a 24-point grid by a
    bounded scalar search and the free amplitudes are then solved by pinv on
    the remaining signal. Without finite tolerances the result equals pinv.

    Basis solving uses complex128, independent of input storage precision. The
    returned amplitudes and fitted components retain input signal units. This
    operation is eager, deterministic for fixed library/options, preserves
    inputs and retains no result cache.

    Examples
    --------
    >>> components, areas = solve_for_complex_amplitudes(
    ...     measured_fid, frequencies_hz, decay_times_s, times_s,
    ...     signal_domain="temporal"
    ... )  # doctest: +SKIP"""
    solver = _normalize_amplitude_solver(amp_solver, pseudo_inv)
    sim_signal = fid_basefunc(
        freq_hz=freq_hz,
        t2_star_s=t2_star_s,
        time_axis_s=time_axis_s,
        gauss_fwhm_hz=gauss_fwhm_hz,
    )

    if signal_domain == "spectral":
        meas_signal = np.fft.ifft(np.fft.ifftshift(values))
    else:
        meas_signal = values

    X = np.asarray(sim_signal.T, dtype=np.complex128)
    y = np.asarray(meas_signal, dtype=np.complex128)
    tolerance_rad = _phase_band_tolerance_rad(
        phase_band_tolerance_deg, phase_band_tau_s, solver, X.shape[1]
    )

    if tolerance_rad is not None and np.any(np.isfinite(tolerance_rad)):
        beta = _solve_phase_band_amplitudes(
            X,
            y,
            np.asarray(np.real(freq_hz), dtype=float),
            tolerance_rad,
            tau_s=float(phase_band_tau_s),
        )
    elif solver == "pinv":
        beta = np.linalg.pinv(X).dot(y)
    elif solver == "normal_eq":
        beta = np.linalg.pinv(X.conj().T.dot(X)).dot(X.conj().T).dot(y)
    else:  # ridge; the solver name has already been validated above.
        lam = max(float(amp_ridge_lambda), 0.0)
        gram = X.conj().T.dot(X)
        if lam > 0.0:
            gram = gram + lam * np.eye(gram.shape[0], dtype=gram.dtype)
        rhs = X.conj().T.dot(y)
        try:
            beta = np.linalg.solve(gram, rhs)
        except np.linalg.LinAlgError:
            beta = np.linalg.pinv(gram).dot(rhs)
    final_signal = sim_signal.T * beta
    return final_signal, beta


def _phase_band_tolerance_rad(
    tolerance_deg: np.ndarray | None,
    tau_s: float | None,
    solver: str,
    n_metabs: int,
) -> np.ndarray | None:
    """Return validated phase-band tolerances in radians, or None without a band.

    Free lines keep ``inf``. Both options must be given together, with one
    tolerance per metabolite, and only the pinv solver accepts them.
    """
    if tolerance_deg is None and tau_s is None:
        return None
    if tolerance_deg is None or tau_s is None:
        raise ValueError(
            "solve_for_complex_amplitudes: phase_band_tolerance_deg and "
            "phase_band_tau_s must be given together."
        )
    if solver != "pinv":
        raise ValueError(
            f"solve_for_complex_amplitudes: the phase band requires amp_solver="
            f"'pinv', not {solver!r}."
        )
    if not np.isfinite(float(tau_s)):
        raise ValueError("phase_band_tau_s must be finite, in seconds.")
    tolerance_deg = np.asarray(tolerance_deg, dtype=float)
    if tolerance_deg.shape != (n_metabs,):
        raise ValueError(
            f"phase_band_tolerance_deg needs one value per metabolite "
            f"({n_metabs}), got shape {tolerance_deg.shape}."
        )
    finite = np.isfinite(tolerance_deg)
    if np.any(~finite & (tolerance_deg != np.inf)) or np.any(
        finite & ((tolerance_deg < 0) | (tolerance_deg > _PHASE_BAND_MAX_TOLERANCE_DEG))
    ):
        raise ValueError(
            "phase_band_tolerance_deg values must lie in "
            f"[0, {_PHASE_BAND_MAX_TOLERANCE_DEG}] degrees, or be inf for a free line."
        )
    return np.radians(tolerance_deg)


def _project_out_columns(basis: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Return ``matrix`` minus its orthogonal projection onto the columns of ``basis``.

    The column space is taken from a rank-thresholded SVD, so zero or
    duplicate columns (for example a non-positive T2*) are ignored.
    """
    left, singular, _ = np.linalg.svd(basis, full_matrices=False)
    if not singular.size or singular[0] == 0:
        return matrix
    cutoff = singular[0] * max(basis.shape) * np.finfo(float).eps
    span = left[:, singular > cutoff]
    return matrix - span @ (span.conj().T @ matrix)


def _solve_phase_band_amplitudes(
    basis: np.ndarray,
    signal: np.ndarray,
    freq_hz: np.ndarray,
    tolerance_rad: np.ndarray,
    *,
    tau_s: float,
) -> np.ndarray:
    """Return least-squares complex amplitudes under the phase band.

    ``basis`` has shape ``(samples, metabolites)`` and ``signal`` shape
    ``(samples,)``, both complex128; ``freq_hz`` gives one frequency in Hz per
    metabolite, ``tolerance_rad`` one tolerance in ``[0, pi/2]`` or ``inf``.

    For a banded line, the allowed amplitudes around its law phase form a
    convex wedge, spanned with non-negative weights by rays at ``-tol``,
    ``0`` and ``+tol`` (exactly, because ``tol <= pi/2``). After projecting out
    the free lines, the banded amplitudes solve one NNLS problem per trial
    ``phi0``. The real-stacked ray matrix is reduced once by QR, so every
    trial only rotates the reduced data and solves NNLS with the small
    triangular factor; this equals NNLS on the full matrix. The law is written
    around the mean banded frequency ``f_ref`` so that the searched phase and
    ``tau_s`` are nearly uncorrelated. The free amplitudes are solved last by
    pinv on the signal that the banded lines leave. The solution is
    deterministic and the inputs are not modified.

    If the free least-squares amplitudes already satisfy the band for one
    common phase (see :func:`_phase_band_holds`), they are the constrained
    optimum and are returned without the phase search. The search reaches the
    same amplitudes up to its phase precision ``_PHASE_BAND_PHASE_XATOL_RAD``,
    which matters only for lines with zero tolerance.
    """
    banded = np.flatnonzero(np.isfinite(tolerance_rad))
    free = np.flatnonzero(~np.isfinite(tolerance_rad))
    f_ref_hz = float(np.mean(freq_hz[banded]))
    unconstrained = np.linalg.pinv(basis).dot(signal)
    if _phase_band_holds(
        unconstrained[banded],
        2 * np.pi * (freq_hz[banded] - f_ref_hz) * tau_s,
        tolerance_rad[banded],
    ):
        return unconstrained
    banded_basis = basis[:, banded]
    projected_signal = signal
    if free.size:
        banded_basis = _project_out_columns(basis[:, free], banded_basis)
        projected_signal = _project_out_columns(basis[:, free], signal)

    owner, offsets = [], []
    for local, tolerance in enumerate(tolerance_rad[banded]):
        rays = (0.0,) if tolerance == 0 else (-tolerance, 0.0, tolerance)
        owner += [local] * len(rays)
        offsets += rays
    owner = np.asarray(owner)
    ray_phase = 2 * np.pi * (freq_hz[banded][owner] - f_ref_hz) * tau_s + np.asarray(
        offsets
    )
    rays = banded_basis[:, owner] * np.exp(1j * ray_phase)
    orthonormal, triangular = np.linalg.qr(np.vstack([rays.real, rays.imag]))
    data_cos = orthonormal.T @ np.concatenate(
        [projected_signal.real, projected_signal.imag]
    )
    data_sin = orthonormal.T @ np.concatenate(
        [projected_signal.imag, -projected_signal.real]
    )
    signal_energy = float(np.vdot(projected_signal, projected_signal).real)
    max_iterations = 50 * triangular.shape[1]

    def solve(phase_rad: float) -> tuple[float, np.ndarray]:
        target = np.cos(phase_rad) * data_cos + np.sin(phase_rad) * data_sin
        weights, residual = nnls(triangular, target, maxiter=max_iterations)
        return residual * residual + signal_energy - float(target @ target), weights

    grid = np.linspace(-np.pi, np.pi, _PHASE_BAND_PHASE_GRID_POINTS, endpoint=False)
    grid_rss = [solve(phase)[0] for phase in grid]
    best = int(np.argmin(grid_rss))
    step = 2 * np.pi / _PHASE_BAND_PHASE_GRID_POINTS
    refined = minimize_scalar(
        lambda phase: solve(phase)[0],
        bounds=(grid[best] - step, grid[best] + step),
        method="bounded",
        options={"xatol": _PHASE_BAND_PHASE_XATOL_RAD},
    )
    phase_rad = float(refined.x) if refined.fun <= grid_rss[best] else grid[best]
    weights = solve(phase_rad)[1]

    amplitudes = np.zeros(basis.shape[1], dtype=np.complex128)
    np.add.at(amplitudes, banded[owner], weights * np.exp(1j * (phase_rad + ray_phase)))
    if free.size:
        remainder = signal - basis[:, banded] @ amplitudes[banded]
        amplitudes[free] = np.linalg.pinv(basis[:, free]).dot(remainder)
    return amplitudes


def _phase_band_holds(
    amplitudes: np.ndarray, law_phase_rad: np.ndarray, tolerance_rad: np.ndarray
) -> bool:
    """Return whether one common phase puts every amplitude inside its band.

    A non-zero amplitude ``a_k`` admits the common phases ``phi0`` with
    ``abs(wrap(angle(a_k) - law_phase_rad[k] - phi0)) <= tolerance_rad[k]``,
    an arc shorter than pi because ``tolerance_rad <= pi/2``. Zero
    amplitudes admit every phase. Arcs shorter than pi intersect in one arc
    or not at all, and a non-empty intersection contains an end point of
    one of the arcs, so testing all end points decides the question. End
    points are tested with a margin of ``_PHASE_BAND_EDGE_MARGIN_RAD`` for
    rounding.
    """
    nonzero = amplitudes != 0
    centres = np.angle(amplitudes[nonzero]) - law_phase_rad[nonzero]
    tolerances = tolerance_rad[nonzero]
    if not centres.size:
        return True
    candidates = np.concatenate([centres - tolerances, centres + tolerances])
    deviation = np.abs(
        (centres[:, None] - candidates[None, :] + np.pi) % (2 * np.pi) - np.pi
    )
    within = deviation <= tolerances[:, None] + _PHASE_BAND_EDGE_MARGIN_RAD
    return bool(np.any(np.all(within, axis=0)))


def _estimate_diff(y: np.ndarray, y_hat: np.ndarray, method: str = "sse"):
    """Quantify the difference between observed and fitted signals."""
    if method == "sse":
        return float(np.sum(np.abs(y - y_hat) ** 2))
    elif method == "sse_mag":
        return float(np.sum((np.abs(y) - np.abs(y_hat)) ** 2))
    else:
        raise ValueError(f"Unknown method '{method}'")


def _get_scalar_fit_param(
    fit_params: dict,
    key: str,
    default: float,
    *,
    min_value: float | None = None,
) -> float:
    """Return a scalar fit parameter with robust casting and optional clamping."""
    raw_value = fit_params.get(key, default)
    try:
        value = float(np.asarray(raw_value, dtype=np.float64).reshape(-1)[0])
    except (TypeError, ValueError, IndexError):
        value = float(default)
    if min_value is not None:
        value = max(value, min_value)
    return value


def _get_min_peak_separation_hz(fit_params: dict) -> float:
    """Return the non-negative minimum peak separation constraint in Hz."""
    return _get_scalar_fit_param(
        fit_params,
        "min_peak_separation_hz",
        DEFAULT_MIN_PEAK_SEPARATION_HZ,
        min_value=0.0,
    )


def _violates_min_peak_separation(
    freqs_hz: np.ndarray,
    min_peak_separation_hz: float,
) -> bool:
    """Return True when any fitted frequency pair is closer than allowed."""
    if min_peak_separation_hz <= 0.0:
        return False

    freqs_hz = np.asarray(np.real(freqs_hz), dtype=np.float64)
    finite_freqs = np.sort(freqs_hz[np.isfinite(freqs_hz)])
    if finite_freqs.size < 2:
        return False

    return bool(np.any(np.diff(finite_freqs) < min_peak_separation_hz))


def _score_fit_parameters(
    values: np.ndarray,
    freq_hz: np.ndarray,
    t2_star_s: np.ndarray,
    fit_params: dict,
) -> tuple[float, np.ndarray]:
    """Return the selected SSE and amplitudes for one complete parameter set.

    Reject frequency pairs below the minimum separation with infinite error
    and zero amplitudes. Otherwise solve amplitudes with the same options as
    the grid searches and score the summed FID or spectrum magnitude according
    to ``use_sse_mag``. Input arrays and fit settings remain unchanged.
    """
    amplitude_options = _prepare_amplitude_solver_kwargs(fit_params)
    signal_domain = str(fit_params.get("signal_domain", "temporal")).lower()
    min_peak_separation_hz = _get_min_peak_separation_hz(fit_params)
    if _violates_min_peak_separation(freq_hz, min_peak_separation_hz):
        return float("inf"), np.zeros(len(freq_hz), dtype=np.complex128)

    if signal_domain == "spectral":
        input_data_temporal = np.fft.ifft(np.fft.ifftshift(values))
    else:
        input_data_temporal = values.copy()

    fitted_signal, complex_amps = solve_for_complex_amplitudes(
        values=input_data_temporal,
        freq_hz=freq_hz,
        t2_star_s=t2_star_s,
        **amplitude_options,
    )
    fitted_signal_sum = np.sum(fitted_signal, axis=1)

    if fit_params.get("use_sse_mag", False):
        base_score = _estimate_diff(
            np.fft.fftshift(np.fft.fft(input_data_temporal)),
            np.fft.fftshift(np.fft.fft(fitted_signal_sum)),
            method="sse_mag",
        )
    else:
        base_score = _estimate_diff(input_data_temporal, fitted_signal_sum)

    return float(base_score), complex_amps


def _prepare_grid_search(
    values: np.ndarray, fit_params: dict
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Prepare one spectrum and amplitude-solver options for a parameter search.

    Return a temporal copy of the input, its centered spectrum for magnitude
    diagnostics, and keyword arguments for ``solve_for_complex_amplitudes``.
    Spectral input is converted with the centered inverse FFT first, preserving
    the existing round trip used to evaluate errors. The measured FFT is reused
    for all candidates. Neither the input signal nor fit options are modified.
    """
    if fit_params["signal_domain"] == "spectral":
        temporal_signal = np.fft.ifft(np.fft.ifftshift(values))
    else:
        temporal_signal = values.copy()
    measured_spectrum = np.fft.fftshift(np.fft.fft(temporal_signal))
    amplitude_options = _prepare_amplitude_solver_kwargs(fit_params)
    return temporal_signal, measured_spectrum, amplitude_options


def _evaluate_grid_candidate(
    input_data_temporal: np.ndarray,
    measured_spectrum: np.ndarray,
    freq_hz: np.ndarray,
    t2_star_s: np.ndarray,
    amplitude_options: dict,
    *,
    return_amplitudes: bool = False,
) -> tuple[float, float] | tuple[float, float, np.ndarray]:
    """Solve candidate amplitudes and return complex-FID and magnitude-spectrum SSE.

    Both searches record both errors, independently of which one selects the
    winning candidate. Reuse the component FIDs returned by the amplitude solver
    instead of rebuilding their basis. Callers retain responsibility for rejecting
    invalid frequencies or T2* values before evaluation.
    With ``return_amplitudes``, also return the already solved amplitudes for
    complete history recording, without running a second amplitude fit.
    """
    fitted_components, amplitudes = solve_for_complex_amplitudes(
        values=input_data_temporal,
        freq_hz=freq_hz,
        t2_star_s=t2_star_s,
        **amplitude_options,
    )
    fitted_signal_sum = np.sum(fitted_components, axis=1)
    temporal_sse = _estimate_diff(input_data_temporal, fitted_signal_sum)
    spectral_sse = _estimate_diff(
        measured_spectrum,
        np.fft.fftshift(np.fft.fft(fitted_signal_sum)),
        method="sse_mag",
    )
    if return_amplitudes:
        return temporal_sse, spectral_sse, amplitudes
    return temporal_sse, spectral_sse


def _validate_search_state(frequencies, t2s, fit_params):
    """Reject invalid starting states, including user-edited parameter maps."""
    for values, lower_key, upper_key in (
        (frequencies, "freq_range_lower_bound_hz", "freq_range_upper_bound_hz"),
        (t2s, "min_t2_s", "max_t2_s"),
    ):
        lower = np.asarray(np.real(fit_params[lower_key]), dtype=float)
        upper = np.asarray(np.real(fit_params[upper_key]), dtype=float)
        if (
            not np.all(np.isfinite(values))
            or not np.all(np.isfinite(lower))
            or not np.all(np.isfinite(upper))
            or np.any(lower > upper)
            or np.any(values < lower)
            or np.any(values > upper)
        ):
            raise ValueError(
                f"Initial parameters must lie within finite {lower_key}/{upper_key} bounds."
            )
    if np.any(t2s <= 0) or np.any(np.asarray(fit_params["min_t2_s"]) <= 0):
        raise ValueError("T2* values and bounds must be positive.")
    for key in ("metabs_freq_ranges", "metabs_t2_ranges"):
        values = np.asarray(fit_params[key])
        if not np.all(np.isfinite(values)) or np.any(values < 0):
            raise ValueError(f"{key} must be finite and non-negative.")


def _fit_parameter_grid(values: np.ndarray, fit_params: dict, *, parameter: str):
    """Search one parameter family while retaining complete feasible states.

    Each candidate solves all complex amplitudes. Out-of-bounds candidates are
    recorded with infinite errors without solving them. The current value is
    always on the grid (also for even npoints); ties retain it. A zero width
    freezes exactly the user-selected parameter. No selected value is clipped.
    In the T2* search, fixed ``t2_coupling`` dependents are not searched; each
    leader candidate also sets its dependents and is rejected if any leaves its
    bounds. A dependent with a ratio band is searched itself, and its candidates
    are rejected if the ratio to its leader leaves the band. A band leader is
    searched twice per iteration (``_t2_search_moves``): first its candidates
    move the banded dependents with the ratio they had when that move began,
    then the leader moves alone, its banded dependents keep their T2* and the
    bands limit its candidates. The diagnostics row of a band leader holds the
    second move; the optimization history records both, the second as
    ``t2_band_leader_candidate``.

    In the frequency search, every line keeps its own slot, search width and
    bounds for the whole call, and each candidate must lie in its own line's
    bounds. The names, T2* values and roles of each ``t2_coupling`` group
    belong to frequency positions (``_coupled_frequency_positions``): every
    candidate is scored, and recorded in the optimization history, with the
    group's lines given to the names in frequency order
    (``_coupled_line_order``). A candidate that moves a line past another line
    of its group therefore exchanges the two names and their T2* constraints,
    never the windows. Diagnostics row ``m`` holds the candidates of the line
    in slot ``m`` at entry, as tried.

    Returns the component spectra, the selected values in input order (for
    frequencies: after the final exchange of names), complex amplitudes,
    diagnostics with axes ``(metabolite, iteration, candidate, field)`` and
    the final line permutation from ``_coupled_line_order``. Slot ``k`` of the
    result holds the line from input slot ``order[k]``; callers move the
    frequency windows with it (``_move_frequency_windows``). The permutation
    is the identity for the T2* search and without ``t2_coupling``.
    """
    npoints = fit_params.get("npoints", NPOINTS_DEFAULT)
    niter = fit_params.get("niter", NITER_DEFAULT)
    use_sse_mag = fit_params.get("use_sse_mag", USE_SSE_MAG_DEFAULT)
    temporal, spectrum, amplitude_options = _prepare_grid_search(values, fit_params)
    frequencies = np.asarray(np.real(fit_params["metabs_freqs_hz"]), dtype=float).copy()
    t2s = np.asarray(
        np.real(fit_params.get("initial_t2s_s", fit_params.get("metabs_t2s"))),
        dtype=float,
    ).copy()
    # Low-level grid callers historically only supplied their own bounds/widths.
    validation_params = {
        "freq_range_lower_bound_hz": frequencies,
        "freq_range_upper_bound_hz": frequencies,
        "min_t2_s": t2s,
        "max_t2_s": t2s,
        "metabs_freq_ranges": np.zeros_like(frequencies),
        "metabs_t2_ranges": np.zeros_like(t2s),
        **fit_params,
    }
    _validate_search_state(frequencies, t2s, validation_params)
    if parameter == "frequency":
        current = frequencies
        widths = np.asarray(np.real(fit_params["metabs_freq_ranges"]), dtype=float)
        lower = np.asarray(
            np.real(fit_params["freq_range_lower_bound_hz"]), dtype=float
        )
        upper = np.asarray(
            np.real(fit_params["freq_range_upper_bound_hz"]), dtype=float
        )
    else:
        current = t2s
        widths = np.asarray(np.real(fit_params["metabs_t2_ranges"]), dtype=float)
        lower = np.asarray(np.real(fit_params["min_t2_s"]), dtype=float)
        upper = np.asarray(np.real(fit_params["max_t2_s"]), dtype=float)
    separation = _get_min_peak_separation_hz(fit_params)
    coupling = _t2_coupling_pairs(fit_params) if parameter == "t2" else []
    bands = _t2_band_pairs(fit_params) if parameter == "t2" else []
    positions = (
        _coupled_frequency_positions(fit_params) if parameter == "frequency" else []
    )
    visit_order = _search_visit_order(
        temporal, frequencies, t2s, amplitude_options, fit_params
    )
    moves = _t2_search_moves(len(current), coupling, bands, visit_order)
    diagnostics = np.full((len(current), niter, npoints, 3), np.nan)
    history = fit_params.get("_fit_history")

    for iteration in range(niter):
        for m, bands_follow in moves:
            followers, checked_bands = _t2_move_constraints(
                m, bands_follow, current, coupling, bands
            )
            if parameter == "frequency":
                history_event = "frequency_candidate"
            elif bands_follow:
                history_event = "t2_candidate"
            else:
                history_event = "t2_band_leader_candidate"
            previous = current[m]
            grid = np.linspace(
                previous - widths[m] / 2, previous + widths[m] / 2, npoints
            )
            center = int(np.argmin(np.abs(grid - previous)))
            grid[center] = previous
            scores = np.full(npoints, np.inf)
            history_indices = []
            for index, candidate in enumerate(grid):
                diagnostics[m, iteration, index, 0] = candidate
                current[m] = candidate
                for dependent, ratio in followers:
                    current[dependent] = ratio * candidate
                scored_frequencies = (
                    frequencies[_coupled_line_order(frequencies, positions)]
                    if positions
                    else frequencies
                )
                reason = "none"
                if (
                    candidate < lower[m]
                    or candidate > upper[m]
                    or any(
                        not lower[dependent] <= current[dependent] <= upper[dependent]
                        for dependent, _ in followers
                    )
                    or not _inside_t2_bands(current, checked_bands)
                ):
                    reason = "bounds"
                elif _violates_min_peak_separation(scored_frequencies, separation):
                    reason = "peak_separation"
                if reason != "none":
                    diagnostics[m, iteration, index, 1:] = np.inf
                    if history is not None:
                        history_indices.append(
                            history.append(
                                scored_frequencies,
                                t2s,
                                complex(np.nan, np.nan),
                                np.nan,
                                np.inf,
                                fit_params,
                                event=history_event,
                                iteration=iteration,
                                metabolite=m,
                                candidate=index,
                                evaluated=False,
                                reason=reason,
                            )
                        )
                    continue
                try:
                    errors = _evaluate_grid_candidate(
                        temporal,
                        spectrum,
                        scored_frequencies,
                        t2s,
                        amplitude_options,
                        **({"return_amplitudes": True} if history is not None else {}),
                    )
                except np.linalg.LinAlgError:
                    if history is not None:
                        history.append(
                            scored_frequencies,
                            t2s,
                            complex(np.nan, np.nan),
                            np.nan,
                            np.nan,
                            fit_params,
                            event=history_event,
                            iteration=iteration,
                            metabolite=m,
                            candidate=index,
                            evaluated=False,
                            reason="linear_algebra_error",
                        )
                    raise
                if history is not None:
                    history_indices.append(
                        history.append(
                            scored_frequencies,
                            t2s,
                            errors[2],
                            errors[0],
                            errors[1 if use_sse_mag else 0],
                            fit_params,
                            event=history_event,
                            iteration=iteration,
                            metabolite=m,
                            candidate=index,
                            reason="none"
                            if np.all(np.isfinite(errors[:2]))
                            else "nonfinite_objective",
                        )
                    )
                    errors = errors[:2]
                diagnostics[m, iteration, index, 1:] = errors
                scores[index] = errors[1 if use_sse_mag else 0]
                if not np.all(np.isfinite(errors)):
                    raise FIDNumericalError("nonfinite_objective")
            current[m] = previous
            finite = np.isfinite(scores)
            if np.any(finite):
                best = int(np.argmin(np.where(finite, scores, np.inf)))
                if scores[best] < scores[center]:
                    current[m] = grid[best]
                    if history is not None:
                        history.accept(history_indices[best])
            for dependent, ratio in followers:
                current[dependent] = ratio * current[m]

    line_order = _coupled_line_order(frequencies, positions)
    if positions:
        frequencies = frequencies[line_order]
        current = frequencies
    fitted, amplitudes = solve_for_complex_amplitudes(
        temporal, frequencies, t2s, **amplitude_options
    )
    fitted_spectral = np.fft.fftshift(np.fft.fft(fitted, axis=0), axes=0)
    return fitted_spectral, current, amplitudes, diagnostics, line_order


def fit_freq_pseudo_inv(values: np.ndarray, fit_params: dict) -> tuple:
    """Refine frequencies within user bounds using complex amplitude refits.

    Parameters
    ----------
    values : numpy.ndarray
        One complex FID or centered spectrum, shape (n_samples,), in signal units.
        The mapping's signal_domain controls conversion to a temporal FID.
    fit_params : dict
        Single-spectrum options with sampling_time_axis_s in seconds, initial
        frequencies in Hz, decay times in seconds and matching per-metabolite
        bounds/search widths. Scientific defaults remain owned by preparation.
        Voigt refinement is unsupported here; use the complete spectrum fitter.

    Returns
    -------
    fitted_spectral : numpy.ndarray
        Complex component spectra, shape (n_samples, n_metabolites), with the
        existing centered FFT convention and signal scaling.
    frequencies_hz : numpy.ndarray
        Refined frequencies in Hz, shape (n_metabolites,), in input order.
    amplitudes : numpy.ndarray
        Complex line areas at t=0 in signal units, shape (n_metabolites,).
    diagnostics : numpy.ndarray
        Float64 candidates with axes (metabolite, iteration, candidate, field).
        Fields are candidate Hz, temporal complex RSS and spectral magnitude RSS;
        legacy r2 names denote residual sums, not a coefficient of determination.

    Raises
    ------
    ValueError
        If options, search bounds, sampling or array shapes are inconsistent,
        or a Voigt lineshape is requested.
    FIDNumericalError
        If no finite numerical state can be selected.

    Notes
    -----
    The established amplitude solver uses complex128 calculation. This eager,
    fixed-order search preserves inputs and retains no mutable cache. Candidate
    bounds, minimum peak separation, iteration order and tie handling are unchanged.
    With ``t2_coupling``, names, T2* values and coupling roles of each group
    belong to frequency positions, ordered by ``metabs_freqs_hz`` (equal values
    by name): every candidate is scored with the group's lines given to the
    names in frequency order, and the returned frequencies are in that order.
    Each line is searched within its own ``metabs_freq_ranges`` and bounds.
    Which line ended under which name is not returned; the complete spectrum
    fitter uses it to keep every line's window with the line in later passes.

    Examples
    --------
    >>> spectra, frequencies, areas, candidates = fit_freq_pseudo_inv(
    ...     measured_fid, options
    ... )  # doctest: +SKIP"""
    return _fit_parameter_grid(values, fit_params, parameter="frequency")[:4]


def fit_t2_pseudo_inv(values: np.ndarray, fit_params: dict) -> tuple:
    """Refine positive decay times within bounds using complex amplitude refits.

    Parameters
    ----------
    values : numpy.ndarray
        One complex FID or centered spectrum, shape (n_samples,), in signal units.
        The mapping's signal_domain controls conversion to a temporal FID.
    fit_params : dict
        Single-spectrum options with sampling_time_axis_s in seconds, current
        frequencies in Hz, initial decay times and positive decay bounds/search
        widths in seconds. Metabolite vector order must agree across options.
        Voigt refinement is unsupported here; use the complete spectrum fitter.

    Returns
    -------
    fitted_spectral : numpy.ndarray
        Complex component spectra, shape (n_samples, n_metabolites), with the
        existing centered FFT convention and signal scaling.
    decay_times_s : numpy.ndarray
        Refined decay times in seconds, shape (n_metabolites,), in input order.
    amplitudes : numpy.ndarray
        Complex line areas at t=0 in signal units, shape (n_metabolites,).
    diagnostics : numpy.ndarray
        Float64 candidates with axes (metabolite, iteration, candidate, field).
        Fields are candidate seconds, temporal complex RSS and spectral magnitude
        RSS; legacy r2 names denote residual sums, not a coefficient of determination.
        The row of a ``t2_coupling`` band leader holds its second, own move;
        fixed dependents keep NaN rows because they are not searched.

    Raises
    ------
    ValueError
        If options, search bounds, sampling or shapes are inconsistent,
        or a Voigt lineshape is requested.
    FIDNumericalError
        If no finite numerical state can be selected.

    Notes
    -----
    The established amplitude solver uses complex128 calculation. This eager,
    fixed-order search preserves inputs and retains no cache. Scientific defaults,
    iteration order and tie handling are unchanged.

    Examples
    --------
    >>> spectra, decay_times, areas, candidates = fit_t2_pseudo_inv(
    ...     measured_fid, options
    ... )  # doctest: +SKIP"""
    return _fit_parameter_grid(values, fit_params, parameter="t2")[:4]


def _fit_lorentz_fwhm_grid(
    values: np.ndarray, fit_params: dict, *, widths_hz: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Refine each Lorentz part on a grid in Lorentz FWHM (Hz) for Voigt fits.

    A linear grid in Hz reaches long Lorentz ``T2`` values, which a linear
    ``T2`` grid of fixed width cannot. Per metabolite, candidates are
    ``1 / (pi * T2) +/- widths_hz / 2`` with the current value forced onto the
    grid; that centre keeps the exact current ``T2``. Candidates with
    non-positive FWHM or ``T2`` outside ``[min_t2_s, max_t2_s]`` are rejected
    without solving. A candidate replaces the current value only if it strictly
    lowers the selected error (``use_sse_mag``). The current shared Gaussian
    FWHM is read from the private options key ``"_gauss_fwhm_hz"``.
    Fixed ``t2_coupling`` dependents are not searched; each leader candidate
    also sets their Lorentz ``T2`` and is rejected if any leaves its bounds.
    Banded dependents are searched and kept inside their ratio band, and a
    band leader gets its second, own move, as in ``_fit_parameter_grid``.

    Parameters
    ----------
    values : numpy.ndarray
        One FID or centred spectrum, as described by ``signal_domain``.
    fit_params : dict
        Single-spectrum options with ``metabs_freqs_hz``, ``metabs_t2s``,
        ``min_t2_s``, ``max_t2_s`` (one value per metabolite) and search counts.
        Not modified.
    widths_hz : numpy.ndarray
        Full Lorentz-FWHM search width in Hz per metabolite; zero freezes it.

    Returns
    -------
    t2s : numpy.ndarray
        Selected Lorentz ``T2`` values in seconds.
    diagnostics : numpy.ndarray
        Shape ``(metabolite, iteration, candidate, 3)``: candidate ``T2`` in
        seconds (NaN for non-positive FWHM), complex temporal RSS and spectral
        magnitude RSS; rejected candidates have infinite errors.

    Raises
    ------
    FIDNumericalError
        If an evaluated candidate yields a non-finite error.
    """
    npoints = fit_params.get("npoints", NPOINTS_DEFAULT)
    niter = fit_params.get("niter", NITER_DEFAULT)
    selected_error = 1 if fit_params.get("use_sse_mag", USE_SSE_MAG_DEFAULT) else 0
    temporal, spectrum, amplitude_options = _prepare_grid_search(values, fit_params)
    frequencies = np.asarray(fit_params["metabs_freqs_hz"], dtype=float).copy()
    t2s = np.asarray(fit_params["metabs_t2s"], dtype=float).copy()
    lower = np.asarray(fit_params["min_t2_s"], dtype=float)
    upper = np.asarray(fit_params["max_t2_s"], dtype=float)
    separated = not _violates_min_peak_separation(
        frequencies, _get_min_peak_separation_hz(fit_params)
    )
    coupling = _t2_coupling_pairs(fit_params)
    bands = _t2_band_pairs(fit_params)
    visit_order = _search_visit_order(
        temporal, frequencies, t2s, amplitude_options, fit_params
    )
    moves = _t2_search_moves(len(t2s), coupling, bands, visit_order)
    diagnostics = np.full((len(t2s), niter, npoints, 3), np.nan)

    for iteration in range(niter):
        for m, bands_follow in moves:
            followers, checked_bands = _t2_move_constraints(
                m, bands_follow, t2s, coupling, bands
            )
            previous = float(t2s[m])
            centre_fwhm = 1.0 / (np.pi * previous)
            grid = np.linspace(
                centre_fwhm - widths_hz[m] / 2, centre_fwhm + widths_hz[m] / 2, npoints
            )
            centre = int(np.argmin(np.abs(grid - centre_fwhm)))
            grid[centre] = centre_fwhm
            candidates = np.full(npoints, np.nan)
            scores = np.full(npoints, np.inf)
            for index, fwhm in enumerate(grid.tolist()):
                candidate = None
                if fwhm == centre_fwhm:
                    candidate = previous
                elif fwhm > 0:
                    # Python floats give inf instead of an overflow error.
                    candidate = 1.0 / (np.pi * fwhm)
                if candidate is not None:
                    candidates[index] = candidate
                    diagnostics[m, iteration, index, 0] = candidate
                if not (
                    separated
                    and candidate is not None
                    and lower[m] <= candidate <= upper[m]
                    and all(
                        lower[dependent] <= ratio * candidate <= upper[dependent]
                        for dependent, ratio in followers
                    )
                ):
                    diagnostics[m, iteration, index, 1:] = np.inf
                    continue
                t2s[m] = candidate
                if not _inside_t2_bands(t2s, checked_bands):
                    t2s[m] = previous
                    diagnostics[m, iteration, index, 1:] = np.inf
                    continue
                for dependent, ratio in followers:
                    t2s[dependent] = ratio * candidate
                errors = _evaluate_grid_candidate(
                    temporal, spectrum, frequencies, t2s, amplitude_options
                )
                diagnostics[m, iteration, index, 1:] = errors
                scores[index] = errors[selected_error]
                if not np.all(np.isfinite(errors)):
                    raise FIDNumericalError("nonfinite_objective")
            t2s[m] = previous
            best = int(np.argmin(scores))
            if scores[best] < scores[centre]:
                t2s[m] = candidates[best]
            for dependent, ratio in followers:
                t2s[dependent] = ratio * t2s[m]
    return t2s, diagnostics


def _fit_gauss_fwhm_grid(
    values: np.ndarray,
    fit_params: dict,
    *,
    gauss_fwhm_hz: float,
    width_hz: float,
) -> tuple[np.ndarray, float]:
    """Refine the shared Gaussian FWHM while keeping every total linewidth.

    Lorentz and Gaussian widths are strongly anti-correlated, so a plain
    one-dimensional Gaussian step stalls. Each candidate ``g`` therefore also
    moves every Lorentz part such that the Olivero-Longbothum Voigt FWHM of
    each line stays unchanged: ``f_L = OL^-1(f_V, g)`` and ``T2 = 1/(pi f_L)``.
    The grid is ``gauss_fwhm_hz +/- width_hz / 2`` with the current value
    forced onto it; that centre keeps the exact current state. Candidates are
    rejected without solving if ``g < 0``, ``g > gauss_max_fwhm_hz``,
    ``g >= f_V`` for any line, or a derived ``T2`` leaves
    ``[min_t2_s, max_t2_s]``. Only strict improvements of the selected error
    are accepted. The search runs ``niter`` times with the same width.
    With ``t2_coupling``, each fixed dependent's derived ``T2`` is replaced by
    its leader's times ``t2_ratio`` before the bounds check, so the coupling
    holds exactly and the dependent's total width follows its leader. A banded
    dependent keeps its own total width; candidates whose derived ratio leaves
    the band are rejected.

    Parameters
    ----------
    values : numpy.ndarray
        One FID or centred spectrum, as described by ``signal_domain``.
    fit_params : dict
        Single-spectrum options with ``metabs_freqs_hz``, ``metabs_t2s``
        (Lorentz part), ``min_t2_s``, ``max_t2_s``, ``gauss_max_fwhm_hz`` and
        search counts. Not modified.
    gauss_fwhm_hz : float
        Current shared Gaussian FWHM in Hz.
    width_hz : float
        Full search width in Hz; positive.

    Returns
    -------
    t2s : numpy.ndarray
        Lorentz ``T2`` values in seconds that belong to the selected width.
    gauss_fwhm_hz : float
        Selected shared Gaussian FWHM in Hz.

    Raises
    ------
    FIDNumericalError
        If an evaluated candidate yields a non-finite error.
    """
    npoints = fit_params.get("npoints", NPOINTS_DEFAULT)
    niter = fit_params.get("niter", NITER_DEFAULT)
    selected_error = 1 if fit_params.get("use_sse_mag", USE_SSE_MAG_DEFAULT) else 0
    temporal, spectrum, amplitude_options = _prepare_grid_search(values, fit_params)
    frequencies = np.asarray(fit_params["metabs_freqs_hz"], dtype=float).copy()
    t2s = np.asarray(fit_params["metabs_t2s"], dtype=float).copy()
    lower = np.asarray(fit_params["min_t2_s"], dtype=float)
    upper = np.asarray(fit_params["max_t2_s"], dtype=float)
    gauss_max_fwhm_hz = float(fit_params["gauss_max_fwhm_hz"])
    gauss = float(gauss_fwhm_hz)
    bands = _t2_band_pairs(fit_params)
    if _violates_min_peak_separation(
        frequencies, _get_min_peak_separation_hz(fit_params)
    ):
        return t2s, gauss

    for _ in range(niter):
        voigt_fwhm_hz = _voigt_fwhm_hz(1.0 / (np.pi * t2s), gauss)
        grid = np.linspace(gauss - width_hz / 2, gauss + width_hz / 2, npoints)
        centre = int(np.argmin(np.abs(grid - gauss)))
        grid[centre] = gauss
        candidates = [None] * npoints
        scores = np.full(npoints, np.inf)
        for index, candidate in enumerate(grid.tolist()):
            if candidate == gauss:
                candidate_t2s = t2s
            elif 0 <= candidate <= gauss_max_fwhm_hz and np.all(
                candidate < voigt_fwhm_hz
            ):
                lorentz_fwhm_hz = _lorentz_fwhm_from_voigt_hz(voigt_fwhm_hz, candidate)
                candidate_t2s = _apply_t2_coupling(
                    [
                        1.0 / (np.pi * fwhm) if fwhm > 0 else np.inf
                        for fwhm in lorentz_fwhm_hz.tolist()
                    ],
                    fit_params,
                )
                if (
                    np.any(candidate_t2s < lower)
                    or np.any(candidate_t2s > upper)
                    or not _inside_t2_bands(candidate_t2s, bands)
                ):
                    continue
            else:
                continue
            errors = _evaluate_grid_candidate(
                temporal,
                spectrum,
                frequencies,
                candidate_t2s,
                {**amplitude_options, "gauss_fwhm_hz": candidate},
            )
            if not np.all(np.isfinite(errors)):
                raise FIDNumericalError("nonfinite_objective")
            candidates[index] = candidate_t2s
            scores[index] = errors[selected_error]
        best = int(np.argmin(scores))
        if scores[best] < scores[centre]:
            t2s, gauss = candidates[best].copy(), float(grid[best])
    return t2s, gauss


def _prepare_single_spectrum_data(
    data_slice: np.ndarray, fit_options: dict
) -> tuple[np.ndarray, dict]:
    """Return the fitted sample prefix and a private copy of global options.

    The sampling axis sets the target length, with ``nsamplepoints`` as fallback.
    Preserve the historical clamp to at least one and at most the input size.
    A shortened spectrum is first converted to an FID, then truncated; mark the
    returned options as temporal so later searches do not transform it twice.
    Full-length spectral input stays spectral. Neither input is modified.
    """
    data_slice = np.asarray(data_slice)
    kwargs = fit_options.copy()
    signal_domain = str(kwargs.get("signal_domain", "temporal")).lower()
    target_nsamplepoints = fitted_sample_count(data_slice.size, kwargs)
    if target_nsamplepoints < data_slice.size:
        if signal_domain == "spectral":
            data_slice = np.fft.ifft(np.fft.ifftshift(data_slice))[
                :target_nsamplepoints
            ]
            kwargs["signal_domain"] = "temporal"
        else:
            data_slice = data_slice[:target_nsamplepoints]

    return data_slice, kwargs


def fitted_sample_count(n_input_samples: int, fit_options: dict) -> int:
    """Return how many complex FID samples one spectrum's fit uses.

    The length of ``sampling_time_axis_s`` sets the target, with
    ``nsamplepoints`` as fallback, clamped to at least one and at most the
    input length. Without either option every input sample is used. This is
    the single owner of the fitted sample count: the solver, the final RSS
    and the residual degrees of freedom all use it.

    Parameters
    ----------
    n_input_samples : int
        Number of samples of one input spectrum along the spectral dimension.
    fit_options : dict
        Global fit options; read only.

    Returns
    -------
    int
        Number of complex samples entering the fit residual.
    """
    sampling_time_axis_s = fit_options.get("sampling_time_axis_s")
    if sampling_time_axis_s is not None:
        target = int(np.asarray(sampling_time_axis_s).size)
    elif fit_options.get("nsamplepoints") is not None:
        target = int(fit_options["nsamplepoints"])
    else:
        return int(n_input_samples)
    return max(1, min(target, int(n_input_samples)))


def count_free_parameters(
    active: np.ndarray,
    freq_search_ranges_hz: np.ndarray,
    t2_search_ranges_s: np.ndarray,
    min_t2s_s: np.ndarray,
    max_t2s_s: np.ndarray,
    freq_lower_bounds_hz: np.ndarray,
    freq_upper_bounds_hz: np.ndarray,
    *,
    metabolites: Sequence[str],
    fit_options: dict,
) -> int:
    """Count the real parameters one spectrum's FID fit adjusts.

    The count is the number of real degrees of freedom that the fitted signal
    loses to the fit, used by the reduced chi-squared. Lines that are skipped
    or were excluded as low-amplitude are inactive and contribute nothing.

    * Complex amplitude: 2 per active line. With ``phase_band_tolerance_deg``
      a listed line with tolerance 0 keeps only its magnitude (1), plus one
      shared phase ``phi0`` when at least one such line is active; a listed
      line with a positive tolerance keeps magnitude and phase deviation (2).
    * Frequency: 1 per active line with a positive search range and a
      non-empty bound interval.
    * T2*: 1 per active line with a positive search range and
      ``min_t2_s < max_t2_s``. A fixed-ratio ``t2_coupling`` group (a leader
      and its fixed dependents) has one T2*, counted when any group member is
      active and the leader's range is free; a banded dependent counts its
      own T2*.
    * Voigt Gaussian FWHM: 1 per spectrum with an active line when
      ``gauss_search_width_hz > 0``.

    A parameter that ends on a bound still counts as free.

    Parameters
    ----------
    active : numpy.ndarray
        Boolean flag per metabolite, in ``metabolites`` order.
    freq_search_ranges_hz, t2_search_ranges_s : numpy.ndarray
        Full search widths per metabolite in Hz and s (``metabs_freq_ranges``
        and ``metabs_t2_ranges``).
    min_t2s_s, max_t2s_s : numpy.ndarray
        Absolute T2* bounds per metabolite in s.
    freq_lower_bounds_hz, freq_upper_bounds_hz : numpy.ndarray
        Absolute frequency bounds per metabolite in Hz.
    metabolites : Sequence[str]
        Metabolite names in solver order.
    fit_options : dict
        Prepared global options with the normalized ``t2_coupling``,
        ``phase_band_tolerance_deg`` and ``lineshape`` entries; read only.

    Returns
    -------
    int
        Number of free real parameters, at least 0.
    """
    names = [str(name) for name in metabolites]
    active = np.asarray(active, dtype=bool)
    frequency_free = (
        active
        & (np.asarray(freq_search_ranges_hz) > 0)
        & (np.asarray(freq_upper_bounds_hz) > np.asarray(freq_lower_bounds_hz))
    )
    t2_range_free = (np.asarray(t2_search_ranges_s) > 0) & (
        np.asarray(max_t2s_s) > np.asarray(min_t2s_s)
    )
    t2_free = active & t2_range_free
    groups: dict[int, list[int]] = {}
    for dependent, (leader, ratio) in (fit_options.get("t2_coupling") or {}).items():
        if isinstance(ratio, tuple) or dependent not in names or leader not in names:
            continue
        leader_index = names.index(leader)
        groups.setdefault(leader_index, [leader_index]).append(names.index(dependent))
    for leader_index, members in groups.items():
        t2_free[members] = False
        t2_free[leader_index] = bool(np.any(active[members])) and bool(
            t2_range_free[leader_index]
        )

    tolerances = fit_options.get("phase_band_tolerance_deg") or {}
    listed = np.array([name in tolerances for name in names], dtype=bool)
    tolerance = np.array([float(tolerances.get(name, np.nan)) for name in names])
    fixed_phase = active & listed & (tolerance == 0)
    free_amplitude = active & ~fixed_phase
    n_amplitude = 2 * int(free_amplitude.sum()) + int(fixed_phase.sum())
    n_amplitude += int(np.any(fixed_phase))

    n_gauss = int(
        _uses_voigt_lineshape(fit_options)
        and float(fit_options.get("gauss_search_width_hz", 0.0)) > 0
        and bool(np.any(active))
    )
    return n_amplitude + int(frequency_free.sum()) + int(t2_free.sum()) + n_gauss


@dataclass(frozen=True)
class _FitState:
    """A complete accepted parameter state, with one common objective.

    ``gauss_fwhm_hz`` is the shared Gaussian FWHM in Hz of a Voigt fit and
    ``None`` for Lorentz fits.
    """

    frequencies: np.ndarray
    t2s: np.ndarray
    amplitudes: np.ndarray
    objective: float
    gauss_fwhm_hz: float | None = None


# Relative stopping tolerance of the joint refinement. Far below scipy's 1e-8
# default, so the refinement stops at the minimum rather than near it. Serial
# and multi-process fits, whose linear algebra may round differently, then
# differ only by the rounding of the objective near its minimum (measured:
# below 1e-9 relative in T2*).
_JOINT_REFINEMENT_TOLERANCE = 1e-14
# Largest cosine between a banded amplitude direction and the FID residual
# for which the analytic Jacobian treats the direction as free (see
# ``_joint_refinement_jacobian``). At the amplitude solution the residual is
# orthogonal to every free direction up to rounding of the NNLS solve.
_JOINT_REFINEMENT_ORTHOGONALITY = 1e-9


def _joint_refinement_proposal(
    data_slice: np.ndarray, state: _FitState, options: dict
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return frequencies and T2* after a joint local least-squares refinement.

    The grid search moves one line and one parameter at a time. Strongly
    coupled lines, for example a weak line next to a strong one, then stop in
    a state from which no single move improves the fit although a joint move
    would. This step starts at ``state`` and minimizes the same objective as
    the grid search over all free frequencies and T2* values together with
    ``scipy.optimize.least_squares`` (trust-region reflective, box bounds).
    The complex amplitudes are solved linearly for every proposal, as in the
    grid search, so they are not optimization variables. For the FID
    objective with the ``pinv`` amplitude solver the derivatives are
    analytic (:func:`_joint_refinement_jacobian`); the magnitude objective
    and the other solvers use two-point finite differences.

    Free variables, each inside its absolute bounds:

    * frequency in Hz of every line with a positive ``metabs_freq_ranges``,
      bounded by ``freq_range_lower_bound_hz``/``freq_range_upper_bound_hz``;
    * T2* in ms of every line with a positive ``metabs_t2_ranges`` that is not
      a ``t2_coupling`` dependent, bounded by ``min_t2_s``/``max_t2_s`` and,
      for a leader, by the bounds of its fixed-ratio dependents divided by
      their ratio;
    * T2* ratio to its leader of every banded dependent with a positive
      ``metabs_t2_ranges``, bounded by its ratio band.

    Fixed-ratio dependents follow their leader. The Gaussian FWHM of a Voigt
    fit stays at ``state.gauss_fwhm_hz``. The residuals are the real and
    imaginary parts of the FID residual, or the magnitude-spectrum residual
    with ``use_sse_mag``, so their sum of squares is the grid objective. The
    step is deterministic.

    Parameters
    ----------
    data_slice : numpy.ndarray
        One FID (or centered spectrum with ``signal_domain='spectral'``),
        shape ``(n_samples,)``; not modified.
    state : _FitState
        Accepted state of the grid search, in the slot order of ``options``.
    options : dict
        Working options of the alternating search, whose frequency windows
        have moved with the lines; not modified.

    Returns
    -------
    tuple of numpy.ndarray or None
        Proposed frequencies in Hz and T2* in s, shape ``(n_lines,)`` each,
        inside the absolute bounds up to rounding of the coupling ratios.
        None if no variable is free or the optimizer fails. The caller
        decides with the full objective whether to accept the proposal.
    """
    freq_lower = np.asarray(np.real(options["freq_range_lower_bound_hz"]), float)
    freq_upper = np.asarray(np.real(options["freq_range_upper_bound_hz"]), float)
    t2_min = np.asarray(np.real(options["min_t2_s"]), float)
    t2_max = np.asarray(np.real(options["max_t2_s"]), float)
    t2_searched = np.asarray(options["metabs_t2_ranges"], float) > 0
    coupling = _t2_coupling_pairs(options)
    bands = _t2_band_pairs(options)
    dependents = {dependent for dependent, _, _ in coupling} | {
        dependent for dependent, _, _, _ in bands
    }

    free_freqs = np.flatnonzero(
        (np.asarray(options["metabs_freq_ranges"], float) > 0)
        & (freq_lower < freq_upper)
    )
    free_t2s, t2_lower_ms, t2_upper_ms = [], [], []
    for line in range(len(state.t2s)):
        if line in dependents or not t2_searched[line]:
            continue
        lower, upper = t2_min[line], t2_max[line]
        for dependent, leader, ratio in coupling:
            if leader == line:
                lower = max(lower, t2_min[dependent] / ratio)
                upper = min(upper, t2_max[dependent] / ratio)
        if lower < upper:
            free_t2s.append(line)
            t2_lower_ms.append(1e3 * lower)
            t2_upper_ms.append(1e3 * upper)
    free_bands = [band for band in bands if t2_searched[band[0]] and band[2] < band[3]]

    lower = np.concatenate(
        [freq_lower[free_freqs], t2_lower_ms, [band[2] for band in free_bands]]
    )
    upper = np.concatenate(
        [freq_upper[free_freqs], t2_upper_ms, [band[3] for band in free_bands]]
    )
    if lower.size == 0:
        return None
    start = np.concatenate(
        [
            state.frequencies[free_freqs],
            1e3 * state.t2s[free_t2s],
            [
                state.t2s[dependent] / state.t2s[leader]
                for dependent, leader, *_ in free_bands
            ],
        ]
    )
    start = np.clip(start, lower, upper)
    n_freqs, n_t2s = len(free_freqs), len(free_t2s)

    def unclipped_parameters(x):
        """Map optimizer variables to frequencies in Hz and unclipped T2* in s."""
        frequencies = state.frequencies.copy()
        frequencies[free_freqs] = x[:n_freqs]
        t2s = state.t2s.copy()
        t2s[free_t2s] = 1e-3 * x[n_freqs : n_freqs + n_t2s]
        t2s = _apply_t2_coupling(t2s, options)
        for ratio, (dependent, leader, _, _) in zip(
            x[n_freqs + n_t2s :], free_bands, strict=True
        ):
            t2s[dependent] = ratio * t2s[leader]
        return frequencies, t2s

    def parameters(x):
        """Map optimizer variables to frequencies in Hz and T2* in s."""
        frequencies, t2s = unclipped_parameters(x)
        return frequencies, np.clip(t2s, t2_min, t2_max)

    if str(options.get("signal_domain", "temporal")).lower() == "spectral":
        measured = np.fft.ifft(np.fft.ifftshift(data_slice))
    else:
        measured = np.asarray(data_slice)
    use_sse_mag = bool(options.get("use_sse_mag", USE_SSE_MAG_DEFAULT))
    measured_magnitude = np.abs(np.fft.fft(measured))
    scoring_options = {**options, "_gauss_fwhm_hz": state.gauss_fwhm_hz}
    if state.gauss_fwhm_hz is None:
        scoring_options.pop("_gauss_fwhm_hz")
    amplitude_options = _prepare_amplitude_solver_kwargs(scoring_options)

    last_solve: dict = {}

    def residuals(x):
        """Return the residual vector whose sum of squares is the objective."""
        frequencies, t2s = parameters(x)
        components, _ = solve_for_complex_amplitudes(
            values=measured,
            freq_hz=frequencies,
            t2_star_s=t2s,
            **amplitude_options,
        )
        fitted = np.sum(components, axis=1)
        if use_sse_mag:
            return measured_magnitude - np.abs(np.fft.fft(fitted))
        difference = np.concatenate(
            [(measured - fitted).real, (measured - fitted).imag]
        )
        last_solve.update(
            x=np.array(x, copy=True), components=components, residual=difference
        )
        return difference

    def jacobian(x):
        """Return the variable-projection Jacobian of ``residuals`` at ``x``."""
        if "x" not in last_solve or not np.array_equal(last_solve["x"], x):
            residuals(x)
        return _joint_refinement_jacobian(
            x,
            last_solve["components"],
            last_solve["residual"],
            unclipped_parameters,
            n_freqs=n_freqs,
            free_freqs=free_freqs,
            free_t2s=free_t2s,
            free_bands=free_bands,
            t2_bounds_s=(t2_min, t2_max),
            options=options,
            amplitude_options=amplitude_options,
        )

    analytic = not use_sse_mag and amplitude_options["amp_solver"] == "pinv"
    try:
        solution = least_squares(
            residuals,
            start,
            jac=jacobian if analytic else "2-point",
            bounds=(lower, upper),
            method="trf",
            ftol=_JOINT_REFINEMENT_TOLERANCE,
            xtol=_JOINT_REFINEMENT_TOLERANCE,
            gtol=_JOINT_REFINEMENT_TOLERANCE,
        )
    except (ValueError, np.linalg.LinAlgError):
        return None
    if not np.all(np.isfinite(solution.x)):
        return None
    return parameters(solution.x)


def _joint_refinement_jacobian(
    x: np.ndarray,
    components: np.ndarray,
    residual: np.ndarray,
    unclipped_parameters,
    *,
    n_freqs: int,
    free_freqs: np.ndarray,
    free_t2s: list[int],
    free_bands: list[tuple[int, int, float, float]],
    t2_bounds_s: tuple[np.ndarray, np.ndarray],
    options: dict,
    amplitude_options: dict,
) -> np.ndarray:
    """Return the Jacobian of the joint-refinement FID residual.

    The residual is ``y - sum_k c_k`` with the fitted components
    ``c_k = a_k * b_k(t)`` and ``b_k = exp(2j*pi*f_k*t - t/T2*_k)`` (times
    the fixed Gaussian envelope of a Voigt fit). With the amplitudes held at
    their solution, the derivatives are ``dc_k/df_k = 2j*pi*t*c_k`` and
    ``dc_k/dT2*_k = t/T2*_k**2 * c_k``. Under the phase band a banded
    amplitude also follows ``exp(2j*pi*(f_k - f_ref)*tau)`` with ``f_ref``
    the mean banded frequency, which adds
    ``2j*pi*tau*(c_k - mean of the banded c)`` to ``dc/df_k`` of a banded
    line.

    Because the amplitudes are solved again for every proposal, the
    derivatives are then projected onto the orthogonal complement of the
    amplitude directions that this solve can follow (variable projection,
    Kaufman 1975): both complex directions of every line without a band,
    and for the banded lines the magnitude of every non-zero line, the
    common phase and the rotation of every line whose deviation from the
    law lies inside its tolerance. A rotation counts as inside when the
    residual is orthogonal to it (``_JOINT_REFINEMENT_ORTHOGONALITY``); a
    line at its tolerance edge cannot rotate further and is not projected
    out. The residual is orthogonal to all projected directions at the
    amplitude solution, so ``J.T @ r`` is the exact gradient of
    ``|r|**2 / 2`` (under the band up to the common-phase search tolerance),
    and ``J`` approximates the Gauss-Newton matrix including the amplitude
    re-solve. The result is deterministic and the inputs are not modified.

    T2* variables are in ms; fixed-ratio dependents follow their leader and a
    banded dependent's T2* is its ratio times the leader's. A T2* clipped to
    ``min_t2_s``/``max_t2_s`` has zero derivative, as in ``parameters``.

    Parameters
    ----------
    x : numpy.ndarray
        Optimizer variables: free frequencies (Hz), free T2* (ms), free
        ratio bands, in this order.
    components : numpy.ndarray
        Fitted components at ``x``, shape ``(samples, lines)``.
    residual : numpy.ndarray
        Residual at ``x``, real parts above imaginary parts, shape
        ``(2 * samples,)``.
    unclipped_parameters : callable
        Maps ``x`` to frequencies in Hz and unclipped T2* in s.
    n_freqs : int
        Number of free frequencies.
    free_freqs, free_t2s, free_bands : sequence
        Line indices of the free frequencies and T2*, and the free
        ``(dependent, leader, low, high)`` ratio bands.
    t2_bounds_s : tuple of numpy.ndarray
        Per-line ``min_t2_s`` and ``max_t2_s``.
    options : dict
        Working fit options with the ``t2_coupling`` of the lines.
    amplitude_options : dict
        Amplitude-solver options: ``time_axis_s``, the Voigt
        ``gauss_fwhm_hz`` if any and the phase band if one applies.

    Returns
    -------
    numpy.ndarray
        Real Jacobian, shape ``(2 * samples, x.size)``: real parts of the
        residual derivatives stacked over imaginary parts.
    """
    frequencies, t2s = unclipped_parameters(x)
    time_s = np.asarray(amplitude_options["time_axis_s"], dtype=float)
    n_lines = components.shape[1]
    t2_min, t2_max = t2_bounds_s
    n_t2s = len(free_t2s)

    # dT2*/dx per line (rows) and variable (columns), before clipping.
    t2_derivative = np.zeros((n_lines, x.size))
    for position, line in enumerate(free_t2s):
        column = n_freqs + position
        t2_derivative[line, column] = 1e-3
        for dependent, leader, ratio in _t2_coupling_pairs(options):
            if leader == line:
                t2_derivative[dependent, column] = 1e-3 * ratio
        for (dependent, leader, _, _), ratio in zip(
            free_bands, x[n_freqs + n_t2s :], strict=True
        ):
            if leader == line:
                t2_derivative[dependent, column] = 1e-3 * ratio
    for position, (dependent, leader, _, _) in enumerate(free_bands):
        t2_derivative[dependent, n_freqs + n_t2s + position] = t2s[leader]
    inside = (t2s >= t2_min) & (t2s <= t2_max)
    t2_derivative[~inside] = 0.0

    with np.errstate(divide="ignore", invalid="ignore"):
        decay_derivative = (
            np.where(inside, time_s[:, None] / np.where(inside, t2s, 1.0) ** 2, 0.0)
            * components
        )
    derivative = decay_derivative @ t2_derivative

    tolerances = amplitude_options.get("phase_band_tolerance_deg")
    banded = (
        np.isfinite(np.asarray(tolerances, dtype=float))
        if tolerances is not None
        else np.zeros(n_lines, dtype=bool)
    )
    tau_s = float(amplitude_options.get("phase_band_tau_s") or 0.0)
    banded_mean = components[:, banded].mean(axis=1) if banded.any() else 0.0
    for column, line in enumerate(free_freqs):
        derivative[:, column] += 2j * np.pi * time_s * components[:, line]
        if banded[line]:
            derivative[:, column] += (
                2j * np.pi * tau_s * (components[:, line] - banded_mean)
            )

    columns = []
    unconstrained = np.flatnonzero(~banded)
    if unconstrained.size:
        basis = fid_basefunc(
            frequencies[unconstrained],
            np.clip(t2s, t2_min, t2_max)[unconstrained],
            time_s,
            gauss_fwhm_hz=amplitude_options.get("gauss_fwhm_hz"),
        ).T
        columns += [_real_stacked(basis), _real_stacked(1j * basis)]
    if banded.any():
        active = banded & np.any(components != 0, axis=0)
        rotations = _real_stacked(1j * components[:, active])
        columns += [
            _real_stacked(components[:, active]),
            _real_stacked(1j * components[:, banded].sum(axis=1)[:, None]),
            rotations[:, _orthogonal_columns(rotations, residual)],
        ]
    stacked = np.concatenate([derivative.real, derivative.imag])
    if columns:
        stacked = _project_out_columns(np.hstack(columns), stacked)
    return -stacked


def _orthogonal_columns(columns: np.ndarray, residual: np.ndarray) -> np.ndarray:
    """Return a mask of the non-zero ``columns`` orthogonal to ``residual``.

    A column counts as orthogonal when the cosine of its angle to the
    residual is at most ``_JOINT_REFINEMENT_ORTHOGONALITY``.
    """
    norms = np.linalg.norm(columns, axis=0) * np.linalg.norm(residual)
    overlap = np.abs(residual @ columns)
    return (norms > 0) & (overlap <= _JOINT_REFINEMENT_ORTHOGONALITY * norms)


def _real_stacked(matrix: np.ndarray) -> np.ndarray:
    """Return complex columns as real columns, real parts above imaginary parts."""
    return np.concatenate([matrix.real, matrix.imag])


def _run_alternating_search(data_slice: np.ndarray, fit_params: dict) -> tuple:
    """Accept or reject frequencies, T2*, amplitudes and objective together.

    Each proposal refits all complex amplitudes against the current other
    parameters. Only strict improvements replace the shared best state. Search
    widths shrink after each pass without changing the user's absolute bounds.
    Re-solve and verify the final state's objective before returning it.

    Voigt fits (private key ``"_gauss_fwhm_hz"`` present) replace the T2*
    step by a Lorentz-FWHM grid step (see :func:`_fit_lorentz_fwhm_grid`)
    whose grid spans ``1/(pi*min_t2_s) - 1/(pi*max_t2_s)`` Hz where
    ``metabs_t2_ranges`` is positive and is frozen where it is zero. If
    ``gauss_search_width_hz`` is positive, every pass then tries the
    width-preserving Gaussian step (:func:`_fit_gauss_fwhm_grid`); otherwise
    the Gaussian FWHM stays at its start value.

    After the last pass, with ``joint_refinement`` (default
    :data:`JOINT_REFINEMENT_DEFAULT`), all free frequencies and T2* values
    are refined together (:func:`_joint_refinement_proposal`). The proposal
    replaces the grid state only if its objective is finite and strictly
    lower, the T2* ratio bands hold, the coupled lines keep their order and
    the minimum peak separation holds; the history records it as event
    ``"joint_refinement"`` either way.

    With ``t2_coupling``, names, T2* values and coupling roles belong to
    frequency positions (:func:`_coupled_frequency_positions`), while every
    line keeps its own frequency search width and bounds. The search first
    gives each coupled line to the name at its position; after every accepted
    frequency step, whose lines may have passed each other
    (:func:`_fit_parameter_grid`), the windows move with the lines
    (:func:`_move_frequency_windows`). Diagnostics row ``m`` of a repetition
    belongs to the name ``m`` when that step began.

    Returns amplitudes, frequencies, T2*, frequency diagnostics, T2*
    diagnostics, the shared Gaussian FWHM in Hz (``None`` for Lorentz fits)
    and the line permutation: slot ``k`` of the result holds the line that
    started in slot ``order[k]``, whose original frequency window the
    amplitude-filter refit must reuse. It is the identity without
    ``t2_coupling``.
    """
    options = fit_params.copy()
    # The positional T2* map is the working state, not a stale global alias.
    options.pop("initial_t2s_s", None)
    for key in ("metabs_freq_ranges", "metabs_t2_ranges"):
        options[key] = np.array(fit_params[key], dtype=float, copy=True)
    frequencies = np.array(options["metabs_freqs_hz"], copy=True)
    t2s = np.array(options["metabs_t2s"], copy=True)
    _validate_search_state(frequencies, t2s, options)
    voigt = "_gauss_fwhm_hz" in options
    positions = _coupled_frequency_positions(options)
    line_order = np.arange(len(frequencies))
    if positions:
        options["_frequency_position_rank"] = _frequency_position_rank(options)
        line_order = _coupled_line_order(frequencies, positions)
        frequencies = frequencies[line_order]
        _move_frequency_windows(options, line_order)

    def evaluate(freqs, decay, gauss_fwhm_hz=None):
        scoring_options = (
            options
            if gauss_fwhm_hz is None
            else {**options, "_gauss_fwhm_hz": gauss_fwhm_hz}
        )
        objective, amplitudes = _score_fit_parameters(
            data_slice, freqs, decay, scoring_options
        )
        if not np.all(np.isfinite(amplitudes)):
            raise FIDNumericalError("nonfinite_amplitudes")
        if not np.isfinite(objective) and not _violates_min_peak_separation(
            freqs, _get_min_peak_separation_hz(options)
        ):
            raise FIDNumericalError("nonfinite_objective")
        return _FitState(
            freqs.copy(),
            decay.copy(),
            amplitudes.copy(),
            objective,
            scoring_options.get("_gauss_fwhm_hz"),
        )

    state = evaluate(frequencies, t2s)
    history = options.get("_fit_history")

    def record(state, event, accepted):
        if history is not None:
            rss, _ = final_fit_errors(
                data_slice,
                state.amplitudes,
                state.frequencies,
                state.t2s,
                fit_options=options,
                gauss_fwhm_hz=state.gauss_fwhm_hz,
            )
            history.append(
                state.frequencies,
                state.t2s,
                state.amplitudes,
                rss,
                state.objective,
                options,
                event=event,
                accepted=accepted,
                evaluated=np.isfinite(state.objective),
                reason="none" if np.isfinite(state.objective) else "peak_separation",
            )

    record(state, "initial", np.isfinite(state.objective))

    zoomfactor = options.get("zoomfactor", ZOOMFACTOR_DEFAULT)
    fit_gauss = voigt and options["gauss_search_width_hz"] > 0
    if voigt:
        lorentz_widths_hz = np.where(
            options["metabs_t2_ranges"] > 0,
            1.0 / (np.pi * np.asarray(options["min_t2_s"], dtype=float))
            - 1.0 / (np.pi * np.asarray(options["max_t2_s"], dtype=float)),
            0.0,
        )
        gauss_width_hz = float(options["gauss_search_width_hz"])
    frequency_diagnostics, t2_diagnostics = [], []
    for repetition in range(int(options.get("rep_fitting", REP_FITTING_DEFAULT))):
        options["_fit_repetition"] = repetition
        if voigt:
            options["_gauss_fwhm_hz"] = state.gauss_fwhm_hz
        options["metabs_freqs_hz"] = state.frequencies.copy()
        options["metabs_t2s"] = state.t2s.copy()
        _, frequencies, _, diagnostic, step_order = _fit_parameter_grid(
            data_slice, options, parameter="frequency"
        )
        candidate = evaluate(frequencies, state.t2s)
        accepted = (
            np.isfinite(candidate.objective) and candidate.objective < state.objective
        )
        record(candidate, "frequency_step", accepted)
        if accepted:
            state = candidate
            if positions:
                line_order = line_order[step_order]
                _move_frequency_windows(options, step_order)
        frequency_diagnostics.append(diagnostic)
        options["metabs_freq_ranges"] /= zoomfactor

        options["metabs_freqs_hz"] = state.frequencies.copy()
        options["metabs_t2s"] = state.t2s.copy()
        if voigt:
            t2s, diagnostic = _fit_lorentz_fwhm_grid(
                data_slice, options, widths_hz=lorentz_widths_hz
            )
        else:
            _, t2s, _, diagnostic, _ = _fit_parameter_grid(
                data_slice, options, parameter="t2"
            )
        candidate = evaluate(state.frequencies, t2s)
        accepted = (
            np.isfinite(candidate.objective) and candidate.objective < state.objective
        )
        record(candidate, "t2_step", accepted)
        if accepted:
            state = candidate
        t2_diagnostics.append(diagnostic)
        if voigt:
            lorentz_widths_hz = lorentz_widths_hz / zoomfactor
        else:
            options["metabs_t2_ranges"] /= zoomfactor

        if fit_gauss:
            options["metabs_t2s"] = state.t2s.copy()
            t2s, gauss_fwhm_hz = _fit_gauss_fwhm_grid(
                data_slice,
                options,
                gauss_fwhm_hz=state.gauss_fwhm_hz,
                width_hz=gauss_width_hz,
            )
            candidate = evaluate(state.frequencies, t2s, gauss_fwhm_hz)
            if (
                np.isfinite(candidate.objective)
                and candidate.objective < state.objective
            ):
                state = candidate
            gauss_width_hz /= zoomfactor
    if voigt:
        options["_gauss_fwhm_hz"] = state.gauss_fwhm_hz

    if options.get("joint_refinement", JOINT_REFINEMENT_DEFAULT):
        proposal = _joint_refinement_proposal(data_slice, state, options)
        if proposal is not None:
            candidate = evaluate(*proposal, state.gauss_fwhm_hz)
            accepted = (
                np.isfinite(candidate.objective)
                and candidate.objective < state.objective
                and _inside_t2_bands(candidate.t2s, _t2_band_pairs(options))
                and not (
                    positions
                    and np.any(
                        _coupled_line_order(candidate.frequencies, positions)
                        != np.arange(len(candidate.frequencies))
                    )
                )
            )
            record(candidate, "joint_refinement", accepted)
            if accepted:
                state = candidate

    verified = evaluate(state.frequencies, state.t2s)
    consistent = np.isclose(
        verified.objective, state.objective, rtol=1e-10, atol=1e-12
    ) and np.allclose(verified.amplitudes, state.amplitudes, rtol=1e-10, atol=1e-12)
    record(verified, "verified", np.isfinite(verified.objective) and consistent)
    if not np.isfinite(verified.objective):
        raise FIDNumericalError("no_feasible_state")
    if not consistent:
        raise FIDNumericalError("inconsistent_final_state")
    return (
        verified.amplitudes,
        verified.frequencies,
        verified.t2s,
        np.stack(frequency_diagnostics, axis=1),
        np.stack(t2_diagnostics, axis=1),
        verified.gauss_fwhm_hz,
        line_order,
    )


def _select_metabolite_options(fit_options: dict, indices: np.ndarray) -> dict:
    """Copy runtime options for a subset in the supplied metabolite order.

    Keep global options and select metabolite names. Direct numerical callers
    may also supply the grid searches' ``initial_t2s_s`` alias; select its values
    consistently so a skipped or thresholded component cannot shift T2* values
    onto the wrong retained peak. A supplied ``_frequency_position_rank`` is
    selected the same way, so the remaining coupled lines keep their relative
    frequency positions. Positional parameter arrays are sliced by the caller.
    Neither the original options nor their arrays are modified.
    """
    selected = fit_options.copy()
    parent_indices = np.asarray(
        fit_options.get("_fit_indices", np.arange(len(fit_options["metabs"])))
    )
    selected["_fit_indices"] = parent_indices[indices]
    selected["metabs"] = [fit_options["metabs"][i] for i in indices]
    if fit_options.get("_frequency_position_rank") is not None:
        selected["_frequency_position_rank"] = np.asarray(
            fit_options["_frequency_position_rank"]
        )[indices]
    if selected.get("initial_t2s_s") is not None:
        selected["initial_t2s_s"] = _coerce_metabolite_values(
            selected["initial_t2s_s"],
            len(fit_options["metabs"]),
            parameter_name="initial_t2s_s",
        )[indices]
    return selected


def _empty_single_spectrum_result(
    initial_freq_hz: np.ndarray,
    initial_t2_s: np.ndarray,
    fit_options: dict,
    initial_gauss_fwhm_hz: float | None = None,
) -> tuple:
    """Return six outputs for components that have not been fitted.

    Keep initial guesses and metabolite order, with zero complex amplitudes,
    NaN search diagnostics and false amplitude-filter flags. The same template
    serves empty signals and skipped metabolites; callers can insert fitted
    subsets into these newly allocated arrays without changing their inputs.
    A given ``initial_gauss_fwhm_hz`` (Voigt fits) is appended unchanged as a
    seventh output.
    """
    total_metabs = len(initial_freq_hz)
    diagnostic_shape = (
        total_metabs,
        fit_options.get("rep_fitting", REP_FITTING_DEFAULT),
        fit_options.get("niter", NITER_DEFAULT),
        fit_options.get("npoints", NPOINTS_DEFAULT),
        3,
    )
    return (
        np.zeros(total_metabs, dtype=np.complex128),
        initial_freq_hz.copy(),
        initial_t2_s.copy(),
        np.full(diagnostic_shape, np.nan),
        np.full(diagnostic_shape, np.nan),
        np.zeros(total_metabs, dtype=bool),
    ) + (() if initial_gauss_fwhm_hz is None else (float(initial_gauss_fwhm_hz),))


def _fit_single_spectrum_pseudo_inv(
    data_slice: np.ndarray,
    initial_freq_hz: np.ndarray,
    initial_t2_s: np.ndarray,
    freq_search_range_hz: np.ndarray,
    t2_search_range_s: np.ndarray,
    min_t2_s: np.ndarray,
    max_t2_s: np.ndarray,
    freq_search_range_lower_bound_hz: np.ndarray,
    freq_search_range_upper_bound_hz: np.ndarray,
    skip_fitting_mask: bool | np.ndarray | None = None,
    initial_gauss_fwhm_hz: float | None = None,
    **kwargs,
) -> tuple:
    """Prepare one spectrum, refine its parameters and optionally refit strong peaks.

    ``data_slice`` is a one-dimensional FID or centered spectrum. The parameter
    arrays contain one value per metabolite: frequencies and their full search
    widths/bounds in Hz; T2*, full widths and bounds in seconds. Legacy complex
    parameter arrays use their real parts. Global ``kwargs`` supply metabolite
    names, sampling times, input domain, search counts and amplitude options.

    Match the input to the sampling length, then run alternating frequency/T2*
    searches. If positive ``snr_threshold`` is given, flag amplitudes strictly
    below this absolute threshold. Refit retained metabolites once, starting at
    the fitted guesses but using the original search widths and bounds of the
    same lines (with ``t2_coupling`` a window stays with its line). Disable
    thresholding for that refit; excluded amplitudes become zero, while their
    fitted frequencies, T2* and diagnostics remain available. All-zero input
    bypasses searching with zero amplitudes. Any nonfinite input sample yields
    NaN active amplitudes without dropping samples or reporting zero signal.
    Both shortcuts retain initial guesses, NaN diagnostics and false flags.

    Return amplitudes, frequencies, T2*, frequency diagnostics, T2* diagnostics
    and exclusion flags in metabolite order. Diagnostic axes are
    ``(metabolite, repetition, iteration, candidate, 3)``. Inputs are not modified.

    ``skip_fitting_mask`` accepts a boolean scalar or one boolean per metabolite
    (0/1 also accepted). True removes that component before amplitude solving,
    parameter searches and peak-separation checks. Return zero amplitude, initial
    guesses, NaN diagnostics and a false amplitude-filter flag for skipped peaks.
    Fit the active subset using the same workflow, including its optional
    amplitude filter. Restore all six outputs to the original metabolite order.
    If every component is skipped, bypass numerical searching entirely.

    With ``lineshape='voigt'`` in ``kwargs``, ``initial_gauss_fwhm_hz`` is the
    required start value of the shared Gaussian FWHM in Hz, within
    ``[0, gauss_max_fwhm_hz]``; ``initial_t2_s`` and the T2* outputs then
    describe the Lorentz part only. A seventh output returns the fitted shared
    Gaussian FWHM in Hz; the shortcuts above return the start value. The
    amplitude-filter refit starts from the fitted Gaussian FWHM. For nonfinite
    input the start value is not checked and may itself be NaN, because no
    search runs. Lorentz fits reject ``initial_gauss_fwhm_hz``.

    With ``t2_coupling`` in ``kwargs``, every fixed dependent starts at, and
    returns, its leader's T2* times ``t2_ratio``; its own initial T2* is not
    used. A banded dependent starts at its own initial T2*, which must lie in
    its ratio band. A skipped or amplitude-filtered leader hands its group to
    the first remaining fixed dependent, as described in
    ``_promote_t2_coupling``. Names, T2* values and roles belong to frequency
    positions and each line keeps its frequency window
    (``_run_alternating_search``); the private ``_frequency_position_rank``
    in ``kwargs`` fixes the position order, otherwise ``initial_freq_hz``
    defines it.
    """
    data_slice, kwargs = _prepare_single_spectrum_data(data_slice, kwargs)
    initial_freq_hz = np.asarray(np.real(initial_freq_hz), dtype=np.float64)
    initial_t2_s = _apply_t2_coupling(np.real(initial_t2_s), kwargs)
    freq_search_range_hz = np.asarray(np.real(freq_search_range_hz), dtype=np.float64)
    t2_search_range_s = np.asarray(np.real(t2_search_range_s), dtype=np.float64)
    min_t2_s = np.asarray(np.real(min_t2_s), dtype=np.float64)
    max_t2_s = np.asarray(np.real(max_t2_s), dtype=np.float64)
    freq_search_range_lower_bound_hz = np.asarray(
        np.real(freq_search_range_lower_bound_hz),
        dtype=np.float64,
    )
    freq_search_range_upper_bound_hz = np.asarray(
        np.real(freq_search_range_upper_bound_hz),
        dtype=np.float64,
    )
    skip_fitting_mask = _coerce_skip_fitting_mask(
        skip_fitting_mask, len(initial_freq_hz)
    )
    voigt = _uses_voigt_lineshape(kwargs)
    data_is_finite = bool(np.all(np.isfinite(data_slice)))
    if voigt and not data_is_finite and initial_gauss_fwhm_hz is not None:
        # Start maps may already be NaN at voxels whose input is not finite
        # (mask_fit_dataset_where_input_nonfinite); such a voxel is not fitted.
        initial_gauss_fwhm_hz = float(np.real(np.asarray(initial_gauss_fwhm_hz)))
    elif voigt:
        if initial_gauss_fwhm_hz is None:
            raise ValueError(
                "lineshape='voigt' requires initial_gauss_fwhm_hz, the start "
                "Gaussian FWHM in Hz; prepare parameters with def_fit_params."
            )
        initial_gauss_fwhm_hz = _coerce_finite_scalar(
            initial_gauss_fwhm_hz, parameter_name="initial_gauss_fwhm_hz"
        )
        gauss_max_fwhm_hz = float(
            kwargs.get("gauss_max_fwhm_hz", GAUSS_MAX_FWHM_HZ_DEFAULT)
        )
        if not 0 <= initial_gauss_fwhm_hz <= gauss_max_fwhm_hz:
            raise ValueError(
                f"initial_gauss_fwhm_hz={initial_gauss_fwhm_hz} Hz must lie in "
                f"[0, gauss_max_fwhm_hz={gauss_max_fwhm_hz}] Hz."
            )
    elif initial_gauss_fwhm_hz is not None:
        raise ValueError(
            "initial_gauss_fwhm_hz is only used with lineshape='voigt'; remove "
            "it or set lineshape='voigt'."
        )

    if not data_is_finite:
        result = _empty_single_spectrum_result(
            initial_freq_hz, initial_t2_s, kwargs, initial_gauss_fwhm_hz
        )
        # A spectrum with any NaN or infinite sample (real or imaginary part)
        # cannot be fitted: report every active parameter as NaN instead of
        # echoing the start values, which would look like a converged fit.
        active = ~skip_fitting_mask
        result[0][active] = complex(np.nan, np.nan)
        result[1][active] = np.nan
        result[2][active] = np.nan
        if voigt and np.any(active):
            return (*result[:6], np.nan)
        return result
    if np.all(data_slice == 0):
        return _empty_single_spectrum_result(
            initial_freq_hz, initial_t2_s, kwargs, initial_gauss_fwhm_hz
        )

    if np.any(skip_fitting_mask):
        result = _empty_single_spectrum_result(
            initial_freq_hz, initial_t2_s, kwargs, initial_gauss_fwhm_hz
        )
        active_indices = np.flatnonzero(~skip_fitting_mask)
        if not active_indices.size:
            return result

        subset_kwargs, subset_t2_ranges_s = _promote_t2_coupling(
            kwargs, ~skip_fitting_mask, t2_search_range_s
        )
        active_kwargs = _select_metabolite_options(subset_kwargs, active_indices)
        active_result = _fit_single_spectrum_pseudo_inv(
            data_slice=data_slice,
            initial_freq_hz=initial_freq_hz[active_indices],
            initial_t2_s=initial_t2_s[active_indices],
            freq_search_range_hz=freq_search_range_hz[active_indices],
            t2_search_range_s=subset_t2_ranges_s[active_indices],
            min_t2_s=min_t2_s[active_indices],
            max_t2_s=max_t2_s[active_indices],
            freq_search_range_lower_bound_hz=freq_search_range_lower_bound_hz[
                active_indices
            ],
            freq_search_range_upper_bound_hz=freq_search_range_upper_bound_hz[
                active_indices
            ],
            initial_gauss_fwhm_hz=initial_gauss_fwhm_hz,
            **active_kwargs,
        )
        for full_values, active_values in zip(
            result[:6], active_result[:6], strict=True
        ):
            full_values[active_indices] = active_values
        result[2][:] = _apply_t2_coupling(result[2], subset_kwargs)
        if voigt:
            return (*result[:6], active_result[6])
        return result

    fit_params_voxel = kwargs.copy()
    fit_params_voxel.update(
        {
            "metabs_freqs_hz": initial_freq_hz,
            "metabs_t2s": initial_t2_s,
            "metabs_freq_ranges": freq_search_range_hz,
            "metabs_t2_ranges": t2_search_range_s,
            "min_t2_s": min_t2_s,
            "max_t2_s": max_t2_s,
            "freq_range_lower_bound_hz": freq_search_range_lower_bound_hz,
            "freq_range_upper_bound_hz": freq_search_range_upper_bound_hz,
        }
    )
    if voigt:
        fit_params_voxel["_gauss_fwhm_hz"] = initial_gauss_fwhm_hz

    (
        best_fit_amps,
        best_freq_hz,
        best_t2_s,
        combined_r2_freq,
        combined_r2_t2,
        best_gauss_fwhm_hz,
        line_order,
    ) = _run_alternating_search(data_slice, fit_params_voxel)
    low_snr_refitted = np.zeros(len(best_freq_hz), dtype=bool)

    # This is an absolute fitted-amplitude threshold, not a signal/noise ratio.
    # Metabolites below it are removed and the remaining metabolites are fitted
    # once more so that a weak component cannot bias their final parameters.
    snr_threshold = fit_params_voxel.get("snr_threshold")
    if snr_threshold is not None and float(snr_threshold) > 0:
        low_snr_refitted = np.abs(best_fit_amps) < float(snr_threshold)
        high_snr_indices = np.flatnonzero(~low_snr_refitted)

        if np.any(low_snr_refitted) and high_snr_indices.size:
            promoted_kwargs, refit_t2_ranges_s = _promote_t2_coupling(
                kwargs, ~low_snr_refitted, t2_search_range_s
            )
            refit_kwargs = _select_metabolite_options(promoted_kwargs, high_snr_indices)
            refit_kwargs["snr_threshold"] = None
            refit_kwargs["_fit_round"] = 1
            refit_result = _fit_single_spectrum_pseudo_inv(
                data_slice=data_slice,
                initial_freq_hz=best_freq_hz[high_snr_indices],
                initial_t2_s=best_t2_s[high_snr_indices],
                freq_search_range_hz=freq_search_range_hz[line_order][high_snr_indices],
                t2_search_range_s=refit_t2_ranges_s[high_snr_indices],
                min_t2_s=min_t2_s[high_snr_indices],
                max_t2_s=max_t2_s[high_snr_indices],
                freq_search_range_lower_bound_hz=(
                    freq_search_range_lower_bound_hz[line_order][high_snr_indices]
                ),
                freq_search_range_upper_bound_hz=(
                    freq_search_range_upper_bound_hz[line_order][high_snr_indices]
                ),
                initial_gauss_fwhm_hz=best_gauss_fwhm_hz,
                **refit_kwargs,
            )
            refit_amps, refit_freqs, refit_t2s, refit_r2_freq, refit_r2_t2, _ = (
                refit_result[:6]
            )
            if voigt:
                best_gauss_fwhm_hz = refit_result[6]
            best_fit_amps[high_snr_indices] = refit_amps
            best_freq_hz[high_snr_indices] = refit_freqs
            best_t2_s[high_snr_indices] = refit_t2s
            combined_r2_freq[high_snr_indices] = refit_r2_freq
            combined_r2_t2[high_snr_indices] = refit_r2_t2
            # An excluded dependent reports the T2* implied by its refitted leader.
            best_t2_s = _apply_t2_coupling(best_t2_s, promoted_kwargs)

        best_fit_amps[low_snr_refitted] = 0.0

    result = (
        best_fit_amps,
        best_freq_hz,
        best_t2_s,
        combined_r2_freq,
        combined_r2_t2,
        low_snr_refitted,
    )
    if voigt:
        return (*result, best_gauss_fwhm_hz)
    return result


def fid_model(
    amplitude: xr.DataArray,
    freq_hz: xr.DataArray,
    t2_star_s: xr.DataArray,
    time_axis_s: np.ndarray | None = None,
    *,
    gauss_fwhm_hz: float | xr.DataArray | None = None,
    **kwargs,
) -> xr.DataArray:
    """Reconstruct metabolite FIDs from fit results.

    Parameters
    ----------
    amplitude, freq_hz, t2_star_s : xr.DataArray
        Complex line areas at ``t = 0``, frequencies in Hz and T2* in seconds
        (the Lorentz part for Voigt fits).
    time_axis_s : numpy.ndarray, optional
        Evaluation times in seconds. Defaults to
        ``kwargs['sampling_time_axis_s']``.
    gauss_fwhm_hz : float or xr.DataArray, optional
        Shared Gaussian FWHM in Hz of a Voigt fit, for example
        ``fit_results['fit_gauss_fwhm_hz']``. ``None`` gives Lorentz lines.
    **kwargs
        Legacy fit options; only ``sampling_time_axis_s`` is read.

    Returns
    -------
    xr.DataArray
        Components as returned by :func:`reconstruct_fid_signal`.

    Raises
    ------
    ValueError
        If no sampling axis is available.

    Notes
    -----
    The labelled reconstruction preserves parameter dimensions/coordinates and
    input time orientation. Dtype follows xarray/NumPy input promotion, with no
    storage conversion. Evaluation is eager for NumPy inputs and keeps xarray's
    array backend for labelled inputs. Fixed inputs are deterministic; no input
    is mutated and no result cache is retained. Unknown legacy keyword entries
    retain their historical ignored behavior.

    Examples
    --------
    >>> components = fid_model(areas, frequencies, decays, time_axis_s=times_s)
    ... # doctest: +SKIP"""
    if time_axis_s is None:
        time_axis_s = kwargs.get("sampling_time_axis_s")

    if time_axis_s is None:
        raise ValueError("FID reconstruction requires a sampling axis `time_axis_s`.")

    return reconstruct_fid_signal(
        amplitude=amplitude,
        freq_hz=freq_hz,
        t2_star_s=t2_star_s,
        time_axis_s=time_axis_s,
        gauss_fwhm_hz=gauss_fwhm_hz,
    )


def reconstruct_fid_signal(
    *,
    amplitude: xr.DataArray,
    freq_hz: xr.DataArray,
    t2_star_s: xr.DataArray,
    time_axis_s: np.ndarray | xr.DataArray,
    gauss_fwhm_hz: float | xr.DataArray | None = None,
) -> xr.DataArray:
    """Evaluate individual metabolite FIDs from fitted parameter maps.

    Parameters
    ----------
    amplitude, freq_hz, t2_star_s : xr.DataArray
        Complex amplitudes, frequencies in Hz and T2* values in seconds.
    time_axis_s : np.ndarray or xr.DataArray
        Model evaluation times in seconds. A DataArray preserves its dimension
        name and sample labels. A NumPy array retains the legacy convention:
        dimension ``acquisition_idx`` with the time values as its labels.
    gauss_fwhm_hz : float or xr.DataArray, optional
        Shared Gaussian FWHM in Hz of a Voigt fit, without a metabolite
        dimension. ``None`` evaluates the Lorentz expression unchanged.

    Returns
    -------
    xr.DataArray
        Components ``amplitude * exp(2j*pi*freq_hz*t) * exp(-t/t2_star_s)``,
        multiplied by ``exp(-(pi*gauss_fwhm_hz*t)**2 / (4*ln 2))`` for Voigt
        fits. The metabolite dimension is retained; no components are summed.

    Raises
    ------
    ValueError
        If labelled arrays cannot align or the Gaussian width is negative or
        non-finite.

    Notes
    -----
    Dimension order follows the existing parameter/time xarray broadcasting.
    Coordinate values/direction and signal units are preserved; dtype follows
    input promotion without an added precision conversion. NumPy inputs are
    computed eagerly, while labelled array backends are retained. This pure,
    deterministic operation neither mutates inputs nor caches results.

    Examples
    --------
    >>> components = reconstruct_fid_signal(
    ...     amplitude=areas, freq_hz=frequencies, t2_star_s=decays,
    ...     time_axis_s=times_s
    ... )  # doctest: +SKIP"""
    if isinstance(time_axis_s, xr.DataArray):
        time_axis_da = time_axis_s
    else:
        time_axis_da = xr.DataArray(
            np.asarray(time_axis_s),
            dims=["acquisition_idx"],
            coords={"acquisition_idx": np.asarray(time_axis_s)},
        )
    components = (
        amplitude
        * np.exp(1j * 2.0 * np.pi * freq_hz * time_axis_da)
        * np.exp(-time_axis_da / t2_star_s)
    )
    if gauss_fwhm_hz is None:
        return components
    return components * _gaussian_decay(time_axis_da, gauss_fwhm_hz)


def final_fit_errors(
    data, amplitudes, frequencies, t2s, *, fit_options, gauss_fwhm_hz=None
):
    """Reconstruct the returned joint state on exactly the samples used to fit.

    Parameters
    ----------
    data : numpy.ndarray
        One FID or centred spectrum, as described by ``signal_domain``.
    amplitudes : numpy.ndarray
        Complex line areas at ``t = 0``, one per metabolite.
    frequencies : numpy.ndarray
        Frequencies in Hz, one per metabolite.
    t2s : numpy.ndarray
        T2* in seconds (the Lorentz part for Voigt fits), one per metabolite.
    fit_options : dict
        Global options with ``sampling_time_axis_s``; not modified.
    gauss_fwhm_hz : float, optional
        Shared Gaussian FWHM in Hz of a Voigt fit. ``None`` evaluates the
        Lorentz model exactly as before.

    Returns
    -------
    rss : float
        ``sum(abs(measured_FID - fitted_FID)**2)``; NaN for non-finite input.
    objective : float
        The RSS, or the spectral magnitude RSS with ``use_sse_mag``.

    Raises
    ------
    ValueError
        If the scientific options, sampling or parameter/data shapes are invalid.

    Notes
    -----
    RSS has squared signal units and uses complex128 fitting arithmetic. Input
    sample order is preserved; the operation is eager, deterministic and does
    not mutate inputs or retain a cache. Non-finite data/areas return NaN errors.

    Examples
    --------
    >>> rss, objective = final_fit_errors(fid, areas, frequencies, decays, options)
    ... # doctest: +SKIP"""
    data, options = _prepare_single_spectrum_data(data, fit_options)
    if str(options.get("signal_domain", "temporal")).lower() == "spectral":
        data = np.fft.ifft(np.fft.ifftshift(data))
    if not np.all(np.isfinite(data)) or not np.all(np.isfinite(amplitudes)):
        return np.nan, np.nan
    basis = fid_basefunc(
        frequencies,
        t2s,
        options["sampling_time_axis_s"],
        gauss_fwhm_hz=gauss_fwhm_hz,
    )
    fitted = basis.T @ amplitudes
    rss = float(np.sum(np.abs(data - fitted) ** 2))
    if options.get("use_sse_mag", False):
        objective = float(
            np.sum((np.abs(np.fft.fft(data)) - np.abs(np.fft.fft(fitted))) ** 2)
        )
    else:
        objective = rss
    return rss, objective


__all__ = [
    "count_free_parameters",
    "fitted_sample_count",
    "MODEL_ID",
    "RESULT_MODEL_NAME",
    "DEFAULT_MIN_PEAK_SEPARATION_HZ",
    "NPOINTS_DEFAULT",
    "NITER_DEFAULT",
    "REP_FITTING_DEFAULT",
    "MIN_T2_S_DEFAULT",
    "MAX_T2_S_DEFAULT",
    "METABS_FREQ_RANGE_DEFAULT",
    "METABS_T2_RANGE_DEFAULT",
    "INITIAL_T2_S_DEFAULT",
    "PSEUDO_INV_DEFAULT",
    "SNR_THRESHOLD_DEFAULT",
    "USE_SSE_MAG_DEFAULT",
    "ZOOMFACTOR_DEFAULT",
    "SHOW_TQDM_DEFAULT",
    "UNCERTAINTY_BOOTSTRAP_SAMPLES_DEFAULT",
    "UNCERTAINTY_CONFIDENCE_LEVEL_DEFAULT",
    "UNCERTAINTY_SEED_DEFAULT",
    "LINESHAPES",
    "LINESHAPE_DEFAULT",
    "SEARCH_ORDERS",
    "SEARCH_ORDER_DEFAULT",
    "JOINT_REFINEMENT_DEFAULT",
    "GAUSS_INITIAL_FWHM_HZ_DEFAULT",
    "GAUSS_MAX_FWHM_HZ_DEFAULT",
    "GAUSS_SEARCH_WIDTH_HZ_DEFAULT",
    "VOIGT_MAX_T2_S_DEFAULT",
    "get_fid_pseudo_inv_params_info",
    "def_fit_params",
    "fid_basefunc",
    "solve_for_complex_amplitudes",
    "fit_freq_pseudo_inv",
    "fit_t2_pseudo_inv",
    "fid_model",
    "reconstruct_fid_signal",
    "final_fit_errors",
]
