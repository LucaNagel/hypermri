"""Spectroscopy fitting workflows for labelled xarray data.

This module owns parameter alignment, serial/parallel execution and result
assembly, fit reconstruction and iterative parameter updates. The generic
fitting API and legacy imports delegate here; the numerical FID model and
single-spectrum solver live in ``models.spectroscopy.fid_model``.
"""

from __future__ import annotations

import json
import warnings
from collections.abc import Sequence
from typing import TypeAlias

import numpy as np
import xarray as xr

from hypermri.analysis.spectroscopy.frequency import _convert_ppm_value, freq_hz_to_ppm
from hypermri.analysis.spectroscopy.frequency import _ppm_delta_to_hz as _delta_ppm2hz
from hypermri.compatibility import assign_frequency_axis, resolve_frequency_axis

from .models.spectroscopy.diagnostics import (
    FIT_FAILURE_REASONS,
    FIT_STATUSES,
    HISTORY_EVENTS,
    HISTORY_FIELDS,
    HISTORY_REASONS,
    FIDNumericalError,
    FitHistory,
    history_capacity,
)
from .models.spectroscopy.fid_model import (
    _GAUSS_BOUND_TOLERANCE_HZ,
    DEFAULT_MIN_PEAK_SEPARATION_HZ,
    INITIAL_T2_S_DEFAULT,
    LINESHAPE_DEFAULT,
    MAGNITUDE_LORENTZ_FWHM_FACTOR,
    MAX_T2_S_DEFAULT,
    METABS_FREQ_RANGE_DEFAULT,
    METABS_T2_RANGE_DEFAULT,
    MIN_T2_S_DEFAULT,
    NITER_DEFAULT,
    NPOINTS_DEFAULT,
    REP_FITTING_DEFAULT,
    VOIGT_MAX_T2_S_DEFAULT,
    _coerce_finite_scalar,
    _coerce_metabolite_values,
    _coerce_skip_fitting_mask,
    _derive_frequency_bounds,
    _empty_single_spectrum_result,
    _fit_single_spectrum_pseudo_inv,
    _magnitude_fwhm_hz,
    _normalize_t2_coupling,
    _prepare_single_spectrum_data,
    _require_same_lineshape,
    _update_frequency_bounds,
    _uses_voigt_lineshape,
    _voigt_fwhm_hz,
    count_free_parameters,
    fid_basefunc,
    fid_model,
    final_fit_errors,
    fitted_sample_count,
    reconstruct_fid_signal,
    solve_for_complex_amplitudes,
)
from .models.spectroscopy.fid_model import (
    def_fit_params as _def_fit_params_impl,
)
from .models.spectroscopy.model_comparison import (
    DETECTION_REASONS,
    DETECTION_STATUSES,
    annotate_detection_result,
    compare_single_spectrum,
    detection_schema,
    source_provenance,
)
from .result_metadata import inherit_input_xarray_metadata

__all__ = [
    "_fit_single_spectrum_pseudo_inv",
    "add_metab_to_fit_params",
    "def_fit_params",
    "fid_basefunc",
    "fid_model",
    "fit_best_of_starts",
    "fit_data_pseudo_inv",
    "initialise_fit_params_from_results",
    "mask_fit_dataset_where_input_nonfinite",
    "phase_fit_results",
    "reconstruct_fitted_signal",
    "refit_gauss_width_from_neighbours",
    "remove_metab_from_fit_params",
    "solve_for_complex_amplitudes",
]

FitSingleSpectrumResult: TypeAlias = tuple[np.ndarray, ...]

_GLOBAL_FIT_CONSTRAINT_NAMES = ("min_peak_separation_hz",)


def _fit_output_schema(options):
    """Define names, core dimensions and dtypes once for every execution path."""
    grid_dims = ["metabolite", "rep_fitting", "niter", "npoints", "r2_data"]
    schema = {
        "fit_amps": (["metabolite"], np.complex128),
        "fit_freqs": (["metabolite"], np.float64),
        "fit_t2s": (["metabolite"], np.float64),
        "fit_r2_freq": (grid_dims, np.float64),
        "fit_r2_t2": (grid_dims, np.float64),
        "fit_low_snr_refitted": (["metabolite"], bool),
        **(
            {"fit_gauss_fwhm_hz": ([], np.float64)}
            if _uses_voigt_lineshape(options)
            else {}
        ),
        "fit_status": ([], np.int16),
        "fit_failure_reason": ([], np.int16),
        "fit_rss": ([], np.float64),
        "fit_objective": ([], np.float64),
    }
    if options.get("store_optimization_history", False):
        schema.update(
            {
                "fit_history_count": ([], np.int64),
                "fit_history_amps": (["history_step", "metabolite"], np.complex128),
                "fit_history_freqs": (["history_step", "metabolite"], np.float64),
                "fit_history_t2s": (["history_step", "metabolite"], np.float64),
                "fit_history_active": (["history_step", "metabolite"], bool),
                "fit_history_rss": (["history_step"], np.float64),
                "fit_history_objective": (["history_step"], np.float64),
                "fit_history_info": (["history_step", "history_field"], np.int32),
            }
        )
    if options.get("detection_target") is not None:
        schema.update(detection_schema(options))
    return schema


def _fit_output_sizes(options, n_metabolites):
    sizes = {
        "metabolite": n_metabolites,
        "rep_fitting": int(options.get("rep_fitting", REP_FITTING_DEFAULT)),
        "niter": int(options.get("niter", NITER_DEFAULT)),
        "npoints": int(options.get("npoints", NPOINTS_DEFAULT)),
        "r2_data": 3,
    }
    if options.get("store_optimization_history", False):
        sizes.update(
            history_step=history_capacity(options, n_metabolites),
            history_field=len(HISTORY_FIELDS),
        )
    if options.get("detection_target") is not None:
        sizes["sha256_byte"] = 32
        if options.get("detection_store_null", False):
            sizes["null_replicate"] = options["detection_null_samples"]
    return sizes


def _fit_single_spectrum_outputs(data, *parameter_maps, **options):
    """Use the same numerical/comparison entry point for every execution path."""
    if options.get("detection_target") is not None:
        return compare_single_spectrum(
            data,
            parameter_maps[:9],
            int(parameter_maps[9]),
            {**options, "detection_sigma": float(parameter_maps[10])},
            _fit_single_spectrum_numerical_outputs,
        )
    return _fit_single_spectrum_numerical_outputs(data, *parameter_maps, **options)


def _fit_single_spectrum_numerical_outputs(data, *parameter_maps, **options):
    """Preserve the six-output solver API and append status/optional full history.

    Voigt fits pass the start Gaussian FWHM as ``parameter_maps[9]`` and keep
    the solver's seventh output, the fitted Gaussian FWHM in Hz, before the
    status fields. User configuration errors propagate. Numerical failures affect only this
    spectrum and return NaN active amplitudes, never a zero-signal result.
    No object holding diagnostics is shared across spectra or workers.
    """
    frequencies, t2s = (
        np.asarray(np.real(values), dtype=float) for values in parameter_maps[:2]
    )
    skipped = _coerce_skip_fitting_mask(parameter_maps[8], len(frequencies))
    data_prefix, _ = _prepare_single_spectrum_data(data, options)
    history = (
        FitHistory(frequencies, t2s, options)
        if options.get("store_optimization_history")
        else None
    )
    run_options = {**options, "_fit_indices": np.arange(len(frequencies))}
    if history is not None:
        run_options["_fit_history"] = history
    status, reason = "success", "none"
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            result = _fit_single_spectrum_pseudo_inv(
                data, *parameter_maps, **run_options
            )
            if not np.all(np.isfinite(data_prefix)):
                status, reason = "invalid_input", "nonfinite_samples"
            elif np.all(skipped):
                status = "skipped"
            elif np.all(data_prefix == 0):
                status = "zero_signal"
            elif not np.all(np.isfinite(result[0])):
                raise FIDNumericalError("nonfinite_amplitudes")
            rss, objective = final_fit_errors(
                data,
                *result[:3],
                fit_options=options,
                gauss_fwhm_hz=result[6] if _uses_voigt_lineshape(options) else None,
            )
            if status != "invalid_input" and not np.all(np.isfinite([rss, objective])):
                raise FIDNumericalError("nonfinite_objective")
    except (FIDNumericalError, np.linalg.LinAlgError, FloatingPointError) as exc:
        status = "fit_failed"
        reason = (
            exc.reason
            if isinstance(exc, FIDNumericalError)
            else (
                "linear_algebra_error"
                if isinstance(exc, np.linalg.LinAlgError)
                else "nonfinite_objective"
            )
        )
        result = _empty_single_spectrum_result(
            frequencies,
            t2s,
            options,
            parameter_maps[9] if _uses_voigt_lineshape(options) else None,
        )
        result[0][~skipped] = complex(np.nan, np.nan)
        rss, objective = np.nan, np.nan
    outputs = result + (
        np.int16(FIT_STATUSES.index(status)),
        np.int16(FIT_FAILURE_REASONS.index(reason)),
        rss,
        objective,
    )
    if history is not None:
        active = ~(skipped | result[5])
        final_options = {
            **options,
            "_fit_indices": np.flatnonzero(active),
            "_fit_round": int(np.any(result[5])),
        }
        history.append(
            result[1][active],
            result[2][active],
            result[0][active],
            rss,
            objective,
            final_options,
            event="returned",
            accepted=status in {"success", "zero_signal", "skipped"},
            evaluated=np.isfinite(objective),
        )
        # Preserve the actual returned parameters, including excluded components.
        history.amplitudes[history.count - 1] = result[0]
        history.frequencies[history.count - 1] = result[1]
        history.t2s[history.count - 1] = result[2]
        outputs += history.outputs()
    return outputs


# Parameter maps in the positional order required by the single-spectrum solver.
_FIT_PARAM_MAP_NAMES = (
    "initial_freqs_hz",
    "initial_t2s_s",
    "metabs_freq_ranges",
    "metabs_t2_ranges",
    "min_t2_s",
    "max_t2_s",
    "freq_range_lower_bound_hz",
    "freq_range_upper_bound_hz",
    "skip_fitting",
)


_FIT_SINGLE_SPECTRUM_POSITIONAL_ARG_NAMES = frozenset(
    {
        "data_slice",
        "initial_freq_hz",
        "initial_t2_s",
        "freq_search_range_hz",
        "t2_search_range_s",
        "min_t2_s",
        "max_t2_s",
        "freq_search_range_lower_bound_hz",
        "freq_search_range_upper_bound_hz",
        "skip_fitting_mask",
        "initial_gauss_fwhm_hz",
    }
)


def def_fit_params(
    data: xr.DataArray = None,
    fit_params: dict = None,
    return_params_info: bool = False,
    mask_nonfinite_input: bool = True,
) -> xr.Dataset | None:
    """Prepare metabolite parameter maps and global spectroscopy fit options.

    The preparation is deterministic. Besides the sampling coordinates and
    the shape of ``data`` it reads the sample values only to find voxels
    with a NaN or infinite sample (this computes dask-backed input).

    Parameters
    ----------
    data : xr.DataArray
        Spectroscopy data with a sampling-time coordinate in seconds. The loop
        dimensions (all except the spectral dimension) define the shape of the
        returned parameter maps.
    fit_params : dict, optional
        Metabolites and fitting options. The accepted keys are those listed by
        ``get_fid_pseudo_inv_params_info()["optional_fit_params"]`` and
        ``["required_fit_params"]``, the notebook spelling ``metabs_freqs_Hz``
        and the markers ``model_name`` and ``fit_type`` that this function
        writes, so ``dict(prepared.attrs)`` can be prepared again. Use
        ``metabs_freqs_hz`` for frequencies in Hz; neither Hz spelling requires
        a ppm axis. Explicit ``metabs_freqs_ppm`` values take precedence and
        require a calibrated ``freq_axis_ppm`` coordinate. T2* guesses in
        seconds must lie within positive, ordered bounds. Search widths must be
        finite and non-negative; zero freezes the parameter. These settings
        accept a scalar or one entry per metabolite. The mapping is copied and
        not mutated.
    return_params_info : bool, optional
        Return None without preparing parameters when True.
    mask_nonfinite_input : bool, optional
        When True (default), every float and complex parameter map is NaN in
        voxels where ``data`` has a NaN or infinite sample in any part, as
        ``mask_fit_dataset_where_input_nonfinite`` does. Such voxels are not
        fitted (``fit_status`` ``invalid_input``) and their results are NaN,
        so start values, ranges and bounds then match the results. False
        keeps the configured values in every voxel.

    Returns
    -------
    xr.Dataset or None
        Parameter maps with a ``metabolite`` dimension and the loop dimensions
        of ``data``, and the effective options as attributes, ready for
        ``fit_data_pseudo_inv``. None when ``return_params_info`` is True.

    Raises
    ------
    ValueError
        Before the numerical search starts, if ``fit_params`` contains an
        unknown key (the message names it, suggests the closest valid key and
        lists all valid keys), the Hz aliases conflict, a ppm value lacks a
        calibrated axis, or a setting is outside its valid range.

    Examples
    --------
    A misspelled option fails instead of silently keeping its default:

    >>> options = {"metabs": ["pyruvate"], "metabs_freq_range": 20.0}
    >>> def_fit_params(data, options)  # doctest: +SKIP
    Traceback (most recent call last):
    ValueError: def_fit_params: unknown fit_params key(s) 'metabs_freq_range' (did you mean 'metabs_freq_ranges'?). ...
    """
    params = _def_fit_params_impl(
        data=data,
        fit_params=fit_params,
        return_params_info=return_params_info,
    )
    if params is None or data is None or not mask_nonfinite_input:
        return params
    return mask_fit_dataset_where_input_nonfinite(params, data)


def phase_fit_results(
    fit_results: xr.Dataset,
    *,
    amplitude_var: str = "fit_amps",
    reduce_dims: Sequence[str] | str | None = None,
    inplace: bool = False,
    phase_var: str = "fit_phase_correction_rad",
) -> xr.Dataset:
    """Phase complex fitted amplitudes so their real component is maximized.

    By default, each fitted complex amplitude is rotated independently, making
    ``real(fit_amps) == abs(fit_amps)`` and leaving only numerical imaginary
    residuals.  Passing ``reduce_dims`` computes one phase correction from the
    complex sum over those dimensions, which is useful when a shared phase
    should be applied per metabolite across voxels or repetitions.
    """
    if amplitude_var not in fit_results:
        raise KeyError(f"fit_results does not contain '{amplitude_var}'.")

    result = fit_results if inplace else fit_results.copy(deep=True)
    amplitudes = result[amplitude_var]

    if not np.issubdtype(amplitudes.dtype, np.complexfloating):
        raise ValueError(f"fit_results['{amplitude_var}'] must be complex-valued.")

    if isinstance(reduce_dims, str):
        reduce_dims = [reduce_dims]
    elif reduce_dims is not None:
        reduce_dims = list(reduce_dims)

    if reduce_dims:
        missing_dims = [dim for dim in reduce_dims if dim not in amplitudes.dims]
        if missing_dims:
            raise ValueError(
                f"Cannot reduce over dimensions not present in '{amplitude_var}': "
                f"{missing_dims}"
            )
        phase_source = amplitudes.sum(dim=reduce_dims, skipna=True)
    else:
        phase_source = amplitudes

    phase_correction = xr.apply_ufunc(
        lambda values: -np.angle(values),
        phase_source,
        dask="allowed",
    )
    phase_correction.name = phase_var
    phase_correction.attrs.update(
        {
            "unit": "rad",
            "description": (
                f"Phase correction applied to '{amplitude_var}' so the fitted "
                "amplitude real component is maximized."
            ),
        }
    )

    result[amplitude_var] = amplitudes * np.exp(1j * phase_correction)
    result[phase_var] = phase_correction
    result.attrs["phased_to_real"] = True
    result.attrs["phase_amplitude_var"] = amplitude_var
    result.attrs["phase_correction_var"] = phase_var
    if reduce_dims:
        result.attrs["phase_reduce_dims"] = tuple(reduce_dims)

    return result


def _dataarray_is_constant_along_dim(data: xr.DataArray, dim: str) -> bool:
    """Return True when all slices along ``dim`` carry the same values."""
    reference = data.isel({dim: 0}, drop=True)
    reference_broadcast, data_broadcast = xr.broadcast(reference, data)
    reference_values = np.asarray(reference_broadcast.values)
    data_values = np.asarray(data_broadcast.values)

    if np.issubdtype(data_values.dtype, np.number) or np.iscomplexobj(data_values):
        return bool(np.allclose(data_values, reference_values, equal_nan=True))

    return bool(np.array_equal(data_values, reference_values))


def _align_fit_params_to_input_data(
    *,
    data: xr.DataArray,
    fit_params: xr.Dataset,
) -> xr.Dataset:
    """Broadcast parameter maps to the input's non-spectral dimensions.

    Each map is ordered as the input loop dimensions followed by ``metabolite``.
    A Voigt start map ``initial_gauss_fwhm_hz`` is shared by all metabolites and
    is aligned to the loop dimensions only. Dimensions removed from the input
    may be dropped from the parameters only when their values are constant.
    Missing skip masks default to False. The returned dataset retains global
    options without modifying the caller's maps.
    """
    spectral_dim = fit_params.attrs.get("spectral_dim", "acquisition_idx")
    if spectral_dim not in data.dims:
        raise ValueError(
            f"Input data is missing the spectral dimension '{spectral_dim}'."
        )

    if "skip_fitting" not in fit_params:
        skip_template = xr.full_like(fit_params["initial_freqs_hz"], False, dtype=bool)
        fit_params = fit_params.assign(skip_fitting=skip_template)

    loop_dims = [dim for dim in data.dims if dim != spectral_dim]
    target_slice = data.isel({spectral_dim: 0}, drop=True)
    metabolite_coord = fit_params["initial_freqs_hz"].coords["metabolite"]
    target_with_metab = target_slice.expand_dims(metabolite=metabolite_coord).transpose(
        *loop_dims, "metabolite"
    )

    gauss_map_names = (
        ("initial_gauss_fwhm_hz",) if "initial_gauss_fwhm_hz" in fit_params else ()
    )
    aligned_params: dict[str, xr.DataArray] = {}
    for key in _FIT_PARAM_MAP_NAMES + gauss_map_names:
        data_array = fit_params[key]
        target = target_with_metab
        if key == "initial_gauss_fwhm_hz":
            if "metabolite" in data_array.dims:
                raise ValueError(
                    "fit_params['initial_gauss_fwhm_hz'] is one Gaussian FWHM "
                    "shared by all metabolites; remove its 'metabolite' dimension."
                )
            target = target_slice
        extra_dims = [
            dim
            for dim in data_array.dims
            if dim not in loop_dims and dim != "metabolite"
        ]
        for dim in extra_dims:
            if data_array.sizes[dim] == 1 or _dataarray_is_constant_along_dim(
                data_array, dim
            ):
                data_array = data_array.isel({dim: 0}, drop=True)
                continue

            raise ValueError(
                f"fit_params['{key}'] still varies along '{dim}', but data no "
                f"longer has that dimension. Reduce or select fit_params along "
                f"'{dim}' as well, or regenerate fit_params from the reduced input."
            )

        aligned_params[key] = data_array.broadcast_like(target).transpose(*target.dims)

    return fit_params.assign(aligned_params)


def _extract_fit_single_global_kwargs(fit_params: xr.Dataset) -> dict[str, object]:
    """Return attrs safe to pass as kwargs to ``_fit_single_spectrum_pseudo_inv``."""
    data_var_names = set(fit_params.data_vars)
    return {
        key: value
        for key, value in fit_params.attrs.items()
        if key not in data_var_names
        and key not in _FIT_SINGLE_SPECTRUM_POSITIONAL_ARG_NAMES
    }


# Start values that ``_clip_t2_coupling_start_values`` places on a band edge
# may miss it by rounding; the fitter's own band test uses the same tolerance.
_T2_BAND_START_RTOL = 1e-12


def _clip_t2_coupling_start_values(fit_params: xr.Dataset) -> xr.Dataset:
    """Return ``fit_params`` with coupled start T2* values moved inside their limits.

    Used when ``t2_coupling_clip_start`` is true. A fixed dependent starts at
    ``t2_ratio`` times its leader's start, so the leader's ``initial_t2s_s`` is
    clipped to the range in which the leader and all its fixed dependents lie
    within their ``[min_t2_s, max_t2_s]``. Afterwards each banded dependent's
    ``initial_t2s_s`` is clipped to ``[low, high]`` times the leader's start,
    intersected with its own T2* bounds. Values already inside stay unchanged,
    so the clipping is the smallest change per spectrum. Spectra without any
    admissible start keep their values and are reported by
    ``_check_t2_coupling_start_values``. ``fit_params`` is not modified.
    """
    coupling = fit_params.attrs.get("t2_coupling") or {}
    initial_t2s_s = fit_params["initial_t2s_s"].copy()
    min_t2_s, max_t2_s = fit_params["min_t2_s"], fit_params["max_t2_s"]

    def clip(values, low, high):
        admissible = low <= high
        return xr.where(admissible, values.clip(min=low, max=high), values)

    fixed_by_leader: dict[str, list[tuple[str, float]]] = {}
    for dependent, (leader, ratio) in coupling.items():
        if not isinstance(ratio, tuple):
            fixed_by_leader.setdefault(leader, []).append((dependent, ratio))
    for leader, dependents in fixed_by_leader.items():
        low = min_t2_s.sel(metabolite=leader)
        high = max_t2_s.sel(metabolite=leader)
        for dependent, ratio in dependents:
            # nextafter keeps ratio * start inside the dependent's bounds
            # although the division may round the limit outward.
            dependent_low = min_t2_s.sel(metabolite=dependent) / ratio
            dependent_low = xr.where(
                dependent_low * ratio < min_t2_s.sel(metabolite=dependent),
                np.nextafter(dependent_low, np.inf),
                dependent_low,
            )
            dependent_high = max_t2_s.sel(metabolite=dependent) / ratio
            dependent_high = xr.where(
                dependent_high * ratio > max_t2_s.sel(metabolite=dependent),
                np.nextafter(dependent_high, -np.inf),
                dependent_high,
            )
            low = np.maximum(low, dependent_low)
            high = np.minimum(high, dependent_high)
        initial_t2s_s.loc[{"metabolite": leader}] = clip(
            initial_t2s_s.sel(metabolite=leader), low, high
        )

    for dependent, (leader, ratio) in coupling.items():
        if not isinstance(ratio, tuple):
            continue
        band_low, band_high = ratio
        leader_start = initial_t2s_s.sel(metabolite=leader)
        low = np.maximum(band_low * leader_start, min_t2_s.sel(metabolite=dependent))
        high = np.minimum(band_high * leader_start, max_t2_s.sel(metabolite=dependent))
        initial_t2s_s.loc[{"metabolite": dependent}] = clip(
            initial_t2s_s.sel(metabolite=dependent), low, high
        )
    return fit_params.assign(initial_t2s_s=initial_t2s_s)


def _check_t2_coupling_start_values(fit_params: xr.Dataset) -> None:
    """Require every coupled start T2* to be admissible before fitting.

    The fitter starts each fixed dependent at ``t2_ratio`` times its leader's
    ``initial_t2s_s``. A start outside ``[min_t2_s, max_t2_s]`` of the dependent
    would leave the leader's search without an admissible candidate. A banded
    dependent starts at its own ``initial_t2s_s``, whose ratio to the leader's
    start must lie in the band. Fail before fitting and name the pair.
    """
    coupling = fit_params.attrs.get("t2_coupling") or {}
    initial_t2s_s = fit_params["initial_t2s_s"]
    for dependent, (leader, ratio) in coupling.items():
        if isinstance(ratio, tuple):
            low, high = ratio
            start_ratio = initial_t2s_s.sel(metabolite=dependent) / initial_t2s_s.sel(
                metabolite=leader
            )
            inside = (start_ratio >= low * (1 - _T2_BAND_START_RTOL)) & (
                start_ratio <= high * (1 + _T2_BAND_START_RTOL)
            )
            if not bool(inside.all()):
                raise ValueError(
                    f"t2_coupling: initial_t2s_s of {dependent!r} divided by "
                    f"initial_t2s_s of {leader!r} leaves the ratio band "
                    f"({low}, {high}) in at least one spectrum; adjust the start "
                    "T2* values or the band, or set t2_coupling_clip_start=True "
                    "to move the start values inside automatically."
                )
            continue
        start_t2_s = ratio * fit_params["initial_t2s_s"].sel(metabolite=leader)
        bounds = fit_params[["min_t2_s", "max_t2_s"]].sel(metabolite=dependent)
        inside = (start_t2_s >= bounds["min_t2_s"]) & (start_t2_s <= bounds["max_t2_s"])
        if not bool(inside.all()):
            raise ValueError(
                f"t2_coupling: {ratio} * initial_t2s_s of {leader!r} leaves "
                f"[min_t2_s, max_t2_s] of {dependent!r} in at least one spectrum; "
                "adjust the leader's start T2*, the dependent's T2* bounds or "
                "t2_ratio, or set t2_coupling_clip_start=True to clip the "
                "leader's start T2* automatically."
            )


def _check_start_separation(
    fit_params: xr.Dataset, min_peak_separation_hz: float, *, operation: str
) -> None:
    """Require the start frequencies to keep the minimum peak separation.

    The fit scores every frequency set with a pair closer than
    ``min_peak_separation_hz`` as inadmissible. A start state of that kind
    gives the first frequency step no valid comparison value, so that step
    would replace the first searched line even if it already sits on its peak.
    Compare the finite ``initial_freqs_hz`` of the lines that are not skipped,
    separately in every spectrum, and fail before any spectrum is fitted.

    Parameters
    ----------
    fit_params : xr.Dataset
        Fit parameters with ``initial_freqs_hz`` (Hz) along ``metabolite`` and
        optional voxel dimensions, and an optional boolean ``skip_fitting`` map
        on the same dimensions.
    min_peak_separation_hz : float
        Minimum allowed distance between two fitted lines in Hz. Zero disables
        the check.
    operation : str
        Name of the calling public function, used in the error message.

    Raises
    ------
    ValueError
        If two active start frequencies of one spectrum are closer than
        ``min_peak_separation_hz``. The message names the first such pair,
        the positional indices of its spectrum (usable with ``isel``) and the
        number of affected spectra. Coordinate values are not reported,
        because a coordinate may share a dimension's name while spanning
        several dimensions, as the 3-D voxel positions of CSI data do.
    """
    if min_peak_separation_hz <= 0.0:
        return
    starts = fit_params["initial_freqs_hz"].astype(float)
    if "skip_fitting" in fit_params:
        starts = starts.where(~fit_params["skip_fitting"].astype(bool))
    voxel_dims = [dim for dim in starts.dims if dim != "metabolite"]
    starts = starts.transpose(*voxel_dims, "metabolite")
    values = starts.values.reshape(-1, starts.sizes["metabolite"])
    values = np.where(np.isfinite(values), values, np.nan)
    order = np.argsort(values, axis=1)
    with np.errstate(invalid="ignore"):
        too_close = (
            np.diff(np.take_along_axis(values, order, axis=1), axis=1)
            < min_peak_separation_hz
        )
    if not too_close.any():
        return
    spectrum, pair = np.argwhere(too_close)[0]
    lower, upper = order[spectrum, pair], order[spectrum, pair + 1]
    names = [str(name) for name in starts["metabolite"].values]
    lower_hz, upper_hz = values[spectrum, lower], values[spectrum, upper]
    voxel = np.unravel_index(spectrum, [starts.sizes[dim] for dim in voxel_dims])
    location = {dim: int(index) for dim, index in zip(voxel_dims, voxel, strict=True)}
    where = (
        f"in the spectrum at positions {location} (for isel)"
        if location
        else "in the single spectrum"
    )
    affected = int(too_close.any(axis=1).sum())
    raise ValueError(
        f"{operation}: the start frequencies (initial_freqs_hz) of "
        f"{names[lower]!r} ({lower_hz:.6g} Hz) and {names[upper]!r} "
        f"({upper_hz:.6g} Hz) are {upper_hz - lower_hz:.3g} Hz apart {where}, "
        "closer than min_peak_separation_hz="
        f"{min_peak_separation_hz:.3g} Hz ({affected} of {len(values)} spectra "
        "affected). The fit rejects every frequency set with such a pair, so it "
        "cannot compare candidates with this start. Move the start frequencies "
        "apart, lower min_peak_separation_hz, or skip or remove one of the lines "
        "in these spectra."
    )


def _prepare_detection_maps(data, fit_params):
    """Align explicit per-spectrum stream IDs and supplied quadrature sigmas.

    Optional maps let independent blocks use the existing fitter backends without
    resetting random streams or serializing spectra with different noise scales.
    They do not estimate noise or change the assumed white-noise likelihood.
    """
    spectral_dim = fit_params.attrs.get("spectral_dim", "acquisition_idx")
    template = data.isel({spectral_dim: 0}, drop=True)
    defaults = {
        "detection_stream_id": xr.DataArray(
            np.arange(template.size, dtype=np.int64).reshape(template.shape),
            dims=template.dims,
            coords=template.coords,
        ),
        "detection_sigma_map": xr.DataArray(
            np.full(template.shape, fit_params.attrs["detection_sigma"], dtype=float),
            dims=template.dims,
            coords=template.coords,
        ),
    }
    maps = []
    for name, default in defaults.items():
        value = fit_params[name] if name in fit_params else default
        if set(value.dims) - set(template.dims):
            raise ValueError(f"{name} may only have non-spectral input dimensions.")
        value, _ = xr.align(value, template, join="exact")
        value = value.broadcast_like(template).transpose(*template.dims)
        values = np.asarray(value.values)
        if name == "detection_stream_id":
            if (
                values.dtype.kind not in "iu"
                or np.any(values < 0)
                or np.any(values > np.iinfo(np.int64).max)
                or np.unique(values).size != values.size
            ):
                raise ValueError(
                    "detection_stream_id requires unique nonnegative int64 IDs."
                )
            value = value.astype(np.int64)
        else:
            with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                valid = (
                    values.dtype.kind in "fiu"
                    and np.all(np.isfinite(values))
                    and np.all(values > 0)
                    and np.all(np.isfinite(values.astype(float) ** 2))
                    and np.all(values.astype(float) ** 2 > 0)
                )
            if not valid:
                raise ValueError(
                    "detection_sigma_map requires finite positive quadrature sigmas with finite positive squares."
                )
            value = value.astype(float)
        maps.append(value)
    return tuple(maps)


def _add_voigt_width_outputs(result_ds: xr.Dataset, options: dict) -> None:
    """Add Voigt linewidths and the Gaussian-width status to fit results in place.

    ``fit_fwhm_hz`` enters as the Lorentz FWHM ``1 / (pi * fit_t2s)``; it is
    kept as ``fit_lorentz_fwhm_hz`` and replaced by the Olivero-Longbothum
    total absorption FWHM. ``fit_magnitude_fwhm_hz`` is replaced by the exact
    magnitude FWHM of the Voigt line. ``fit_gauss_fwhm_status`` is, in order of precedence,
    ``not_fitted`` (``fit_status`` is not ``success``), ``fixed``
    (``gauss_search_width_hz == 0``), ``at_lower_bound`` (0 Hz),
    ``at_upper_bound`` (``gauss_max_fwhm_hz``) or ``estimated``.
    """
    lorentz = result_ds["fit_fwhm_hz"]
    lorentz.attrs.update(unit="Hz", definition="1 / (pi * fit_t2s)")
    gauss = result_ds["fit_gauss_fwhm_hz"]
    gauss.attrs.update(
        unit="Hz",
        definition="Gaussian FWHM shared by all lines of the spectrum",
    )
    result_ds["fit_lorentz_fwhm_hz"] = lorentz
    result_ds["fit_fwhm_hz"] = _voigt_fwhm_hz(lorentz, gauss)
    result_ds["fit_fwhm_hz"].attrs.update(
        unit="Hz",
        definition=(
            "absorption-spectrum Olivero-Longbothum Voigt FWHM: "
            "0.5346 * fit_lorentz_fwhm_hz + "
            "sqrt(0.2166 * fit_lorentz_fwhm_hz**2 + fit_gauss_fwhm_hz**2)"
        ),
    )
    result_ds["fit_magnitude_fwhm_hz"] = _magnitude_fwhm_hz(lorentz, gauss)
    result_ds["fit_magnitude_fwhm_hz"].attrs.update(
        unit="Hz",
        definition=(
            "magnitude-spectrum FWHM of the Voigt line from "
            "fit_lorentz_fwhm_hz and fit_gauss_fwhm_hz, solved exactly"
        ),
    )
    upper_hz = float(options["gauss_max_fwhm_hz"]) - _GAUSS_BOUND_TOLERANCE_HZ
    status = xr.where(
        gauss <= _GAUSS_BOUND_TOLERANCE_HZ,
        "at_lower_bound",
        xr.where(gauss >= upper_hz, "at_upper_bound", "estimated"),
    )
    if float(options["gauss_search_width_hz"]) == 0:
        status = xr.full_like(status, "fixed")
    result_ds["fit_gauss_fwhm_status"] = xr.where(
        result_ds["fit_status"] != "success", "not_fitted", status
    )
    result_ds["fit_gauss_fwhm_status"].attrs["description"] = (
        "not_fitted, fixed, at_lower_bound, at_upper_bound or estimated; a "
        "bound status means the Gaussian FWHM is not determined by the data"
    )


# A fitted value closer than this to a search bound sits at the bound. Both
# lie far below the frequency resolution (Hz) and the T2* search steps (s).
_FREQ_BOUND_TOLERANCE_HZ = 1e-3
_T2_BOUND_TOLERANCE_S = 1e-6
_BOUND_STATUS_DTYPE = "U14"
_BOUND_STATUS_DESCRIPTION = (
    "not_fitted, fixed, at_lower_bound, at_upper_bound or estimated; a bound "
    "status means the search window, not the data, limits the value"
)


def _bound_status_labels(
    values, lower, upper, searched, fitted, *, tolerance: float
) -> np.ndarray:
    """Return bound status labels for fitted values and their search bounds.

    All arrays broadcast to one shape. The label is, in order of precedence,
    ``not_fitted`` (``fitted`` is False), ``fixed`` (``searched`` is False),
    ``at_lower_bound`` (``value <= lower + tolerance``), ``at_upper_bound``
    (``value >= upper - tolerance``) or ``estimated``.
    """
    values = np.asarray(values, dtype=float)
    status = np.where(
        values <= np.asarray(lower, dtype=float) + tolerance,
        "at_lower_bound",
        np.where(
            values >= np.asarray(upper, dtype=float) - tolerance,
            "at_upper_bound",
            "estimated",
        ),
    )
    status = np.where(np.asarray(searched, dtype=bool), status, "fixed")
    return np.where(np.asarray(fitted, dtype=bool), status, "not_fitted").astype(
        _BOUND_STATUS_DTYPE
    )


def _t2_searched(fit_params: xr.Dataset) -> xr.DataArray:
    """Return whether each line's T2* was searched.

    A line with ``metabs_t2_ranges > 0`` is searched. A fixed-ratio
    ``t2_coupling`` dependent follows its leader, so it is searched exactly
    when its leader is; a ratio-band dependent uses its own range.
    """
    searched = (fit_params["metabs_t2_ranges"] > 0).copy()
    coupling = fit_params.attrs.get("t2_coupling") or {}
    for dependent, (leader, ratio) in coupling.items():
        if isinstance(ratio, (tuple, list)):
            continue
        searched.loc[{"metabolite": dependent}] = searched.sel(metabolite=leader).values
    return searched


def _add_bound_statuses(result_ds: xr.Dataset, fit_params: xr.Dataset) -> None:
    """Add ``fit_freq_status`` and ``fit_t2_status`` to fit results in place.

    The fitted frequency is compared with ``freq_range_lower_bound_hz`` and
    ``freq_range_upper_bound_hz``, the fitted T2* with ``min_t2_s`` and
    ``max_t2_s``, per spectrum and line in solver order (the bounds each line
    was searched in). Lines that were skipped or in spectra whose fit did not
    succeed are ``not_fitted``; lines with a zero search width are ``fixed``.
    Dask-backed results stay lazy.
    """
    fitted = (result_ds["fit_status"] == "success") & ~result_ds["fit_skipped"].astype(
        bool
    )
    lower = fit_params["freq_range_lower_bound_hz"]
    upper = fit_params["freq_range_upper_bound_hz"]
    outputs = {
        "fit_freq_status": (
            "fit_freqs",
            lower,
            upper,
            (fit_params["metabs_freq_ranges"] > 0) & (lower < upper),
            _FREQ_BOUND_TOLERANCE_HZ,
            "frequency window freq_range_lower_bound_hz to freq_range_upper_bound_hz",
        ),
        "fit_t2_status": (
            "fit_t2s",
            fit_params["min_t2_s"],
            fit_params["max_t2_s"],
            _t2_searched(fit_params),
            _T2_BOUND_TOLERANCE_S,
            "T2* bounds min_t2_s to max_t2_s",
        ),
    }
    for name, (source, low, high, searched, tolerance, bounds) in outputs.items():
        status = xr.apply_ufunc(
            _bound_status_labels,
            result_ds[source],
            low,
            high,
            searched,
            fitted,
            kwargs={"tolerance": tolerance},
            dask="parallelized",
            output_dtypes=[np.dtype(_BOUND_STATUS_DTYPE)],
        )
        result_ds[name] = status.transpose(*result_ds[source].dims)
        result_ds[name].attrs.update(
            description=_BOUND_STATUS_DESCRIPTION, bounds=bounds
        )


def _warn_about_unusable_frequency_windows(
    fit_params: xr.Dataset, *, operation: str
) -> None:
    """Warn when frequency windows cannot resolve or separate the lines.

    A searched window (``metabs_freq_ranges > 0``) narrower than the spectral
    point spacing ``bw_hz / nsamplepoints`` (``1 / (n_samples * dwell)``)
    cannot move the line by one point, so the fit ends at a window edge.
    Searched windows of two lines outside one ``t2_coupling`` group that
    overlap in a spectrum where neither line is skipped let both lines reach
    the same peak. Each check emits one ``UserWarning`` that names
    ``operation``, the lines and the limits. Nothing is changed.
    """
    attrs = fit_params.attrs
    names = [str(name) for name in fit_params.metabolite.values]
    ranges = fit_params["metabs_freq_ranges"].transpose("metabolite", ...)
    active = (ranges > 0) & ~fit_params["skip_fitting"].astype(bool).transpose(
        *ranges.dims
    )
    ranges_hz = ranges.values.reshape(len(names), -1)
    active_flags = active.values.reshape(len(names), -1)

    if attrs.get("bw_hz") is not None and attrs.get("nsamplepoints"):
        spacing_hz = float(attrs["bw_hz"]) / int(attrs["nsamplepoints"])
        narrow = [
            f"{name} ({float(np.min(ranges_hz[index][too_narrow])):.3g} Hz)"
            for index, name in enumerate(names)
            if (
                too_narrow := active_flags[index] & (ranges_hz[index] < spacing_hz)
            ).any()
        ]
        if narrow:
            warnings.warn(
                f"{operation}: the frequency window (metabs_freq_ranges) of "
                f"{', '.join(narrow)} is narrower than the spectral point "
                f"spacing {spacing_hz:.3g} Hz, so the fitted frequency is "
                "likely to end at a window edge. Use a window of at least "
                "one point spacing, or 0 to freeze the frequency.",
                UserWarning,
                stacklevel=3,
            )

    group = {name: name for name in names}
    for dependent, (leader, _) in (attrs.get("t2_coupling") or {}).items():
        group[dependent] = leader
    lower = fit_params["freq_range_lower_bound_hz"].transpose(*ranges.dims)
    upper = fit_params["freq_range_upper_bound_hz"].transpose(*ranges.dims)
    lower_hz = lower.values.reshape(len(names), -1)
    upper_hz = upper.values.reshape(len(names), -1)
    overlapping = []
    for first in range(len(names)):
        for second in range(first + 1, len(names)):
            if group[names[first]] == group[names[second]]:
                continue
            both = active_flags[first] & active_flags[second]
            overlap = both & (
                np.maximum(lower_hz[first], lower_hz[second])
                < np.minimum(upper_hz[first], upper_hz[second])
            )
            if overlap.any():
                overlapping.append(
                    f"{names[first]} [{np.min(lower_hz[first][overlap]):.4g}, "
                    f"{np.max(upper_hz[first][overlap]):.4g}] Hz and "
                    f"{names[second]} [{np.min(lower_hz[second][overlap]):.4g}, "
                    f"{np.max(upper_hz[second][overlap]):.4g}] Hz"
                )
    if overlapping:
        warnings.warn(
            f"{operation}: the frequency windows of uncoupled lines overlap: "
            f"{'; '.join(overlapping)}. Both lines can then reach the same "
            "peak. Narrow the windows (metabs_freq_ranges) or move the starts "
            "apart.",
            UserWarning,
            stacklevel=3,
        )


_FREE_PARAMETER_MAP_NAMES = (
    "metabs_freq_ranges",
    "metabs_t2_ranges",
    "min_t2_s",
    "max_t2_s",
    "freq_range_lower_bound_hz",
    "freq_range_upper_bound_hz",
)

_CARRIED_NOISE_COORDINATE = "noise_sigma_quadrature"
_CARRIED_NOISE_SOURCE = "acquisition_tail"


def _add_fit_quality(
    result_ds: xr.Dataset,
    fit_params: xr.Dataset,
    *,
    loop_dims: list[str],
    n_input_samples: int,
    carried_noise: xr.DataArray | None,
) -> xr.Dataset:
    """Add free-parameter count, residual degrees of freedom and chi2/N_eff.

    Runs in solver component order, before any relabelling; the per-spectrum
    count is a sum over lines and therefore does not depend on the order.
    ``carried_noise`` is the input's acquisition-tail noise coordinate, or None,
    in which case ``fit_reduced_chi_squared`` is NaN.
    """
    # fit_diagnostics imports the FID model through this package, so a
    # module-level import would be circular when fit_diagnostics loads first.
    from hypermri.analysis.spectroscopy.fit_diagnostics import (
        REDUCED_CHI_SQUARED_DEFINITION,
        calculate_reduced_chi_squared,
    )

    options = dict(fit_params.attrs)
    metabolites = [str(name) for name in fit_params.metabolite.values]
    active = ~(result_ds.fit_skipped | result_ds.fit_low_snr_refitted)
    maps = [fit_params[name] for name in _FREE_PARAMETER_MAP_NAMES]
    n_free = xr.apply_ufunc(
        count_free_parameters,
        active,
        *maps,
        input_core_dims=[["metabolite"]] * (1 + len(maps)),
        vectorize=True,
        dask="parallelized",
        output_dtypes=[np.int64],
        kwargs={"metabolites": metabolites, "fit_options": options},
    )
    n_free = n_free.transpose(*loop_dims).astype(np.int64)
    n_samples = fitted_sample_count(n_input_samples, options)
    result_ds["fit_n_free_parameters"] = n_free
    result_ds.fit_n_free_parameters.attrs.update(
        unit="1",
        definition=(
            "real parameters adjusted by the fit: 2 per free complex amplitude "
            "(1 per fixed-phase phase-band line plus 1 shared phase), 1 per "
            "free frequency, 1 per free T2* (one per fixed-ratio t2_coupling "
            "group), 1 for a fitted Voigt Gaussian FWHM; see "
            "count_free_parameters"
        ),
    )
    result_ds["fit_residual_degrees_of_freedom"] = 2 * n_samples - n_free
    result_ds.fit_residual_degrees_of_freedom.attrs.update(
        unit="1",
        definition="2 * fitted complex samples - fit_n_free_parameters",
        n_fitted_complex_samples=n_samples,
    )
    if carried_noise is None:
        chi2 = xr.full_like(result_ds.fit_rss, np.nan, dtype=float)
        chi2.attrs = {
            "unit": "dimensionless",
            "definition": REDUCED_CHI_SQUARED_DEFINITION,
            "noise_source": "none",
        }
    else:
        chi2 = calculate_reduced_chi_squared(result_ds, carried_noise)
    result_ds["fit_reduced_chi_squared"] = chi2
    return result_ds


def _carried_acquisition_noise(data: xr.DataArray) -> xr.DataArray | None:
    """Return the input's acquisition-tail noise coordinate, if it carries one."""
    if _CARRIED_NOISE_COORDINATE not in data.coords:
        return None
    carried = data.coords[_CARRIED_NOISE_COORDINATE]
    source = carried.attrs.get("noise_source")
    if source != _CARRIED_NOISE_SOURCE:
        raise ValueError(
            f"fit_data_pseudo_inv: input coordinate '{_CARRIED_NOISE_COORDINATE}' "
            f"has noise_source={source!r}; only '{_CARRIED_NOISE_SOURCE}' noise "
            "from propagate_acquisition_tail_noise is used automatically. Drop "
            "the coordinate and pass other noise explicitly to "
            "calculate_reduced_chi_squared or fit_spectrum(noise=...)."
        )
    noise = xr.DataArray(
        np.asarray(carried.values),
        dims=carried.dims,
        coords={dim: data.coords[dim] for dim in carried.dims if dim in data.coords},
        attrs=dict(carried.attrs),
    )
    return noise


def _build_fit_data_pseudo_inv_result_dataset(
    *,
    fit_results_tuple: FitSingleSpectrumResult,
    fit_params: xr.Dataset,
    loop_dims: list[str],
    loop_coords: dict[str, xr.DataArray],
) -> xr.Dataset:
    """Assemble the public fit-results dataset from raw fitting outputs."""
    global_params = fit_params.attrs
    schema = _fit_output_schema(global_params)
    sizes = _fit_output_sizes(global_params, fit_params.sizes["metabolite"])
    coordinates = {
        **loop_coords,
        **{name: np.arange(size) for name, size in sizes.items()},
        "metabolite": fit_params.metabolite,
    }
    if "history_field" in sizes:
        coordinates["history_field"] = list(HISTORY_FIELDS)
    result_arrays = {}
    for (name, (core_dims, _)), values in zip(
        schema.items(), fit_results_tuple, strict=True
    ):
        dims = loop_dims + core_dims
        result_arrays[name] = xr.DataArray(
            values.data if isinstance(values, xr.DataArray) else values,
            dims=dims,
            coords={key: value for key, value in coordinates.items() if key in dims},
        )

    result_ds = xr.Dataset(result_arrays)
    label_fields = [
        ("fit_status", FIT_STATUSES),
        ("fit_failure_reason", FIT_FAILURE_REASONS),
    ]
    if global_params.get("detection_target") is not None:
        label_fields.extend(
            [
                ("detection_status", DETECTION_STATUSES),
                ("detection_reason", DETECTION_REASONS),
                ("m0_fit_status", FIT_STATUSES),
                ("m0_fit_failure_reason", FIT_FAILURE_REASONS),
            ]
        )
    for name, labels in label_fields:
        result_ds[name] = xr.apply_ufunc(
            _decode_fit_labels,
            result_ds[name],
            kwargs={"labels": labels},
            dask="parallelized",
            output_dtypes=[np.dtype("U32")],
        )

    result_ds.attrs.update(global_params)
    result_ds.attrs.setdefault("lineshape", LINESHAPE_DEFAULT)
    result_ds["fit_amps"].attrs["unit"] = "a.u."
    result_ds["fit_freqs"].attrs["unit"] = "Hz"
    result_ds["fit_t2s"].attrs["unit"] = "s"
    result_ds["fit_fwhm_hz"] = 1 / (np.pi * result_ds["fit_t2s"])
    result_ds["fit_fwhm_hz"].attrs.update(
        unit="Hz", definition="absorption-spectrum FWHM 1 / (pi * fit_t2s)"
    )
    result_ds["fit_magnitude_fwhm_hz"] = (
        MAGNITUDE_LORENTZ_FWHM_FACTOR * result_ds["fit_fwhm_hz"]
    )
    result_ds["fit_magnitude_fwhm_hz"].attrs.update(
        unit="Hz", definition="magnitude-spectrum FWHM sqrt(3) / (pi * fit_t2s)"
    )
    if _uses_voigt_lineshape(global_params):
        _add_voigt_width_outputs(result_ds, global_params)
    for suffix, candidate_unit in (("freq", "Hz"), ("t2", "s")):
        legacy = f"fit_r2_{suffix}"
        canonical = f"fit_search_diagnostics_{suffix}"
        result_ds[canonical] = (
            result_ds[legacy]
            .rename({"r2_data": "search_metric"})
            .assign_coords(
                search_metric=["candidate", "temporal_rss", "spectral_magnitude_rss"]
            )
        )
        attributes = {
            "description": "Sequential candidate diagnostics; not R-squared and not a joint optimization history.",
            "candidate_unit": candidate_unit,
            "rss_unit": "a.u.^2",
            "spectral_fft_normalization": "numpy default: unnormalized forward FFT",
        }
        result_ds[canonical].attrs.update(attributes)
        result_ds[legacy].attrs.update(attributes, deprecated_alias_for=canonical)
    result_ds["fit_low_snr_refitted"].attrs["description"] = (
        "Boolean flag indicating metabolites that were excluded and refitted due to low amplitude (|amplitude| < snr_threshold)"
    )
    result_ds.attrs["model_name"] = "FID"
    result_ds.attrs["fit_objective_name"] = (
        "spectral_magnitude_rss"
        if global_params.get("use_sse_mag")
        else "temporal_complex_rss"
    )
    result_ds.attrs["numerical_contract_version"] = "3"

    result_ds.fit_status.attrs["description"] = (
        "Numerical status only, not signal detection: success, zero_signal, skipped, invalid_input, fit_failed"
    )
    result_ds.fit_failure_reason.attrs["description"] = (
        "Stable failure reason; none for a valid numerical result"
    )
    result_ds.fit_rss.attrs.update(
        unit="a.u.^2", definition="sum(abs(measured_FID - fitted_FID)**2)"
    )
    result_ds.fit_objective.attrs.update(
        unit="a.u.^2", definition=result_ds.attrs["fit_objective_name"]
    )
    result_ds["fit_skipped"] = fit_params.skip_fitting.transpose(
        *loop_dims, "metabolite"
    )
    _add_bound_statuses(result_ds, fit_params)
    if global_params.get("store_optimization_history"):
        result_ds.fit_history_info.attrs.update(
            event_codes=json.dumps(HISTORY_EVENTS, sort_keys=True),
            rejection_reason_codes=json.dumps(HISTORY_REASONS, sort_keys=True),
            description="Candidate acceptance is local to its coordinate grid; step acceptance is global. Indices are zero-based; -1 means not applicable or unused padding.",
        )
        result_ds.fit_history_count.attrs["description"] = (
            "Number of populated history rows; later rows are padding"
        )
        for name, unit in (
            ("amps", "a.u."),
            ("freqs", "Hz"),
            ("t2s", "s"),
            ("rss", "a.u.^2"),
            ("objective", "a.u.^2"),
        ):
            result_ds[f"fit_history_{name}"].attrs["unit"] = unit
        result_ds.fit_history_active.attrs["description"] = (
            "Components included in this candidate/model, in solver component "
            "order; fit_component_index maps labels to these components"
        )
        result_ds.attrs["optimization_history_version"] = "1"

    for unit in ("ppm", "Hz"):
        result_ds = assign_frequency_axis(
            result_ds, resolve_frequency_axis(fit_params, unit), unit
        )
    result_ds.attrs["signal_domain"] = fit_params.attrs.get("signal_domain", None)
    result_ds.attrs["domain"] = fit_params.attrs.get("domain", None)
    result_ds.attrs["spectral_dim"] = fit_params.attrs.get(
        "spectral_dim", "acquisition_idx"
    )
    result_ds.attrs["git_commit_hash"] = fit_params.attrs.get("git_commit_hash", None)

    if global_params.get("detection_target") is not None:
        result_ds = annotate_detection_result(result_ds, fit_params)
    return result_ds


def _frequency_position_order(fit_params: xr.Dataset) -> list[int]:
    """Return metabolite indices in the frequency position order of a fit.

    The order is ascending mean ``initial_freqs_hz`` over all spectra (NaN
    ignored), equal means ordered by metabolite name. It is the target order
    of the relabelling of uncoupled lines (``_exchangeable_metabolite_groups``)
    and, as ``_frequency_position_rank``, the order of the frequency positions
    that own the names and roles of ``t2_coupling`` lines during the search.
    """
    names = [str(name) for name in fit_params.metabolite.values]
    loop_dims = [
        dim for dim in fit_params["initial_freqs_hz"].dims if dim != "metabolite"
    ]
    mean_initial_hz = (
        fit_params["initial_freqs_hz"].mean(dim=loop_dims, skipna=True).values
        if loop_dims
        else fit_params["initial_freqs_hz"].values
    )
    return sorted(range(len(names)), key=lambda i: (mean_initial_hz[i], names[i]))


def _exchangeable_metabolite_groups(fit_params: xr.Dataset) -> list[np.ndarray]:
    """Group metabolite indices whose fitted components may exchange labels.

    Each group lists indices in the target order of
    ``_frequency_position_order``. Components exchange labels only within a
    group, which requires the same constraint role: lines in ``t2_coupling`` (leaders and dependents) keep
    their label, and lines with different ``phase_band_tolerance_deg`` entries
    (absent counts as its own entry) are never exchanged.
    """
    names = [str(name) for name in fit_params.metabolite.values]
    coupled = set()
    for dependent, (leader, _) in (fit_params.attrs.get("t2_coupling") or {}).items():
        coupled.update((str(dependent), str(leader)))
    tolerances = fit_params.attrs.get("phase_band_tolerance_deg") or {}
    groups: dict[object, list[int]] = {}
    for index in _frequency_position_order(fit_params):
        if names[index] in coupled:
            continue
        groups.setdefault(tolerances.get(names[index]), []).append(index)
    return [np.asarray(group) for group in groups.values() if len(group) > 1]


def _component_index_by_frequency(freqs_hz, skipped, *, groups):
    """Return, per spectrum, the solver component to report under each label.

    ``freqs_hz`` and ``skipped`` have the metabolite axis last. Within each
    group the active components (not skipped, finite frequency) are sorted by
    fitted frequency and assigned to the active labels in target order; a
    stable sort keeps equal frequencies in target order. Inactive components
    keep their label.
    """
    freqs_hz, skipped = np.broadcast_arrays(freqs_hz, skipped)
    component_index = np.broadcast_to(
        np.arange(freqs_hz.shape[-1]), freqs_hz.shape
    ).copy()
    flat_freqs = freqs_hz.reshape(-1, freqs_hz.shape[-1])
    flat_active = (~skipped & np.isfinite(freqs_hz)).reshape(flat_freqs.shape)
    flat_index = component_index.reshape(flat_freqs.shape)
    for group in groups:
        active = flat_active[:, group]
        complete = active.all(axis=1)
        order = np.argsort(flat_freqs[complete][:, group], axis=1, kind="stable")
        flat_index[np.ix_(complete, group)] = group[order]
        for spectrum in np.flatnonzero(~complete & (active.sum(axis=1) > 1)):
            members = group[active[spectrum]]
            order = np.argsort(flat_freqs[spectrum, members], kind="stable")
            flat_index[spectrum, members] = members[order]
    return component_index


def _take_components(values, component_index):
    """Reorder the last (metabolite) axis of ``values`` per spectrum."""
    shape = np.broadcast_shapes(values.shape, component_index.shape)
    return np.take_along_axis(
        np.broadcast_to(values, shape), np.broadcast_to(component_index, shape), axis=-1
    )


def _sort_metabolites_by_initial_frequency(
    result_ds: xr.Dataset, fit_params: xr.Dataset
) -> xr.Dataset:
    """Relabel fitted components so their frequencies follow the initial order.

    Applied by ``fit_data_pseudo_inv`` unless ``keep_fit_order`` is True. The
    exchange groups come from ``_exchangeable_metabolite_groups``. Every data
    variable with a ``metabolite`` dimension moves with its component except
    the optimization history, whose rows keep the solver order. Adds
    ``fit_component_index``. Dask-backed results stay lazy; the total fitted
    signal is invariant because only labels change.
    """
    groups = _exchangeable_metabolite_groups(fit_params)
    apply_options = {
        "input_core_dims": [["metabolite"], ["metabolite"]],
        "output_core_dims": [["metabolite"]],
        "dask": "parallelized",
    }
    component_index = xr.apply_ufunc(
        _component_index_by_frequency,
        result_ds["fit_freqs"],
        result_ds["fit_skipped"],
        kwargs={"groups": groups},
        output_dtypes=[np.int64],
        **apply_options,
    )
    sorted_ds = result_ds.copy()
    if groups:
        for name, variable in result_ds.data_vars.items():
            if "metabolite" not in variable.dims or "history_step" in variable.dims:
                continue
            sorted_ds[name] = xr.apply_ufunc(
                _take_components,
                variable,
                component_index,
                output_dtypes=[variable.dtype],
                keep_attrs=True,
                **apply_options,
            ).transpose(*variable.dims)
    sorted_ds["fit_component_index"] = component_index.transpose(
        *result_ds["fit_freqs"].dims
    )
    sorted_ds["fit_component_index"].attrs["description"] = (
        "Index, in metabolite order, of the solver component reported under "
        "this label (see keep_fit_order); history variables keep solver order"
    )
    return sorted_ds


def _decode_fit_labels(codes, *, labels):
    """Avoid NumPy vectorize's truncation of fixed-width Unicode output types."""
    # Preserve a zero-dimensional array for scalar Dask comparisons/broadcasts.
    return np.asarray(np.asarray(labels, dtype="U32")[codes])


def _fit_data_pseudo_inv_apply_ufunc(
    data: xr.DataArray,
    param_maps: tuple[xr.DataArray, ...],
    global_kwargs: dict[str, object],
) -> FitSingleSpectrumResult:
    """Fit prepared spectra with xarray using the shared numerical output schema.

    ``data`` has the sample dimension last; ``param_maps`` follow
    ``_FIT_PARAM_MAP_NAMES`` with matching loop dimensions and metabolite last.
    ``global_kwargs`` contains only options not already passed positionally.
    Dask-backed input uses xarray's parallelized execution.
    """
    spectral_dim = data.dims[-1]
    schema = _fit_output_schema(global_kwargs)
    sizes = _fit_output_sizes(global_kwargs, param_maps[0].sizes["metabolite"])
    gufunc_options = {"output_sizes": sizes}
    output_dtypes = [dtype for _, dtype in schema.values()]
    if data.chunks is not None:
        # One full FID per task; preserve the user's loop-dimension partitioning.
        data = data.chunk({spectral_dim: -1})
        loop_chunks = {dim: data.chunksizes[dim] for dim in data.dims[:-1]}
        param_maps = tuple(
            param.chunk(
                {
                    dim: (-1 if dim == "metabolite" else loop_chunks[dim])
                    for dim in param.dims
                }
            )
            for param in param_maps
        )
        # Supply empty output prototypes instead of casting the complex input
        # prototype to real/int output dtypes during Dask's graph construction.
        gufunc_options["meta"] = tuple(
            np.empty((0,) * len(dims), dtype=dtype) for dims, dtype in schema.values()
        )
        output_dtypes = None
    return xr.apply_ufunc(
        _fit_single_spectrum_outputs,
        data,
        *param_maps,
        input_core_dims=[[spectral_dim]]
        + [
            ["metabolite"] if "metabolite" in param.dims else [] for param in param_maps
        ],
        output_core_dims=[dims for dims, _ in schema.values()],
        exclude_dims={spectral_dim},
        dask="parallelized",
        vectorize=True,
        output_dtypes=output_dtypes,
        dask_gufunc_kwargs=gufunc_options,
        kwargs=global_kwargs,
    )


def _fit_data_pseudo_inv_with_tqdm(
    data: xr.DataArray,
    param_maps: tuple[xr.DataArray, ...],
    global_kwargs: dict[str, object],
) -> FitSingleSpectrumResult:
    """Fit prepared spectra serially with progress and the shared output schema.

    Inputs use the same dimension order, parameter order and filtered global
    options as ``_fit_data_pseudo_inv_apply_ufunc``. Arrays retain the input's
    loop-dimension order, including when only a single spectrum is supplied.
    """
    from tqdm.auto import tqdm

    loop_dims = data.dims[:-1]
    loop_shape = data.shape[:-1]
    schema = _fit_output_schema(global_kwargs)
    sizes = _fit_output_sizes(global_kwargs, param_maps[0].sizes["metabolite"])
    outputs = tuple(
        np.empty(loop_shape + tuple(sizes[dim] for dim in dims), dtype=dtype)
        for dims, dtype in schema.values()
    )
    index_iterator = np.ndindex(loop_shape) if loop_dims else [()]
    total_voxels = int(np.prod(loop_shape, dtype=np.int64)) if loop_dims else 1
    with tqdm(total=total_voxels, desc="Fitting spectra", leave=True) as pbar:
        for idx in index_iterator:
            selector = {dim: index for dim, index in zip(loop_dims, idx, strict=True)}
            values = _fit_single_spectrum_outputs(
                np.asarray(data.isel(selector).values),
                *(np.asarray(param.isel(selector).values) for param in param_maps),
                **global_kwargs,
            )
            for destination, value in zip(outputs, values, strict=True):
                destination[idx] = value
            pbar.update(1)
    return outputs


def _normalize_joblib_n_jobs(n_jobs: int | None) -> int:
    """Validate and normalize the public joblib worker-count option."""
    if n_jobs is None:
        return 1
    if isinstance(n_jobs, bool) or not isinstance(n_jobs, (int, np.integer)):
        raise TypeError("n_jobs must be an integer or None.")

    normalized_n_jobs = int(n_jobs)
    if normalized_n_jobs == 0:
        raise ValueError("n_jobs=0 is invalid; use 1, -1, or another non-zero integer.")
    return normalized_n_jobs


def _resolve_joblib_batch_size(
    batch_size: int | str,
    *,
    total_spectra: int,
    n_jobs: int,
) -> int:
    """Return the number of spectra handled by each submitted joblib task."""
    if batch_size == "auto":
        from joblib import effective_n_jobs

        worker_count = max(1, effective_n_jobs(n_jobs))
        target_batch_count = max(1, 4 * worker_count)
        return max(1, int(np.ceil(total_spectra / target_batch_count)))

    if isinstance(batch_size, bool) or not isinstance(batch_size, (int, np.integer)):
        raise TypeError("batch_size must be 'auto' or a positive integer.")

    normalized_batch_size = int(batch_size)
    if normalized_batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")
    return normalized_batch_size


def _fit_spectrum_batch_joblib(
    batch_start: int,
    data_batch: np.ndarray,
    param_batches: tuple[np.ndarray, ...],
    safe_kwargs: dict[str, object],
) -> tuple[int, FitSingleSpectrumResult]:
    """Fit one batch of spectra in a joblib worker."""
    batch_results = [
        _fit_single_spectrum_outputs(
            data_batch[batch_idx],
            *(param_batch[batch_idx] for param_batch in param_batches),
            **safe_kwargs,
        )
        for batch_idx in range(data_batch.shape[0])
    ]
    return batch_start, tuple(
        np.stack([np.asarray(result[index], dtype=dtype) for result in batch_results])
        for index, (_, dtype) in enumerate(_fit_output_schema(safe_kwargs).values())
    )


def _fit_data_pseudo_inv_with_joblib(
    data: xr.DataArray,
    param_maps: tuple[xr.DataArray, ...],
    global_kwargs: dict[str, object],
    *,
    n_jobs: int,
    batch_size: int | str,
    parallel_backend: str,
    show_tqdm: bool,
) -> FitSingleSpectrumResult:
    """Fit NumPy spectra in joblib batches using the shared output schema.

    Inputs follow ``_fit_data_pseudo_inv_apply_ufunc``'s preparation contract.
    Batches may finish out of order; their starting offsets restore the original
    spectrum order before outputs are reshaped to the input's loop dimensions.
    """
    from contextlib import nullcontext

    from joblib import Parallel, delayed, parallel_config
    from tqdm.auto import tqdm

    loop_shape = data.shape[:-1]
    total_spectra = int(np.prod(loop_shape, dtype=np.int64)) if loop_shape else 1
    spectral_size = data.shape[-1]
    flat_input = np.asarray(data.values).reshape(total_spectra, spectral_size)
    flat_param_maps = tuple(
        np.asarray(param_map.values).reshape(
            (total_spectra,) + param_map.shape[len(loop_shape) :]
        )
        for param_map in param_maps
    )
    resolved_batch_size = _resolve_joblib_batch_size(
        batch_size,
        total_spectra=total_spectra,
        n_jobs=n_jobs,
    )
    batch_bounds = [
        (start, min(start + resolved_batch_size, total_spectra))
        for start in range(0, total_spectra, resolved_batch_size)
    ]

    def make_tasks():
        """Yield fresh batch tasks, including for older joblib's non-streaming path."""
        return (
            delayed(_fit_spectrum_batch_joblib)(
                start,
                flat_input[start:stop],
                tuple(param_map[start:stop] for param_map in flat_param_maps),
                global_kwargs,
            )
            for start, stop in batch_bounds
        )

    parallel_context = (
        parallel_config(backend="loky", inner_max_num_threads=1)
        if parallel_backend == "loky"
        else nullcontext()
    )
    parallel_kwargs = {
        "n_jobs": n_jobs,
        "backend": parallel_backend,
        "batch_size": 1,
    }

    flat_results: list[np.ndarray] | None = None
    progress_context = (
        tqdm(total=total_spectra, desc="Fitting spectra (parallel)", leave=True)
        if show_tqdm
        else nullcontext()
    )
    with parallel_context, progress_context as pbar:
        try:
            indexed_results = Parallel(
                **parallel_kwargs,
                return_as="generator_unordered",
            )(make_tasks())
        except TypeError:
            # joblib < 1.3 has no streaming ``return_as`` option.
            indexed_results = Parallel(**parallel_kwargs)(make_tasks())

        for batch_start, batch_results in indexed_results:
            batch_length = batch_results[0].shape[0]
            if flat_results is None:
                flat_results = [
                    np.empty((total_spectra, *result.shape[1:]), dtype=result.dtype)
                    for result in batch_results
                ]
            for destination, batch_result in zip(
                flat_results, batch_results, strict=True
            ):
                destination[batch_start : batch_start + batch_length] = batch_result
            if show_tqdm:
                pbar.update(batch_length)

    if flat_results is None:  # pragma: no cover - a DataArray always has >= 1 spectrum
        raise RuntimeError("No spectra were submitted for fitting.")

    return tuple(
        result.reshape(loop_shape + result.shape[1:]) for result in flat_results
    )


def _trim_input_to_sampling_axis(
    *,
    data: xr.DataArray,
    fit_params: xr.Dataset,
) -> xr.DataArray:
    """Trim temporal input when fit_params use fewer samples than data."""
    spectral_dim = fit_params.attrs.get("spectral_dim", "acquisition_idx")
    if spectral_dim not in data.dims:
        return data

    signal_domain = str(
        fit_params.attrs.get(
            "signal_domain",
            data.attrs.get("signal_domain", "temporal"),
        )
    ).lower()
    if signal_domain == "spectral":
        return data

    sampling_axis = fit_params.attrs.get("sampling_time_axis_s")
    if sampling_axis is not None:
        target_nsamplepoints = int(np.asarray(sampling_axis).size)
    else:
        target_nsamplepoints = int(
            fit_params.attrs.get("nsamplepoints", data.sizes[spectral_dim])
        )

    if target_nsamplepoints <= 0:
        return data

    spectral_size = int(data.sizes[spectral_dim])
    target_nsamplepoints = min(target_nsamplepoints, spectral_size)
    if target_nsamplepoints >= spectral_size:
        return data

    return data.isel({spectral_dim: slice(0, target_nsamplepoints)})


def fit_data_pseudo_inv(
    data: xr.DataArray,
    fit_params: xr.Dataset,
    show_tqdm: bool | None = None,
    n_jobs: int | None = 1,
    batch_size: int | str = "auto",
    parallel_backend: str = "loky",
) -> xr.Dataset:
    """Fit spectral data across all voxels with an iterative pseudo-inverse.

    NumPy-backed input is fitted in parallel with joblib when ``n_jobs`` is not
    1. ``n_jobs=-1`` uses all available CPU cores. ``batch_size`` controls how
    many spectra each joblib task fits; ``"auto"`` creates approximately four
    batches per worker. The default ``parallel_backend="loky"`` uses processes
    and limits numerical-library threads inside each worker to avoid CPU
    oversubscription. Dask-backed input continues to use xarray's lazy
    ``apply_ufunc`` path, for which the joblib arguments are ignored.

    All execution paths share input trimming, parameter-map alignment and
    result assembly. Parameters are broadcast over the non-spectral input
    dimensions and passed to the solver in a fixed order. Results preserve
    that dimension order and inherit compatible coordinates and input attrs.

    Results include per-spectrum ``fit_status`` and ``fit_failure_reason``.
    Numerical failures yield NaN active amplitudes and errors while other
    spectra continue; invalid user configuration still raises an exception.
    A spectrum with any NaN or infinite sample, in the real or imaginary
    part, is not fitted: its active amplitudes, frequencies, T2* values,
    linewidths and (Voigt) Gaussian FWHM are NaN and ``fit_status`` is
    ``invalid_input``.
    ``success`` means a verified finite numerical state, not detection or
    proof of convergence to a global minimum. Set
    ``fit_params.attrs['store_optimization_history'] = True`` to record all
    candidates and joint-state decisions. This is off by default.

    Set ``lineshape='voigt'`` in ``def_fit_params`` to multiply every line by
    one Gaussian decay whose FWHM is shared within each spectrum and fitted
    after the Lorentz search (see ``def_fit_params(return_params_info=True)``).
    Voigt results add ``fit_gauss_fwhm_hz``, ``fit_gauss_fwhm_status`` and
    ``fit_lorentz_fwhm_hz``; ``fit_t2s`` is then the Lorentz part only and
    ``fit_fwhm_hz`` the total Voigt FWHM. The default ``'lorentz'`` is unchanged.

    ``fit_fwhm_hz`` is the absorption-spectrum FWHM, ``1 / (pi * fit_t2s)``
    for Lorentz lines. ``fit_magnitude_fwhm_hz`` is the wider FWHM of the
    magnitude spectrum: ``sqrt(3) / (pi * fit_t2s)`` for Lorentz lines and the
    exactly solved magnitude width for Voigt lines. Both describe infinitely
    long lines: line broadening applied before the fit is contained in
    ``fit_t2s``, and broadening by FID truncation is not included.

    By default (``keep_fit_order=False``) fitted components are relabelled per
    spectrum so that fitted frequencies follow the order of the mean initial
    frequencies (equal means ordered by metabolite name). This prevents label
    swaps between lines with overlapping search ranges. Only lines with the
    same constraint role are exchanged (not skipped, not in ``t2_coupling``,
    same ``phase_band_tolerance_deg`` entry); statistical mode does not reorder.
    All per-metabolite outputs except the optimization history move together
    and ``fit_component_index`` records the solver component behind each label.
    ``keep_fit_order=True`` keeps the solver order.

    Every result carries the fit quality per spectrum: int64
    ``fit_n_free_parameters`` (see
    :func:`~hypermri.analysis.fitting.models.spectroscopy.fid_model.count_free_parameters`),
    ``fit_residual_degrees_of_freedom`` ``N_eff = 2 * n - k`` for ``n``
    fitted complex samples and ``k`` free real parameters, and
    ``fit_reduced_chi_squared`` ``fit_rss / sigma**2 / N_eff``. Sigma is the
    acquisition-tail noise when ``data`` carries it as the coordinate
    ``noise_sigma_quadrature`` over its non-spectral dimensions (from
    :func:`~hypermri.analysis.spectroscopy.noise_estimation.propagate_acquisition_tail_noise`,
    with ``noise_source='acquisition_tail'``); otherwise the reduced
    chi-squared is NaN. Other noise is applied with
    :func:`~hypermri.analysis.spectroscopy.fit_diagnostics.calculate_reduced_chi_squared`
    or ``fit_spectrum(noise=...)``. The effective fit options (line shape,
    phase-band ``tau`` and tolerances, couplings, search settings) are stored
    in the result attrs.

    Set ``detection_target`` to the bicarbonate label, and supply explicit
    ``detection_sigma`` (SD per temporal real/imaginary quadrature),
    ``detection_null_samples`` and ``detection_seed`` to enable single-target
    M0/M1 comparison. Both models are fully refitted for the observation and
    every null replicate. The fixed-budget plug-in calibration is approximate;
    independent false-positive/power validation remains necessary. Statistical
    mode requires no legacy amplitude filtering and an unregularized complex
    RSS objective. Raw M1 parameters remain in ``fit_amps/freqs/t2s``; target
    acceptance and NaN-masked amplitudes are separate outputs. Weak-component
    diagnostics never restrict parameters or select which spectra are tested.

    Parameters
    ----------
    data : xarray.DataArray
        Complex FIDs or centered spectra with the prepared spectral dimension;
        other labelled dimensions identify spectra. Signal units are retained.
    fit_params : xarray.Dataset
        Metabolite maps and validated scientific options from def_fit_params:
        sampling seconds, frequencies Hz and decay seconds. Prepared maps may
        broadcast across input spectra; both inputs are preserved.
    show_tqdm : bool or None, optional
        Display execution progress. None uses the prepared show_tqdm setting.
    n_jobs : int or None, default 1
        Joblib workers for NumPy input; -1 uses available cores and None means 1.
        Dask input retains its scheduler and ignores joblib settings.
    batch_size : int or str, default "auto"
        Positive spectra per joblib task, or automatic batching per worker.
    parallel_backend : str, default "loky"
        Joblib backend; loky uses processes and limits nested numerical threads.

    Returns
    -------
    xarray.Dataset
        Complex128 fitted line areas in signal units, float64 frequencies (Hz),
        decay times (s), widths (Hz), squared-signal RSS and status/search arrays.
        Non-spectral input dimension order and compatible labels/attrs are
        retained; metabolite and search/history axes follow the existing schema.
        With ``keep_fit_order=False`` the result adds int64
        ``fit_component_index`` (loop dimensions, ``metabolite``).
        Numerical failure is local to a spectrum; inactive metabolites retain
        their specified exclusion behavior. NumPy execution is eager and Dask
        retains laziness. No input mutation or persistent result cache occurs.
        Fixed options/seeds and deterministic backends preserve repeatability.

    Raises
    ------
    ValueError
        If sampling, alignment, scientific options, lineshape or execution settings
        are inconsistent, if ``data`` carries a ``noise_sigma_quadrature``
        coordinate whose ``noise_source`` is not ``'acquisition_tail'``, or if
        two start frequencies of lines that are not
        skipped lie closer than ``min_peak_separation_hz`` in any spectrum. All
        of these are checked before the first spectrum is fitted.
    TypeError
        If an execution setting has an invalid type.

    Warns
    -----
    UserWarning
        If stored ``freq_range_lower_bound_hz`` or ``freq_range_upper_bound_hz``
        differ from ``initial_freqs_hz`` -/+ ``metabs_freq_ranges`` / 2. The
        bounds always follow start and full search width, so the derived
        bounds are used; ``fit_params`` itself is not modified.

    Examples
    --------
    >>> fitted = fit_data_pseudo_inv(signals, prepared, n_jobs=2)
    ... # doctest: +SKIP"""
    normalized_n_jobs = _normalize_joblib_n_jobs(n_jobs)
    if not isinstance(parallel_backend, str) or not parallel_backend:
        raise TypeError("parallel_backend must be a non-empty string.")
    carried_noise = _carried_acquisition_noise(data)

    fit_params = _derive_frequency_bounds(fit_params, operation="fit_data_pseudo_inv")
    data = _trim_input_to_sampling_axis(
        data=data,
        fit_params=fit_params,
    )
    fit_params = _align_fit_params_to_input_data(
        data=data,
        fit_params=fit_params,
    )
    # Revalidate edited attrs, and attach fresh source provenance only when the
    # caller explicitly opts into supplied-noise statistical comparison.
    from .models.spectroscopy.fid_model import _prepare_global_fit_options

    global_params = _prepare_global_fit_options(fit_params.attrs)
    if global_params.get("detection_target") is not None:
        global_params.update(source_provenance())
    fit_params = fit_params.assign_attrs(global_params)
    if global_params["t2_coupling_clip_start"]:
        fit_params = _clip_t2_coupling_start_values(fit_params)
    _check_t2_coupling_start_values(fit_params)
    _check_start_separation(
        fit_params,
        global_params["min_peak_separation_hz"],
        operation="fit_data_pseudo_inv",
    )
    _warn_about_unusable_frequency_windows(fit_params, operation="fit_data_pseudo_inv")
    if not isinstance(
        global_params.get("store_optimization_history", False), (bool, np.bool_)
    ):
        raise ValueError("store_optimization_history must be boolean.")
    spectral_dim = global_params.get("spectral_dim", "acquisition_idx")
    loop_dims = [dim for dim in data.dims if dim != spectral_dim]
    aligned_input = data.transpose(*loop_dims, spectral_dim)
    param_maps = tuple(fit_params[name] for name in _FIT_PARAM_MAP_NAMES)
    voigt = _uses_voigt_lineshape(global_params)
    if voigt != ("initial_gauss_fwhm_hz" in fit_params):
        raise ValueError(
            "lineshape='voigt' requires the start map "
            "fit_params['initial_gauss_fwhm_hz'] and Lorentz fits must not "
            "contain it; create the parameters with def_fit_params using the "
            "intended lineshape."
        )
    if voigt:
        param_maps += (fit_params["initial_gauss_fwhm_hz"],)
    if global_params.get("detection_target") is not None:
        param_maps += _prepare_detection_maps(aligned_input, fit_params)
    global_kwargs = _extract_fit_single_global_kwargs(fit_params)
    if global_params.get("t2_coupling"):
        # One position order for all spectra, as in the relabelling below.
        global_kwargs["_frequency_position_rank"] = np.argsort(
            _frequency_position_order(fit_params)
        )

    if show_tqdm is None:
        show_tqdm = bool(global_params.get("show_tqdm", False))

    if data.chunks is not None:
        if normalized_n_jobs != 1:
            warnings.warn(
                "Joblib options are ignored for Dask-backed input. Configure the "
                "Dask scheduler to control parallel execution instead.",
                stacklevel=2,
            )
        if show_tqdm:
            warnings.warn(
                "fit_data_pseudo_inv(show_tqdm=True) currently uses eager "
                "voxel-wise iteration. Dask-backed input detected, so falling "
                "back to the vectorized apply_ufunc path without a tqdm progress "
                "bar.",
                stacklevel=2,
            )
        fit_results_tuple = _fit_data_pseudo_inv_apply_ufunc(
            data=aligned_input,
            param_maps=param_maps,
            global_kwargs=global_kwargs,
        )
    elif normalized_n_jobs != 1:
        fit_results_tuple = _fit_data_pseudo_inv_with_joblib(
            data=aligned_input,
            param_maps=param_maps,
            global_kwargs=global_kwargs,
            n_jobs=normalized_n_jobs,
            batch_size=batch_size,
            parallel_backend=parallel_backend,
            show_tqdm=show_tqdm,
        )
    elif show_tqdm:
        fit_results_tuple = _fit_data_pseudo_inv_with_tqdm(
            data=aligned_input,
            param_maps=param_maps,
            global_kwargs=global_kwargs,
        )
    else:
        fit_results_tuple = _fit_data_pseudo_inv_apply_ufunc(
            data=aligned_input,
            param_maps=param_maps,
            global_kwargs=global_kwargs,
        )

    loop_coords = {
        name: coord
        for name, coord in aligned_input.coords.items()
        if spectral_dim not in coord.dims
    }
    result = _build_fit_data_pseudo_inv_result_dataset(
        fit_results_tuple=fit_results_tuple,
        fit_params=fit_params,
        loop_dims=loop_dims,
        loop_coords=loop_coords,
    )
    result = _add_fit_quality(
        result,
        fit_params,
        loop_dims=loop_dims,
        n_input_samples=int(aligned_input.sizes[spectral_dim]),
        carried_noise=carried_noise,
    )
    if not (
        global_params["keep_fit_order"]
        or global_params.get("detection_target") is not None
    ):
        result = _sort_metabolites_by_initial_frequency(result, fit_params)
    return inherit_input_xarray_metadata(result, data)


_START_CANDIDATE_DIM = "start_candidate"
# Start maps of every candidate recorded by fit_best_of_starts: output name ->
# (start-map variable, unit).
_START_VALUE_OUTPUTS = {
    "fit_start_initial_freqs_hz": ("initial_freqs_hz", "Hz"),
    "fit_start_initial_t2s_s": ("initial_t2s_s", "s"),
    "fit_start_initial_gauss_fwhm_hz": ("initial_gauss_fwhm_hz", "Hz"),
}
# Option attrs from which def_fit_params builds the per-metabolite maps. The
# maps are what the fit reads, so start candidates may differ in these attrs.
_START_MAP_ATTRS = frozenset(
    {
        "gauss_initial_fwhm_hz",
        "initial_t2s_s",
        "max_t2_s",
        "metabs_freq_ranges",
        "metabs_freqs_hz",
        "metabs_freqs_ppm",
        "metabs_t2_ranges",
        "min_t2_s",
    }
)


def _attr_values_equal(first: object, second: object) -> bool:
    """Return whether two fit option values are equal, NaN equal to NaN."""
    if isinstance(first, dict) or isinstance(second, dict):
        return (
            isinstance(first, dict)
            and isinstance(second, dict)
            and first.keys() == second.keys()
            and all(_attr_values_equal(first[key], second[key]) for key in first)
        )
    if isinstance(first, (list, tuple)) or isinstance(second, (list, tuple)):
        if isinstance(first, np.ndarray) or isinstance(second, np.ndarray):
            return np.array_equal(np.asarray(first), np.asarray(second))
        return (
            isinstance(first, (list, tuple))
            and isinstance(second, (list, tuple))
            and len(first) == len(second)
            and all(
                _attr_values_equal(a, b) for a, b in zip(first, second, strict=True)
            )
        )
    first_array, second_array = np.asarray(first), np.asarray(second)
    if first_array.shape != second_array.shape:
        return False
    if first_array.dtype.kind in "fc" and second_array.dtype.kind in "fc":
        return bool(np.array_equal(first_array, second_array, equal_nan=True))
    return bool(np.array_equal(first_array, second_array))


def _check_start_candidates(start_params: Sequence[xr.Dataset]) -> None:
    """Reject start candidates that do not describe the same fit model.

    Candidates may differ in their per-metabolite maps (start values, search
    widths and bounds, T2* limits) and in the option attrs that only build
    these maps, but must share the metabolites, all other option attrs, the
    skipped lines and the frequency order of the lines,
    because the chosen result keeps the labels of every candidate.
    """
    reference = start_params[0]
    reference_order = _frequency_position_order(reference)
    for index, params in enumerate(start_params):
        if not isinstance(params, xr.Dataset):
            raise TypeError(
                "fit_best_of_starts: start_params must contain xr.Dataset "
                f"parameters from def_fit_params; entry {index} is "
                f"{type(params).__name__}."
            )
        if index == 0:
            continue
        reference_names = [str(name) for name in reference.metabolite.values]
        names = [str(name) for name in params.metabolite.values]
        if names != reference_names:
            raise ValueError(
                f"fit_best_of_starts: start candidate {index} fits the lines "
                f"{names}, candidate 0 fits {reference_names}. Every candidate "
                "must fit the same lines in the same order."
            )
        for key in sorted(
            (set(reference.attrs) | set(params.attrs)) - _START_MAP_ATTRS
        ):
            if not _attr_values_equal(reference.attrs.get(key), params.attrs.get(key)):
                raise ValueError(
                    f"fit_best_of_starts: start candidate {index} differs from "
                    f"candidate 0 in the fit option attrs['{key}']. Candidates "
                    "may differ only in their start and search-range maps; "
                    "prepare them with the same options."
                )
        if "skip_fitting" in reference and not reference["skip_fitting"].equals(
            params.get("skip_fitting")
        ):
            raise ValueError(
                f"fit_best_of_starts: start candidate {index} skips other lines "
                "than candidate 0 (skip_fitting differs). Every candidate must "
                "fit the same lines in every spectrum."
            )
        if _frequency_position_order(params) != reference_order:
            raise ValueError(
                f"fit_best_of_starts: start candidate {index} orders the lines "
                "differently by mean start frequency than candidate 0. Results "
                "are labelled by this order, so the same label would name "
                "different lines; give every candidate the same line order."
            )
    if reference.attrs.get("detection_target") is not None:
        raise ValueError(
            "fit_best_of_starts does not support the statistical detection "
            "mode (detection_target): its M0/M1 comparison is defined for one "
            "start. Fit the candidates with fit_data_pseudo_inv instead."
        )


def _lowest_objective_candidate(objectives: xr.DataArray) -> xr.DataArray:
    """Return the per-spectrum index of the lowest finite objective.

    Ties go to the lowest index; a spectrum without any finite objective
    gets candidate 0.
    """
    finite = objectives.where(np.isfinite(objectives), np.inf)
    return finite.argmin(dim=_START_CANDIDATE_DIM).astype(np.int64)


def fit_best_of_starts(
    data: xr.DataArray,
    *,
    start_params: Sequence[xr.Dataset],
    show_tqdm: bool | None = None,
    n_jobs: int | None = 1,
    batch_size: int | str = "auto",
    parallel_backend: str = "loky",
) -> xr.Dataset:
    """Fit several start candidates and keep the best fit of each spectrum.

    The FID fit searches frequencies, T2* values and (Voigt) the Gaussian
    width locally from its start values, so different starts can end in
    different local minima. This function runs :func:`fit_data_pseudo_inv`
    once per start candidate on the same data and keeps, for every spectrum,
    the result with the lowest ``fit_objective``, the quantity the fit
    minimises. For the default objective (temporal complex RSS) this equals
    ``fit_rss``, so the choice is also the lowest chi-squared for any fixed
    noise level. Ties go to the candidate listed first. A spectrum whose
    candidates all failed keeps candidate 0's result and status.

    Start candidates are made with the existing parameter functions:

    * global starts: :func:`def_fit_params` (the start that
      :func:`fit_data_pseudo_inv` uses on its own);
    * from previous results: :func:`initialise_fit_params_from_results`;
    * nested: :func:`add_metab_to_fit_params` with ``fit_results`` of the
      model without the new line. It starts the existing lines at their
      fitted values. Amplitudes are solved linearly at every step and have no
      start value.

    Every candidate is a complete fit, so the run time is the sum of the
    candidates' fit times. Fitting a single candidate is not different from
    :func:`fit_data_pseudo_inv`, which remains the default fit.

    Parameters
    ----------
    data : xarray.DataArray
        Complex FIDs or centred spectra, as for :func:`fit_data_pseudo_inv`.
        Not modified.
    start_params : sequence of xarray.Dataset
        At least two start candidates from :func:`def_fit_params` or the
        functions above. They must fit the same metabolites in the same
        order, carry identical option attrs (line shape, objective, sampling,
        search settings, couplings and phase band), skip the same lines and
        order the lines identically by mean start frequency. They may differ
        in every per-metabolite map: start frequencies (Hz), T2* starts and
        limits (s), search widths and the Voigt Gaussian FWHM start (Hz),
        and in the option attrs from which :func:`def_fit_params` builds
        these maps. Not modified.
    show_tqdm : bool or None, optional
        Progress display of each candidate fit; None uses the prepared
        ``show_tqdm`` setting.
    n_jobs : int or None, default 1
        Joblib workers for each candidate fit, as in
        :func:`fit_data_pseudo_inv`. The choice does not depend on it.
    batch_size : int or str, default "auto"
        Spectra per joblib task, as in :func:`fit_data_pseudo_inv`.
    parallel_backend : str, default "loky"
        Joblib backend, as in :func:`fit_data_pseudo_inv`.

    Returns
    -------
    xarray.Dataset
        The :func:`fit_data_pseudo_inv` result of the chosen candidate per
        spectrum, with its schema, dimensions, coordinates, units and attrs,
        plus:

        * ``fit_start_candidate`` (int64, input loop dimensions): zero-based
          index into ``start_params`` of the kept candidate;
        * ``fit_start_objective`` (float64, ``start_candidate`` followed by
          the loop dimensions): every candidate's ``fit_objective``, NaN where
          that candidate failed;
        * ``fit_start_initial_freqs_hz`` (Hz) and ``fit_start_initial_t2s_s``
          (s), float64, ``start_candidate`` followed by the start-map
          dimensions: every candidate's start frequency and T2* map, so the
          starts can be repeated; ``fit_start_initial_gauss_fwhm_hz`` (Hz)
          for Voigt fits;
        * ``attrs['start_selection']``: the selection rule.

    Raises
    ------
    TypeError
        If ``start_params`` is a single Dataset or contains another type, or an
        execution setting has an invalid type.
    ValueError
        If fewer than two candidates are given, the candidates disagree in
        metabolites, option attrs, skipped lines or line order, the options
        request the statistical detection mode, or a candidate fit rejects its
        parameters (see :func:`fit_data_pseudo_inv`). All candidates are
        checked before the first fit.

    Examples
    --------
    Start a seven-line fit globally and from the six-line result, and keep
    the better fit per voxel:

    >>> six_line = fit_data_pseudo_inv(signals, six_line_params)
    ... # doctest: +SKIP
    >>> nested = add_metab_to_fit_params(
    ...     -35.0,
    ...     new_metab_name="lac_2",
    ...     fit_params=six_line_params,
    ...     fit_results=six_line,
    ...     freeze_other_freqs=False,
    ...     freeze_other_t2s=False,
    ... )  # doctest: +SKIP
    >>> best = fit_best_of_starts(
    ...     signals, start_params=[seven_line_params, nested]
    ... )  # doctest: +SKIP
    >>> best.fit_start_candidate  # 0: global start, 1: nested start
    ... # doctest: +SKIP
    """
    if isinstance(start_params, (xr.Dataset, str, bytes)):
        raise TypeError(
            "fit_best_of_starts: start_params must be a sequence of at least "
            "two parameter Datasets, not a single "
            f"{type(start_params).__name__}."
        )
    candidates = list(start_params)
    if len(candidates) < 2:
        raise ValueError(
            "fit_best_of_starts needs at least two start candidates; with one "
            "candidate call fit_data_pseudo_inv directly."
        )
    _check_start_candidates(candidates)
    results = [
        fit_data_pseudo_inv(
            data,
            params,
            show_tqdm=show_tqdm,
            n_jobs=n_jobs,
            batch_size=batch_size,
            parallel_backend=parallel_backend,
        )
        for params in candidates
    ]

    objectives = xr.concat(
        [result["fit_objective"].astype(np.float64) for result in results],
        dim=_START_CANDIDATE_DIM,
        coords="minimal",
        compat="override",
    ).assign_coords({_START_CANDIDATE_DIM: np.arange(len(results))})
    chosen = _lowest_objective_candidate(objectives)
    loop_dims = set(chosen.dims)

    best = results[0].copy()
    for name, variable in results[0].data_vars.items():
        if not loop_dims.issubset(variable.dims):
            continue
        selected = variable
        for index in range(1, len(results)):
            selected = selected.where(chosen != index, results[index][name])
        best[name] = selected.transpose(*variable.dims).astype(variable.dtype)
        best[name].attrs = dict(variable.attrs)

    best["fit_start_candidate"] = chosen.drop_vars(
        _START_CANDIDATE_DIM, errors="ignore"
    )
    best["fit_start_candidate"].attrs.update(
        unit="1",
        description="Zero-based index into start_params of the kept candidate",
    )
    best["fit_start_objective"] = objectives
    best["fit_start_objective"].attrs.update(
        unit=results[0]["fit_objective"].attrs.get("unit", "a.u.^2"),
        description=("fit_objective of every start candidate; NaN where it failed"),
    )
    best.attrs["start_selection"] = (
        "lowest finite fit_objective per spectrum over "
        f"{len(results)} start candidates; ties to the lowest index"
    )
    for name, (source, unit) in _START_VALUE_OUTPUTS.items():
        if source not in candidates[0]:
            continue
        best[name] = xr.concat(
            [params[source].astype(np.float64) for params in candidates],
            dim=_START_CANDIDATE_DIM,
            coords="minimal",
            compat="override",
        ).assign_coords({_START_CANDIDATE_DIM: np.arange(len(candidates))})
        best[name].attrs.update(
            unit=unit, description=f"{source} of every start candidate"
        )
    return best


def reconstruct_fitted_signal(
    fit_results: xr.Dataset,
    data: xr.DataArray,
) -> xr.DataArray:
    """Reconstruct the total spectroscopy fit on the input data's coordinates.

    Parameters
    ----------
    fit_results : xr.Dataset
        FID fit results containing ``fit_amps``, ``fit_freqs`` and ``fit_t2s``,
        plus ``fit_gauss_fwhm_hz`` for ``lineshape='voigt'`` results.
        Evaluation uses the fitted model's ``sampling_time_axis_s`` attribute.
    data : xr.DataArray
        The corresponding fitted samples. Its values are not used; dimensions,
        coordinates and metadata define the returned array. The sample count
        and non-spectral coordinates must match the fit. An explicit sampling
        time coordinate or attribute is checked up to a constant time origin.
        Select the fitted samples first when only part of an FID was fitted.

    Returns
    -------
    xr.DataArray
        Complex total signal with the same dimensions, order and coordinates
        as ``data``. ``signal_domain='temporal'`` returns the summed FID;
        ``'spectral'`` returns its centered FFT. The domain comes from input
        metadata, falling back to the fit metadata and then ``'temporal'``.
        Invalid component values propagate to the total instead of being skipped.

    Raises
    ------
    ValueError
        If model times, parameter maps or input coordinates do not match, or
        if the signal domain is unsupported.

    Examples
    --------
    >>> fitted = reconstruct_fitted_signal(fit_results, data)
    >>> residual = data - fitted
    """
    spectral_dim = fit_results.attrs.get("spectral_dim", "acquisition_idx")
    if spectral_dim not in data.dims:
        raise ValueError(f"data must contain the fitted dimension '{spectral_dim}'.")

    times_s = fit_results.attrs.get("sampling_time_axis_s")
    if times_s is None:
        raise ValueError("fit_results is missing 'sampling_time_axis_s'.")
    times_s = np.asarray(times_s, dtype=float)
    if (
        times_s.ndim != 1
        or times_s.size != data.sizes[spectral_dim]
        or times_s.size == 0
        or not np.isfinite(times_s).all()
    ):
        raise ValueError(
            "data must contain exactly the samples described by the fit's "
            "one-dimensional 'sampling_time_axis_s'. Select the fitted samples first."
        )

    reference_times = data.coords.get("sampling_time_axis_s")
    if reference_times is None:
        reference_times = data.attrs.get("sampling_time_axis_s")
    if reference_times is None:
        reference_times = data.coords.get("acquisition_time_s")
    if reference_times is not None:
        reference_times = np.asarray(reference_times, dtype=float)
        if reference_times.shape != times_s.shape or not np.allclose(
            reference_times - reference_times[0],
            times_s - times_s[0],
            rtol=1e-9,
            atol=1e-12,
        ):
            raise ValueError("data sampling times do not match the fitted time grid.")

    expected_dims = (set(data.dims) - {spectral_dim}) | {"metabolite"}
    for name in ("fit_amps", "fit_freqs", "fit_t2s"):
        if name not in fit_results or set(fit_results[name].dims) != expected_dims:
            raise ValueError(
                f"fit_results['{name}'] must match data's non-spectral "
                "dimensions and contain a 'metabolite' dimension."
            )
    gauss_fwhm_hz = None
    if _uses_voigt_lineshape(fit_results.attrs):
        gauss_fwhm_hz = fit_results.get("fit_gauss_fwhm_hz")
        if gauss_fwhm_hz is None or set(gauss_fwhm_hz.dims) != expected_dims - {
            "metabolite"
        }:
            raise ValueError(
                "Voigt fit_results['fit_gauss_fwhm_hz'] must exist and match "
                "data's non-spectral dimensions."
            )
    try:
        xr.align(fit_results["fit_amps"], data, join="exact", copy=False)
    except ValueError as exc:
        raise ValueError(
            "Fit maps and data must have matching non-spectral coordinates."
        ) from exc

    time_axis = xr.DataArray(
        times_s,
        dims=[spectral_dim],
        coords={spectral_dim: data.coords[spectral_dim]},
    )
    components = reconstruct_fid_signal(
        amplitude=fit_results["fit_amps"],
        freq_hz=fit_results["fit_freqs"],
        t2_star_s=fit_results["fit_t2s"],
        time_axis_s=time_axis,
        gauss_fwhm_hz=gauss_fwhm_hz,
    )
    signal = components.sum("metabolite", skipna=False)
    domain = str(
        data.attrs.get(
            "signal_domain", fit_results.attrs.get("signal_domain", "temporal")
        )
    ).lower()
    if domain == "spectral":
        signal = xr.apply_ufunc(
            np.fft.fft,
            signal,
            input_core_dims=[[spectral_dim]],
            output_core_dims=[[spectral_dim]],
            kwargs={"axis": -1},
            dask="parallelized",
        )
        signal = signal.roll({spectral_dim: times_s.size // 2}, roll_coords=False)
    elif domain != "temporal":
        raise ValueError("data signal_domain must be 'temporal' or 'spectral'.")

    reconstructed = xr.DataArray(
        signal.transpose(*data.dims).data,
        dims=data.dims,
        coords=data.coords,
        attrs=dict(data.attrs),
        name="fitted_signal",
    )
    reconstructed.attrs.update(
        signal_domain=domain,
        model_name=fit_results.attrs.get("model_name", "FID"),
        unit=fit_results["fit_amps"].attrs.get("unit", "a.u."),
    )
    return reconstructed


def _seeding_uses_voigt(fit_params: xr.Dataset, fit_results: xr.Dataset) -> bool:
    """Return whether results seed a Voigt fit; reject mixed lineshapes.

    Lorentz ``fit_t2s`` describe the full linewidth, Voigt ``fit_t2s`` only
    its Lorentz part, so the two cannot seed each other.
    """
    voigt = _require_same_lineshape("Seeding from fit_results", fit_params, fit_results)
    if voigt and (
        "initial_gauss_fwhm_hz" not in fit_params
        or "fit_gauss_fwhm_hz" not in fit_results
    ):
        raise ValueError(
            "Voigt seeding requires fit_params['initial_gauss_fwhm_hz'] and "
            "fit_results['fit_gauss_fwhm_hz']."
        )
    return voigt


def _prepare_base_fit_params(
    fit_params: xr.Dataset,
    fit_results: xr.Dataset | None,
) -> xr.Dataset:
    """Copy fit parameters and optionally seed them from existing fit results.

    Voigt results also seed the shared start Gaussian FWHM map.
    """
    base_params = fit_params.copy(deep=True)
    if fit_results is None:
        return base_params
    voigt = _seeding_uses_voigt(base_params, fit_results)

    try:
        merged_freqs = fit_results["fit_freqs"].combine_first(
            base_params["initial_freqs_hz"]
        )
        merged_t2s = fit_results["fit_t2s"].combine_first(base_params["initial_t2s_s"])
    except KeyError as exc:
        raise KeyError(
            "fit_results must contain 'fit_freqs' and 'fit_t2s' variables."
        ) from exc

    base_params["initial_freqs_hz"] = merged_freqs.transpose(
        *base_params["initial_freqs_hz"].dims
    )
    base_params["initial_t2s_s"] = merged_t2s.transpose(
        *base_params["initial_t2s_s"].dims
    )
    if voigt:
        base_params["initial_gauss_fwhm_hz"] = (
            fit_results["fit_gauss_fwhm_hz"]
            .combine_first(base_params["initial_gauss_fwhm_hz"])
            .transpose(*base_params["initial_gauss_fwhm_hz"].dims)
        )
    return base_params


def _normalize_global_fit_constraints(fit_params: xr.Dataset) -> xr.Dataset:
    """Return parameters with scalar global constraints stored as attributes.

    ``min_peak_separation_hz`` applies to every fitted frequency pair, so it
    cannot vary by metabolite or voxel.  Accept a legacy scalar data variable
    to keep interactive xarray assignments compatible, but promote it to the
    canonical dataset attribute before a metabolite is appended.
    """
    updated = fit_params
    attrs = dict(updated.attrs)
    for name in _GLOBAL_FIT_CONSTRAINT_NAMES:
        if name not in updated.data_vars:
            continue

        value = updated[name]
        if value.ndim != 0:
            raise ValueError(
                f"{name} is a global fit constraint and must be stored as a "
                f"scalar dataset attribute: fit_params.attrs['{name}']."
            )
        scalar_value = _coerce_finite_scalar(value.values, parameter_name=name)
        if scalar_value < 0.0:
            raise ValueError(f"{name} must be non-negative.")
        updated = updated.drop_vars(name)
        attrs[name] = scalar_value

    return updated.assign_attrs(attrs)


def _resolve_fit_frequency_axes(
    fit_params: xr.Dataset,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Return Hz and ppm calibration axes stored on a fit-parameter dataset."""
    hz_axis = resolve_frequency_axis(fit_params, "Hz")
    ppm_axis = resolve_frequency_axis(fit_params, "ppm")

    if hz_axis is None and "freq_axis_hz" in fit_params:
        hz_axis = fit_params["freq_axis_hz"].values
    if ppm_axis is None and "freq_axis_ppm" in fit_params:
        ppm_axis = fit_params["freq_axis_ppm"].values

    return hz_axis, ppm_axis


def _get_repetition_dim(data: xr.DataArray) -> str | None:
    """Return the repetition dimension of a fitted parameter, if present."""
    if "repetition_idx" in data.dims:
        return "repetition_idx"
    return next((dim for dim in data.dims if "rep" in dim.lower()), None)


def _select_reference_repetitions(
    reference_freqs: xr.DataArray,
    *,
    reference_repetitions: int | slice | range | Sequence[int] | str | None,
    reference_metabolite: str,
    fit_results: xr.Dataset | None,
) -> xr.DataArray:
    """Select and broadcast frequency references across repetitions."""
    if reference_repetitions is None:
        return reference_freqs

    repetition_dim = _get_repetition_dim(reference_freqs)
    if repetition_dim is None:
        raise ValueError(
            "reference_repetitions was provided, but no repetition dimension was found."
        )

    if isinstance(reference_repetitions, str):
        if reference_repetitions != "max_fit_amp":
            raise ValueError(
                "reference_repetitions must be an index, slice, range, sequence "
                "of indices, or 'max_fit_amp'."
            )
        if fit_results is None or "fit_amps" not in fit_results:
            raise ValueError(
                "reference_repetitions='max_fit_amp' requires fit_results with "
                "a 'fit_amps' variable."
            )
        reference_amps = fit_results["fit_amps"].sel(
            metabolite=reference_metabolite,
            drop=True,
        )
        if _get_repetition_dim(reference_amps) != repetition_dim:
            raise ValueError(
                "Unable to align repetition dimension between fit_freqs and "
                "fit_amps for reference selection."
            )
        selected_idx = np.abs(reference_amps).argmax(dim=repetition_dim)
        selected = reference_freqs.isel({repetition_dim: selected_idx})
        return selected.broadcast_like(reference_freqs)

    if isinstance(reference_repetitions, slice):
        selected = reference_freqs.isel({repetition_dim: reference_repetitions})
    elif isinstance(reference_repetitions, range):
        selected = reference_freqs.isel({repetition_dim: list(reference_repetitions)})
    elif isinstance(reference_repetitions, Sequence):
        selected = reference_freqs.isel({repetition_dim: list(reference_repetitions)})
    else:
        selected = reference_freqs.isel({repetition_dim: int(reference_repetitions)})

    if repetition_dim in selected.dims:
        if selected.sizes[repetition_dim] == 0:
            raise ValueError(
                "reference_repetitions selected no repetitions for the "
                "reference metabolite."
            )
        selected = selected.median(dim=repetition_dim, skipna=True)

    return selected.broadcast_like(reference_freqs)


def _broadcast_value_like(
    value: float | np.ndarray | xr.DataArray,
    template: xr.DataArray,
) -> xr.DataArray:
    """Broadcast a scalar or leading-dimension array to a parameter template."""
    array_value = np.asarray(value.data if isinstance(value, xr.DataArray) else value)
    if array_value.ndim == 0:
        return xr.full_like(template, array_value.item())
    if array_value.ndim > template.ndim:
        raise ValueError(
            f"Cannot align a {array_value.ndim}D value with the "
            f"{template.ndim}D fit-parameter template."
        )

    input_dims = template.dims[: array_value.ndim]
    value_da = xr.DataArray(array_value, dims=input_dims)
    missing_dims = [dim for dim in template.dims if dim not in input_dims]
    return value_da.expand_dims(missing_dims).transpose(*template.dims)


def add_metab_to_fit_params(
    new_freq: float | np.ndarray | xr.DataArray | None,
    new_t2_s: float | np.ndarray | xr.DataArray | None = None,
    new_metab_name: str = "new_metabolite",
    fit_params: xr.Dataset | None = None,
    fit_results: xr.Dataset | None = None,
    new_freq_range: float | None = None,
    new_t2_range_s: float | None = None,
    new_min_t2_s: float | None = None,
    new_max_t2_s: float | None = None,
    new_freq_unit: str = "Hz",
    reference_metabolite: str | None = None,
    freq_offset: float = 0.0,
    freq_offset_unit: str = "Hz",
    reference_repetitions: int | slice | range | Sequence[int] | str | None = None,
    freeze_other_freqs: bool = True,
    freeze_other_t2s: bool = True,
    t2_coupling: str | tuple[str, float | tuple[float, float]] | None = None,
) -> xr.Dataset:
    """Add a metabolite to an existing spectroscopy fit configuration.

    This is the one owner of adding a line: scripts, the CSI pipeline and the
    interactive review plugin call it, so the related entries (metabolite
    list, frequency start and window, T2* start, range and bounds, coupling)
    stay consistent. The initial frequency can be supplied directly or derived
    from a fitted reference metabolite plus ``freq_offset``. Existing fitted
    frequencies and T2* values are used as updated initial values when
    ``fit_results`` is provided. Settings left at None take the
    ``def_fit_params`` defaults, so with both freeze flags False adding a line
    equals preparing the extended parameter set from scratch.

    Parameters
    ----------
    new_freq : float, numpy.ndarray, xarray.DataArray, or None
        Initial frequency of the new metabolite. Must be ``None`` when
        ``reference_metabolite`` is set; otherwise it is required.
    new_t2_s : float, numpy.ndarray, xarray.DataArray, or None, optional
        Initial T2* value in seconds, a scalar or one value per spectrum
        (leading loop dimensions of the parameter maps). None uses the
        ``def_fit_params`` default of 0.025 s.
    new_metab_name : str, optional
        Name of the new metabolite. Default is ``"new_metabolite"``.
    fit_params : xr.Dataset
        Existing fit-parameter dataset to extend. Global constraints such as
        ``min_peak_separation_hz`` belong in dataset attributes, because they
        apply to every frequency pair. A legacy scalar data variable with this
        name is promoted to the corresponding attribute; a non-scalar one is
        rejected.
    fit_results : xr.Dataset, optional
        Previous fit results used to refresh initial frequencies and T2*
        values before adding the metabolite. Voigt results also refresh the
        shared start Gaussian FWHM ``initial_gauss_fwhm_hz``; results and
        parameters must use the same lineshape.
    new_freq_range : float or None, optional
        Full frequency search width in ``new_freq_unit``; the search spans the
        start frequency +/- half of it. It must not exceed the spectral width.
        None uses the ``def_fit_params`` default of 70 Hz (in Hz whatever
        ``new_freq_unit`` is).
    new_t2_range_s : float or None, optional
        Full T2* search range in seconds; 0 freezes T2*. None uses the
        ``def_fit_params`` default of 0.04 s.
    new_min_t2_s : float or None, optional
        Minimum allowed T2* in seconds. None uses the ``def_fit_params``
        default of 0.005 s.
    new_max_t2_s : float or None, optional
        Maximum allowed T2* in seconds. None uses the ``def_fit_params``
        default: 0.07 s, or 1.0 s for ``lineshape='voigt'`` (where it bounds
        only the Lorentz part, so that a mainly Gaussian line stays
        reachable).
    new_freq_unit : str, optional
        Unit of ``new_freq`` and ``new_freq_range``: ``"Hz"`` or ``"ppm"``.
    reference_metabolite : str or None, optional
        Existing metabolite whose fitted frequency provides the reference. If
        supplied, ``new_freq`` must be ``None``. If omitted, ``new_freq`` is
        required.
    freq_offset : float, optional
        Offset from the reference frequency. Default is 0.0.
    freq_offset_unit : str, optional
        Unit of ``freq_offset``: ``"Hz"`` or ``"ppm"``.
    reference_repetitions : int, slice, range, Sequence[int], str, or None, optional
        Repetition selection for the reference frequency. ``None`` preserves
        each repetition's own value; a multi-repetition selection uses its
        median; ``"max_fit_amp"`` selects the repetition with maximum fitted
        amplitude.
    freeze_other_freqs : bool, optional
        Set existing metabolites' frequency search widths to zero. Default is
        True.
    freeze_other_t2s : bool, optional
        Set existing metabolites' T2* search ranges to zero. Default is True.
    t2_coupling : str, tuple, or None, optional
        Couple the new line's T2* to an existing leader, in the
        ``def_fit_params`` ``t2_coupling`` form: a leader name (ratio 1.0), a
        ``(leader, t2_ratio)`` pair with ``t2_ratio > 0`` or a
        ``(leader, (low, high))`` ratio band. The entry is added to the
        ``t2_coupling`` attribute. None (default) leaves the new line
        uncoupled.

    Returns
    -------
    xr.Dataset
        A copied and expanded fit-parameter dataset. Variables with a
        ``metabolite`` dimension gain the new line; per-line variables that
        this function does not set, such as start amplitudes, are NaN for it.
        Variables without ``metabolite``, such as the shared Gaussian start
        map ``initial_gauss_fwhm_hz``, keep their dimensions and values. The
        attributes ``metabs``, ``metabs_freqs_hz`` and ``metabs_freqs_ppm``
        (when present) and ``t2_coupling`` include the new line, so
        ``dict(result.attrs)`` can be prepared again. The input datasets are
        not mutated.

    Raises
    ------
    ValueError
        If required fit parameters, units, frequency axes, or reference
        selections are invalid, or ``fit_results`` has another lineshape; if
        the name is already present; if the start frequency lies outside the
        frequency axis stored on ``fit_params`` or the frequency window is
        wider than its spectral width; if the coupling leader is unknown or
        itself a dependent. Also
        if, in any spectrum, two start frequencies of lines that are not
        skipped (the new line included) lie closer than
        ``min_peak_separation_hz`` (attribute of ``fit_params``, default 5 Hz),
        because ``fit_data_pseudo_inv`` would reject them. With ``fit_results``
        the existing starts are the refreshed fitted frequencies.

    Examples
    --------
    Add lactate 400 Hz above the fitted pyruvate line with default windows,
    leaving the other lines free:

    >>> extended = add_metab_to_fit_params(  # doctest: +SKIP
    ...     None,
    ...     new_metab_name="lactate",
    ...     fit_params=params,
    ...     reference_metabolite="pyruvate",
    ...     freq_offset=400.0,
    ...     freeze_other_freqs=False,
    ...     freeze_other_t2s=False,
    ... )
    """
    if fit_params is None:
        raise ValueError("fit_params must be provided.")

    use_reference = reference_metabolite is not None
    if not use_reference and new_freq is None:
        raise ValueError(
            "Either new_freq must be provided or reference_metabolite must be set."
        )
    if use_reference and new_freq is not None:
        raise ValueError(
            "new_freq and reference_metabolite are mutually exclusive. Set "
            "new_freq=None when deriving the frequency from a reference "
            "metabolite, or set reference_metabolite=None to use an absolute "
            "frequency."
        )

    if not isinstance(new_metab_name, str) or not new_metab_name.strip():
        raise ValueError("new_metab_name must be a non-empty string.")
    if (
        "metabolite" in fit_params.coords
        and new_metab_name in fit_params.metabolite.values
    ):
        raise ValueError(
            f"new_metab_name '{new_metab_name}' is already present in fit_params."
        )

    if not use_reference and reference_repetitions is not None:
        raise ValueError(
            "reference_repetitions requires reference_metabolite to be set."
        )

    freq_offset = _coerce_finite_scalar(
        freq_offset,
        parameter_name="freq_offset",
    )
    if not use_reference and freq_offset != 0.0:
        raise ValueError("freq_offset requires reference_metabolite to be set.")

    # Defaults are the def_fit_params defaults; the frequency window default
    # is in Hz whatever new_freq_unit is.
    freq_range_in_hz = new_freq_range is None
    if freq_range_in_hz:
        new_freq_range = METABS_FREQ_RANGE_DEFAULT
    if new_t2_s is None:
        new_t2_s = INITIAL_T2_S_DEFAULT
    if new_t2_range_s is None:
        new_t2_range_s = METABS_T2_RANGE_DEFAULT
    if new_min_t2_s is None:
        new_min_t2_s = MIN_T2_S_DEFAULT
    if new_max_t2_s is None:
        new_max_t2_s = (
            VOIGT_MAX_T2_S_DEFAULT
            if _uses_voigt_lineshape(fit_params.attrs)
            else MAX_T2_S_DEFAULT
        )

    new_freq_range = _coerce_finite_scalar(
        new_freq_range,
        parameter_name="new_freq_range",
    )
    if new_freq_range < 0.0:
        raise ValueError("new_freq_range must be non-negative.")

    new_t2_range_s = _coerce_finite_scalar(
        new_t2_range_s,
        parameter_name="new_t2_range_s",
    )
    if new_t2_range_s < 0.0:
        raise ValueError("new_t2_range_s must be non-negative.")

    new_min_t2_s = _coerce_finite_scalar(
        new_min_t2_s,
        parameter_name="new_min_t2_s",
    )
    new_max_t2_s = _coerce_finite_scalar(
        new_max_t2_s,
        parameter_name="new_max_t2_s",
    )
    if new_min_t2_s <= 0.0:
        raise ValueError("new_min_t2_s must be positive.")
    if new_max_t2_s < new_min_t2_s:
        raise ValueError("new_max_t2_s must be greater than or equal to new_min_t2_s.")

    try:
        new_t2_values = np.asarray(
            new_t2_s.data if isinstance(new_t2_s, xr.DataArray) else new_t2_s,
            dtype=float,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("new_t2_s must contain finite numeric values.") from exc
    if not np.all(np.isfinite(new_t2_values)):
        raise ValueError("new_t2_s must contain finite numeric values.")
    if np.any(new_t2_values < new_min_t2_s) or np.any(new_t2_values > new_max_t2_s):
        raise ValueError("new_t2_s must lie within [new_min_t2_s, new_max_t2_s].")

    allowed_units = {"hz", "ppm"}
    for parameter_name, unit in (
        ("new_freq_unit", new_freq_unit),
        ("freq_offset_unit", freq_offset_unit),
    ):
        if unit.lower() not in allowed_units:
            raise ValueError(
                f"Unknown {parameter_name} '{unit}'. Expected one of {allowed_units}."
            )

    base_params = _normalize_global_fit_constraints(
        _prepare_base_fit_params(fit_params, fit_results)
    )
    hz_axis, ppm_axis = _resolve_fit_frequency_axes(fit_params)

    if new_freq_unit.lower() == "ppm" and not freq_range_in_hz:
        new_freq_range_arr = np.asarray(new_freq_range)
        if np.any(np.abs(new_freq_range_arr) > 1.0):
            warnings.warn(
                "new_freq_range is larger than 1 ppm. If you meant to specify the search range in Hz, convert it to ppm before calling add_metab_to_fit_params; otherwise the search range may become very large.",
                UserWarning,
                stacklevel=2,
            )
        new_freq_range = np.abs(
            _delta_ppm2hz(
                new_freq_range,
                hz_axis=hz_axis,
                ppm_axis=ppm_axis,
            )
        )
    if new_freq_unit.lower() == "ppm" and not use_reference:
        new_freq = _convert_ppm_value(
            new_freq,
            hz_axis=hz_axis,
            ppm_axis=ppm_axis,
        )

    freq_offset_hz = freq_offset
    if freq_offset_unit.lower() == "ppm":
        freq_offset_hz = _delta_ppm2hz(
            freq_offset,
            hz_axis=hz_axis,
            ppm_axis=ppm_axis,
        )

    freq_offset_hz = _coerce_finite_scalar(
        freq_offset_hz,
        parameter_name="converted freq_offset",
    )

    if use_reference:
        if reference_metabolite not in base_params.metabolite.values:
            raise ValueError(
                f"reference_metabolite '{reference_metabolite}' is not present in the fit parameters."
            )
        reference_freqs = base_params["initial_freqs_hz"].sel(
            metabolite=reference_metabolite, drop=True
        )
        reference_freqs = _select_reference_repetitions(
            reference_freqs,
            reference_repetitions=reference_repetitions,
            reference_metabolite=reference_metabolite,
            fit_results=fit_results,
        )
        new_freq = np.asarray(reference_freqs.data) + freq_offset_hz
    elif new_freq is None:
        raise ValueError(
            "new_freq must be provided when reference_metabolite is not specified."
        )

    if isinstance(new_freq, xr.DataArray):
        new_freq = np.asarray(new_freq.data)
    elif isinstance(new_freq, list):
        new_freq = np.asarray(new_freq)

    if isinstance(new_freq, np.ndarray):
        new_freq = np.asarray(new_freq, dtype=float)
        if new_freq.ndim == 0:
            new_freq = float(new_freq.item())

    new_freq_range = _coerce_finite_scalar(
        new_freq_range,
        parameter_name="converted new_freq_range",
    )
    _check_new_line_inside_spectrum(
        new_metab_name, new_freq, new_freq_range, hz_axis, base_params.attrs
    )

    min_peak_separation_hz = base_params.attrs.get("min_peak_separation_hz", None)

    if min_peak_separation_hz is not None:
        try:
            min_peak_separation_hz = float(
                np.asarray(min_peak_separation_hz, dtype=float).reshape(-1)[0]
            )
        except (TypeError, ValueError, IndexError):
            min_peak_separation_hz = 0.0
        if (
            min_peak_separation_hz > 0.0
            and new_freq_range > 0.0
            and new_freq_range < 2.0 * min_peak_separation_hz
        ):
            warnings.warn(
                "new_freq_range is interpreted as the full search width "
                f"(initial frequency +/- {new_freq_range / 2.0:.3g} Hz). "
                f"With min_peak_separation_hz={min_peak_separation_hz:.3g}, "
                "this may be too narrow for close or drifting peaks.",
                UserWarning,
                stacklevel=2,
            )

    template = base_params.isel(metabolite=0, drop=True)
    new_peak_data = xr.Dataset(coords={"metabolite": [new_metab_name]})
    new_peak_data["initial_freqs_hz"] = _broadcast_value_like(
        new_freq,
        template["initial_freqs_hz"],
    )
    new_peak_data["initial_t2s_s"] = _broadcast_value_like(
        new_t2_s,
        template["initial_t2s_s"],
    )

    new_peak_data["metabs_freq_ranges"] = xr.full_like(
        template["metabs_freq_ranges"], new_freq_range
    )
    new_peak_data["metabs_t2_ranges"] = xr.full_like(
        template["metabs_t2_ranges"], float(new_t2_range_s)
    )
    new_peak_data["min_t2_s"] = xr.full_like(template["min_t2_s"], new_min_t2_s)
    new_peak_data["max_t2_s"] = xr.full_like(template["max_t2_s"], new_max_t2_s)
    if "skip_fitting" in base_params:
        new_peak_data["skip_fitting"] = xr.full_like(
            template["initial_freqs_hz"], False, dtype=bool
        )
    # Only variables with a "metabolite" dimension gain the new line; shared
    # variables such as the Gaussian start map keep their dimensions and values.
    # Per-line variables the new line does not set are NaN for it.
    updated_params = xr.concat(
        [base_params, new_peak_data], dim="metabolite", data_vars="minimal"
    )

    if "skip_fitting" in updated_params:
        skip_da = updated_params["skip_fitting"]
        if np.issubdtype(skip_da.dtype, np.floating):
            skip_da = skip_da.fillna(0.0)
        updated_params["skip_fitting"] = skip_da.astype(bool)

    # xr.where puts the condition's "metabolite" dimension first; the range
    # maps keep the dimension order of the input.
    is_old_peak = updated_params.metabolite != new_metab_name
    for frozen, name in (
        (freeze_other_freqs, "metabs_freq_ranges"),
        (freeze_other_t2s, "metabs_t2_ranges"),
    ):
        if frozen:
            ranges = updated_params[name]
            updated_params[name] = xr.where(is_old_peak, 0.0, ranges).transpose(
                *ranges.dims
            )

    updated_params.attrs = _attrs_with_added_line(
        base_params.attrs,
        metabolites=updated_params.metabolite.values.tolist(),
        new_freq_hz=updated_params["initial_freqs_hz"].sel(metabolite=new_metab_name),
        hz_axis=hz_axis,
        ppm_axis=ppm_axis,
        t2_coupling=t2_coupling,
    )
    _check_start_separation(
        updated_params,
        DEFAULT_MIN_PEAK_SEPARATION_HZ
        if min_peak_separation_hz is None
        else min_peak_separation_hz,
        operation="add_metab_to_fit_params",
    )

    return _update_frequency_bounds(updated_params)


def _check_new_line_inside_spectrum(
    name: str,
    new_freq_hz: float | np.ndarray,
    freq_range_hz: float,
    hz_axis: np.ndarray | None,
    attrs: dict,
) -> None:
    """Reject an added line whose start or window leaves the spectral width.

    The spectral width is ``attrs['bw_hz']`` when stored, else the extent of
    the Hz axis (bin count times bin spacing). Without a Hz axis nothing is
    checked, because the parameters then carry no spectral calibration.
    """
    if hz_axis is None or np.size(hz_axis) < 2:
        return
    axis = np.asarray(hz_axis, dtype=float)
    low, high = float(np.min(axis)), float(np.max(axis))
    starts = np.asarray(new_freq_hz, dtype=float)
    if not np.all(np.isfinite(starts)):
        raise ValueError(
            f"add_metab_to_fit_params: the start frequency of {name!r} must be finite."
        )
    if np.any(starts < low) or np.any(starts > high):
        raise ValueError(
            f"add_metab_to_fit_params: the start frequency of {name!r} "
            f"({float(np.min(starts)):.6g} to {float(np.max(starts)):.6g} Hz) "
            f"lies outside the frequency axis [{low:.6g}, {high:.6g}] Hz. Use a "
            "start inside the spectrum (check the unit, Hz or ppm)."
        )
    bandwidth_hz = attrs.get("bw_hz")
    if bandwidth_hz is None:
        bandwidth_hz = axis.size * abs(float(axis[1] - axis[0]))
    bandwidth_hz = float(bandwidth_hz)
    if freq_range_hz > bandwidth_hz:
        raise ValueError(
            f"add_metab_to_fit_params: the frequency window of {name!r} "
            f"({freq_range_hz:.6g} Hz) is wider than the spectral width "
            f"({bandwidth_hz:.6g} Hz). Use a smaller new_freq_range."
        )


def _attrs_with_added_line(
    attrs: dict,
    *,
    metabolites: list[str],
    new_freq_hz: xr.DataArray,
    hz_axis: np.ndarray | None,
    ppm_axis: np.ndarray | None,
    t2_coupling: str | tuple | None,
) -> dict:
    """Return copied fit options that list the added (last) line.

    ``metabs_freqs_hz`` and ``metabs_freqs_ppm`` gain the new line's mean
    start frequency when they list every previous line; the per-spectrum
    starts stay in ``initial_freqs_hz``. A new coupling entry is validated
    together with the existing ones.
    """
    updated = dict(attrs)
    name = metabolites[-1]
    updated["metabs"] = list(metabolites)
    start_hz = float(np.mean(np.asarray(new_freq_hz, dtype=float)))
    n_previous = len(metabolites) - 1
    if len(updated.get("metabs_freqs_hz") or []) == n_previous:
        updated["metabs_freqs_hz"] = [*updated["metabs_freqs_hz"], start_hz]
    if len(updated.get("metabs_freqs_ppm") or []) == n_previous:
        start_ppm = None
        if hz_axis is not None and ppm_axis is not None:
            start_ppm = float(freq_hz_to_ppm(start_hz, hz_axis, ppm_axis))
        updated["metabs_freqs_ppm"] = [*updated["metabs_freqs_ppm"], start_ppm]
    if t2_coupling is not None:
        updated["t2_coupling"] = _normalize_t2_coupling(
            {**(updated.get("t2_coupling") or {}), name: t2_coupling}, metabolites
        )
    return updated


# Option attrs that hold one entry per line in ``metabs`` order. A list of
# another length (or a scalar) applies to every line and is kept unchanged.
_PER_LINE_OPTION_ATTRS = ("metabs", *sorted(_START_MAP_ATTRS))


def remove_metab_from_fit_params(
    fit_params: xr.Dataset, *, metab_name: str
) -> xr.Dataset:
    """Remove one line from a spectroscopy fit configuration.

    The counterpart of :func:`add_metab_to_fit_params`: the line leaves every
    variable with a ``metabolite`` dimension and every per-line option
    attribute, and its ``t2_coupling`` entry is dropped, so removing a line
    equals preparing the reduced line set with :func:`def_fit_params`.

    Parameters
    ----------
    fit_params : xr.Dataset
        Fit-parameter dataset with a ``metabolite`` coordinate, as returned by
        :func:`def_fit_params` or :func:`add_metab_to_fit_params`. Not
        mutated.
    metab_name : str
        Name of the line to remove. It must be present and must not lead the
        T2* of another line.

    Returns
    -------
    xr.Dataset
        A copy without the line. Variables without a ``metabolite`` dimension,
        such as the Voigt start map ``initial_gauss_fwhm_hz``, are unchanged.
        The attributes ``metabs`` and the per-line options
        (``metabs_freqs_hz``, ``metabs_freqs_ppm``, ``initial_t2s_s`` and the
        other start-map options, where they list one entry per line) lose the
        line's entry. A ``t2_coupling`` attribute loses the line's entry and is
        removed when no coupling is left.

    Raises
    ------
    ValueError
        If the line is not in ``fit_params``, if it is the only line, or if it
        leads the T2* of other lines. Remove those dependents first, or prepare
        the parameters again without the coupling.

    Examples
    --------
    >>> reduced = remove_metab_from_fit_params(params, metab_name="alanine")  # doctest: +SKIP
    """
    names = [str(name) for name in fit_params.metabolite.values]
    if metab_name not in names:
        raise ValueError(
            f"remove_metab_from_fit_params: line {metab_name!r} is not in "
            f"fit_params (lines: {names})."
        )
    if len(names) == 1:
        raise ValueError(
            f"remove_metab_from_fit_params: {metab_name!r} is the only line; a "
            "fit needs at least one line."
        )
    coupling = dict(fit_params.attrs.get("t2_coupling") or {})
    dependents = sorted(
        dependent for dependent, (leader, _) in coupling.items() if leader == metab_name
    )
    if dependents:
        raise ValueError(
            f"remove_metab_from_fit_params: {metab_name!r} leads the T2* of "
            f"{dependents} through t2_coupling. Remove those lines first, or "
            "prepare the parameters again without this coupling."
        )

    index = names.index(metab_name)
    reduced = fit_params.drop_sel(metabolite=[metab_name])
    attrs = dict(fit_params.attrs)
    for key in _PER_LINE_OPTION_ATTRS:
        value = attrs.get(key)
        if isinstance(value, (list, tuple)) and len(value) == len(names):
            attrs[key] = [entry for i, entry in enumerate(value) if i != index]
    coupling.pop(metab_name, None)
    if coupling:
        attrs["t2_coupling"] = coupling
    else:
        attrs.pop("t2_coupling", None)
    reduced.attrs = attrs
    return reduced


def _broadcast_search_ranges(
    values: float | Sequence[float] | np.ndarray | xr.DataArray | None,
    template: xr.DataArray,
    *,
    parameter_name: str,
) -> xr.DataArray:
    """Align optional finite, non-negative search widths to a parameter map.

    None retains existing widths. Scalars apply everywhere and sequences follow
    metabolite order. A DataArray may also vary along the template's voxel
    dimensions, but must contain metabolite and have matching coordinate labels.
    """
    if values is None:
        return template
    if isinstance(values, xr.DataArray):
        if "metabolite" not in values.dims or not set(values.dims) <= set(
            template.dims
        ):
            raise ValueError(
                f"{parameter_name} must contain 'metabolite' and only dimensions "
                "present in fit_params."
            )
        try:
            _, parameter = xr.align(template, values, join="exact", copy=False)
        except ValueError as exc:
            raise ValueError(
                f"{parameter_name} coordinates must match fit_params."
            ) from exc
        if np.iscomplexobj(parameter):
            raise ValueError(f"{parameter_name} must contain finite real values.")
        try:
            parameter = parameter.astype(float)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{parameter_name} must contain finite real values."
            ) from exc
    else:
        parameter = xr.DataArray(
            _coerce_metabolite_values(
                values, template.sizes["metabolite"], parameter_name=parameter_name
            ),
            dims=["metabolite"],
            coords={"metabolite": template.coords["metabolite"]},
        )
    if not np.all(np.isfinite(parameter)) or np.any(parameter < 0):
        raise ValueError(f"{parameter_name} must contain finite, non-negative widths.")
    return parameter.broadcast_like(template).transpose(*template.dims)


def initialise_fit_params_from_results(
    fit_params: xr.Dataset,
    fit_results: xr.Dataset,
    freq_ranges_hz: float | Sequence[float] | np.ndarray | xr.DataArray | None = None,
    t2_ranges_s: float | Sequence[float] | np.ndarray | xr.DataArray | None = None,
) -> xr.Dataset:
    """Copy fitted frequencies and T2* values into the next fit's initial guesses.

    Parameters
    ----------
    fit_params : xr.Dataset
        Existing parameter maps and options. The input dataset is not modified.
    fit_results : xr.Dataset
        Results containing ``fit_freqs`` and ``fit_t2s`` for the same metabolites
        and voxels. Dimension order may differ; coordinate labels must match.
        Voigt results also seed ``initial_gauss_fwhm_hz`` from
        ``fit_gauss_fwhm_hz``; results and parameters must use the same
        lineshape.
    freq_ranges_hz, t2_ranges_s : scalar, sequence, xr.DataArray or None
        Optional full search widths, in Hz and seconds respectively. None keeps
        existing widths, a scalar applies everywhere, and a sequence follows
        metabolite order. Labelled DataArrays may vary by voxel. Widths must be
        finite and non-negative; zero freezes the corresponding parameter.

    Returns
    -------
    xr.Dataset
        Updated initial guesses and search widths, with frequency bounds
        recalculated as initial frequency +/- half the full search width.
        Where a fitted value is NaN or infinite, or a fitted T2* is not
        positive (for example voxels whose input data contain NaN), the
        existing initial value is kept.

    Raises
    ------
    ValueError
        If dimensions or coordinate labels disagree, supplied widths are
        invalid, or the lineshapes differ. Coordinate mismatches are rejected
        before xarray can fill missing labels with NaNs during assignment.
    """
    if fit_params is None or fit_results is None:
        raise ValueError("fit_params and fit_results must both be provided.")

    updated_params = fit_params.copy(deep=True)
    seeded_maps = (
        ("initial_freqs_hz", "fit_freqs"),
        ("initial_t2s_s", "fit_t2s"),
    )
    if _seeding_uses_voigt(fit_params, fit_results):
        seeded_maps += (("initial_gauss_fwhm_hz", "fit_gauss_fwhm_hz"),)
    try:
        for target, source in seeded_maps:
            template = fit_params[target]
            fitted = fit_results[source].transpose(*template.dims)
            _, fitted = xr.align(template, fitted, join="exact", copy=False)
            # Unfittable voxels (for example NaN input) report NaN results;
            # keep their previous start values so the next fit stays valid.
            usable = np.isfinite(fitted)
            if target != "initial_freqs_hz":
                usable &= fitted > 0 if target == "initial_t2s_s" else fitted >= 0
            updated_params[target] = fitted.where(usable, template)
    except KeyError as exc:
        raise KeyError(
            "fit_results must contain 'fit_freqs' and 'fit_t2s' variables."
        ) from exc
    except ValueError as exc:
        raise ValueError(
            "Unable to align fit_results with fit_params; check dimensions and "
            "coordinate labels."
        ) from exc

    updated_params["metabs_freq_ranges"] = _broadcast_search_ranges(
        freq_ranges_hz,
        updated_params["metabs_freq_ranges"],
        parameter_name="freq_ranges_hz",
    )
    updated_params["metabs_t2_ranges"] = _broadcast_search_ranges(
        t2_ranges_s,
        updated_params["metabs_t2_ranges"],
        parameter_name="t2_ranges_s",
    )

    return _update_frequency_bounds(updated_params)


_NEIGHBOUR_OFFSETS = tuple(
    (row, column)
    for row in (-1, 0, 1)
    for column in (-1, 0, 1)
    if (row, column) != (0, 0)
)


def _median_of_estimated_neighbours(
    gauss_fwhm_hz: xr.DataArray,
    status: xr.DataArray,
    plane_dims: tuple[str, str],
) -> xr.DataArray:
    """Return the median Gaussian FWHM of the 8 in-plane neighbours per voxel.

    Only neighbours with the status ``estimated`` count; voxels without such
    a neighbour get NaN. The edge of the plane has fewer neighbours.
    """
    estimated = gauss_fwhm_hz.where(status == "estimated")
    neighbours = xr.concat(
        [
            estimated.shift({plane_dims[0]: row, plane_dims[1]: column})
            for row, column in _NEIGHBOUR_OFFSETS
        ],
        dim="neighbour",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return neighbours.median("neighbour", skipna=True)


def refit_gauss_width_from_neighbours(
    data: xr.DataArray,
    fit_params: xr.Dataset,
    fit_results: xr.Dataset,
    plane_dims: tuple[str, str] = ("read_idx", "phase_idx"),
    show_tqdm: bool | None = None,
    n_jobs: int | None = 1,
) -> xr.Dataset:
    """Refit voxels without a usable Gaussian width with the neighbours' median.

    A Voigt fit leaves the Gaussian FWHM undetermined in weak voxels: it ends
    at 0 Hz (``fit_gauss_fwhm_status`` ``at_lower_bound``) or the fit fails
    (``fit_status`` ``fit_failed``). This optional second pass fixes the
    Gaussian FWHM of each such voxel to the median of its 8 neighbours in the
    plane ``plane_dims`` whose status is ``estimated``, and fits the voxel
    again with all other parameters from ``fit_params``. Voxels without an
    estimated neighbour, and all other voxels, keep their results.

    Parameters
    ----------
    data : xr.DataArray
        The spectra that gave ``fit_results``, passed unchanged to
        :func:`fit_data_pseudo_inv`.
    fit_params : xr.Dataset
        The Voigt fit parameters that gave ``fit_results``. They are not
        modified.
    fit_results : xr.Dataset
        Voigt results of :func:`fit_data_pseudo_inv` for ``data`` and
        ``fit_params``, with ``fit_gauss_fwhm_hz`` (Hz),
        ``fit_gauss_fwhm_status`` and ``fit_status``. They are not modified.
    plane_dims : tuple of str, optional
        The two dimensions of the result that span the neighbourhood, by
        default ``("read_idx", "phase_idx")``. Every other dimension, such as
        a slice or a repetition, is treated separately.
    show_tqdm, n_jobs : optional
        Passed to :func:`fit_data_pseudo_inv` for the second pass.

    Returns
    -------
    xr.Dataset
        A copy of ``fit_results`` in which the refitted voxels hold the
        results of the second pass. Their ``fit_gauss_fwhm_status`` is
        ``neighbour_median`` when the second fit succeeds and ``not_fitted``
        otherwise. The attribute ``gauss_neighbour_fill`` records the rule.

    Raises
    ------
    ValueError
        If the fit is not a Voigt fit or a dimension in ``plane_dims`` is
        missing from ``fit_results``.

    Notes
    -----
    This is a scientific choice, off unless called: the filled width comes
    from the neighbours, not from the voxel's own data. The second pass fits
    only the target voxels; the others are skipped. The result is
    deterministic for deterministic fits.

    Examples
    --------
    >>> results = fit_data_pseudo_inv(data, fit_params)  # doctest: +SKIP
    >>> filled = refit_gauss_width_from_neighbours(
    ...     data, fit_params, results, plane_dims=("read_idx", "phase_idx")
    ... )  # doctest: +SKIP
    """
    if not _uses_voigt_lineshape(fit_params.attrs) or (
        "fit_gauss_fwhm_hz" not in fit_results
    ):
        raise ValueError(
            "refit_gauss_width_from_neighbours needs a Voigt fit: fit_params "
            "with lineshape='voigt' and fit_results with fit_gauss_fwhm_hz."
        )
    missing = [dim for dim in plane_dims if dim not in fit_results.dims]
    if len(plane_dims) != 2 or missing:
        raise ValueError(
            "refit_gauss_width_from_neighbours: plane_dims must name two "
            f"dimensions of fit_results {tuple(fit_results.dims)}; missing "
            f"{missing}."
        )
    status = fit_results["fit_gauss_fwhm_status"]
    median_hz = _median_of_estimated_neighbours(
        fit_results["fit_gauss_fwhm_hz"], status, tuple(plane_dims)
    )
    targets = (
        (status == "at_lower_bound") | (fit_results["fit_status"] == "fit_failed")
    ) & np.isfinite(median_hz)
    filled = fit_results.copy(deep=True)
    filled.attrs["gauss_neighbour_fill"] = (
        "Gaussian FWHM fixed to the median of the estimated 8 neighbours in "
        f"{tuple(plane_dims)} where it was at_lower_bound or the fit failed"
    )
    if not bool(targets.any()):
        return filled

    refit_params = fit_params.copy(deep=True)
    refit_params["initial_gauss_fwhm_hz"] = median_hz.where(
        targets, refit_params["initial_gauss_fwhm_hz"]
    )
    skip = refit_params.get("skip_fitting")
    skip = (
        xr.zeros_like(refit_params["initial_freqs_hz"], dtype=bool)
        if skip is None
        else skip.astype(bool)
    )
    refit_params["skip_fitting"] = skip | ~targets
    refit_params.attrs["gauss_search_width_hz"] = 0.0
    refit = fit_data_pseudo_inv(data, refit_params, show_tqdm=show_tqdm, n_jobs=n_jobs)
    refit["fit_gauss_fwhm_status"] = xr.where(
        refit["fit_status"] == "success", "neighbour_median", "not_fitted"
    )
    for name, variable in fit_results.data_vars.items():
        if name not in refit or not set(targets.dims) <= set(variable.dims):
            continue
        merged = xr.where(targets, refit[name], variable, keep_attrs=True)
        merged = merged.transpose(*variable.dims)
        if variable.dtype.kind not in "OSU":
            merged = merged.astype(variable.dtype)
        filled[name] = merged
    return filled


def mask_fit_dataset_where_input_nonfinite(
    dataset: xr.Dataset,
    data: xr.DataArray,
) -> xr.Dataset:
    """Set per-voxel fit values to NaN where the fitted input is not finite.

    A voxel is invalid when any sample of ``data`` along the spectral
    dimension is NaN or infinite, in the real or the imaginary part. Every
    floating-point or complex data variable of ``dataset`` that has at least
    one non-spectral dimension of ``data`` becomes NaN (``nan+nanj``
    for complex values) at the invalid voxels. This applies to start values,
    search ranges and bounds of ``fit_params`` as well as to amplitudes,
    frequencies, T2* values, linewidths and diagnostics of ``fit_results``.

    Parameters
    ----------
    dataset : xr.Dataset
        Fit parameters from :func:`def_fit_params` or results from
        :func:`fit_data_pseudo_inv`. ``attrs['spectral_dim']`` names the
        spectral dimension of ``data``; ``'acquisition_idx'`` is used
        when it is absent. Not modified.
    data : xr.DataArray
        Data that the parameters belong to or that were fitted, with the
        spectral dimension and the voxel dimensions (for example
        ``read_idx``, ``phase_idx``, ``slice_idx`` or repetitions). Voxel
        dimensions with coordinates in both objects are matched by label;
        dataset positions missing from ``data`` stay unchanged.
        Dimensions without coordinates are matched by position.

    Returns
    -------
    xr.Dataset
        Deep copy of ``dataset`` with identical dimensions, dimension order,
        coordinates, attributes and dtypes. Boolean, integer and string
        variables (``skip_fitting``, ``fit_status``, ``fit_component_index``
        and so on) and variables without a voxel dimension are unchanged.

    Raises
    ------
    ValueError
        If ``data`` lacks the spectral dimension, or if a voxel
        dimension without coordinates has a different size in ``dataset``.

    Notes
    -----
    A variable that lacks some voxel dimensions of ``data``, for
    example a parameter shared by all repetitions, is set to NaN only where
    the input is invalid for every value of the missing dimensions.

    Examples
    --------
    ::

        result = fit_data_pseudo_inv(data, params)
        params = mask_fit_dataset_where_input_nonfinite(params, data)
        result = mask_fit_dataset_where_input_nonfinite(result, data)
    """
    spectral_dim = dataset.attrs.get("spectral_dim", "acquisition_idx")
    if spectral_dim not in data.dims:
        raise ValueError(
            "mask_fit_dataset_where_input_nonfinite: data has no spectral "
            f"dimension '{spectral_dim}' (dims {data.dims}); pass the "
            "data that were fitted or set dataset.attrs['spectral_dim']."
        )
    invalid = xr.DataArray(
        ~np.isfinite(np.asarray(data.values)),
        dims=data.dims,
        coords={
            dim: data.coords[dim]
            for dim in data.dims
            if dim != spectral_dim and dim in data.coords
        },
    ).any(spectral_dim)

    masked = dataset.copy(deep=True)
    for name, variable in dataset.data_vars.items():
        if variable.dtype.kind not in "fc":
            continue
        voxel_dims = [dim for dim in invalid.dims if dim in variable.dims]
        if not voxel_dims:
            continue
        missing_dims = [dim for dim in invalid.dims if dim not in variable.dims]
        voxel_invalid = invalid.all(missing_dims) if missing_dims else invalid
        for dim in voxel_dims:
            if dim in voxel_invalid.indexes and dim in variable.indexes:
                voxel_invalid = voxel_invalid.reindex(
                    {dim: variable.indexes[dim]}, fill_value=False
                )
            elif variable.sizes[dim] != voxel_invalid.sizes[dim]:
                raise ValueError(
                    "mask_fit_dataset_where_input_nonfinite: dimension "
                    f"'{dim}' of '{name}' has size {variable.sizes[dim]}, but "
                    f"data has size {voxel_invalid.sizes[dim]}; pass the "
                    "data that belong to this dataset."
                )
        voxel_invalid = voxel_invalid.transpose(*voxel_dims).values.astype(bool)
        expand = tuple(
            slice(None) if dim in voxel_dims else np.newaxis for dim in variable.dims
        )
        order = [voxel_dims.index(dim) for dim in variable.dims if dim in voxel_dims]
        mask = np.transpose(voxel_invalid, order)[expand]
        fill = complex(np.nan, np.nan) if variable.dtype.kind == "c" else np.nan
        values = np.array(variable.values, copy=True)
        values[np.broadcast_to(mask, values.shape)] = fill
        masked[name].values = values
    return masked
