"""Monoexponential magnitude T2 model for quantitative echo images.

Unweighted nonlinear least squares fits signal space without an offset. Every
finite echo magnitude above threshold participates; at least three valid echoes
and two distinct echo times are required. Calculation uses float64 and seconds.
T2 is bounded above at 5 seconds. Only the returned T2 map converts to ms.
No fit uncertainty is estimated.
"""

from __future__ import annotations

from numbers import Integral, Real

import numpy as np
import xarray as xr
from scipy.optimize import least_squares

from hypermri.analysis.fitting.execution import copy_execution_params

MODEL_ID = "t2"
RESULT_MODEL_NAME = "monoexponential_t2"
MODEL_EQUATION = "s0 * exp(-echo_time_s / (t2 / 1000))"
OUTPUT_PARAM_NAMES = ["t2", "s0", "rss_t2"]
_MILLISECONDS_PER_SECOND = 1000.0
_DEFAULT_SIGNAL_THRESHOLD = 1e-5
_MINIMUM_VALID_ECHOES = 3
_MAXIMUM_T2_S = 5.0


def _validate_options(params):
    allowed = {
        "signal_threshold",
        "dynamic_dim",
        "use_multiprocessing",
        "number_of_cpu_cores",
        "fit_type",
        "model_id",
        "model_name",
    }
    unknown = set(params) - allowed
    if unknown:
        raise ValueError(f"Unknown T2 fit options: {sorted(unknown)}.")
    threshold = params.get("signal_threshold", _DEFAULT_SIGNAL_THRESHOLD)
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, Real)
        or not np.isfinite(threshold)
        or threshold < 0
    ):
        raise ValueError("T2 signal_threshold must be finite and nonnegative.")
    dim = params.get("dynamic_dim", "echo_idx")
    if not isinstance(dim, str) or not dim:
        raise TypeError("T2 dynamic_dim must be a nonempty dimension name.")
    if "use_multiprocessing" in params and not isinstance(
        params["use_multiprocessing"], bool
    ):
        raise TypeError("use_multiprocessing must be a boolean.")
    workers = params.get("number_of_cpu_cores", -1)
    if isinstance(workers, bool) or not isinstance(workers, Integral) or workers == 0:
        raise ValueError("number_of_cpu_cores must be a nonzero integer.")


def _echo_times(data, dim):
    if dim not in data.dims or dim not in data.coords:
        raise ValueError(
            f"T2 fitting requires the {dim!r} echo dimension and coordinate in seconds."
        )
    coord = data.coords[dim]
    if coord.dims != (dim,) or coord.attrs.get("unit", coord.attrs.get("units")) != "s":
        raise ValueError(
            f"T2 coordinate {dim!r} must be one-dimensional with unit='s'."
        )
    times = np.asarray(coord.values, dtype=float)
    if (
        times.size < _MINIMUM_VALID_ECHOES
        or not np.all(np.isfinite(times))
        or np.any(times < 0)
        or np.unique(times).size < 2
    ):
        raise ValueError(
            "T2 fitting requires at least three finite nonnegative echo times in seconds and two distinct times."
        )
    return times


def _validate_prepared(data, params):
    _validate_options(params)
    if not {"dynamic_dim", "signal_threshold"}.issubset(params):
        raise ValueError(
            "Prepared T2 settings require dynamic_dim and signal_threshold; use prepare_fit_params(data, 't2', options)."
        )
    _echo_times(data, params["dynamic_dim"])


def prepare_t2_fit_params_dataset(
    data: xr.DataArray,
    fit_params: dict | xr.Dataset | None = None,
    return_params_info: bool = False,
) -> xr.Dataset | None:
    """Validate T2 sampling and prepare the effective scientific settings.

    Parameters
    ----------
    data : xarray.DataArray
        Real/complex echo signal; selected dynamic coordinate must declare
        seconds, contain at least three finite nonnegative times and at least
        two distinct times. Remaining dimensions identify independent voxels.
    fit_params : dict or xarray.Dataset or None, optional
        Signal threshold, dynamic_dim and shared worker settings. None retains
        model defaults (threshold 1e-5, echo_idx). The model's fixed T2 upper
        bound is 5 s. Prepared attributes are copied.
        Unknown settings fail explicitly. No input or option mutation occurs.
    return_params_info : bool, optional
        Return None after validation instead of preparing a Dataset.

    Returns
    -------
    xarray.Dataset or None
        Attributes with model identifiers and effective settings. No acquired
        samples are read or fitted; coordinates are inspected eagerly, with no
        cache. All echo times and numerical calculations remain in seconds.

    Raises
    ------
    TypeError, ValueError
        Invalid options, coordinate units or incompatible echo sampling.
    """
    params = dict(
        fit_params.attrs if isinstance(fit_params, xr.Dataset) else fit_params or {}
    )
    _validate_options(params)
    dim = params.get("dynamic_dim", "echo_idx")
    _echo_times(data, dim)
    if return_params_info:
        return None
    attrs = {
        "fit_type": MODEL_ID,
        "model_id": MODEL_ID,
        "model_name": RESULT_MODEL_NAME,
        "dynamic_dim": dim,
        "signal_threshold": params.get("signal_threshold", _DEFAULT_SIGNAL_THRESHOLD),
    }
    copy_execution_params(params, attrs)
    return xr.Dataset(attrs=attrs)


def _fit_decay(signal, times_s, threshold):
    """Fit float64 seconds; normalize signal scale only for solver conditioning."""
    magnitude = np.abs(np.asarray(signal)).astype(np.float64)
    valid = np.isfinite(magnitude) & (magnitude > threshold)
    failed = dict.fromkeys(OUTPUT_PARAM_NAMES, np.nan)
    times = np.asarray(times_s, dtype=np.float64)[valid]
    if times.size < _MINIMUM_VALID_ECHOES or np.unique(times).size < 2:
        return failed
    values = magnitude[valid]
    scale = np.max(values)
    normalized = values / scale
    # Log regression supplies initialization only; the objective is signal RSS.
    slope = np.linalg.lstsq(
        np.column_stack((np.ones(times.size), times)), np.log(normalized), rcond=None
    )[0][1]
    fallback_t2 = min(max(np.ptp(times), np.max(times), 0.001), _MAXIMUM_T2_S)
    initial_t2 = min(-1.0 / slope, _MAXIMUM_T2_S) if slope < 0 else fallback_t2
    decay = np.exp(-times / initial_t2)
    if np.sum(decay**2) == 0:
        initial_t2 = fallback_t2
        decay = np.exp(-times / initial_t2)
    initial_s0 = np.dot(decay, normalized) / np.dot(decay, decay)
    try:
        result = least_squares(
            lambda params: params[0] * np.exp(-times / params[1]) - normalized,
            [initial_s0, initial_t2],
            bounds=([0.0, np.finfo(float).tiny], [np.inf, _MAXIMUM_T2_S]),
            x_scale="jac",
            ftol=1e-10,
            xtol=1e-10,
            gtol=1e-10,
            max_nfev=2000,
        )
    except (ValueError, FloatingPointError, np.linalg.LinAlgError):
        return failed
    s0, t2_s = result.x
    if not result.success or not np.all(np.isfinite(result.x)) or t2_s <= 0:
        return failed
    s0 *= scale
    residuals = s0 * np.exp(-times / t2_s) - values
    return {
        "t2": t2_s * _MILLISECONDS_PER_SECOND,
        "s0": s0,
        "rss_t2": float(np.sum(residuals**2)),
    }


def _attach_metadata(result, data, params):
    dim = params["dynamic_dim"]
    result["echo_times_s"] = data.coords[dim].copy(deep=True)
    result["echo_times_s"].attrs.update(unit="s", units="s")
    result["t2"].attrs.update(
        unit="ms", units="ms", long_name="transverse relaxation time"
    )
    signal_unit = data.attrs.get("unit", data.attrs.get("units", "input signal units"))
    result["s0"].attrs["unit"] = signal_unit
    result["rss_t2"].attrs.update(
        unit=f"({signal_unit})^2",
        definition="sum of squared magnitude signal residuals over included echoes",
    )
    result.attrs.update(
        t2_output_units="ms",
        echo_time_units="s",
        fit_uncertainty="not calculated",
        minimum_valid_echoes=_MINIMUM_VALID_ECHOES,
        maximum_t2_s=_MAXIMUM_T2_S,
        fit_objective="unweighted magnitude signal RSS; no offset",
    )


def monoexponential_t2_model(
    echo_times_s: xr.DataArray | np.ndarray,
    s0: xr.DataArray,
    t2: xr.DataArray,
    *,
    dynamic_dim: str = "echo_idx",
) -> xr.DataArray:
    """Reconstruct magnitude echo signals from S0 and a T2 map in milliseconds.

    Parameters
    ----------
    echo_times_s : xarray.DataArray or numpy.ndarray
        One-dimensional finite nonnegative echo times in seconds, in acquisition
        order. Labelled arrays retain coordinates; arrays use dynamic_dim.
    s0 : xarray.DataArray
        Initial magnitude in input signal units, with arbitrary voxel dimensions.
    t2 : xarray.DataArray
        Transverse relaxation time in ms, with coordinates compatible with s0.
        Invalid fits may contain NaN and produce NaN reconstructed signals.
    dynamic_dim : str, optional
        Dimension name for an unlabelled echo array; default echo_idx.

    Returns
    -------
    xarray.DataArray
        Float64 predicted magnitudes in signal units, with voxel dimensions
        followed by the echo dimension. Inputs, order and coordinates are kept.
        Calculation converts T2 ms to seconds centrally, eagerly, without cache.

    Raises
    ------
    ValueError
        Echo times are not a finite nonnegative one-dimensional array.

    Examples
    --------
    >>> prediction = monoexponential_t2_model(times_s, result.s0, result.t2)  # doctest: +SKIP
    """
    times = np.asarray(echo_times_s, dtype=float)
    if times.ndim != 1 or not np.all(np.isfinite(times)) or np.any(times < 0):
        raise ValueError(
            "T2 reconstruction requires finite nonnegative echo times in seconds."
        )
    axis = (
        echo_times_s
        if isinstance(echo_times_s, xr.DataArray)
        else xr.DataArray(times, dims=[dynamic_dim], coords={dynamic_dim: times})
    )
    return s0.astype(float) * np.exp(
        -axis / (t2.astype(float) / _MILLISECONDS_PER_SECOND)
    )


def _describe_params():
    return {
        "required_fit_params": {},
        "optional_fit_params": {
            "signal_threshold": "Minimum included magnitude in signal units (default 1e-5).",
            "dynamic_dim": "Echo coordinate in seconds (default echo_idx).",
        },
        "output_parameters": {
            "t2": "T2 in ms, bounded above at 5000 ms (5 s)",
            "s0": "initial magnitude in signal units",
            "rss_t2": "unweighted signal RSS",
        },
        "model_equation": MODEL_EQUATION,
    }
