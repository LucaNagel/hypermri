"""Descriptive empirical log-logistic DCE concentration model.

Times are seconds and concentrations are mM. The model has six fitted parameters
and requires no AIF. Arrival and rise duration can trade off; their sum is
reported as the derived half-rise time. Functions are eager and deterministic,
preserve inputs and retain no caches.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import xarray as xr
from scipy.optimize import least_squares

from hypermri.analysis.dce.fit_metrics import _residual_statistics
from hypermri.analysis.dce.fit_window import select_fit_window
from hypermri.analysis.dce.sampling import _coerce_numeric_1d, _validate_dce_time_axis
from hypermri.analysis.fitting.result_metadata import select_reconstruction_array

from ..fit_quality import _dataarray_isclose, _parameters_at_bounds
from ..result_metadata import _attach_dce_reconstruction_arrays, _set_var_attrs
from ..sampling import _derive_time_points_from_concentration_map


@dataclass(frozen=True)
class EmpiricalLogLogisticFitOptions:
    """Immutable scientific overrides for the empirical log logistic curve fit.

    None leaves a value unspecified: the existing fitter remains the sole owner
    of scientific defaults and data-dependent guesses/bounds. Construction does
    not fit, read data or allocate a cache; supplied sequences become tuples.

    Parameters
    ----------
    fit_start_index : int | None, optional
        First zero-based sample index; included.
    fit_end_index : int | None, optional
        Exclusive end index. Negative ends retain this model's historical rule.
    fit_start_time_s : float | None, optional
        Inclusive start time in seconds.
    fit_end_time_s : float | None, optional
        Inclusive end time in seconds.
    fit_indices : tuple[int, ...] | None, optional
        Explicit sample indices; negative entries follow NumPy indexing.
    fit_times_s : tuple[float, ...] | None, optional
        Requested times in seconds; select the nearest sample, first on ties.
    initial_parameters : tuple[float, ...] | None, optional
        Initial parameters in the order and units described below.
    lower_bounds : tuple[float, ...] | None, optional
        Lower parameter bounds in the same order and units.
    upper_bounds : tuple[float, ...] | None, optional
        Upper parameter bounds in the same order and units.
    loss : str | None, optional
        SciPy least-squares loss name; None retains the model default.
    f_scale_mm : float | None, optional
        Positive loss transition scale in mM concentration residuals.
    max_nfev : int | None, optional
        Maximum number of least-squares function evaluations.

    Raises
    ------
    ValueError
        If supplied parameter vectors have an incompatible length.

    Notes
    -----
    Parameter order is (bolus arrival, rise time, rise shape, washout rate,
    amplitude, baseline), with units (s, s, dimensionless, 1/s, mM, mM).
    Negative ends select no samples. Existing data-dependent starts/bounds and
    the tradeoff between arrival and rise duration are retained.

    Examples
    --------
    >>> EmpiricalLogLogisticFitOptions(fit_start_time_s=5.0).to_fit_params()
    {'fit_start_time_s': 5.0}
    """

    fit_start_index: int | None = None
    fit_end_index: int | None = None
    fit_start_time_s: float | None = None
    fit_end_time_s: float | None = None
    fit_indices: tuple[int, ...] | None = None
    fit_times_s: tuple[float, ...] | None = None
    initial_parameters: tuple[float, ...] | None = None
    lower_bounds: tuple[float, ...] | None = None
    upper_bounds: tuple[float, ...] | None = None
    loss: str | None = None
    f_scale_mm: float | None = None
    max_nfev: int | None = None

    def __post_init__(self) -> None:
        """Copy sequences into immutable tuples and validate parameter counts."""
        for name in (
            "fit_indices",
            "fit_times_s",
            "initial_parameters",
            "lower_bounds",
            "upper_bounds",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, tuple(value))
        for name in ("initial_parameters", "lower_bounds", "upper_bounds"):
            value = getattr(self, name)
            if value is not None and len(value) != 6:
                raise ValueError(
                    f"{type(self).__name__}.{name} must contain exactly "
                    f"{6} parameter values."
                )

    def to_fit_params(self) -> dict[str, object]:
        """Translate supplied scientific overrides to the existing fit mapping.

        Returns
        -------
        dict[str, object]
            Fresh mapping with canonical legacy keys. None fields are omitted
            so defaults are resolved once by the fitter. Parameter vectors and
            selections are immutable tuples; units and numeric values are
            unchanged. The options object is not mutated.
        """
        values = {
            "fit_start_index": self.fit_start_index,
            "fit_end_index": self.fit_end_index,
            "fit_start_time_s": self.fit_start_time_s,
            "fit_end_time_s": self.fit_end_time_s,
            "fit_indices": self.fit_indices,
            "fit_times_s": self.fit_times_s,
            "x0": self.initial_parameters,
            "lower_bounds": self.lower_bounds,
            "upper_bounds": self.upper_bounds,
            "loss": self.loss,
            "f_scale": self.f_scale_mm,
            "max_nfev": self.max_nfev,
        }
        return {name: value for name, value in values.items() if value is not None}


_EMPIRICAL_LOG_LOGISTIC_MODEL_EQUATION = (
    "Ct(t) = baseline + amplitude * log_logistic_rise(t) * "
    "exp(-washout_rate * max(t - bolus_arrival, 0))"
)

_EMPIRICAL_LOG_LOGISTIC_OUTPUT_NAMES = [
    "bolus_arrival_s",
    "rise_time_s",
    "rise_shape",
    "washout_rate_s",
    "peak_amplitude",
    "baseline_offset",
    "time_to_peak_s",
    "half_rise_time_s",
    "rss",
    "rmse",
    "fit_success",
    "n_points_fit",
]

_EMPIRICAL_HALF_RISE_TIME_ATTRS = {
    "long_name": "half_rise_time_s",
    "description": (
        "Absolute time when the log-logistic rise term reaches 0.5, before "
        "baseline and washout; bolus_arrival_s + rise_time_s."
    ),
    "units": "s",
    "model": "empirical_log_logistic",
}


def empirical_log_logistic_model(
    time_points_s: np.ndarray,
    bolus_arrival_s: float,
    rise_time_s: float,
    rise_shape: float,
    washout_rate_s: float,
    peak_amplitude: float,
    baseline_offset: float = 0.0,
) -> np.ndarray:
    """Evaluate the descriptive empirical log-logistic DCE curve.

    Parameters
    ----------
    time_points_s : numpy.ndarray
        Evaluation times in seconds; any shape and ordering are preserved.
    bolus_arrival_s : float
        Bolus-arrival time in seconds. The rise term is zero at earlier times.
    rise_time_s : float
        Positive half-rise duration in seconds, floored at float64 tiny.
    rise_shape : float
        Dimensionless rise exponent, floored at float64 tiny.
    washout_rate_s : float
        Exponential washout rate in 1/s, clipped to at least zero.
    peak_amplitude : float
        Rise/washout amplitude in mM, not necessarily the attained curve peak.
    baseline_offset : float, default 0.0
        Additive baseline concentration in mM.

    Returns
    -------
    numpy.ndarray
        Float64 concentration in mM with the input time shape and order.
        Evaluation is eager and deterministic and preserves inputs without caches.

    Notes
    -----
    This is a package-specific descriptive model, not a mechanistic compartment
    model. It multiplies a shifted log-logistic rise by exponential washout.
    Arrival and rise duration can trade off under noise; their sum defines the
    absolute half-rise time of the rise term.

    Examples
    --------
    >>> empirical_log_logistic_model(np.array([0., 2.]), 2., 10., 4., 0.02, 0.1)
    array([0., 0.])
    """
    time_points = np.asarray(time_points_s, dtype=np.float64)
    tau = np.clip(time_points - float(bolus_arrival_s), 0.0, None)
    safe_tau = np.maximum(tau, np.finfo(np.float64).tiny)
    safe_rise_time = max(float(rise_time_s), np.finfo(np.float64).tiny)
    safe_rise_shape = max(float(rise_shape), np.finfo(np.float64).tiny)
    ratio = safe_tau / safe_rise_time
    inverse_power = np.exp(np.clip(-safe_rise_shape * np.log(ratio), -700.0, 700.0))
    rise_term = np.where(
        tau > 0.0,
        1.0 / (1.0 + inverse_power),
        0.0,
    )
    washout_term = np.exp(-max(float(washout_rate_s), 0.0) * tau)
    return float(baseline_offset) + float(peak_amplitude) * rise_term * washout_term


def _reconstruct_empirical_log_logistic_curve_scalar(
    bolus_arrival_s: float,
    rise_time_s: float,
    rise_shape: float,
    washout_rate_s: float,
    peak_amplitude: float,
    baseline_offset: float,
    *,
    time_points_s: np.ndarray,
) -> np.ndarray:
    """Scalar adapter used by xarray vectorization for empirical reconstruction."""
    return empirical_log_logistic_model(
        time_points_s,
        bolus_arrival_s,
        rise_time_s,
        rise_shape,
        washout_rate_s,
        peak_amplitude,
        baseline_offset,
    )


def _default_empirical_log_logistic_bounds(
    time_points_s: np.ndarray,
    signal_curve: np.ndarray,
    fit_params: Mapping[str, object] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return initial guesses and bounds for the empirical log-logistic fit."""
    fit_params = dict(fit_params or {})
    time_points = np.asarray(time_points_s, dtype=np.float64)
    signal_curve = np.asarray(signal_curve, dtype=np.float64)

    delta_t = float(np.median(np.diff(time_points))) if len(time_points) > 1 else 1.0
    time_span = float(max(time_points[-1] - time_points[0], delta_t))
    baseline_guess = float(
        np.nanmedian(signal_curve[: max(1, min(3, len(signal_curve)))])
    )
    peak_value = float(np.nanmax(signal_curve))
    peak_index = int(np.nanargmax(signal_curve))
    bolus_guess = float(
        max(time_points[0], time_points[max(0, peak_index - 1)] - delta_t)
    )
    rise_time_guess = float(
        max(delta_t, min(time_span, time_points[peak_index] - bolus_guess + delta_t))
    )
    amplitude_guess = float(max(peak_value - baseline_guess, 1e-4))
    washout_guess = float(1.0 / max(time_span, delta_t))

    x0 = np.asarray(
        fit_params.get(
            "x0",
            [
                bolus_guess,
                rise_time_guess,
                4.0,
                washout_guess,
                amplitude_guess,
                baseline_guess,
            ],
        ),
        dtype=np.float64,
    )
    lower_bounds = np.asarray(
        fit_params.get(
            "lower_bounds",
            [
                time_points[0],
                max(delta_t * 0.25, 1e-6),
                0.5,
                0.0,
                0.0,
                float(np.nanmin(signal_curve) - max(abs(baseline_guess), 1e-3)),
            ],
        ),
        dtype=np.float64,
    )
    upper_bounds = np.asarray(
        fit_params.get(
            "upper_bounds",
            [
                time_points[-1],
                max(time_span * 1.5, 5.0 * delta_t),
                20.0,
                max(2.0 / max(delta_t, 1e-6), 1.0),
                max(peak_value * 5.0, amplitude_guess * 10.0, 1.0),
                float(np.nanmax(signal_curve) + max(abs(peak_value), 1e-3)),
            ],
        ),
        dtype=np.float64,
    )

    if x0.shape != (6,) or lower_bounds.shape != (6,) or upper_bounds.shape != (6,):
        raise ValueError(
            "Empirical log-logistic x0 and bounds must each contain exactly six values."
        )

    return np.clip(x0, lower_bounds, upper_bounds), lower_bounds, upper_bounds


def fit_empirical_log_logistic_curve(
    tissue_concentration: np.ndarray,
    *,
    time_points_s: np.ndarray,
    fit_params: Mapping[str, object] | EmpiricalLogLogisticFitOptions | None = None,
) -> dict[str, float | bool | np.ndarray]:
    """Fit the empirical log-logistic DCE model to one concentration curve.

    Parameters
    ----------
    tissue_concentration : numpy.ndarray
        Real tissue concentration in mM, shape ``(n_time,)`` after squeezing.
    time_points_s : numpy.ndarray
        Acquisition times in seconds on the same axis.
    fit_params : Mapping[str, object] or EmpiricalLogLogisticFitOptions or None, optional
        Window selectors of ``select_fit_window`` and initial guesses/bounds
        of ``_default_empirical_log_logistic_bounds``. ``loss`` (default linear),
        ``f_scale`` (default 1) and ``max_nfev`` (default 500) configure least
        squares. This empirical model does not require an AIF.

    Returns
    -------
    dict
        Bolus arrival and rise/peak times in seconds, dimensionless rise shape,
        washout rate in 1/s, amplitude and baseline in mM. ``prediction`` is
        float64 mM on the full axis. RSS is mM², RMSE mM and R² dimensionless
        (NaN for constant observations). The boolean ``fit_window_mask`` excludes
        non-finite tissue samples; ``n_points_fit`` and ``fit_success`` report
        optimization. Fewer than six selected samples return NaN parameters and
        prediction with false success. Covariance is not estimated. Inputs are
        preserved and evaluation is eager and deterministic.
        ``half_rise_time_s`` is the absolute acquisition time in seconds,
        ``bolus_arrival_s + rise_time_s``, when the log-logistic rise term is
        0.5 before baseline and washout. It is derived after fitting, not an
        extra fitted parameter or the half-height time of the complete curve.
        It is NaN when fewer than six samples permit a fit. Its uncertainty
        is not estimated; ``fit_success`` must still be checked.

    Raises
    ------
    ValueError
        If curve/time shapes, bounds, starts or optimizer options are invalid.
    IndexError
        If an explicitly selected sample index is outside the time array.

    Notes
    -----
    This is the package-specific descriptive empirical model. Arrival time and
    rise duration can trade off strongly; their sum is the reported half-rise
    time. Reporting it does not change the six-parameter optimizer or establish
    identifiability of the individual timing parameters.

    Examples
    --------
    >>> fitted = fit_empirical_log_logistic_curve(tissue, time_points_s=times_s)  # doctest: +SKIP
    """
    if isinstance(fit_params, EmpiricalLogLogisticFitOptions):
        fit_params = fit_params.to_fit_params()
    fit_params = dict(fit_params or {})
    time_points = _coerce_numeric_1d("time_points_s", time_points_s)
    tissue_curve = _coerce_numeric_1d("tissue_concentration", tissue_concentration)

    if len(time_points) != len(tissue_curve):
        raise ValueError(
            "tissue_concentration and time_points_s must have matching lengths."
        )

    fit_mask = np.asarray(
        select_fit_window(
            time_points,
            fit_params,
            fit_type="empirical_log_logistic",
        ).values,
        dtype=bool,
    )
    fit_mask &= np.isfinite(tissue_curve)

    if int(fit_mask.sum()) < 6:
        empty_prediction = np.full_like(time_points, np.nan, dtype=np.float64)
        return {
            "bolus_arrival_s": np.nan,
            "rise_time_s": np.nan,
            "rise_shape": np.nan,
            "washout_rate_s": np.nan,
            "peak_amplitude": np.nan,
            "baseline_offset": np.nan,
            "time_to_peak_s": np.nan,
            "half_rise_time_s": np.nan,
            "rss": np.nan,
            "rmse": np.nan,
            "r_squared": np.nan,
            "fit_success": False,
            "n_points_fit": int(fit_mask.sum()),
            "prediction": empty_prediction,
            "fit_window_mask": fit_mask,
        }

    x0, lower_bounds, upper_bounds = _default_empirical_log_logistic_bounds(
        time_points,
        tissue_curve,
        fit_params,
    )

    def residuals(params: np.ndarray) -> np.ndarray:
        prediction = empirical_log_logistic_model(time_points, *params)
        return prediction[fit_mask] - tissue_curve[fit_mask]

    result = least_squares(
        residuals,
        x0=x0,
        bounds=(lower_bounds, upper_bounds),
        loss=str(fit_params.get("loss", "linear")),
        f_scale=float(fit_params.get("f_scale", 1.0)),
        max_nfev=int(fit_params.get("max_nfev", 500)),
    )

    prediction = empirical_log_logistic_model(time_points, *result.x)
    metrics = _residual_statistics(tissue_curve[fit_mask], prediction[fit_mask])
    peak_index = int(np.argmax(prediction)) if np.all(np.isfinite(prediction)) else -1
    time_to_peak_s = float(time_points[peak_index]) if peak_index >= 0 else np.nan

    return {
        "bolus_arrival_s": float(result.x[0]),
        "rise_time_s": float(result.x[1]),
        "rise_shape": float(result.x[2]),
        "washout_rate_s": float(result.x[3]),
        "peak_amplitude": float(result.x[4]),
        "baseline_offset": float(result.x[5]),
        "time_to_peak_s": time_to_peak_s,
        "half_rise_time_s": float(result.x[0] + result.x[1]),
        "rss": metrics["rss"],
        "rmse": metrics["rmse"],
        "r_squared": metrics["r_squared"],
        "fit_success": bool(result.success),
        "n_points_fit": int(fit_mask.sum()),
        "prediction": prediction,
        "fit_window_mask": fit_mask,
    }


def _fit_empirical_log_logistic_voxel(
    *,
    signal: np.ndarray,
    dynamic_coords: np.ndarray,
    fit_params: dict,
    indices: tuple,
) -> tuple[dict[str, float | bool | int], tuple]:
    """Adapter that lets the shared voxelwise executor run empirical fits."""
    time_points_s = np.asarray(
        fit_params.get("time_points_s", dynamic_coords),
        dtype=np.float64,
    )
    fit_result = fit_empirical_log_logistic_curve(
        signal,
        time_points_s=time_points_s,
        fit_params=fit_params,
    )
    return (
        {
            name: fit_result[name]
            for name in _EMPIRICAL_LOG_LOGISTIC_OUTPUT_NAMES
            if name in fit_result
        },
        indices,
    )


def empirical_log_logistic_reconstruction(
    time_points_s: np.ndarray,
    bolus_arrival_s: xr.DataArray,
    rise_time_s: xr.DataArray,
    rise_shape: xr.DataArray,
    washout_rate_s: xr.DataArray,
    peak_amplitude: xr.DataArray,
    baseline_offset: xr.DataArray,
) -> xr.DataArray:
    """Reconstruct labelled concentration curves from existing model parameters.

    Parameters
    ----------
    time_points_s : numpy.ndarray
        Finite, strictly increasing acquisition times in seconds, shape (n_time,).
    bolus_arrival_s, rise_time_s : xarray.DataArray
        Arrival and half-rise-duration maps in seconds.
    rise_shape : xarray.DataArray
        Dimensionless rise-exponent map.
    washout_rate_s : xarray.DataArray
        Nonnegative washout-rate map in 1/s.
    peak_amplitude, baseline_offset : xarray.DataArray
        Amplitude and baseline maps in mM.

    Returns
    -------
    xarray.DataArray
        Float64 predicted concentration in mM. Parameter dimensions/coordinates
        broadcast in xarray order, followed by ``repetition_idx`` with zero-based
        indices and ``time_s`` in seconds. Inputs remain unchanged; eager NumPy
        evaluation is deterministic, with xarray's existing dask behavior retained.

    Raises
    ------
    ValueError
        If the time axis is invalid or parameter coordinates cannot align.

    Examples
    --------
    >>> prediction = empirical_log_logistic_reconstruction(times_s, *parameter_maps)  # doctest: +SKIP
    """
    time_points = _validate_dce_time_axis("time_points_s", time_points_s)
    time_dim_name = "repetition_idx"
    rep_coord = np.arange(len(time_points), dtype=int)
    base_coords = {
        time_dim_name: rep_coord,
        "time_s": (time_dim_name, time_points),
    }
    dask_kwargs = {"output_sizes": {time_dim_name: len(time_points)}}

    prediction = xr.apply_ufunc(
        _reconstruct_empirical_log_logistic_curve_scalar,
        bolus_arrival_s,
        rise_time_s,
        rise_shape,
        washout_rate_s,
        peak_amplitude,
        baseline_offset,
        input_core_dims=[[], [], [], [], [], []],
        output_core_dims=[[time_dim_name]],
        vectorize=True,
        output_dtypes=[np.float64],
        dask="allowed",
        dask_gufunc_kwargs=dask_kwargs,
        kwargs={"time_points_s": time_points},
    )
    return prediction.assign_coords(**base_coords)


__all__ = [
    "EmpiricalLogLogisticFitOptions",
    "empirical_log_logistic_model",
    "fit_empirical_log_logistic_curve",
    "empirical_log_logistic_reconstruction",
]


def _annotate_model_output(dataset: xr.Dataset):
    """Attach the existing model-specific descriptions in place."""
    for var_name, units, description in (
        (
            "bolus_arrival_s",
            "s",
            "Estimated bolus arrival time of the empirical fit.",
        ),
        (
            "rise_time_s",
            "s",
            "Characteristic rise-time parameter of the log-logistic term.",
        ),
        (
            "rise_shape",
            "1",
            "Shape parameter controlling the steepness of the log-logistic rise.",
        ),
        ("washout_rate_s", "s^-1", "Mono-exponential washout rate constant."),
        (
            "peak_amplitude",
            "mM",
            "Amplitude scaling of the empirical concentration curve.",
        ),
        ("baseline_offset", "mM", "Constant offset term of the empirical model."),
        ("time_to_peak_s", "s", "Time of the reconstructed concentration peak."),
        (
            "fit_success",
            "bool",
            "Boolean flag indicating empirical fit convergence.",
        ),
        (
            "n_points_fit",
            "count",
            "Number of dynamic samples retained in the empirical fit window.",
        ),
        ("time_points_s", "s", "Time axis used for empirical fitting."),
    ):
        _set_var_attrs(
            dataset,
            var_name,
            long_name=var_name,
            description=description,
            units=units,
            model="empirical_log_logistic",
        )
    if "half_rise_time_s" in dataset:
        dataset["half_rise_time_s"].attrs.update(_EMPIRICAL_HALF_RISE_TIME_ATTRS)


def _reconstruction_parameters(
    data_to_reconstruct: xr.Dataset, kwargs: dict, voxel_coords: dict | None = None
):
    """Translate legacy reconstruction options without changing precedence."""
    dynamic_dim_override = kwargs.get("dynamic_dim")
    time_points_s = dynamic_dim_override
    if time_points_s is None:
        time_points_s = kwargs.get("time_points_s")
    if time_points_s is None:
        time_points_s = select_reconstruction_array(
            data_to_reconstruct,
            "time_points_s",
            voxel_coords,
        )
    if time_points_s is None:
        raise ValueError(
            "Empirical log-logistic reconstruction requires time_points_s via kwargs or dataset variables."
        )
    params = {
        "time_points_s": (
            time_points_s.values
            if isinstance(time_points_s, xr.DataArray)
            else np.asarray(time_points_s)
        ),
        "bolus_arrival_s": data_to_reconstruct["bolus_arrival_s"],
        "rise_time_s": data_to_reconstruct["rise_time_s"],
        "rise_shape": data_to_reconstruct["rise_shape"],
        "washout_rate_s": data_to_reconstruct["washout_rate_s"],
        "peak_amplitude": data_to_reconstruct["peak_amplitude"],
        "baseline_offset": data_to_reconstruct["baseline_offset"],
    }
    return params


def _reconstruct_empirical_log_logistic_prediction(
    fit_results: xr.Dataset,
    concentration_map: xr.DataArray,
    *,
    time_points_s: np.ndarray,
    fit_params: Mapping[str, object] | None = None,
) -> tuple[xr.DataArray, int, xr.DataArray, xr.DataArray]:
    rep_coord = concentration_map.coords.get(
        "repetition_idx", np.arange(len(time_points_s), dtype=int)
    )
    dask_kwargs = {"output_sizes": {"repetition_idx": len(time_points_s)}}

    prediction = xr.apply_ufunc(
        _reconstruct_empirical_log_logistic_curve_scalar,
        fit_results["bolus_arrival_s"],
        fit_results["rise_time_s"],
        fit_results["rise_shape"],
        fit_results["washout_rate_s"],
        fit_results["peak_amplitude"],
        fit_results["baseline_offset"],
        input_core_dims=[[], [], [], [], [], []],
        output_core_dims=[["repetition_idx"]],
        vectorize=True,
        output_dtypes=[np.float64],
        dask="allowed",
        dask_gufunc_kwargs=dask_kwargs,
        kwargs={"time_points_s": np.asarray(time_points_s, dtype=np.float64)},
    ).assign_coords(repetition_idx=rep_coord)

    if "time_s" in concentration_map.coords:
        prediction = prediction.assign_coords(time_s=concentration_map.coords["time_s"])

    parameter_count = 6
    finite_params = (
        np.isfinite(fit_results["bolus_arrival_s"])
        & np.isfinite(fit_results["rise_time_s"])
        & np.isfinite(fit_results["rise_shape"])
        & np.isfinite(fit_results["washout_rate_s"])
        & np.isfinite(fit_results["peak_amplitude"])
        & np.isfinite(fit_results["baseline_offset"])
    )

    success_flag = fit_results.get("fit_success")
    if success_flag is not None:
        finite_params = finite_params & success_flag.astype(bool)

    fit_params = dict(fit_params or {})
    time_points = np.asarray(time_points_s, dtype=np.float64)
    delta_t = float(np.median(np.diff(time_points))) if len(time_points) > 1 else 1.0
    time_span = float(max(time_points[-1] - time_points[0], delta_t))
    lower_bounds = np.asarray(
        fit_params.get(
            "lower_bounds",
            [time_points[0], max(delta_t * 0.25, 1e-6), 0.5, 0.0, 0.0, -np.inf],
        ),
        dtype=np.float64,
    )
    upper_bounds = np.asarray(
        fit_params.get(
            "upper_bounds",
            [
                time_points[-1],
                max(time_span * 1.5, 5.0 * delta_t),
                20.0,
                max(2.0 / max(delta_t, 1e-6), 1.0),
                np.inf,
                np.inf,
            ],
        ),
        dtype=np.float64,
    )
    tol = 1e-6
    hit_bound = (
        _dataarray_isclose(
            fit_results["bolus_arrival_s"], float(lower_bounds[0]), atol=tol
        )
        | _dataarray_isclose(
            fit_results["bolus_arrival_s"], float(upper_bounds[0]), atol=tol
        )
        | _dataarray_isclose(
            fit_results["rise_time_s"], float(lower_bounds[1]), atol=tol
        )
        | _dataarray_isclose(
            fit_results["rise_time_s"], float(upper_bounds[1]), atol=tol
        )
        | _dataarray_isclose(
            fit_results["rise_shape"], float(lower_bounds[2]), atol=tol
        )
        | _dataarray_isclose(
            fit_results["rise_shape"], float(upper_bounds[2]), atol=tol
        )
        | _dataarray_isclose(
            fit_results["washout_rate_s"], float(lower_bounds[3]), atol=tol
        )
        | _dataarray_isclose(
            fit_results["washout_rate_s"], float(upper_bounds[3]), atol=tol
        )
        | _dataarray_isclose(
            fit_results["peak_amplitude"], float(lower_bounds[4]), atol=tol
        )
        | _dataarray_isclose(
            fit_results["peak_amplitude"], float(upper_bounds[4]), atol=tol
        )
        | _dataarray_isclose(
            fit_results["baseline_offset"], float(lower_bounds[5]), atol=tol
        )
        | _dataarray_isclose(
            fit_results["baseline_offset"], float(upper_bounds[5]), atol=tol
        )
    )
    return prediction, parameter_count, finite_params, hit_bound


def _quality_inputs(
    fit_results: xr.Dataset,
    concentration_map: xr.DataArray,
    *,
    aif_concentration: np.ndarray,
    aif_time_s: np.ndarray,
    time_points_for_fit: np.ndarray,
    fit_params: dict,
    rep_coord,
):
    """Prepare unchanged predictions, parameter counts and quality flags."""
    (
        prediction,
        parameter_count,
        finite_params,
        hit_bound,
    ) = _reconstruct_empirical_log_logistic_prediction(
        fit_results,
        concentration_map,
        time_points_s=time_points_for_fit,
        fit_params=fit_params,
    )
    selected_tofts_model = None
    return prediction, parameter_count, finite_params, hit_bound, selected_tofts_model


RESULT_MODEL_NAME = "empirical_log_logistic"
_reconstruct = empirical_log_logistic_reconstruction
_fit_voxel = _fit_empirical_log_logistic_voxel
_OUTPUT_DTYPES = {"fit_success": bool, "n_points_fit": np.int64}
_DEFAULT_MODEL_MODE = "empirical_log_logistic"
_quality_time_points = _derive_time_points_from_concentration_map


def _output_param_names(params):
    return _EMPIRICAL_LOG_LOGISTIC_OUTPUT_NAMES


def _quality_window(times, plasma, params):
    return select_fit_window(times, params, fit_type="empirical_log_logistic").values


_BACKEND_DESCRIPTION = "direct concentration-space empirical fit"
