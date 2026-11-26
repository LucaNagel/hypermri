"""Sourbron renal concentration model, curve fitting and scientific options.

Times are seconds and concentrations are mM. The reduced model and optional
finite plasma-transit extension use their respective documented quadrature.
Finite-Tp calculations integrate both stages on one internally oversampled
linearly interpolated AIF before returning acquisition samples. Scientific
starts/bounds, result units and precision retain their existing contracts.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import xarray as xr
from scipy.optimize import least_squares
from scipy.signal import fftconvolve

from hypermri.analysis.dce.fit_metrics import _residual_statistics
from hypermri.analysis.dce.fit_window import _select_fit_samples
from hypermri.analysis.fitting.execution import copy_execution_params
from hypermri.analysis.fitting.models.quantitative.convolution import (
    trapezoidal_convolution,
    trapezoidal_convolution_weights_s,
)
from hypermri.analysis.fitting.result_metadata import select_reconstruction_array

from ..fit_quality import _dataarray_isclose, _parameters_at_bounds
from ..result_metadata import _attach_dce_reconstruction_arrays, _set_var_attrs
from .extended_compartment import _extended_compartment_concentration


@dataclass(frozen=True)
class SourbronFitOptions:
    """Immutable scientific overrides for the sourbron curve fit.

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
    aif_threshold_mm : float | None, optional
        Use plasma samples strictly above this mM threshold.
    estimate_tp : bool | None, optional
        Estimate the finite plasma-transit time Tp as a fourth parameter.
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
    Parameter order is (vp, Ft, Tt), with units (fraction, 1/s, s), plus Tp in
    seconds when estimate_tp=True. Negative ends select no samples; explicit
    requested times are unsupported. Existing sampling-dependent guesses/bounds
    are retained. The finite-Tp variant integrates both stages internally with
    a maximum 0.1 s step; this does not change the measured acquisition times.

    Examples
    --------
    >>> SourbronFitOptions(fit_start_time_s=5.0).to_fit_params()
    {'fit_start_time_s': 5.0}
    """

    fit_start_index: int | None = None
    fit_end_index: int | None = None
    fit_start_time_s: float | None = None
    fit_end_time_s: float | None = None
    fit_indices: tuple[int, ...] | None = None
    aif_threshold_mm: float | None = None
    estimate_tp: bool | None = None
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
            "initial_parameters",
            "lower_bounds",
            "upper_bounds",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, tuple(value))
        for name in ("initial_parameters", "lower_bounds", "upper_bounds"):
            value = getattr(self, name)
            if value is not None and len(value) != (4 if self.estimate_tp else 3):
                raise ValueError(
                    f"{type(self).__name__}.{name} must contain exactly "
                    f"{4 if self.estimate_tp else 3} parameter values."
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
            "aif_threshold": self.aif_threshold_mm,
            "estimate_tp": self.estimate_tp,
            "x0": self.initial_parameters,
            "lower_bounds": self.lower_bounds,
            "upper_bounds": self.upper_bounds,
            "loss": self.loss,
            "f_scale": self.f_scale_mm,
            "max_nfev": self.max_nfev,
        }
        return {name: value for name, value in values.items() if value is not None}


MODEL_ID = "sourbron"
RESULT_MODEL_NAME = "sourbron_separable_compartment"
_PER_S_TO_ML_MIN_PER_100ML = 6000.0
_FINITE_TRANSIT_MAX_STEP_S = 0.1


def _coerce_numeric_1d(name: str, values: object) -> np.ndarray:
    """Return a squeezed 1D float64 array or raise on incompatible input."""
    arr = np.asarray(values, dtype=np.float64).squeeze()
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D after squeezing. Got shape {arr.shape}.")
    return arr


def _validate_time_axis(time_s: np.ndarray) -> np.ndarray:
    """Validate a strictly increasing, finite time axis."""
    time_s = _coerce_numeric_1d("time_s", time_s)
    if time_s.size < 3:
        raise ValueError("time_s must contain at least three samples.")
    if not np.all(np.isfinite(time_s)):
        raise ValueError("time_s must contain only finite values.")
    if np.any(np.diff(time_s) <= 0):
        raise ValueError("time_s must be strictly increasing.")
    return time_s


def _exponential_convolution(
    aif_concentration: np.ndarray,
    time_s: np.ndarray,
    transit_time_s: float,
) -> np.ndarray:
    """Convolve an AIF with ``exp(-t / T)`` by the trapezoidal rule."""
    if transit_time_s <= 0:
        raise ValueError("transit_time_s must be positive.")

    aif = _coerce_numeric_1d("aif_concentration", aif_concentration)
    time_s = _validate_time_axis(time_s)
    if len(aif) != len(time_s):
        raise ValueError("aif_concentration and time_s must have the same length.")
    return trapezoidal_convolution(
        aif, time_s, lambda lag_s: np.exp(-lag_s / float(transit_time_s))
    )


def _normalized_exponential_convolution(
    aif_concentration: np.ndarray,
    time_s: np.ndarray,
    transit_time_s: float,
) -> np.ndarray:
    """Convolve with a normalized exponential kernel ``exp(-t/T) / T``."""
    safe_transit_time_s = float(transit_time_s)
    if safe_transit_time_s <= 0:
        raise ValueError("transit_time_s must be positive.")
    return (
        _exponential_convolution(aif_concentration, time_s, safe_transit_time_s)
        / safe_transit_time_s
    )


def _prepare_finite_transit_input(time_s, aif):
    """Interpolate the AIF once onto a uniform grid with step at most 0.1 s.

    Both states start at zero at the first supplied sample. The original
    samples and absolute time origin are preserved; no earlier AIF is inferred.
    """
    sample_lags_s = time_s - time_s[0]
    n_intervals = int(np.ceil(sample_lags_s[-1] / _FINITE_TRANSIT_MAX_STEP_S))
    integration_lags_s = np.linspace(0.0, sample_lags_s[-1], n_intervals + 1)
    step_s = sample_lags_s[-1] / n_intervals
    integration_aif = np.interp(integration_lags_s, sample_lags_s, aif)
    return sample_lags_s, integration_lags_s, integration_aif, step_s


def _finite_transit_concentration(
    sample_lags_s, integration_lags_s, integration_aif, step_s, vp, ft_s, tt_s, tp_s
):
    """Integrate plasma and tubular states before sampling tissue once.

    Inputs are local float64 arrays, transit times are seconds and flow is
    1/s. FFT convolution is linear (zero padded) with trapezoidal endpoint
    correction; neither the AIF nor a parameter map is mutated or cached.
    """
    if tt_s <= 0 or tp_s <= 0:
        raise ValueError("transit_time_s must be positive.")
    if any(np.isnan(value) for value in (vp, ft_s, tt_s, tp_s)):
        return np.full_like(sample_lags_s, np.nan, dtype=np.float64)

    def convolve(signal, kernel):
        result = fftconvolve(signal, kernel)[: integration_lags_s.size] * step_s
        result -= 0.5 * step_s * (signal[0] * kernel + signal * kernel[0])
        result[0] = 0.0
        return result

    plasma = convolve(integration_aif, np.exp(-integration_lags_s / tp_s) / tp_s)
    tubular = convolve(plasma, np.exp(-integration_lags_s / tt_s))
    return np.interp(sample_lags_s, integration_lags_s, vp * plasma + ft_s * tubular)


def sourbron_separable_compartment_model(
    time_s: np.ndarray,
    aif_concentration: np.ndarray,
    *,
    plasma_volume_fraction: float | xr.DataArray,
    tubular_flow_s: float | xr.DataArray,
    tubular_transit_time_s: float | xr.DataArray,
    plasma_transit_time_s: float | xr.DataArray | None = None,
) -> np.ndarray | xr.DataArray:
    """
    Evaluate the Sourbron separable renal compartment model.

    With a precomputed tissue-plasma input ``Cp(t)``, the reduced model is

    ``Ct(t) = vp * Cp(t) + Ft * integral(Cp(u) * exp(-(t-u)/Tt) du)``

    where ``vp`` is plasma volume fraction, ``Ft`` is the tubular / outflow
    term in ``s^-1``, and ``Tt`` is tubular transit time in seconds.

    When ``plasma_transit_time_s`` is provided, the supplied AIF is interpreted
    as the arterial plasma concentration ``Ca(t)`` and the full paper-style
    separable model is evaluated via

    ``Cp(t) = (1 / Tp) * integral(Ca(u) * exp(-(t-u)/Tp) du)``

    followed by the same tissue equation above. This adds the optional tissue
    plasma transit-time parameter ``Tp`` while keeping the reduced model as the
    default behavior.

    The reduced integral uses the acquisition-grid trapezoidal rule of
    :func:`~hypermri.analysis.fitting.models.quantitative.convolution.trapezoidal_convolution`,
    which is valid for non-uniform sampling. With ``Tp``, the sampled AIF is
    linearly interpolated once onto a uniform internal grid with spacing at
    most 0.1 s. Both convolutions use that grid with trapezoidal endpoint
    correction; only the final tissue curve is interpolated back to acquisition
    samples. Both states start at zero at the first sample, without extrapolating
    a pre-acquisition AIF. The integration resolution is a numerical policy, not
    a physical lower bound on Tp or a guarantee of noisy parameter recovery.

    Parameters
    ----------
    time_s : numpy.ndarray
        Sample times in s, at least three, finite and strictly increasing.
        The spacing may be non-uniform.
    aif_concentration : numpy.ndarray
        Arterial plasma concentration in mM at ``time_s``, shape
        ``(n_time,)``.
    plasma_volume_fraction : float or xarray.DataArray
        Plasma volume fraction ``vp``, dimensionless.
    tubular_flow_s : float or xarray.DataArray
        Tubular flow ``Ft`` in 1/s.
    tubular_transit_time_s : float or xarray.DataArray
        Tubular transit time ``Tt`` in s, positive.
    plasma_transit_time_s : float or xarray.DataArray, optional
        Plasma transit time ``Tp`` in s, positive.  Omit for the reduced
        model.

    Returns
    -------
    numpy.ndarray or xarray.DataArray
        Tissue concentration in mM.  For scalar parameters a float64 array
        of shape ``(n_time,)``.  When any parameter is a ``DataArray``, a
        ``DataArray`` with the parameter dimensions and ``repetition_idx``,
        whose coordinate holds ``time_s``; it is evaluated eagerly for all
        voxels. Finite-Tp parameter maps use the same scalar integration per voxel.
        Calculation is float64 and preserves inputs/coordinate
        orientation. Fixed inputs produce deterministic results without caching.

    Raises
    ------
    ValueError
        If the time axis is invalid, the AIF length differs from it, or a
        transit time is not positive. NaN parameters propagate NaN predictions.

    Notes
    -----
    The reduced model shares its equation with extended Tofts while preserving
    the native ``exp(-lag/Tt)`` arithmetic. Model-specific guesses, bounds and
    optimizers are retained. The finite-Tp convolution remains a separate path.

    Examples
    --------
    >>> times = np.array([0.0, 1.0, 2.0])
    >>> sourbron_separable_compartment_model(
    ...     times, np.ones_like(times), plasma_volume_fraction=0.25,
    ...     tubular_flow_s=0.03, tubular_transit_time_s=10.0
    ... ).shape
    (3,)
    """
    time_s = _validate_time_axis(time_s)
    aif = _coerce_numeric_1d("aif_concentration", aif_concentration)
    if len(aif) != len(time_s):
        raise ValueError("aif_concentration and time_s must have matching lengths.")

    if plasma_transit_time_s is None:
        labelled = any(
            isinstance(param, xr.DataArray)
            for param in (
                plasma_volume_fraction,
                tubular_flow_s,
                tubular_transit_time_s,
            )
        )
        if labelled:
            if tubular_transit_time_s is None:
                raise ValueError("tubular_transit_time_s is required.")
            plasma = xr.DataArray(
                aif, dims=["repetition_idx"], coords={"repetition_idx": time_s}
            )
            volume, flow, transit = (
                plasma_volume_fraction,
                tubular_flow_s,
                tubular_transit_time_s,
            )
        else:
            plasma = aif
            volume, flow, transit = (
                float(plasma_volume_fraction),
                float(tubular_flow_s),
                float(tubular_transit_time_s),
            )
            if transit <= 0:
                raise ValueError("transit_time_s must be positive.")
        return _extended_compartment_concentration(
            time_s,
            plasma,
            plasma_volume_fraction=volume,
            transfer_rate_s=flow,
            retention_kernel=lambda lag_s: np.exp(-lag_s / transit),
        )

    prepared = _prepare_finite_transit_input(time_s, aif)
    parameters = (
        plasma_volume_fraction,
        tubular_flow_s,
        tubular_transit_time_s,
        plasma_transit_time_s,
    )

    def evaluate(vp, ft_s, tt_s, tp_s):
        return _finite_transit_concentration(*prepared, vp, ft_s, tt_s, tp_s)

    if any(isinstance(param, xr.DataArray) for param in parameters):
        prediction = xr.apply_ufunc(
            evaluate,
            *parameters,
            input_core_dims=[[], [], [], []],
            output_core_dims=[["repetition_idx"]],
            vectorize=True,
            join="inner",
            output_dtypes=[np.float64],
        )
        # Retain the existing vascular-then-tubular broadcast dimension order.
        dimensions = list(getattr(plasma_volume_fraction, "dims", ()))
        dimensions.append("repetition_idx")
        for parameter in (
            plasma_transit_time_s,
            tubular_flow_s,
            tubular_transit_time_s,
        ):
            dimensions.extend(
                dim for dim in getattr(parameter, "dims", ()) if dim not in dimensions
            )
        return (
            prediction.assign_coords(repetition_idx=time_s)
            .transpose(*dimensions)
            .rename(None)
        )

    return evaluate(*(float(parameter) for parameter in parameters))


def _build_sourbron_fit_mask(
    time_s: np.ndarray,
    *,
    fit_params: Mapping[str, object] | None = None,
    aif_concentration: np.ndarray | None = None,
) -> np.ndarray:
    """Build the temporal mask for Sourbron fitting."""
    fit_params = dict(fit_params or {})
    fit_start_index = int(fit_params.get("fit_start_index", 0) or 0)
    fit_end_index = fit_params.get(
        "fit_end_index",
        fit_params.get("fit_stop_index", None),
    )
    fit_start_time_s = float(fit_params.get("fit_start_time_s", -np.inf))
    fit_end_time_s = float(
        fit_params.get("fit_end_time_s", fit_params.get("fit_stop_time_s", np.inf))
    )

    mask = _select_fit_samples(
        time_s,
        start_index=fit_start_index,
        end_index=None if fit_end_index is None else int(fit_end_index),
        start_time_s=fit_start_time_s,
        end_time_s=fit_end_time_s,
        fit_indices=fit_params.get("fit_indices"),
    )

    if aif_concentration is not None:
        aif_threshold = float(fit_params.get("aif_threshold", 0.0))
        mask &= np.asarray(aif_concentration, dtype=np.float64) > aif_threshold

    return mask


def _default_sourbron_bounds(
    time_s: np.ndarray,
    fit_params: Mapping[str, object] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return initial guesses and bounds for the Sourbron fit."""
    fit_params = dict(fit_params or {})
    time_s = _validate_time_axis(time_s)
    delta_t = float(np.median(np.diff(time_s))) if len(time_s) > 1 else 1.0
    time_span = float(time_s[-1] - time_s[0])
    estimate_tp = bool(fit_params.get("estimate_tp", False))

    if estimate_tp:
        x0 = np.asarray(
            fit_params.get(
                "x0",
                [
                    0.10,
                    0.01,
                    max(5.0 * delta_t, min(60.0, max(time_span / 4.0, delta_t))),
                    max(delta_t, min(max(time_span / 10.0, delta_t), 20.0)),
                ],
            ),
            dtype=np.float64,
        )
        lower_bounds = np.asarray(
            fit_params.get(
                "lower_bounds",
                [0.0, 0.0, max(delta_t * 0.5, 1e-3), max(delta_t * 0.25, 1e-3)],
            ),
            dtype=np.float64,
        )
        upper_bounds = np.asarray(
            fit_params.get(
                "upper_bounds",
                [
                    1.0,
                    0.5,
                    max(time_span * 3.0, 10.0 * delta_t),
                    max(time_span, 5.0 * delta_t),
                ],
            ),
            dtype=np.float64,
        )
        if x0.shape != (4,) or lower_bounds.shape != (4,) or upper_bounds.shape != (4,):
            raise ValueError(
                "With estimate_tp=True, x0 and bounds must each contain exactly four values."
            )
        return np.clip(x0, lower_bounds, upper_bounds), lower_bounds, upper_bounds

    x0 = np.asarray(
        fit_params.get(
            "x0",
            [
                0.10,
                0.01,
                max(5.0 * delta_t, min(60.0, max(time_span / 4.0, delta_t))),
            ],
        ),
        dtype=np.float64,
    )
    lower_bounds = np.asarray(
        fit_params.get("lower_bounds", [0.0, 0.0, max(delta_t * 0.5, 1e-3)]),
        dtype=np.float64,
    )
    upper_bounds = np.asarray(
        fit_params.get(
            "upper_bounds", [1.0, 0.5, max(time_span * 3.0, 10.0 * delta_t)]
        ),
        dtype=np.float64,
    )
    if x0.shape != (3,) or lower_bounds.shape != (3,) or upper_bounds.shape != (3,):
        raise ValueError("x0 and bounds must each contain exactly three values.")
    return np.clip(x0, lower_bounds, upper_bounds), lower_bounds, upper_bounds


def get_sourbron_params_info() -> dict[str, object]:
    """Return descriptions of the existing Sourbron inputs and outputs.

    Returns
    -------
    dict[str, object]
        Fresh mappings for required/optional fit parameters, output variables and
        model equation. Time is seconds, concentration is mM, Ft is 1/s and vp a
        fraction. The optional Tp extension uses the shared internally
        oversampled double convolution, without changing scientific defaults.
        No fitting, data reads or caching occur.
    """
    return {
        "required_fit_params": {
            "aif_signal": "Arterial input function concentration values",
            "time_points_s": "Time points in seconds corresponding to the dynamic acquisition",
        },
        "optional_fit_params": {
            "fit_start_index": "First repetition index to include (default 0)",
            "fit_end_index": "Index after the last repetition to include (default None)",
            "fit_start_time_s": "Start time in seconds for the window (default -inf)",
            "fit_end_time_s": "End time in seconds for the window (default inf)",
            "fit_indices": "List of specific repetition indices to include",
            "aif_threshold": "Minimum AIF concentration retained for fitting (default 0.0)",
            "x0": "Initial guess [vp, Ft, Tt]",
            "lower_bounds": "Lower bounds for [vp, Ft, Tt]",
            "upper_bounds": "Upper bounds for [vp, Ft, Tt]",
            "use_multiprocessing": "Enable shared voxelwise parallel processing. Default: True",
            "number_of_cpu_cores": "Number of joblib workers for voxelwise processing (-1 uses all). Default: -1",
        },
        "output_parameters": {
            "vp": "Plasma volume fraction",
            "Ft": "Tubular flow / filtration parameter in s^-1",
            "Tt": "Tubular transit time in seconds",
            "tubular_flow_s": "Alias of Ft",
            "tubular_flow_ml_min_100ml": "Tubular flow converted to mL/min/100mL",
            "tubular_transit_time_s": "Alias of Tt",
            "rss": "Residual sum of squares on fitted samples",
            "rmse": "Root mean square fitting error on fitted samples",
            "r_squared": "Coefficient of determination on fitted samples",
            "fit_success": "Optimizer convergence flag",
            "n_points_fit": "Number of temporal samples used in the fit",
        },
        "model_equation": "Ct(t) = vp * Cp(t) + Ft * integral(Cp(u) * exp(-(t-u)/Tt) du)",
    }


def prepare_sourbron_fit_params_dataset(
    data: xr.DataArray,
    fit_params: dict | None = None,
    return_params_info: bool = False,
    *,
    base_attrs: dict[str, object] | None = None,
) -> xr.Dataset | None:
    """Prepare the existing Sourbron fit-parameter dataset.

    Parameters
    ----------
    data : xarray.DataArray
        Real concentration in mM with a dynamic dimension named by ``dynamic_dim``
        (default ``repetition_idx``) and arbitrary labelled spatial dimensions.
    fit_params : dict or None, optional
        Scientific and execution options. ``aif_signal`` is required in mM;
        ``time_points_s`` is seconds and defaults to the dynamic coordinate.
        The input mapping is preserved.
    return_params_info : bool, default False
        Return None without preparing a dataset when only parameter info is wanted.
    base_attrs : dict[str, object] or None, optional
        Initial dataset attributes; model metadata and supplied options take
        precedence according to the existing preparation contract.

    Returns
    -------
    xarray.Dataset or None
        Dataset without variables, carrying effective model/time/AIF and execution
        attributes. Array values are referenced rather than eagerly reconstructed.
        This deterministic metadata operation fits no curves and keeps no cache.

    Raises
    ------
    ValueError
        If the required AIF is missing.

    Examples
    --------
    >>> prepared = prepare_sourbron_fit_params_dataset(
    ...     concentrations, {"aif_signal": plasma_mm, "time_points_s": times_s}
    ... )  # doctest: +SKIP
    """
    if return_params_info:
        return None
    fit_params = fit_params or {}
    aif_signal = fit_params.get("aif_signal", None)
    if aif_signal is None:
        raise ValueError("Sourbron fitting requires 'aif_signal' in fit_params.")

    attrs = dict(base_attrs or {})
    attrs.update(fit_params)
    dynamic_dim = fit_params.get("dynamic_dim", "repetition_idx")
    time_points_s = fit_params.get("time_points_s", data.coords.get(dynamic_dim, None))
    attrs.update(
        {
            "fit_type": "sourbron",
            "model_id": "sourbron",
            "model_name": RESULT_MODEL_NAME,
            "dynamic_dim": dynamic_dim,
            "time_points_s": time_points_s,
            "aif_signal": aif_signal,
            "fit_start_time_s": fit_params.get("fit_start_time_s", -np.inf),
            "fit_stop_time_s": fit_params.get("fit_stop_time_s", np.inf),
            "fit_start_index": fit_params.get("fit_start_index", 0),
            "fit_stop_index": fit_params.get("fit_stop_index", None),
            "aif_threshold": fit_params.get("aif_threshold", 0.0),
        }
    )
    copy_execution_params(fit_params, attrs)
    return xr.Dataset(attrs=attrs)


def fit_sourbron_separable_compartment_curve(
    tissue_concentration: np.ndarray,
    *,
    aif_concentration: np.ndarray,
    time_s: np.ndarray,
    fit_params: Mapping[str, object] | SourbronFitOptions | None = None,
) -> dict[str, float | bool | np.ndarray]:
    """Fit the Sourbron model to one plasma-input concentration curve.

    Parameters
    ----------
    tissue_concentration : numpy.ndarray
        Tissue concentration in mM, shape ``(n_time,)`` after squeezing.
    aif_concentration : numpy.ndarray
        Plasma concentration in mM on the same time axis.
    time_s : numpy.ndarray
        Finite, strictly increasing acquisition times in seconds; sampling may
        be non-uniform. The complete AIF axis is used for convolution.
    fit_params : Mapping[str, object] or SourbronFitOptions or None, optional
        Index/time bounds and ``fit_indices`` select samples. End indices are
        exclusive absolute indices; time bounds are inclusive. ``fit_times_s``
        is not read. ``aif_threshold`` defaults to 0 mM. ``estimate_tp`` adds
        the finite plasma-transit extension: linear AIF interpolation and both
        stages on a uniform internal grid with step at most 0.1 s. That AIF
        grid is prepared once per fit and shared by all optimizer evaluations
        and the returned prediction. Initial guesses and bounds follow
        ``_default_sourbron_bounds``; ``loss`` (default linear), ``f_scale``
        (default 1) and ``max_nfev`` (default 300) configure least squares.

    Returns
    -------
    dict
        Vascular fraction ``vp``, flow ``Ft``/``tubular_flow_s`` in 1/s and
        transit time ``Tt``/``tubular_transit_time_s`` in seconds. Flow converted
        to mL/min/100 mL is also returned. The Tp extension adds plasma transit,
        plasma flow and extraction-fraction aliases. ``prediction`` is float64
        concentration in mM on the complete axis; ``fit_window_mask`` is boolean
        and excludes non-finite tissue/AIF samples. RSS is in mM², RMSE in mM,
        and R² is dimensionless (NaN for constant observations). Success flags
        and ``n_points_fit`` describe optimization. Fewer than three selected
        samples, or five with Tp, return NaN parameters/prediction and false
        success. Covariance is not estimated. Inputs are not mutated and fixed
        inputs/options produce deterministic, eager results.

    Raises
    ------
    ValueError
        If time/curve shapes or sampling are invalid, or optimizer bounds,
        starts or options are inconsistent.
    IndexError
        If an explicitly selected sample index is outside the time array.

    Examples
    --------
    >>> fitted = fit_sourbron_separable_compartment_curve(
    ...     tissue, aif_concentration=plasma, time_s=times_s,
    ...     fit_params={"estimate_tp": False}
    ... )  # doctest: +SKIP
    """
    if isinstance(fit_params, SourbronFitOptions):
        fit_params = fit_params.to_fit_params()
    fit_params = dict(fit_params or {})
    time_s = _validate_time_axis(time_s)
    tissue_curve = _coerce_numeric_1d("tissue_concentration", tissue_concentration)
    aif = _coerce_numeric_1d("aif_concentration", aif_concentration)
    estimate_tp = bool(fit_params.get("estimate_tp", False))

    if len(tissue_curve) != len(time_s) or len(aif) != len(time_s):
        raise ValueError(
            "tissue_concentration, aif_concentration, and time_s must have matching lengths."
        )

    fit_mask = _build_sourbron_fit_mask(
        time_s,
        fit_params=fit_params,
        aif_concentration=aif,
    )
    fit_mask &= np.isfinite(tissue_curve) & np.isfinite(aif)

    minimum_points_required = 5 if estimate_tp else 3
    if int(fit_mask.sum()) < minimum_points_required:
        empty_prediction = np.full_like(time_s, np.nan, dtype=np.float64)
        result = {
            "vp": np.nan,
            "Ft": np.nan,
            "Tt": np.nan,
            "tubular_flow_s": np.nan,
            "tubular_flow_ml_min_100ml": np.nan,
            "tubular_transit_time_s": np.nan,
            "rss": np.nan,
            "rmse": np.nan,
            "r_squared": np.nan,
            "success": False,
            "fit_success": False,
            "n_points_fit": int(fit_mask.sum()),
            "prediction": empty_prediction,
            "fit_window_mask": fit_mask,
        }
        if estimate_tp:
            result.update(
                {
                    "Tp": np.nan,
                    "plasma_transit_time_s": np.nan,
                    "Fp": np.nan,
                    "plasma_flow_s": np.nan,
                    "plasma_flow_ml_min_100ml": np.nan,
                    "extraction_fraction": np.nan,
                }
            )
        return result

    prepared = _prepare_finite_transit_input(time_s, aif) if estimate_tp else None
    x0, lower_bounds, upper_bounds = _default_sourbron_bounds(
        time_s,
        fit_params=fit_params,
    )

    def predict(params: np.ndarray) -> np.ndarray:
        vp, ft_s, tt_s = params[:3]
        if estimate_tp:
            return _finite_transit_concentration(*prepared, vp, ft_s, tt_s, params[3])
        return sourbron_separable_compartment_model(
            time_s,
            aif,
            plasma_volume_fraction=vp,
            tubular_flow_s=ft_s,
            tubular_transit_time_s=tt_s,
        )

    def residuals(params: np.ndarray) -> np.ndarray:
        return predict(params)[fit_mask] - tissue_curve[fit_mask]

    result = least_squares(
        residuals,
        x0=x0,
        bounds=(lower_bounds, upper_bounds),
        loss=str(fit_params.get("loss", "linear")),
        f_scale=float(fit_params.get("f_scale", 1.0)),
        max_nfev=int(fit_params.get("max_nfev", 300)),
    )

    vp, ft_s, tt_s = result.x[:3]
    tp_s = float(result.x[3]) if estimate_tp else np.nan
    prediction = predict(result.x)

    metrics = _residual_statistics(tissue_curve[fit_mask], prediction[fit_mask])
    fp_s = np.nan if not estimate_tp or tp_s <= 0 else float(vp / tp_s)
    extraction_fraction = (
        np.nan if not np.isfinite(fp_s) or fp_s <= 0 else float(ft_s / fp_s)
    )

    output = {
        "vp": float(vp),
        "Ft": float(ft_s),
        "Tt": float(tt_s),
        "tubular_flow_s": float(ft_s),
        "tubular_flow_ml_min_100ml": float(ft_s * _PER_S_TO_ML_MIN_PER_100ML),
        "tubular_transit_time_s": float(tt_s),
        "rss": metrics["rss"],
        "rmse": metrics["rmse"],
        "r_squared": metrics["r_squared"],
        "success": bool(result.success),
        "fit_success": bool(result.success),
        "n_points_fit": int(fit_mask.sum()),
        "prediction": prediction,
        "fit_window_mask": fit_mask,
    }
    if estimate_tp:
        output.update(
            {
                "Tp": tp_s,
                "plasma_transit_time_s": tp_s,
                "Fp": fp_s,
                "plasma_flow_s": fp_s,
                "plasma_flow_ml_min_100ml": (
                    np.nan
                    if not np.isfinite(fp_s)
                    else float(fp_s * _PER_S_TO_ML_MIN_PER_100ML)
                ),
                "extraction_fraction": extraction_fraction,
            }
        )
    return output


def sourbron_model_reconstruction(
    time_points_s: np.ndarray,
    aif_signal: np.ndarray,
    vp: xr.DataArray,
    Ft: xr.DataArray,
    Tt: xr.DataArray,
    Tp: xr.DataArray | float | None = None,
) -> xr.DataArray:
    """Reconstruct labelled concentration curves from Sourbron fit parameters.

    Parameters
    ----------
    time_points_s : numpy.ndarray
        Finite, increasing acquisition times in seconds, shape ``(n_time,)``.
        At least three samples are required; irregular sampling is supported.
    aif_signal : numpy.ndarray
        Plasma concentration in mM on the same acquisition samples.
    vp : xarray.DataArray
        Dimensionless plasma-volume-fraction map.
    Ft : xarray.DataArray
        Normalized tubular-flow parameter map in 1/s.
    Tt : xarray.DataArray
        Positive tubular-transit-time map in seconds.
    Tp : xarray.DataArray or float or None, optional
        Positive plasma transit time in seconds for the finite-transit model.
        None retains the reduced model that uses the supplied plasma input
        directly. With Tp, both stages use one linearly interpolated AIF and
        internally oversampled grid (maximum step 0.1 s), and tissue is returned
        only at the supplied acquisition times. This numerical resolution
        does not establish noisy parameter identifiability.

    Returns
    -------
    xarray.DataArray
        Float64 tissue concentration in mM. Parameter dimensions and coordinates
        broadcast in the existing vascular-then-tubular order. The
        ``repetition_idx`` coordinate holds ``time_points_s`` in seconds;
        accessors restore dataset sample labels separately. Computation is eager and deterministic;
        inputs are preserved and no mutable cache is retained.

    Raises
    ------
    ValueError
        If input axes, sampling or transit times are invalid, or parameter maps
        cannot align.

    Examples
    --------
    >>> prediction = sourbron_model_reconstruction(
    ...     times_s, plasma_mm, vp_map, ft_map, tt_map
    ... )  # doctest: +SKIP
    """
    return sourbron_separable_compartment_model(
        np.asarray(time_points_s, dtype=np.float64),
        np.asarray(aif_signal, dtype=np.float64),
        plasma_volume_fraction=vp,
        tubular_flow_s=Ft,
        tubular_transit_time_s=Tt,
        plasma_transit_time_s=Tp,
    )


__all__ = [
    "SourbronFitOptions",
    "sourbron_separable_compartment_model",
    "get_sourbron_params_info",
    "prepare_sourbron_fit_params_dataset",
    "fit_sourbron_separable_compartment_curve",
    "sourbron_model_reconstruction",
    "MODEL_ID",
    "RESULT_MODEL_NAME",
]


def _annotate_model_output(dataset: xr.Dataset):
    """Attach the existing model-specific descriptions in place."""
    for var_name, units, description in (
        ("vp", "1", "Sourbron plasma volume fraction."),
        ("Ft", "s^-1", "Sourbron tubular / outflow term."),
        ("Tt", "s", "Sourbron tubular transit time."),
        (
            "Tp",
            "s",
            "Optional Sourbron tissue plasma transit time estimated by the full 4-parameter variant.",
        ),
        (
            "plasma_transit_time_s",
            "s",
            "Clear-name alias of the optional Sourbron Tp parameter.",
        ),
        (
            "Fp",
            "s^-1",
            "Optional Sourbron plasma flow term derived as vp / Tp.",
        ),
        (
            "plasma_flow_s",
            "s^-1",
            "Clear-name alias of the optional Sourbron Fp term.",
        ),
        (
            "plasma_flow_ml_min_100ml",
            "mL/min/100mL",
            "Optional Sourbron plasma flow converted to mL/min/100mL.",
        ),
        (
            "extraction_fraction",
            "1",
            "Optional Sourbron extraction fraction Ft / Fp from the 4-parameter variant.",
        ),
        ("tubular_flow_s", "s^-1", "Clear-name alias of the Sourbron Ft term."),
        (
            "tubular_flow_ml_min_100ml",
            "mL/min/100mL",
            "Sourbron tubular flow converted to mL/min/100mL.",
        ),
        (
            "tubular_transit_time_s",
            "s",
            "Clear-name alias of the Sourbron tubular transit time.",
        ),
        (
            "fit_success",
            "bool",
            "Boolean flag indicating Sourbron fit convergence.",
        ),
        (
            "n_points_fit",
            "count",
            "Number of dynamic samples retained in the Sourbron fit window.",
        ),
        ("time_points_s", "s", "Time axis used for Sourbron fitting."),
    ):
        _set_var_attrs(
            dataset,
            var_name,
            long_name=var_name,
            description=description,
            units=units,
            model="sourbron",
        )


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
    aif_signal = kwargs.get("aif_signal")
    if aif_signal is None:
        aif_signal = select_reconstruction_array(
            data_to_reconstruct,
            "aif_signal",
            voxel_coords,
        )

    if time_points_s is None or aif_signal is None:
        raise ValueError(
            "time_points_s (via `dynamic_dim` or `time_points_s`) and aif_signal must be provided via kwargs or exist in dataset variables for Sourbron reconstruction."
        )

    params = {
        "time_points_s": (
            time_points_s.values
            if isinstance(time_points_s, xr.DataArray)
            else np.asarray(time_points_s)
        ),
        "aif_signal": (
            aif_signal.values
            if isinstance(aif_signal, xr.DataArray)
            else np.asarray(aif_signal)
        ),
        "vp": data_to_reconstruct["vp"],
        "Ft": data_to_reconstruct["Ft"],
        "Tt": data_to_reconstruct["Tt"],
        "Tp": data_to_reconstruct.get("Tp"),
    }
    return params


def _fit_voxel(signal, dynamic_coords, fit_params, indices):
    """Adapt this model to the existing shared voxel executor."""
    return (
        fit_sourbron_separable_compartment_curve(
            signal,
            aif_concentration=fit_params["aif_signal"],
            time_s=fit_params["time_points_s"],
            fit_params=fit_params,
        ),
        indices,
    )


def _output_param_names(params: dict):
    """Return the existing ordered output names for these options."""
    output_param_names = [
        "vp",
        "Ft",
        "Tt",
        "tubular_flow_s",
        "tubular_flow_ml_min_100ml",
        "tubular_transit_time_s",
        "rss",
        "rmse",
        "r_squared",
        "fit_success",
        "n_points_fit",
    ]
    if bool(params.get("estimate_tp", False)):
        output_param_names.extend(
            [
                "Tp",
                "plasma_transit_time_s",
                "Fp",
                "plasma_flow_s",
                "plasma_flow_ml_min_100ml",
                "extraction_fraction",
            ]
        )
    return output_param_names


def _attach_fit_options(output_dataset: xr.Dataset, fit_params: dict):
    """Record this model's existing effective reconstruction options."""
    output_dataset.attrs["fit_start_index"] = fit_params.get("fit_start_index", 0)
    output_dataset.attrs["fit_end_index"] = fit_params.get("fit_end_index")
    output_dataset.attrs["fit_start_time_s"] = fit_params.get(
        "fit_start_time_s", -np.inf
    )
    output_dataset.attrs["fit_end_time_s"] = fit_params.get("fit_end_time_s", np.inf)
    output_dataset.attrs["estimate_tp"] = bool(fit_params.get("estimate_tp", False))
    output_dataset.attrs["sourbron_model_variant"] = (
        "full_with_tp"
        if bool(fit_params.get("estimate_tp", False))
        else "reduced_cp_input"
    )
    if output_dataset.attrs["estimate_tp"]:
        output_dataset.attrs.update(_finite_transit_metadata())


def _finite_transit_metadata():
    """Name the numerical resolution and AIF interpretation used by finite Tp."""
    return {
        "sourbron_convolution_method": "oversampled_trapezoidal",
        "sourbron_integration_max_step_s": _FINITE_TRANSIT_MAX_STEP_S,
        "sourbron_aif_interpolation": "linear",
    }


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
    _, lower_bounds, upper_bounds = _default_sourbron_bounds(
        time_points_for_fit,
        fit_params,
    )
    tol = 1e-6
    estimate_tp = "Tp" in fit_results
    parameter_count = 4 if estimate_tp else 3
    finite_params = (
        np.isfinite(fit_results["vp"])
        & np.isfinite(fit_results["Ft"])
        & np.isfinite(fit_results["Tt"])
    )
    if estimate_tp:
        finite_params = finite_params & np.isfinite(fit_results["Tp"])
    hit_bound = (
        _dataarray_isclose(fit_results["vp"], float(lower_bounds[0]), atol=tol)
        | _dataarray_isclose(fit_results["vp"], float(upper_bounds[0]), atol=tol)
        | _dataarray_isclose(fit_results["Ft"], float(lower_bounds[1]), atol=tol)
        | _dataarray_isclose(fit_results["Ft"], float(upper_bounds[1]), atol=tol)
        | _dataarray_isclose(fit_results["Tt"], float(lower_bounds[2]), atol=tol)
        | _dataarray_isclose(fit_results["Tt"], float(upper_bounds[2]), atol=tol)
    )
    if estimate_tp:
        hit_bound = (
            hit_bound
            | _dataarray_isclose(fit_results["Tp"], float(lower_bounds[3]), atol=tol)
            | _dataarray_isclose(fit_results["Tp"], float(upper_bounds[3]), atol=tol)
        )
    prediction = sourbron_model_reconstruction(
        np.asarray(aif_time_s, dtype=np.float64),
        np.asarray(aif_concentration, dtype=np.float64),
        fit_results["vp"],
        fit_results["Ft"],
        fit_results["Tt"],
        fit_results.get("Tp"),
    )
    prediction = prediction.assign_coords(repetition_idx=rep_coord)
    if "time_s" in concentration_map.coords:
        prediction = prediction.assign_coords(time_s=concentration_map.coords["time_s"])
    selected_tofts_model = None
    return prediction, parameter_count, finite_params, hit_bound, selected_tofts_model


_prepare_fit_params = prepare_sourbron_fit_params_dataset
_describe_params = get_sourbron_params_info
_reconstruct = sourbron_model_reconstruction
_OUTPUT_DTYPES = {"fit_success": bool, "n_points_fit": np.int32}
_DEFAULT_MODEL_MODE = "sourbron"


def _quality_window(times, plasma, params):
    return _build_sourbron_fit_mask(times, fit_params=params, aif_concentration=plasma)


def _attach_reconstruction_arrays(output_dataset, fit_params, dynamic_dim_name, data):
    """Attach the existing seconds and plasma concentration arrays."""
    _attach_dce_reconstruction_arrays(
        output_dataset,
        fit_params,
        dynamic_dim_name,
        data,
        aif_times_key="time_points_s",
    )


def _finalize_pipeline_metadata(final_ds, fit_params_pk):
    """Retain model-specific variant and equation-equivalence attributes."""
    final_ds.attrs["estimate_tp"] = bool(fit_params_pk.get("estimate_tp", False))
    final_ds.attrs["sourbron_model_variant"] = (
        "full_with_tp"
        if bool(fit_params_pk.get("estimate_tp", False))
        else "reduced_cp_input"
    )
    if final_ds.attrs["estimate_tp"]:
        final_ds.attrs.update(_finite_transit_metadata())
    if not bool(
        final_ds.attrs.get("estimate_tp", fit_params_pk.get("estimate_tp", False))
    ):
        final_ds.attrs.setdefault(
            "equation_equivalent_fit_types",
            "extended_tofts_plasma,kidney_two_compartment_outflow,sourbron",
        )
        final_ds.attrs.setdefault(
            "fit_type_equivalence_note",
            "The implemented Sourbron separable renal compartment fit is mathematically equivalent to the extended Tofts 2-compartment form, but it is optimized and reported in the renal vp/Ft/Tt parameterization.",
        )
