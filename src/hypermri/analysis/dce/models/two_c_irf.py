"""Cortical renal 2C-IRF concentration model and curve fitting.

All time axes are in seconds; input concentrations are mM. Flow parameters are
normalized rates in 1/s, not absolute organ flows. This module preserves the
existing parameterization and sampling rule. A single bounded start can converge
to a local minimum, and sample-grid quadrature can bias parameter recovery.
Optimizer success and a high R² do not establish flow identifiability or accuracy;
the curve fitter documents the characterized limits. Functions are eager,
deterministic and preserve inputs without caches.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import xarray as xr
from scipy.integrate import trapezoid
from scipy.optimize import least_squares

from hypermri.analysis.dce.fit_metrics import _residual_statistics
from hypermri.analysis.dce.fit_window import select_fit_window
from hypermri.analysis.dce.sampling import _coerce_numeric_1d, _validate_dce_time_axis
from hypermri.analysis.fitting.models.quantitative.convolution import (
    trapezoidal_convolution,
)
from hypermri.analysis.fitting.result_metadata import select_reconstruction_array

from ..fit_quality import _dataarray_isclose, _parameters_at_bounds
from ..result_metadata import _attach_dce_reconstruction_arrays, _set_var_attrs


@dataclass(frozen=True)
class TwoCIrfFitOptions:
    """Immutable scientific overrides for the two c irf curve fit.

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
    insufficient_signal_threshold_mm : float | None, optional
        Minimum selected tissue peak magnitude in mM.
    flow_scale_guess_per_s : float | None, optional
        Lower flow-scale guess in 1/s, combined with the measured tissue/AIF scale.
    gfr_guess_per_s : float | None, optional
        Initial normalized GFR guess in 1/s.
    rpf_guess_per_s : float | None, optional
        Initial normalized RPF guess in 1/s.
    vascular_delay_guess_s : float | None, optional
        Initial vascular delay in seconds; the existing tissue-onset heuristic may replace it.
    vascular_dispersion_guess_per_s : float | None, optional
        Initial vascular dispersion rate in 1/s.
    tubular_delay_guess_s : float | None, optional
        Initial tubular delay in seconds.
    tubular_dispersion_guess_per_s : float | None, optional
        Initial tubular dispersion rate in 1/s.
    flow_upper_bound_per_s : float | None, optional
        Upper bound for GFR and vascular excess flow in 1/s.
    dispersion_rate_upper_bound_per_s : float | None, optional
        Upper bound for both dispersion rates in 1/s.
    delay_upper_bound_s : float | None, optional
        Upper bound for both delays in seconds.

    Raises
    ------
    ValueError
        If supplied parameter vectors have an incompatible length.

    Notes
    -----
    Optimizer order is (GFR, RPF - GFR, vascular delay, vascular dispersion
    rate, tubular delay, tubular dispersion rate), with units (1/s, 1/s, s, 1/s,
    s, 1/s). Negative ends select no samples. These are normalized rates, not
    absolute organ flows. One bounded optimizer start is used; an explicit
    initial vector does not eliminate sampling error or parameter tradeoffs.
    See ``fit_two_c_irf_curve`` for the characterized recovery limits. All
    existing sampling-dependent starts/bounds remain unchanged.

    Examples
    --------
    >>> TwoCIrfFitOptions(fit_start_time_s=5.0).to_fit_params()
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
    insufficient_signal_threshold_mm: float | None = None
    flow_scale_guess_per_s: float | None = None
    gfr_guess_per_s: float | None = None
    rpf_guess_per_s: float | None = None
    vascular_delay_guess_s: float | None = None
    vascular_dispersion_guess_per_s: float | None = None
    tubular_delay_guess_s: float | None = None
    tubular_dispersion_guess_per_s: float | None = None
    flow_upper_bound_per_s: float | None = None
    dispersion_rate_upper_bound_per_s: float | None = None
    delay_upper_bound_s: float | None = None

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
            "insufficient_signal_threshold": self.insufficient_signal_threshold_mm,
            "flow_scale_guess": self.flow_scale_guess_per_s,
            "gfr_guess": self.gfr_guess_per_s,
            "rpf_guess": self.rpf_guess_per_s,
            "vascular_delay_guess": self.vascular_delay_guess_s,
            "vascular_dispersion_guess": self.vascular_dispersion_guess_per_s,
            "tubular_delay_guess": self.tubular_delay_guess_s,
            "tubular_dispersion_guess": self.tubular_dispersion_guess_per_s,
            "flow_upper_bound": self.flow_upper_bound_per_s,
            "dispersion_rate_upper_bound": self.dispersion_rate_upper_bound_per_s,
            "delay_upper_bound": self.delay_upper_bound_s,
        }
        return {name: value for name, value in values.items() if value is not None}


_PER_S_TO_ML_MIN_PER_100ML = 6000.0


def two_c_irf_retention_function(
    lag_time_s: np.ndarray,
    *,
    gfr_s: float,
    rpf_s: float,
    vascular_delay_s: float,
    vascular_dispersion_rate_s: float,
    tubular_delay_s: float,
    tubular_dispersion_rate_s: float,
) -> np.ndarray:
    """Evaluate the existing cortical renal 2C-IRF retention kernel.

    Parameters
    ----------
    lag_time_s : numpy.ndarray
        Evaluation lags in seconds; any shape is preserved. Negative lags give zero.
    gfr_s : float
        Normalized filtration rate in 1/s, clipped to at least zero.
    rpf_s : float
        Normalized plasma flow in 1/s, clipped to at least the effective GFR.
    vascular_delay_s : float
        Vascular delay in seconds, clipped to at least zero.
    vascular_dispersion_rate_s : float
        Vascular dispersion rate in 1/s, floored at float64 tiny.
    tubular_delay_s : float
        Tubular delay in seconds, clipped to at least zero.
    tubular_dispersion_rate_s : float
        Tubular dispersion rate in 1/s, floored at float64 tiny.

    Returns
    -------
    numpy.ndarray
        Float64 retention kernel in 1/s with the input shape and orientation.
        Nearly equal dispersion rates use the existing analytic limiting form.
        Evaluation is eager, deterministic and preserves all inputs.

    Notes
    -----
    This preserves the package's modified cortical 2C-IRF parameterization;
    Flow, delay and dispersion parameters can trade off during fitting; see
    ``fit_two_c_irf_curve`` for the recovery limits. Rates are normalized to
    tissue volume.

    Examples
    --------
    >>> kernel = two_c_irf_retention_function(
    ...     np.array([-1., 0., 3.]), gfr_s=0.01, rpf_s=0.05,
    ...     vascular_delay_s=2., vascular_dispersion_rate_s=0.2,
    ...     tubular_delay_s=4., tubular_dispersion_rate_s=0.05
    ... )
    >>> kernel.shape
    (3,)
    """
    lag = np.asarray(lag_time_s, dtype=np.float64)
    gfr = max(float(gfr_s), 0.0)
    rpf = max(float(rpf_s), gfr)
    t_a = max(float(vascular_delay_s), 0.0)
    m_a = max(float(vascular_dispersion_rate_s), np.finfo(np.float64).tiny)
    t_t = max(float(tubular_delay_s), 0.0)
    m_t = max(float(tubular_dispersion_rate_s), np.finfo(np.float64).tiny)

    kernel = np.zeros_like(lag, dtype=np.float64)
    positive = lag >= 0.0
    if not np.any(positive):
        return kernel

    lag_positive = lag[positive]
    early = lag_positive <= t_a
    middle = (lag_positive > t_a) & (lag_positive <= (t_a + t_t))
    late = lag_positive > (t_a + t_t)

    kernel_positive = np.zeros_like(lag_positive, dtype=np.float64)
    kernel_positive[early] = rpf
    if np.any(middle):
        tau = lag_positive[middle] - t_a
        kernel_positive[middle] = (rpf - gfr) * np.exp(-m_a * tau) + gfr
    if np.any(late):
        tau = lag_positive[late] - t_a
        late_tail = (rpf - gfr) * np.exp(-m_a * tau)
        x = lag_positive[late] - t_a - t_t
        if np.isclose(m_t, m_a, rtol=1e-6, atol=1e-9):
            e1 = np.exp(-m_a * x) * (1.0 + m_a * x)
        else:
            e1 = (m_t * np.exp(-m_a * x) - m_a * np.exp(-m_t * x)) / (m_t - m_a)
        kernel_positive[late] = late_tail + gfr * e1

    kernel[positive] = kernel_positive
    return kernel


def _prepare_two_c_irf_prediction(
    time_points_s: np.ndarray,
    aif_concentration: np.ndarray,
) -> dict[str, np.ndarray]:
    """Validate the time axis and AIF reused across repeated 2C-IRF evaluations."""
    time_points = _validate_dce_time_axis("time_points_s", time_points_s)
    aif = _coerce_numeric_1d("aif_concentration", aif_concentration)
    if len(aif) != len(time_points):
        raise ValueError(
            "aif_concentration and time_points_s must have matching lengths."
        )
    return {"time_points_s": time_points, "aif_concentration": aif}


def _predict_two_c_irf_curve(
    prepared: Mapping[str, object],
    *,
    gfr_s: float,
    rpf_s: float,
    vascular_delay_s: float,
    vascular_dispersion_rate_s: float,
    tubular_delay_s: float,
    tubular_dispersion_rate_s: float,
) -> np.ndarray:
    """Evaluate the 2C-IRF model by trapezoidal convolution on the samples."""
    time_points = np.asarray(prepared["time_points_s"], dtype=np.float64)
    aif = np.asarray(prepared["aif_concentration"], dtype=np.float64)
    return trapezoidal_convolution(
        aif,
        time_points,
        lambda lag_s: two_c_irf_retention_function(
            lag_s,
            gfr_s=gfr_s,
            rpf_s=rpf_s,
            vascular_delay_s=vascular_delay_s,
            vascular_dispersion_rate_s=vascular_dispersion_rate_s,
            tubular_delay_s=tubular_delay_s,
            tubular_dispersion_rate_s=tubular_dispersion_rate_s,
        ),
    )


def two_c_irf_model(
    time_points_s: np.ndarray,
    aif_concentration: np.ndarray,
    *,
    gfr_s: float,
    rpf_s: float,
    vascular_delay_s: float,
    vascular_dispersion_rate_s: float,
    tubular_delay_s: float,
    tubular_dispersion_rate_s: float,
) -> np.ndarray:
    """Evaluate the existing cortical renal 2C-IRF concentration model.

    Parameters
    ----------
    time_points_s : numpy.ndarray
        Finite, strictly increasing acquisition times in seconds, shape (n_time,)
        after squeezing; irregular sampling is supported.
    aif_concentration : numpy.ndarray
        Arterial plasma concentration in mM on the same acquisition samples.
    gfr_s, rpf_s : float
        Normalized filtration and plasma-flow rates in 1/s. The kernel clips GFR
        to nonnegative values and RPF to at least GFR.
    vascular_delay_s, tubular_delay_s : float
        Nonnegative vascular and tubular delays in seconds; negative inputs are
        clipped to zero by the kernel.
    vascular_dispersion_rate_s, tubular_dispersion_rate_s : float
        Dispersion rates in 1/s, floored at float64 tiny by the kernel.

    Returns
    -------
    numpy.ndarray
        Float64 tissue concentration in mM, shape (n_time,), in acquisition order.
        The existing kernel is convolved with the full sampled AIF using the
        shared trapezoidal rule. Inputs are preserved; calculation is eager,
        deterministic and retains no caches.

    Raises
    ------
    ValueError
        If the time/AIF axes are not 1D, times are empty/nonfinite/nonincreasing,
        or sample lengths differ.

    Notes
    -----
    The fit key is ``2c_irf``. Normalized rates require external tissue volume
    for conversion to absolute organ flows; whether all parameters are identifiable
    from typical data has not been established.

    References
    ----------
    Chen B, Zhang Y, Song X, et al. Quantitative Estimation of Renal Function
    with Dynamic Contrast-Enhanced MRI Using a Modified Two-Compartment Model.
    PLoS One. 2014;9(8):e105087. DOI: https://doi.org/10.1371/journal.pone.0105087

    Examples
    --------
    >>> predicted = two_c_irf_model(
    ...     times_s, plasma_mm, gfr_s=0.01, rpf_s=0.05,
    ...     vascular_delay_s=2., vascular_dispersion_rate_s=0.2,
    ...     tubular_delay_s=4., tubular_dispersion_rate_s=0.05
    ... )  # doctest: +SKIP
    """
    prepared = _prepare_two_c_irf_prediction(time_points_s, aif_concentration)
    return _predict_two_c_irf_curve(
        prepared,
        gfr_s=gfr_s,
        rpf_s=rpf_s,
        vascular_delay_s=vascular_delay_s,
        vascular_dispersion_rate_s=vascular_dispersion_rate_s,
        tubular_delay_s=tubular_delay_s,
        tubular_dispersion_rate_s=tubular_dispersion_rate_s,
    )


def _default_two_c_irf_bounds(
    time_points_s: np.ndarray,
    aif_concentration: np.ndarray,
    fit_params: Mapping[str, object] | None = None,
    *,
    tissue_curve: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return initial guesses and bounds for the 2C-IRF fit."""
    fit_params = dict(fit_params or {})
    time_points = _validate_dce_time_axis("time_points_s", time_points_s)
    aif = _coerce_numeric_1d("aif_concentration", aif_concentration)
    if len(aif) != len(time_points):
        raise ValueError(
            "aif_concentration and time_points_s must have matching lengths."
        )

    delta_t = float(np.median(np.diff(time_points))) if len(time_points) > 1 else 1.0
    time_span = float(max(time_points[-1] - time_points[0], delta_t))
    aif_auc = float(trapezoid(np.clip(aif, 0.0, None), time_points))
    tissue_peak = 0.0
    tissue = None
    if tissue_curve is not None:
        tissue = _coerce_numeric_1d("tissue_curve", tissue_curve)
        if len(tissue) != len(time_points):
            raise ValueError(
                "tissue_curve and time_points_s must have matching lengths."
            )
        tissue_peak = float(np.nanmax(np.clip(tissue, 0.0, None)))

    flow_scale_guess = max(
        tissue_peak / max(aif_auc, delta_t),
        float(fit_params.get("flow_scale_guess", 0.01)),
        1e-4,
    )
    gfr_guess = float(fit_params.get("gfr_guess", flow_scale_guess))
    rpf_guess = float(
        fit_params.get("rpf_guess", max(gfr_guess * 1.5, gfr_guess + 1e-3))
    )
    vascular_excess_guess = max(rpf_guess - gfr_guess, 1e-4)

    vascular_delay_guess = float(fit_params.get("vascular_delay_guess", delta_t))
    if (
        tissue is not None
        and tissue_peak > 0.0
        and np.nanmax(np.clip(aif, 0.0, None)) > 0.0
    ):
        tissue_onset = int(np.argmax(np.clip(tissue, 0.0, None) >= 0.1 * tissue_peak))
        aif_onset = int(
            np.argmax(np.clip(aif, 0.0, None) >= 0.1 * float(np.nanmax(aif)))
        )
        vascular_delay_guess = float(
            np.clip(
                time_points[tissue_onset] - time_points[aif_onset], 0.0, time_span * 0.5
            )
        )
    vascular_dispersion_guess = float(
        fit_params.get("vascular_dispersion_guess", 1.0 / max(time_span * 0.2, delta_t))
    )
    tubular_delay_guess = float(
        fit_params.get("tubular_delay_guess", min(2.0 * delta_t, time_span * 0.25))
    )
    tubular_dispersion_guess = float(
        fit_params.get(
            "tubular_dispersion_guess", 1.0 / max(time_span * 0.4, 2.0 * delta_t)
        )
    )

    flow_upper = float(fit_params.get("flow_upper_bound", 5.0))
    rate_upper = float(
        fit_params.get(
            "dispersion_rate_upper_bound", max(5.0 / max(delta_t, 1e-6), 1.0)
        )
    )
    delay_upper = float(
        fit_params.get("delay_upper_bound", max(time_span, 5.0 * delta_t))
    )

    x0 = np.asarray(
        fit_params.get(
            "x0",
            [
                gfr_guess,
                vascular_excess_guess,
                vascular_delay_guess,
                vascular_dispersion_guess,
                tubular_delay_guess,
                tubular_dispersion_guess,
            ],
        ),
        dtype=np.float64,
    )
    lower_bounds = np.asarray(
        fit_params.get(
            "lower_bounds",
            [0.0, 0.0, 0.0, 1e-6, 0.0, 1e-6],
        ),
        dtype=np.float64,
    )
    upper_bounds = np.asarray(
        fit_params.get(
            "upper_bounds",
            [
                flow_upper,
                flow_upper,
                delay_upper,
                rate_upper,
                delay_upper,
                rate_upper,
            ],
        ),
        dtype=np.float64,
    )

    if x0.shape != (6,) or lower_bounds.shape != (6,) or upper_bounds.shape != (6,):
        raise ValueError("2C-IRF x0 and bounds must each contain exactly six values.")

    return np.clip(x0, lower_bounds, upper_bounds), lower_bounds, upper_bounds


def fit_two_c_irf_curve(
    tissue_concentration: np.ndarray,
    *,
    aif_concentration: np.ndarray,
    time_points_s: np.ndarray,
    fit_params: Mapping[str, object] | TwoCIrfFitOptions | None = None,
) -> dict[str, float | bool | np.ndarray]:
    """Fit the cortical renal 2C-IRF model to one concentration curve.

    Parameters
    ----------
    tissue_concentration : numpy.ndarray
        Tissue concentration in mM, shape ``(n_time,)`` after squeezing.
    aif_concentration : numpy.ndarray
        Plasma concentration in mM on the same time axis.
    time_points_s : numpy.ndarray
        Finite, strictly increasing acquisition times in seconds; non-uniform
        sampling is supported. Convolution retains the complete input axis.
    fit_params : Mapping[str, object] or TwoCIrfFitOptions or None, optional
        Window selectors of ``select_fit_window`` and the initial guesses/bounds
        of ``_default_two_c_irf_bounds``. ``insufficient_signal_threshold``
        defaults to 1e-6 mM. ``loss`` (default linear), ``f_scale`` (default 1)
        and ``max_nfev`` (default 600) configure bounded least squares.

    Returns
    -------
    dict
        ``GFR`` and ``RPF`` normalized flow in 1/s; delay/transit times in
        seconds, dispersion rates in 1/s and dimensionless filtration fraction.
        Converted flows are in mL/min/100 mL; absolute mL/min requires an
        external tissue volume. ``prediction`` is float64 mM on the full time
        axis. RSS is mM², RMSE mM and R² dimensionless (NaN for a constant curve).
        ``fit_window_mask`` is boolean and excludes non-finite tissue/AIF
        samples; ``n_points_fit`` and ``fit_success`` describe optimization.
        Insufficient signal or fewer than six selected points return NaN
        parameters/prediction and false success. Covariance is not estimated.
        Inputs are preserved; evaluation is eager and deterministic.

    Raises
    ------
    ValueError
        If curve shapes, sampling, bounds, initial guesses or optimizer options
        are inconsistent.
    IndexError
        If an explicitly selected sample index is outside the time array.

    Notes
    -----
    Fitting uses one bounded least-squares start. ``fit_success`` reports the
    optimizer's termination status; it does not establish a global minimum,
    parameter identifiability or physiological accuracy. R² and RSS assess the
    sampled concentration curve, not recovery of the individual flow parameters.

    A noise-free synthetic dense-reference case with GFR/RPF 0.018/0.032 1/s,
    vascular delay/dispersion 4 s/0.16 1/s and tubular delay/dispersion
    8 s/0.08 1/s returns default GFR about 0.0121/0.0110 1/s on uniform 4 s
    and non-uniform acquisition grids over 176 s, respectively, despite
    successful fits and R² above 0.9999. The reference linearly interpolates
    the sampled AIF and integrates independently at 0.01 s spacing; refining
    it to 0.0025 s does not resolve the discrepancy. Starting at the known
    synthetic parameters gives a GFR closer to truth but a larger residual.
    Additional starts can reveal local minima without correcting sample-grid
    quadrature or guaranteeing accurate GFR. These are limitations of this
    characterized case, not general error bounds. No multi-start selection,
    identifiability threshold or parameter uncertainty estimate is applied.

    Examples
    --------
    >>> fitted = fit_two_c_irf_curve(
    ...     tissue, aif_concentration=plasma, time_points_s=times_s
    ... )  # doctest: +SKIP
    """
    if isinstance(fit_params, TwoCIrfFitOptions):
        fit_params = fit_params.to_fit_params()
    fit_params = dict(fit_params or {})
    time_points = _validate_dce_time_axis("time_points_s", time_points_s)
    tissue_curve = _coerce_numeric_1d("tissue_concentration", tissue_concentration)
    aif = _coerce_numeric_1d("aif_concentration", aif_concentration)

    if len(tissue_curve) != len(time_points) or len(aif) != len(time_points):
        raise ValueError(
            "tissue_concentration, aif_concentration, and time_points_s must have matching lengths."
        )

    fit_mask = np.asarray(
        select_fit_window(
            time_points,
            fit_params,
            fit_type="2c_irf",
        ).values,
        dtype=bool,
    )
    fit_mask &= np.isfinite(tissue_curve) & np.isfinite(aif)
    peak_signal = (
        float(np.nanmax(np.abs(tissue_curve[fit_mask]))) if np.any(fit_mask) else 0.0
    )

    if int(fit_mask.sum()) < 6 or peak_signal <= float(
        fit_params.get("insufficient_signal_threshold", 1e-6)
    ):
        empty_prediction = np.full_like(time_points, np.nan, dtype=np.float64)
        return {
            "GFR": np.nan,
            "RPF": np.nan,
            "vascular_delay_s": np.nan,
            "vascular_dispersion_rate_s": np.nan,
            "tubular_delay_s": np.nan,
            "tubular_dispersion_rate_s": np.nan,
            "vascular_mean_transit_time_s": np.nan,
            "tubular_mean_transit_time_s": np.nan,
            "filtration_fraction": np.nan,
            "gfr_ml_min_100ml": np.nan,
            "rpf_ml_min_100ml": np.nan,
            "rss": np.nan,
            "rmse": np.nan,
            "r_squared": np.nan,
            "fit_success": False,
            "n_points_fit": int(fit_mask.sum()),
            "prediction": empty_prediction,
            "fit_window_mask": fit_mask,
        }

    x0, lower_bounds, upper_bounds = _default_two_c_irf_bounds(
        time_points,
        aif,
        fit_params,
        tissue_curve=tissue_curve,
    )
    prepared_prediction = _prepare_two_c_irf_prediction(time_points, aif)

    def residuals(params: np.ndarray) -> np.ndarray:
        (
            gfr_s,
            vascular_excess_s,
            vascular_delay_s,
            vascular_dispersion_s,
            tubular_delay_s,
            tubular_dispersion_s,
        ) = params
        rpf_s = gfr_s + vascular_excess_s
        prediction = _predict_two_c_irf_curve(
            prepared_prediction,
            gfr_s=gfr_s,
            rpf_s=rpf_s,
            vascular_delay_s=vascular_delay_s,
            vascular_dispersion_rate_s=vascular_dispersion_s,
            tubular_delay_s=tubular_delay_s,
            tubular_dispersion_rate_s=tubular_dispersion_s,
        )
        return prediction[fit_mask] - tissue_curve[fit_mask]

    result = least_squares(
        residuals,
        x0=x0,
        bounds=(lower_bounds, upper_bounds),
        loss=str(fit_params.get("loss", "linear")),
        f_scale=float(fit_params.get("f_scale", 1.0)),
        max_nfev=int(fit_params.get("max_nfev", 600)),
    )

    (
        gfr_s,
        vascular_excess_s,
        vascular_delay_s,
        vascular_dispersion_s,
        tubular_delay_s,
        tubular_dispersion_s,
    ) = result.x
    rpf_s = float(gfr_s + vascular_excess_s)
    prediction = _predict_two_c_irf_curve(
        prepared_prediction,
        gfr_s=float(gfr_s),
        rpf_s=rpf_s,
        vascular_delay_s=float(vascular_delay_s),
        vascular_dispersion_rate_s=float(vascular_dispersion_s),
        tubular_delay_s=float(tubular_delay_s),
        tubular_dispersion_rate_s=float(tubular_dispersion_s),
    )

    metrics = _residual_statistics(tissue_curve[fit_mask], prediction[fit_mask])

    return {
        "GFR": float(gfr_s),
        "RPF": rpf_s,
        "vascular_delay_s": float(vascular_delay_s),
        "vascular_dispersion_rate_s": float(vascular_dispersion_s),
        "tubular_delay_s": float(tubular_delay_s),
        "tubular_dispersion_rate_s": float(tubular_dispersion_s),
        "vascular_mean_transit_time_s": float(
            vascular_delay_s + 1.0 / max(vascular_dispersion_s, 1e-12)
        ),
        "tubular_mean_transit_time_s": float(
            tubular_delay_s + 1.0 / max(tubular_dispersion_s, 1e-12)
        ),
        "filtration_fraction": np.nan if rpf_s <= 0 else float(gfr_s / rpf_s),
        "gfr_ml_min_100ml": float(gfr_s * _PER_S_TO_ML_MIN_PER_100ML),
        "rpf_ml_min_100ml": float(rpf_s * _PER_S_TO_ML_MIN_PER_100ML),
        "rss": metrics["rss"],
        "rmse": metrics["rmse"],
        "r_squared": metrics["r_squared"],
        "fit_success": bool(result.success),
        "n_points_fit": int(fit_mask.sum()),
        "prediction": prediction,
        "fit_window_mask": fit_mask,
    }


_TWO_C_IRF_OUTPUT_NAMES = (
    "GFR",
    "RPF",
    "vascular_delay_s",
    "vascular_dispersion_rate_s",
    "tubular_delay_s",
    "tubular_dispersion_rate_s",
    "vascular_mean_transit_time_s",
    "tubular_mean_transit_time_s",
    "filtration_fraction",
    "gfr_ml_min_100ml",
    "rpf_ml_min_100ml",
    "rss",
    "rmse",
    "fit_success",
    "n_points_fit",
)


def _fit_two_c_irf_voxel(
    *,
    signal: np.ndarray,
    dynamic_coords: np.ndarray,
    fit_params: dict,
    indices: tuple[int, ...],
) -> tuple[dict, tuple[int, ...]]:
    """Adapt a concentration curve to the shared executor using explicit seconds."""
    return (
        fit_two_c_irf_curve(
            signal,
            aif_concentration=fit_params["aif_concentration"],
            time_points_s=fit_params["time_points_s"],
            fit_params=fit_params,
        ),
        indices,
    )


def _fit_two_c_irf_curve_tuple(
    signal_curve: np.ndarray,
    *,
    aif_concentration: np.ndarray,
    time_points_s: np.ndarray,
    fit_params: Mapping[str, object] | None = None,
) -> tuple[
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    bool,
    int,
]:
    result = fit_two_c_irf_curve(
        signal_curve,
        aif_concentration=aif_concentration,
        time_points_s=time_points_s,
        fit_params=fit_params,
    )
    return (
        float(result["GFR"]),
        float(result["RPF"]),
        float(result["vascular_delay_s"]),
        float(result["vascular_dispersion_rate_s"]),
        float(result["tubular_delay_s"]),
        float(result["tubular_dispersion_rate_s"]),
        float(result["vascular_mean_transit_time_s"]),
        float(result["tubular_mean_transit_time_s"]),
        float(result["filtration_fraction"]),
        float(result["gfr_ml_min_100ml"]),
        float(result["rpf_ml_min_100ml"]),
        float(result["rss"]),
        float(result["rmse"]),
        bool(result["fit_success"]),
        int(result["n_points_fit"]),
    )


def two_c_irf_reconstruction(
    time_points_s: np.ndarray,
    aif_concentration: np.ndarray,
    GFR: xr.DataArray,
    RPF: xr.DataArray,
    vascular_delay_s: xr.DataArray,
    vascular_dispersion_rate_s: xr.DataArray,
    tubular_delay_s: xr.DataArray,
    tubular_dispersion_rate_s: xr.DataArray,
) -> xr.DataArray:
    """Reconstruct labelled concentration curves from existing model parameters.

    Parameters
    ----------
    time_points_s : numpy.ndarray
        Finite, strictly increasing acquisition times in seconds, shape (n_time,).
    aif_concentration : numpy.ndarray
        Plasma concentration in mM on the same samples.
    GFR, RPF : xarray.DataArray
        Normalized filtration/plasma-flow parameter maps in 1/s.
    vascular_delay_s, tubular_delay_s : xarray.DataArray
        Vascular/tubular delay maps in seconds.
    vascular_dispersion_rate_s, tubular_dispersion_rate_s : xarray.DataArray
        Dispersion-rate maps in 1/s.

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
        Also raised for incompatible AIF length/shape. Nonfinite scalar
        parameters return an all-NaN curve.

    Examples
    --------
    >>> prediction = two_c_irf_reconstruction(times_s, *parameter_maps)  # doctest: +SKIP
    """
    time_points = _validate_dce_time_axis("time_points_s", time_points_s)
    aif = _coerce_numeric_1d("aif_concentration", aif_concentration)
    if len(aif) != len(time_points):
        raise ValueError(
            "aif_concentration and time_points_s must have matching lengths."
        )

    time_dim_name = "repetition_idx"
    rep_coord = np.arange(len(time_points), dtype=int)
    base_coords = {
        time_dim_name: rep_coord,
        "time_s": (time_dim_name, time_points),
    }
    dask_kwargs = {"output_sizes": {time_dim_name: len(time_points)}}

    prediction = xr.apply_ufunc(
        _reconstruct_two_c_irf_curve_scalar,
        GFR,
        RPF,
        vascular_delay_s,
        vascular_dispersion_rate_s,
        tubular_delay_s,
        tubular_dispersion_rate_s,
        input_core_dims=[[], [], [], [], [], []],
        output_core_dims=[[time_dim_name]],
        vectorize=True,
        output_dtypes=[np.float64],
        dask="allowed",
        dask_gufunc_kwargs=dask_kwargs,
        kwargs={
            "aif_concentration": aif,
            "time_points_s": time_points,
        },
    )
    return prediction.assign_coords(**base_coords)


def _reconstruct_two_c_irf_curve_scalar(
    gfr_s: float,
    rpf_s: float,
    vascular_delay_s: float,
    vascular_dispersion_rate_s: float,
    tubular_delay_s: float,
    tubular_dispersion_rate_s: float,
    *,
    aif_concentration: np.ndarray,
    time_points_s: np.ndarray,
) -> np.ndarray:
    """Scalar adapter used by xarray vectorization for 2C-IRF reconstruction."""
    if not all(
        np.isfinite(value)
        for value in (
            gfr_s,
            rpf_s,
            vascular_delay_s,
            vascular_dispersion_rate_s,
            tubular_delay_s,
            tubular_dispersion_rate_s,
        )
    ):
        return np.full(len(time_points_s), np.nan, dtype=np.float64)

    return two_c_irf_model(
        time_points_s,
        aif_concentration,
        gfr_s=gfr_s,
        rpf_s=rpf_s,
        vascular_delay_s=vascular_delay_s,
        vascular_dispersion_rate_s=vascular_dispersion_rate_s,
        tubular_delay_s=tubular_delay_s,
        tubular_dispersion_rate_s=tubular_dispersion_rate_s,
    )


__all__ = [
    "TwoCIrfFitOptions",
    "two_c_irf_retention_function",
    "two_c_irf_model",
    "fit_two_c_irf_curve",
    "two_c_irf_reconstruction",
]


def _annotate_model_output(dataset: xr.Dataset):
    """Attach the existing model-specific descriptions in place."""
    for var_name, units, description in (
        (
            "GFR",
            "s^-1",
            "Renal cortical filtration term from the 2C-IRF model, normalized per voxel volume.",
        ),
        (
            "RPF",
            "s^-1",
            "Renal plasma flow term from the 2C-IRF model, normalized per voxel volume.",
        ),
        (
            "vascular_delay_s",
            "s",
            "Minimum vascular transit delay of the cortical 2C-IRF retention kernel.",
        ),
        (
            "vascular_dispersion_rate_s",
            "s^-1",
            "Exponential washout / dispersion rate of the vascular 2C-IRF compartment.",
        ),
        (
            "tubular_delay_s",
            "s",
            "Minimum tubular transit delay of the cortical 2C-IRF retention kernel.",
        ),
        (
            "tubular_dispersion_rate_s",
            "s^-1",
            "Exponential washout / dispersion rate of the tubular 2C-IRF compartment.",
        ),
        (
            "vascular_mean_transit_time_s",
            "s",
            "Derived mean transit time of the vascular compartment (delay + 1/rate).",
        ),
        (
            "tubular_mean_transit_time_s",
            "s",
            "Derived mean transit time of the tubular compartment (delay + 1/rate).",
        ),
        (
            "filtration_fraction",
            "1",
            "Derived filtration fraction GFR / RPF from the 2C-IRF model.",
        ),
        (
            "gfr_ml_min_100ml",
            "mL/min/100mL",
            "Normalized GFR map derived from GFR * 6000. Multiply by voxel_volume_ml / 100 for absolute voxelwise mL/min.",
        ),
        (
            "rpf_ml_min_100ml",
            "mL/min/100mL",
            "Normalized RPF map derived from RPF * 6000.",
        ),
        (
            "fit_success",
            "bool",
            "Boolean flag indicating 2C-IRF fit convergence.",
        ),
        (
            "n_points_fit",
            "count",
            "Number of dynamic samples retained in the 2C-IRF fit window.",
        ),
        ("time_points_s", "s", "Time axis used for 2C-IRF fitting."),
    ):
        _set_var_attrs(
            dataset,
            var_name,
            long_name=var_name,
            description=description,
            units=units,
            model="2c_irf",
        )


def _reconstruction_parameters(
    data_to_reconstruct: xr.Dataset, kwargs: dict, voxel_coords: dict | None = None
):
    """Translate legacy reconstruction options without changing precedence."""
    dynamic_dim_override = kwargs.get("dynamic_dim")
    time_points_s = dynamic_dim_override
    if time_points_s is None:
        time_points_s = kwargs.get("time_points_s", kwargs.get("aif_time_s"))
    if time_points_s is None:
        time_points_s = select_reconstruction_array(
            data_to_reconstruct,
            "time_points_s",
            voxel_coords,
        )
    if time_points_s is None:
        time_points_s = select_reconstruction_array(
            data_to_reconstruct,
            "aif_time_s",
            voxel_coords,
        )

    aif_concentration = kwargs.get("aif_concentration")
    if aif_concentration is None:
        aif_concentration = select_reconstruction_array(
            data_to_reconstruct,
            "aif_concentration",
            voxel_coords,
        )

    if time_points_s is None or aif_concentration is None:
        raise ValueError(
            "2C-IRF reconstruction requires time_points_s/aif_time_s and aif_concentration via kwargs or dataset variables."
        )

    params = {
        "time_points_s": (
            time_points_s.values
            if isinstance(time_points_s, xr.DataArray)
            else np.asarray(time_points_s)
        ),
        "aif_concentration": (
            aif_concentration.values
            if isinstance(aif_concentration, xr.DataArray)
            else np.asarray(aif_concentration)
        ),
        "GFR": data_to_reconstruct["GFR"],
        "RPF": data_to_reconstruct["RPF"],
        "vascular_delay_s": data_to_reconstruct["vascular_delay_s"],
        "vascular_dispersion_rate_s": data_to_reconstruct["vascular_dispersion_rate_s"],
        "tubular_delay_s": data_to_reconstruct["tubular_delay_s"],
        "tubular_dispersion_rate_s": data_to_reconstruct["tubular_dispersion_rate_s"],
    }
    return params


def _reconstruct_two_c_irf_prediction(
    fit_results: xr.Dataset,
    concentration_map: xr.DataArray,
    *,
    aif_concentration: np.ndarray,
    time_points_s: np.ndarray,
    fit_params: Mapping[str, object] | None = None,
) -> tuple[xr.DataArray, int, xr.DataArray, xr.DataArray]:
    rep_coord = concentration_map.coords.get(
        "repetition_idx", np.arange(len(time_points_s), dtype=int)
    )
    dask_kwargs = {"output_sizes": {"repetition_idx": len(time_points_s)}}

    prediction = xr.apply_ufunc(
        _reconstruct_two_c_irf_curve_scalar,
        fit_results["GFR"],
        fit_results["RPF"],
        fit_results["vascular_delay_s"],
        fit_results["vascular_dispersion_rate_s"],
        fit_results["tubular_delay_s"],
        fit_results["tubular_dispersion_rate_s"],
        input_core_dims=[[], [], [], [], [], []],
        output_core_dims=[["repetition_idx"]],
        vectorize=True,
        output_dtypes=[np.float64],
        dask="allowed",
        dask_gufunc_kwargs=dask_kwargs,
        kwargs={
            "aif_concentration": np.asarray(aif_concentration, dtype=np.float64),
            "time_points_s": np.asarray(time_points_s, dtype=np.float64),
        },
    ).assign_coords(repetition_idx=rep_coord)

    if "time_s" in concentration_map.coords:
        prediction = prediction.assign_coords(time_s=concentration_map.coords["time_s"])

    parameter_count = 6
    finite_params = (
        np.isfinite(fit_results["GFR"])
        & np.isfinite(fit_results["RPF"])
        & np.isfinite(fit_results["vascular_delay_s"])
        & np.isfinite(fit_results["vascular_dispersion_rate_s"])
        & np.isfinite(fit_results["tubular_delay_s"])
        & np.isfinite(fit_results["tubular_dispersion_rate_s"])
    )
    success_flag = fit_results.get("fit_success")
    if success_flag is not None:
        finite_params = finite_params & success_flag.astype(bool)

    _, lower_bounds, upper_bounds = _default_two_c_irf_bounds(
        time_points_s,
        aif_concentration,
        fit_params,
    )
    tol = 1e-6
    vascular_excess = fit_results["RPF"] - fit_results["GFR"]
    hit_bound = (
        _dataarray_isclose(fit_results["GFR"], float(lower_bounds[0]), atol=tol)
        | _dataarray_isclose(fit_results["GFR"], float(upper_bounds[0]), atol=tol)
        | _dataarray_isclose(vascular_excess, float(lower_bounds[1]), atol=tol)
        | _dataarray_isclose(vascular_excess, float(upper_bounds[1]), atol=tol)
        | _dataarray_isclose(
            fit_results["vascular_delay_s"], float(lower_bounds[2]), atol=tol
        )
        | _dataarray_isclose(
            fit_results["vascular_delay_s"], float(upper_bounds[2]), atol=tol
        )
        | _dataarray_isclose(
            fit_results["vascular_dispersion_rate_s"],
            float(lower_bounds[3]),
            atol=tol,
        )
        | _dataarray_isclose(
            fit_results["vascular_dispersion_rate_s"],
            float(upper_bounds[3]),
            atol=tol,
        )
        | _dataarray_isclose(
            fit_results["tubular_delay_s"], float(lower_bounds[4]), atol=tol
        )
        | _dataarray_isclose(
            fit_results["tubular_delay_s"], float(upper_bounds[4]), atol=tol
        )
        | _dataarray_isclose(
            fit_results["tubular_dispersion_rate_s"],
            float(lower_bounds[5]),
            atol=tol,
        )
        | _dataarray_isclose(
            fit_results["tubular_dispersion_rate_s"],
            float(upper_bounds[5]),
            atol=tol,
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
    ) = _reconstruct_two_c_irf_prediction(
        fit_results,
        concentration_map,
        aif_concentration=np.asarray(aif_concentration, dtype=np.float64),
        time_points_s=time_points_for_fit,
        fit_params=fit_params,
    )
    selected_tofts_model = None
    return prediction, parameter_count, finite_params, hit_bound, selected_tofts_model


RESULT_MODEL_NAME = "2c_irf"
_reconstruct = two_c_irf_reconstruction
_fit_voxel = _fit_two_c_irf_voxel
_OUTPUT_DTYPES = {"fit_success": bool, "n_points_fit": np.int64}
_DEFAULT_MODEL_MODE = "2c_irf"


def _output_param_names(params):
    return _TWO_C_IRF_OUTPUT_NAMES


def _quality_window(times, plasma, params):
    return select_fit_window(times, params, fit_type="2c_irf").values


_BACKEND_DESCRIPTION = "direct concentration-space 2C-IRF fit"
