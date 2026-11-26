"""Standard and extended Tofts concentration models and scientific options.

Times are seconds and concentrations are mM. Numerical curve operations and
legacy signal/parameter adapters share the existing trapezoidal convolution.
The historical window, bounds, precision and optimizer behavior are retained.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr
from scipy.interpolate import interp1d
from scipy.optimize import least_squares

from hypermri.analysis.dce.conversions import (
    DEFAULT_R1_PER_MM_S,
    concentration_to_signal,
)
from hypermri.analysis.dce.fit_metrics import _sum_squared_residuals
from hypermri.analysis.dce.fit_window import _select_fit_samples
from hypermri.analysis.dce.sampling import _validate_dce_time_axis
from hypermri.analysis.fitting.execution import copy_execution_params
from hypermri.analysis.fitting.models.quantitative.convolution import (
    trapezoidal_convolution,
)
from hypermri.analysis.fitting.result_metadata import select_reconstruction_array

from ..fit_quality import _dataarray_isclose, _parameters_at_bounds
from ..result_metadata import _attach_dce_reconstruction_arrays, _set_var_attrs
from .extended_compartment import _extended_compartment_concentration


@dataclass(frozen=True)
class ToftsFitOptions:
    """Immutable scientific overrides for the tofts curve fit.

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
    initial_1comp_per_s : tuple[float, float] | None, optional
        Initial (Ktrans, kep) values in 1/s.
    lower_1comp_per_s : tuple[float, float] | None, optional
        Lower (Ktrans, kep) bounds in 1/s; supply together with upper bounds.
    upper_1comp_per_s : tuple[float, float] | None, optional
        Upper (Ktrans, kep) bounds in 1/s; supply together with lower bounds.
    initial_2comp : tuple[float, float, float] | None, optional
        Initial (Ktrans, kep, vp), with units (1/s, 1/s, fraction).
    lower_2comp : tuple[float, float, float] | None, optional
        Lower extended-model bounds in the same order/units; paired with upper bounds.
    upper_2comp : tuple[float, float, float] | None, optional
        Upper extended-model bounds in the same order/units; paired with lower bounds.

    Raises
    ------
    ValueError
        If supplied parameter vectors have an incompatible length or Tofts
        lower/upper bounds are not paired.

    Notes
    -----
    Model choice remains the explicit model_type argument of
    fit_single_voxel_tofts_helper. Negative ends retain Python slicing. None
    leaves the original helper's start values and
    TOFTS_1COMP_BOUNDS/TOFTS_2COMP_BOUNDS in force. Window options intersect an
    explicitly supplied fit_mask.

    Examples
    --------
    >>> ToftsFitOptions(fit_start_time_s=5.0).to_fit_params()
    {'fit_start_time_s': 5.0}
    """

    fit_start_index: int | None = None
    fit_end_index: int | None = None
    fit_start_time_s: float | None = None
    fit_end_time_s: float | None = None
    fit_indices: tuple[int, ...] | None = None
    fit_times_s: tuple[float, ...] | None = None
    initial_1comp_per_s: tuple[float, float] | None = None
    lower_1comp_per_s: tuple[float, float] | None = None
    upper_1comp_per_s: tuple[float, float] | None = None
    initial_2comp: tuple[float, float, float] | None = None
    lower_2comp: tuple[float, float, float] | None = None
    upper_2comp: tuple[float, float, float] | None = None

    def __post_init__(self) -> None:
        """Copy sequences into immutable tuples and validate parameter counts."""
        for name in (
            "fit_indices",
            "fit_times_s",
            "initial_1comp_per_s",
            "lower_1comp_per_s",
            "upper_1comp_per_s",
            "initial_2comp",
            "lower_2comp",
            "upper_2comp",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, tuple(value))
        for name in (
            "initial_1comp_per_s",
            "lower_1comp_per_s",
            "upper_1comp_per_s",
            "initial_2comp",
            "lower_2comp",
            "upper_2comp",
        ):
            value = getattr(self, name)
            if value is not None and len(value) != (2 if "1comp" in name else 3):
                raise ValueError(
                    f"ToftsFitOptions.{name} must contain exactly "
                    f"{2 if '1comp' in name else 3} parameter values."
                )
        for lower, upper in (
            (self.lower_1comp_per_s, self.upper_1comp_per_s),
            (self.lower_2comp, self.upper_2comp),
        ):
            if (lower is None) != (upper is None):
                raise ValueError(
                    "ToftsFitOptions requires paired lower and upper bounds."
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
            "x0_1comp": self.initial_1comp_per_s,
            "x0_2comp": self.initial_2comp,
        }
        if self.lower_1comp_per_s is not None:
            values["bounds_1comp"] = (self.lower_1comp_per_s, self.upper_1comp_per_s)
        if self.lower_2comp is not None:
            values["bounds_2comp"] = (self.lower_2comp, self.upper_2comp)
        return {name: value for name, value in values.items() if value is not None}


MODEL_ID = "tofts"
RESULT_MODEL_NAME = "tofts_model"
MODEL_EQUATION = "Ct(t) = vp*Cp(t) + Ktrans * integral(Cp(tau)*exp(-kep*(t-tau))dtau)"


def get_tofts_params_info() -> dict[str, object]:
    """Return the public parameter-info entry for the Tofts model.

    Returns
    -------
    dict[str, object]
        ``"required_fit_params"`` and ``"optional_fit_params"`` map fit-option
        names to descriptions, ``"output_parameters"`` maps result variables
        to descriptions and ``"model_equation"`` holds :data:`MODEL_EQUATION`.
    """
    return {
        "required_fit_params": {
            "aif_signal": "Arterial Input Function concentration values (required).",
            "aif_times": "Time points in seconds for the AIF (required).",
        },
        "optional_fit_params": {
            "t1_map": "np.ndarray or xr.DataArray, pre-contrast T1 map in ms.",
            "s0_map": "np.ndarray or xr.DataArray, baseline signal map.",
            "n_precontrast": "number of pre-contrast frames to auto-calculate s0_map if missing.",
            "tr_s": "repetition time in seconds (for signal-to-concentration).",
            "fa_deg": "flip angle in degrees (for signal-to-concentration).",
            "r1": f"relaxivity in 1/(mM s) (default {DEFAULT_R1_PER_MM_S}).",
            "model_mode": '"1comp", "2comp" or "both" (default).',
            "fit_start_index": "first repetition index to include.",
            "fit_end_index": "index after the last repetition to include.",
            "fit_start_time_s": "start time in seconds for the window.",
            "fit_end_time_s": "end time in seconds for the window.",
            "fit_indices": "specific repetition indices to include.",
            "fit_times_s": "specific time points in seconds to include.",
            "use_multiprocessing": "Enable shared voxelwise parallel processing. Default: True",
            "number_of_cpu_cores": "Number of joblib workers for voxelwise processing (-1 uses all). Default: -1",
        },
        "output_parameters": {
            "Ktrans_1c": "Ktrans from 1-compartment Tofts model",
            "kep_1c": "kep from 1-compartment Tofts model",
            "Ktrans_2c": "Ktrans from 2-compartment Extended Tofts model",
            "kep_2c": "kep from 2-compartment Extended Tofts model",
            "vp_2c": "vp from 2-compartment Extended Tofts model",
            "ve_2c": "Ktrans/kep clipped to [0, 1] from the 2-compartment model",
            "rss_1c": "Residual sum of squares of the 1-compartment fit in mM^2",
            "rss_2c": "Residual sum of squares of the 2-compartment fit in mM^2",
        },
        "model_equation": MODEL_EQUATION,
    }


def prepare_tofts_fit_params_dataset(
    data: xr.DataArray,
    fit_params: dict | None = None,
    return_params_info: bool = False,
    *,
    base_attrs: dict[str, object] | None = None,
) -> xr.Dataset | None:
    """Prepare the public quantitative fit-params dataset for Tofts fitting.

    The effective fit options are stored as attributes of an empty dataset.
    Missing options get their defaults; ``r1`` defaults to
    :data:`~hypermri.analysis.dce.conversions.DEFAULT_R1_PER_MM_S`.
    The inputs are not modified.

    Parameters
    ----------
    data : xarray.DataArray
        Data to fit.  Its coordinate along ``fit_params["dynamic_dim"]``
        (default ``"repetition_idx"``) is the default ``time_points_s``.
    fit_params : dict or None, optional
        Fit options.  ``"aif_signal"`` (plasma concentration in mM) and
        ``"aif_times"`` (s) are required; the keys listed by
        :func:`get_tofts_params_info` and the execution options are read.
    return_params_info : bool, default False
        If True, return None because the caller requests the parameter
        information instead of a dataset.
    base_attrs : dict[str, object] or None, optional
        Attributes to start from; the fit options take precedence.

    Returns
    -------
    xarray.Dataset or None
        Dataset without variables whose attributes hold the effective fit
        options, including ``"cp_func"``, a linear interpolator of the AIF in
        mM over time in s that extrapolates outside ``aif_times``.  None if
        ``return_params_info`` is True.

    Raises
    ------
    ValueError
        If ``"aif_signal"`` or ``"aif_times"`` is missing.
    """
    if return_params_info:
        return None
    fit_params = fit_params or {}
    aif_signal = fit_params.get("aif_signal", None)
    aif_times = fit_params.get("aif_times", None)
    if aif_signal is None or aif_times is None:
        raise ValueError(
            "Both 'aif_signal' and 'aif_times' must be provided for Tofts fitting."
        )

    cp_func = interp1d(aif_times, aif_signal, kind="linear", fill_value="extrapolate")

    attrs = dict(base_attrs or {})
    dynamic_dim = fit_params.get("dynamic_dim", "repetition_idx")
    attrs.update(
        {
            "dynamic_dim": dynamic_dim,
            "time_points_s": fit_params.get(
                "time_points_s", data.coords.get(dynamic_dim, None)
            ),
            "cp_func": cp_func,
            "t1_map": fit_params.get("t1_map", None),
            "s0_map": fit_params.get("s0_map", None),
            "tr_s": fit_params.get("tr_s", None),
            "fa_deg": fit_params.get("fa_deg", None),
            "r1": fit_params.get("r1", DEFAULT_R1_PER_MM_S),
            "model_mode": fit_params.get("model_mode", "both"),
            "fit_start_index": fit_params.get("fit_start_index", 0),
            "fit_end_index": fit_params.get(
                "fit_end_index", fit_params.get("fit_stop_index", None)
            ),
            "fit_start_time_s": fit_params.get("fit_start_time_s", -np.inf),
            "fit_end_time_s": fit_params.get(
                "fit_end_time_s", fit_params.get("fit_stop_time_s", np.inf)
            ),
            "fit_indices": fit_params.get("fit_indices", None),
            "fit_times_s": fit_params.get("fit_times_s", None),
            "aif_signal": aif_signal,
            "aif_times": aif_times,
            "aif_start_index": fit_params.get("aif_start_index", 0),
            "aif_end_index": fit_params.get("aif_end_index", None),
        }
    )
    copy_execution_params(fit_params, attrs)
    return xr.Dataset(attrs=attrs)


#: Bounds of the standard Tofts fit as ``(lower, upper)`` for (Ktrans, kep),
#: both in 1/s.  The fit and the "parameter at bound" quality check in
#: ``analysis/dce/fitting_support.py`` read this value.
TOFTS_1COMP_BOUNDS = ((0.0, 0.0), (5.0, 5.0))

#: Bounds of the extended Tofts fit as ``(lower, upper)`` for (Ktrans, kep)
#: in 1/s and the plasma volume fraction vp.
TOFTS_2COMP_BOUNDS = ((0.0, 0.0, 0.0), (5.0, 5.0, 1.0))


def tofts_model_1comp(t, ktrans, kep, cp_func):
    """Evaluate the standard one-compartment Tofts model.

    ``Ct(t) = Ktrans * integral from t[0] to t of Cp(u) exp(-kep (t - u)) du``,
    evaluated with :func:`~.convolution.trapezoidal_convolution` on the
    samples ``t``.

    Parameters
    ----------
    t : numpy.ndarray
        Sample times in s, shape ``(n_time,)``, finite and strictly
        increasing.  The spacing may be non-uniform.
    ktrans : float
        Volume transfer constant Ktrans in 1/s.
    kep : float
        Efflux rate constant kep in 1/s.
    cp_func : callable
        Plasma concentration in mM as a function of time in s; it is
        evaluated only at ``t``.

    Returns
    -------
    numpy.ndarray
        Tissue concentration in mM, float64, shape ``(n_time,)``; 0 at
        ``t[0]``.

    Raises
    ------
    ValueError
        If ``t`` is not 1D, not finite or not strictly increasing.
    """
    return ktrans * trapezoidal_convolution(
        cp_func(t), t, lambda lag_s: np.exp(-kep * lag_s)
    )


def extended_tofts_model_2comp(t, ktrans, kep, vp, cp_func):
    """Evaluate the extended Tofts model with a plasma volume term.

    ``Ct(t) = vp * Cp(t) + Ktrans * integral of Cp(u) exp(-kep (t - u)) du``
    with the shared reduced extended-compartment equation. The native Tofts
    rate kernel retains its multiplication order; fit defaults remain local.

    Parameters
    ----------
    t : numpy.ndarray
        Sample times in s, shape ``(n_time,)``, finite and strictly
        increasing.  The spacing may be non-uniform.
    ktrans : float
        Volume transfer constant Ktrans in 1/s.
    kep : float
        Efflux rate constant kep in 1/s.
    vp : float
        Plasma volume fraction, dimensionless, in ``[0, 1]``.
    cp_func : callable
        Deterministic plasma concentration in mM as a function of time in s,
        returning shape ``(n_time,)``. Evaluated once at ``t`` per call.

    Returns
    -------
    numpy.ndarray
        Tissue concentration in mM, float64, shape ``(n_time,)``. Integration
        uses float64; vascular multiplication retains the AIF operand precision.
        Inputs are preserved. Evaluation is eager, deterministic for a fixed
        plasma function and retains no cache.

    Raises
    ------
    ValueError
        If ``t`` is not 1D, not finite or not strictly increasing.

    Examples
    --------
    >>> times = np.array([0.0, 1.0, 2.0])
    >>> extended_tofts_model_2comp(times, 0.03, 0.1, 0.25, np.ones_like).shape
    (3,)
    """
    return _extended_compartment_concentration(
        t,
        cp_func(t),
        plasma_volume_fraction=vp,
        transfer_rate_s=ktrans,
        retention_kernel=lambda lag_s: np.exp(-kep * lag_s),
    )


def _residual_sum_of_squares(result) -> float:
    """Return the RSS of a ``least_squares`` result; its ``cost`` is half of it."""
    return float(_sum_squared_residuals(result.fun))


def fit_single_voxel_tofts_helper(
    t,
    c_t_measured,
    cp_func,
    model_type="1comp",
    x0_1comp=None,
    bounds_1comp=None,
    x0_2comp=None,
    bounds_2comp=None,
    fit_mask=None,
    *,
    options: ToftsFitOptions | None = None,
):
    """Fit the Tofts or extended Tofts model to one concentration curve.

    The model is fitted by bounded non-linear least squares
    (``scipy.optimize.least_squares``) on the samples selected by
    ``fit_mask``.  The result is deterministic for given inputs.

    Parameters
    ----------
    t : array_like
        Sample times in s, shape ``(n_time,)``, strictly increasing; the
        spacing may be non-uniform.
    c_t_measured : array_like
        Tissue concentration in mM, shape ``(n_time,)``.
    cp_func : callable
        Plasma concentration in mM as a function of time in s.
    model_type : {"1comp", "2comp"}, default "1comp"
        ``"1comp"`` fits (Ktrans, kep) of the standard Tofts model,
        ``"2comp"`` fits (Ktrans, kep, vp) of the extended Tofts model.
    x0_1comp, x0_2comp : sequence of float, optional
        Start values for the respective model; Ktrans and kep in 1/s, vp as a
        fraction.  Default ``[0.01, 0.1]`` and ``[0.01, 0.1, 0.01]``.
    bounds_1comp, bounds_2comp : tuple of sequences, optional
        Lower and upper bounds in the same units.  Default
        ``TOFTS_1COMP_BOUNDS`` and ``TOFTS_2COMP_BOUNDS``.
    fit_mask : array_like of bool, optional
        Samples to include, shape ``(n_time,)``.  All samples are used when
        omitted.
    options : ToftsFitOptions or None, optional
        Immutable window, initial-value and bound overrides. Its window
        intersects ``fit_mask``; its starts/bounds cannot be combined with
        the individual ``x0_*`` or ``bounds_*`` arguments. None fields retain
        the existing defaults. The options object is not mutated.

    Returns
    -------
    params : np.ndarray
        Fitted parameters in the order given for ``model_type``.  All zero
        when no sample is positive; NaN when the inputs are not finite, the
        mask has the wrong shape, fewer samples than parameters are selected,
        or the optimizer fails.
    rss : float
        Residual sum of squares in mM^2 over the samples selected by
        ``fit_mask``, ``sum((model - c_t_measured)**2)``. Sourbron uses the same
        concentration space; Patlak reports dimensionless graphical RSS after
        AIF normalization. ``least_squares`` reports half the squared residual
        sum as ``cost``. NaN when ``params`` is NaN.

    Raises
    ------
    ValueError
        If ``model_type`` is not ``"1comp"`` or ``"2comp"``, starts/bounds
        are supplied through both interfaces, or option-mask shapes cannot
        be combined.

    Notes
    -----
    Times and concentrations are calculated in float64 without modifying the
    acquired arrays. This eager function reads no files and retains no caches.

    Examples
    --------
    >>> parameters, rss = fit_single_voxel_tofts_helper(
    ...     times_s, tissue_mm, plasma_interpolator, model_type="1comp"
    ... )  # doctest: +SKIP
    >>> options = ToftsFitOptions(fit_end_index=-2)
    >>> parameters, rss = fit_single_voxel_tofts_helper(
    ...     times_s, tissue_mm, plasma_interpolator, options=options
    ... )  # doctest: +SKIP
    """

    if options is not None:
        if any(
            value is not None
            for value in (x0_1comp, x0_2comp, bounds_1comp, bounds_2comp)
        ):
            raise ValueError(
                "Tofts starts and bounds must be supplied through options or "
                "the individual arguments, not both."
            )
        params = options.to_fit_params()
        x0_1comp = params.get("x0_1comp")
        x0_2comp = params.get("x0_2comp")
        bounds_1comp = params.get("bounds_1comp")
        bounds_2comp = params.get("bounds_2comp")
        option_mask = build_tofts_fit_mask(t, params)
        fit_mask = (
            option_mask
            if fit_mask is None
            else np.asarray(fit_mask, dtype=bool) & option_mask
        )

    def nan_result():
        n_params = 2 if model_type == "1comp" else 3
        return (np.full(n_params, np.nan, dtype=np.float64), np.nan)

    if x0_1comp is None:
        x0_1comp = [0.01, 0.1]
    if bounds_1comp is None:
        bounds_1comp = TOFTS_1COMP_BOUNDS
    if x0_2comp is None:
        x0_2comp = [0.01, 0.1, 0.01]
    if bounds_2comp is None:
        bounds_2comp = TOFTS_2COMP_BOUNDS

    t = np.asarray(t, dtype=np.float64)
    c_t_measured = np.asarray(c_t_measured, dtype=np.float64)
    if fit_mask is not None:
        fit_mask = np.asarray(fit_mask, dtype=bool).squeeze()
        if fit_mask.shape != c_t_measured.shape:
            return nan_result()

    if not np.all(np.isfinite(t)) or not np.all(np.isfinite(c_t_measured)):
        return nan_result()

    valid_mask = np.ones_like(c_t_measured, dtype=bool)
    if fit_mask is not None:
        valid_mask &= fit_mask

    if np.all(c_t_measured <= 0):
        # The model with zero parameters is zero, so the residual is the curve.
        zero_model_rss = float(_sum_squared_residuals(c_t_measured[valid_mask]))
        return (
            ((0, 0), zero_model_rss)
            if model_type == "1comp"
            else ((0, 0, 0), zero_model_rss)
        )

    n_params = 2 if model_type == "1comp" else 3
    if int(np.sum(valid_mask)) < n_params:
        return nan_result()

    def residual_func(model_vals):
        res = model_vals - c_t_measured
        return res[valid_mask]

    if model_type == "1comp":
        try:
            res = least_squares(
                lambda p: residual_func(tofts_model_1comp(t, p[0], p[1], cp_func)),
                x0=np.array(x0_1comp, dtype=np.float64),
                bounds=bounds_1comp,
                ftol=1e-3,
            )
        except (RuntimeError, ValueError, TypeError):
            return nan_result()
        return (res.x, _residual_sum_of_squares(res)) if res.success else nan_result()
    if model_type == "2comp":
        try:
            res = least_squares(
                lambda p: residual_func(
                    extended_tofts_model_2comp(t, p[0], p[1], p[2], cp_func)
                ),
                x0=np.array(x0_2comp, dtype=np.float64),
                bounds=bounds_2comp,
                ftol=1e-3,
            )
        except (RuntimeError, ValueError, TypeError):
            return nan_result()
        return (res.x, _residual_sum_of_squares(res)) if res.success else nan_result()
    raise ValueError(f"Unknown model_type: {model_type}")


def build_tofts_fit_mask(
    fit_times: np.ndarray,
    fit_params: dict | None = None,
) -> np.ndarray:
    """Build the Tofts window with its historical exclusive-end convention.

    Parameters
    ----------
    fit_times : numpy.ndarray
        Sample times in seconds, 1D after squeezing, in acquisition order.
    fit_params : dict or None, optional
        Intersect ``fit_start_index`` (default 0), exclusive ``fit_end_index``
        (alias ``fit_stop_index``), inclusive ``fit_start_time_s`` and
        ``fit_end_time_s`` (alias ``fit_stop_time_s``), ``fit_indices`` and
        ``fit_times_s``. Time bounds default to infinity. Explicit times select
        the nearest sample; a tie selects the first index. Negative explicit
        indices follow NumPy indexing. A negative end index follows Python
        slicing from the end; this differs from the other DCE model windows.

    Returns
    -------
    numpy.ndarray
        New boolean mask of shape ``(n_time,)`` in the input order. Inputs and
        options are not mutated; selection is eager and deterministic.

    Raises
    ------
    IndexError
        If an explicit sample index is outside the time array.
    TypeError
        If the squeezed input is scalar or a selector is not numeric.
    ValueError
        If selectors cannot be converted to their required numeric form.

    Examples
    --------
    >>> build_tofts_fit_mask(np.array([0., 2., 5.]), {"fit_end_index": -1})
    array([ True,  True, False])
    """
    fit_params = fit_params or {}
    fit_times = np.asarray(fit_times, dtype=np.float64).squeeze()
    fit_start_idx = int(fit_params.get("fit_start_index", 0) or 0)
    fit_end_idx = fit_params.get(
        "fit_end_index", fit_params.get("fit_stop_index", None)
    )
    fit_start_time = fit_params.get("fit_start_time_s", -np.inf)
    fit_end_time = fit_params.get(
        "fit_end_time_s", fit_params.get("fit_stop_time_s", np.inf)
    )
    # Tofts keeps Python slice semantics for a negative exclusive end index.
    if fit_end_idx is not None:
        fit_end_idx = int(fit_end_idx)
        if fit_end_idx < 0:
            fit_end_idx = max(len(fit_times) + fit_end_idx, 0)
    return _select_fit_samples(
        fit_times,
        start_index=max(fit_start_idx, 0),
        end_index=fit_end_idx,
        start_time_s=fit_start_time,
        end_time_s=fit_end_time,
        fit_indices=fit_params.get("fit_indices"),
        fit_times_s=fit_params.get("fit_times_s"),
    )


def fit_single_curve_tofts(
    signal_1d_values: np.ndarray,
    dynamic_coords_for_fit: np.ndarray,
    fit_params: dict,
    indices: tuple | None = None,
    signal_to_concentration_fn=None,
) -> dict:
    """Fit a single Tofts curve for a scalar voxel or an ROI-mean curve.

    The curve is fitted in concentration space.  When ``fit_params`` holds
    ``"t1_map"``, ``"s0_map"``, ``"tr_s"`` and ``"fa_deg"``, the curve is a
    signal and is first converted with ``signal_to_concentration_fn``.  The
    standard and the extended Tofts model are fitted on the samples selected
    by :func:`build_tofts_fit_mask`.  The inputs are not modified.

    Parameters
    ----------
    signal_1d_values : numpy.ndarray
        Tissue concentration in mM, or signal in arbitrary units for a
        signal-space fit, shape ``(n_time,)``.
    dynamic_coords_for_fit : numpy.ndarray
        Sample times in s, shape ``(n_time,)``.  Used when
        ``fit_params["time_points_s"]`` is missing or has another length.
    fit_params : dict
        Fit options.  ``"cp_func"`` (plasma concentration in mM as a function
        of time in s) is required.  Read keys are ``"model_mode"`` (default
        ``"both"``), ``"time_points_s"``, ``"aif_signal"``, ``"aif_times"``,
        ``"t1_map"`` (ms), ``"s0_map"``, ``"tr_s"``, ``"fa_deg"``, ``"r1"``
        in 1/(mM s) (default
        :data:`~hypermri.analysis.dce.conversions.DEFAULT_R1_PER_MM_S`)
        and the fit-window keys of :func:`build_tofts_fit_mask`.
    indices : tuple or None, optional
        Index of the voxel in ``"t1_map"`` and ``"s0_map"``.  None means that
        both maps already hold the single value for this curve.
    signal_to_concentration_fn : callable, optional
        Function ``(signal, s0, t10_s, tr_s, fa_deg, r1)`` that returns the
        concentration in mM.  Required for a signal-space fit.

    Returns
    -------
    dict
        ``"Ktrans_1c"``, ``"kep_1c"`` (1/s) and ``"rss_1c"`` (mM^2) for
        ``model_mode`` ``"1comp"`` or ``"both"``, and ``"Ktrans_2c"``,
        ``"kep_2c"`` (1/s), ``"vp_2c"``, ``"ve_2c"`` (fractions) and
        ``"rss_2c"`` (mM^2) for ``"2comp"`` or ``"both"``. ``ve_2c`` is
        clipped to [0, 1]; the public DCE dataset workflows separately report
        unbounded ``ve_equivalent`` and boolean ``ve_was_clipped`` after fitting.
        All values are NaN
        when the curve, time axis or AIF is not finite, ``"cp_func"`` is
        missing, the voxel T1 lies outside (0, 5000) ms, S0 is at most 1 or
        the converted concentration stays below 0.001 mM.

    Raises
    ------
    ValueError
        If a signal-space fit is requested without
        ``signal_to_concentration_fn``.
    """

    def empty_result(current_model_mode: str) -> dict:
        result = {}
        if current_model_mode in ["1comp", "both"]:
            result.update({"Ktrans_1c": np.nan, "kep_1c": np.nan, "rss_1c": np.nan})
        if current_model_mode in ["2comp", "both"]:
            result.update(
                {
                    "Ktrans_2c": np.nan,
                    "kep_2c": np.nan,
                    "vp_2c": np.nan,
                    "ve_2c": np.nan,
                    "rss_2c": np.nan,
                }
            )
        return result

    t1_map = fit_params.get("t1_map", None)
    s0_map = fit_params.get("s0_map", None)
    tr_s = fit_params.get("tr_s")
    fa_deg = fit_params.get("fa_deg")
    r1 = fit_params.get("r1", DEFAULT_R1_PER_MM_S)
    cp_func = fit_params.get("cp_func")
    model_mode = fit_params.get("model_mode", "both")
    nan_result = empty_result(model_mode)

    tps = fit_params.get("time_points_s")
    if tps is not None:
        tps_arr = np.asarray(tps).squeeze()
        if len(tps_arr) == len(dynamic_coords_for_fit):
            fit_times = tps_arr.astype(np.float64)
        else:
            fit_times = np.asarray(dynamic_coords_for_fit, dtype=np.float64)
    else:
        fit_times = np.asarray(dynamic_coords_for_fit, dtype=np.float64)

    c_t = np.asarray(signal_1d_values, dtype=np.float64)
    if not np.all(np.isfinite(c_t)) or not np.all(np.isfinite(fit_times)):
        return nan_result

    aif_signal = fit_params.get("aif_signal", None)
    aif_times = fit_params.get("aif_times", None)
    if aif_signal is not None and not np.all(np.isfinite(np.asarray(aif_signal))):
        return nan_result
    if aif_times is not None and not np.all(np.isfinite(np.asarray(aif_times))):
        return nan_result
    if cp_func is None:
        return nan_result

    if (
        t1_map is not None
        and s0_map is not None
        and tr_s is not None
        and fa_deg is not None
    ):
        if indices is None:
            t1v = (
                np.asarray(t1_map.values).squeeze()
                if hasattr(t1_map, "values")
                else np.asarray(t1_map).squeeze()
            )
            s0v = (
                np.asarray(s0_map.values).squeeze()
                if hasattr(s0_map, "values")
                else np.asarray(s0_map).squeeze()
            )
        elif isinstance(t1_map, np.ndarray) and t1_map.ndim != len(indices):
            try:
                t1v = t1_map[indices[-t1_map.ndim :]]
                s0v = s0_map[indices[-s0_map.ndim :]]
            except Exception:
                return nan_result
        else:
            t1v = t1_map[indices]
            s0v = s0_map[indices]

        t1v = np.asarray(t1v).squeeze()
        s0v = np.asarray(s0v).squeeze()

        if np.size(t1v) != 1 or np.size(s0v) != 1:
            return nan_result

        t1v = float(t1v)
        s0v = float(s0v)
        if np.isfinite(t1v) and np.isfinite(s0v) and t1v > 0 and t1v < 5000 and s0v > 1:
            if signal_to_concentration_fn is None:
                raise ValueError(
                    "signal_to_concentration_fn is required for signal-space Tofts fitting."
                )
            c_t = signal_to_concentration_fn(
                signal_1d_values, s0v, t1v / 1000.0, tr_s, fa_deg, r1
            )
            if not np.all(np.isfinite(c_t)):
                return nan_result
            if np.nanmax(c_t) < 0.001:
                return nan_result
        else:
            return nan_result

    valid_mask = build_tofts_fit_mask(fit_times, fit_params)
    fit_results = {}

    if model_mode in ["1comp", "both"]:
        p1, cost_1c = fit_single_voxel_tofts_helper(
            fit_times, c_t, cp_func, "1comp", fit_mask=valid_mask
        )
        fit_results["Ktrans_1c"] = p1[0]
        fit_results["kep_1c"] = p1[1]
        fit_results["rss_1c"] = cost_1c

    if model_mode in ["2comp", "both"]:
        p2, cost_2c = fit_single_voxel_tofts_helper(
            fit_times, c_t, cp_func, "2comp", fit_mask=valid_mask
        )
        fit_results["Ktrans_2c"] = p2[0]
        fit_results["kep_2c"] = p2[1]
        fit_results["vp_2c"] = p2[2]
        fit_results["ve_2c"] = (
            np.clip(p2[0] / p2[1], 0, 1)
            if np.isfinite(p2[0]) and np.isfinite(p2[1]) and p2[1] > 0
            else np.nan
        )
        fit_results["rss_2c"] = cost_2c

    return fit_results


def _reconstruct_tofts_curve_scalar(
    ktrans: float,
    kep: float,
    vp: float,
    selected_model: float,
    s0: float,
    t10_ms: float,
    time_points_s: np.ndarray,
    aif_signal: np.ndarray,
    aif_times: np.ndarray | None = None,
    tr_s: float | None = None,
    fa_deg: float | None = None,
    r1: float = DEFAULT_R1_PER_MM_S,
    reco_space: str = "concentration",
) -> np.ndarray:
    """Reconstruct one Tofts-family curve from scalar parameters."""
    if not np.all(np.isfinite([ktrans, kep, vp])):
        return np.full(len(time_points_s), np.nan, dtype=np.float64)
    if kep <= 0:
        return np.full(len(time_points_s), np.nan, dtype=np.float64)

    source_time = time_points_s if aif_times is None else np.asarray(aif_times)
    cp_func = interp1d(
        source_time,
        aif_signal,
        kind="linear",
        fill_value="extrapolate",
    )

    if selected_model == 2:
        concentration = extended_tofts_model_2comp(
            time_points_s,
            ktrans,
            kep,
            vp,
            cp_func,
        )
    else:
        concentration = tofts_model_1comp(
            time_points_s,
            ktrans,
            kep,
            cp_func,
        )

    if reco_space != "signal":
        return concentration

    if (
        any(val is None for val in (tr_s, fa_deg))
        or not np.isfinite([s0, t10_ms]).all()
    ):
        return np.full(len(time_points_s), np.nan, dtype=np.float64)

    return concentration_to_signal(
        concentration,
        float(s0),
        float(t10_ms) / 1000.0,
        float(tr_s),
        float(fa_deg),
        float(r1),
    )


def _spgr_baseline_input(
    value: xr.DataArray | float | None, *, name: str
) -> xr.DataArray | float:
    """Return a baseline map or scalar that broadcasts by dimension name."""
    if value is None:
        return np.nan
    if isinstance(value, xr.DataArray):
        return value
    if np.ndim(value) != 0:
        raise ValueError(
            f"tofts_model_reconstruction received {name} as an unlabelled array "
            f"of shape {np.shape(value)}. Pass {name} as a scalar or as an "
            "xarray.DataArray whose dimensions match the parameter maps."
        )
    return float(value)


def tofts_model_reconstruction(
    time_points_s: np.ndarray,
    aif_signal: np.ndarray,
    aif_times: np.ndarray = None,
    model_mode: str = "both",
    Ktrans_1c: xr.DataArray = None,
    kep_1c: xr.DataArray = None,
    Ktrans_2c: xr.DataArray = None,
    kep_2c: xr.DataArray = None,
    vp_2c: xr.DataArray = None,
    ve_2c: xr.DataArray = None,
    rss_1c: xr.DataArray = None,
    rss_2c: xr.DataArray = None,
    selected_model: xr.DataArray = None,
    s0: xr.DataArray | float = None,
    t10_ms: xr.DataArray | float = None,
    tr_s: float = None,
    fa_deg: float = None,
    r1: float = DEFAULT_R1_PER_MM_S,
    reco_space: str = "concentration",
) -> xr.DataArray:
    """Reconstruct concentration or signal curves from Tofts-family parameters.

    Each voxel is evaluated eagerly with the standard or the extended Tofts
    model on ``time_points_s``.  With ``model_mode="both"`` the model is
    chosen per voxel by ``selected_model``, else by the smaller residual sum
    of squares, else the standard model is used.

    Parameters
    ----------
    time_points_s : numpy.ndarray
        Strictly increasing sample times in s, shape ``(n_time,)``.  The
        spacing may be non-uniform.
    aif_signal : numpy.ndarray
        Plasma concentration in mM, shape ``(n_time,)``.
    aif_times : numpy.ndarray, optional
        Strictly increasing times of ``aif_signal`` in s.  The AIF is
        interpolated linearly, with extrapolation, onto ``time_points_s``.
        Defaults to ``time_points_s``.
    model_mode : {"1comp", "2comp", "both"}, default "both"
        Model to reconstruct.
    Ktrans_1c, kep_1c : xarray.DataArray, optional
        Standard Tofts parameter maps in 1/s.
    Ktrans_2c, kep_2c : xarray.DataArray, optional
        Extended Tofts parameter maps in 1/s.
    vp_2c : xarray.DataArray, optional
        Plasma volume fraction map of the extended model.
    ve_2c : xarray.DataArray, optional
        Extravascular extracellular volume fraction map.  Accepted with the
        fit results and not used.
    rss_1c, rss_2c : xarray.DataArray, optional
        Residual sums of squares in mM^2 used to choose the model per voxel
        when ``selected_model`` is missing.
    selected_model : xarray.DataArray, optional
        Model per voxel, 1 (standard) or 2 (extended).
    s0 : xarray.DataArray or float, optional
        Baseline signal in arbitrary units, as a scalar or as a map that
        broadcasts against the parameter maps by dimension name.  Signal
        space only.
    t10_ms : xarray.DataArray or float, optional
        Pre-contrast T1 in ms, as a scalar or as a map like ``s0``.  Signal
        space only.
    tr_s : float, optional
        Repetition time in s.  Signal space only.
    fa_deg : float, optional
        Flip angle in degrees.  Signal space only.
    r1 : float, default DEFAULT_R1_PER_MM_S
        Contrast-agent relaxivity in 1/(mM s), see
        :data:`~hypermri.analysis.dce.conversions.DEFAULT_R1_PER_MM_S`.
        Signal space only.
    reco_space : {"concentration", "signal"}, default "concentration"
        Return concentration in mM or SPGR signal in arbitrary units.

    Returns
    -------
    xarray.DataArray or None
        float64 curves with the dimensions of the parameter maps followed by
        ``repetition_idx`` and a ``time_s`` coordinate.  Voxels with
        non-finite parameters or ``kep <= 0``, voxels of a signal
        reconstruction with a missing or non-finite ``s0`` or ``t10_ms``, and
        all voxels of a signal reconstruction without ``tr_s`` or ``fa_deg``
        are NaN.  Dimensions of ``s0`` or ``t10_ms`` maps that the parameter
        maps lack are added.  None if the parameters of the requested model
        are missing.

    Raises
    ------
    ValueError
        If a time axis is not finite and strictly increasing, the AIF
        lengths do not match the time axes, or ``s0`` or ``t10_ms`` is an
        unlabelled array with more than zero dimensions.
    """
    time_points_s = _validate_dce_time_axis(
        name="time_points_s", values=np.asarray(time_points_s, dtype=np.float64)
    )
    aif_signal = np.asarray(aif_signal, dtype=np.float64).squeeze()
    if len(aif_signal) != len(time_points_s):
        raise ValueError("aif_signal and time_points_s must have the same length.")

    if aif_times is not None:
        aif_times = _validate_dce_time_axis(
            name="aif_times", values=np.asarray(aif_times, dtype=np.float64)
        )
        if len(aif_times) != len(aif_signal):
            raise ValueError("aif_times and aif_signal must have the same length.")

    time_dim_name = "repetition_idx"
    rep_coord = np.arange(len(time_points_s), dtype=int)
    base_coords = {
        time_dim_name: rep_coord,
        "time_s": (time_dim_name, time_points_s),
    }
    dask_kwargs = {"output_sizes": {time_dim_name: len(time_points_s)}}

    s0 = _spgr_baseline_input(s0, name="s0")
    t10_ms = _spgr_baseline_input(t10_ms, name="t10_ms")

    def _apply_model(ktrans, kep, vp, selector):
        # s0 and t10_ms are inputs, not kwargs, so that maps broadcast against
        # the parameter maps by dimension name and reach the voxel as scalars.
        return xr.apply_ufunc(
            _reconstruct_tofts_curve_scalar,
            ktrans,
            kep,
            vp,
            selector,
            s0,
            t10_ms,
            input_core_dims=[[], [], [], [], [], []],
            output_core_dims=[[time_dim_name]],
            vectorize=True,
            output_dtypes=[np.float64],
            dask="allowed",
            dask_gufunc_kwargs=dask_kwargs,
            kwargs={
                "time_points_s": time_points_s,
                "aif_signal": aif_signal,
                "aif_times": aif_times,
                "tr_s": tr_s,
                "fa_deg": fa_deg,
                "r1": r1,
                "reco_space": reco_space,
            },
        ).assign_coords(**base_coords)

    recon_1c = None
    recon_2c = None

    if model_mode in {"1comp", "both"} and Ktrans_1c is not None and kep_1c is not None:
        recon_1c = _apply_model(
            Ktrans_1c,
            kep_1c,
            xr.zeros_like(Ktrans_1c),
            xr.zeros_like(Ktrans_1c),
        )

    if (
        model_mode in {"2comp", "both"}
        and Ktrans_2c is not None
        and kep_2c is not None
        and vp_2c is not None
    ):
        recon_2c = _apply_model(
            Ktrans_2c,
            kep_2c,
            vp_2c,
            xr.ones_like(Ktrans_2c) * 2,
        )

    if model_mode == "1comp":
        return recon_1c
    if model_mode == "2comp":
        return recon_2c

    selector = selected_model
    if selector is None and rss_1c is not None and rss_2c is not None:
        selector = xr.where(rss_2c < rss_1c, 2, 1).astype(np.int8)

    if selector is None:
        selector = xr.zeros_like(Ktrans_1c if recon_1c is not None else Ktrans_2c) + 1

    if recon_1c is None:
        return recon_2c
    if recon_2c is None:
        return recon_1c
    return xr.where(selector == 2, recon_2c, recon_1c)


__all__ = [
    "ToftsFitOptions",
    "get_tofts_params_info",
    "prepare_tofts_fit_params_dataset",
    "tofts_model_1comp",
    "extended_tofts_model_2comp",
    "fit_single_voxel_tofts_helper",
    "build_tofts_fit_mask",
    "fit_single_curve_tofts",
    "tofts_model_reconstruction",
    "MODEL_ID",
    "RESULT_MODEL_NAME",
    "MODEL_EQUATION",
    "TOFTS_1COMP_BOUNDS",
    "TOFTS_2COMP_BOUNDS",
]


def _annotate_model_output(dataset: xr.Dataset):
    """Attach the existing model-specific descriptions in place."""
    for var_name, units, description in (
        (
            "Ktrans_1c",
            "s^-1",
            "Tofts 1-compartment transfer parameter. Do not equate this with GFR.",
        ),
        (
            "kep_1c",
            "s^-1",
            "Tofts 1-compartment efflux rate constant.",
        ),
        (
            "Ktrans_2c",
            "s^-1",
            "Extended Tofts transfer parameter. Do not equate this with GFR.",
        ),
        ("kep_2c", "s^-1", "Extended Tofts efflux rate constant."),
        ("vp_2c", "1", "Extended Tofts plasma volume fraction."),
        (
            "ve_2c",
            "1",
            "Extended Tofts extracellular volume fraction clipped to [0, 1]; "
            "ve_equivalent retains the unbounded ratio.",
        ),
    ):
        _set_var_attrs(
            dataset,
            var_name,
            long_name=var_name,
            description=description,
            units=units,
            model="tofts",
        )
    _set_var_attrs(
        dataset,
        "selected_tofts_model",
        long_name="Selected Tofts Submodel",
        description="1 for standard Tofts, 2 for extended Tofts when model_mode='both'.",
        units="index",
        model="tofts",
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
    aif_time_s = kwargs.get("aif_time_s")
    if aif_time_s is None:
        aif_time_s = select_reconstruction_array(
            data_to_reconstruct,
            "aif_time_s",
            voxel_coords,
        )

    if time_points_s is None or aif_signal is None:
        raise ValueError(
            "time_points_s (via `dynamic_dim` or `time_points_s`) and aif_signal must be provided via kwargs or exist in dataset variables for Tofts reconstruction."
        )

    time_points_s_np = (
        time_points_s.values
        if isinstance(time_points_s, xr.DataArray)
        else np.asarray(time_points_s)
    )
    aif_times_np = None
    if aif_time_s is not None:
        aif_times_np = (
            aif_time_s.values
            if isinstance(aif_time_s, xr.DataArray)
            else np.asarray(aif_time_s)
        )

    aif_signal_np = (
        aif_signal.values
        if isinstance(aif_signal, xr.DataArray)
        else np.asarray(aif_signal)
    )

    reco_space = str(kwargs.get("reco_space", "concentration")).lower()
    s0_val = None
    t10_val = None

    if reco_space == "signal":
        if "s0_map" in data_to_reconstruct:
            s0_val = data_to_reconstruct["s0_map"]
        elif "s0" in data_to_reconstruct:
            s0_val = data_to_reconstruct["s0"]

        if "t1_map" in data_to_reconstruct:
            t10_val = data_to_reconstruct["t1_map"]
        elif "t1" in data_to_reconstruct:
            t10_val = data_to_reconstruct["t1"]

    params = {
        "time_points_s": time_points_s_np,
        "aif_signal": aif_signal_np,
        "aif_times": aif_times_np,
        "model_mode": kwargs.get(
            "model_mode", data_to_reconstruct.attrs.get("model_mode", "both")
        ),
        "Ktrans_1c": data_to_reconstruct.get("Ktrans_1c"),
        "kep_1c": data_to_reconstruct.get("kep_1c"),
        "Ktrans_2c": data_to_reconstruct.get("Ktrans_2c"),
        "kep_2c": data_to_reconstruct.get("kep_2c"),
        "vp_2c": data_to_reconstruct.get("vp_2c"),
        "ve_2c": data_to_reconstruct.get("ve_2c"),
        "rss_1c": data_to_reconstruct.get("rss_1c"),
        "rss_2c": data_to_reconstruct.get("rss_2c"),
        "selected_model": data_to_reconstruct.get("selected_tofts_model"),
        "s0": s0_val,
        "t10_ms": t10_val,
        "tr_s": kwargs.get("tr_s", data_to_reconstruct.attrs.get("tr_s")),
        "fa_deg": kwargs.get("fa_deg", data_to_reconstruct.attrs.get("fa_deg")),
        "r1": kwargs.get(
            "r1", data_to_reconstruct.attrs.get("r1", DEFAULT_R1_PER_MM_S)
        ),
        "reco_space": reco_space,
    }
    return params


def _fit_voxel(signal, dynamic_coords, fit_params, indices):
    """Adapt this model to the existing shared voxel executor."""
    return (
        fit_single_curve_tofts(
            signal_1d_values=signal,
            dynamic_coords_for_fit=dynamic_coords,
            fit_params=fit_params,
            indices=indices,
        ),
        indices,
    )


def _output_param_names(params: dict):
    """Return the existing ordered output names for these options."""
    model_mode = str(params.get("model_mode", "both")).lower()
    if model_mode == "1comp":
        return ["Ktrans_1c", "kep_1c", "rss_1c"]
    if model_mode == "2comp":
        return ["Ktrans_2c", "kep_2c", "vp_2c", "ve_2c", "rss_2c"]
    return [
        "Ktrans_1c",
        "kep_1c",
        "rss_1c",
        "Ktrans_2c",
        "kep_2c",
        "vp_2c",
        "ve_2c",
        "rss_2c",
    ]


def _attach_fit_options(output_dataset: xr.Dataset, fit_params: dict):
    """Record this model's existing effective reconstruction options."""
    output_dataset.attrs["model_mode"] = fit_params.get("model_mode", "both")
    output_dataset.attrs["tr_s"] = fit_params.get("tr_s")
    output_dataset.attrs["fa_deg"] = fit_params.get("fa_deg")
    output_dataset.attrs["r1"] = fit_params.get("r1", DEFAULT_R1_PER_MM_S)
    output_dataset.attrs["fit_start_index"] = fit_params.get("fit_start_index", 0)
    output_dataset.attrs["fit_end_index"] = fit_params.get("fit_end_index")
    output_dataset.attrs["aif_start_index"] = fit_params.get("aif_start_index", 0)
    output_dataset.attrs["aif_end_index"] = fit_params.get("aif_end_index")


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
    model_mode = str(
        fit_results.attrs.get("model_mode", fit_params.get("model_mode", "both"))
    ).lower()
    tol = 1e-6
    selected_tofts_model = None
    if model_mode == "1comp":
        parameter_count = 2
        finite_params = np.isfinite(fit_results["Ktrans_1c"]) & np.isfinite(
            fit_results["kep_1c"]
        )
        hit_bound = _parameters_at_bounds(
            fit_results, _TOFTS_1COMP_NAMES, TOFTS_1COMP_BOUNDS, atol=tol
        )
    elif model_mode == "2comp":
        parameter_count = 3
        finite_params = (
            np.isfinite(fit_results["Ktrans_2c"])
            & np.isfinite(fit_results["kep_2c"])
            & np.isfinite(fit_results["vp_2c"])
        )
        hit_bound = _parameters_at_bounds(
            fit_results, _TOFTS_2COMP_NAMES, TOFTS_2COMP_BOUNDS, atol=tol
        )
    else:
        selected_tofts_model = xr.where(
            fit_results["rss_2c"] < fit_results["rss_1c"], 2, 1
        ).astype(np.int8)
        parameter_count = xr.where(selected_tofts_model == 2, 3, 2).astype(np.int8)
        finite_1c = np.isfinite(fit_results["Ktrans_1c"]) & np.isfinite(
            fit_results["kep_1c"]
        )
        finite_2c = (
            np.isfinite(fit_results["Ktrans_2c"])
            & np.isfinite(fit_results["kep_2c"])
            & np.isfinite(fit_results["vp_2c"])
        )
        hit_bound_1c = _parameters_at_bounds(
            fit_results, _TOFTS_1COMP_NAMES, TOFTS_1COMP_BOUNDS, atol=tol
        )
        hit_bound_2c = _parameters_at_bounds(
            fit_results, _TOFTS_2COMP_NAMES, TOFTS_2COMP_BOUNDS, atol=tol
        )
        finite_params = xr.where(selected_tofts_model == 2, finite_2c, finite_1c)
        hit_bound = xr.where(selected_tofts_model == 2, hit_bound_2c, hit_bound_1c)

    prediction = tofts_model_reconstruction(
        np.asarray(aif_time_s, dtype=np.float64),
        np.asarray(aif_concentration, dtype=np.float64),
        aif_times=np.asarray(aif_time_s, dtype=np.float64),
        model_mode=model_mode,
        Ktrans_1c=fit_results.get("Ktrans_1c"),
        kep_1c=fit_results.get("kep_1c"),
        Ktrans_2c=fit_results.get("Ktrans_2c"),
        kep_2c=fit_results.get("kep_2c"),
        vp_2c=fit_results.get("vp_2c"),
        ve_2c=fit_results.get("ve_2c"),
        rss_1c=fit_results.get("rss_1c"),
        rss_2c=fit_results.get("rss_2c"),
        selected_model=selected_tofts_model,
    )
    prediction = prediction.assign_coords(repetition_idx=rep_coord)
    if "time_s" in concentration_map.coords:
        prediction = prediction.assign_coords(time_s=concentration_map.coords["time_s"])
    return prediction, parameter_count, finite_params, hit_bound, selected_tofts_model


_prepare_fit_params = prepare_tofts_fit_params_dataset
_describe_params = get_tofts_params_info
_reconstruct = tofts_model_reconstruction
_OUTPUT_DTYPES = {}
_DEFAULT_MODEL_MODE = "both"
_TOFTS_1COMP_NAMES = ("Ktrans_1c", "kep_1c")
_TOFTS_2COMP_NAMES = ("Ktrans_2c", "kep_2c", "vp_2c")


def _quality_window(times, plasma, params):
    return build_tofts_fit_mask(times, params)


def _attach_reconstruction_arrays(output_dataset, fit_params, dynamic_dim_name, data):
    """Attach the existing seconds and plasma concentration arrays."""
    _attach_dce_reconstruction_arrays(
        output_dataset,
        fit_params,
        dynamic_dim_name,
        data,
        aif_times_key="aif_times",
    )


def _finalize_pipeline_metadata(final_ds, fit_params_pk):
    """Retain model-specific variant and equation-equivalence attributes."""
    if str(final_ds.attrs.get("fit_model_mode")).lower() == "2comp":
        final_ds.attrs.setdefault(
            "equation_equivalent_fit_types",
            "extended_tofts_plasma,kidney_two_compartment_outflow,sourbron",
        )
        final_ds.attrs.setdefault(
            "fit_type_equivalence_note",
            "The implemented extended Tofts 2-compartment form is mathematically equivalent to the Sourbron parameterization and the deprecated kidney_two_compartment_outflow alias.",
        )


_KEEP_LEGACY_MODEL_MODE = True


def _effective_model_mode(fit_params, default_model_mode):
    return str(fit_params.get("model_mode", default_model_mode)).lower()
