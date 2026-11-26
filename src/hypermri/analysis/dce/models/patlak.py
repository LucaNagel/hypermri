"""Patlak concentration model, graphical regression and scientific options.

Times are seconds and concentrations are mM. Regression RSS is calculated
after AIF normalization; concentration-space quality RSS is a separate metric.
Functions preserve inputs and retain the existing model-specific fit window.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import xarray as xr
from scipy.integrate import cumulative_trapezoid

from hypermri.analysis.dce.conversions import DEFAULT_R1_PER_MM_S
from hypermri.analysis.dce.fit_metrics import _sum_squared_residuals
from hypermri.analysis.dce.fit_window import _select_fit_samples
from hypermri.analysis.fitting.execution import copy_execution_params
from hypermri.analysis.fitting.result_metadata import select_reconstruction_array

from ..fit_quality import _dataarray_isclose, _parameters_at_bounds
from ..result_metadata import _attach_dce_reconstruction_arrays, _set_var_attrs


@dataclass(frozen=True)
class PatlakFitOptions:
    """Immutable scientific overrides for the patlak curve fit.

    None leaves a value unspecified: the existing fitter remains the sole owner
    of scientific defaults and data-dependent guesses/bounds. Construction does
    not fit, read data or allocate a cache; supplied sequences become tuples.

    Parameters
    ----------
    fit_start_index : int or float or None, optional
        First zero-based sample index; included.
    fit_end_index : int or float or None, optional
        Exclusive end index. Negative ends retain this model's historical rule.
    fit_start_time_s : float | None, optional
        Inclusive start time in seconds.
    fit_end_time_s : float | None, optional
        Inclusive end time in seconds.
    aif_threshold_mm : float | None, optional
        Use samples with plasma concentration strictly above this threshold in mM.

    Notes
    -----
    Patlak ignores explicit index/time lists and uses absolute index ends. It
    has no configurable regression bounds; its existing clipping of negative
    K_trans after graphical RSS calculation is preserved.

    Examples
    --------
    >>> PatlakFitOptions(fit_start_time_s=5.0).to_fit_params()
    {'fit_start_time_s': 5.0}
    """

    fit_start_index: int | float | None = None
    fit_end_index: int | float | None = None
    fit_start_time_s: float | None = None
    fit_end_time_s: float | None = None
    aif_threshold_mm: float | None = None

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
            "aif_threshold": self.aif_threshold_mm,
        }
        return {name: value for name, value in values.items() if value is not None}


MODEL_ID = "patlak"
RESULT_MODEL_NAME = "patlak"
MODEL_EQUATION = "Ct(T) = K_trans * integral(Cp(t)dt) + Vp * Cp(T)"
OUTPUT_PARAM_NAMES = ["K_trans", "Vp", "rss_patlak"]


def get_patlak_params_info() -> dict[str, object]:
    """Return descriptions of the existing Patlak inputs and outputs.

    Returns
    -------
    dict[str, object]
        Fresh mappings for required/optional fit parameters, output variables and
        the concentration model equation. Time is seconds, concentration is mM,
        K_trans is 1/s, Vp is a fraction and graphical RSS is dimensionless.
        No fitting, input mutation, data reads or caching occur.
    """
    return {
        "required_fit_params": {
            "aif_signal": "Arterial input function (AIF) sampled at the same time points as input_data",
            "time_points_s": "Time axis in seconds corresponding to the dynamic acquisition",
        },
        "optional_fit_params": {
            "fit_start_time_s": "Start time (in seconds) for fitting window. Default: -inf (all times)",
            "fit_stop_time_s": "Stop time (in seconds) for fitting window. Default: inf (all times)",
            "fit_start_index": "Start index for fitting window. Default: 0",
            "fit_stop_index": "Stop index (exclusive) for fitting window. Default: None (all indices)",
            "aif_threshold": "Minimum AIF value to include in fit (avoids division by small numbers). Default: 1e-6",
            "use_multiprocessing": "Enable parallel processing. Default: True",
            "number_of_cpu_cores": "Number of CPU cores for parallel processing (-1 uses all). Default: -1",
        },
        "output_parameters": {
            "K_trans": "Transfer constant from plasma to tissue (s^-1 when time_points_s is provided in seconds)",
            "Vp": "Plasma volume fraction",
            "rss_patlak": "Residual sum of squares for the linear regression",
        },
        "model_equation": MODEL_EQUATION,
    }


def prepare_patlak_fit_params_dataset(
    data: xr.DataArray,
    fit_params: dict | None = None,
    return_params_info: bool = False,
    *,
    base_attrs: dict[str, object] | None = None,
) -> xr.Dataset | None:
    """Prepare the existing Patlak fit-parameter dataset.

    Parameters
    ----------
    data : xarray.DataArray
        Real concentration in mM with a dynamic dimension named by ``dynamic_dim``
        (default ``repetition_idx``) and arbitrary labelled spatial dimensions.
    fit_params : dict or None, optional
        Scientific and execution options. ``aif_signal`` is required in mM;
        ``time_points_s`` is seconds and defaults to the dynamic coordinate.
        The input mapping is preserved. Complex data emits a warning for Patlak.
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
    >>> prepared = prepare_patlak_fit_params_dataset(
    ...     concentrations, {"aif_signal": plasma_mm, "time_points_s": times_s}
    ... )  # doctest: +SKIP
    """
    if return_params_info:
        return None
    fit_params = fit_params or {}
    aif_signal = fit_params.get("aif_signal", None)
    if aif_signal is None:
        raise ValueError(
            "No AIF signal (arterial input function) provided for Patlak fitting. "
            "Please provide 'aif_signal' in fit_params."
        )

    attrs = dict(base_attrs or {})
    dynamic_dim = fit_params.get("dynamic_dim", "repetition_idx")
    time_points_s = fit_params.get("time_points_s", data.coords.get(dynamic_dim, None))
    attrs.update(
        {
            "fit_type": "patlak",
            "model_id": "patlak",
            "model_name": RESULT_MODEL_NAME,
            "dynamic_dim": dynamic_dim,
            "time_points_s": time_points_s,
            "aif_signal": aif_signal,
            "fit_start_time_s": fit_params.get("fit_start_time_s", -np.inf),
            "fit_stop_time_s": fit_params.get("fit_stop_time_s", np.inf),
            "fit_start_index": fit_params.get("fit_start_index", 0),
            "fit_stop_index": fit_params.get("fit_stop_index", None),
            "aif_threshold": fit_params.get("aif_threshold", 1e-6),
        }
    )
    copy_execution_params(fit_params, attrs)

    if np.iscomplexobj(data.values):
        warnings.warn(
            "Input data for Patlak fitting is complex-valued. The Patlak model expects "
            "real-valued concentration data. Consider taking the magnitude or real part "
            "of the signal before fitting.",
            UserWarning,
            stacklevel=2,
        )

    return xr.Dataset(attrs=attrs)


def _build_patlak_fit_mask(
    time_points_s: np.ndarray,
    aif_signal: np.ndarray,
    fit_params: Mapping[str, object] | None = None,
) -> np.ndarray:
    """Select Patlak regression samples for both fitting and quality metrics."""
    fit_params = fit_params or {}
    mask = _select_fit_samples(
        time_points_s,
        start_index=fit_params.get("fit_start_index", 0),
        end_index=fit_params.get(
            "fit_end_index", fit_params.get("fit_stop_index", None)
        ),
        start_time_s=fit_params.get("fit_start_time_s", -np.inf),
        end_time_s=fit_params.get(
            "fit_end_time_s", fit_params.get("fit_stop_time_s", np.inf)
        ),
    )
    mask &= aif_signal > fit_params.get("aif_threshold", 1e-6)
    return mask


def fit_single_voxel_patlak_helper(
    signal: np.ndarray = None,
    aif_signal: np.ndarray = None,
    time_points_s: np.ndarray = None,
    fit_params: dict | PatlakFitOptions | None = None,
):
    """Fit a Patlak concentration curve by graphical linear regression.

    The AIF integral uses the full acquisition time axis before selecting fit
    samples. Regression and RSS are computed after division by the AIF, so RSS
    is dimensionless and differs from concentration-space quality-map RSS.

    Parameters
    ----------
    signal : numpy.ndarray, optional
        Real tissue concentration in mM, shape ``(n_time,)`` after squeezing.
    aif_signal : numpy.ndarray, optional
        Plasma concentration in mM on the same samples.
    time_points_s : numpy.ndarray, optional
        Increasing acquisition times in seconds on the same samples.
    fit_params : dict or PatlakFitOptions or None, optional
        ``signal``, ``aif_signal`` and ``time_points_s`` override the arguments.
        Index bounds ``fit_start_index`` and exclusive ``fit_end_index`` (alias
        ``fit_stop_index``) and inclusive time bounds ``fit_start_time_s`` and
        ``fit_end_time_s`` (alias ``fit_stop_time_s``) select the window.
        Only samples with AIF above ``aif_threshold`` (default 1e-6 mM) are
        used. Negative end indices select no samples; ``fit_indices`` and
        ``fit_times_s`` are not read by this model. The internal
        ``_allow_integer_second_time_axis`` option permits times equal to
        consecutive integer indices when their seconds are already established.

    Returns
    -------
    dict
        ``K_trans`` in 1/s, vascular fraction ``Vp`` and dimensionless graphical
        ``rss_patlak``. Values are NaN for non-finite inputs, fewer than two
        selected samples or a rank-deficient/failed regression. Negative fitted
        K_trans is clipped to zero after calculating the original regression RSS.
        Inputs are not mutated; calculation is eager and deterministic.

    Raises
    ------
    ValueError
        If times are missing, appear to be unconverted integer indices, or
        sample arrays have incompatible lengths.
    TypeError
        If a required concentration is missing from both the argument and
        fit_params, or a squeezed input is scalar instead of 1D.

    Examples
    --------
    >>> result = fit_single_voxel_patlak_helper(tissue, plasma, times_s)  # doctest: +SKIP
    """
    if isinstance(fit_params, PatlakFitOptions):
        fit_params = fit_params.to_fit_params()
    fit_params = fit_params or {}
    signal = fit_params.get("signal", signal)
    aif_signal = fit_params.get("aif_signal", aif_signal)
    time_points_s = fit_params.get("time_points_s", time_points_s)
    allow_integer_second_time_axis = fit_params.get(
        "_allow_integer_second_time_axis", False
    )

    if time_points_s is None:
        raise ValueError(
            "time_points_s is required for Patlak fitting but was not provided. "
            "Please provide time values in seconds via fit_params['time_points_s']."
        )

    signal = np.asarray(signal).squeeze()
    aif_signal = np.asarray(aif_signal).squeeze()
    time_points_s = np.asarray(time_points_s).squeeze()
    if (
        not np.all(np.isfinite(signal))
        or not np.all(np.isfinite(aif_signal))
        or not np.all(np.isfinite(time_points_s))
    ):
        return {"K_trans": np.nan, "Vp": np.nan, "rss_patlak": np.nan}
    if len(time_points_s) > 2:
        if not allow_integer_second_time_axis and np.allclose(
            time_points_s, np.arange(len(time_points_s)), rtol=0.01
        ):
            raise ValueError(
                "time_points_s appears to contain indices (0, 1, 2, ...) rather than actual "
                "time values in seconds. The Patlak model requires time values in seconds. "
                "If your coordinate contains indices, multiply by the repetition time (TR) or "
                "provide the actual acquisition times in seconds."
            )

    valid_mask = _build_patlak_fit_mask(time_points_s, aif_signal, fit_params)

    if np.sum(valid_mask) < 2:
        return {"K_trans": np.nan, "Vp": np.nan, "rss_patlak": np.nan}

    aif_filt = aif_signal[valid_mask]
    signal_filt = signal[valid_mask]
    integral_aif = cumulative_trapezoid(aif_signal, time_points_s, initial=0)
    integral_aif_filt = integral_aif[valid_mask]

    patlak_x = integral_aif_filt / aif_filt
    patlak_y = signal_filt / aif_filt
    design = np.vstack([np.ones_like(patlak_x, dtype=float), patlak_x.astype(float)]).T

    try:
        coeffs, _, rank, _ = np.linalg.lstsq(design, patlak_y, rcond=None)

        if rank < 2:
            return {"K_trans": np.nan, "Vp": np.nan, "rss_patlak": np.nan}

        vp_val = coeffs[0]
        ki_val = coeffs[1]
        predicted_patlak_y = design @ coeffs
        rss = _sum_squared_residuals(patlak_y - predicted_patlak_y)

        if ki_val < 0:
            ki_val = 0.0

        return {"K_trans": ki_val, "Vp": vp_val, "rss_patlak": rss}

    except np.linalg.LinAlgError:
        return {"K_trans": np.nan, "Vp": np.nan, "rss_patlak": np.nan}


def patlak_model(
    time_points_s: np.ndarray,
    aif_signal: np.ndarray,
    k_trans: xr.DataArray,
    vp: xr.DataArray,
) -> xr.DataArray:
    """Reconstruct labelled concentration curves from Patlak parameters.

    Parameters
    ----------
    time_points_s : numpy.ndarray
        Acquisition times in seconds, shape ``(n_time,)`` in acquisition order.
    aif_signal : numpy.ndarray
        Plasma concentration in mM on the same acquisition samples.
    k_trans : xarray.DataArray
        Transfer-rate parameter map in 1/s.
    vp : xarray.DataArray
        Dimensionless vascular-fraction parameter map.

    Returns
    -------
    xarray.DataArray
        Tissue concentration in mM. Parameter maps broadcast in xarray order,
        followed by ``repetition_idx`` whose values are the supplied seconds.
        The AIF integral uses cumulative trapezoidal integration over the full
        axis. Floating-point precision follows the inputs and SciPy/xarray
        arithmetic. Evaluation is eager, deterministic and preserves inputs.

    Raises
    ------
    ValueError
        If sample lengths differ or parameter coordinates cannot align.

    Examples
    --------
    >>> prediction = patlak_model(times_s, plasma_mm, k_trans_map, vp_map)  # doctest: +SKIP
    """
    time_dim_name = "repetition_idx"
    time_da = xr.DataArray(
        time_points_s, dims=[time_dim_name], coords={time_dim_name: time_points_s}
    )
    aif_da = xr.DataArray(
        aif_signal, dims=[time_dim_name], coords={time_dim_name: time_points_s}
    )

    integral_aif = cumulative_trapezoid(aif_signal, time_points_s, initial=0)
    integral_aif_da = xr.DataArray(
        integral_aif, dims=[time_dim_name], coords=time_da.coords
    )

    return k_trans * integral_aif_da + vp * aif_da


__all__ = [
    "PatlakFitOptions",
    "get_patlak_params_info",
    "prepare_patlak_fit_params_dataset",
    "fit_single_voxel_patlak_helper",
    "patlak_model",
    "MODEL_ID",
    "RESULT_MODEL_NAME",
    "MODEL_EQUATION",
    "OUTPUT_PARAM_NAMES",
]


def _annotate_model_output(dataset: xr.Dataset):
    """Attach the existing model-specific descriptions in place."""
    description = (
        "Patlak slope / uptake parameter fitted on concentration curves. "
        "It is filtration-related but is not a direct GFR estimate."
    )
    _set_var_attrs(
        dataset,
        "K_trans",
        long_name="Patlak Slope",
        description=description,
        units="s^-1",
        model="patlak",
    )
    _set_var_attrs(
        dataset,
        "Vp",
        long_name="Patlak Intercept",
        description="Patlak intercept representing the vascular/plasma term in the linear model.",
        units="1",
        model="patlak",
    )
    _set_var_attrs(
        dataset,
        "patlak_slope",
        long_name="Patlak Slope",
        description=description,
        units="s^-1",
        model="patlak",
    )
    _set_var_attrs(
        dataset,
        "patlak_intercept",
        long_name="Patlak Intercept",
        description="Renal-facing alias for the Patlak intercept term.",
        units="1",
        model="patlak",
    )
    _set_var_attrs(
        dataset,
        "filtration_index",
        long_name="Filtration Index",
        description=(
            "Renal-facing alias of the Patlak slope. "
            "It should be treated as a filtration-related uptake parameter, not direct GFR."
        ),
        units="s^-1",
        model="patlak",
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
            "time_points_s (via `dynamic_dim` or `time_points_s`) and aif_signal must be provided via kwargs or exist in dataset variables."
        )

    time_points_s_np = (
        time_points_s.values
        if isinstance(time_points_s, xr.DataArray)
        else np.asarray(time_points_s)
    )
    aif_signal_np = (
        aif_signal.values
        if isinstance(aif_signal, xr.DataArray)
        else np.asarray(aif_signal)
    )

    params = {
        "k_trans": data_to_reconstruct["K_trans"],
        "vp": data_to_reconstruct["Vp"],
        "time_points_s": time_points_s_np,
        "aif_signal": aif_signal_np,
    }
    return params


def _fit_voxel(signal, dynamic_coords, fit_params, indices):
    """Adapt this model to the existing shared voxel executor."""
    return (
        fit_single_voxel_patlak_helper(signal=signal, fit_params=fit_params),
        indices,
    )


def _output_param_names(params: dict):
    """Return the existing ordered Patlak result variables."""
    return OUTPUT_PARAM_NAMES


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
    prediction = patlak_model(
        time_points_for_fit,
        np.asarray(aif_concentration, dtype=np.float64),
        fit_results["K_trans"],
        fit_results["Vp"],
    )
    prediction = prediction.assign_coords(repetition_idx=rep_coord)
    if "time_s" in concentration_map.coords:
        prediction = prediction.assign_coords(time_s=concentration_map.coords["time_s"])
    parameter_count = 2
    finite_params = np.isfinite(fit_results["K_trans"]) & np.isfinite(fit_results["Vp"])
    hit_bound = xr.zeros_like(fit_results["K_trans"], dtype=bool)
    selected_tofts_model = None
    return prediction, parameter_count, finite_params, hit_bound, selected_tofts_model


_prepare_fit_params = prepare_patlak_fit_params_dataset
_describe_params = get_patlak_params_info
_reconstruct = patlak_model
_OUTPUT_DTYPES = {}
_DEFAULT_MODEL_MODE = "1comp"


def _quality_window(times, plasma, params):
    return _build_patlak_fit_mask(times, plasma, params)


def _attach_reconstruction_arrays(output_dataset, fit_params, dynamic_dim_name, data):
    """Attach the existing seconds and plasma concentration arrays."""
    _attach_dce_reconstruction_arrays(
        output_dataset,
        fit_params,
        dynamic_dim_name,
        data,
        aif_times_key="time_points_s",
    )


_KEEP_LEGACY_MODEL_MODE = True
