"""Prepare and fit inversion-recovery T1 grids with one scientific solver.

Public preparation converts trial T1 values from seconds to float64 milliseconds;
inversion-time coordinates and fitted T1 use milliseconds. Supplied weighting,
coefficient bounds and regularization reach the existing grid-search solver.
Omitted settings retain that solver's defaults. Importing or preparing parameters
fits no signal, changes no acquired data and retains no reconstruction/fit cache.
"""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import xarray as xr
from scipy.optimize import lsq_linear

from hypermri.analysis.fitting.execution import copy_execution_params

MODEL_ID = "t1"
RESULT_MODEL_NAME = "t1_inversion_recovery"
T1_GRID_INPUT_UNIT = "s"
T1_OUTPUT_UNIT = "ms"
INVERSION_TIME_UNIT = "ms"
MODEL_EQUATION = "S(t_ms) = A * (1 - 2 * exp(-t_ms / T1_ms)) + B"
OUTPUT_PARAM_NAMES = ["t1", "A", "B", "rss_t1"]


def _t1_grid_seconds_to_milliseconds(t1_values_s: object) -> np.ndarray:
    """Convert the public T1 search grid from seconds to model milliseconds."""
    return np.atleast_1d(np.asarray(t1_values_s, dtype=np.float64)) * 1000.0


def get_t1_params_info() -> dict[str, object]:
    """Describe the public T1 settings, outputs and millisecond model equation.

    Returns
    -------
    dict of str to object
        New mapping describing grid units, supported scientific/execution options
        and output units. Coefficients have input signal units; RSS has squared
        signal units, with dimensionless weights when requested.

    Notes
    -----
    Deterministic metadata only: no signal read, fitting, mutation or caching.
    """
    return {
        "required_fit_params": {
            "t1_values_s": (
                "Array of T1 values in seconds to test during grid search. "
                "The grid is converted to milliseconds internally; the fitted "
                "'t1' output is reported in milliseconds."
            ),
        },
        "optional_fit_params": {
            "fit_baseline": "If True, fits baseline parameter B in model S(t) = A*(1-2*exp(-t/T1)) + B. Default: False",
            "weight_by_signal": "Weight squared residuals by abs(signal)/(max(abs(signal))+1e-8). This signal-magnitude heuristic is not a calibrated noise likelihood. Default: False",
            "bounds_A": "Lower/upper amplitude bounds in input signal units; strictly ordered, with infinities allowed. Complex real/imaginary coefficients use these bounds separately. Default: unbounded",
            "bounds_B": "Lower/upper baseline bounds in input signal units, applied when fit_baseline=True. Default: unbounded",
            "rel_bounds_A": "Dimensionless scalar r gives (1-r, 1+r)*max(abs(signal)); a pair is multiplied directly by that magnitude. Overrides bounds_A. Default: None",
            "rel_bounds_B": "Dimensionless scalar r gives (-r, r)*max(abs(signal)); a pair is multiplied directly by that magnitude. Overrides bounds_B. Default: None",
            "auto_calc_rel_bound": "Use the existing inversion-recovery heuristic for relative A/B bounds, overriding supplied relative bounds. Default: False",
            "inversion_imperfection": "Dimensionless inversion-imperfection setting used by automatic bounds. Default: 0.1",
            "regularization_alpha": "Positive dimensionless ridge coefficient for all fitted coefficients when bounds are unbounded; any finite bound selects bounded least squares instead. Non-positive values retain ordinary least squares. Default: 0.0",
            "use_multiprocessing": "Enable parallel processing. Default: True",
            "number_of_cpu_cores": "Number of CPU cores for parallel processing (-1 uses all). Default: -1",
        },
        "output_parameters": {
            "t1": "Longitudinal relaxation time (milliseconds)",
            "A": "Signal amplitude parameter (can be complex if input is complex)",
            "B": "Baseline offset parameter (can be complex if input is complex, 0 if fit_baseline=False)",
            "rss_t1": "Residual sum of squares in squared signal units, weighted if requested; excludes the regularization penalty",
        },
        "model_equation": MODEL_EQUATION,
    }


def prepare_t1_fit_params_dataset(
    data: xr.DataArray,
    fit_params: dict | None = None,
    return_params_info: bool = False,
    *,
    base_attrs: dict[str, object] | None = None,
) -> xr.Dataset | None:
    """Prepare T1 grid metadata and preserve supported supplied solver settings.

    Parameters
    ----------
    data : xarray.DataArray
        Real or complex signal with a sample dimension, normally ``inversion_idx``.
        Its coordinate provides inversion times in ms in the original order;
        remaining labelled dimensions identify voxels. Signal values are not read.
    fit_params : dict or None, optional
        Supplied settings. ``t1_values_s`` is a one-dimensional trial grid in
        seconds, converted to float64 ms exactly once; omitted grid retains
        500 trials from 0.01 to 10 s. ``dynamic_dim`` selects the sample coordinate;
        ``inversion_times_ms`` can supply explicit ms metadata. ``fit_baseline``
        defaults to False. Supported weighting, absolute/relative A/B bounds,
        automatic bounds, inversion imperfection and ridge regularization are
        described by :func:`get_t1_params_info` and copied only when supplied.
        Omitted scientific settings remain owned by the solver. Worker settings
        follow the shared execution contract. The mapping is not modified.
    return_params_info : bool, optional
        Return None immediately without constructing parameters or fitting.
        Use :func:`get_t1_params_info` for the information mapping.
    base_attrs : dict of str to object or None, optional
        Initial metadata copied into the result; model/grid fields take precedence.

    Returns
    -------
    xarray.Dataset or None
        Attribute-only dataset containing model identifiers, float64 seconds/ms
        grids, inversion-time labels, units and supplied scientific/worker options.
        New supported option values are deep-copied. None when information-only
        behavior is requested. No signal transformation, numerical fit or cache.

    Raises
    ------
    TypeError, ValueError
        If trial-grid values cannot be converted to a numerical float64 array.
        The solver checks applicable coefficient bounds during fitting.

    Notes
    -----
    Preparation is deterministic and metadata-only. Relative bounds override
    absolute bounds; automatic bounds override supplied relative bounds. Finite
    bounds select the existing bounded solver rather than ridge regularization.
    No phase correction or scientific defaults are introduced here.

    Examples
    --------
    >>> params = prepare_t1_fit_params_dataset(
    ...     data, {"t1_values_s": [0.5, 0.7, 0.9], "weight_by_signal": True}
    ... )  # doctest: +SKIP
    """
    if return_params_info:
        return None
    fit_params = fit_params or {}
    attrs = dict(base_attrs or {})
    t1_values_s = np.atleast_1d(
        np.asarray(
            fit_params.get("t1_values_s", np.linspace(0.01, 10, 500)),
            dtype=np.float64,
        )
    )
    t1_values_ms = _t1_grid_seconds_to_milliseconds(t1_values_s)
    dynamic_dim = fit_params.get("dynamic_dim", "inversion_idx")
    attrs.update(
        {
            "fit_type": "t1",
            "model_id": "t1",
            "model_name": RESULT_MODEL_NAME,
            "dynamic_dim": dynamic_dim,
            "t1_values_s": t1_values_s,
            "t1_values_ms": t1_values_ms,
            "t1_grid_input_units": T1_GRID_INPUT_UNIT,
            "t1_output_units": T1_OUTPUT_UNIT,
            "inversion_time_units": INVERSION_TIME_UNIT,
            "fit_baseline": fit_params.get("fit_baseline", False),
            "inversion_times_ms": fit_params.get(
                "inversion_times_ms", data.coords.get(dynamic_dim, None)
            ),
        }
    )
    for name in (
        "weight_by_signal",
        "bounds_A",
        "bounds_B",
        "rel_bounds_A",
        "rel_bounds_B",
        "auto_calc_rel_bound",
        "inversion_imperfection",
        "regularization_alpha",
    ):
        if name in fit_params:
            attrs[name] = deepcopy(fit_params[name])
    copy_execution_params(fit_params, attrs)
    return xr.Dataset(attrs=attrs)


def _find_t1_bounds(
    signal: np.ndarray = None,
    inversion_times_ms: np.ndarray = None,
    t1_values_ms: np.ndarray = None,
    fit_params: dict | None = None,
):
    """Find appropriate A/B bounds for the inversion-recovery model."""
    fit_params = fit_params or {}
    signal = fit_params.get("signal", signal)
    auto_calc_rel_bound = fit_params.get("auto_calc_rel_bound", False)
    bounds_B = fit_params.get("bounds_B", (-np.inf, np.inf))
    bounds_A = fit_params.get("bounds_A", (-np.inf, np.inf))

    inversion_imperfection = fit_params.get("inversion_imperfection", 0.1)
    fit_params["inversion_imperfection"] = inversion_imperfection

    rel_bounds_B = fit_params.get("rel_bounds_B", None)
    rel_bounds_A = fit_params.get("rel_bounds_A", None)

    signal = np.asarray(signal).squeeze()
    s_ref = np.max(np.abs(signal)) if signal.size > 0 else 1.0

    if auto_calc_rel_bound:
        timax = np.max(inversion_times_ms)
        t1max = np.max(t1_values_ms)

        # "Worst case": Maximum possible signal recovery at the longest inversion time
        # and slowest T1 value:
        max_recov_sig = np.abs(1 - 2 * np.exp(-timax / t1max))
        lb_A = 1 - inversion_imperfection / 2
        ub_A = 1 / max(max_recov_sig, 0.1)
        rel_bounds_A = [lb_A, ub_A]
        # If everything is perfect, baseline B is 0 -> lower bound of B is 0
        lb_B = 0
        #
        ub_B = inversion_imperfection / 2 * (1 / max(max_recov_sig, 0.1))
        rel_bounds_B = [lb_B, ub_B]

    if rel_bounds_B is not None:
        if isinstance(rel_bounds_B, int | float):
            bounds_B = (-rel_bounds_B * s_ref, rel_bounds_B * s_ref)
        else:
            bounds_B = (rel_bounds_B[0] * s_ref, rel_bounds_B[1] * s_ref)

    if rel_bounds_A is not None:
        if isinstance(rel_bounds_A, int | float):
            bounds_A = ((1.0 - rel_bounds_A) * s_ref, (1.0 + rel_bounds_A) * s_ref)
        else:
            bounds_A = (rel_bounds_A[0] * s_ref, rel_bounds_A[1] * s_ref)

    return bounds_A, bounds_B


def fit_t1_inversion_recovery_helper(
    signal: np.ndarray = None,
    inversion_times_ms: np.ndarray = None,
    t1_values_ms: np.ndarray = None,
    fit_baseline: bool = False,
    fit_params: dict | None = None,
):
    """Fit a single inversion-recovery curve across a millisecond T1 grid.

    ``inversion_times_ms`` and ``t1_values_ms`` use milliseconds. The returned
    ``"t1"`` value therefore also uses milliseconds. Public callers normally
    provide ``t1_values_s`` to :func:`hypermri.analysis.fitting.fit`; that grid
    is converted exactly once by the T1 parameter preparation step.
    """
    fit_params = fit_params or {}

    inversion_times_ms = fit_params.get("inversion_times_ms", inversion_times_ms)
    t1_values_ms = fit_params.get("t1_values_ms", t1_values_ms)
    fit_baseline = fit_params.get("fit_baseline", fit_baseline)
    signal = fit_params.get("signal", signal)
    signal = np.asarray(signal).squeeze()

    reg_alpha = fit_params.get("regularization_alpha", 0.0)

    bounds_A, bounds_B = _find_t1_bounds(
        signal=signal,
        inversion_times_ms=inversion_times_ms,
        t1_values_ms=t1_values_ms,
        fit_params=fit_params,
    )
    use_lsq_linear = any(np.isfinite(bounds_B)) or any(np.isfinite(bounds_A))

    t1_values_ms = np.asarray(t1_values_ms).squeeze()
    inversion_times_ms = np.asarray(inversion_times_ms).squeeze()

    weight_by_signal = fit_params.get("weight_by_signal", False)
    if weight_by_signal:
        weights = np.abs(signal) / (np.max(np.abs(signal)) + 1e-8)
        w_sqrt = np.sqrt(weights)
    else:
        w_sqrt = None

    rss_values = []
    coeffs_for_t1_values = []

    nan_param_val = np.nan + 0j * np.nan if np.iscomplexobj(signal) else np.nan

    for t1_guess in t1_values_ms:
        if t1_guess <= 0:
            rss_values.append(np.inf)
            coeffs_for_t1_values.append(
                [nan_param_val, nan_param_val] if fit_baseline else [nan_param_val]
            )
            continue

        exp_term = np.exp(-inversion_times_ms / float(t1_guess))

        if fit_baseline:
            m = np.vstack(
                [
                    np.ones_like(inversion_times_ms, dtype=float),
                    (1 - 2 * exp_term),
                ]
            ).T
        else:
            m = (1 - 2 * exp_term).reshape(-1, 1)

        if w_sqrt is not None:
            m_fit = m * w_sqrt[:, np.newaxis]
            signal_fit = signal * w_sqrt
        else:
            m_fit = m
            signal_fit = signal

        try:
            if use_lsq_linear:
                if fit_baseline:
                    lb = [bounds_B[0], bounds_A[0]]
                    ub = [bounds_B[1], bounds_A[1]]
                else:
                    lb = [bounds_A[0]]
                    ub = [bounds_A[1]]

                if np.iscomplexobj(signal_fit):
                    res = lsq_linear(m_fit, signal_fit.real, bounds=(lb, ub))
                    coeffs_real = res.x
                    res_imag = lsq_linear(m_fit, signal_fit.imag, bounds=(lb, ub))
                    coeffs_imag = res_imag.x
                    coeffs = coeffs_real + 1j * coeffs_imag
                else:
                    res = lsq_linear(m_fit, signal_fit, bounds=(lb, ub))
                    coeffs = res.x

                predicted_signals = m @ coeffs
                if w_sqrt is not None:
                    rss = np.sum(np.abs((signal - predicted_signals) * w_sqrt) ** 2)
                else:
                    rss = np.sum(np.abs(signal - predicted_signals) ** 2)

            elif reg_alpha > 0:
                n_params = m_fit.shape[1]
                identity = np.eye(n_params)
                lhs = m_fit.T @ m_fit + reg_alpha * identity
                rhs = m_fit.T @ signal_fit
                coeffs = np.linalg.solve(lhs, rhs)

                predicted_signals = m @ coeffs
                if w_sqrt is not None:
                    rss = np.sum(np.abs((signal - predicted_signals) * w_sqrt) ** 2)
                else:
                    rss = np.sum(np.abs(signal - predicted_signals) ** 2)

            else:
                coeffs, _, _, _ = np.linalg.lstsq(m_fit, signal_fit, rcond=None)
                predicted_signals = m @ coeffs
                if w_sqrt is not None:
                    rss = np.sum(np.abs((signal - predicted_signals) * w_sqrt) ** 2)
                else:
                    rss = np.sum(np.abs(signal - predicted_signals) ** 2)

            rss_values.append(rss)
            coeffs_for_t1_values.append(coeffs)

        except np.linalg.LinAlgError:
            rss_values.append(np.inf)
            coeffs_for_t1_values.append(
                [nan_param_val, nan_param_val] if fit_baseline else [nan_param_val]
            )

    if not rss_values or np.all(np.isinf(rss_values) | np.isnan(rss_values)):
        return {
            "t1": np.nan,
            "A": nan_param_val,
            "B": nan_param_val,
            "rss_t1": np.nan,
        }

    min_rss_index = np.nanargmin(rss_values)
    best_t1 = t1_values_ms[min_rss_index]
    best_coeffs = coeffs_for_t1_values[min_rss_index]
    min_rss = rss_values[min_rss_index]

    best_A = nan_param_val
    best_B = nan_param_val

    if fit_baseline:
        if len(best_coeffs) == 2 and not np.any(np.isnan(best_coeffs)):
            best_B = best_coeffs[0]
            best_A = best_coeffs[1]
    else:
        if len(best_coeffs) == 1 and not np.any(np.isnan(best_coeffs)):
            best_A = best_coeffs[0]
            best_B = 0.0
        if np.any(np.isnan(best_coeffs)):
            best_B = nan_param_val

    return {"t1": best_t1, "A": best_A, "B": best_B, "rss_t1": min_rss}


def inversion_recovery_model(
    inversion_times_ms: np.ndarray,
    a: xr.DataArray | np.ndarray | float | complex | None = None,
    b: xr.DataArray | np.ndarray | float | complex | None = None,
    t1_ms: xr.DataArray | np.ndarray | float | None = None,
    db_mode: bool = False,
    **kwargs,
) -> xr.DataArray:
    """Reconstruct a T1 inversion-recovery signal using millisecond time units.

    Parameters
    ----------
    inversion_times_ms
        Inversion times in milliseconds.
    a, b
        Signal amplitude and baseline parameters.
    t1_ms
        Longitudinal relaxation time in milliseconds, matching the ``t1``
        variable returned by the T1 fitter.

    Notes
    -----
    For backward compatibility this function accepts both:
    - modern keyword names: ``a``, ``b``, ``t1_ms``
    - legacy keyword names: ``A``, ``B``, ``t1``

    The historical ``t1_s`` keyword is also accepted without conversion. It
    was a misnamed alias whose values were always interpreted as milliseconds.
    """
    # Backward-compatible aliases used by external scripts and older notebooks.
    if a is None:
        a = kwargs.pop("A", None)
    if b is None:
        b = kwargs.pop("B", None)
    if t1_ms is None:
        t1_ms = kwargs.pop("t1", kwargs.pop("t1_s", None))

    if kwargs:
        unexpected = ", ".join(sorted(kwargs.keys()))
        raise TypeError(f"Unexpected keyword argument(s): {unexpected}")

    if a is None or b is None or t1_ms is None:
        raise TypeError(
            "inversion_recovery_model requires amplitude, baseline, and T1 via "
            "'a'/'A', 'b'/'B', and 't1_ms'/'t1'."
        )

    if db_mode:
        print("Calculating inversion recovery model:")
        print(f"  - inversion_times_ms: {inversion_times_ms}")
        print(f"  - A: {a}")
        print(f"  - B: {b}")
        print(f"  - t1_ms: {t1_ms}")

    if not isinstance(inversion_times_ms, xr.DataArray):
        inversion_times_ms = xr.DataArray(
            inversion_times_ms,
            dims=["inversion_idx"],
            coords={"inversion_idx": inversion_times_ms},
        )

    return a * (1 - 2 * np.exp(-inversion_times_ms / t1_ms)) + b
