"""State and steps of the "curve fit" tool of the browser viewer.

The curve-fit tool fits a relaxation or diffusion model along the curve axis
of the viewer, for every voxel at once:

======================  ==========  ===================  ==================
curve axis              model       axis unit            result map
======================  ==========  ===================  ==================
``bvalue_idx``          ADC         b-value, s/mm2       ``adc`` in mm²/s
``echo_idx``            T2          echo time, s         ``t2`` in ms
``inversion_idx``       T1          inversion time, ms   ``t1`` in ms
======================  ==========  ===================  ==================

The fits themselves are :func:`~hypermri.analysis.quantitative.fit_adc`,
:func:`~hypermri.analysis.quantitative.fit_t2` and
:func:`~hypermri.analysis.quantitative.fit_t1`; the fitted curves come from
the model equations of those fits. This module only chooses the model,
checks the axis and prepares what the browser shows. It builds no widgets.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import xarray as xr

from hypermri.analysis.fitting.execution import ExecutionConfig
from hypermri.analysis.fitting.models.quantitative.adc import (
    monoexponential_adc_model,
)
from hypermri.analysis.fitting.models.quantitative.t1 import inversion_recovery_model
from hypermri.analysis.fitting.models.quantitative.t2 import (
    monoexponential_t2_model,
)
from hypermri.analysis.quantitative import (
    ADCFitOptions,
    T1FitOptions,
    T2FitOptions,
    fit_adc,
    fit_t1,
    fit_t2,
)
from hypermri.visualization.contracts import RenderPayload
from hypermri.visualization.render_data import (
    FIT_RESIDUAL_OVERLAY,
    FIT_RESIDUAL_ZERO_OVERLAY,
    add_fit_residual_overlay,
    select_complex_part,
)

__all__ = [
    "CURVE_MODELS",
    "CurveFitState",
    "CurveModel",
    "check_curve_axis",
    "curve_fit_curves",
    "curve_fit_settings",
    "curve_fit_summary",
    "curve_model",
    "model_signal",
    "run_curve_fit",
    "set_curve_fit_option",
    "voxel_parameters",
]

# Points of the drawn fit curve between the first and last sample.
_FIT_CURVE_POINTS = 200
# Colour of the drawn fit curve, as for the spectral fit.
_FIT_COLOUR = "#d62728"


@dataclass(frozen=True)
class CurveModel:
    """One model of the curve-fit tool.

    Attributes
    ----------
    key : str
        Short model name, also the ``kind`` of a saved fit
        (:class:`~hypermri.visualization.saved_fits.SavedFit`): ``"adc"``,
        ``"t2"`` or ``"t1"``.
    label : str
        Name shown in the browser.
    dim : str
        Curve dimension the model is fitted along.
    axis_unit : str
        Required ``unit`` attribute of the coordinate of ``dim``.
    result_variable : str
        Variable of the fit result shown as map.
    result_unit : str
        Unit of ``result_variable``.
    min_points : int
        Smallest number of distinct axis values the fit needs.
    """

    key: str
    label: str
    dim: str
    axis_unit: str
    result_variable: str
    result_unit: str
    min_points: int


CURVE_MODELS: dict[str, CurveModel] = {
    model.dim: model
    for model in (
        CurveModel("adc", "ADC (diffusion)", "bvalue_idx", "s/mm2", "adc", "mm²/s", 2),
        CurveModel("t2", "T2 (echo decay)", "echo_idx", "s", "t2", "ms", 3),
        CurveModel(
            "t1", "T1 (inversion recovery)", "inversion_idx", "ms", "t1", "ms", 3
        ),
    )
}
"""Curve-fit model of each curve dimension."""


@dataclass
class CurveFitState:
    """Settings and last note of the curve-fit tool.

    The object is changed in place by the functions of this module.

    Attributes
    ----------
    fit_baseline : bool
        T1 only: fit a constant offset ``B`` besides the amplitude ``A``
        (:class:`~hypermri.analysis.quantitative.T1FitOptions`).
    message : str
        Last note or error shown below the tool.
    """

    fit_baseline: bool = False
    message: str = ""


def curve_model(data: xr.DataArray, curve_dim: str | None) -> CurveModel | None:
    """Return the model fitted along a curve dimension.

    Parameters
    ----------
    data : xarray.DataArray
        Data shown by the viewer. Not modified.
    curve_dim : str or None
        Dimension shown on the curve axis.

    Returns
    -------
    CurveModel or None
        The model of ``curve_dim`` from :data:`CURVE_MODELS`, or None when
        no model belongs to it or ``data`` lacks the dimension. The axis is
        not checked; see :func:`check_curve_axis`.
    """
    model = CURVE_MODELS.get(curve_dim) if curve_dim else None
    if model is None or model.dim not in data.dims:
        return None
    return model


def check_curve_axis(data: xr.DataArray, model: CurveModel) -> np.ndarray:
    """Return the axis values of a curve fit after checking them.

    Parameters
    ----------
    data : xarray.DataArray
        Data to fit. Not modified.
    model : CurveModel
        Model to fit along ``model.dim``.

    Returns
    -------
    numpy.ndarray
        Float axis values in ``model.axis_unit``, in data order.

    Raises
    ------
    ValueError
        If the coordinate of ``model.dim`` is missing, not one-dimensional,
        has another ``unit`` attribute, has non-finite or negative values or
        fewer than ``model.min_points`` distinct values.
    """
    dim = model.dim
    what = f"Curve fit {model.label}"
    if dim not in data.coords:
        raise ValueError(
            f"{what}: the data have no coordinate {dim!r}; it must hold the "
            f"axis values in {model.axis_unit}."
        )
    coord = data.coords[dim]
    unit = coord.attrs.get("unit", coord.attrs.get("units"))
    if coord.dims != (dim,):
        raise ValueError(
            f"{what}: coordinate {dim!r} must be one-dimensional; it has dims "
            f"{coord.dims}. Select one slice first."
        )
    if unit != model.axis_unit:
        raise ValueError(
            f"{what}: coordinate {dim!r} has unit {unit!r}, but "
            f"{model.axis_unit!r} is needed. Assign the axis values in "
            f"{model.axis_unit} with attrs={{'unit': {model.axis_unit!r}}}."
        )
    values = np.asarray(coord.values, dtype=float)
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError(
            f"{what}: the values of {dim!r} must be finite and not negative."
        )
    if np.unique(values).size < model.min_points:
        raise ValueError(
            f"{what}: {dim!r} needs at least {model.min_points} distinct values; "
            f"it has {np.unique(values).size}."
        )
    return values


def set_curve_fit_option(state: CurveFitState, name: str, value: Any) -> None:
    """Change one setting of the curve-fit tool.

    Parameters
    ----------
    state : CurveFitState
        Tool state; the attribute ``name`` is replaced.
    name : str
        ``"fit_baseline"``.
    value : Any
        New value, converted to bool.

    Raises
    ------
    ValueError
        If the setting does not exist.
    """
    if name != "fit_baseline":
        raise ValueError(f"Curve fit: unknown setting {name!r}; choose 'fit_baseline'.")
    state.fit_baseline = bool(value)


def run_curve_fit(
    data: xr.DataArray,
    model: CurveModel,
    state: CurveFitState,
    execution: ExecutionConfig | None = None,
) -> xr.Dataset:
    """Fit the model along the curve axis in every voxel.

    Parameters
    ----------
    data : xarray.DataArray
        Signal with dimension ``model.dim``; all other dimensions are voxels.
        Not modified. ADC and T2 fit the magnitude; T1 fits the values as
        given (real or complex).
    model : CurveModel
        Model to fit.
    state : CurveFitState
        Tool settings. Not modified.
    execution : ExecutionConfig or None, optional
        Parallel execution of the fit; None uses the default of the fit
        options (all CPU cores).

    Returns
    -------
    xarray.Dataset
        The result of :func:`~hypermri.analysis.quantitative.fit_adc`,
        :func:`~hypermri.analysis.quantitative.fit_t2` or
        :func:`~hypermri.analysis.quantitative.fit_t1`, with the voxel
        dimensions of ``data``. Failed voxels are NaN.

    Raises
    ------
    ValueError
        If the axis fails :func:`check_curve_axis` or the fit rejects the data.
    """
    check_curve_axis(data, model)
    if model.key == "adc":
        return fit_adc(data, ADCFitOptions(execution=execution))
    if model.key == "t2":
        return fit_t2(data, T2FitOptions(execution=execution))
    return fit_t1(
        data, T1FitOptions(fit_baseline=state.fit_baseline, execution=execution)
    )


def model_signal(
    model: CurveModel, parameters: dict[str, Any], x: np.ndarray
) -> np.ndarray:
    """Return the fitted model signal of one voxel.

    Parameters
    ----------
    model : CurveModel
        Fitted model.
    parameters : dict
        Fit result of the voxel: ``s0`` and ``adc`` (mm²/s) for ADC, ``s0``
        and ``t2`` (ms) for T2, ``A``, ``B`` and ``t1`` (ms) for T1.
    x : numpy.ndarray
        One-dimensional axis values in ``model.axis_unit``.

    Returns
    -------
    numpy.ndarray
        Model signal at ``x`` in the signal units of the fit; complex for a
        complex T1 fit.
    """
    x = np.asarray(x, dtype=float)
    if model.key == "adc":
        signal = monoexponential_adc_model(
            x, np.float64(parameters["s0"]), np.float64(parameters["adc"])
        )
    elif model.key == "t2":
        signal = monoexponential_t2_model(
            x, np.float64(parameters["s0"]), np.float64(parameters["t2"])
        )
    else:
        signal = inversion_recovery_model(
            x, a=parameters["A"], b=parameters["B"], t1_ms=parameters["t1"]
        )
    return np.asarray(signal)


def voxel_parameters(result: xr.Dataset, indices: dict[str, int]) -> dict[str, Any]:
    """Return the fit result of one voxel.

    Parameters
    ----------
    result : xarray.Dataset
        Fit result of :func:`run_curve_fit`. Not modified.
    indices : dict
        Position of the voxel per dimension; dimensions not named use 0.

    Returns
    -------
    dict
        Scalar value of every result variable.
    """
    voxel = {dim: int(indices.get(dim, 0)) for dim in result.dims}
    selected = result.isel(voxel)
    return {name: selected[name].values[()] for name in selected.data_vars}


def curve_fit_curves(
    model: CurveModel,
    parameters: dict[str, Any],
    x: np.ndarray,
    measured: np.ndarray,
    state: Any,
) -> list[dict[str, Any]]:
    """Return the fit curve and residual of one voxel for the curve panel.

    Parameters
    ----------
    model : CurveModel
        Fitted model.
    parameters : dict
        Fit result of the voxel, see :func:`model_signal`.
    x : numpy.ndarray
        Axis values of the measured curve in ``model.axis_unit``.
    measured : numpy.ndarray
        Displayed measured curve at ``x`` (already the shown complex part).
    state : PlotState
        Viewer state: ``state.spectral.spec_part`` chooses the complex part
        of the fit, ``state.spectral.show_fit_residual`` the residual.

    Returns
    -------
    list of dict
        ``{"name", "x", "y", "color"}`` of the fit curve, dense between the
        smallest and largest ``x``, then the residual and its dashed zero
        line as in the spectral fit (:func:`add_fit_residual_overlay`).
        Empty when the voxel fit failed.
    """
    x = np.asarray(x, dtype=float).ravel()
    measured = np.asarray(measured, dtype=float).ravel()
    if x.size == 0 or not all(np.all(np.isfinite(v)) for v in parameters.values()):
        return []
    part = state.spectral.spec_part
    dense_x = np.linspace(float(x.min()), float(x.max()), _FIT_CURVE_POINTS)
    dense_y = select_complex_part(model_signal(model, parameters, dense_x), part)
    curves = [
        {
            "name": f"Fit {model.label}",
            "x": dense_x.tolist(),
            "y": _finite_list(dense_y),
            "color": _FIT_COLOUR,
        }
    ]
    payload = RenderPayload(
        spectral_x=x,
        spectral_y=measured,
        overlays={"Fit": select_complex_part(model_signal(model, parameters, x), part)},
    )
    add_fit_residual_overlay(payload, state)
    labels = payload.metadata.get("spectral_overlay_legend_labels", {})
    colours = payload.metadata.get("spectral_overlay_colors", {})
    for name in (FIT_RESIDUAL_OVERLAY, FIT_RESIDUAL_ZERO_OVERLAY):
        if name not in payload.overlays:
            continue
        own_x = payload.metadata.get(f"overlay_x_{name}", x)
        curve = {
            "name": str(labels.get(name, name)),
            "x": np.asarray(own_x, dtype=float).tolist(),
            "y": _finite_list(payload.overlays[name]),
            "color": colours.get(name, "gray"),
        }
        if name == FIT_RESIDUAL_ZERO_OVERLAY:
            curve["dash"] = "dash"
        curves.append(curve)
    return curves


def curve_fit_summary(model: CurveModel, result: xr.Dataset) -> str:
    """Return a one-line note on a finished curve fit.

    Parameters
    ----------
    model : CurveModel
        Fitted model.
    result : xarray.Dataset
        Fit result of :func:`run_curve_fit`. Not modified.

    Returns
    -------
    str
        Number of voxels with a finite result and the median result with
        its unit.
    """
    values = np.asarray(result[model.result_variable].values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return f"{model.label}: no voxel could be fitted."
    return (
        f"{model.label}: {finite.size} of {values.size} voxels fitted, median "
        f"{model.result_variable} {np.median(finite):.4g} {model.result_unit}."
    )


def curve_fit_settings(
    state: CurveFitState, model: CurveModel | None
) -> dict[str, Any]:
    """Return the tool state as the browser shows it.

    Parameters
    ----------
    state : CurveFitState
        Tool state. Not modified.
    model : CurveModel or None
        Model of the current curve axis; None when the tool is not offered.

    Returns
    -------
    dict
        The settings of :class:`CurveFitState` and ``offered``, ``model``
        (key), ``label``, ``axis_unit`` and ``result_unit`` of the model.
    """
    settings = asdict(state)
    settings["offered"] = model is not None
    if model is not None:
        settings.update(
            model=model.key,
            label=model.label,
            axis_unit=model.axis_unit,
            result_unit=model.result_unit,
        )
    return settings


def _finite_list(values: Any) -> list[float | None]:
    """Return values as a JSON list with None for non-finite samples."""
    array = np.asarray(values, dtype=float).ravel()
    return [float(v) if np.isfinite(v) else None for v in array]
